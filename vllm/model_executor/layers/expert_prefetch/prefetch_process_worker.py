# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The expert-prefetch worker process.

A direct port of `ExpertCache._prefetch_worker`'s body into a process of its
own: same staging order (ids D2H -> host maps -> chunked copies -> event), same
degrade-on-failure semantics, but with its own interpreter and CUDA context so
none of it contends with the model's kernel launches.

Per prefetch (buffer `b`, request sequence `s`), against the protocol the
parent's `submit_prefetch` speaks:

  1. wake on `cmd_seq`, find `req_seq[b] > done_seq[b]`
  2. host-sync `ids_ready[b]` -- after this the predicted ids sit in
     `ids_staging[b]` AND the parent's compute tail (including the previous
     reader of this buffer) has drained, so copying into the buffer is safe
  3. write the slot<->expert maps (shared memory), set `maps_ready[b]`
  4. issue the staging copies on this process's stream, paced with blocking
     events exactly like the thread worker
  5. record `copies_done[b]` (interprocess), publish telemetry, set
     `copies_issued[b]`, `done_seq[b] = s`

Failure at any point still sets the flags (a stale map degrades to cache
misses in the parent, not a hang) and is loudly logged.
"""

import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.expert_prefetch.constants import NATIVE_BITS
from vllm.model_executor.layers.expert_prefetch.dequant_staging import DequantStaging
from vllm.model_executor.layers.expert_prefetch.expert_buffer import ExpertBuffer
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    QUANT_STORE_ATTR,
    QuantBlobLayout,
    QuantizedExpertWeight,
    blob_aliases,
)
from vllm.model_executor.layers.expert_prefetch.prefetch_process import (
    ControlBlock,
    WorkerBootstrap,
)
from vllm.model_executor.layers.expert_prefetch.shared_host_weights import (
    SharedHostArena,
)

logger = init_logger(__name__)

# Wake-up strategy while idle: spin hard briefly (prefetches arrive every
# layer, milliseconds apart, so the hot path should not sleep), then yield,
# then sleep in short ticks, polling the death pipe once per tick batch.
_SPINS = 2000
_YIELDS = 200
_SLEEP_S = 0.0001


def _device_by_uuid(gpu_uuid: str) -> torch.device:
    for i in range(torch.cuda.device_count()):
        if str(torch.cuda.get_device_properties(i).uuid) == gpu_uuid:
            return torch.device("cuda", i)
    raise RuntimeError(f"No visible CUDA device has UUID {gpu_uuid}")


def _chunk_bounds(n: int, num_chunks: int) -> list[tuple[int, int]]:
    """Mirror of `expert_cache._chunk_bounds` (not imported: that module pulls
    the whole cache stack in, and this process wants a minimal import set)."""
    num_chunks = max(1, min(num_chunks, n))
    if n == 0:
        return []
    size = -(-n // num_chunks)
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


class _Worker:
    def __init__(self, bootstrap: WorkerBootstrap):
        self.b = bootstrap
        self.device = _device_by_uuid(bootstrap.gpu_uuid)
        torch.cuda.set_device(self.device)

        # Host weight store: same physical pages as the parent's, registered
        # here so this context's H2D copies are true DMAs.
        self.weights = SharedHostArena.attach(bootstrap.weights)
        self.weights.register()

        # Control block: CPU-only access from this side, no registration.
        control = SharedHostArena.attach(bootstrap.control)
        self.cb = ControlBlock(
            control.tensor(bootstrap.control_key),
            bootstrap.num_slots,
            bootstrap.num_experts,
        )
        if int(self.cb.header["magic"][0]) != 0x45585052_46435442:
            raise RuntimeError("control block magic mismatch")

        # The GPU cache slots, imported once via CUDA IPC.
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        def _import(args: tuple) -> torch.Tensor:
            largs = list(args)
            largs[6] = self.device.index
            return rebuild_cuda_tensor(*largs)

        self.owners = self._build_owners()
        self.buffers: list[ExpertBuffer] = []
        for idx, name in enumerate(("ping", "pong")):
            buf = ExpertBuffer(name, bootstrap.param_names)
            buf.num_slots = bootstrap.num_slots
            buf.device = self.device
            buf.params = {
                p: _import(args) for p, args in bootstrap.tensor_ipc[name].items()
            }
            # The maps live in the control block, so writing them here is
            # publishing them to the parent.
            buf.slot_to_expert = self.cb.bufs["slot_to_expert"][idx]
            buf.expert_to_slot = self.cb.bufs["expert_to_slot"][idx]
            self.buffers.append(buf)

        self.stream = torch.cuda.Stream()
        self.pace = torch.cuda.Event(blocking=True)
        self.ids_ready = [
            torch.cuda.Event.from_ipc_handle(self.device, h)
            for h in bootstrap.ids_ready_handles
        ]
        self.copies_done = [torch.cuda.Event(interprocess=True) for _ in range(2)]

        # Dequant ring, local to this process (the packed blobs are transient;
        # nothing in the parent ever reads them).
        self.staging: DequantStaging | None = None
        if bootstrap.quant_bits:
            self.staging = DequantStaging("prefetch-worker")
            self.staging.allocate(
                bootstrap.quant_shapes, bootstrap.quant_bits, self.device
            )
            with torch.cuda.stream(self.stream):
                self.buffers[0].warmup_dequant(
                    self.staging, bootstrap.quant_bits, bootstrap.quant_group_size
                )

    def _build_owners(self) -> list[SimpleNamespace]:
        """One shim per MoE layer, shaped like the slice of `RoutedExperts`
        the staging code reads: the weight params plus the quant store."""
        owners = []
        for layer_name in self.b.layer_names:
            owner = SimpleNamespace(layer_name=layer_name)
            for p in self.b.param_names:
                setattr(owner, p, self.weights.tensor(f"{layer_name}.{p}"))
            if self.b.quant_bits:
                store: dict[str, dict[int, QuantizedExpertWeight]] = {}
                for p in self.b.param_names:
                    rows, cols, num_groups = self.b.quant_shapes[p]
                    store[p] = {}
                    for bits in self.b.quant_bits:
                        layout = QuantBlobLayout(
                            rows=rows,
                            cols=cols,
                            num_bits=bits,
                            num_groups=num_groups,
                        )
                        blob = self.weights.tensor(f"{layer_name}.{p}:int{bits}")
                        qweight, scale = blob_aliases(blob, layout)
                        store[p][bits] = QuantizedExpertWeight(
                            blob=blob, layout=layout, qweight=qweight, scale=scale
                        )
                owner.__dict__[QUANT_STORE_ATTR] = store
            owners.append(owner)
        return owners

    # ------------------------------------------------------------ serve loop

    def serve(self, death_pipe) -> None:
        header = self.cb.header
        bufs = self.cb.bufs
        last_cmd = int(header["cmd_seq"][0])
        while True:
            cmd = int(header["cmd_seq"][0])
            if cmd == last_cmd:
                if not self._idle_wait(header, last_cmd, death_pipe):
                    return
                continue
            last_cmd = cmd
            if int(header["shutdown"][0]):
                logger.info("Expert prefetch worker: shutdown requested")
                return
            pending = sorted(
                (int(bufs["req_seq"][idx]), idx)
                for idx in range(2)
                if bufs["req_seq"][idx] > bufs["done_seq"][idx]
            )
            for seq, idx in pending:
                self._handle(idx, seq)

    def _idle_wait(self, header, last_cmd: int, death_pipe) -> bool:
        """Wait for a new command; False means the process should exit."""
        for _ in range(_SPINS):
            if int(header["cmd_seq"][0]) != last_cmd:
                return True
        for _ in range(_YIELDS):
            os.sched_yield()
            if int(header["cmd_seq"][0]) != last_cmd:
                return True
        while int(header["cmd_seq"][0]) == last_cmd:
            # Parent gone (EOF makes the pipe readable) -> exit. The parent
            # never writes on this pipe.
            if death_pipe.poll(0):
                logger.info("Expert prefetch worker: parent exited, stopping")
                return False
            for _ in range(50):
                if int(header["cmd_seq"][0]) != last_cmd:
                    return True
                time.sleep(_SLEEP_S)
        return True

    # -------------------------------------------------------------- one job

    def _handle(self, idx: int, seq: int) -> None:
        bufs = self.cb.bufs
        buf = self.buffers[idx]
        try:
            n = int(bufs["n_ids"][idx])
            num_bits = int(bufs["num_bits"][idx])
            num_chunks = int(bufs["num_chunks"][idx])
            sampling = bool(bufs["sampling"][idx])
            owner = self.owners[int(bufs["layer_id"][idx])]

            # Gate on the parent's side stream: the ids D2H has landed and the
            # compute tail (the previous reader of this buffer) has drained.
            self.ids_ready[idx].synchronize()
            expert_list = self.cb.bufs["ids_staging"][idx][:n].tolist()

            with torch.cuda.stream(self.stream):
                buf.stage_ids(np.asarray(expert_list, dtype=np.int32))
                # `staged_layer` before `maps_ready`: the parent trusts the
                # maps only for the layer this echoes back.
                bufs["staged_layer"][idx] = bufs["layer_id"][idx]
                bufs["maps_ready"][idx] = 1

                elapsed = 0.0
                timed = 0
                if num_bits == NATIVE_BITS:
                    for lo, hi in _chunk_bounds(n, num_chunks):
                        start, end = self._chunk_events(sampling)
                        for slot_id in range(lo, hi):
                            buf.copy_expert(owner, slot_id, expert_list[slot_id])
                        if end is not None:
                            end.record(self.stream)
                        if hi < n:
                            # self.pace.record(self.stream)
                            # self.pace.synchronize()
                            self.stream.synchronize()
                            if end is not None:
                                elapsed += start.elapsed_time(end)
                                timed += hi - lo
                else:
                    assert self.staging is not None
                    ring = self.staging.num_slots
                    for lo in range(0, n, ring):
                        chunk_experts = expert_list[lo : lo + ring]
                        chunk_slots = list(range(lo, lo + len(chunk_experts)))
                        start, end = self._chunk_events(sampling)
                        buf._dequant_chunk(
                            owner,
                            self.staging,
                            num_bits,
                            self.b.quant_group_size,
                            chunk_experts,
                            chunk_slots,
                        )
                        if end is not None:
                            end.record(self.stream)
                        # self.pace.record(self.stream)
                        # self.pace.synchronize()
                        self.stream.synchronize()
                        if end is not None:
                            elapsed += start.elapsed_time(end)
                            timed += len(chunk_experts)

                self.copies_done[idx].record(self.stream)

                if timed:
                    # `t_e_count` last: it is the parent's new-sample guard.
                    bufs["t_e_ms"][idx] = elapsed / timed
                    bufs["t_e_bits"][idx] = num_bits
                    bufs["t_e_count"][idx] += 1
        except Exception:
            self.cb.header["child_error"][0] += 1
            logger.exception(
                "Expert prefetch worker failed on buffer %d (seq %d)", idx, seq
            )
        finally:
            bufs["maps_ready"][idx] = 1
            bufs["copies_issued"][idx] = 1
            bufs["done_seq"][idx] = seq

    def _chunk_events(self, sampling: bool):
        if not sampling:
            return None, None
        start = torch.cuda.Event(enable_timing=True)
        start.record(self.stream)
        return start, torch.cuda.Event(enable_timing=True)


def prefetch_worker_main(bootstrap_bytes: bytes, ready_pipe, death_pipe) -> None:
    """Process entry point (spawned; see `PrefetchProcessClient.start`)."""
    try:
        bootstrap: WorkerBootstrap = pickle.loads(bootstrap_bytes)
        worker = _Worker(bootstrap)
        ready_pipe.send(
            {
                "status": "ready",
                "copies_done_handles": [e.ipc_handle() for e in worker.copies_done],
            }
        )
        ready_pipe.close()
    except Exception as e:
        logger.exception("Expert prefetch worker failed to start")
        try:
            ready_pipe.send({"status": "error", "error": repr(e)})
            ready_pipe.close()
        except Exception:
            pass
        return

    try:
        worker.serve(death_pipe)
    except Exception:
        logger.exception("Expert prefetch worker crashed")
    finally:
        # Release the registration explicitly: interpreter teardown order in a
        # spawned child is otherwise happy to unmap before CUDA lets go.
        worker.weights.unregister()
