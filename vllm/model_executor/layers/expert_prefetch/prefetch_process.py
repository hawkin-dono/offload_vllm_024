# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parent-side plumbing for running the prefetch worker in its own process.

The thread worker shares the interpreter and the CUDA context with the model's
forward pass, so its Python enqueue loops, Triton launches and stream waits
contend with the compute thread's own kernel launches (GIL and driver context
lock). Moving the worker into a separate process removes both: the child gets
its own interpreter and CUDA context, H2D staging copies ride the copy engines
(which overlap another context's kernels), and readiness crosses back over
interprocess CUDA events plus a small shared-memory control block.

Cross-process state, established once at startup:

  * GPU cache slots -- exported to the child via CUDA IPC (`reduce_tensor`).
  * Host expert store -- a `SharedHostArena` both processes register.
  * Control block -- one small shared-memory array (this module owns the
    layout): per-buffer request fields, readiness flags, the slot<->expert
    maps, an `ids_staging` landing row for the D2H of the predicted ids, and
    copy-time telemetry. Plain stores/loads with no locks: each word has a
    single writer at any point of the protocol, and x86's store ordering makes
    a field written before its guard (`req_seq` / `maps_ready`) visible before
    the guard flips.
  * Two interprocess CUDA events per buffer: `ids_ready` (parent records on
    the side stream after enqueuing the ids D2H; the child host-syncs it,
    which also discharges the buffer's write-after-read hazard because the
    side stream was made to wait on compute first) and `copies_done` (the
    child records after the staging copies; the parent's compute stream waits
    on it).

The per-prefetch protocol and the child's serve loop live in
`prefetch_process_worker`.
"""

import atexit
import os
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.expert_prefetch.constants import DEQUANT_RING_SLOTS
from vllm.model_executor.layers.expert_prefetch.shared_host_weights import (
    ArenaDescriptor,
    SharedHostArena,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.expert_prefetch.expert_buffer import ExpertBuffer
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

_MAGIC = 0x45585052_46435442  # "EXPRFCTB"

# How long a readiness wait may stall before the worker process is declared
# dead and the cache degrades to on-demand misses. Generous: a healthy child
# publishes maps in microseconds-to-milliseconds.
DEAD_WORKER_TIMEOUT_S = 5.0


def mps_active() -> bool:
    """Best-effort check that the CUDA MPS control daemon is reachable."""
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    return os.path.exists(os.path.join(pipe_dir, "control"))


def control_dtypes(num_slots: int, num_experts: int) -> tuple[np.dtype, np.dtype]:
    """(header, per-buffer) numpy layouts of the control block.

    Built from the same call in both processes, so the layouts always agree.
    """
    header = np.dtype(
        [
            ("magic", np.uint64),
            ("cmd_seq", np.uint64),
            ("shutdown", np.int32),
            ("child_error", np.int32),
        ],
        align=True,
    )
    per_buffer = np.dtype(
        [
            ("req_seq", np.uint64),
            ("done_seq", np.uint64),
            ("maps_ready", np.int32),
            ("copies_issued", np.int32),
            ("staged_layer", np.int32),
            ("layer_id", np.int32),
            ("n_ids", np.int32),
            ("num_bits", np.int32),
            ("num_chunks", np.int32),
            ("sampling", np.int32),
            ("t_e_ms", np.float64),
            ("t_e_bits", np.int32),
            ("t_e_count", np.int32),
            ("ids_staging", np.int32, (num_slots,)),
            ("slot_to_expert", np.int32, (num_slots,)),
            ("expert_to_slot", np.int32, (num_experts,)),
        ],
        align=True,
    )
    return header, per_buffer


class ControlBlock:
    """Typed views over the shared control-block tensor.

    `header` is a length-1 structured array; `bufs` a length-2 one (ping=0,
    pong=1). Field access (`cb.bufs["maps_ready"][idx]`) reads/writes the
    shared pages directly. `ids_tensor(idx)` is the torch view of a buffer's
    `ids_staging` row, for the GPU-side D2H copy.
    """

    def __init__(self, backing: torch.Tensor, num_slots: int, num_experts: int):
        header_dt, buf_dt = control_dtypes(num_slots, num_experts)
        raw = backing.numpy()
        assert raw.dtype == np.uint8 and raw.ndim == 1
        need = header_dt.itemsize + 2 * buf_dt.itemsize
        if raw.nbytes < need:
            raise ValueError(f"control block is {raw.nbytes}B, needs {need}B")
        self.header = raw[: header_dt.itemsize].view(header_dt)
        self.bufs = raw[
            header_dt.itemsize : header_dt.itemsize + 2 * buf_dt.itemsize
        ].view(buf_dt)
        self._ids_tensors = [
            torch.from_numpy(self.bufs["ids_staging"][i]) for i in range(2)
        ]

    @staticmethod
    def nbytes(num_slots: int, num_experts: int) -> int:
        header_dt, buf_dt = control_dtypes(num_slots, num_experts)
        return header_dt.itemsize + 2 * buf_dt.itemsize

    def ids_tensor(self, idx: int) -> torch.Tensor:
        return self._ids_tensors[idx]


class SharedFlag:
    """`threading.Event`-shaped flag over one int32 of the control block.

    `wait` spins briefly, then yields, and after `DEAD_WORKER_TIMEOUT_S`
    consults `liveness` -- if the worker process is gone, the flag is set
    (degrading this buffer to cache misses, mirroring the thread worker's
    `finally`) instead of hanging the model forever.
    """

    def __init__(
        self,
        arr: np.ndarray,
        idx: int,
        liveness: Callable[[], bool] | None = None,
        on_dead: Callable[[], None] | None = None,
    ):
        self._arr = arr
        self._idx = idx
        self._liveness = liveness
        self._on_dead = on_dead

    def set(self) -> None:
        self._arr[self._idx] = 1

    def clear(self) -> None:
        self._arr[self._idx] = 0

    def is_set(self) -> bool:
        return bool(self._arr[self._idx])

    def wait(self, timeout: float | None = None) -> bool:
        arr, idx = self._arr, self._idx
        if arr[idx]:
            return True
        for _ in range(200):
            if arr[idx]:
                return True
        deadline = time.monotonic() + (
            timeout if timeout is not None else DEAD_WORKER_TIMEOUT_S
        )
        while not arr[idx]:
            os.sched_yield()
            if time.monotonic() >= deadline:
                if self._liveness is not None and not self._liveness():
                    if self._on_dead is not None:
                        self._on_dead()
                    return bool(arr[idx])
                # Worker alive but slow (e.g. a huge staging burst): keep
                # waiting, checking liveness once per timeout period.
                deadline = time.monotonic() + DEAD_WORKER_TIMEOUT_S
        return True


@dataclass
class WorkerBootstrap:
    """Everything the worker process needs, pickled through spawn."""

    gpu_uuid: str
    weights: ArenaDescriptor
    control: ArenaDescriptor
    control_key: str
    num_slots: int
    num_experts: int
    param_names: tuple[str, ...]
    layer_names: list[str]  # index = layer_id
    # {buffer_name: {param_name: reduce_tensor() args}}
    tensor_ipc: dict[str, dict[str, tuple[Any, ...]]]
    ids_ready_handles: list[bytes]
    quant_bits: tuple[int, ...] = ()
    quant_group_size: int = 128
    # {param_name: (rows, cols, num_groups)} -- sizes the child's dequant ring.
    quant_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    ring_slots: int = DEQUANT_RING_SLOTS


class PrefetchProcessClient:
    """Owns the worker process and the parent side of the protocol."""

    def __init__(
        self,
        param_names: tuple[str, ...],
        layer_names: list[str],
        num_slots: int,
        num_experts: int,
        shm_dir: str,
        quant_bits: tuple[int, ...] = (),
        quant_group_size: int = 128,
        quant_shapes: dict[str, tuple[int, int, int]] | None = None,
    ):
        self.param_names = param_names
        self.layer_names = list(layer_names)
        self.layer_ids = {name: i for i, name in enumerate(self.layer_names)}
        self.num_slots = num_slots
        self.num_experts = num_experts
        self.shm_dir = shm_dir
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.quant_shapes = quant_shapes or {}

        self.cb: ControlBlock | None = None
        self._control_arena: SharedHostArena | None = None
        self._ids_ready: list[torch.cuda.Event] = []
        self._copies_done: list[torch.cuda.Event] = []
        self._proc = None
        self._death_write = None
        self._next_seq = 1
        self._dead = False
        self._last_t_e_count = [0, 0]
        atexit.register(self.shutdown)

    # ------------------------------------------------------------- lifecycle

    def start(
        self,
        buffers: "list[ExpertBuffer]",
        weights_arena: SharedHostArena,
        device: torch.device,
    ) -> None:
        """Spawn and handshake the worker; wire the buffers to shared state."""
        from vllm.utils.system_utils import get_mp_context

        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in alloc_conf.replace(" ", ""):
            raise RuntimeError(
                "prefetch_worker_mode='process' shares the GPU cache buffers "
                "over CUDA IPC, which expandable_segments allocations do not "
                "support. Unset expandable_segments in "
                "PYTORCH_CUDA_ALLOC_CONF."
            )
        if self.quant_bits and not mps_active():
            logger.warning(
                "expert_quant_bits with prefetch_worker_mode='process' but no "
                "CUDA MPS daemon detected: dequant kernels launched from the "
                "worker's CUDA context will time-slice against compute "
                "kernels instead of sharing SMs, likely erasing the prefetch "
                "overlap. Start MPS (nvidia-cuda-mps-control -d) or stage in "
                "bf16."
            )

        # Control block: shared memory registered in this (parent) context so
        # the per-prefetch ids D2H onto `ids_staging` is a true async DMA.
        self._control_arena = SharedHostArena(self.shm_dir)
        backing = self._control_arena.add(
            "control",
            torch.zeros(
                ControlBlock.nbytes(self.num_slots, self.num_experts),
                dtype=torch.uint8,
            ),
        )
        self._control_arena.register()
        cb = ControlBlock(backing, self.num_slots, self.num_experts)
        cb.bufs["staged_layer"][:] = -1
        cb.bufs["maps_ready"][:] = 1
        cb.bufs["copies_issued"][:] = 1
        cb.header["magic"][0] = _MAGIC
        self.cb = cb

        # The GPU slot tensors cross once, by IPC handle. Everything already
        # written to them must be visible before the handle is shared.
        from torch.multiprocessing.reductions import reduce_tensor

        torch.cuda.synchronize()
        tensor_ipc = {
            ("ping", "pong")[i]: {
                name: reduce_tensor(buf.params[name])[1] for name in self.param_names
            }
            for i, buf in enumerate(buffers)
        }

        self._ids_ready = [torch.cuda.Event(interprocess=True) for _ in range(2)]
        gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)

        bootstrap = WorkerBootstrap(
            gpu_uuid=gpu_uuid,
            weights=weights_arena.descriptor(),
            control=self._control_arena.descriptor(),
            control_key="control",
            num_slots=self.num_slots,
            num_experts=self.num_experts,
            param_names=self.param_names,
            layer_names=self.layer_names,
            tensor_ipc=tensor_ipc,
            ids_ready_handles=[e.ipc_handle() for e in self._ids_ready],
            quant_bits=self.quant_bits,
            quant_group_size=self.quant_group_size,
            quant_shapes=self.quant_shapes,
        )

        ctx = get_mp_context()
        ready_recv, ready_send = ctx.Pipe(duplex=False)
        # The child exits when this pipe hits EOF, so a hard-killed parent
        # cannot leave an orphan worker holding GPU memory.
        death_recv, death_send = ctx.Pipe(duplex=False)
        self._death_write = death_send

        from vllm.model_executor.layers.expert_prefetch.prefetch_process_worker import (  # noqa: E501
            prefetch_worker_main,
        )

        proc = ctx.Process(
            target=prefetch_worker_main,
            args=(pickle.dumps(bootstrap), ready_send, death_recv),
            name="expert-prefetch-worker",
            daemon=True,
        )
        proc.start()
        ready_send.close()
        death_recv.close()
        self._proc = proc

        if not ready_recv.poll(300):
            self._fail_start("worker process did not report ready in 300s")
        try:
            msg = ready_recv.recv()
        except EOFError:
            proc.join(timeout=5.0)
            self._fail_start(
                "prefetch worker process died during startup (exit code "
                f"{proc.exitcode}); see its traceback above"
            )
        if msg.get("status") != "ready":
            self._fail_start(f"worker process failed to start: {msg.get('error')}")
        self._copies_done = [
            torch.cuda.Event.from_ipc_handle(device, h)
            for h in msg["copies_done_handles"]
        ]
        # The worker has attached; only the names need to go.
        self._control_arena.unlink()
        logger.info(
            "Expert prefetch worker process started (pid %d, %d layers, %d slots)",
            proc.pid,
            len(self.layer_names),
            self.num_slots,
        )

        for idx, buf in enumerate(buffers):
            buf.attach_shared_state(
                cb,
                idx,
                self.layer_names,
                self.layer_ids,
                liveness=self.is_alive,
                on_dead=self.mark_dead,
            )

    def _fail_start(self, why: str) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
        raise RuntimeError(why)

    def is_alive(self) -> bool:
        return not self._dead and self._proc is not None and self._proc.is_alive()

    @property
    def usable(self) -> bool:
        return not self._dead

    def mark_dead(self) -> None:
        """Degrade permanently to on-demand misses; never hang the model.

        Mirrors the thread worker's `finally`: flags released, staged state
        invalidated. Called from a flag wait that timed out on a dead child.
        """
        if self._dead:
            return
        self._dead = True
        cb = self.cb
        if cb is not None:
            cb.bufs["staged_layer"][:] = -1
            cb.bufs["slot_to_expert"][:] = -1
            cb.bufs["expert_to_slot"][:] = -1
            cb.bufs["maps_ready"][:] = 1
            cb.bufs["copies_issued"][:] = 1
        logger.error(
            "Expert prefetch worker process died; prefetching is disabled and "
            "every expert will be fetched on demand from here on."
        )

    def shutdown(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.is_alive() and self.cb is not None:
                self.cb.header["shutdown"][0] = 1
                self.cb.header["cmd_seq"][0] += 1
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)
        except Exception:
            pass
        if self._control_arena is not None:
            self._control_arena.cleanup()

    # ------------------------------------------------------------- hot path

    def submit_prefetch(
        self,
        buf: "ExpertBuffer",
        idx: int,
        owner: "RoutedExperts",
        expert_ids: torch.Tensor,
        stream: torch.cuda.Stream,
        num_chunks: int,
        reference_ids: torch.Tensor | None,
        num_bits: int,
        sampling: bool,
    ) -> None:
        """Hand one prefetch to the worker process. Microseconds, no syncs.

        The caller (`ExpertCache.prefetch`) has already truncated the ids,
        ordered `stream` after compute (the write-after-read fence) and
        cleared the buffer's flags.
        """
        cb = self.cb
        assert cb is not None
        n = expert_ids.numel()
        bufs = cb.bufs
        bufs["layer_id"][idx] = self.layer_ids[owner.layer_name]
        bufs["n_ids"][idx] = n
        bufs["num_bits"][idx] = num_bits
        bufs["num_chunks"][idx] = num_chunks
        bufs["sampling"][idx] = 1 if sampling else 0

        with torch.cuda.stream(stream):
            # The ids land in shared pinned memory; the child reads them there
            # after `ids_ready`, so the D2H sync this used to cost the worker
            # thread happens in the child process instead.
            ids32 = expert_ids.reshape(-1).to(torch.int32)
            cb.ids_tensor(idx)[:n].copy_(ids32, non_blocking=True)
            if reference_ids is None:
                buf.reference_ids = None
            else:
                # Parent-local pinned row: only this process reads them (hit
                # telemetry in `resolve`), gated behind `maps_ready`, which the
                # child sets only after `ids_ready` -- by which point this D2H
                # (enqueued before the record) has landed.
                buf.reference_ids = buf.stage_host_ref(reference_ids).numpy()
        self._ids_ready[idx].record(stream)

        bufs["req_seq"][idx] = self._next_seq
        self._next_seq += 1
        cb.header["cmd_seq"][0] += 1

        # Re-armed by the child's record each prefetch; consumed (set to None)
        # by `wait_prefetch_event` after `copies_issued`, exactly like the
        # thread worker's per-prefetch event.
        buf.prefetch_event = self._copies_done[idx]

    def drain_copy_times(self, sink: dict[int, Any]) -> None:
        """Pull new per-expert copy-time samples into `sink` (drain_stats)."""
        cb = self.cb
        if cb is None:
            return
        for idx in range(2):
            count = int(cb.bufs["t_e_count"][idx])
            if count != self._last_t_e_count[idx]:
                self._last_t_e_count[idx] = count
                sink[int(cb.bufs["t_e_bits"][idx])].append(
                    float(cb.bufs["t_e_ms"][idx])
                )
