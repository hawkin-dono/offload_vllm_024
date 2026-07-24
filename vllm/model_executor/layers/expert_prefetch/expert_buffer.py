# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One side of the ping-pong cache: GPU storage for a layer's staged experts."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.model_executor.layers.expert_prefetch.constants import (
    EMPTY_SLOT,
    NATIVE_BITS,
)
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    QUANT_STORE_ATTR,
    dequant_experts_into_,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.expert_prefetch.dequant_staging import (
        DequantStaging,
    )
    from vllm.model_executor.layers.expert_prefetch.prefetch_process import (
        ControlBlock,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


@dataclass
class _SharedStagedFor:
    """`staged_for` backing when the worker runs in a separate process.

    The worker echoes the layer *id* into the control block after staging the
    maps; this translates it back to the layer name `resolve` compares
    against. Parent-side writes (reset, the staleness path) go through the
    same word."""

    staged_layer: np.ndarray  # (2,) int32 view into the control block
    idx: int
    layer_names: list[str]
    layer_ids: dict[str, int]

    def get(self) -> str | None:
        layer = int(self.staged_layer[self.idx])
        return self.layer_names[layer] if layer >= 0 else None

    def set(self, value: str | None) -> None:
        self.staged_layer[self.idx] = -1 if value is None else self.layer_ids[value]


class ExpertBuffer:
    """One side of the ping-pong cache: GPU storage for `num_slots` experts."""

    def __init__(self, name: str, param_names: tuple[str, ...]):
        self.name = name
        self.param_names = param_names
        self.num_slots = 0

        # Staged expert weights, keyed by the RoutedExperts param name they
        # shadow (e.g. "w13_weight"). Shape (num_slots, *expert_shape).
        self.params: dict[str, torch.Tensor] = {}

        # Slot<->expert maps, kept on the HOST as numpy arrays: the whole resolve
        # decision (remap, miss detection, eviction) runs on the CPU at decode,
        # so nothing here is a GPU tensor. `slot_to_expert[slot]` = the global
        # expert id staged in that slot (or EMPTY_SLOT); `expert_to_slot[e]` =
        # the slot holding expert `e` (or EMPTY_SLOT). The prefetch worker fills
        # them from the ids it already has on CPU.
        self.slot_to_expert: np.ndarray = np.empty(0, dtype=np.int32)
        self.expert_to_slot: np.ndarray = np.empty(0, dtype=np.int32)
        # The accelerator the weight buffers live on; set in `allocate`.
        self.device: torch.device | None = None

        # Which MoE layer the staged weights belong to. Buffers are recycled
        # across layers, so an id table alone is not enough to trust a slot:
        # slot 3 may well hold "expert 7", but expert 7 *of a previous layer*.
        # Every read checks this before treating a slot as a hit. (A property:
        # in process mode it lives in the shared control block instead.)
        self._shared_staged: _SharedStagedFor | None = None
        self.staged_for = None

        # A prefetch is issued from a worker thread, so readiness has two parts:
        # `copies_issued` (the thread finished enqueuing the copies) and
        # `prefetch_event` (the GPU finished executing them).
        self.maps_ready = threading.Event()
        self.maps_ready.set()
        self.copies_issued = threading.Event()
        self.copies_issued.set()
        self.prefetch_event: torch.cuda.Event | None = None

        # The predictor's top-`K` picks for this layer (K = the router's top_k,
        # not `prefetch_top_k`). Only used to measure accuracy: comparing the
        # *staged* set against what the layer routed to would conflate the
        # predictor being wrong with us having deliberately staged less, and the
        # Poisson model needs the former on its own.
        self.reference_ids: np.ndarray | None = None

        # Pinned landing rows for the worker's D2H of the staged/reference ids
        # (see `stage_host_ids`). Grown, never shrunk; per-buffer so a view
        # stays valid until the next prefetch into this same buffer.
        self._ids_host: torch.Tensor | None = None
        self._ref_host: torch.Tensor | None = None

    @property
    def staged_for(self) -> str | None:
        if self._shared_staged is not None:
            return self._shared_staged.get()
        return self._staged_local

    @staged_for.setter
    def staged_for(self, value: str | None) -> None:
        if self._shared_staged is not None:
            self._shared_staged.set(value)
        else:
            self._staged_local = value

    def attach_shared_state(
        self,
        cb: "ControlBlock",
        idx: int,
        layer_names: list[str],
        layer_ids: dict[str, int],
        liveness: Callable[[], bool],
        on_dead: Callable[[], None],
    ) -> None:
        """Rewire this buffer's cross-worker state onto the shared control
        block, for a worker running in a separate process.

        The readiness events become `SharedFlag`s, the slot<->expert maps
        become views into shared memory (so the worker's `stage_ids` publishes
        directly to what `resolve` reads), and `staged_for` reads/writes the
        shared `staged_layer` word. Everything else -- `prefetch_event`,
        `reference_ids`, the pinned id rows -- stays parent-local.
        """
        from vllm.model_executor.layers.expert_prefetch.prefetch_process import (
            SharedFlag,
        )

        self.maps_ready = SharedFlag(cb.bufs["maps_ready"], idx, liveness, on_dead)
        self.copies_issued = SharedFlag(
            cb.bufs["copies_issued"], idx, liveness, on_dead
        )
        cb.bufs["slot_to_expert"][idx][:] = self.slot_to_expert
        cb.bufs["expert_to_slot"][idx][: len(self.expert_to_slot)] = self.expert_to_slot
        self.slot_to_expert = cb.bufs["slot_to_expert"][idx]
        self.expert_to_slot = cb.bufs["expert_to_slot"][idx]
        self._shared_staged = _SharedStagedFor(
            staged_layer=cb.bufs["staged_layer"],
            idx=idx,
            layer_names=layer_names,
            layer_ids=layer_ids,
        )

    def allocate(
        self,
        owner: "RoutedExperts",
        num_slots: int,
        device: torch.device,
    ) -> None:
        self.num_slots = num_slots
        self.device = device
        for name in self.param_names:
            src = getattr(owner, name)
            self.params[name] = torch.zeros(
                (num_slots, *src.shape[1:]),
                dtype=src.dtype,
                device=device,
            )
        self.slot_to_expert = np.full(num_slots, EMPTY_SLOT, dtype=np.int32)
        self.expert_to_slot = np.full(
            owner.global_num_experts, EMPTY_SLOT, dtype=np.int32
        )

    def stage_ids(self, expert_ids: np.ndarray) -> None:
        """Record that slots 0..len-1 now hold `expert_ids` (host-side only).

        Called by the prefetch worker from the ids it already has on CPU, so the
        maps `resolve` reads are ready without any GPU round-trip.
        """
        n = len(expert_ids)
        self.slot_to_expert[:] = EMPTY_SLOT
        self.slot_to_expert[:n] = expert_ids
        self.expert_to_slot[:] = EMPTY_SLOT
        if n:
            self.expert_to_slot[expert_ids] = np.arange(n, dtype=np.int32)

    def clear_ids(self) -> None:
        """Drop the slot<->expert maps: every expert becomes a miss."""
        self.slot_to_expert[:] = EMPTY_SLOT
        self.expert_to_slot[:] = EMPTY_SLOT

    def wait_maps_ready(self) -> None:
        """Block (host-side) until the slot<->expert maps are published.

        The worker fills the maps and sets `maps_ready` up front -- before
        enqueuing the staging copies -- so this is all `resolve` needs to remap
        `topk_ids` on the host, and it returns without waiting for the (MiB) H2D
        to execute. It does NOT imply the copies are enqueued or `prefetch_event`
        is recorded; the weights are gated separately by `wait_prefetch_event`,
        which a caller reaches only after `copies_issued`.
        """
        self.maps_ready.wait()

    def wait_prefetch_event(self) -> None:
        """Make the current (compute) stream wait for the staged copies to land.

        GPU-side only: enqueues a wait on the compute stream and returns without
        blocking the host. `prefetch_event` is recorded by the worker only after
        the copies are enqueued (well after `maps_ready`), so a caller must first
        ensure the worker has got that far -- via `copies_issued` -- before
        relying on this; otherwise the event may still be unset and the wait a
        silent no-op.
        """
        if self.prefetch_event is not None:
            torch.cuda.current_stream().wait_event(self.prefetch_event)
            self.prefetch_event = None

    def wait_until_ready(self) -> None:
        """Block until this buffer's staged weights are usable.

        Waits for the worker thread to finish enqueuing copies, then makes the
        current stream wait on the copy stream. Cheap when no prefetch is in
        flight, since both events are already set.
        """
        self.copies_issued.wait()
        self.wait_prefetch_event()

    def stage_host_ids(
        self,
        expert_ids: torch.Tensor,
        reference_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Enqueue D2H of the staged/reference ids into reusable pinned rows.

        Returns pinned views the caller may read only after draining the stream
        it enqueued on. Pageable `.cpu()` here would allocate on every prefetch
        and spin-sync the stream mid-call; these copies stay async so the worker
        pays one blocking wait for both. The views stay valid until the next
        prefetch into this same buffer -- the lifetime `reference_ids` already
        has.
        """
        n = expert_ids.numel()
        host = self._ids_host
        if host is None or host.numel() < n or host.dtype != expert_ids.dtype:
            host = torch.empty(n, dtype=expert_ids.dtype, pin_memory=True)
            self._ids_host = host
        ids_view = host[:n]
        ids_view.copy_(expert_ids.reshape(-1), non_blocking=True)
        if reference_ids is None:
            return ids_view, None
        return ids_view, self.stage_host_ref(reference_ids)

    def stage_host_ref(self, reference_ids: torch.Tensor) -> torch.Tensor:
        """Enqueue D2H of the reference ids alone (see `stage_host_ids`)."""
        m = reference_ids.numel()
        ref = self._ref_host
        if ref is None or ref.numel() < m or ref.dtype != reference_ids.dtype:
            ref = torch.empty(m, dtype=reference_ids.dtype, pin_memory=True)
            self._ref_host = ref
        ref_view = ref[:m]
        ref_view.copy_(reference_ids.reshape(-1), non_blocking=True)
        return ref_view

    def copy_expert(
        self,
        owner: "RoutedExperts",
        slot_id: int,
        expert_id: int,
    ) -> None:
        """Copy a single expert from pinned CPU storage into `slot_id`.

        The copy is async only because each `src[expert_id]` is a *view* into
        the pinned CPU storage the offloader set up. Gathering the rows first
        (`src[expert_ids]`) would allocate a new, unpinned tensor and silently
        make the copy synchronous — do not "optimize" this into a batched index.
        """
        for name in self.param_names:
            src = getattr(owner, name)
            self.params[name][slot_id].copy_(src[expert_id], non_blocking=True)

    def _dequant_chunk(
        self,
        owner: "RoutedExperts",
        staging: "DequantStaging",
        num_bits: int,
        group_size: int,
        experts: list[int],
        slots: list[int],
    ) -> None:
        """Stage one ring-full: copy the packed blobs in, then dequant them out.

        `experts`/`slots` hold at most `staging.num_slots` expert ids and their
        destination cache slots. Each expert crosses the bus as a single blob
        (packed weights and scales together, which is why the store interleaves
        them); one kernel per parameter then unpacks the ring into `params`.
        Both are issued on the caller's current stream, so the dequant orders
        after the copies and the next chunk's copies order after this dequant --
        the ring write-after-read hazard is stream order alone.
        """
        store = owner.__dict__[QUANT_STORE_ATTR]
        for ring_slot, expert_id in enumerate(experts):
            for name in self.param_names:
                staging.landing_for(name, num_bits)[ring_slot].copy_(
                    store[name][num_bits].blob[expert_id], non_blocking=True
                )
        dst_index = staging.dst_index(slots)
        src_index = staging.slot_index[: len(experts)]
        for name in self.param_names:
            packed, scale = staging.aliases(name, num_bits)
            dequant_experts_into_(
                packed,
                scale,
                self.params[name],
                src_index,
                group_size,
                num_bits,
                dst_index,
            )

    def fetch(
        self,
        owner: "RoutedExperts",
        expert_ids: np.ndarray,
        slot_ids: np.ndarray,
        staging: "DequantStaging | None" = None,
        num_bits: int = NATIVE_BITS,
        group_size: int = 128,
    ) -> None:
        """Copy `expert_ids` from CPU into `slot_ids` of this buffer.

        At `NATIVE_BITS` the bf16 originals are copied per slot. Otherwise the
        experts are staged a ring-full at a time through `staging`: the small
        quantized blob crosses PCIe and is unpacked on the GPU into the same
        bf16 `params` slots the MoE kernel reads.
        """
        experts = expert_ids.tolist()
        slots = slot_ids.tolist()
        if num_bits == NATIVE_BITS:
            for slot_id, expert_id in zip(slots, experts):
                self.copy_expert(owner, slot_id, expert_id)
            return
        if staging is None:
            raise ValueError("A quantized fetch needs a DequantStaging ring.")
        chunk = staging.num_slots
        for start in range(0, len(experts), chunk):
            self._dequant_chunk(
                owner,
                staging,
                num_bits,
                group_size,
                experts[start : start + chunk],
                slots[start : start + chunk],
            )

    def warmup_dequant(
        self, staging: "DequantStaging", bits: tuple[int, ...], group_size: int
    ) -> None:
        """Compile the dequant kernel here, on the calling thread, per width.

        Otherwise the first launch happens inside the prefetch worker while
        `copies_issued` is cleared, stalling the consumer for the whole Triton
        compile -- and possibly racing a concurrent compile from the
        fetch-on-demand path. The kernel cache is keyed by the constexpr bits /
        group / block, so one launch per resident width compiles the variants
        both rings and the on-demand path reuse. Writing garbage into ring slot
        0 is safe: every cache slot is EMPTY_SLOT until a real fetch overwrites
        it.
        """
        slot_index = staging.slot_index[:1]
        for num_bits in bits:
            for name in self.param_names:
                packed, scale = staging.aliases(name, num_bits)
                dequant_experts_into_(
                    packed,
                    scale,
                    self.params[name],
                    slot_index,
                    group_size,
                    num_bits,
                    slot_index,
                )
        torch.cuda.current_stream().synchronize()
