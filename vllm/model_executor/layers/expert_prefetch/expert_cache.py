# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ping-pong GPU cache for routed MoE expert weights.

Expert weights live in pinned CPU memory (see `ExpertCacheOffloader`). Only the
experts a layer actually routes to are staged onto the GPU, into one of two
buffers:

  * the **active** buffer holds the experts for the layer running right now;
  * the **inactive** buffer is concurrently filled, on a side stream, with the
    experts predicted for the *next* layer.

The two swap after every MoE layer, so GPU-resident expert weights cost two
layers rather than the whole model. A single cache is shared by every MoE layer
in the model.

Slots are addressed by position, not by expert id: `slot_to_expert[i]` is the
global expert id currently staged in slot `i` (or -1 if empty), and its inverse
`expert_to_slot` remaps `topk_ids` into slot indices before the kernel runs. Both
maps are kept on the HOST (numpy): at decode `topk_ids` is a handful of ids, so
the remap, miss detection and eviction all run on the CPU -- the only device
traffic is one D2H of `topk_ids` and one H2D of the result, with no per-layer
bookkeeping kernels on the compute stream. Experts predicted incorrectly are
simply cache misses: they get fetched on demand, which is correct but synchronous.
"""

import os
import re
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    MAX_EXPERT_QUANT_BITS,
    QUANT_STORE_ATTR,
    ExpertQuantSpec,
    QuantBlobLayout,
    blob_aliases,
    dequant_experts_into_,
    validate_quantizable,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

EMPTY_SLOT = -1

# Bit width standing for the native (bf16) originals. Always a prefetch
# candidate, costs no extra memory, and takes the plain per-slot copy path.
NATIVE_BITS = 16

# Experts a dequant ring holds before recycling. Sized in bytes at the widest
# supported width so one ring serves int8/int4/int2 alike.
DEQUANT_RING_SLOTS = 16


class DequantStaging:
    """Small GPU ring the packed experts land in before being unpacked.

    Quantization only ever shrinks what crosses PCIe; the MoE kernel still reads
    bf16, so every staged expert has to be dequantized into an `ExpertBuffer`
    slot anyway. That makes the packed bytes *transient* -- alive only between
    an expert's H2D copy and the dequant kernel that reads it -- so they do not
    need a slot per cache slot, and they do not need a buffer per bit width.
    One ring of `num_slots` experts, sized in bytes at the widest supported
    width, serves every resident width at once: at 128 cache slots that is
    ~150 MiB instead of the ~2 GiB a full-size buffer per width would cost.

    A fetch walks its experts a ring-full at a time. Copies and dequant are
    issued on the *same* stream, so the copies of chunk k+1 are ordered behind
    the dequant of chunk k that reads those ring slots -- the write-after-read
    hazard is handled by stream order alone, with no events to get wrong.

    One ring cannot be shared by the whole cache, though: `ExpertCache.prefetch`
    runs on a side stream while `ExpertCache.resolve` fetches misses on the
    compute stream, and the two overlap in time. Each path owns one.
    """

    def __init__(self, name: str, num_slots: int = DEQUANT_RING_SLOTS):
        self.name = name
        self.num_slots = num_slots
        # Packed blobs, keyed by param name: (num_slots, blob_bytes) uint8,
        # sized at the widest bit width.
        self.blobs: dict[str, torch.Tensor] = {}
        # {name: {num_bits: landing}} -- the blob rows trimmed to what a store
        # of that width occupies, so an H2D copy is one contiguous row-sized
        # transfer. A narrower store leaves the tail of each ring row untouched.
        self.landing: dict[str, dict[int, torch.Tensor]] = {}
        # {name: {num_bits: (qweight, scale)}} -- strided aliases of the blob,
        # built once here so a fetch never re-derives them.
        self._aliases: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        # Ring slot indices, sliced per chunk; kept on device so a fetch does
        # not have to build one.
        self.slot_index: torch.Tensor = torch.empty(0, dtype=torch.int32)

    def allocate(
        self,
        shapes: dict[str, tuple[int, int, int]],
        bits: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Size the ring from `shapes` ({name: (rows, cols, num_groups)}).

        Rows/cols/num_groups are the same across widths (they depend on the
        weight shape and group size, not the bit width), so the ring is built
        once at `MAX_EXPERT_QUANT_BITS` and every resident width's landing and
        aliases address the same storage.
        """
        for name, (rows, cols, num_groups) in shapes.items():
            widest = QuantBlobLayout(
                rows=rows,
                cols=cols,
                num_bits=MAX_EXPERT_QUANT_BITS,
                num_groups=num_groups,
            )
            blob = torch.zeros(
                (self.num_slots, widest.blob_bytes), dtype=torch.uint8, device=device
            )
            self.blobs[name] = blob
            self.landing[name] = {}
            self._aliases[name] = {}
            for num_bits in bits:
                layout = QuantBlobLayout(
                    rows=rows, cols=cols, num_bits=num_bits, num_groups=num_groups
                )
                self.landing[name][num_bits] = blob[:, : layout.blob_bytes]
                self._aliases[name][num_bits] = blob_aliases(blob, layout)
        self.slot_index = torch.arange(self.num_slots, dtype=torch.int32, device=device)

    def aliases(self, name: str, num_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (qweight, scale) views of `name`'s ring at `num_bits`."""
        return self._aliases[name][num_bits]

    def landing_for(self, name: str, num_bits: int) -> torch.Tensor:
        """The landing region of `name`'s ring at `num_bits`."""
        return self.landing[name][num_bits]

    def nbytes(self) -> int:
        return sum(blob.numel() for blob in self.blobs.values())


@dataclass
class CacheStats:
    """One forward pass of measurements, for `PrefetchController`.

    All host-side ints: the resolve decision runs on the CPU, so the hit/needed
    tallies are already on the host and never touch the device.
    """

    # Summed over the layers that ran against a prefetched buffer.
    hits: int = 0
    needed: int = 0
    # Hits the predictor's full top-`K` picks *would* have got. The accuracy the
    # Poisson model wants, uncontaminated by how much we chose to stage.
    reference_hits: int = 0
    staged: int = 0
    layers: int = 0
    truncated: bool = False
    # Unread (start, end) event pairs; drained lazily, a forward pass later.
    t_comp_events: list["_TCompSample"] = field(default_factory=list)


@dataclass
class ForwardStats:
    """What one forward pass measured, read off the device exactly once."""

    hits: int
    reference_hits: int
    needed: int
    staged: int
    layers: int
    truncated: bool
    t_comp_ms: list[float]
    # Per-expert copy time in ms, keyed by the precision (num_bits) that staged.
    # NATIVE_BITS is the bf16 copy; only widths a forward actually ran appear.
    t_e_ms: dict[int, float]


@dataclass
class _TCompSample:
    """Events bracketing one layer's compute, H2D excluded.

    `t_comp = elapsed(prev_ready, begin)` -- the compute-stream span from the
    previous layer's resolve finishing (`prev_ready`, after its wait and any
    on-demand fetches) to this layer's resolve starting (`begin`, before its
    wait). That interval is exactly one layer's compute -- the previous layer's
    MoE kernel plus this layer's attention and gating -- with both layers' H2D
    stalls left out, which is the bubble a single prefetch can hide behind.

    Anchored at the previous *resolve*, not at when the prefetch was issued: a
    prefetch driven by an `attn_input` predictor is issued a whole attention
    block before that resolve, so measuring from the issue point would fold the
    previous layer's attention into the window and roughly double `t_comp`.
    """

    begin: torch.cuda.Event
    prev_ready: torch.cuda.Event

    def ready(self) -> bool:
        return self.begin.query() and self.prev_ready.query()

    def elapsed_ms(self) -> float:
        return self.prev_ready.elapsed_time(self.begin)


LOG_ACCURACY = os.getenv("LOG_ACCURACY", "0") == "1"
ACCURACY_LOG_INTERVAL = int(os.getenv("ACCURACY_LOG_INTERVAL", "1"))
# When set, the per-forward accuracy summary is also appended to this file.
ACCURACY_LOG_FILE = os.getenv("ACCURACY_LOG_FILE", "/tmp/vllm_expert_accuracy.log")


def _chunk_bounds(n: int, num_chunks: int) -> list[tuple[int, int]]:
    """Split `range(n)` into at most `num_chunks` contiguous, non-empty spans."""
    num_chunks = max(1, min(num_chunks, n))
    if n == 0:
        return []
    size = -(-n // num_chunks)  # ceil, so the last chunk is the short one
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def _layer_index(layer_name: str) -> int:
    """Best-effort decoder-layer index from a param prefix, for ordered logs."""
    match = re.search(r"layers\.(\d+)", layer_name)
    return int(match.group(1)) if match else -1


class AccuracyTracker:
    """Accumulates prefetch accuracy overall and per MoE layer.

    For each MoE layer the cache reports how many of the experts the layer
    actually routed to (`needed`) had already been staged by the predictor
    (`hits`). The hit rate is the prediction accuracy: 1.0 means every routed
    expert was prefetched, 0.0 means all were fetched on demand.

    `record_topk` adds the other half of the picture: the `prefetch_top_k` the
    controller chose for that layer. Reading accuracy without it is misleading,
    since a layer scoring badly because the predictor was wrong and one scoring
    badly because we deliberately staged two experts look identical.
    """

    def __init__(self) -> None:
        self._hits: dict[str, int] = {}
        self._needed: dict[str, int] = {}
        self._topk_sum: dict[str, int] = {}
        self._topk_count: dict[str, int] = {}
        # Running means of the two times the controller solves against: how long
        # one layer computes for, and how long one expert takes to copy in.
        self._t_comp_sum = 0.0
        self._t_comp_count = 0
        # Per-precision (num_bits) running copy-time sums; NATIVE_BITS for bf16.
        self._t_e_sum: dict[int, float] = {}
        self._t_e_count: dict[int, int] = {}
        self._forwards = 0

    def update(self, layer_name: str, hits: int, needed: int) -> None:
        with open(ACCURACY_LOG_FILE, "a") as f:
            f.write(f"Layer {layer_name}: hits={hits}, needed={needed}\n")
        self._hits[layer_name] = self._hits.get(layer_name, 0) + hits
        self._needed[layer_name] = self._needed.get(layer_name, 0) + needed

    def record_topk(self, layer_name: str, prefetch_top_k: int) -> None:
        """Record the `prefetch_top_k` one layer's prefetch was issued at.

        Keyed by the layer being *staged*, not the layer whose hidden state fed
        the predictor, so it lines up with the accuracy the same layer reports
        from `resolve`. Kept as a mean rather than a single value: the
        controller picks per batch-size bucket, so a run that mixes batch sizes
        genuinely has more than one.
        """
        self._topk_sum[layer_name] = (
            self._topk_sum.get(layer_name, 0) + prefetch_top_k
        )
        self._topk_count[layer_name] = self._topk_count.get(layer_name, 0) + 1

    def mean_topk(self, layer_name: str) -> float | None:
        """The mean `prefetch_top_k` for one layer, or None if never staged."""
        count = self._topk_count.get(layer_name, 0)
        return self._topk_sum[layer_name] / count if count else None

    def record_timings(
        self, t_comp_ms: list[float], t_e_ms: dict[int, float]
    ) -> None:
        """Record one forward's compute and per-precision copy times, in ms.

        Not per layer: `t_comp_ms` arrives as a flat list of per-layer windows
        with no layer attached, and `t_e_ms` is a property of the link (per
        precision) rather than of any one layer. Both are also drained lazily --
        an event that had not completed by the last drain is reported a forward
        or two late -- so these are running means over the whole run, not this
        pass's values.
        """
        self._t_comp_sum += sum(t_comp_ms)
        self._t_comp_count += len(t_comp_ms)
        for bits, t in t_e_ms.items():
            if t > 0.0:
                self._t_e_sum[bits] = self._t_e_sum.get(bits, 0.0) + t
                self._t_e_count[bits] = self._t_e_count.get(bits, 0) + 1

    def on_forward_end(self, interval: int = ACCURACY_LOG_INTERVAL) -> None:
        """Log the running accuracy every `interval` forward passes."""
        self._forwards += 1
        if interval and self._forwards % interval == 0:
            self.log()

    def overall(self) -> float:
        hits = sum(self._hits.values())
        needed = sum(self._needed.values())
        return hits / needed if needed else 0.0

    def _topk_suffix(self, layer_name: str) -> str:
        """`/k<mean>` for a staged layer, empty for one that never was.

        The absence is the useful part: a layer with no suffix was never
        prefetched at all -- no predictor, or its prefetch never landed -- which
        is a different failure from one that was staged and mispredicted.
        """
        mean = self.mean_topk(layer_name)
        return f"/k{mean:.1f}" if mean is not None else ""

    def log(self) -> None:
        if not self._needed:
            return
        per_layer = " ".join(
            f"L{_layer_index(name)}="
            f"{self._hits[name] / self._needed[name]:.2f}"
            f"{self._topk_suffix(name)}"
            for name in sorted(self._needed, key=_layer_index)
            if self._needed[name]
        )
        staged = sum(self._topk_count.values())
        overall_topk = sum(self._topk_sum.values()) / staged if staged else 0.0
        t_comp = (
            self._t_comp_sum / self._t_comp_count if self._t_comp_count else 0.0
        )
        # `t_comp/t_e(p)` is the bubble budget the controller solves against per
        # precision: how many expert copies of that width fit under one layer's
        # compute. Printed per width so a `p` that looks wrong can be traced to
        # whichever term produced it.
        te_parts = []
        for bits in sorted(self._t_e_sum, reverse=True):
            count = self._t_e_count.get(bits, 0)
            if not count:
                continue
            t_e = self._t_e_sum[bits] / count
            budget = t_comp / t_e if t_e > 0.0 else 0.0
            label = "bf16" if bits == NATIVE_BITS else f"int{bits}"
            te_parts.append(f"{label}(t_e={t_e:.3f}ms,budget={budget:.1f})")
        te_str = " ".join(te_parts) if te_parts else "n/a"
        summary = (
            f"[ExpertAcc] forwards={self._forwards} overall={self.overall():.3f} "
            f"(hits={sum(self._hits.values())}, "
            f"needed={sum(self._needed.values())}) topk={overall_topk:.1f} "
            f"t_comp={t_comp:.3f}ms t_e=[{te_str}] "
            f"| {per_layer}"
        )
        logger.info("%s", summary)
        if ACCURACY_LOG_FILE:
            try:
                with open(ACCURACY_LOG_FILE, "a") as f:
                    f.write(summary + "\n")
            except OSError as e:
                logger.warning("Could not write accuracy log to %s: %s",
                               ACCURACY_LOG_FILE, e)

    def reset(self) -> None:
        self._hits.clear()
        self._needed.clear()
        self._topk_sum.clear()
        self._topk_count.clear()
        self._t_comp_sum = 0.0
        self._t_comp_count = 0
        self._t_e_sum.clear()
        self._t_e_count.clear()
        self._forwards = 0


# Global tracker, mirroring offload_vllm: `resolve` records into it and the
# model's forward-end hook logs it. A single instance is shared across layers.
accuracy_tracker = AccuracyTracker()


def maybe_create_expert_cache(
    routed_experts: list["RoutedExperts"],
) -> "ExpertCache | None":
    """Build the shared cache for a model's MoE layers, if the expert_cache
    offload backend is active. Returns None otherwise, leaving the model on the
    stock (fully GPU-resident) path.
    """
    from vllm.model_executor.offloader import ExpertCacheOffloader, get_offloader

    offloader = get_offloader()
    if not isinstance(offloader, ExpertCacheOffloader) or not routed_experts:
        return None

    cache = ExpertCache()
    cache.bind(routed_experts)
    # Buffers are allocated later, from the offloader's post_init: expert
    # weights are not in their final runtime layout until then.
    offloader.register_expert_cache(cache)
    return cache


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
        # Every read checks this before treating a slot as a hit.
        self.staged_for: str | None = None

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
        # Built here, not hoisted: the allocator attributes it to whichever
        # stream is current at this point, which is the one the kernel runs on.
        dst_index = torch.tensor(slots, dtype=torch.int32, device=self.device)
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


class ExpertCache:
    """Two `ExpertBuffer`s, flipped after each MoE layer.

    Shared by every MoE layer in the model: `bind` attaches it to each layer's
    `RoutedExperts`, and the MoE forward reaches it via `layer.expert_cache`.
    """

    def __init__(self, num_cache_slots: int = 0):
        # 0 means "size to hold a whole layer" — resolved in `allocate`, once we
        # can see the expert weights.
        self._requested_slots = num_cache_slots
        self.param_names: tuple[str, ...] = ()
        self.num_slots = 0
        self.allocated = False

        self.ping = ExpertBuffer("ping", ())
        self.pong = ExpertBuffer("pong", ())
        self.active_name = "ping"

        # Quantized-store state, populated by `allocate`. `quant_bits` is the
        # resident set (fidelity-descending, e.g. (8, 4, 2)); `qspecs` maps each
        # width to its spec; `_ondemand_bits` is the width a cache miss is served
        # at -- the coarsest resident width (int2 when present), else NATIVE_BITS.
        self.quant_bits: tuple[int, ...] = ()
        self.quant_group_size = 128
        self.qspecs: dict[int, ExpertQuantSpec] = {}
        self._ondemand_bits = NATIVE_BITS
        # One dequant ring per issuing stream: prefetch runs on a side stream,
        # on-demand on the compute stream, and the two overlap in time.
        self.prefetch_staging = DequantStaging("prefetch")
        self.ondemand_staging = DequantStaging("on-demand")

        self._owner: RoutedExperts | None = None
        # Single worker: prefetches are issued in layer order and a second one
        # cannot start until the previous buffer has been consumed anyway.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="expert-prefetch"
        )

        # Prefetch accuracy: of the experts a layer turned out to need, how many
        # were already staged. Host-side ints -- the resolve decision is on CPU.
        self._hits: int = 0
        self._needed: int = 0

        # Per-forward measurements for the adaptive controller, plus the timing
        # state that feeds them. `stats` is reset by `drain_stats`, and counts
        # only layers that ran against a prefetched buffer -- unlike `_hits` /
        # `_needed`, which are lifetime totals over every layer.
        self.stats = CacheStats()
        self._sampling = False
        self._prev_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        # Per-expert copy times measured by the worker thread, in milliseconds,
        # keyed by the precision (num_bits) the pass staged at. Only the width a
        # forward actually ran refines at runtime; the others keep their
        # calibrated seed (the cross-width ratio is dominated by blob size, which
        # is known, and calibration measured each width directly).
        self._copy_times: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=64)
        )

        # Reusable staging buffers for the per-layer `resolve` transfers, so no
        # tensor is allocated on the hot path. `_host_topk` receives the D2H of
        # `topk_ids`; `_host_out`/`_dev_out` carry the remapped ids H2D. Grown
        # (never shrunk) when a larger batch or a new dtype appears. Only ever
        # touched from the compute thread in `resolve`, so no locking is needed.
        self._host_topk: torch.Tensor | None = None
        self._host_out: torch.Tensor | None = None
        self._dev_out: torch.Tensor | None = None

    @property
    def owner(self) -> "RoutedExperts":
        """Any one MoE layer; all of them share the same expert weight layout."""
        if self._owner is None:
            raise RuntimeError("ExpertCache used before bind().")
        return self._owner

    @staticmethod
    def _check_supported(owner: "RoutedExperts") -> None:
        """Reject configurations the cache would silently get wrong.

        Each of these produces plausible-looking but incorrect output rather than
        an error, so they are checked up front, at load time.
        """
        if owner.expert_map is not None:
            raise NotImplementedError(
                "Expert cache does not support expert parallelism: topk_ids are "
                "global expert ids, and the cache indexes this rank's local "
                "expert weights, so the two would silently disagree."
            )
        if owner.moe_config.has_bias:
            raise NotImplementedError(
                "Expert cache does not support MoE layers with expert bias: the "
                "kernel indexes w13_bias/w2_bias by expert id, but the cached "
                "path passes cache slot indices, so the wrong bias would be "
                "applied."
            )
        if owner.quant_method.is_monolithic:
            raise NotImplementedError(
                "Expert cache requires the modular MoE path; this layer uses a "
                f"monolithic kernel ({owner.quant_method.__class__.__name__}), "
                "which routes internally and never exposes topk_ids to remap."
            )

    @staticmethod
    def _check_quant_supported(
        owner: "RoutedExperts",
        param_names: tuple[str, ...],
        quant_bits: tuple[int, ...],
        group_size: int,
    ) -> None:
        """Reject layers the quantized store would get wrong.

        Shape and group-size constraints are checked by `validate_quantizable`
        instead, since the quantization pass runs before this and has to fail
        with the same message. Checked per resident width, at load time.
        """
        quant_method = owner.quant_method.__class__.__name__
        if quant_method != "UnquantizedFusedMoEMethod":
            raise NotImplementedError(
                f"expert_quant_bits requires unquantized expert weights; this "
                f"layer uses {quant_method}. Re-quantizing already-quantized "
                "weights is not supported."
            )
        for name in param_names:
            src = getattr(owner, name)
            if not src.dtype.is_floating_point:
                raise NotImplementedError(
                    f"expert_quant_bits requires float expert weights; {name} "
                    f"is {src.dtype}."
                )
            for num_bits in quant_bits:
                validate_quantizable(src, group_size, name, num_bits)

    def bind(self, routed_experts: list["RoutedExperts"]) -> None:
        """Attach this cache to every MoE layer that will share it."""
        if not routed_experts:
            raise ValueError("ExpertCache.bind requires at least one MoE layer.")
        self._owner = routed_experts[0]
        for layer in routed_experts:
            # Bypass nn.Module.__setattr__ so the shared cache does not become a
            # submodule of every layer (which would duplicate it in state_dict,
            # and make weight loading complain about unexpected parameters).
            layer.__dict__["expert_cache"] = self

    def allocate(
        self,
        param_names: tuple[str, ...],
        default_num_slots: int = 0,
        quant_bits: tuple[int, ...] = (),
        quant_group_size: int = 128,
    ) -> None:
        """Allocate both GPU buffers. Called from the offloader's `post_init`,
        once expert weights are in their final runtime layout."""
        if self.allocated:
            return
        owner = self._owner
        if owner is None:
            raise RuntimeError("ExpertCache.allocate called before bind().")
        if not param_names:
            raise ValueError("ExpertCache has no expert parameters to cache.")

        self._check_supported(owner)

        # Resident quantized widths, fidelity-descending. On-demand misses are
        # served at the coarsest resident width to keep the critical-path stall
        # smallest (requirement 3); NATIVE_BITS when quantization is disabled.
        self.quant_bits = tuple(sorted(set(quant_bits), reverse=True))
        self.quant_group_size = quant_group_size
        self._ondemand_bits = min(self.quant_bits) if self.quant_bits else NATIVE_BITS
        if self.quant_bits:
            self._check_quant_supported(
                owner, param_names, self.quant_bits, quant_group_size
            )
            self.qspecs = {
                bits: ExpertQuantSpec(num_bits=bits, group_size=quant_group_size)
                for bits in self.quant_bits
            }

        num_experts = owner.local_num_experts
        num_slots = self._requested_slots or default_num_slots or num_experts
        if num_slots > num_experts:
            num_slots = num_experts
        if num_slots < owner.local_num_experts:
            raise ValueError(
                f"num_cache_slots ({num_slots}) is smaller than the layer's "
                f"local_num_experts ({owner.local_num_experts}); can not handle "
                "the prefill phase. Raise --num-cache-slots or set "
                "expert_cache.num_cache_slots in the config."
            )

        self.param_names = param_names
        self.num_slots = num_slots
        self.ping = ExpertBuffer("ping", param_names)
        self.pong = ExpertBuffer("pong", param_names)

        # The cache is the GPU-resident mirror, so it must land on the
        # accelerator, not alongside the offloaded (CPU) source weights.
        device = torch.device(torch.cuda.current_device())
        self.ping.allocate(owner, num_slots, device)
        self.pong.allocate(owner, num_slots, device)

        staging_note = ""
        if self.quant_bits:
            store = owner.__dict__.get(QUANT_STORE_ATTR, {})
            missing = [
                (name, bits)
                for name in param_names
                for bits in self.quant_bits
                if bits not in store.get(name, {})
            ]
            if missing:
                raise RuntimeError(
                    f"No quantized store for {missing} on {owner.layer_name}. "
                    "The offloader builds it in post_init, before allocating the "
                    "cache; allocating with quantization enabled outside that "
                    "order cannot work."
                )
            # Rows/cols/num_groups are width-independent, so any resident width's
            # layout sizes the (shared) ring.
            shapes = {
                name: (
                    store[name][self.quant_bits[0]].layout.rows,
                    store[name][self.quant_bits[0]].layout.cols,
                    store[name][self.quant_bits[0]].layout.num_groups,
                )
                for name in param_names
            }
            self.prefetch_staging.allocate(shapes, self.quant_bits, device)
            self.ondemand_staging.allocate(shapes, self.quant_bits, device)
            # Once, not per buffer: both rings share the same kernel variants,
            # so the second call would only recompile.
            self.ping.warmup_dequant(
                self.prefetch_staging, self.quant_bits, quant_group_size
            )
            staging_bytes = (
                self.prefetch_staging.nbytes() + self.ondemand_staging.nbytes()
            )
            staging_note = (
                f" (staged as int{'/'.join(str(b) for b in self.quant_bits)} "
                f"through 2 x {DEQUANT_RING_SLOTS}-expert dequant rings, "
                f"{staging_bytes / 1024**2:.0f} MiB; on-demand at "
                f"int{self._ondemand_bits})"
            )

        self._hits = 0
        self._needed = 0
        self.stats = CacheStats()
        self.allocated = True

        bytes_per_buffer = sum(
            t.numel() * t.element_size() for t in self.ping.params.values()
        )
        logger.info(
            "Expert cache: 2 x %d slots (%d experts/layer), %.2f GiB on GPU%s",
            num_slots,
            num_experts,
            2 * bytes_per_buffer / 1024**3,
            staging_note,
        )

    def calibrate_copy_time(
        self, num_samples: int = 8, warmup: int = 2
    ) -> dict[int, float]:
        """Time one expert's staging cost per precision, in milliseconds.

        Returns `{num_bits: ms_per_expert}` over NATIVE_BITS (the bf16 per-slot
        copy) and each resident quantized width (its ring-full H2D *plus* the
        dequant kernel, since the bubble budget must account for both). Run once
        from the prefetcher's first forward, so the controller has a bubble
        constraint per precision before enough passes go by to measure one. Off
        the hot path, so full syncs are fine here.
        """
        if not self.allocated:
            raise RuntimeError("calibrate_copy_time called before allocate().")
        owner = self.owner
        buf = self.ping

        src = getattr(owner, self.param_names[0])
        if not src.is_pinned():
            logger.warning(
                "Expert weights are not in pinned memory, so every H2D copy is "
                "synchronous and cannot overlap compute. Prefetching will not "
                "help. Check VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY."
            )

        num_experts = min(owner.local_num_experts, buf.num_slots)

        def _timed(run, per_call: int) -> float:
            for _ in range(warmup):
                run()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_samples):
                run()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / (num_samples * per_call)

        def _native() -> None:
            for i in range(num_experts):
                buf.copy_expert(owner, i % buf.num_slots, i % num_experts)

        times = {NATIVE_BITS: _timed(_native, num_experts)}

        ring = min(num_experts, self.prefetch_staging.num_slots)
        experts = [i % num_experts for i in range(ring)]
        slots = list(range(ring))
        for num_bits in self.quant_bits:

            def _quant(num_bits: int = num_bits) -> None:
                buf._dequant_chunk(
                    owner,
                    self.prefetch_staging,
                    num_bits,
                    self.quant_group_size,
                    experts,
                    slots,
                )

            times[num_bits] = _timed(_quant, ring)

        # Leave no trace: these slots hold weights nothing staged, and the id
        # maps must not claim otherwise.
        buf.clear_ids()
        buf.staged_for = None

        bytes_per_expert = sum(
            t[0].numel() * t[0].element_size() for t in buf.params.values()
        )
        summary = ", ".join(
            f"{'bf16' if b == NATIVE_BITS else f'int{b}'}={t:.3f}ms"
            for b, t in times.items()
        )
        logger.info(
            "Expert cache copy calibration: %s (bf16 expert %.1f MiB)",
            summary,
            bytes_per_expert / 1024**2,
        )
        return times

    def begin_forward(self, sampling: bool) -> None:
        """Start a forward pass, optionally recording timing events.

        Sampling is off by default: the events are cheap but not free, and the
        controller only consumes them every `adapt_interval` passes.
        """
        self._sampling = sampling and not torch.cuda.is_current_stream_capturing()
        self._prev_events = None

    def drain_stats(self) -> "ForwardStats":
        """Read this forward's measurements, and reset them.

        All counters are host ints (the resolve decision is on CPU), so this is
        free -- no device sync.
        """
        stats = self.stats
        # Partitioned in one pass: `ready()` is a live device query, so asking
        # twice can see an event complete in between and drop it unread.
        ready, pending = [], []
        for sample in stats.t_comp_events:
            (ready if sample.ready() else pending).append(sample)

        result = ForwardStats(
            hits=int(stats.hits),
            reference_hits=int(stats.reference_hits),
            needed=int(stats.needed),
            staged=stats.staged,
            layers=stats.layers,
            truncated=stats.truncated,
            # Pending samples are carried to the next drain rather than waited
            # on: blocking here would defeat the point, and dropping them would
            # usually lose the whole batch, all recorded moments ago.
            t_comp_ms=[t for t in (s.elapsed_ms() for s in ready) if t > 0.0],
            t_e_ms={
                bits: sum(times) / len(times)
                for bits, times in self._copy_times.items()
                if times
            },
        )
        self._copy_times.clear()
        stats.hits = 0
        stats.needed = 0
        stats.reference_hits = 0
        stats.staged = 0
        stats.layers = 0
        stats.truncated = False
        stats.t_comp_events = pending
        return result

    def hit_rate(self) -> float:
        """Fraction of needed experts that prediction had already staged.

        1.0 means every expert a layer routed to was prefetched; 0.0 means all
        were fetched on demand (which is what you get with no predictor).
        """
        return self._hits / self._needed if self._needed else 0.0

    def reset_stats(self) -> None:
        self._hits = 0
        self._needed = 0

    @property
    def sampling(self) -> bool:
        """Whether this forward pass is recording timing/accuracy telemetry.

        Set by `begin_forward` at the first prefetch of the pass, so it is
        stable by the time a layer ranks or resolves against the cache.
        """
        return self._sampling

    @property
    def active(self) -> ExpertBuffer:
        return self.ping if self.active_name == "ping" else self.pong

    @property
    def inactive(self) -> ExpertBuffer:
        return self.pong if self.active_name == "ping" else self.ping

    def flip(self) -> None:
        self.active_name = "pong" if self.active_name == "ping" else "ping"

    def reset(self) -> None:
        """Invalidate both buffers and return to a known state.

        Called at the end of the last MoE layer so a new forward pass starts
        from `ping` and cannot read weights staged for the previous one.
        """
        for buf in (self.ping, self.pong):
            buf.wait_until_ready()
            buf.staged_for = None
            buf.reference_ids = None
            if buf.num_slots:
                buf.clear_ids()
        self.active_name = "ping"
        self._prev_events = None

    def _record_event(self) -> torch.cuda.Event | None:
        """A timing event on the compute stream, or None when not sampling."""
        if not self._sampling:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _close_window(self, begin: torch.cuda.Event | None) -> None:
        """Finish this layer's timing window and pair it with the last one."""
        if begin is None:
            return
        ready = torch.cuda.Event(enable_timing=True)
        ready.record()
        if self._prev_events is not None:
            _, prev_ready = self._prev_events
            self.stats.t_comp_events.append(
                _TCompSample(begin=begin, prev_ready=prev_ready)
            )
        self._prev_events = (begin, ready)

    def _record_layer_stats(
        self,
        buf: ExpertBuffer,
        needed: np.ndarray,
        hit_count: int,
        needed_count: int,
    ) -> None:
        """Tally one prefetched layer for the controller (host-side ints).

        `needed` is the distinct experts this layer routed to. Everything here
        is a plain int -- the totals are drained once per forward.
        """
        stats = self.stats
        stats.hits += hit_count
        stats.needed += needed_count
        stats.layers += 1
        if buf.reference_ids is not None:
            stats.reference_hits += int(np.isin(needed, buf.reference_ids).sum())
        else:
            stats.reference_hits += hit_count

    def _to_host(self, topk_ids: torch.Tensor) -> np.ndarray:
        """D2H `topk_ids` into a reusable pinned buffer; return a numpy view.

        Replaces `topk_ids.cpu().numpy()`: the pinned buffer turns the transfer
        into a DMA and avoids a pageable allocation on every layer. The read is
        host-side, so the copy is synced with an event first. The returned view
        aliases the buffer and is only read downstream (the remap indexes it),
        so reuse is safe until the next `resolve`.
        """
        n = topk_ids.numel()
        host = self._host_topk
        if host is None or host.numel() < n or host.dtype != topk_ids.dtype:
            host = torch.empty(n, dtype=topk_ids.dtype, pin_memory=True)
            self._host_topk = host
        dst = host[:n]
        dst.copy_(topk_ids.reshape(-1), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        return dst.numpy().reshape(tuple(topk_ids.shape))

    def _to_device(
        self,
        cached: np.ndarray,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """H2D the remapped ids through reusable pinned/device buffers.

        Replaces `torch.as_tensor(...).to(device)`. The copy is non-blocking:
        the MoE kernel that reads the result is issued right after, on the same
        (compute) stream, so it is ordered after the transfer without a sync.

        Both buffers are reused across layers, which is safe in eager mode (the
        feature requires `--enforce-eager`):
          * `_dev_out` -- layer L's kernel reads it before layer L+1's H2D, which
            is ordered after that kernel on the shared compute stream;
          * `_host_out` -- every `resolve` calls `_to_host` (a full-stream sync)
            before this, so the previous layer's H2D has drained and the host
            write here cannot race a DMA still reading the pinned buffer.
        """
        n = cached.size
        host = self._host_out
        if host is None or host.numel() < n or host.dtype != dtype:
            host = torch.empty(n, dtype=dtype, pin_memory=True)
            self._host_out = host
        dev = self._dev_out
        if (
            dev is None
            or dev.numel() < n
            or dev.dtype != dtype
            or dev.device != device
        ):
            dev = torch.empty(n, dtype=dtype, device=device)
            self._dev_out = dev
        host_v = host[:n]
        host_v.numpy()[:] = cached.reshape(-1)
        dev_v = dev[:n]
        dev_v.copy_(host_v, non_blocking=True)
        return dev_v.view(shape)

    def resolve(
        self,
        owner: "RoutedExperts",
        topk_ids: torch.Tensor,
    ) -> tuple[ExpertBuffer, torch.Tensor]:
        """Make the active buffer usable for `owner`, and remap `topk_ids`.

        Waits for any in-flight prefetch, fetches whatever the predictor missed,
        and rewrites `topk_ids` from global expert ids into cache slot indices.
        Returns the buffer to run against, and the remapped ids.

        The remap, miss detection and eviction all run on the CPU: at decode
        `topk_ids` is a handful of ids, and deciding which pinned rows to fetch
        is a host decision anyway. The only device traffic is one D2H of
        `topk_ids`, the (rare) on-demand weight copies, and one H2D of the
        remapped ids -- no per-layer bookkeeping kernels on the compute stream.
        """
        buf = self.active

        # Recorded before the wait, deliberately: a stall here is time the
        # prefetch failed to hide, and must not be counted as compute.
        begin = self._record_event()

        # The one D2H: bring the routing to the host. Independent of the prefetch
        # -- it needs only the router's topk -- so it is issued first, before any
        # wait, and its blocking sync then costs ~microseconds (topkGating) rather
        # than sitting behind the staging H2D. `topk_ids` is (T, K) ints.
        topk_np = self._to_host(topk_ids)
        flat = topk_np.reshape(-1)

        # Host maps: published by the worker as soon as it knows the ids, before
        # the staging copies land, so the remap is not serialized behind the
        # (MiB) H2D. This does not wait for the weights themselves.
        buf.wait_maps_ready()

        # If this buffer was not staged for *this* layer, nothing in it can be
        # trusted (see `ExpertBuffer.staged_for`). Dropping the maps turns every
        # expert into a miss -- slow but always correct, and the path taken when
        # prediction is disabled entirely.
        prefetched = buf.staged_for == owner.layer_name
        if not prefetched:
            buf.clear_ids()
            buf.staged_for = owner.layer_name

        # Remap on the host: expert id -> slot, or EMPTY_SLOT for a miss. Fancy
        # indexing returns a fresh array, so patching it below is safe.
        cached = buf.expert_to_slot[flat]
        miss = cached < 0

        # `needed` (the distinct routed experts) is wanted by both the telemetry
        # and the miss handler, but neither runs on the common all-hit non-sampled
        # path, so compute it at most once and only when something needs it.
        needed: np.ndarray | None = None

        # Accuracy telemetry for the controller -- consumed once per forward and
        # only on sampled passes. Computed pre-fetch so a hit means prediction.
        if self._sampling or LOG_ACCURACY:
            needed = np.unique(flat)
            self._record_hits(owner, buf, flat, needed, prefetched)

        # Order the compute stream after this buffer's staged weights before the
        # on-demand fetches and the MoE kernel read it. `copies_issued` also
        # guarantees the worker has recorded `prefetch_event` (`maps_ready` fires
        # earlier, before the copies), so it must precede `wait_prefetch_event`.
        # This waits only for the *active* buffer's prefetch; the next layer's
        # prefetch keeps streaming on the side stream, so its chunked copies still
        # interleave with the on-demand fetches issued just below.
        buf.copies_issued.wait()
        buf.wait_prefetch_event()

        # Fetch whatever missed. Host-side decision; the weight copies go on the
        # compute stream so the MoE kernel is ordered after them.
        if miss.any():
            if needed is None:
                needed = np.unique(flat)
            cached = self._resolve_misses(owner, buf, flat, cached, miss, needed)

        # Everything between `begin` and here is dead time: the D2H stall plus
        # the on-demand copies. The next layer's window starts at `ready`, so it
        # is excluded rather than subtracted.
        self._close_window(begin)

        # Ship the remapped ids back for the kernel (T*K ints); the MoE kernel
        # that follows is ordered after this H2D on the compute stream.
        cached_topk_ids = self._to_device(
            cached, tuple(topk_ids.shape), topk_ids.dtype, topk_ids.device
        )
        return buf, cached_topk_ids

    def _record_hits(
        self,
        owner: "RoutedExperts",
        buf: ExpertBuffer,
        flat: np.ndarray,
        needed: np.ndarray,
        prefetched: bool,
    ) -> None:
        """Hit/needed accounting on the host, on sampled passes only.

        `flat` is the routed expert ids and `needed` their distinct values
        (deduped once by the caller). A routed expert is a hit iff its pre-fetch
        slot is valid, so this must run before `_resolve_misses`.
        """
        needed_count = int(needed.size)
        hit_count = int((buf.expert_to_slot[needed] >= 0).sum())

        self._hits += hit_count
        self._needed += needed_count
        if prefetched:
            self._record_layer_stats(buf, needed, hit_count, needed_count)
        if LOG_ACCURACY:
            accuracy_tracker.update(owner.layer_name, hit_count, needed_count)

    def _resolve_misses(
        self,
        owner: "RoutedExperts",
        buf: ExpertBuffer,
        flat: np.ndarray,
        cached: np.ndarray,
        miss: np.ndarray,
        needed: np.ndarray,
    ) -> np.ndarray:
        """Fetch routed experts that were not staged; return the patched slots.

        Runs entirely on the host -- the `topk_ids` D2H already paid the only
        sync. Mutates the buffer's slot maps and issues the on-demand weight
        copies on the compute stream, so the MoE kernel is ordered after them.
        `needed` is the distinct routed experts, deduped once by the caller.
        """
        # Distinct missing experts.
        missing = np.unique(flat[miss])

        # A slot is evictable unless it holds an expert this layer still needs.
        # Empty slots (EMPTY_SLOT) never match `needed`, so they are evictable.
        occupied = buf.slot_to_expert >= 0
        holds_needed = occupied & np.isin(buf.slot_to_expert, needed)
        evictable = np.nonzero(~holds_needed)[0]
        if evictable.size < missing.size:
            raise RuntimeError(
                f"Expert cache too small: layer {owner.layer_name} routes to "
                f"{int(needed.size)} experts but the cache has "
                f"{buf.num_slots} slots. Raise --num-cache-slots."
            )
        victims = evictable[: missing.size].astype(np.int32)

        # Update the host maps, then copy the weights in on the compute stream.
        # `resolve` has already waited on the buffer's prefetch, so the on-demand
        # copies land into slots the prefetch is done writing. On-demand is the
        # coarsest resident width (int2 when present): a miss is on the critical
        # path, so it wants the smallest transfer. When quantization is off,
        # `_ondemand_bits` is NATIVE_BITS and this is the plain bf16 copy. The
        # dequant runs on the compute stream inside the timing window, so it is
        # charged as dead time and correctly excluded from `t_comp`.
        buf.slot_to_expert[victims] = missing
        buf.expert_to_slot[missing] = victims
        buf.fetch(
            owner,
            missing,
            victims,
            staging=self.ondemand_staging,
            num_bits=self._ondemand_bits,
            group_size=self.quant_group_size,
        )

        # Patch the caller's remap for the just-fetched experts.
        cached[miss] = buf.expert_to_slot[flat[miss]]
        return cached

    def prefetch(
        self,
        owner: "RoutedExperts",
        expert_ids: torch.Tensor,
        stream: torch.cuda.Stream,
        num_chunks: int = 1,
        reference_ids: torch.Tensor | None = None,
        num_bits: int = NATIVE_BITS,
    ) -> None:
        """Stage `expert_ids` of `owner` (the *next* MoE layer) into the
        inactive buffer.

        `expert_ids` must already be deduplicated and ranked best-first: when it
        does not fit the cache the tail is dropped, so the caller's ordering
        decides what survives. `num_bits` is the precision to stage at, chosen
        by the controller (NATIVE_BITS for bf16). Returns as soon as the copies
        are handed to a worker thread; the consumer synchronizes via
        `wait_maps_ready` (host maps) and `wait_prefetch_event` (staged weights).
        """
        buf = self.inactive

        if expert_ids.numel() > buf.num_slots:
            expert_ids = expert_ids[: buf.num_slots]

        # The copy stream must not run ahead of the compute stream. It overwrites
        # a buffer the *previous* layer's MoE kernel may still be reading, and
        # `expert_ids` was produced on this same side stream by the ranker. This
        # orders the copies after the compute stream's current tail; the
        # single-worker executor serializes successive prefetches of this buffer
        # (each buffer is re-staged only every other layer, after being consumed),
        # so no explicit wait for the previous copy is needed here -- and doing
        # one would stall the compute stream, which is exactly what we are
        # removing.
        stream.wait_stream(torch.cuda.current_stream())

        # Keep `expert_ids` alive until the copy stream is done with it, since
        # the allocator only tracks the stream it was created on.
        expert_ids.record_stream(stream)

        buf.maps_ready.clear()
        buf.copies_issued.clear()
        device = buf.device
        self._executor.submit(
            self._prefetch_worker,
            owner,
            buf,
            expert_ids,
            stream,
            device,
            num_chunks,
            reference_ids,
            num_bits,
        )

    def _prefetch_worker(
        self,
        owner: "RoutedExperts",
        buf: ExpertBuffer,
        expert_ids: torch.Tensor,
        stream: torch.cuda.Stream,
        device: torch.device,
        num_chunks: int,
        reference_ids: torch.Tensor | None,
        num_bits: int = NATIVE_BITS,
    ) -> None:
        try:
            torch.cuda.set_device(device)
            # Written here, off the main thread, so the ranking that produced
            # these stays off the compute stream. `staged_for` is read by
            # `resolve` after `maps_ready`, so it must be set before that fires.
            buf.staged_for = owner.layer_name
            with torch.cuda.stream(stream):
                expert_list = expert_ids.cpu().tolist()
                n = len(expert_list)

                # Reference ids are host-side telemetry; bring them over here on
                # the side stream (off the main thread), where they were produced.
                # Before `maps_ready`, since `resolve`'s hit accounting reads them
                # right after that wait.
                buf.reference_ids = (
                    None
                    if reference_ids is None
                    else reference_ids.detach().cpu().numpy()
                )

                # Record the slot<->expert maps on the host, from the ids we
                # already have on CPU, so `resolve` reads them without any GPU
                # round-trip. Staged experts are distinct (no collisions). Publish
                # `maps_ready` immediately: this is all `resolve` needs to remap,
                # and it fires before the (MiB) staging copies below execute, so
                # the consumer is not serialized behind the H2D. The weights
                # themselves are gated separately, by `prefetch_event`.
                buf.stage_ids(np.asarray(expert_list, dtype=np.int32))
                buf.maps_ready.set()

                # Copies go out in chunks with a drain between them, so that
                # on-demand fetches -- issued on the compute stream by a layer
                # that mispredicted -- get the DMA engine instead of queueing
                # behind the whole prefetch. Within a chunk we only enqueue:
                # draining after *every* copy would serialize the transfer, and
                # ordering the copies against their reader is `prefetch_event`'s
                # job, which it does on the GPU without blocking this thread.
                #
                # A quantized prefetch instead walks a dequant ring a ring-full
                # at a time (copy the packed blobs in, then unpack them into the
                # bf16 slots), syncing between ring-fulls -- which both enforces
                # the ring write-after-read hazard and paces on-demand fetches.
                # The ring size dictates the chunking, so `num_chunks` is unused.
                elapsed = 0.0
                timed = 0
                if num_bits == NATIVE_BITS:
                    for lo, hi in _chunk_bounds(n, num_chunks):
                        start, end = self._chunk_events(stream)
                        for slot_id in range(lo, hi):
                            buf.copy_expert(owner, slot_id, expert_list[slot_id])
                        if end is not None:
                            end.record(stream)
                        if hi < n:
                            # Pacing gap. Also makes this chunk's events complete,
                            # so reading them below costs nothing extra.
                            stream.synchronize()
                            if end is not None:
                                elapsed += start.elapsed_time(end)
                                timed += hi - lo
                else:
                    ring = self.prefetch_staging.num_slots
                    for lo in range(0, n, ring):
                        chunk_experts = expert_list[lo : lo + ring]
                        chunk_slots = list(range(lo, lo + len(chunk_experts)))
                        start, end = self._chunk_events(stream)
                        buf._dequant_chunk(
                            owner,
                            self.prefetch_staging,
                            num_bits,
                            self.quant_group_size,
                            chunk_experts,
                            chunk_slots,
                        )
                        if end is not None:
                            end.record(stream)
                        # Sync every ring-full: ring reuse, on-demand pacing, and
                        # completing this ring-full's timing events.
                        stream.synchronize()
                        if end is not None:
                            elapsed += start.elapsed_time(end)
                            timed += len(chunk_experts)

                event = torch.cuda.Event()
                event.record(stream)
                buf.prefetch_event = event

                if timed:
                    self._copy_times[num_bits].append(elapsed / timed)
        except Exception:
            # Nobody calls `.result()` on the future, so without this a bug here
            # is invisible: it degrades to a 0% hit rate and a mysteriously slow
            # model rather than a traceback.
            logger.exception("Expert prefetch failed for %s", owner.layer_name)
        finally:
            # Unblock a consumer even on failure (a stale map degrades to cache
            # misses, not a hang), and mark the worker fully exited for `reset`.
            buf.maps_ready.set()
            buf.copies_issued.set()

    def _chunk_events(
        self, stream: torch.cuda.Stream
    ) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
        """A recorded start event and its (unrecorded) end, when sampling."""
        if not self._sampling:
            return None, None
        start = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        return start, torch.cuda.Event(enable_timing=True)

