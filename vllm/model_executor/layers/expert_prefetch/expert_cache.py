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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

EMPTY_SLOT = -1


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
    t_e_ms: float


@dataclass
class _TCompSample:
    """Events for one layer's compute time, excluding H2D.

    `t_comp = elapsed(issue, begin) - elapsed(prev_begin, prev_ready)`

    The first term is the whole window a prefetch had to hide behind: from when
    it was issued, in the previous decoder layer, to when this layer's MoE block
    needs the weights. The second subtracts the previous layer's dead time --
    its stall waiting for its own prefetch, plus its on-demand fetches, which
    are the only H2D on the compute stream.
    """

    issue: torch.cuda.Event
    begin: torch.cuda.Event
    prev_begin: torch.cuda.Event
    prev_ready: torch.cuda.Event

    def ready(self) -> bool:
        return self.begin.query() and self.prev_ready.query()

    def elapsed_ms(self) -> float:
        window = self.issue.elapsed_time(self.begin)
        dead = self.prev_begin.elapsed_time(self.prev_ready)
        return window - dead


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
        self._t_e_sum = 0.0
        self._t_e_count = 0
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

    def record_timings(self, t_comp_ms: list[float], t_e_ms: float) -> None:
        """Record one forward's compute and copy times, in milliseconds.

        Not per layer: `t_comp_ms` arrives as a flat list of per-layer windows
        with no layer attached, and `t_e_ms` is a property of the link rather
        than of any one layer. Both are also drained lazily -- an event that had
        not completed by the last drain is reported a forward or two late -- so
        these are running means over the whole run, not this pass's values.
        """
        self._t_comp_sum += sum(t_comp_ms)
        self._t_comp_count += len(t_comp_ms)
        if t_e_ms > 0.0:
            self._t_e_sum += t_e_ms
            self._t_e_count += 1

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
        t_e = self._t_e_sum / self._t_e_count if self._t_e_count else 0.0
        # `t_comp/t_e` is the bubble budget the controller solves against: how
        # many expert copies fit under one layer's compute. Printed alongside so
        # a `p` that looks wrong can be traced to whichever term produced it.
        budget = t_comp / t_e if t_e > 0.0 else 0.0
        summary = (
            f"[ExpertAcc] forwards={self._forwards} overall={self.overall():.3f} "
            f"(hits={sum(self._hits.values())}, "
            f"needed={sum(self._needed.values())}) topk={overall_topk:.1f} "
            f"t_comp={t_comp:.3f}ms t_e={t_e:.3f}ms budget={budget:.1f} "
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
        self._t_e_sum = 0.0
        self._t_e_count = 0
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
        self.copies_issued = threading.Event()
        self.copies_issued.set()
        self.prefetch_event: torch.cuda.Event | None = None

        # Recorded on the compute stream when this buffer's prefetch was issued.
        # Paired with the consuming `resolve` to measure how much compute the
        # copies had to hide behind. None when timing is not being sampled.
        self.issue_event: torch.cuda.Event | None = None

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

    def wait_copies_issued(self) -> None:
        """Block (host-side) until the worker has enqueued the copies.

        This is what publishes the slot<->expert maps (`stage_ids` runs before
        `copies_issued` is set), so it is all `resolve` needs before it can remap
        `topk_ids` on the host. Crucially it does NOT wait for the copies to
        *execute* on the GPU -- that is `wait_prefetch_event`'s job -- so a
        blocking D2H issued right after is not serialized behind the staging H2D.
        """
        self.copies_issued.wait()

    def wait_prefetch_event(self) -> None:
        """Make the current (compute) stream wait for the staged copies to land.

        GPU-side only: enqueues a wait on the compute stream and returns without
        blocking the host. Split from `wait_copies_issued` so the (large) staging
        H2D is only waited on right before the MoE kernel reads the weights, not
        before the unrelated `topk_ids` D2H.
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
        self.wait_copies_issued()
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

    def fetch(
        self,
        owner: "RoutedExperts",
        expert_ids: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> None:
        """Copy `expert_ids` from CPU into `slot_ids` of this buffer."""
        for slot_id, expert_id in zip(slot_ids.tolist(), expert_ids.tolist()):
            self.copy_expert(owner, slot_id, expert_id)


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
        # Per-expert copy times measured by the worker thread, in milliseconds.
        self._copy_times: deque[float] = deque(maxlen=64)

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
        self, param_names: tuple[str, ...], default_num_slots: int = 0
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

        num_experts = owner.local_num_experts
        num_slots = self._requested_slots or default_num_slots or num_experts
        if num_slots > num_experts:
            num_slots = num_experts
        if num_slots < owner.local_num_experts:
            raise ValueError(
                f"num_cache_slots ({num_slots}) is smaller than the layer's "
                f"local_num_experts ({owner.local_num_experts}); can not handle the prefill phase. Raise --num-cache-slots or set expert_cache.num_cache_slots in the config."
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
        self._hits = 0
        self._needed = 0
        self.stats = CacheStats()
        self.allocated = True

        bytes_per_buffer = sum(
            t.numel() * t.element_size() for t in self.ping.params.values()
        )
        logger.info(
            "Expert cache: 2 x %d slots (%d experts/layer), %.2f GiB on GPU",
            num_slots,
            num_experts,
            2 * bytes_per_buffer / 1024**3,
        )

    def calibrate_copy_time(self, num_samples: int = 8, warmup: int = 2) -> float:
        """Time a single expert's H2D copy, in milliseconds.

        Run once from `allocate`, so the controller's bubble constraint has a
        sane value before enough forward passes have gone by to measure one.
        Off the hot path, so a full sync is fine here.
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
        for i in range(warmup):
            buf.copy_expert(owner, i % buf.num_slots, i % num_experts)
        torch.cuda.synchronize()

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for i in range(num_samples):
            buf.copy_expert(owner, i % buf.num_slots, i % num_experts)
        end.record()
        torch.cuda.synchronize()

        # Leave no trace: these slots hold weights nothing staged, and the id
        # maps must not claim otherwise.
        buf.clear_ids()
        buf.staged_for = None

        t_e = start.elapsed_time(end) / num_samples
        bytes_per_expert = sum(
            t[0].numel() * t[0].element_size() for t in buf.params.values()
        )
        logger.info(
            "Expert cache: %.3f ms per expert copy (%.1f MiB, %.1f GiB/s)",
            t_e,
            bytes_per_expert / 1024**2,
            bytes_per_expert / (t_e * 1e-3) / 1024**3,
        )
        return t_e

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
            t_e_ms=(
                sum(self._copy_times) / len(self._copy_times)
                if self._copy_times
                else 0.0
            ),
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
            buf.issue_event = None
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

    def _close_window(
        self, buf: ExpertBuffer, begin: torch.cuda.Event | None
    ) -> None:
        """Finish this layer's timing window and pair it with the last one."""
        if begin is None:
            return
        ready = torch.cuda.Event(enable_timing=True)
        ready.record()
        if buf.issue_event is not None and self._prev_events is not None:
            prev_begin, prev_ready = self._prev_events
            self.stats.t_comp_events.append(
                _TCompSample(
                    issue=buf.issue_event,
                    begin=begin,
                    prev_begin=prev_begin,
                    prev_ready=prev_ready,
                )
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

        # Host-side wait only: this publishes the slot<->expert maps. The staged
        # weights may still be copying on the GPU -- we defer waiting on them
        # (`wait_prefetch_event`) until just before they are read, so the tiny
        # `topk_ids` D2H below is not serialized behind the (MiB, ~1ms) staging
        # H2D. The D2H depends only on the router's topk, not on the prefetch.
        buf.wait_copies_issued()

        # If this buffer was not staged for *this* layer, nothing in it can be
        # trusted (see `ExpertBuffer.staged_for`). Dropping the maps turns every
        # expert into a miss -- slow but always correct, and the path taken when
        # prediction is disabled entirely.
        prefetched = buf.staged_for == owner.layer_name
        if not prefetched:
            buf.clear_ids()
            buf.staged_for = owner.layer_name

        # The one D2H: bring the routing to the host. `topk_ids` is (T, K) ints.
        topk_np = self._to_host(topk_ids)
        flat = topk_np.reshape(-1)

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

        # Only now make the compute stream wait for the staged weights: the
        # on-demand fetches below write into the same buffer, and the MoE kernel
        # that follows reads it, so both must be ordered after the prefetch.
        buf.wait_prefetch_event()

        # Fetch whatever missed. Host-side decision; the weight copies go on the
        # compute stream so the MoE kernel is ordered after them.
        if miss.any():
            if needed is None:
                needed = np.unique(flat)
            cached = self._resolve_misses(owner, buf, flat, cached, miss, needed)

        # Everything between `begin` and here is dead time: the D2H stall plus
        # the on-demand copies. The next layer subtracts it.
        self._close_window(buf, begin)

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
        buf.slot_to_expert[victims] = missing
        buf.expert_to_slot[missing] = victims
        buf.fetch(owner, missing, victims)

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
    ) -> None:
        """Stage `expert_ids` of `owner` (the *next* MoE layer) into the
        inactive buffer.

        `expert_ids` must already be deduplicated and ranked best-first: when it
        does not fit the cache the tail is dropped, so the caller's ordering
        decides what survives. Returns as soon as the copies are handed to a
        worker thread; the consumer synchronizes via `wait_until_ready`.
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

        buf.issue_event = self._record_event()
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
    ) -> None:
        try:
            torch.cuda.set_device(device)
            # Published to the consuming `resolve` via `copies_issued` (set in
            # `finally`). The main thread no longer writes these, so the ranking
            # that produced them stays off the compute stream.
            buf.staged_for = owner.layer_name
            with torch.cuda.stream(stream):
                expert_list = expert_ids.cpu().tolist()
                n = len(expert_list)
                # Reference ids are host-side telemetry; bring them over here on
                # the side stream (off the main thread), where they were produced.
                buf.reference_ids = (
                    None
                    if reference_ids is None
                    else reference_ids.detach().cpu().numpy()
                )

                # Copies go out in chunks with a drain between them, so that
                # on-demand fetches -- issued on the compute stream by a layer
                # that mispredicted -- get the DMA engine instead of queueing
                # behind the whole prefetch. Within a chunk we only enqueue:
                # draining after *every* copy would serialize the transfer, and
                # ordering the copies against their reader is `prefetch_event`'s
                # job, which it does on the GPU without blocking this thread.
                elapsed = 0.0
                timed = 0
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

                # Record the slot<->expert maps on the host, from the ids we
                # already have on CPU, so `resolve` reads them without any GPU
                # round-trip. Staged experts are distinct (no collisions).
                buf.stage_ids(np.asarray(expert_list, dtype=np.int32))

                event = torch.cuda.Event()
                event.record(stream)
                buf.prefetch_event = event

                if timed:
                    self._copy_times.append(elapsed / timed)
        except Exception:
            # Nobody calls `.result()` on the future, so without this a bug here
            # is invisible: it degrades to a 0% hit rate and a mysteriously slow
            # model rather than a traceback.
            logger.exception("Expert prefetch failed for %s", owner.layer_name)
        finally:
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

