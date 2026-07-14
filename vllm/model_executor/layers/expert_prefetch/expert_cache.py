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

Slots are addressed by position, not by expert id: `cached_expert_ids[i]` is the
global expert id currently staged in slot `i` (or -1 if the slot is empty), and
the MoE forward remaps `topk_ids` through that table before invoking the kernel.
Experts predicted incorrectly are simply cache misses: they get fetched on
demand, which is correct but synchronous.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

EMPTY_SLOT = -1


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

        # cached_expert_ids[slot] = global expert id staged there, or EMPTY_SLOT.
        self.cached_expert_ids: torch.Tensor = torch.empty(0, dtype=torch.int32)

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

    def allocate(
        self,
        owner: "RoutedExperts",
        num_slots: int,
        device: torch.device,
    ) -> None:
        self.num_slots = num_slots
        for name in self.param_names:
            src = getattr(owner, name)
            self.params[name] = torch.zeros(
                (num_slots, *src.shape[1:]),
                dtype=src.dtype,
                device=device,
            )
        self.cached_expert_ids = torch.full(
            (num_slots,), EMPTY_SLOT, dtype=torch.int32, device=device
        )

    def wait_until_ready(self) -> None:
        """Block until this buffer's staged weights are usable.

        Waits for the worker thread to finish enqueuing copies, then makes the
        current stream wait on the copy stream. Cheap when no prefetch is in
        flight, since both events are already set.
        """
        self.copies_issued.wait()
        if self.prefetch_event is not None:
            torch.cuda.current_stream().wait_event(self.prefetch_event)
            self.prefetch_event = None

    def fetch(
        self,
        owner: "RoutedExperts",
        expert_ids: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> None:
        """Copy `expert_ids` from CPU into `slot_ids` of this buffer.

        The copies are async only because each `src[expert_id]` is a *view* into
        the pinned CPU storage the offloader set up. Gathering the rows first
        (`src[expert_ids]`) would allocate a new, unpinned tensor and silently
        make every copy synchronous — do not "optimize" this loop into a
        batched index.
        """
        for slot_id, expert_id in zip(slot_ids.tolist(), expert_ids.tolist()):
            for name in self.param_names:
                src = getattr(owner, name)
                self.params[name][slot_id].copy_(src[expert_id], non_blocking=True)


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
        # were already staged. Allocated with the buffers, on the GPU.
        self._hits: torch.Tensor = torch.zeros((), dtype=torch.long)
        self._needed: torch.Tensor = torch.zeros((), dtype=torch.long)

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
        if num_slots < owner.top_k:
            raise ValueError(
                f"num_cache_slots ({num_slots}) is smaller than the layer's "
                f"top_k ({owner.top_k}); a single token would not fit in the "
                "cache."
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
        self._hits = torch.zeros((), dtype=torch.long, device=device)
        self._needed = torch.zeros((), dtype=torch.long, device=device)
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

    def hit_rate(self) -> float:
        """Fraction of needed experts that prediction had already staged.

        1.0 means every expert a layer routed to was prefetched; 0.0 means all
        were fetched on demand (which is what you get with no predictor). Forces
        a device sync, so read it between forward passes, not inside one.
        """
        needed = int(self._needed)
        return float(self._hits) / needed if needed else 0.0

    def reset_stats(self) -> None:
        self._hits.zero_()
        self._needed.zero_()

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
            if buf.num_slots:
                buf.cached_expert_ids.fill_(EMPTY_SLOT)
        self.active_name = "ping"

    def resolve(
        self,
        owner: "RoutedExperts",
        topk_ids: torch.Tensor,
    ) -> tuple[ExpertBuffer, torch.Tensor]:
        """Make the active buffer usable for `owner`, and remap `topk_ids`.

        Waits for any in-flight prefetch, fetches whatever the predictor missed,
        and rewrites `topk_ids` from global expert ids into cache slot indices.
        Returns the buffer to run against, and the remapped ids.
        """
        buf = self.active
        buf.wait_until_ready()

        # If this buffer was not staged for *this* layer, nothing in it can be
        # trusted (see `ExpertBuffer.staged_for`). Dropping the id table turns
        # every expert into a miss, which is slow but always correct -- this is
        # also the path taken when prediction is disabled entirely.
        if buf.staged_for != owner.layer_name:
            buf.cached_expert_ids.fill_(EMPTY_SLOT)
            buf.staged_for = owner.layer_name

        slot_ids = buf.cached_expert_ids
        needed = torch.unique(topk_ids.reshape(-1)).to(slot_ids.device, torch.int32)

        hit = torch.isin(needed, slot_ids)
        # Accumulate on-device: reading these would force a sync, so the tally is
        # kept on the GPU and only materialized by `hit_rate`.
        self._hits += hit.sum()
        self._needed += needed.numel()

        if not bool(hit.all()):
            missing = needed[~hit]
            # A slot is evictable unless it holds an expert this layer needs.
            # Empty slots (EMPTY_SLOT) are never "needed", so they go first only
            # by virtue of ordering -- correctness does not depend on that.
            evictable = torch.nonzero(~torch.isin(slot_ids, needed), as_tuple=True)[0]
            if evictable.numel() < missing.numel():
                raise RuntimeError(
                    f"Expert cache too small: layer {owner.layer_name} routes to "
                    f"{needed.numel()} experts but the cache has "
                    f"{buf.num_slots} slots. Raise --num-cache-slots."
                )
            victims = evictable[: missing.numel()]
            slot_ids[victims] = missing
            # Issued on the current stream, so the MoE kernel that follows is
            # ordered after these copies without an explicit sync.
            buf.fetch(owner, missing.cpu(), victims.cpu())

        # topk_ids are global expert ids; the kernel indexes the cache buffer, so
        # they have to become slot indices.
        lookup = torch.full(
            (owner.global_num_experts,),
            EMPTY_SLOT,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )
        occupied = slot_ids >= 0
        positions = torch.arange(buf.num_slots, device=slot_ids.device)
        lookup[slot_ids[occupied].long()] = positions[occupied].to(topk_ids.dtype)

        cached_topk_ids = lookup[topk_ids]
        return buf, cached_topk_ids

    def prefetch(
        self,
        owner: "RoutedExperts",
        expert_ids: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        """Stage `expert_ids` of `owner` (the *next* MoE layer) into the
        inactive buffer.

        Returns as soon as the copies are handed to a worker thread; the
        consumer synchronizes via `ExpertBuffer.wait_until_ready`.
        """
        buf = self.inactive
        # The buffer we are about to overwrite must not still be in flight.
        buf.wait_until_ready()

        expert_ids = torch.unique(expert_ids.reshape(-1))[: buf.num_slots]

        # The copy stream must not run ahead of the compute stream. It reads
        # `expert_ids`, which the predictor just produced there, and it
        # overwrites a buffer the *previous* layer's MoE kernel may still be
        # reading. Both are silent data races without this: the copies land
        # while the reads are in flight, and the corruption surfaces as subtly
        # wrong logits, not a crash.
        
        stream.wait_stream(torch.cuda.current_stream())
        
        # Keep `expert_ids` alive until the copy stream is done with it, since
        # the allocator only tracks the stream it was created on.
        
        expert_ids.record_stream(stream)

        buf.staged_for = owner.layer_name
        buf.copies_issued.clear()
        device = buf.cached_expert_ids.device
        self._executor.submit(
            self._prefetch_worker, owner, buf, expert_ids, stream, device
        )

    def _prefetch_worker(
        self,
        owner: "RoutedExperts",
        buf: ExpertBuffer,
        expert_ids: torch.Tensor,
        stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        try:
            torch.cuda.set_device(device)
            with torch.cuda.stream(stream):
                slot_ids = torch.arange(expert_ids.numel(), dtype=torch.long)
                buf.fetch(owner, expert_ids.cpu(), slot_ids)

                ids = buf.cached_expert_ids
                ids.fill_(EMPTY_SLOT)
                ids[: expert_ids.numel()] = expert_ids.to(device, dtype=torch.int32)

                event = torch.cuda.Event()
                event.record(stream)
                buf.prefetch_event = event
        finally:
            buf.copies_issued.set()
