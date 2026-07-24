# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ping-pong expert-cache CPU offloading for routed MoE experts.

Routed expert weights are parked in pinned CPU memory and staged into a small
GPU cache one layer at a time. This backend only owns the CPU side: it decides
which parameters live on the host and keeps their storage pinned so that H2D
copies can be async. The GPU-side staging (predicted prefetch on a side stream,
fetch-on-demand for cache misses, ping/pong flip) is driven by the MoE forward
path, so unlike `UVAOffloader` this backend installs no module forward hook.
"""

from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    QUANT_STORE_ATTR,
    SUPPORTED_EXPERT_QUANT_BITS,
    QuantizedExpertWeight,
    blob_aliases,
    quantize_experts,
)
from vllm.model_executor.layers.expert_prefetch.shared_host_weights import (
    DEFAULT_SHM_DIR,
    SharedHostArena,
)
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.utils.mem_utils import format_gib

if TYPE_CHECKING:
    from vllm.model_executor.layers.expert_prefetch import ExpertCache

logger = init_logger(__name__)


class ExpertCacheOffloader(BaseOffloader):
    """Keeps routed expert weights in pinned CPU memory.

    Args:
        expert_cache_params: Parameter name segments to keep on CPU. Matched
            per segment, so "routed_experts.w13_weight" matches
            "...experts.routed_experts.w13_weight" but not "...w13_weight_scale".
        num_cache_slots: Experts per ping/pong GPU buffer. 0 means one slot per
            expert. Consumed by the MoE layer when it allocates the cache.
        quant_bits: Resident quantized precisions of the CPU-side expert store,
            a subset of {8, 4, 2}. Empty keeps the weights in their original
            dtype. Each width builds a second pinned copy the cache can stage.
        quant_group_size: Scale group along the reduction dim, or -1 for one
            scale per row. Only used when `quant_bits` is non-empty.
        worker_mode: "thread" keeps the host store in process-private pinned
            memory. "process" places it in a `SharedHostArena` instead, so the
            prefetch worker process can register and copy from the same pages.
        shm_dir: Directory for the arena's backing files ("process" mode only).
    """

    def __init__(
        self,
        expert_cache_params: set[str],
        num_cache_slots: int = 0,
        quant_bits: tuple[int, ...] = (),
        quant_group_size: int = 128,
        worker_mode: str = "thread",
        shm_dir: str = DEFAULT_SHM_DIR,
    ):
        bad = sorted(b for b in set(quant_bits) if b not in SUPPORTED_EXPERT_QUANT_BITS)
        if bad:
            raise ValueError(
                f"quant_bits {bad} not supported; each must be one of "
                f"{sorted(SUPPORTED_EXPERT_QUANT_BITS)}."
            )
        if not expert_cache_params:
            raise ValueError(
                "expert_cache_params is empty, so the expert_cache offload "
                "backend would not offload anything. Either set it or use a "
                "different offload_backend."
            )
        self.expert_cache_params = expert_cache_params
        self.num_cache_slots = num_cache_slots
        # Fidelity-descending and deduped so every consumer sees the same order.
        self.quant_bits = tuple(sorted(set(quant_bits), reverse=True))
        self.quant_group_size = quant_group_size
        self.pin_memory = should_pin_memory()
        self.worker_mode = worker_mode
        self.shm_dir = shm_dir
        # In process mode the host store lives in shared memory instead of
        # process-private pinned pages, so the worker process can reach it.
        # Pinning then happens per process, via `SharedHostArena.register`.
        self.arena: SharedHostArena | None = (
            SharedHostArena(shm_dir) if worker_mode == "process" else None
        )
        self.offloaded_bytes = 0

        # (owning module, dotted param name) rather than Parameter refs:
        # process_weights_after_loading may re-register the parameter object
        # (replace_parameter), which would leave a captured ref stale.
        self._offloaded: list[tuple[nn.Module, str]] = []
        self._caches: list[ExpertCache] = []

    def _should_offload(self, name: str) -> bool:
        # Dots on both sides so we only ever match whole segments.
        return any(f".{p}." in f".{name}." for p in self.expert_cache_params)

    def cached_param_names(self, owner: nn.Module) -> tuple[str, ...]:
        """The expert params of `owner` this backend offloads, e.g.
        ("w13_weight", "w2_weight"). These are exactly the tensors the GPU cache
        has to shadow."""
        names = (p.rpartition(".")[2] for p in sorted(self.expert_cache_params))
        return tuple(n for n in names if hasattr(owner, n))

    def register_expert_cache(self, cache: "ExpertCache") -> None:
        """Hand the GPU-side cache to the offloader so it can be allocated once
        weight loading has settled (see `post_init`)."""
        self._caches.append(cache)

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        modules = [self._offload_experts(module) for module in modules_generator]

        if self.offloaded_bytes == 0:
            raise ValueError(
                "The expert_cache offload backend matched no parameters. "
                f"expert_cache_params={sorted(self.expert_cache_params)} did "
                "not match any parameter name. Note that in the current MoE "
                "layout expert weights are nested under `routed_experts`, e.g. "
                "`mlp.experts.routed_experts.w13_weight`."
            )

        logger.info(
            "Expert cache: offloaded %s of routed expert weights to CPU%s",
            format_gib(self.offloaded_bytes),
            "" if self.pin_memory else " (unpinned; H2D copies will be sync)",
        )
        return modules

    def _offload_experts(self, module: nn.Module) -> nn.Module:
        if (params := next(module.parameters(), None)) is None:
            return module
        if params.device == torch.device("cpu"):
            return module

        for name, p in module.named_parameters():
            if not self._should_offload(name):
                continue
            p.data = self._to_host(p.data)
            self.offloaded_bytes += p.data.numel() * p.data.element_size()
            self._offloaded.append((module, name))

        return module

    def _to_host(self, data: torch.Tensor) -> torch.Tensor:
        cpu_data = data.to(device="cpu")
        # In process mode the final home is the shared arena (post_init moves
        # the tensor there and registration pins it); pinning here would only
        # be thrown away with this interim copy.
        if self.arena is not None:
            return cpu_data
        return cpu_data.pin_memory() if self.pin_memory else cpu_data

    def _arena_key(self, module: nn.Module, name: str) -> str:
        """A stable, unique arena key for `module`'s parameter `name`.

        The dotted `name` repeats across decoder layers, so the key is rooted
        at the owning RoutedExperts' `layer_name` instead -- which is also how
        the worker process finds the tensor again.
        """
        parent, _, leaf = name.rpartition(".")
        target = module.get_submodule(parent) if parent else module
        return f"{target.layer_name}.{leaf}"

    def post_init(self) -> None:
        """Re-pin offloaded parameters, then allocate the GPU caches.

        `process_weights_after_loading` runs each module under
        `device_loading_context`, which copies CPU parameters to the GPU, lets
        the quant method reshape/re-register them, and then restores them with a
        plain `.to("cpu")` — dropping the pinning we set up in `wrap_modules`.
        This runs after all of that, so it is the first point at which the
        parameters are both pinned-restorable and in their final runtime layout,
        which is exactly what the cache buffers have to mirror.

        In process mode the parameters move into the shared arena here instead
        (same timing, same reason), and the arena is registered -- the
        process-mode analog of pinning -- once the quantized store has joined
        it.
        """
        if self.arena is None:
            self._repin()
        else:
            self._move_to_arena()
        self._quantize_offloaded()
        if self.arena is not None:
            self.arena.register()

        for cache in self._caches:
            owner = cache.owner
            cache.allocate(
                param_names=self.cached_param_names(owner),
                default_num_slots=self.num_cache_slots,
                quant_bits=self.quant_bits,
                quant_group_size=self.quant_group_size,
            )
            if self.arena is not None:
                cache.start_process_worker(self.arena, self.shm_dir)

        if self.arena is not None:
            # Every worker has attached (its ready message arrived inside
            # start_process_worker): the mappings keep the pages alive, so the
            # names can go away now -- a crashed run leaks nothing in /dev/shm.
            self.arena.unlink()

    def _move_to_arena(self) -> None:
        """Move every offloaded parameter into the shared arena, one at a time
        so peak host memory stays one parameter above steady state (mirroring
        `_repin`, which reallocates the same way)."""
        arena = self.arena
        assert arena is not None
        # Everything that will land in the arena: the bf16 originals plus one
        # quantized mirror per resident width (~bits/16 of the originals each,
        # padded up a little for the interleaved scales).
        quant_fraction = sum(bits / 16 * 1.05 for bits in self.quant_bits)
        arena.check_capacity(int(self.offloaded_bytes * (1 + quant_fraction)))
        moved = 0
        for module, name in self._offloaded:
            p = module.get_parameter(name)
            if p.device.type != "cpu":
                continue
            p.data = arena.add(self._arena_key(module, name), p.data.contiguous())
            moved += 1
        logger.info(
            "Expert cache: moved %d expert parameters (%s) into the shared host arena",
            moved,
            format_gib(arena.nbytes()),
        )

    def _quantize_offloaded(self) -> None:
        """Build a pinned quantized mirror of every offloaded expert weight, one
        per resident width.

        Runs here rather than in `wrap_modules` for the same reason `_repin`
        does: this is the first point at which the weights are in their final
        runtime layout, which is what the mirror has to reproduce. The store is
        nested `{leaf: {num_bits: QuantizedExpertWeight}}` so the cache can pick
        a precision per fetch.
        """
        if not self.quant_bits:
            return

        device = torch.device(torch.cuda.current_device())
        quantized_bytes = 0
        for module, name in self._offloaded:
            p = module.get_parameter(name)
            # `module` is the decoder layer, but the cache looks the store up on
            # the RoutedExperts that owns the parameter, by leaf name.
            parent, _, leaf = name.rpartition(".")
            target = module.get_submodule(parent) if parent else module
            # Through __dict__ so the store stays out of state_dict().
            per_leaf = target.__dict__.setdefault(QUANT_STORE_ATTR, {}).setdefault(
                leaf, {}
            )
            for bits in self.quant_bits:
                store = quantize_experts(
                    p.data,
                    bits,
                    self.quant_group_size,
                    device,
                    # In process mode the blob's final home is the arena; a
                    # pinned interim copy would be pure setup cost.
                    self.pin_memory and self.arena is None,
                    name=leaf,
                )
                if self.arena is not None:
                    key = f"{self._arena_key(module, name)}:int{bits}"
                    blob = self.arena.add(key, store.blob)
                    qweight, scale = blob_aliases(blob, store.layout)
                    store = QuantizedExpertWeight(
                        blob=blob,
                        layout=store.layout,
                        qweight=qweight,
                        scale=scale,
                    )
                per_leaf[bits] = store
                quantized_bytes += store.nbytes()

        # The chunk buffers are freed, but the allocator still holds their
        # reserved blocks, which would skew the later memory profiling.
        torch.cuda.empty_cache()

        logger.info(
            "Expert cache: int%s store adds %s of pinned host memory (%.0f%% of "
            "the original expert weights, which are kept)",
            "/".join(str(b) for b in self.quant_bits),
            format_gib(quantized_bytes),
            100 * quantized_bytes / max(self.offloaded_bytes, 1),
        )

    def _repin(self) -> None:
        if not self.pin_memory:
            return

        repinned = 0
        for module, name in self._offloaded:
            p = module.get_parameter(name)
            if p.device.type != "cpu" or p.data.is_pinned():
                continue
            p.data = p.data.pin_memory()
            repinned += 1

        if repinned:
            logger.debug(
                "Expert cache: re-pinned %d/%d expert parameters after loading",
                repinned,
                len(self._offloaded),
            )
