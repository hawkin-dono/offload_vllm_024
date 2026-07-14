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
    """

    def __init__(
        self,
        expert_cache_params: set[str],
        num_cache_slots: int = 0,
    ):
        if not expert_cache_params:
            raise ValueError(
                "expert_cache_params is empty, so the expert_cache offload "
                "backend would not offload anything. Either set it or use a "
                "different offload_backend."
            )
        self.expert_cache_params = expert_cache_params
        self.num_cache_slots = num_cache_slots
        self.pin_memory = should_pin_memory()
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
        return cpu_data.pin_memory() if self.pin_memory else cpu_data

    def post_init(self) -> None:
        """Re-pin offloaded parameters, then allocate the GPU caches.

        `process_weights_after_loading` runs each module under
        `device_loading_context`, which copies CPU parameters to the GPU, lets
        the quant method reshape/re-register them, and then restores them with a
        plain `.to("cpu")` — dropping the pinning we set up in `wrap_modules`.
        This runs after all of that, so it is the first point at which the
        parameters are both pinned-restorable and in their final runtime layout,
        which is exactly what the cache buffers have to mirror.
        """
        self._repin()

        for cache in self._caches:
            owner = cache.owner
            cache.allocate(
                param_names=self.cached_param_names(owner),
                default_num_slots=self.num_cache_slots,
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
