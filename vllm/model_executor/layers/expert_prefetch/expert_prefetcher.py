# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drives expert prefetching from inside a model's decoder layers.

Ties together the two halves of the feature: `ExpertPredictor` says which
experts the next MoE layer will want, `ExpertCache` stages them. A model opts in
by building one of these and calling `maybe_prefetch` at the two points where a
predictor input is available.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.expert_prefetch.expert_cache import (
    ExpertCache,
    maybe_create_expert_cache,
)
from vllm.model_executor.layers.expert_prefetch.expert_predictor import (
    ATTN_INPUT,
    MOE_INPUT,
    ExpertPredictor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)


class ExpertPrefetcher:
    """Predicts and stages the next MoE layer's experts, one layer ahead."""

    def __init__(
        self,
        cache: ExpertCache,
        predictor: ExpertPredictor | None,
        moe_layers: dict[int, "RoutedExperts"],
    ):
        self.cache = cache
        self.predictor = predictor
        self.moe_layers = moe_layers
        # Copies run here so they overlap the compute on the default stream.
        self.stream = torch.cuda.Stream()

    def maybe_prefetch(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        input_type: str,
    ) -> None:
        """Stage layer `layer_idx + 1`'s experts, if this is the signal its
        predictor was trained on.

        Called at both candidate points in every decoder layer; returns
        immediately unless this particular layer's predictor wants this
        particular input. A layer with no predictor is simply never prefetched,
        and falls back to fetch-on-demand.
        """
        if self.predictor is None:
            return
        if self.predictor.input_type(layer_idx) != input_type:
            return

        next_moe = self.moe_layers.get(layer_idx + 1)
        if next_moe is None:
            return

        expert_ids = self.predictor.predict(hidden_states.detach(), layer_idx)
        self.cache.prefetch(next_moe, expert_ids, self.stream)

    def on_forward_end(self) -> None:
        """Drop staged state so the next forward pass starts cold."""
        if self.cache.accuracy is not None:
            # The one point in the pass where reading the counters is safe: no
            # MoE layer is waiting on the sync it may cost.
            self.cache.accuracy.on_forward_end()
        self.cache.reset()


def _collect_moe_layers(layers: nn.ModuleList) -> dict[int, "RoutedExperts"]:
    """Map decoder layer index -> its routed experts, skipping dense layers and
    (under pipeline parallelism) layers that live on another rank."""
    moe_layers: dict[int, RoutedExperts] = {}
    for layer_idx, layer in enumerate(layers):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        routed_experts = getattr(experts, "routed_experts", None)
        if routed_experts is not None:
            moe_layers[layer_idx] = routed_experts
    return moe_layers


def maybe_create_expert_prefetcher(
    vllm_config: "VllmConfig",
    layers: nn.ModuleList,
) -> ExpertPrefetcher | None:
    """Build the prefetcher for a model, if expert offloading is enabled.

    Returns None when the expert_cache offload backend is not active, leaving
    the model on the stock fully-GPU-resident path.
    """
    offload_config = vllm_config.offload_config.expert_cache
    moe_layers = _collect_moe_layers(layers)
    cache = maybe_create_expert_cache(
        list(moe_layers.values()),
        log_accuracy_interval=offload_config.log_accuracy_interval,
    )
    if cache is None:
        return None

    predictor: ExpertPredictor | None = None
    if offload_config.expert_predictor_dir:
        model_config = vllm_config.model_config
        predictor = ExpertPredictor(
            checkpoint_dir=offload_config.expert_predictor_dir,
            device=torch.device(vllm_config.device_config.device),
            dtype=model_config.dtype,
        )
        _validate_predictor(predictor, moe_layers)
    else:
        logger.warning(
            "Expert cache is enabled but no expert_predictor_dir is set: experts "
            "will be fetched on demand, which is correct but synchronous. Set "
            "--expert-predictor-dir to prefetch them."
        )

    return ExpertPrefetcher(cache, predictor, moe_layers)


def _validate_predictor(
    predictor: ExpertPredictor,
    moe_layers: dict[int, "RoutedExperts"],
) -> None:
    """Catch a predictor trained for a different model than the one loaded.

    A mismatch here is not an error the cache would notice -- wrong expert ids
    are simply cache misses -- so it would show up only as a mysteriously low hit
    rate. Fail at load instead.
    """
    if not moe_layers:
        return
    num_experts = next(iter(moe_layers.values())).global_num_experts
    if predictor.num_experts != num_experts:
        raise ValueError(
            f"Expert predictor was trained for {predictor.num_experts} experts "
            f"but this model has {num_experts}. Wrong checkpoint directory?"
        )

    predicted = set(predictor.layer_to_input)
    # Predictor for layer i stages layer i+1, so only layers with a successor
    # that actually has experts are useful.
    prefetchable = {idx - 1 for idx in moe_layers}
    if not predicted & prefetchable:
        raise ValueError(
            "No expert predictor covers a layer that precedes an MoE layer. "
            f"Predictors exist for layers {sorted(predicted)[:5]}..., but the "
            f"MoE layers are {sorted(moe_layers)[:5]}.... Wrong checkpoints?"
        )
