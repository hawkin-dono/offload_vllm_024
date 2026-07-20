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
    LOG_ACCURACY,
    ExpertCache,
    accuracy_tracker,
    maybe_create_expert_cache,
)
from vllm.model_executor.layers.expert_prefetch.expert_predictor import (
    ExpertPredictor,
)
from vllm.model_executor.layers.expert_prefetch.prefetch_controller import (
    PrefetchController,
)
from vllm.v1.utils import record_function_or_nullcontext

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.offload import ExpertCacheOffloadConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)


class ExpertPrefetcher:
    """Predicts and stages the next MoE layer's experts, one layer ahead."""

    def __init__(
        self,
        cache: ExpertCache,
        predictor: ExpertPredictor | None,
        moe_layers: dict[int, "RoutedExperts"],
        config: "ExpertCacheOffloadConfig | None" = None,
    ):
        self.cache = cache
        self.predictor = predictor
        self.moe_layers = moe_layers
        self.config = config
        self.num_chunks = config.prefetch_num_chunks if config else 1
        # Copies run here so they overlap the compute on the default stream.
        self.stream = torch.cuda.Stream()
        # Built on first use: it needs the cache's slot count and a calibrated
        # copy time, and neither exists until the offloader's `post_init` has
        # allocated the buffers -- which happens after the model is constructed.
        self.controller: PrefetchController | None = None
        # The batch size this forward pass is running at. Fixed for the whole
        # pass, so the controller's per-bucket state cannot be split across it.
        self._num_tokens = 0

    def _ensure_controller(self) -> PrefetchController | None:
        if self.controller is None and self.config is not None and self.predictor:
            self.controller = PrefetchController(
                top_k=self.predictor.top_k,
                num_experts=self.predictor.num_experts,
                num_slots=self.cache.num_slots,
                cfg=self.config,
                t_e_ms=self.cache.calibrate_copy_time(),
            )
        return self.controller

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

        hidden_states = hidden_states.detach()
        num_tokens = hidden_states.shape[0] if hidden_states.dim() > 1 else 1
        controller = self._ensure_controller()
        if self._num_tokens == 0:
            self._num_tokens = num_tokens
            # The controller only needs timings every `adapt_interval` passes,
            # but the accuracy log prints every one, so accuracy tracking forces
            # them on. That also feeds the controller more samples than it would
            # otherwise get: LOG_ACCURACY already syncs per layer, so this is a
            # debug mode either way, not a configuration to measure against.
            self.cache.begin_forward(
                sampling=LOG_ACCURACY
                or (controller is not None and controller.due_to_sample())
            )

        # The predictor is two GEMMs and an activation -- microseconds, and
        # nothing on a profile timeline names them. Without a scope it is
        # indistinguishable from the model's own kernels on the same stream.
        with record_function_or_nullcontext("expert_prefetch: predict"):
            logits = self.predictor.predict(hidden_states, layer_idx)
        top_k = self.predictor.top_k
        prefetch_top_k = controller.topk_for(num_tokens) if controller else top_k

        if LOG_ACCURACY:
            # Attributed to the layer being staged, so the summary can print it
            # next to the hit rate that same layer goes on to report.
            accuracy_tracker.record_topk(next_moe.layer_name, prefetch_top_k)
            logger.debug(
                "Prefetching for layer %d, input %s, num_tokens=%d, top_k=%d, "
                "prefetch_top_k=%d",
                layer_idx,
                input_type,
                num_tokens,
                top_k,
                prefetch_top_k,
            )
        # Rank by predictor score, not by expert id: what does not fit the cache
        # is dropped from the tail, and `torch.unique` would sort numerically and
        # so drop by id -- a systematically biased subset rather than the least
        # likely experts.
        with record_function_or_nullcontext("expert_prefetch: rank"):
            expert_ids = _rank_unique(logits, prefetch_top_k)

            # The full top-`K` picks, for measuring accuracy only. Nothing is
            # copied for these; the cache just intersects them with what the
            # layer routed to, which is the accuracy the Poisson model is
            # written in terms of.
            reference_ids = None
            if controller is not None and prefetch_top_k < top_k:
                reference_ids = _rank_unique(logits, top_k)

        # Covers only the handoff to the worker thread, not the copies: those
        # run on `self.stream` and outlive this scope by design.
        with record_function_or_nullcontext("expert_prefetch: issue"):
            self.cache.prefetch(
                next_moe,
                expert_ids,
                self.stream,
                num_chunks=self.num_chunks,
                reference_ids=reference_ids,
            )

    def on_forward_end(self) -> None:
        """Feed the controller this pass's measurements, then start cold."""
        # Before the summary, not after: `_adapt` is what drains the timing
        # events, so logging first would always report the previous pass's.
        if self.controller is not None:
            self._adapt()
        if LOG_ACCURACY:
            accuracy_tracker.on_forward_end()
        self._num_tokens = 0
        self.cache.reset()

    def _adapt(self) -> None:
        assert self.controller is not None
        stats = self.cache.drain_stats()
        num_tokens = self._num_tokens
        if num_tokens:
            self.controller.observe(
                num_tokens=num_tokens,
                hits=stats.reference_hits,
                needed=stats.needed,
                staged=stats.staged,
                layers=stats.layers,
                truncated=stats.truncated,
            )
            for t_comp in stats.t_comp_ms:
                self.controller.observe_t_comp(num_tokens, t_comp)
        self.controller.observe_t_e(stats.t_e_ms)
        if LOG_ACCURACY:
            accuracy_tracker.record_timings(stats.t_comp_ms, stats.t_e_ms)
        if self.controller.on_forward_end():
            self.controller.step()
            logger.debug("%s", self.controller.summary())


def _rank_unique(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """The union of each token's top-`top_k` experts, best-scoring first.

    An expert is scored by its best score over the batch, so a prefix of the
    result is always the most likely experts -- which is what makes truncating
    it safe when the union outgrows the cache.
    """
    num_experts = logits.shape[-1]
    top = torch.topk(logits, top_k, dim=-1)
    ids = top.indices.reshape(-1)
    scores = top.values.reshape(-1).float()

    floor = torch.finfo(torch.float32).min
    best = torch.full((num_experts,), floor, device=logits.device)
    best.scatter_reduce_(0, ids, scores, reduce="amax", include_self=True)

    candidates = torch.nonzero(best > floor, as_tuple=True)[0]
    return candidates[torch.argsort(best[candidates], descending=True)]
    # return torch.unique(ids)

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
    moe_layers = _collect_moe_layers(layers)
    cache = maybe_create_expert_cache(list(moe_layers.values()))
    if cache is None:
        return None

    offload_config = vllm_config.offload_config.expert_cache
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

    if predictor is not None and not vllm_config.model_config.enforce_eager:
        # `resolve` reads expert ids back to the host to decide what to fetch,
        # and the prefetch worker records events from another thread; neither is
        # capturable. Without this the graph captures one forward's cache state
        # and replays it forever, which is wrong rather than merely slow.
        raise ValueError(
            "The expert_cache offload backend requires --enforce-eager: the "
            "cache makes host-side decisions inside the MoE forward, which "
            "cannot be captured into a CUDA graph."
        )

    return ExpertPrefetcher(cache, predictor, moe_layers, offload_config)


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
