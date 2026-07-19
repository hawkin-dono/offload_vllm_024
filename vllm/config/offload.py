# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch", "expert_cache"]

# 0 disables quantization of the CPU-side expert store. One width for the whole
# model for now; the GPU staging ring is already sized so a per-expert choice
# costs no extra memory (see `DequantStaging`).
ExpertQuantBits = Literal[0, 2, 4, 8]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class ExpertCacheOffloadConfig:
    """Configuration for ping-pong expert-cache CPU offloading.

    Keeps routed MoE expert weights in pinned CPU memory and stages only the
    experts a layer actually needs into a small GPU cache. Unlike the other
    backends, staging is driven by the MoE forward path (predicted prefetch
    plus fetch-on-demand), not by a module forward hook.
    """

    expert_cache_params: set[str] = Field(
        default_factory=lambda: {
            "routed_experts.w13_weight",
            "routed_experts.w2_weight",
        }
    )
    """The set of parameter name segments to keep in CPU memory. Uses the same
    segment matching as `cpu_offload_params`: "routed_experts.w13_weight"
    matches "...mlp.experts.routed_experts.w13_weight" but not
    "...routed_experts.w13_weight_scale".

    Expert bias is deliberately absent: the fused kernel indexes it by expert id
    while the cached path passes slot indices, so MoE layers with bias are
    rejected at load time rather than cached.
    """

    num_cache_slots: int = Field(default=0, ge=0)
    """Number of expert slots in each of the two (ping/pong) GPU buffers.
    Default 0 means one slot per expert, i.e. a buffer can hold a whole layer.
    Lowering this trades GPU memory for a higher chance of a cache miss (which
    costs a synchronous fetch-on-demand), and must be at least as large as the
    number of distinct experts a single forward pass routes to.
    """

    expert_quant_bits: ExpertQuantBits = 0
    """Bit width of the CPU-side copy of routed expert weights that is staged
    over PCIe. Default 0 keeps them in their original dtype.

    Setting this to 2, 4 or 8 stores a second, symmetric round-to-nearest copy
    of each expert alongside the original and stages *that* across the bus,
    unpacking it on the GPU into the same cache slots the MoE kernel already
    reads. H2D traffic drops roughly 2x/4x/8x. In exchange pinned host memory
    grows by that fraction of the expert weights (the originals are kept), and
    accuracy takes the hit of uncalibrated RTN -- ~11% relative error on the
    weights at int4, and far more at int2, where a group has four levels.
    Evaluate before enabling; this is a model-quality change, not just a perf
    one.

    The GPU cost is fixed and small: the packed weights land in a shared
    dequant ring of a few experts rather than a buffer per cache slot, so the
    GPU cache stays the bf16 ping/pong plus ~150 MiB.
    """

    expert_quant_group_size: int = Field(default=128)
    """Quantization group size along the reduction (last) dim of each expert
    weight, or -1 for one scale per output row. Only used when
    `expert_quant_bits` is nonzero.

    The reduction dim -- `hidden_size` for w13_weight,
    `intermediate_size_per_partition` for w2_weight -- must be divisible by
    this. Note the latter is sharded by tensor parallelism, so a group size
    that works at tp=1 can fail at tp=4; use -1 in that case.
    """

    expert_predictor_dir: str = ""
    """Directory of trained expert-predictor checkpoints, named
    `{input_type}_layer_{first}_{last}.ckpt`. Each predicts, from layer i's
    hidden state, which experts layer i+1 will route to, so they can be
    prefetched a full layer ahead.

    Empty (the default) disables prediction: experts are then fetched on demand
    when the MoE layer discovers it needs them. That is correct but synchronous,
    and is the baseline prediction is measured against.

    How many experts each prediction stages is not configured here -- it is
    `ExpertPredictor.prefetch_top_k`, tuned at runtime.
    """


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
      Never selects "expert_cache", which must be requested explicitly.
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    - "expert_cache": Ping-pong GPU cache for routed MoE expert weights.
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    expert_cache: ExpertCacheOffloadConfig = Field(
        default_factory=ExpertCacheOffloadConfig
    )
    """Parameters for the expert-cache offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )

        if self.expert_cache.expert_quant_bits:
            group_size = self.expert_cache.expert_quant_group_size
            # A group must not straddle a byte boundary, and a byte holds
            # 8 // bits values: 4 at int2, 2 at int4, 1 at int8.
            per_byte = 8 // self.expert_cache.expert_quant_bits
            if group_size != -1 and (group_size < per_byte or group_size % per_byte):
                raise ValueError(
                    f"expert_quant_group_size ({group_size}) must be -1 (one "
                    f"scale per row) or a multiple of {per_byte} for "
                    f"expert_quant_bits={self.expert_cache.expert_quant_bits}"
                )
            if self.offload_backend != "expert_cache":
                warnings.warn(
                    "expert_quant_bits is set but offload_backend is "
                    f"'{self.offload_backend}'. It will be ignored.",
                    stacklevel=2,
                )
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
