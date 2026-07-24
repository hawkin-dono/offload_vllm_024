# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch", "expert_cache"]

# Resident CPU-side quantized precisions must be a subset of these. Mirrors
# `expert_quant.SUPPORTED_EXPERT_QUANT_BITS`, duplicated here to keep this
# config module free of the triton import `expert_quant` pulls in.
SUPPORTED_EXPERT_QUANT_BITS = (8, 4, 2)

# Sentinel bit width standing for the native (bf16) originals, which are always
# a prefetch candidate and cost no extra host memory. Mirrors
# `expert_cache.NATIVE_BITS`.
NATIVE_BITS = 16


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

    expert_quant_bits: list[int] = Field(default_factory=list)
    """Resident CPU-side quantized precisions of routed expert weights, a
    subset of {8, 4, 2}. Empty (the default) keeps weights in their original
    dtype and disables quantization.

    Each listed width stores a second, symmetric round-to-nearest copy of every
    expert in pinned host memory alongside the bf16 originals; per batch-size
    bucket the `PrefetchController` picks which of these -- or bf16 -- to stage
    over PCIe, unpacking it on the GPU into the same cache slots the MoE kernel
    reads. H2D traffic for a quantized width drops roughly 2x/4x/8x at
    int8/int4/int2. In exchange pinned host memory grows by the summed fraction
    of the expert weights (1/2 + 1/4 + 1/8 for all three), and accuracy takes
    the hit of uncalibrated RTN -- far more at int2, where a group has four
    levels. Evaluate before enabling; this is a model-quality change.

    bf16 is always an implicit candidate and costs no extra host memory (the
    offloaded originals). Fetch-on-demand always uses the coarsest resident
    width (int2 when present) to keep the critical-path stall smallest. The GPU
    cost is fixed and small: packed weights land in shared dequant rings of a
    few experts, so the GPU cache stays the bf16 ping/pong plus ~150 MiB.
    """

    expert_quant_group_size: int = Field(default=128)
    """Quantization group size along the reduction (last) dim of each expert
    weight, or -1 for one scale per output row. Only used when
    `expert_quant_bits` is non-empty.

    The reduction dim -- `hidden_size` for w13_weight,
    `intermediate_size_per_partition` for w2_weight -- must be divisible by
    this. Note the latter is sharded by tensor parallelism, so a group size
    that works at tp=1 can fail at tp=4; use -1 in that case. It must also be a
    multiple of the smallest resident width's values-per-byte (4 at int2).
    """

    prefetch_pin_bits: int = Field(default=NATIVE_BITS)
    """Precision to stage when `prefetch_topk` pins the staged count.

    Only consulted when `prefetch_topk` is nonzero (adaptation is off, so the
    controller cannot pick a precision). Default 16 stages bf16, reproducing
    the unquantized behaviour; set to a resident width (8, 4 or 2) to benchmark
    a fixed quantized prefetch. Fetch-on-demand is independent of this and still
    uses the coarsest resident width.
    """

    expert_predictor_dir: str = ""
    """Directory of trained expert-predictor checkpoints, named
    `{input_type}_layer_{first}_{last}.ckpt`. Each predicts, from layer i's
    hidden state, which experts layer i+1 will route to, so they can be
    prefetched a full layer ahead.

    Empty (the default) disables prediction: experts are then fetched on demand
    when the MoE layer discovers it needs them. That is correct but synchronous,
    and is the baseline prediction is measured against.

    How many experts each prediction stages is decided at runtime by
    `PrefetchController`; see `prefetch_topk` to pin it instead.
    """

    prefetch_topk: int = Field(default=0, ge=0)
    """How many experts per token to stage on each prediction.

    Default 0 lets `PrefetchController` solve for it from measured prediction
    accuracy and copy bandwidth, separately per batch-size bucket. Any other
    value pins it and disables adaptation, which is what to use when comparing
    against a fixed-size prefetch.
    """

    prefetch_confidence: float = Field(default=0.99)
    """Confidence level for the Poisson bound on mispredicted experts.

    The controller stages enough experts that, with this probability, it has not
    left out one the layer will route to. Higher values stage more and waste
    more bandwidth on predictions that turn out to be wrong. One of 0.95, 0.99,
    0.999.
    """

    prefetch_adapt_interval: int = Field(default=32, ge=1)
    """Forward passes between adaptation steps.

    Also sets how often timing events are recorded: one pass per interval
    carries them, and the step happens at the end of the interval so they have
    had time to complete and are never blocked on.
    """

    prefetch_num_chunks: int = Field(default=4, ge=1)
    """How many pieces to split a prefetch into.

    The copy stream is drained between chunks so that fetch-on-demand -- issued
    on the compute stream by a layer whose prediction missed -- gets the DMA
    engine instead of queueing behind the entire prefetch. 1 disables chunking.
    """

    prefetch_ema_alpha: float = Field(default=0.5, gt=0, le=1)
    """How fast the staged-expert count follows its target. Lower is steadier."""

    prefetch_min_topk: int = Field(default=1, ge=1)
    """Floor on the number of experts per token to stage."""

    prefetch_worker_mode: Literal["thread", "process"] = "thread"
    """Where the prefetch worker runs.

    "thread" (the default) issues the staging copies from a worker thread. That
    thread shares the interpreter (GIL) and the CUDA context with the model's
    forward pass, so its Python enqueue work and stream waits contend with the
    compute thread's own kernel launches.

    "process" runs the worker in a separate process with its own interpreter
    and CUDA context: the host expert store moves into file-backed shared
    memory (see `prefetch_shm_dir`) registered by both processes, the GPU cache
    buffers are shared via CUDA IPC, and readiness crosses over interprocess
    CUDA events. Linux-only. Costs one extra CUDA context (~a few hundred MB of
    GPU memory per rank). With `expert_quant_bits` set, the dequant kernels
    launch from the worker's context; run the CUDA MPS daemon so they share SMs
    with compute instead of time-slicing against it.
    """

    prefetch_shm_dir: str = "/dev/shm"
    """Directory for the shared host weight store's backing files (process
    mode only). Must be tmpfs/shm-like and large enough for the offloaded
    expert weights plus the quantized store; the default tmpfs at /dev/shm is
    typically capped at half of RAM.
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
        # Only these have a tabulated z-score; anything else would silently fall
        # back to some other confidence level.
        valid_confidences = (0.95, 0.99, 0.999)
        if self.expert_cache.prefetch_confidence not in valid_confidences:
            raise ValueError(
                f"prefetch_confidence ({self.expert_cache.prefetch_confidence}) "
                f"must be one of {valid_confidences}"
            )

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

        if self.expert_cache.prefetch_worker_mode == "process":
            import sys

            if not sys.platform.startswith("linux"):
                raise ValueError(
                    "prefetch_worker_mode='process' needs Linux: the shared "
                    "host weight store lives in file-backed shared memory "
                    "(/dev/shm) and the buffers cross processes via CUDA IPC."
                )

        quant_bits = self.expert_cache.expert_quant_bits
        if quant_bits:
            supported = set(SUPPORTED_EXPERT_QUANT_BITS)
            bad = sorted(b for b in set(quant_bits) if b not in supported)
            if bad:
                raise ValueError(
                    f"expert_quant_bits {bad} not supported; each must be one "
                    f"of {sorted(supported, reverse=True)}."
                )
            # Normalise to a deduped, fidelity-descending list so every
            # consumer sees the same order (bf16 first, then 8, 4, 2).
            quant_bits = sorted(set(quant_bits), reverse=True)
            self.expert_cache.expert_quant_bits = quant_bits

            group_size = self.expert_cache.expert_quant_group_size
            # The strictest packing constraint is the smallest resident width:
            # a byte holds 8 // bits values (4 at int2), and a group must not
            # straddle a byte boundary.
            per_byte = 8 // min(quant_bits)
            if group_size != -1 and (group_size < per_byte or group_size % per_byte):
                raise ValueError(
                    f"expert_quant_group_size ({group_size}) must be -1 (one "
                    f"scale per row) or a multiple of {per_byte} for the "
                    f"smallest expert_quant_bits ({min(quant_bits)})."
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

        The expert-cache prefetch policy knobs are the exception: they only
        change how many expert weights get staged and when, never the traced
        graph, so including them would invalidate the cache for nothing.
        `expert_quant_bits` and `expert_quant_group_size` are structural (they
        decide which pinned stores get built), so they stay in the hash.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors={"expert_cache"})
        # `ignored_factors` only reaches top-level fields, so the expert-cache
        # sub-config is re-added by hand with just its structural fields.
        factors["expert_cache"] = get_hash_factors(
            self.expert_cache,
            ignored_factors={
                "prefetch_topk",
                "prefetch_confidence",
                "prefetch_adapt_interval",
                "prefetch_num_chunks",
                "prefetch_ema_alpha",
                "prefetch_min_topk",
                "prefetch_pin_bits",
                # Where the worker runs and where the arena's files land never
                # change the traced graph. (`prefetch_worker_mode` does change
                # host store placement, but that is invisible to compilation.)
                "prefetch_worker_mode",
                "prefetch_shm_dir",
            },
        )
        hash_str = hash_factors(factors)
        return hash_str
