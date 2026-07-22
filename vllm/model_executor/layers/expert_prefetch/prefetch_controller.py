# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Picks how many experts per token to prefetch, from runtime measurements.

`prefetch_top_k` trades latency against PCIe bandwidth: staging fewer experts
than the router will pick shrinks the transfer, and the ones prediction gets
wrong are still served on demand. The right value is not a constant -- it
depends on how accurate the predictor happens to be, and on how much copying
actually fits underneath a layer's compute. This module measures both and solves
for it, per batch-size bucket, every `adapt_interval` forward passes.

Two constraints are combined.

**Confidence (Poisson).** Of the `K` experts a token routes to, the number the
predictor gets wrong is well modelled as Poisson with mean `lam = K*(1-acc)`,
since the per-expert error probability is small and `K` is fixed. Staging more
than `K - E_max` experts is wasted bandwidth: with confidence `C`, everything
past that is a misprediction that would have been fetched on demand anyway.

**Bubble.** Copies only overlap the compute they hide behind. `t_comp / t_e` is
how many experts fit under one layer's compute, and staging fewer than that
leaves bandwidth unused for no benefit -- hence `max`, not `min`: the Poisson
term says what is *useful*, the bubble term says what is *free*.

Both are per-token, but the bubble budget is a count of actual copies, and a
batch stages the *union* over its tokens rather than `prefetch_top_k` per token.
The union ratio is not analytic (it depends on how correlated routing is at that
batch size), so it is measured: `staged / prefetch_top_k` for the bucket.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.offload import ExpertCacheOffloadConfig

logger = init_logger(__name__)

# z-score for a one-sided confidence level, as used by the Gaussian
# approximation of the Poisson CDF.
Z_SCORES = {0.95: 1.645, 0.99: 2.326, 0.999: 3.090}

# A single interval should never move `p` by more than one expert, however
# extreme its measurements were.
_MAX_STEP = 1.0
# Below this many observed layers a bucket's statistics are too noisy to act on.
_MIN_SAMPLES = 8

# `t_e` is measured under contention with on-demand fetches, so it drifts with
# `p`. Smooth it hard and never let it wander far from the startup calibration.
_TE_ALPHA = 0.05
_TE_MIN_FACTOR = 0.5
_TE_MAX_FACTOR = 4.0

# Bit width standing for the native (bf16) originals -- always a candidate and
# the highest fidelity. Mirrors `expert_cache.NATIVE_BITS`.
NATIVE_BITS = 16


@dataclass
class _Accum:
    """Measurements accumulated for one bucket since the last `step`."""

    hits: int = 0
    needed: int = 0
    staged: int = 0
    layers: int = 0
    truncated: bool = False
    t_comp_ms: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.hits = 0
        self.needed = 0
        self.staged = 0
        self.layers = 0
        self.truncated = False
        self.t_comp_ms.clear()


@dataclass
class BucketState:
    """The adaptive state for one batch-size bucket.

    `p` is kept as a float even though only `floor(p)` is ever used: the
    corrections are often smaller than one expert, and rounding at every step
    would discard them instead of letting them accumulate.

    `precision` is the bit width the last `step` chose to stage this bucket at
    (NATIVE_BITS for bf16), read back by `select`.
    """

    p: float
    ema_acc: float = 0.0
    ema_t_comp_ms: float = 0.0
    samples: int = 0
    precision: int = NATIVE_BITS


class PrefetchController:
    """Solves for `prefetch_top_k` per batch-size bucket.

    Fed by `ExpertPrefetcher.on_forward_end`, read by `maybe_prefetch`. All
    inputs are host-side ints and floats; the caller is responsible for reading
    them off the device at a point where a sync is already being paid.
    """

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        num_slots: int,
        cfg: "ExpertCacheOffloadConfig",
        t_e_ms: dict[int, float],
    ):
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_slots = num_slots

        self.min_topk = max(1, cfg.prefetch_min_topk)
        self.max_topk = min(top_k, num_experts)
        if self.min_topk > self.max_topk:
            raise ValueError(
                f"prefetch_min_topk ({self.min_topk}) exceeds the model's top_k "
                f"({self.max_topk}); it can never be satisfied."
            )
        self.z_score = Z_SCORES[cfg.prefetch_confidence]
        self.alpha = cfg.prefetch_ema_alpha
        self.interval = cfg.prefetch_adapt_interval

        # A pinned value disables adaptation entirely: every bucket reports it
        # and no measurement can move it. `pinned_bits` fixes the precision too,
        # since the controller no longer picks one.
        self.pinned: int | None = cfg.prefetch_topk or None
        if self.pinned is not None and not 1 <= self.pinned <= num_experts:
            raise ValueError(
                f"prefetch_topk must be in [1, {num_experts}], got {self.pinned}."
            )
        self._pinned_bits = cfg.prefetch_pin_bits

        # Per-precision copy times, keyed by num_bits (NATIVE_BITS for bf16).
        self._t_e_ms = dict(t_e_ms)
        self._t_e_calibrated = dict(t_e_ms)
        # Candidate precisions, fidelity-descending (16, 8, 4, 2): numeric order
        # is fidelity order, so `k_bubble` rises monotonically down this list.
        self._precisions = sorted(self._t_e_ms, reverse=True)
        if self.pinned is not None and self._pinned_bits not in self._t_e_ms:
            raise ValueError(
                f"prefetch_pin_bits ({self._pinned_bits}) is not a staged "
                f"precision; expected {NATIVE_BITS} (bf16) or a resident width "
                f"in {sorted(b for b in self._t_e_ms if b != NATIVE_BITS)}."
            )
        self._states: dict[int, BucketState] = {}
        self._accum: dict[int, _Accum] = {}
        self._forwards = 0
        # Set by the last `step`, for logging only: (k_poisson, k_bubbles by
        # precision, chosen precision) per bucket.
        self._last_terms: dict[int, tuple[float, dict[int, float], int]] = {}

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    @staticmethod
    def bucket(num_tokens: int) -> int:
        """Power-of-two bucket: 1->1, 2->2, 3..4->3, 5..8->4, ..."""
        return max(1, num_tokens).bit_length()

    def select(self, num_tokens: int) -> tuple[int, int]:
        """The (prefetch_top_k, precision) to stage for a batch of this size.

        Precision is the width `step` chose for this bucket (NATIVE_BITS = bf16),
        or the pinned width when adaptation is disabled.
        """
        if self.pinned is not None:
            return self.pinned, self._pinned_bits
        state = self._state_for(num_tokens)
        return self._clamp_int(state.p), state.precision

    def topk_for(self, num_tokens: int) -> int:
        """How many experts per token to stage for a batch of this size."""
        return self.select(num_tokens)[0]

    def _state_for(self, num_tokens: int) -> BucketState:
        return self._state_for_bucket(self.bucket(num_tokens))

    def _state_for_bucket(self, bucket: int) -> BucketState:
        state = self._states.get(bucket)
        if state is None:
            state = BucketState(p=self._seed(bucket))
            self._states[bucket] = state
        return state

    def _seed(self, bucket: int) -> float:
        """Start a new bucket from its nearest populated neighbour.

        Adjacent buckets differ by a factor of two in token count, so their
        routing statistics are far closer than any constant would be. With
        nothing to copy from, start at the top: over-staging costs bandwidth
        that the bubble term was going to hand out anyway, while under-staging
        costs synchronous on-demand fetches.
        """
        if self._states:
            nearest = min(self._states, key=lambda b: abs(b - bucket))
            return self._states[nearest].p
        return float(self.max_topk)

    def _clamp_int(self, p: float) -> int:
        return int(min(max(math.floor(p), self.min_topk), self.max_topk))

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def observe(
        self,
        num_tokens: int,
        hits: int,
        needed: int,
        staged: int,
        layers: int,
        truncated: bool,
    ) -> None:
        """Record one forward pass worth of cache statistics.

        `hits`/`needed`/`staged` are summed over the MoE layers that actually
        ran against a prefetched buffer; layers whose buffer was dropped are the
        caller's job to exclude, since there everything is a miss by
        construction and would drag `acc` down for no reason.
        """
        if self.pinned is not None or layers <= 0:
            return
        accum = self._accum.setdefault(self.bucket(num_tokens), _Accum())
        accum.hits += hits
        accum.needed += needed
        accum.staged += staged
        accum.layers += layers
        accum.truncated |= truncated

    def observe_t_comp(self, num_tokens: int, t_comp_ms: float) -> None:
        """Record a per-layer compute time (H2D excluded), in milliseconds."""
        if self.pinned is not None or t_comp_ms <= 0.0:
            return
        self._accum.setdefault(self.bucket(num_tokens), _Accum()).t_comp_ms.append(
            t_comp_ms
        )

    def observe_t_e(self, t_e_ms: dict[int, float]) -> None:
        """Record measured per-expert copy times, per precision, in ms.

        Only the precision a forward actually staged at appears in `t_e_ms`; the
        others keep their calibrated seed. Each is clamped to a window around its
        own startup calibration: this is measured under contention with
        on-demand fetches, so a pathological interval can report a wildly
        inflated rate, and letting that through would shrink the bubble budget
        exactly when prefetching matters most. Widths without a calibration
        (never expected) are ignored.
        """
        if self.pinned is not None:
            return
        for bits, sample in t_e_ms.items():
            calibrated = self._t_e_calibrated.get(bits)
            if sample <= 0.0 or calibrated is None:
                continue
            lo = calibrated * _TE_MIN_FACTOR
            hi = calibrated * _TE_MAX_FACTOR
            sample = min(max(sample, lo), hi)
            self._t_e_ms[bits] = (
                1.0 - _TE_ALPHA
            ) * self._t_e_ms[bits] + _TE_ALPHA * sample

    def due_to_sample(self) -> bool:
        """Whether this forward pass should record timing events.

        One pass per interval carries them: the events are cheap but not free,
        and a single pass across every MoE layer is already a few dozen samples
        to average.
        """
        return self.pinned is None and self._forwards % self.interval == 0

    def on_forward_end(self) -> bool:
        """Advance the forward counter; True when an adaptation step is due.

        Sampling happens on the first pass of an interval and the step on the
        last, so the timing events have the rest of the interval to complete and
        are never waited on.
        """
        self._forwards += 1
        return self.pinned is None and self._forwards % self.interval == 0

    # ------------------------------------------------------------------
    # The solve
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Recompute `p` for every bucket that has enough fresh measurements."""
        if self.pinned is not None:
            return
        for bucket, accum in self._accum.items():
            # Created here if this bucket was measured before it was ever read:
            # `self._states.get` would silently skip adaptation for it.
            self._step_bucket(bucket, self._state_for_bucket(bucket), accum)
            accum.clear()

    def _step_bucket(self, bucket: int, state: BucketState, accum: _Accum) -> None:
        if accum.layers < _MIN_SAMPLES or accum.hits <= 0 or accum.needed <= 0:
            return

        p_cur = float(self._clamp_int(state.p))
        acc = accum.hits / accum.needed
        state.ema_acc = self._ema(state.ema_acc, acc, state.samples)

        k_poisson = self._poisson_target(state.ema_acc)

        k_bubbles: dict[int, float] = {}
        if accum.t_comp_ms:
            t_comp = sum(accum.t_comp_ms) / len(accum.t_comp_ms)
            state.ema_t_comp_ms = self._ema(state.ema_t_comp_ms, t_comp, state.samples)
            staged_avg = accum.staged / accum.layers
            # `staged_avg` (the union ratio) and `t_comp` are
            # precision-independent; only `t_e` varies, so one dict of bubble
            # targets -- one per candidate precision -- covers every width.
            k_bubbles = {
                bits: self._bubble_target(
                    state.ema_t_comp_ms, p_cur, staged_avg, self._t_e_ms[bits]
                )
                for bits in self._precisions
            }

        chosen_bits, p_raw = self._select_precision(k_poisson, k_bubbles)

        state.samples += accum.layers
        state.precision = chosen_bits
        self._last_terms[bucket] = (k_poisson, k_bubbles, chosen_bits)

        if accum.truncated:
            # The prefetch overflowed the cache and was cut down. Raising `p`
            # now would discard more of it, drop the hit rate further, and push
            # the bubble term to raise `p` again -- so only let it fall.
            p_raw = min(p_raw, p_cur)
        # No deadband: the EMA must be allowed to converge all the way onto the
        # target. Stopping it "close enough" strands `p` a fraction below the
        # next integer -- a target of 5.005 would settle at 4.9 and floor to 4.
        # Jitter is already damped by the EMA and bounded by `_MAX_STEP`.
        #
        # Smooth first, then rate-limit. The other order caps the EMA's *target*
        # at `p + 1`, which it can only ever approach asymptotically -- `p` would
        # stall a hair below the next integer and never cross it.
        smoothed = (1.0 - self.alpha) * state.p + self.alpha * p_raw
        state.p = min(
            max(smoothed, state.p - _MAX_STEP, float(self.min_topk)),
            state.p + _MAX_STEP,
            float(self.max_topk),
        )

        logger.debug(
            "[Prefetch] bucket=%d p=%.2f (%d) acc=%.3f k_poisson=%.2f "
            "precision=%s k_bubbles=%s t_comp=%.3fms%s",
            bucket,
            state.p,
            self._clamp_int(state.p),
            state.ema_acc,
            k_poisson,
            "bf16" if chosen_bits == NATIVE_BITS else f"int{chosen_bits}",
            {b: round(v, 2) for b, v in k_bubbles.items()},
            state.ema_t_comp_ms,
            " truncated" if accum.truncated else "",
        )

    def _select_precision(
        self, k_poisson: float, k_bubbles: dict[int, float]
    ) -> tuple[int, float]:
        """The precision to stage at, and the prefetch_top_k for it.

        `k_bubble` rises as fidelity drops (a smaller blob copies faster, so more
        fit under one layer's compute), so scanning fidelity-descending and
        taking the first width whose `k_bubble >= k_poisson` yields
        `min{k_opt1(p) : k_opt1(p) >= k_opt2}` at the highest such `p` -- the
        best accuracy that still stages enough experts to cover the Poisson
        target without a bubble. bf16 wins when the compute budget is large.

        If no width qualifies (even the coarsest is too slow to cover the target
        under compute), fall back to the coarsest resident width and the Poisson
        target, accepting some bubble -- exactly `max(k_opt1, k_opt2)` with the
        fastest precision. With quantization off, `_precisions` is just
        [NATIVE_BITS] and this reduces to the original `max` at bf16.
        """
        if not k_bubbles:
            # No compute-time sample yet: the bubble term is unsolved, so keep
            # the highest fidelity and let the Poisson target drive `p`.
            return NATIVE_BITS, k_poisson
        for bits in self._precisions:  # fidelity-descending: 16, 8, 4, 2
            if k_bubbles.get(bits, 0.0) >= k_poisson:
                return bits, k_bubbles[bits]
        return self._precisions[-1], k_poisson

    def _poisson_target(self, acc: float) -> float:
        """Experts per token worth staging at accuracy `acc`.

        `lam = K(1-acc)` is the expected number of mispredictions per token;
        `E_max = lam + Z*sqrt(lam)` bounds it at confidence `C` via the Gaussian
        approximation to the Poisson CDF. So at that confidence `K - E_max` of
        the top-`K` predictions are correct, and staging past it is bandwidth
        spent on experts that are very likely wrong.

        Absolute, not a correction to the current value: `acc` is measured
        against the predictor's full top-`K` picks (see `ExpertBuffer
        .reference_ids`), so it does not move when we stage more or less, and
        the target it implies is a fixed point rather than something to drift
        towards. A multiplicative form here could never raise `p` at all -- at
        `acc = 1` it returns exactly what it was given.
        """
        acc = min(max(acc, 0.0), 1.0)
        lam = self.top_k * (1.0 - acc)
        e_max = lam + self.z_score * math.sqrt(lam)
        return max(self.top_k - e_max, 0.0)

    def _bubble_target(
        self, t_comp_ms: float, p_cur: float, staged_avg: float, t_e_ms: float
    ) -> float:
        """Experts per token that fit under one layer's compute at this width.

        `t_comp / t_e` counts *copies* of a `t_e`-per-expert width, but `p` is
        per token and a batch stages the union over its tokens, so the budget is
        converted through the measured union size at the current `p`.
        """
        if t_comp_ms <= 0.0 or t_e_ms <= 0.0 or staged_avg <= 0.0:
            return 0.0
        budget = t_comp_ms / t_e_ms
        # Cannot stage more than the cache can hold, however much time there is.
        budget = min(budget, float(self.num_slots))
        return p_cur * budget / staged_avg

    @staticmethod
    def _ema(current: float, sample: float, samples: int) -> float:
        """Seed on the first observation instead of decaying up from zero."""
        if samples == 0 or current == 0.0:
            return sample
        return 0.7 * current + 0.3 * sample

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _bits_label(self, bits: int) -> str:
        return "bf16" if bits == NATIVE_BITS else f"int{bits}"

    def summary(self) -> str:
        if self.pinned is not None:
            return (
                f"[Prefetch] pinned top_k={self.pinned} "
                f"precision={self._bits_label(self._pinned_bits)}"
            )
        parts = []
        for bucket in sorted(self._states):
            state = self._states[bucket]
            poisson, k_bubbles, chosen = self._last_terms.get(
                bucket, (0.0, {}, state.precision)
            )
            bubble = k_bubbles.get(chosen, 0.0)
            parts.append(
                f"b{bucket}(n={1 << (bucket - 1)}..{(1 << bucket) - 1}): "
                f"p={self._clamp_int(state.p)}@{self._bits_label(chosen)} "
                f"acc={state.ema_acc:.3f} poisson={poisson:.1f} bubble={bubble:.1f}"
            )
        # Show every candidate precision's copy time and the raw bubble budget
        # (t_comp / t_e, experts that fit under one layer's compute) it implies,
        # at a representative t_comp -- so the widths the selector did *not*
        # pick are still visible. bf16 covering the target is why it wins.
        t_comp = max((s.ema_t_comp_ms for s in self._states.values()), default=0.0)
        te = " ".join(
            f"{self._bits_label(b)}(t_e={self._t_e_ms[b]:.3f}ms"
            + (
                f",budget={t_comp / self._t_e_ms[b]:.1f})"
                if self._t_e_ms[b] > 0.0
                else ")"
            )
            for b in self._precisions
        )
        return (
            f"[Prefetch] forwards={self._forwards} t_e=[{te}] | " + " | ".join(parts)
        )
