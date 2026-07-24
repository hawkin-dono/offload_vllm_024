# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefetch-accuracy accounting and logging for the expert cache."""

import os
import re

from vllm.logger import init_logger
from vllm.model_executor.layers.expert_prefetch.constants import NATIVE_BITS

logger = init_logger(__name__)

LOG_ACCURACY = os.getenv("LOG_ACCURACY", "0") == "1"
ACCURACY_LOG_INTERVAL = int(os.getenv("ACCURACY_LOG_INTERVAL", "1"))
# When set, the per-forward accuracy summary is also appended to this file.
ACCURACY_LOG_FILE = os.getenv("ACCURACY_LOG_FILE", "/tmp/vllm_expert_accuracy.log")


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
