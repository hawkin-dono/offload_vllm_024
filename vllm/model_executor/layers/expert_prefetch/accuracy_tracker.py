# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefetch accuracy accounting for the expert cache.

Of the experts an MoE layer turned out to route to, how many had the predictor
already staged? That ratio is what the whole feature is judged on: 1.0 means
every routed expert was prefetched, 0.0 means all of them were fetched on
demand, which is what an absent (or useless) predictor gets you.

Ground truth is the `topk_ids` the layer is about to run with, so the numbers
describe the routing that actually happened -- nothing is re-run to measure it.

Counters live on the GPU and are accumulated per layer without ever being read.
Reading one would put a host sync on the hot path of every MoE layer in the
model; instead the whole table is copied back at most once per forward pass.
"""

import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _layer_label(layer_name: str, fallback: int) -> str:
    """Best-effort decoder-layer number from a param prefix, for readable logs."""
    match = re.search(r"layers\.(\d+)", layer_name)
    return match.group(1) if match else str(fallback)


class AccuracyTracker:
    """Per-layer prefetch hit counters for one `ExpertCache`.

    Args:
        layer_names: The MoE layers sharing the cache, in model order. Fixes
            each layer's slot in the counter tensors.
        log_interval: Log a summary every this many forward passes. 0 disables
            logging; the counters are still maintained, since they cost two
            device-side adds per layer and `hit_rate` is how a run is verified.
    """

    def __init__(self, layer_names: list[str], log_interval: int = 0):
        self._layers = list(layer_names)
        self._index = {name: i for i, name in enumerate(self._layers)}
        self._log_interval = log_interval
        self._forwards = 0

        # Allocated on the CPU and moved in `allocate`: the cache knows its
        # device only once the expert weights are in their runtime layout.
        self._hits = torch.zeros(len(self._layers), dtype=torch.long)
        self._needed = torch.zeros(len(self._layers), dtype=torch.long)

    def allocate(self, device: torch.device) -> None:
        """Move the counters next to the buffers they describe."""
        self._hits = self._hits.to(device)
        self._needed = self._needed.to(device)

    def record(self, layer_name: str, hits: torch.Tensor, needed: int) -> None:
        """Add one layer's tally.

        Args:
            hits: Boolean mask over the layer's distinct routed experts, true
                where the expert was already staged. Summed on the device, so
                this does not sync.
            needed: How many distinct experts the layer routed to.
        """
        idx = self._index.get(layer_name)
        if idx is None:
            # A layer that never went through `bind` -- nothing sane to count.
            return
        self._hits[idx] += hits.sum()
        self._needed[idx] += needed

    def on_forward_end(self) -> None:
        """Log the running accuracy every `log_interval` forward passes."""
        self._forwards += 1
        if self._log_interval and self._forwards % self._log_interval == 0:
            self.log()

    def totals(self) -> tuple[int, int]:
        """(hits, needed) summed over every layer. Forces a device sync."""
        return int(self._hits.sum()), int(self._needed.sum())

    def hit_rate(self) -> float:
        """Fraction of needed experts that prediction had already staged.

        Forces a device sync, so read it between forward passes, not inside one.
        """
        hits, needed = self.totals()
        return hits / needed if needed else 0.0

    def per_layer_hit_rate(self) -> dict[str, float]:
        """Hit rate keyed by layer name, skipping layers that never ran."""
        # One D2H for the whole table rather than one per layer.
        hits = self._hits.tolist()
        needed = self._needed.tolist()
        return {
            name: hits[i] / needed[i]
            for i, name in enumerate(self._layers)
            if needed[i]
        }

    def log(self) -> None:
        per_layer = self.per_layer_hit_rate()
        if not per_layer:
            return
        hits, needed = self.totals()
        detail = " ".join(
            f"L{_layer_label(name, self._index[name])}={rate:.2f}"
            for name, rate in per_layer.items()
        )
        logger.info(
            "Expert prefetch accuracy: forwards=%d overall=%.3f "
            "(hits=%d, needed=%d) | %s",
            self._forwards,
            hits / needed if needed else 0.0,
            hits,
            needed,
            detail,
        )

    def reset(self) -> None:
        self._hits.zero_()
        self._needed.zero_()
        self._forwards = 0
