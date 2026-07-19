# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predicts which experts the *next* MoE layer will route to.

A small MLP per decoder layer maps that layer's hidden state to a score per
expert of the following layer. Running it during layer `i` gives the prefetcher
a full layer of lead time to stage layer `i+1`'s experts (see `ExpertCache`).

Two input signals are supported, chosen per layer by whichever the predictor was
trained on:

  * ``attn_input`` -- the layer's raw (pre-norm) input, available before
    self-attention runs, so the prefetch overlaps the whole attention block.
  * ``moe_input``  -- the normalized tensor fed to the MoE router, available only
    after attention, so less overlap but a signal much closer to the routing
    decision.

Checkpoints are the ones produced by training, named
``{input_type}_layer_{first}_{last}.ckpt``. They are Lightning checkpoints, but
only the ``state_dict`` and ``hyper_parameters`` are read -- Lightning itself is
not a runtime dependency.
"""

import re
from pathlib import Path

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

ATTN_INPUT = "attn_input"
MOE_INPUT = "moe_input"
INPUT_TYPES = (ATTN_INPUT, MOE_INPUT)

# e.g. "attn_input_layer_10_10" -> ("attn_input", 10, 10)
_CKPT_STEM = re.compile(r"^(?P<input_type>.+)_layer_(?P<first>\d+)_(?P<last>\d+)$")


class _MLP(nn.Module):
    """`mlp`: Linear -> SiLU -> Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.nn.functional.silu(self.linear1(x)))


class _MLPv12(nn.Module):
    """`mlpv12`: Linear -> SiLU -> Linear -> SiLU -> Linear.

    Training had a dropout after the first activation; at inference it is the
    identity, so it is omitted here.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.linear1(x))
        x = torch.nn.functional.silu(self.linear2(x))
        return self.linear3(x)


_ARCHITECTURES = {"mlp": _MLP, "mlpv12": _MLPv12}


def _parse_stem(stem: str) -> tuple[str, int, int]:
    match = _CKPT_STEM.match(stem)
    if match is None:
        raise ValueError(
            f"Checkpoint name must look like '{{input_type}}_layer_{{first}}_"
            f"{{last}}.ckpt', got {stem!r}."
        )
    input_type = match.group("input_type")
    if input_type not in INPUT_TYPES:
        raise ValueError(
            f"Unknown predictor input type {input_type!r} in {stem!r}; "
            f"expected one of {INPUT_TYPES}."
        )
    return input_type, int(match.group("first")), int(match.group("last"))


def _build_from_checkpoint(path: Path) -> tuple[nn.Module, int, int]:
    """Rebuild one predictor from a Lightning checkpoint.

    Layer widths come from the tensor shapes rather than the hyperparameters:
    `hidden_dim` was left at its (architecture-dependent) default during
    training and so is not recorded in `hyper_parameters`.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]

    model_name = hparams["model_name"]
    architecture = _ARCHITECTURES.get(model_name)
    if architecture is None:
        raise ValueError(
            f"{path.name}: unknown predictor architecture {model_name!r}; "
            f"expected one of {sorted(_ARCHITECTURES)}."
        )

    # Lightning saved the inner module under a "model." prefix.
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }

    input_dim = state_dict["linear1.weight"].shape[1]
    hidden_dim = state_dict["linear1.weight"].shape[0]
    num_experts = int(hparams["num_experts"])

    predictor = architecture(input_dim, hidden_dim, num_experts)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor, int(hparams["top_k"]), num_experts


class ExpertPredictor(nn.Module):
    """The per-layer predictors for one model, keyed by decoder layer index.

    `predictor[i]` consumes layer `i`'s hidden state and scores the experts of
    layer `i+1`. Layers with no checkpoint simply have no predictor: their
    successor is not prefetched and falls back to fetch-on-demand.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: torch.device,
        dtype: torch.dtype,
        prefetch_top_k: int = 0,
    ):
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Expert predictor checkpoint directory not found: {checkpoint_dir}"
            )

        self.layer_to_input: dict[int, str] = {}
        # Held off the module tree deliberately: keyed by layer, not by name.
        self._predictors = nn.ModuleDict()

        top_ks: set[int] = set()
        expert_counts: set[int] = set()
        for path in sorted(checkpoint_dir.glob("*.ckpt")):
            input_type, first, last = _parse_stem(path.stem)
            predictor, top_k, num_experts = _build_from_checkpoint(path)
            top_ks.add(top_k)
            expert_counts.add(num_experts)

            for layer_idx in range(first, last + 1):
                if layer_idx in self.layer_to_input:
                    raise ValueError(
                        f"Two predictor checkpoints claim layer {layer_idx}; "
                        f"{path.name} conflicts with an earlier one."
                    )
                self.layer_to_input[layer_idx] = input_type
                self._predictors[str(layer_idx)] = predictor

        if not self.layer_to_input:
            raise FileNotFoundError(
                f"No predictor checkpoints (*.ckpt) found in {checkpoint_dir}."
            )
        if len(top_ks) != 1:
            raise ValueError(f"Predictors disagree on the model's top_k: {top_ks}.")
        if len(expert_counts) != 1:
            raise ValueError(
                f"Predictors disagree on the expert count: {expert_counts}."
            )

        self.top_k = top_ks.pop()
        self.num_experts = expert_counts.pop()
        self._prefetch_top_k = prefetch_top_k or max(1, int(self.top_k))

        self.to(device=device, dtype=dtype)
        logger.info(
            "Expert predictor: %d layers from %s (top_k=%d, prefetching top-%d)",
            len(self.layer_to_input),
            checkpoint_dir,
            self.top_k,
            self.prefetch_top_k,
        )

    @property
    def prefetch_top_k(self) -> int:
        """How many experts per token to stage.

        Tunable at runtime (by profiling, or an adaptive policy): predicting
        fewer experts than the router will actually pick shrinks the prefetch
        and the PCIe traffic with it, while the ones it gets wrong are still
        served on demand. So this trades latency against bandwidth, and can be
        moved freely between forward passes without affecting correctness.
        """
        return self._prefetch_top_k

    @prefetch_top_k.setter
    def prefetch_top_k(self, value: int) -> None:
        if not 1 <= value <= self.num_experts:
            raise ValueError(
                f"prefetch_top_k must be in [1, {self.num_experts}], got {value}."
            )
        self._prefetch_top_k = value

    def input_type(self, layer_idx: int) -> str | None:
        """Which hidden state layer `layer_idx`'s predictor was trained on, or
        None if that layer has no predictor."""
        return self.layer_to_input.get(layer_idx)

    @torch.inference_mode()
    def predict(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Experts that layer `layer_idx + 1` is likely to route to.

        Returns the union over the batch's tokens, as a 1-D tensor of global
        expert ids.
        """
        logits = self._predictors[str(layer_idx)](hidden_states)
        top = torch.topk(logits, self._prefetch_top_k, dim=-1).indices
        return torch.unique(top.reshape(-1))
