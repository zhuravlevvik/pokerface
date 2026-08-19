"""Reusable supervised warm-up for the poker model backbone and auxiliary heads.

This module deliberately knows nothing about corpus generation, curriculum
stages, command-line entry points, or the UI.  It accepts any sequence whose
items expose the same fields as ``curriculum.PretrainingExample``:
``observation``, ``selected_action``, ``equity_target``,
``expected_showdown_share_target`` and ``terminal_pnl_bb``.  That keeps generated datasets, archived traces, and
future streaming adapters interchangeable at this boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import os
import random
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from uuid import uuid4

from .equity import (
    EquityMetrics,
    ExpectedShowdownShareMetrics,
    equity_cross_entropy,
    equity_metrics,
    expected_showdown_share_binary_cross_entropy,
    expected_showdown_share_metrics,
)
from .model import ACTION_NAMES, BET_SIZE_ACTIONS, TORCH_AVAILABLE, PokerAgentModel

if TORCH_AVAILABLE:
    import torch
    from torch.nn import functional as F


PRETRAINING_CHECKPOINT_VERSION = 2


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for poker.pretraining; install the project with `.[rl]`.")


class PretrainingExampleLike(Protocol):
    """Structural contract accepted by :class:`EquityBackbonePretrainer`."""

    observation: Mapping[str, object]
    selected_action: str
    equity_target: Sequence[float]
    expected_showdown_share_target: float
    terminal_pnl_bb: float


BreakdownHook = Callable[[PretrainingExampleLike], str]


@dataclass(frozen=True)
class PretrainingConfig:
    """Configuration for deterministic minibatch supervised warm-up.

    Outcome soft-target cross entropy and scalar expected-showdown-share BCE
    are the primary losses.  Behaviour
    cloning and value warm-up are deliberately opt-in: they are useful for a
    gentle initialisation, but should not force a rule bot's strategy or noisy
    terminal results into the policy by default.
    """

    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 1
    seed: int = 0
    behavior_cloning_coefficient: float = 0.0
    value_warmup_coefficient: float = 0.0
    expected_showdown_share_coefficient: float = 1.0
    value_huber_delta: float = 10.0
    max_grad_norm: float | None = 1.0
    weight_decay: float = 0.0
    calibration_bins: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if min(
            self.behavior_cloning_coefficient,
            self.value_warmup_coefficient,
            self.expected_showdown_share_coefficient,
        ) < 0:
            raise ValueError("pretraining loss coefficients must be non-negative")
        if self.value_huber_delta <= 0:
            raise ValueError("value_huber_delta must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or null")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.calibration_bins < 1:
            raise ValueError("calibration_bins must be positive")


@dataclass(frozen=True)
class PretrainingMetrics:
    """Mean losses from one completed epoch at a safe resume boundary."""

    epoch: int
    global_step: int
    samples: int
    total_loss: float
    equity_loss: float
    behavior_cloning_loss: float
    value_warmup_loss: float
    expected_showdown_share_loss: float


@dataclass(frozen=True)
class PretrainingValidation:
    """Equity quality on a holdout, optionally sliced by user-provided hooks."""

    equity: EquityMetrics
    breakdowns: Mapping[str, Mapping[str, EquityMetrics]]
    predictions: tuple[tuple[float, float, float], ...]
    expected_showdown_share: ExpectedShowdownShareMetrics
    expected_showdown_share_breakdowns: Mapping[str, Mapping[str, ExpectedShowdownShareMetrics]]
    expected_showdown_share_predictions: tuple[float, ...]


def _as_examples(examples: Sequence[PretrainingExampleLike] | Iterable[PretrainingExampleLike]) -> tuple[PretrainingExampleLike, ...]:
    rows = tuple(examples)
    if not rows:
        raise ValueError("pretraining examples must not be empty")
    for example in rows:
        for name in ("observation", "selected_action", "equity_target", "expected_showdown_share_target", "terminal_pnl_bb"):
            if not hasattr(example, name):
                raise TypeError(f"pretraining example has no {name!r} field")
        if not isinstance(example.observation, Mapping):
            raise TypeError("pretraining example observation must be a mapping")
        if not isinstance(example.selected_action, str):
            raise TypeError("pretraining example selected_action must be a string")
    return rows


def _action_labels(examples: Sequence[PretrainingExampleLike], device):
    """Map engine actions to the policy's type + conditional-size labels."""

    _require_torch()
    action_indices: list[int] = []
    size_indices: list[int] = []
    for example in examples:
        action = example.selected_action
        if action in BET_SIZE_ACTIONS:
            action_indices.append(ACTION_NAMES.index("raise"))
            size_indices.append(BET_SIZE_ACTIONS.index(action))
        elif action in ACTION_NAMES and action != "raise":
            action_indices.append(ACTION_NAMES.index(action))
            size_indices.append(-1)
        else:
            # A bare ``raise`` does not identify an engine action, therefore
            # cannot correctly supervise the conditional sizing distribution.
            raise ValueError(f"unsupported selected action for behaviour cloning: {action!r}")
    return (
        torch.tensor(action_indices, dtype=torch.long, device=device),
        torch.tensor(size_indices, dtype=torch.long, device=device),
    )


def factorized_behavior_cloning_loss(output, action_indices, bet_size_indices):
    """Negative log likelihood under ``P(type) * P(size | type=raise)``.

    The model supplies legal-masked logits.  For non-raises, no fictitious
    sizing likelihood is included.  This is the same factorisation used by
    PPO, exposed separately so supervised callers and tests can audit it.
    """

    _require_torch()
    action_log_probs = F.log_softmax(output.action_logits, dim=-1)
    log_probability = action_log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    is_raise = action_indices == ACTION_NAMES.index("raise")
    if bool(is_raise.any()):
        selected_sizes = bet_size_indices[is_raise]
        if bool((selected_sizes < 0).any()):
            raise ValueError("raise behaviour-cloning samples need a raise-size label")
        size_log_probs = F.log_softmax(output.bet_size_logits[is_raise], dim=-1)
        log_probability = log_probability.clone()
        log_probability[is_raise] += size_log_probs.gather(1, selected_sizes.unsqueeze(1)).squeeze(1)
    return -log_probability.mean()


def _rng_state() -> dict[str, Any]:
    _require_torch()
    result: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    _require_torch()
    python_state, torch_state = state.get("python"), state.get("torch")
    if python_state is None or torch_state is None:
        raise ValueError("checkpoint has incomplete random-number-generator state")
    random.setstate(python_state)
    torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint was saved with CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Publish a fully-written checkpoint, never a partial destination file."""

    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:  # Some network filesystems do not allow directory fsync.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


class EquityBackbonePretrainer:
    """Train model representations from soft showdown labels before PPO.

    Minibatch permutations depend only on ``config.seed`` and the completed
    epoch number.  Thus a checkpoint written after an epoch resumes with the
    exact next permutation without relying on a DataLoader worker state.
    """

    def __init__(
        self,
        model: PokerAgentModel,
        config: PretrainingConfig | None = None,
        *,
        provenance: Mapping[str, str] | None = None,
    ) -> None:
        _require_torch()
        self.model = model
        self.config = config or PretrainingConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.epoch = 0
        self.global_step = 0
        self.provenance = dict(provenance or {})
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in self.provenance.items()):
            raise TypeError("pretraining provenance must map strings to strings")

    def _indices_for_epoch(self, size: int) -> list[int]:
        generator = torch.Generator(device="cpu")
        # The multiplication avoids the common early-epoch seeds occupying a
        # suspiciously correlated range while remaining fully reproducible.
        generator.manual_seed(self.config.seed + self.epoch * 1_000_003)
        return torch.randperm(size, generator=generator).tolist()

    def train_epoch(self, examples: Sequence[PretrainingExampleLike] | Iterable[PretrainingExampleLike]) -> PretrainingMetrics:
        rows = _as_examples(examples)
        self.model.train()
        totals = {"total": 0.0, "equity": 0.0, "share": 0.0, "behavior": 0.0, "value": 0.0}
        processed = 0
        permutation = self._indices_for_epoch(len(rows))
        for start in range(0, len(rows), self.config.batch_size):
            indices = permutation[start : start + self.config.batch_size]
            batch = tuple(rows[index] for index in indices)
            output = self.model([item.observation for item in batch])
            targets = torch.tensor([item.equity_target for item in batch], dtype=torch.float32, device=self.device)
            equity_loss = equity_cross_entropy(output.equity_logits, targets)
            share_targets = torch.tensor(
                [item.expected_showdown_share_target for item in batch],
                dtype=torch.float32,
                device=self.device,
            )
            share_loss = expected_showdown_share_binary_cross_entropy(
                output.expected_showdown_share_logit,
                share_targets,
            )
            behavior_loss = torch.zeros((), dtype=equity_loss.dtype, device=self.device)
            if self.config.behavior_cloning_coefficient:
                action_indices, size_indices = _action_labels(batch, self.device)
                behavior_loss = factorized_behavior_cloning_loss(output, action_indices, size_indices)
            value_loss = torch.zeros((), dtype=equity_loss.dtype, device=self.device)
            if self.config.value_warmup_coefficient:
                values = torch.tensor([item.terminal_pnl_bb for item in batch], dtype=torch.float32, device=self.device)
                value_loss = F.huber_loss(output.value, values, reduction="mean", delta=self.config.value_huber_delta)
            loss = (
                equity_loss
                + self.config.expected_showdown_share_coefficient * share_loss
                + self.config.behavior_cloning_coefficient * behavior_loss
                + self.config.value_warmup_coefficient * value_loss
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            count = len(batch)
            processed += count
            totals["total"] += float(loss.detach().item()) * count
            totals["equity"] += float(equity_loss.detach().item()) * count
            totals["share"] += float(share_loss.detach().item()) * count
            totals["behavior"] += float(behavior_loss.detach().item()) * count
            totals["value"] += float(value_loss.detach().item()) * count
            self.global_step += 1
        self.epoch += 1
        return PretrainingMetrics(
            epoch=self.epoch,
            global_step=self.global_step,
            samples=processed,
            total_loss=totals["total"] / processed,
            equity_loss=totals["equity"] / processed,
            behavior_cloning_loss=totals["behavior"] / processed,
            value_warmup_loss=totals["value"] / processed,
            expected_showdown_share_loss=totals["share"] / processed,
        )

    def fit(
        self,
        examples: Sequence[PretrainingExampleLike] | Iterable[PretrainingExampleLike],
        *,
        epochs: int | None = None,
    ) -> tuple[PretrainingMetrics, ...]:
        """Run complete epochs; every returned metric is safe to checkpoint."""

        count = self.config.epochs if epochs is None else epochs
        if count < 1:
            raise ValueError("epochs must be positive")
        rows = _as_examples(examples)
        return tuple(self.train_epoch(rows) for _ in range(count))

    def evaluate(
        self,
        examples: Sequence[PretrainingExampleLike] | Iterable[PretrainingExampleLike],
        *,
        breakdowns: Mapping[str, BreakdownHook] | None = None,
    ) -> PretrainingValidation:
        """Return holdout logloss, Brier and ECE, including requested slices."""

        rows = _as_examples(examples)
        was_training = self.model.training
        self.model.eval()
        predictions: list[tuple[float, float, float]] = []
        share_predictions: list[float] = []
        try:
            with torch.no_grad():
                for start in range(0, len(rows), self.config.batch_size):
                    batch = rows[start : start + self.config.batch_size]
                    output = self.model([item.observation for item in batch])
                    predictions.extend(tuple(float(value) for value in row) for row in output.equity_probabilities.detach().cpu().tolist())
                    share_predictions.extend(float(value) for value in output.expected_showdown_share.detach().cpu().tolist())
        finally:
            if was_training:
                self.model.train()
        targets = [tuple(float(value) for value in item.equity_target) for item in rows]
        overall = equity_metrics(predictions, targets, bins=self.config.calibration_bins)
        share_targets = [float(item.expected_showdown_share_target) for item in rows]
        overall_share = expected_showdown_share_metrics(share_predictions, share_targets, bins=self.config.calibration_bins)
        sliced: dict[str, dict[str, EquityMetrics]] = {}
        sliced_shares: dict[str, dict[str, ExpectedShowdownShareMetrics]] = {}
        for name, hook in (breakdowns or {}).items():
            groups: dict[str, list[int]] = {}
            for index, example in enumerate(rows):
                groups.setdefault(str(hook(example)), []).append(index)
            sliced[name] = {
                key: equity_metrics([predictions[index] for index in indices], [targets[index] for index in indices], bins=self.config.calibration_bins)
                for key, indices in groups.items()
            }
            sliced_shares[name] = {
                key: expected_showdown_share_metrics(
                    [share_predictions[index] for index in indices],
                    [share_targets[index] for index in indices],
                    bins=self.config.calibration_bins,
                )
                for key, indices in groups.items()
            }
        return PretrainingValidation(
            overall,
            sliced,
            tuple(predictions),
            overall_share,
            sliced_shares,
            tuple(share_predictions),
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        """A weights-only-safe payload that inference can load as a model."""

        return {
            "pretraining_checkpoint_version": PRETRAINING_CHECKPOINT_VERSION,
            "metadata": self.model.checkpoint_metadata(),
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pretraining_config": asdict(self.config),
            "progress": {"epoch": self.epoch, "global_step": self.global_step},
            "rng": _rng_state(),
            "provenance": dict(self.provenance),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save the model, optimizer, progress and RNG state."""

        destination = Path(path)
        _atomic_torch_save(self.checkpoint_payload(), destination)
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        model: PokerAgentModel | None = None,
        map_location: str | None = "cpu",
        device: str | None = None,
    ) -> "EquityBackbonePretrainer":
        """Restore an epoch-boundary checkpoint and its exact next minibatch."""

        _require_torch()
        # An explicit device wins over map_location: a CUDA-produced artifact
        # can therefore always be opened and continued on CPU.
        load_location = device if device is not None else map_location
        payload = torch.load(Path(path), map_location=load_location, weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("pretraining_checkpoint_version") != PRETRAINING_CHECKPOINT_VERSION:
            raise ValueError("not a compatible pretraining checkpoint")
        config_data, progress, rng = payload.get("pretraining_config"), payload.get("progress"), payload.get("rng")
        optimizer_state = payload.get("optimizer_state_dict")
        provenance = payload.get("provenance", {})
        if not isinstance(config_data, Mapping) or not isinstance(progress, Mapping) or not isinstance(rng, Mapping) or not isinstance(optimizer_state, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("checkpoint has incomplete pretraining state")
        restored_config = PretrainingConfig(**dict(config_data))
        if device is not None:
            restored_config = replace(restored_config, device=device)
        elif restored_config.device.startswith("cuda") and not torch.cuda.is_available():
            # Metadata describes the source run, not an obligation to have a
            # GPU when inspecting or continuing it elsewhere.
            restored_config = replace(restored_config, device="cpu")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint has no model state")
        if model is None:
            restored_model = PokerAgentModel.load_checkpoint(path, map_location=map_location or "cpu")
        else:
            metadata = payload.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("checkpoint has no model metadata")
            PokerAgentModel._validate_metadata(metadata)
            restored_model = model
            restored_model.load_state_dict(state_dict)
        instance = cls(restored_model, restored_config, provenance=dict(provenance))
        instance.optimizer.load_state_dict(optimizer_state)
        epoch, global_step = progress.get("epoch"), progress.get("global_step")
        if not isinstance(epoch, int) or epoch < 0 or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("checkpoint has invalid pretraining progress")
        instance.epoch, instance.global_step = epoch, global_step
        # Creating the model/optimizer can consume RNG; restore it last.
        _restore_rng_state(rng)
        return instance
