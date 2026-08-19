"""Training-only virtual-showdown labels and equity quality metrics.

The policy observation deliberately omits opponent hole cards and undealt
cards.  This module may use them *only* inside :class:`EquitySnapshot` to
construct supervision for an auxiliary head.  Its public result is a soft
``[win, tie, loss]`` target and an expected **showdown** share target.  The
latter is the hero's fractional share among active hands at a virtual
showdown; it is intentionally not a final-pot or side-pot payout label.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import comb, isfinite, log
from random import Random
from typing import Iterable, Sequence

from .cards import Card
from .evaluator import evaluate
from .game_state import HandState

EQUITY_OUTCOMES = ("win", "tie", "loss")


@dataclass(frozen=True)
class EquitySnapshot:
    """Private, immutable state required to label one decision point.

    This is a training-environment artifact, not a policy input and not a
    serialisable trace record.  ``opponent_hole_cards`` and
    ``remaining_deck`` must never be passed to :mod:`poker.observation` or
    :mod:`poker.model`.
    """

    hero_seat: int
    hero_hole_cards: tuple[Card, Card]
    opponent_hole_cards: tuple[tuple[Card, Card], ...]
    board: tuple[Card, ...]
    remaining_deck: tuple[Card, ...]
    source_seed: int | None = None
    reference: str = ""

    def __post_init__(self) -> None:
        if len(self.hero_hole_cards) != 2:
            raise ValueError("hero must have exactly two hole cards")
        if not self.opponent_hole_cards:
            raise ValueError("equity snapshot needs at least one active opponent")
        if any(len(cards) != 2 for cards in self.opponent_hole_cards):
            raise ValueError("each opponent must have exactly two hole cards")
        if not 0 <= len(self.board) <= 5:
            raise ValueError("board must contain zero to five cards")
        all_cards = (*self.hero_hole_cards, *(card for cards in self.opponent_hole_cards for card in cards), *self.board, *self.remaining_deck)
        if len(all_cards) != len(set(all_cards)):
            raise ValueError("equity snapshot cards must be distinct")
        if len(self.remaining_deck) < 5 - len(self.board):
            raise ValueError("remaining deck cannot complete the board")

    @property
    def cards_to_come(self) -> int:
        return 5 - len(self.board)


@dataclass(frozen=True)
class EquityTarget:
    """Outcome probabilities plus correct multiway expected showdown share.

    For each virtual runout the hero receives zero when losing, one when the
    sole best hand, and ``1 / number_of_best_hands`` when tied.  Averaging that
    value yields ``expected_showdown_share``.  This excludes folded players,
    side pots and the actual continuation of the hand by construction.

    ``None`` is accepted only as a legacy constructor convenience and derives
    the heads-up formula.  Labels emitted by :func:`generate_equity_target`
    always populate the explicit, multiway-correct value.
    """

    win: float
    tie: float
    loss: float
    samples: int
    exact: bool
    expected_showdown_share: float | None = None

    def __post_init__(self) -> None:
        probabilities = self.probabilities
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if any(not isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("equity probabilities must be finite and non-negative")
        if abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError("equity probabilities must sum to one")
        share = self.expected_showdown_share
        if share is None:
            object.__setattr__(self, "expected_showdown_share", self.win + 0.5 * self.tie)
            share = self.expected_showdown_share
        if not isinstance(share, (int, float)) or not isfinite(float(share)) or not 0.0 <= float(share) <= 1.0:
            raise ValueError("expected_showdown_share must be finite and in [0, 1]")

    @property
    def probabilities(self) -> tuple[float, float, float]:
        return (self.win, self.tie, self.loss)

    @property
    def equity(self) -> float:
        """Legacy heads-up display scalar, not a multiway pot-share metric."""

        return self.win + 0.5 * self.tie

    @property
    def expected_share(self) -> float:
        """Compatibility alias for :attr:`expected_showdown_share`."""

        assert self.expected_showdown_share is not None
        return float(self.expected_showdown_share)

    def as_list(self) -> list[float]:
        return list(self.probabilities)


@dataclass(frozen=True)
class EquityCalibrationBin:
    """Calibration statistics for one inclusive/exclusive equity interval."""

    lower: float
    upper: float
    count: int
    mean_prediction: float | None
    mean_target: float | None

    @property
    def gap(self) -> float | None:
        if self.mean_prediction is None or self.mean_target is None:
            return None
        return self.mean_prediction - self.mean_target


@dataclass(frozen=True)
class EquityMetrics:
    """Categorical outcome diagnostics.

    The ECE member retains the legacy ``win + 0.5 * tie`` reduction and is
    therefore suitable only for heads-up compatibility reports.  Multiway
    gates must use :class:`ExpectedShowdownShareMetrics`.
    """

    samples: int
    logloss: float
    brier_score: float
    expected_calibration_error: float
    calibration: tuple[EquityCalibrationBin, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "logloss": self.logloss,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "calibration": [
                {
                    "lower": item.lower,
                    "upper": item.upper,
                    "count": item.count,
                    "mean_prediction": item.mean_prediction,
                    "mean_target": item.mean_target,
                    "gap": item.gap,
                }
                for item in self.calibration
            ],
        }


@dataclass(frozen=True)
class ExpectedShowdownShareMetrics:
    """Proper scalar diagnostics for expected showdown-share predictions."""

    samples: int
    logloss: float
    brier_score: float
    mean_absolute_error: float
    root_mean_squared_error: float
    expected_calibration_error: float
    calibration: tuple[EquityCalibrationBin, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "logloss": self.logloss,
            "brier_score": self.brier_score,
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "expected_calibration_error": self.expected_calibration_error,
            "calibration": [
                {
                    "lower": item.lower,
                    "upper": item.upper,
                    "count": item.count,
                    "mean_prediction": item.mean_prediction,
                    "mean_target": item.mean_target,
                    "gap": item.gap,
                }
                for item in self.calibration
            ],
        }


def capture_equity_snapshot(state: HandState, hero_seat: int, *, reference: str = "") -> EquitySnapshot:
    """Capture the private label context before a decision mutates ``state``.

    Only players who have not folded are included: the target asks what would
    happen if the currently active players all continued to showdown.
    """

    if not 0 <= hero_seat < len(state.players):
        raise ValueError("hero seat does not exist")
    if state.player(hero_seat).folded:
        raise ValueError("a folded player has no current showdown equity")
    opponents = tuple(
        state.hole_cards[player.seat]
        for player in state.players
        if player.seat != hero_seat and not player.folded
    )
    return EquitySnapshot(
        hero_seat=hero_seat,
        hero_hole_cards=state.hole_cards[hero_seat],
        opponent_hole_cards=opponents,
        board=tuple(state.board),
        remaining_deck=state.deck.snapshot(),
        source_seed=state.seed,
        reference=reference,
    )


def generate_equity_target(snapshot: EquitySnapshot, *, samples: int = 16, seed: int | None = None) -> EquityTarget:
    """Build a soft virtual-showdown target for ``snapshot``.

    On the river the outcome is evaluated exactly once.  On earlier streets,
    all runouts are enumerated when ``samples`` covers the combination count;
    otherwise independently sampled runouts approximate the conditional
    distribution.  The actual continuation, folds, and final payout are not read.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    runout_count = comb(len(snapshot.remaining_deck), snapshot.cards_to_come)
    if snapshot.cards_to_come == 0:
        outcomes = [_showdown_result(snapshot, ())]
        exact = True
    elif samples >= runout_count:
        outcomes = [_showdown_result(snapshot, runout) for runout in combinations(snapshot.remaining_deck, snapshot.cards_to_come)]
        exact = True
    else:
        randomizer = Random(_stable_seed(snapshot) if seed is None else seed)
        outcomes = [
            _showdown_result(snapshot, tuple(randomizer.sample(snapshot.remaining_deck, snapshot.cards_to_come)))
            for _ in range(samples)
        ]
        exact = False
    counts = {outcome: sum(result[0] == outcome for result in outcomes) for outcome in EQUITY_OUTCOMES}
    total = len(outcomes)
    return EquityTarget(
        win=counts["win"] / total,
        tie=counts["tie"] / total,
        loss=counts["loss"] / total,
        samples=total,
        exact=exact,
        expected_showdown_share=sum(result[1] for result in outcomes) / total,
    )


def equity_cross_entropy(logits, targets, *, reduction: str = "mean"):
    """Soft-target cross entropy for PyTorch equity logits.

    The local import keeps engine-only installs free of a PyTorch dependency.
    ``targets`` must have a final dimension of three and rows summing to one.
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:  # pragma: no cover - optional path.
        raise RuntimeError("equity_cross_entropy requires PyTorch; install with `.[rl]`.") from error
    if logits.shape != targets.shape or logits.shape[-1] != len(EQUITY_OUTCOMES):
        raise ValueError("logits and targets must have the same final dimension of three")
    if not torch.isfinite(targets).all() or bool((targets < 0).any()):
        raise ValueError("targets must be finite and non-negative")
    if not torch.allclose(targets.sum(dim=-1), torch.ones_like(targets.sum(dim=-1)), atol=1e-6):
        raise ValueError("each target row must sum to one")
    per_sample = -(targets * functional.log_softmax(logits, dim=-1)).sum(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def equity_metrics(predictions: Iterable[Sequence[float] | EquityTarget], targets: Iterable[Sequence[float] | EquityTarget], *, bins: int = 10) -> EquityMetrics:
    """Calculate outcome logloss/Brier and the legacy heads-up scalar ECE."""

    if bins < 1:
        raise ValueError("bins must be positive")
    prediction_rows = [_probabilities(item, "prediction") for item in predictions]
    target_rows = [_probabilities(item, "target") for item in targets]
    if not prediction_rows:
        raise ValueError("predictions must not be empty")
    if len(prediction_rows) != len(target_rows):
        raise ValueError("predictions and targets must have the same length")
    epsilon = 1e-12
    logloss = sum(-sum(target * log(max(prediction, epsilon)) for prediction, target in zip(row, target_row, strict=True)) for row, target_row in zip(prediction_rows, target_rows, strict=True)) / len(prediction_rows)
    brier = sum(sum((prediction - target) ** 2 for prediction, target in zip(row, target_row, strict=True)) for row, target_row in zip(prediction_rows, target_rows, strict=True)) / len(prediction_rows)
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for prediction, target in zip(prediction_rows, target_rows, strict=True):
        predicted_equity = prediction[0] + 0.5 * prediction[1]
        target_equity = target[0] + 0.5 * target[1]
        index = min(int(predicted_equity * bins), bins - 1)
        grouped[index].append((predicted_equity, target_equity))
    calibration: list[EquityCalibrationBin] = []
    expected_calibration_error = 0.0
    for index, values in enumerate(grouped):
        lower, upper = index / bins, (index + 1) / bins
        if values:
            mean_prediction = sum(item[0] for item in values) / len(values)
            mean_target = sum(item[1] for item in values) / len(values)
            expected_calibration_error += len(values) / len(prediction_rows) * abs(mean_prediction - mean_target)
        else:
            mean_prediction = None
            mean_target = None
        calibration.append(EquityCalibrationBin(lower, upper, len(values), mean_prediction, mean_target))
    return EquityMetrics(
        samples=len(prediction_rows),
        logloss=logloss,
        brier_score=brier,
        expected_calibration_error=expected_calibration_error,
        calibration=tuple(calibration),
    )


def expected_showdown_share_binary_cross_entropy(logits, targets, *, reduction: str = "mean"):
    """Proper soft Bernoulli loss for scalar expected showdown share.

    A fractional share is a valid soft target in ``[0, 1]``.  This loss is
    independent of outcome CE so multiway share learning does not force the
    three-class head to encode the number of co-winners.
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:  # pragma: no cover - optional path.
        raise RuntimeError("expected_showdown_share_binary_cross_entropy requires PyTorch; install with `.[rl]`.") from error
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if not torch.isfinite(targets).all() or bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("expected showdown-share targets must be finite and in [0, 1]")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    return functional.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


def expected_showdown_share_metrics(
    predictions: Iterable[float | EquityTarget], targets: Iterable[float | EquityTarget], *, bins: int = 10
) -> ExpectedShowdownShareMetrics:
    """Calculate proper scalar calibration for active-hand showdown share."""

    if bins < 1:
        raise ValueError("bins must be positive")
    prediction_rows = [_share_value(item, "prediction") for item in predictions]
    target_rows = [_share_value(item, "target") for item in targets]
    if not prediction_rows:
        raise ValueError("predictions must not be empty")
    if len(prediction_rows) != len(target_rows):
        raise ValueError("predictions and targets must have the same length")
    epsilon = 1e-12
    logloss = sum(
        -(target * log(max(prediction, epsilon)) + (1.0 - target) * log(max(1.0 - prediction, epsilon)))
        for prediction, target in zip(prediction_rows, target_rows, strict=True)
    ) / len(prediction_rows)
    squared_errors = [(prediction - target) ** 2 for prediction, target in zip(prediction_rows, target_rows, strict=True)]
    brier = sum(squared_errors) / len(squared_errors)
    mae = sum(abs(prediction - target) for prediction, target in zip(prediction_rows, target_rows, strict=True)) / len(prediction_rows)
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for prediction, target in zip(prediction_rows, target_rows, strict=True):
        grouped[min(int(prediction * bins), bins - 1)].append((prediction, target))
    calibration: list[EquityCalibrationBin] = []
    ece = 0.0
    for index, values in enumerate(grouped):
        lower, upper = index / bins, (index + 1) / bins
        if values:
            predicted_mean = sum(item[0] for item in values) / len(values)
            target_mean = sum(item[1] for item in values) / len(values)
            ece += len(values) / len(prediction_rows) * abs(predicted_mean - target_mean)
        else:
            predicted_mean = target_mean = None
        calibration.append(EquityCalibrationBin(lower, upper, len(values), predicted_mean, target_mean))
    return ExpectedShowdownShareMetrics(
        samples=len(prediction_rows),
        logloss=logloss,
        brier_score=brier,
        mean_absolute_error=mae,
        root_mean_squared_error=(brier**0.5),
        expected_calibration_error=ece,
        calibration=tuple(calibration),
    )


def _showdown_result(snapshot: EquitySnapshot, runout: tuple[Card, ...]) -> tuple[str, float]:
    hero_rank = evaluate((*snapshot.hero_hole_cards, *snapshot.board, *runout))
    opponent_ranks = [evaluate((*cards, *snapshot.board, *runout)) for cards in snapshot.opponent_hole_cards]
    best = max(hero_rank, *opponent_ranks)
    if hero_rank != best:
        return "loss", 0.0
    winner_count = sum(rank == best for rank in (hero_rank, *opponent_ranks))
    return ("win" if winner_count == 1 else "tie"), 1.0 / winner_count


def _stable_seed(snapshot: EquitySnapshot) -> int:
    material = f"{snapshot.source_seed!r}:{snapshot.reference}:{snapshot.hero_seat}".encode()
    return int.from_bytes(sha256(material).digest()[:8], "big")


def _probabilities(value: Sequence[float] | EquityTarget, name: str) -> tuple[float, float, float]:
    probabilities = value.probabilities if isinstance(value, EquityTarget) else tuple(value)
    if len(probabilities) != len(EQUITY_OUTCOMES):
        raise ValueError(f"{name} must contain [win, tie, loss]")
    normalized = tuple(float(item) for item in probabilities)
    if any(not isfinite(item) or item < 0.0 for item in normalized) or abs(sum(normalized) - 1.0) > 1e-6:
        raise ValueError(f"{name} must be a finite probability distribution")
    return normalized  # type: ignore[return-value]


def _share_value(value: float | EquityTarget, name: str) -> float:
    raw = value.expected_showdown_share if isinstance(value, EquityTarget) else value
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not isfinite(float(raw)) or not 0.0 <= float(raw) <= 1.0:
        raise ValueError(f"{name} expected showdown share must be finite and in [0, 1]")
    return float(raw)
