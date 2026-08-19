"""Reproducible holdout evaluation for policies, checkpoints, and baselines.

This is deliberately separate from training.  A suite deals the same fixed
hands to a candidate against every supplied opponent, rotates the candidate
through every seat, and records only evaluation-time diagnostics.  In
particular, no action selected here is added to PPO data or a league.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from json import dumps
from math import isfinite, sqrt
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .betting import Action, RAISE_ACTIONS
from .equity import (
    EquityMetrics,
    ExpectedShowdownShareMetrics,
    equity_metrics,
    expected_showdown_share_metrics,
)
from .game_state import HandState
from .league import ModelPolicy
from .model import ACTION_NAMES, BET_SIZE_ACTIONS, TORCH_AVAILABLE
from .observation import observation_for
from .rules import BIG_BLIND, SEAT_COUNT, positions
from .traces import HandTrace


class EvaluablePolicy(Protocol):
    """The minimal non-learning policy interface accepted by this module."""

    def select_action(self, observation: Mapping[str, object], legal_actions: Mapping[str, bool]) -> Action:
        """Select one engine action for the current player-safe observation."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed holdout protocol.

    ``hands_per_opponent`` must be a multiple of ``player_count``.  This makes candidate
    seat/position exposure exactly equal rather than merely approximately
    balanced, which is especially important with the fixed button seat used by
    the cash-game engine.  With ``paired_position_seeds``, each deal seed is
    repeated once per candidate seat and confidence intervals use deal blocks.
    """

    hands_per_opponent: int = 100
    seed_start: int = 100_000
    starting_stack: int = 10_000
    player_count: int = SEAT_COUNT
    allowed_raise_actions: tuple[Action, ...] | None = None
    equity_samples: int = 16
    calibration_bins: int = 10
    paired_position_seeds: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.player_count <= SEAT_COUNT:
            raise ValueError(f"player_count must be in 2..{SEAT_COUNT}")
        if self.hands_per_opponent < self.player_count or self.hands_per_opponent % self.player_count:
            raise ValueError("hands_per_opponent must be a positive multiple of player_count for fair rotation")
        if self.starting_stack < BIG_BLIND:
            raise ValueError("starting_stack must cover the big blind")
        if self.equity_samples < 1 or self.calibration_bins < 1:
            raise ValueError("equity_samples and calibration_bins must be positive")
        if self.allowed_raise_actions is not None:
            normalized = tuple(Action(action) for action in self.allowed_raise_actions)
            if not normalized or any(action not in RAISE_ACTIONS | {Action.ALL_IN} for action in normalized):
                raise ValueError("allowed_raise_actions must contain only raise sizes and all-in")
            object.__setattr__(self, "allowed_raise_actions", normalized)


@dataclass(frozen=True)
class PokerStyleStatistics:
    """Poker-facing tendencies calculated from the candidate's own hands."""

    hands: int
    vpip: float
    pfr: float
    three_bet: float
    fold_to_3bet: float
    aggression_factor: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ModelDiagnostics:
    """Holdout-only health metrics for the policy, value and masking heads."""

    decisions: int
    policy_entropy: float | None
    value_mae_bb: float | None
    value_rmse_bb: float | None
    masked_action_rate: float | None
    illegal_action_count: int

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class SanityScenarioResult:
    """One fixed, player-safe observation used for an inference smoke check."""

    name: str
    legal: bool
    finite: bool
    expected_showdown_share: float | None
    selected_action: str | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MatchupReport:
    """Candidate result against one baseline or historical checkpoint."""

    opponent: str
    hands: int
    seed_blocks: int
    ci_method: str
    pnl_bb: float
    bb_per_100: float
    bb_per_100_standard_error: float
    bb_per_100_ci95_low: float
    bb_per_100_ci95_high: float
    win_rate: float
    tie_rate: float
    league_score: float
    position_hands: Mapping[str, int]
    pnl_by_position_bb: Mapping[str, float]
    statistics: PokerStyleStatistics
    model_diagnostics: ModelDiagnostics
    equity: EquityMetrics | None
    expected_showdown_share: ExpectedShowdownShareMetrics | None

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["statistics"] = self.statistics.as_dict()
        result["model_diagnostics"] = self.model_diagnostics.as_dict()
        result["equity"] = None if self.equity is None else self.equity.as_dict()
        result["expected_showdown_share"] = (
            None if self.expected_showdown_share is None else self.expected_showdown_share.as_dict()
        )
        return result


@dataclass(frozen=True)
class EvaluationSuiteReport:
    """A complete machine-readable fixed-suite report for one candidate."""

    candidate: str
    config: EvaluationConfig
    matchups: tuple[MatchupReport, ...]
    sanity_checks: tuple[SanityScenarioResult, ...]

    @property
    def aggregate_bb_per_100(self) -> float:
        hands = sum(item.hands for item in self.matchups)
        return 0.0 if not hands else sum(item.pnl_bb for item in self.matchups) / hands * 100.0

    @property
    def aggregate_league_score(self) -> float:
        """Mean fixed-suite score (win=1, tie=0.5, loss=0) across matchups."""

        return sum(item.league_score for item in self.matchups) / len(self.matchups)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "2.0",
            "outcome_protocol": "fixed_deal_virtual_showdown_outcome_v1",
            "scalar_metric_protocol": "active_hands_expected_showdown_share_v1",
            "candidate": self.candidate,
            "config": asdict(self.config),
            "aggregate_bb_per_100": self.aggregate_bb_per_100,
            "aggregate_league_score": self.aggregate_league_score,
            "matchups": [item.as_dict() for item in self.matchups],
            "sanity_checks": [item.as_dict() for item in self.sanity_checks],
        }

    def write_json(self, path: str | Path) -> Path:
        """Write an atomic-consumer-friendly, deterministic JSON report."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            try:
                descriptor = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:  # pragma: no cover - filesystem dependent.
                pass
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


@dataclass
class _StyleCounters:
    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    three_bets: int = 0
    faced_three_bet: int = 0
    folded_to_three_bet: int = 0
    aggressive: int = 0
    calls: int = 0


@dataclass
class _ModelAccumulator:
    entropy: list[float]
    value_errors: list[float]
    masked_selected: int = 0
    decisions: int = 0
    equity_predictions: list[tuple[float, float, float]] | None = None
    equity_targets: list[tuple[float, float, float]] | None = None
    expected_showdown_share_predictions: list[float] | None = None
    expected_showdown_share_targets: list[float] | None = None


def evaluate_suite(
    candidate_name: str,
    candidate: EvaluablePolicy,
    opponents: Mapping[str, EvaluablePolicy],
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationSuiteReport:
    """Evaluate ``candidate`` against every fixed baseline/checkpoint.

    Each supplied opponent fills every non-candidate seat.  The candidate moves
    one seat each hand, so every matchup has exact exposure to every position
    for its table size.  Every matchup receives the same seed range; evaluating
    another candidate with the same protocol therefore uses common random
    numbers and is not confounded by a luckier set of boards.
    """

    if not candidate_name:
        raise ValueError("candidate_name must not be empty")
    if not opponents:
        raise ValueError("at least one baseline or historical opponent is required")
    protocol = config or EvaluationConfig()
    reports = tuple(
        _evaluate_matchup(candidate, opponent_name, opponent, protocol)
        for opponent_name, opponent in opponents.items()
    )
    return EvaluationSuiteReport(
        candidate=candidate_name,
        config=protocol,
        matchups=reports,
        sanity_checks=run_sanity_checks(candidate),
    )


def _evaluate_matchup(
    candidate: EvaluablePolicy,
    opponent_name: str,
    opponent: EvaluablePolicy,
    config: EvaluationConfig,
) -> MatchupReport:
    # Some baselines carry an instance-local RNG.  Evaluation must not mutate
    # caller-owned policies, otherwise repeating the same suite changes the
    # answer despite fixed deal seeds.
    candidate_template = _evaluation_copy(candidate)
    opponent_template = _evaluation_copy(opponent)
    style = _StyleCounters()
    position_names = tuple(positions(0, player_count=config.player_count).values())
    position_hands = {position: 0 for position in position_names}
    pnl_by_position = {position: 0.0 for position in position_hands}
    total_pnl = 0.0
    hand_pnl: list[float] = []
    winning_hands = 0
    tied_hands = 0
    model_accumulator = (
        _ModelAccumulator(
            [],
            [],
            equity_predictions=[],
            equity_targets=[],
            expected_showdown_share_predictions=[],
            expected_showdown_share_targets=[],
        )
        if isinstance(candidate_template, ModelPolicy)
        else None
    )

    for hand_index in range(config.hands_per_opponent):
        # Button remains stable; rotating the candidate's physical seat rotates
        # its poker position exactly once per full player-count block.
        candidate_seat = hand_index % config.player_count
        deal_index = hand_index // config.player_count if config.paired_position_seeds else hand_index
        deal_seed = config.seed_start + deal_index
        state = HandState(
            seed=deal_seed,
            starting_stack=config.starting_stack,
            player_count=config.player_count,
            allowed_raise_actions=None
            if config.allowed_raise_actions is None
            else frozenset(config.allowed_raise_actions),
        )
        candidate_policy = _evaluation_copy(candidate_template)
        opponent_policy = _evaluation_copy(opponent_template)
        _seed_evaluation_policy(candidate_policy, deal_seed * 2)
        _seed_evaluation_policy(opponent_policy, deal_seed * 2 + 1)
        position = state.positions[candidate_seat]
        position_hands[position] += 1
        style.hands += 1
        trace = HandTrace(
            hand_id=hand_index,
            seed=state.seed,
            button_seat=state.button_seat,
            starting_stack=config.starting_stack,
            equity_samples=config.equity_samples,
        )
        voluntary = False
        pfr = False
        three_bet = False
        preflop_raises = 0
        model_rows: list[tuple[int, tuple[float, float, float], float, float]] = []
        while not state.complete:
            seat = state.actor
            if seat is None:
                raise RuntimeError("live hand has no actor")
            observation = observation_for(state, seat)
            policy = candidate_policy if seat == candidate_seat else opponent_policy
            is_candidate = seat == candidate_seat
            if is_candidate and model_accumulator is not None:
                action, probabilities, equity, expected_showdown_share, value = _model_action(policy, observation)
                model_accumulator.decisions += 1
                model_accumulator.entropy.append(_entropy(probabilities))
                legal = observation["legal_actions"]
                if not bool(legal.get(action.value, False)):
                    model_accumulator.masked_selected += 1
                model_rows.append((len(trace.decisions), equity, expected_showdown_share, value))
            else:
                action = policy.select_action(observation, observation["legal_actions"])
            if not isinstance(action, Action):
                action = Action(action)
            legal_actions = state.legal_actions(seat)
            if not legal_actions[action]:
                raise ValueError(f"policy selected illegal action {action.value!r} in evaluation")
            if is_candidate:
                if action in RAISE_ACTIONS or action == Action.ALL_IN:
                    style.aggressive += 1
                elif action == Action.CALL:
                    style.calls += 1
                if state.street.value == "preflop":
                    to_call = state.to_call(seat)
                    if to_call > 0 and preflop_raises >= 2:
                        style.faced_three_bet += 1
                        if action == Action.FOLD:
                            style.folded_to_three_bet += 1
                    if action == Action.CALL or action in RAISE_ACTIONS or action == Action.ALL_IN:
                        if not (state.player(seat).committed_street > 0 and action == Action.CALL and to_call == 0):
                            voluntary = True
                    if action in RAISE_ACTIONS or action == Action.ALL_IN:
                        if preflop_raises >= 1:
                            three_bet = True
                        pfr = True
                        preflop_raises += 1
            elif state.street.value == "preflop" and (action in RAISE_ACTIONS or action == Action.ALL_IN):
                preflop_raises += 1
            trace.record_action(state, action)
            state.step(action)
        trace.complete(state)
        pnl = (state.player(candidate_seat).stack - config.starting_stack) / BIG_BLIND
        total_pnl += pnl
        hand_pnl.append(pnl)
        pnl_by_position[position] += pnl
        winning_hands += pnl > 0
        tied_hands += pnl == 0
        style.vpip_hands += voluntary
        style.pfr_hands += pfr
        style.three_bets += three_bet
        if model_accumulator is not None:
            for decision_index, prediction, share_prediction, value in model_rows:
                decision = trace.decisions[decision_index]
                if (
                    decision.equity_target is None
                    or decision.expected_showdown_share_target is None
                    or decision.terminal_pnl_bb is None
                ):
                    raise RuntimeError("completed evaluation trace has no labels")
                assert model_accumulator.equity_predictions is not None
                assert model_accumulator.equity_targets is not None
                model_accumulator.equity_predictions.append(prediction)
                model_accumulator.equity_targets.append(tuple(float(value) for value in decision.equity_target))
                assert model_accumulator.expected_showdown_share_predictions is not None
                assert model_accumulator.expected_showdown_share_targets is not None
                model_accumulator.expected_showdown_share_predictions.append(share_prediction)
                model_accumulator.expected_showdown_share_targets.append(
                    float(decision.expected_showdown_share_target)
                )
                model_accumulator.value_errors.append(value - decision.terminal_pnl_bb)

    diagnostics = _model_diagnostics(model_accumulator)
    equity = None
    if model_accumulator is not None and model_accumulator.equity_predictions:
        equity = equity_metrics(
            model_accumulator.equity_predictions,
            model_accumulator.equity_targets or (),
            bins=config.calibration_bins,
        )
    expected_showdown_share = None
    if model_accumulator is not None and model_accumulator.expected_showdown_share_predictions:
        expected_showdown_share = expected_showdown_share_metrics(
            model_accumulator.expected_showdown_share_predictions,
            model_accumulator.expected_showdown_share_targets or (),
            bins=config.calibration_bins,
        )
    hands = config.hands_per_opponent
    bb_per_100 = total_pnl / hands * 100.0
    if config.paired_position_seeds:
        block_values = [
            sum(hand_pnl[start : start + config.player_count]) / config.player_count
            for start in range(0, hands, config.player_count)
        ]
        ci_method = "paired_position_seed_block_normal_v1"
    else:
        block_values = hand_pnl
        ci_method = "independent_hand_normal_v1"
    if len(block_values) > 1:
        mean = sum(block_values) / len(block_values)
        variance = sum((value - mean) ** 2 for value in block_values) / (len(block_values) - 1)
        standard_error = sqrt(variance / len(block_values)) * 100.0
    else:  # Config currently requires at least two hands; kept defensive.
        standard_error = 0.0
    margin = 1.96 * standard_error
    return MatchupReport(
        opponent=opponent_name,
        hands=hands,
        seed_blocks=len(block_values),
        ci_method=ci_method,
        pnl_bb=total_pnl,
        bb_per_100=bb_per_100,
        bb_per_100_standard_error=standard_error,
        bb_per_100_ci95_low=bb_per_100 - margin,
        bb_per_100_ci95_high=bb_per_100 + margin,
        win_rate=winning_hands / hands,
        tie_rate=tied_hands / hands,
        league_score=(winning_hands + 0.5 * tied_hands) / hands,
        position_hands=position_hands,
        pnl_by_position_bb=pnl_by_position,
        statistics=_style_statistics(style),
        model_diagnostics=diagnostics,
        equity=equity,
        expected_showdown_share=expected_showdown_share,
    )


def _evaluation_copy(policy: EvaluablePolicy) -> EvaluablePolicy:
    """Clone mutable baseline state without needlessly copying a neural model."""

    if isinstance(policy, ModelPolicy):
        return policy
    try:
        return deepcopy(policy)
    except Exception:
        # Protocol implementations may legitimately be backed by a service or
        # another non-copyable immutable object.  They remain evaluable, but
        # callers should make their random state explicit in that case.
        return policy


def _seed_evaluation_policy(policy: EvaluablePolicy, seed: int) -> None:
    """Reset a bot's private RNG per deal block without touching global RNG."""

    rng = getattr(policy, "_rng", None)
    if rng is not None and hasattr(rng, "seed"):
        rng.seed(seed)


def _model_action(
    policy: EvaluablePolicy,
    observation: Mapping[str, object],
) -> tuple[Action, tuple[float, ...], tuple[float, float, float], float, float]:
    if not isinstance(policy, ModelPolicy):  # Defensive: called only after the check above.
        raise TypeError("model diagnostics require ModelPolicy")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate a model checkpoint")
    import torch

    model = policy.model
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model([observation])
    if was_training:
        model.train()
    action_index = int(output.action_probabilities[0].argmax().item())
    action_name = ACTION_NAMES[action_index]
    if action_name == "raise":
        action_name = BET_SIZE_ACTIONS[int(output.bet_size_probabilities[0].argmax().item())]
    return (
        Action(action_name),
        tuple(float(item) for item in output.action_probabilities[0].tolist()),
        tuple(float(item) for item in output.equity_probabilities[0].tolist()),  # type: ignore[return-value]
        float(output.expected_showdown_share[0].item()),
        float(output.value[0].item()),
    )


def _entropy(probabilities: Sequence[float]) -> float:
    from math import log

    return -sum(value * log(value) for value in probabilities if value > 0.0)


def _style_statistics(counter: _StyleCounters) -> PokerStyleStatistics:
    hands = max(1, counter.hands)
    return PokerStyleStatistics(
        hands=counter.hands,
        vpip=counter.vpip_hands / hands,
        pfr=counter.pfr_hands / hands,
        three_bet=counter.three_bets / hands,
        fold_to_3bet=counter.folded_to_three_bet / max(1, counter.faced_three_bet),
        aggression_factor=counter.aggressive / max(1, counter.calls),
    )


def _model_diagnostics(accumulator: _ModelAccumulator | None) -> ModelDiagnostics:
    if accumulator is None or not accumulator.decisions:
        return ModelDiagnostics(0, None, None, None, None, 0)
    errors = accumulator.value_errors
    return ModelDiagnostics(
        decisions=accumulator.decisions,
        policy_entropy=sum(accumulator.entropy) / accumulator.decisions,
        value_mae_bb=sum(abs(value) for value in errors) / len(errors),
        value_rmse_bb=sqrt(sum(value * value for value in errors) / len(errors)),
        masked_action_rate=accumulator.masked_selected / accumulator.decisions,
        illegal_action_count=accumulator.masked_selected,
    )


def run_sanity_checks(policy: EvaluablePolicy) -> tuple[SanityScenarioResult, ...]:
    """Run fixed card/odds situations without defining a promotion threshold.

    These are diagnostics, not training labels: a candidate which fails a
    scenario is reported as such and can be rejected by a promotion policy.
    The engine's real legal masks are retained, while known cards are swapped
    only in a copied player-safe observation.  This keeps the check free of
    opponent private information and suitable for model inference.
    """

    if not isinstance(policy, ModelPolicy):
        return ()
    base = HandState(seed=91)
    seat = base.actor
    if seat is None:
        raise RuntimeError("new hand has no actor")
    original = observation_for(base, seat)
    scenarios = (
        ("nuts_river", ("Th", "3d"), ("Ah", "Kh", "Qh", "Jh", "2c"), "Royal flush made on the river."),
        ("lost_river", ("3d", "4s"), ("Ah", "Kh", "Qh", "Jh", "2c"), "Weak hand facing a four-to-a-flush board."),
        ("flush_draw", ("As", "5s"), ("Ks", "7s", "2d"), "Four-card flush draw on the flop."),
        ("favourable_pot_odds", ("9h", "8h"), ("Th", "7d", "2c"), "Open-ended draw with a small call relative to the pot."),
        ("unfavourable_pot_odds", ("9h", "8h"), ("Th", "7d", "2c"), "Same draw with a large call relative to the pot."),
        ("short_stack_all_in", ("As", "Kd"), (), "Short stack decision with all-in available."),
    )
    results: list[SanityScenarioResult] = []
    for name, hole_cards, board, note in scenarios:
        observation = _scenario_observation(original, hole_cards, board, name)
        try:
            action, probabilities, equity, expected_showdown_share, _ = _model_action(policy, observation)
            legal = bool(observation["legal_actions"].get(action.value, False))
            finite = (
                all(isfinite(value) and value >= 0.0 for value in (*probabilities, *equity))
                and abs(sum(equity) - 1.0) < 1e-5
                and isfinite(expected_showdown_share)
                and 0.0 <= expected_showdown_share <= 1.0
            )
            results.append(
                SanityScenarioResult(name, legal, finite, expected_showdown_share, action.value, note)
            )
        except Exception:
            results.append(SanityScenarioResult(name, False, False, None, None, note))
    return tuple(results)


def _scenario_observation(original: Mapping[str, object], hole_cards: tuple[str, str], board: tuple[str, ...], scenario: str) -> dict[str, Any]:
    """Return a legal-mask-preserving fixed inference observation."""

    observation = deepcopy(dict(original))
    cards = dict(observation["cards"])
    cards["hole_cards"] = list(hole_cards)
    cards["board"] = list(board)
    cards["street"] = "river" if len(board) == 5 else "flop" if len(board) == 3 else "preflop"
    cards["street_index"] = {"preflop": 0, "flop": 1, "river": 3}[cards["street"]]
    observation["cards"] = cards
    observation["hole_cards"] = list(hole_cards)
    observation["board"] = list(board)
    observation["street"] = cards["street"]
    hero = dict(observation["hero"])
    if scenario == "favourable_pot_odds":
        hero["to_call_bb"] = 0.25
        hero["to_call_to_pot"] = 0.05
    elif scenario == "unfavourable_pot_odds":
        hero["to_call_bb"] = 20.0
        hero["to_call_to_pot"] = 0.80
    elif scenario == "short_stack_all_in":
        hero["stack_bb"] = 3.0
        hero["stack_to_pot"] = 0.4
    observation["hero"] = hero
    return observation


__all__ = [
    "EvaluationConfig",
    "EvaluationSuiteReport",
    "MatchupReport",
    "ModelDiagnostics",
    "PokerStyleStatistics",
    "SanityScenarioResult",
    "evaluate_suite",
    "run_sanity_checks",
]
