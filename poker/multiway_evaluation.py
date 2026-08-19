"""Fixed, seat-balanced evaluation for 2/3/5-max candidate policies.

Unlike the heads-up promotion evaluator, this module accepts an explicit,
heterogeneous opponent factory for every non-candidate seat.  The same deal
seed is played once with the candidate in each physical seat; confidence
intervals are calculated from those whole-table seed blocks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from math import isfinite, log, sqrt
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from .betting import Action, RAISE_ACTIONS
from .equity import EquityMetrics, ExpectedShowdownShareMetrics, equity_metrics, expected_showdown_share_metrics
from .game_state import HandState
from .league import ModelPolicy
from .model import ACTION_NAMES, BET_SIZE_ACTIONS, TORCH_AVAILABLE
from .observation import observation_for
from .rules import BIG_BLIND, SEAT_COUNT, positions
from .traces import HandTrace


MULTIWAY_EVALUATION_SCHEMA_VERSION = "1.0"
MULTIWAY_EVALUATION_PROTOCOL = "fixed_common_deal_seat_balanced_multiway_v1"
PAIRED_BLOCK_CI_METHOD = "paired_full_table_seed_block_normal_v1"
EXPECTED_SHOWDOWN_SHARE_PROTOCOL = "active_hands_expected_showdown_share_v1"
PAIRED_TRANSFER_SCRATCH_PROTOCOL = "paired_common_deal_transfer_minus_scratch_v1"


class EvaluablePolicy(Protocol):
    """Minimal policy contract; observations never include private opponent data."""

    def select_action(self, observation: Mapping[str, object], legal_actions: Mapping[str, bool]) -> Action:
        """Select one legal engine action."""


PolicyFactory = Callable[[], EvaluablePolicy]


@dataclass(frozen=True)
class OpponentSeat:
    """One named, independently constructed non-candidate seat policy."""

    identity: str
    factory: PolicyFactory

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("opponent identity must be a non-empty string")
        if not callable(self.factory):
            raise TypeError("opponent factory must be callable")


@dataclass(frozen=True)
class MultiwayEvaluationConfig:
    """Serializable fixed protocol for a common-deal multiway suite."""

    player_count: int = 3
    deal_blocks: int = 32
    seed_start: int = 4_000_000
    starting_stack: int = 10_000
    allowed_raise_actions: tuple[Action, ...] | None = None
    equity_samples: int = 32
    calibration_bins: int = 10
    required_expected_showdown_share_strata: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.player_count not in (2, 3, 5):
            raise ValueError("multiway evaluation supports only 2, 3, or 5 players")
        if self.deal_blocks < 2:
            raise ValueError("deal_blocks must be at least two for a paired-block confidence interval")
        if self.starting_stack < BIG_BLIND:
            raise ValueError("starting_stack must cover the big blind")
        if self.equity_samples < 1 or self.calibration_bins < 1:
            raise ValueError("equity_samples and calibration_bins must be positive")
        if any(not isinstance(item, str) or not item for item in self.required_expected_showdown_share_strata):
            raise ValueError("required_expected_showdown_share_strata must contain non-empty strings")
        if len(set(self.required_expected_showdown_share_strata)) != len(self.required_expected_showdown_share_strata):
            raise ValueError("required_expected_showdown_share_strata must be unique")
        if self.allowed_raise_actions is not None:
            normalized = tuple(Action(action) for action in self.allowed_raise_actions)
            if not normalized or any(action not in RAISE_ACTIONS | {Action.ALL_IN} for action in normalized):
                raise ValueError("allowed_raise_actions must contain only raise sizes and all-in")
            object.__setattr__(self, "allowed_raise_actions", normalized)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        allowed = result["allowed_raise_actions"]
        result["allowed_raise_actions"] = None if allowed is None else [action.value for action in self.allowed_raise_actions or ()]
        return result


@dataclass(frozen=True)
class MultiwayModelDiagnostics:
    """Candidate model-only diagnostics gathered from frozen evaluation traces."""

    decisions: int
    policy_entropy: float | None
    value_mae_bb: float | None
    value_rmse_bb: float | None
    masked_action_rate: float | None
    illegal_action_count: int
    equity: EquityMetrics | None
    expected_showdown_share: ExpectedShowdownShareMetrics | None
    expected_showdown_share_by_stratum: Mapping[str, ExpectedShowdownShareMetrics]
    expected_showdown_share_support: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "decisions": self.decisions,
            "policy_entropy": self.policy_entropy,
            "value_mae_bb": self.value_mae_bb,
            "value_rmse_bb": self.value_rmse_bb,
            "masked_action_rate": self.masked_action_rate,
            "illegal_action_count": self.illegal_action_count,
            "equity": None if self.equity is None else self.equity.as_dict(),
            "expected_showdown_share": (
                None if self.expected_showdown_share is None else self.expected_showdown_share.as_dict()
            ),
            "expected_showdown_share_by_stratum": {
                key: value.as_dict() for key, value in sorted(self.expected_showdown_share_by_stratum.items())
            },
            "expected_showdown_share_support": dict(sorted(self.expected_showdown_share_support.items())),
        }


@dataclass(frozen=True)
class MultiwayEvaluationReport:
    """Hash-safe result of one candidate evaluated across every table seat."""

    candidate: str
    config: MultiwayEvaluationConfig
    opponent_slots: tuple[str, ...]
    hands: int
    seed_blocks: int
    pnl_bb: float
    block_pnl_bb: tuple[float, ...]
    bb_per_100: float
    bb_per_100_standard_error: float
    bb_per_100_ci95_low: float
    bb_per_100_ci95_high: float
    position_hands: Mapping[str, int]
    pnl_by_position_bb: Mapping[str, float]
    model_diagnostics: MultiwayModelDiagnostics

    @property
    def protocol(self) -> dict[str, object]:
        return {
            "name": MULTIWAY_EVALUATION_PROTOCOL,
            "candidate_seat_rotation": "every_physical_seat_once_per_common_deal_block_v1",
            "opponent_slot_assignment": "relative_slot_r_to_physical_seat_candidate_plus_r_mod_player_count_v1",
            "deal_seed_schedule": "seed_start_plus_block_index_v1",
            "ci_method": PAIRED_BLOCK_CI_METHOD,
            "scalar_metric_protocol": EXPECTED_SHOWDOWN_SHARE_PROTOCOL,
        }

    @property
    def protocol_sha256(self) -> str:
        return _canonical_sha256(
            {
                "config": self.config.as_dict(),
                "opponent_slots": list(self.opponent_slots),
                "protocol": self.protocol,
            }
        )

    @property
    def seat_assignments(self) -> dict[str, dict[str, str]]:
        """Deterministic physical-seat identities for every candidate rotation."""

        assignments: dict[str, dict[str, str]] = {}
        for candidate_seat in range(self.config.player_count):
            assignment = {str(seat): "candidate" for seat in range(self.config.player_count)}
            for relative_slot, identity in enumerate(self.opponent_slots, start=1):
                assignment[str((candidate_seat + relative_slot) % self.config.player_count)] = identity
            assignments[str(candidate_seat)] = assignment
        return assignments

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": MULTIWAY_EVALUATION_SCHEMA_VERSION,
            "candidate": self.candidate,
            "config": self.config.as_dict(),
            "opponent_slots": list(self.opponent_slots),
            "protocol": self.protocol,
            "protocol_sha256": self.protocol_sha256,
            "seat_assignments": self.seat_assignments,
            "hands": self.hands,
            "seed_blocks": self.seed_blocks,
            "pnl_bb": self.pnl_bb,
            # One mean candidate PnL per common-deal block.  Consumers may
            # pair this vector with scratch/transfer runs using the same
            # protocol SHA and seed schedule; it contains no private cards.
            "block_pnl_bb": list(self.block_pnl_bb),
            "bb_per_100": self.bb_per_100,
            "bb_per_100_standard_error": self.bb_per_100_standard_error,
            "bb_per_100_ci95_low": self.bb_per_100_ci95_low,
            "bb_per_100_ci95_high": self.bb_per_100_ci95_high,
            "position_hands": dict(sorted(self.position_hands.items())),
            "pnl_by_position_bb": dict(sorted(self.pnl_by_position_bb.items())),
            "model_diagnostics": self.model_diagnostics.as_dict(),
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically write the deterministic, JSON-safe report."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


@dataclass
class _ModelAccumulator:
    entropy: list[float]
    value_errors: list[float]
    masked_selected: int = 0
    decisions: int = 0
    equity_predictions: list[tuple[float, float, float]] | None = None
    equity_targets: list[tuple[float, float, float]] | None = None
    expected_share_predictions: list[float] | None = None
    expected_share_targets: list[float] | None = None
    expected_share_by_stratum: dict[str, tuple[list[float], list[float]]] | None = None


@dataclass(frozen=True)
class PairedMultiwayEvaluation:
    """Transfer-minus-scratch deltas paired by common deal block."""

    transfer_candidate: str
    scratch_candidate: str
    seed_blocks: int
    delta_block_pnl_bb: tuple[float, ...]
    delta_bb_per_100: float
    delta_bb_per_100_standard_error: float
    delta_bb_per_100_ci95_low: float
    delta_bb_per_100_ci95_high: float
    protocol_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": PAIRED_TRANSFER_SCRATCH_PROTOCOL,
            "protocol_sha256": self.protocol_sha256,
            "transfer_candidate": self.transfer_candidate,
            "scratch_candidate": self.scratch_candidate,
            "seed_blocks": self.seed_blocks,
            "delta_block_pnl_bb": list(self.delta_block_pnl_bb),
            "delta_bb_per_100": self.delta_bb_per_100,
            "delta_bb_per_100_standard_error": self.delta_bb_per_100_standard_error,
            "delta_bb_per_100_ci95_low": self.delta_bb_per_100_ci95_low,
            "delta_bb_per_100_ci95_high": self.delta_bb_per_100_ci95_high,
        }


def evaluate_multiway_suite(
    candidate_name: str,
    candidate: EvaluablePolicy,
    opponents: Sequence[OpponentSeat],
    *,
    config: MultiwayEvaluationConfig | None = None,
) -> MultiwayEvaluationReport:
    """Evaluate a candidate against a heterogeneous explicit seat lineup.

    Opponent slot ``r`` (one-based) always occupies physical seat
    ``(candidate_seat + r) % player_count``.  Thus a `(rule, aggro)` 3-max
    lineup stays relative to the candidate while the entire table rotates.
    Identities and the mapping are persisted; factories themselves are
    intentionally excluded from the report/hash because callables are not
    durable artifacts.
    """

    if not isinstance(candidate_name, str) or not candidate_name.strip():
        raise ValueError("candidate_name must be a non-empty string")
    protocol = config or MultiwayEvaluationConfig()
    slots = tuple(opponents)
    if len(slots) != protocol.player_count - 1:
        raise ValueError("one explicit opponent factory is required for every non-candidate seat")
    identities = tuple(slot.identity for slot in slots)
    if len(identities) != len(set(identities)):
        raise ValueError("opponent identities must be unique")

    candidate_template = _isolated_policy(candidate)
    position_names = tuple(positions(0, player_count=protocol.player_count).values())
    position_hands = {position: 0 for position in position_names}
    pnl_by_position = {position: 0.0 for position in position_names}
    total_pnl = 0.0
    block_values: list[float] = []
    accumulator = _new_model_accumulator(candidate_template)

    for block_index in range(protocol.deal_blocks):
        deal_seed = protocol.seed_start + block_index
        block_pnl = 0.0
        for candidate_seat in range(protocol.player_count):
            state = HandState(
                seed=deal_seed,
                starting_stack=protocol.starting_stack,
                player_count=protocol.player_count,
                allowed_raise_actions=None
                if protocol.allowed_raise_actions is None
                else frozenset(protocol.allowed_raise_actions),
            )
            candidate_policy = _isolated_policy(candidate_template)
            _seed_policy(candidate_policy, _policy_seed(deal_seed, candidate_seat, 0))
            opponent_policies = [_isolated_policy(slot.factory()) for slot in slots]
            for slot_index, opponent_policy in enumerate(opponent_policies, start=1):
                _seed_policy(opponent_policy, _policy_seed(deal_seed, candidate_seat, slot_index))
            policies = _seat_policies(candidate_seat, candidate_policy, opponent_policies, protocol.player_count)
            position = state.positions[candidate_seat]
            position_hands[position] += 1
            trace = HandTrace(
                hand_id=block_index * protocol.player_count + candidate_seat,
                seed=state.seed,
                button_seat=state.button_seat,
                starting_stack=protocol.starting_stack,
                equity_samples=protocol.equity_samples,
            )
            model_rows: list[tuple[int, tuple[float, float, float], float, float]] = []
            while not state.complete:
                seat = state.actor
                if seat is None:
                    raise RuntimeError("live hand has no actor")
                observation = observation_for(state, seat)
                policy = policies[seat]
                if seat == candidate_seat and accumulator is not None:
                    action, probabilities, equity, expected_share, value = _model_action(policy, observation)
                    accumulator.decisions += 1
                    accumulator.entropy.append(_entropy(probabilities))
                    if not bool(observation["legal_actions"].get(action.value, False)):
                        accumulator.masked_selected += 1
                    model_rows.append((len(trace.decisions), equity, expected_share, value))
                else:
                    action = policy.select_action(observation, observation["legal_actions"])
                action = Action(action)
                if not state.legal_actions(seat)[action]:
                    raise ValueError(f"policy selected illegal action {action.value!r} in multiway evaluation")
                trace.record_action(state, action)
                state.step(action)
            trace.complete(state)
            pnl = (state.player(candidate_seat).stack - protocol.starting_stack) / BIG_BLIND
            total_pnl += pnl
            block_pnl += pnl
            pnl_by_position[position] += pnl
            _record_model_labels(accumulator, model_rows, trace)
        block_values.append(block_pnl / protocol.player_count)

    standard_error = _block_standard_error(block_values)
    bb_per_100 = total_pnl / (protocol.deal_blocks * protocol.player_count) * 100.0
    margin = 1.96 * standard_error
    return MultiwayEvaluationReport(
        candidate=candidate_name,
        config=protocol,
        opponent_slots=identities,
        hands=protocol.deal_blocks * protocol.player_count,
        seed_blocks=len(block_values),
        pnl_bb=total_pnl,
        block_pnl_bb=tuple(block_values),
        bb_per_100=bb_per_100,
        bb_per_100_standard_error=standard_error,
        bb_per_100_ci95_low=bb_per_100 - margin,
        bb_per_100_ci95_high=bb_per_100 + margin,
        position_hands=position_hands,
        pnl_by_position_bb=pnl_by_position,
        model_diagnostics=_model_diagnostics(
            accumulator,
            protocol.calibration_bins,
            required_strata=protocol.required_expected_showdown_share_strata,
        ),
    )


def _seat_policies(
    candidate_seat: int,
    candidate: EvaluablePolicy,
    opponents: Sequence[EvaluablePolicy],
    player_count: int,
) -> tuple[EvaluablePolicy, ...]:
    result: list[EvaluablePolicy] = [candidate] * player_count
    for relative_slot, opponent in enumerate(opponents, start=1):
        result[(candidate_seat + relative_slot) % player_count] = opponent
    return tuple(result)


def _isolated_policy(policy: EvaluablePolicy) -> EvaluablePolicy:
    """Copy non-model policies so evaluator RNG never escapes to the caller."""

    if isinstance(policy, ModelPolicy):
        return policy
    try:
        return deepcopy(policy)
    except Exception as error:
        raise TypeError("multiway evaluation requires copyable non-model policies for RNG isolation") from error


def _seed_policy(policy: EvaluablePolicy, seed: int) -> None:
    rng = getattr(policy, "_rng", None)
    if rng is not None and hasattr(rng, "seed"):
        rng.seed(seed)


def _policy_seed(deal_seed: int, candidate_seat: int, slot_index: int) -> int:
    return int.from_bytes(
        sha256(f"multiway-policy-v1:{deal_seed}:{candidate_seat}:{slot_index}".encode("ascii")).digest()[:8], "big"
    )


def _new_model_accumulator(candidate: EvaluablePolicy) -> _ModelAccumulator | None:
    if not isinstance(candidate, ModelPolicy):
        return None
    return _ModelAccumulator(
        [], [], equity_predictions=[], equity_targets=[], expected_share_predictions=[], expected_share_targets=[], expected_share_by_stratum={}
    )


def _model_action(
    policy: EvaluablePolicy, observation: Mapping[str, object]
) -> tuple[Action, tuple[float, ...], tuple[float, float, float], float, float]:
    if not isinstance(policy, ModelPolicy):
        raise TypeError("model diagnostics require ModelPolicy")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate a model policy")
    import torch

    model = policy.model
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model([observation])
    if was_training:
        model.train()
    action_name = ACTION_NAMES[int(output.action_probabilities[0].argmax().item())]
    if action_name == "raise":
        action_name = BET_SIZE_ACTIONS[int(output.bet_size_probabilities[0].argmax().item())]
    return (
        Action(action_name),
        tuple(float(item) for item in output.action_probabilities[0].tolist()),
        tuple(float(item) for item in output.equity_probabilities[0].tolist()),  # type: ignore[return-value]
        float(output.expected_showdown_share[0].item()),
        float(output.value[0].item()),
    )


def _record_model_labels(
    accumulator: _ModelAccumulator | None,
    rows: Sequence[tuple[int, tuple[float, float, float], float, float]],
    trace: HandTrace,
) -> None:
    if accumulator is None:
        return
    for decision_index, equity_prediction, share_prediction, value in rows:
        decision = trace.decisions[decision_index]
        if decision.equity_target is None or decision.expected_showdown_share_target is None or decision.terminal_pnl_bb is None:
            raise RuntimeError("completed multiway evaluation trace has no labels")
        assert accumulator.equity_predictions is not None and accumulator.equity_targets is not None
        accumulator.equity_predictions.append(equity_prediction)
        accumulator.equity_targets.append(tuple(float(item) for item in decision.equity_target))
        assert accumulator.expected_share_predictions is not None and accumulator.expected_share_targets is not None
        accumulator.expected_share_predictions.append(share_prediction)
        accumulator.expected_share_targets.append(float(decision.expected_showdown_share_target))
        assert accumulator.expected_share_by_stratum is not None
        stratum = _expected_share_stratum(decision.observation)
        predictions, targets = accumulator.expected_share_by_stratum.setdefault(stratum, ([], []))
        predictions.append(share_prediction)
        targets.append(float(decision.expected_showdown_share_target))
        accumulator.value_errors.append(value - decision.terminal_pnl_bb)


def _model_diagnostics(accumulator: _ModelAccumulator | None, bins: int, *, required_strata: Sequence[str] = ()) -> MultiwayModelDiagnostics:
    if accumulator is None or not accumulator.decisions:
        return MultiwayModelDiagnostics(0, None, None, None, None, 0, None, None, {}, {})
    errors = accumulator.value_errors
    equity = None
    if accumulator.equity_predictions:
        equity = equity_metrics(accumulator.equity_predictions, accumulator.equity_targets or (), bins=bins)
    expected_share = None
    if accumulator.expected_share_predictions:
        expected_share = expected_showdown_share_metrics(
            accumulator.expected_share_predictions, accumulator.expected_share_targets or (), bins=bins
        )
    assert accumulator.expected_share_by_stratum is not None
    by_stratum = {
        key: expected_showdown_share_metrics(predictions, targets, bins=bins)
        for key, (predictions, targets) in sorted(accumulator.expected_share_by_stratum.items())
    }
    return MultiwayModelDiagnostics(
        decisions=accumulator.decisions,
        policy_entropy=sum(accumulator.entropy) / accumulator.decisions,
        value_mae_bb=sum(abs(value) for value in errors) / len(errors),
        value_rmse_bb=sqrt(sum(value * value for value in errors) / len(errors)),
        masked_action_rate=accumulator.masked_selected / accumulator.decisions,
        illegal_action_count=accumulator.masked_selected,
        equity=equity,
        expected_showdown_share=expected_share,
        expected_showdown_share_by_stratum=by_stratum,
        expected_showdown_share_support={
            key: by_stratum[key].samples if key in by_stratum else 0
            for key in sorted(set(by_stratum) | set(required_strata))
        },
    )


def _expected_share_stratum(observation: Mapping[str, object]) -> str:
    cards, table = observation.get("cards"), observation.get("table")
    if not isinstance(cards, Mapping) or not isinstance(table, Mapping):
        raise ValueError("model observation lacks expected-showdown-share stratum fields")
    street, active = cards.get("street"), table.get("active_player_count")
    if not isinstance(street, str) or isinstance(active, bool) or not isinstance(active, int) or active < 2:
        raise ValueError("model observation has invalid expected-showdown-share stratum fields")
    return f"street={street}|active_players={active}"


def pair_multiway_reports(transfer: MultiwayEvaluationReport, scratch: MultiwayEvaluationReport) -> PairedMultiwayEvaluation:
    """Calculate a fail-closed transfer-minus-scratch paired block CI."""

    if transfer.config.as_dict() != scratch.config.as_dict() or transfer.opponent_slots != scratch.opponent_slots:
        raise ValueError("paired reports require identical multiway config and opponent slot identities")
    if len(transfer.block_pnl_bb) != len(scratch.block_pnl_bb) or not transfer.block_pnl_bb:
        raise ValueError("paired reports require equal non-empty common-deal block vectors")
    deltas = tuple(left - right for left, right in zip(transfer.block_pnl_bb, scratch.block_pnl_bb, strict=True))
    standard_error = _block_standard_error(deltas)
    mean_delta_bb_per_100 = sum(deltas) / len(deltas) * 100.0
    margin = 1.96 * standard_error
    protocol_sha = _canonical_sha256(
        {
            "protocol": PAIRED_TRANSFER_SCRATCH_PROTOCOL,
            "config": transfer.config.as_dict(),
            "opponent_slots": list(transfer.opponent_slots),
            "transfer_candidate": transfer.candidate,
            "scratch_candidate": scratch.candidate,
        }
    )
    return PairedMultiwayEvaluation(
        transfer.candidate,
        scratch.candidate,
        len(deltas),
        deltas,
        mean_delta_bb_per_100,
        standard_error,
        mean_delta_bb_per_100 - margin,
        mean_delta_bb_per_100 + margin,
        protocol_sha,
    )


def _block_standard_error(block_values: Sequence[float]) -> float:
    if len(block_values) < 2:
        return 0.0
    mean = sum(block_values) / len(block_values)
    variance = sum((value - mean) ** 2 for value in block_values) / (len(block_values) - 1)
    return sqrt(variance / len(block_values)) * 100.0


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * log(value) for value in probabilities if value > 0.0)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return sha256(dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform/filesystem dependent.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - network filesystems can reject this.
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "EXPECTED_SHOWDOWN_SHARE_PROTOCOL",
    "MULTIWAY_EVALUATION_PROTOCOL",
    "MULTIWAY_EVALUATION_SCHEMA_VERSION",
    "PAIRED_BLOCK_CI_METHOD",
    "PAIRED_TRANSFER_SCRATCH_PROTOCOL",
    "MultiwayEvaluationConfig",
    "MultiwayEvaluationReport",
    "MultiwayModelDiagnostics",
    "OpponentSeat",
    "PairedMultiwayEvaluation",
    "evaluate_multiway_suite",
    "pair_multiway_reports",
]
