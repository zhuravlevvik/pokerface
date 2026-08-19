"""Curriculum contracts for progressively training the Hold'em agent.

This module deliberately does *not* contain PPO, league self-play, or a
trainer.  It makes the boundaries between those future components explicit:
the stage selects a legal game, checkpoints retain their provenance, and a
fixed previous-stage suite detects regressions after transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .betting import Action, RAISE_ACTIONS
from .equity import equity_metrics
from .game_state import HandState
from .model import MODEL_VERSION, TORCH_AVAILABLE, PokerAgentModel
from .observation import observation_for
from .rules import BIG_BLIND
from .simulator import BatchedHoldemEnvironment
from .traces import HandTrace


class CurriculumStage(str, Enum):
    """Ordered curriculum stages from the project specification."""

    A_HEADS_UP_STARTER = "A"
    B_HEADS_UP_FULL = "B"
    C_THREE_MAX = "C"
    D_FIVE_MAX_FIXED = "D"
    E_FIVE_MAX_EXPANDED = "E"


_STARTER_RAISES = frozenset({Action.RAISE_MIN, Action.RAISE_1_2_POT, Action.RAISE_POT, Action.ALL_IN})
_FULL_RAISES = frozenset((*RAISE_ACTIONS, Action.ALL_IN))


@dataclass(frozen=True)
class CurriculumStageSpec:
    """The game distribution and optimiser adjustment for one stage."""

    stage: CurriculumStage
    player_count: int
    allowed_raise_actions: frozenset[Action]
    starting_stacks_bb: tuple[int, ...]
    learning_rate_scale: float

    def __post_init__(self) -> None:
        if self.player_count not in (2, 3, 5):
            raise ValueError("curriculum supports only 2-, 3-, and 5-max stages")
        if not self.allowed_raise_actions or not self.allowed_raise_actions.issubset(RAISE_ACTIONS | {Action.ALL_IN}):
            raise ValueError("allowed_raise_actions must be a non-empty raise-action subset")
        if not self.starting_stacks_bb or any(stack < 1 for stack in self.starting_stacks_bb):
            raise ValueError("starting_stacks_bb must contain positive stack sizes")
        if self.learning_rate_scale <= 0:
            raise ValueError("learning_rate_scale must be positive")

    def starting_stack_chips(self, *, variant_index: int = 0) -> int:
        """Return a deterministic stack variant without hiding sampling policy."""

        return self.starting_stacks_bb[variant_index % len(self.starting_stacks_bb)] * BIG_BLIND


STAGE_SPECS: Mapping[CurriculumStage, CurriculumStageSpec] = {
    CurriculumStage.A_HEADS_UP_STARTER: CurriculumStageSpec(CurriculumStage.A_HEADS_UP_STARTER, 2, _STARTER_RAISES, (100,), 1.0),
    CurriculumStage.B_HEADS_UP_FULL: CurriculumStageSpec(CurriculumStage.B_HEADS_UP_FULL, 2, _FULL_RAISES, (100,), 0.7),
    CurriculumStage.C_THREE_MAX: CurriculumStageSpec(CurriculumStage.C_THREE_MAX, 3, _FULL_RAISES, (100,), 0.5),
    CurriculumStage.D_FIVE_MAX_FIXED: CurriculumStageSpec(CurriculumStage.D_FIVE_MAX_FIXED, 5, _FULL_RAISES, (100,), 0.35),
    CurriculumStage.E_FIVE_MAX_EXPANDED: CurriculumStageSpec(CurriculumStage.E_FIVE_MAX_EXPANDED, 5, _FULL_RAISES, (50, 100, 200), 0.25),
}
"""Immutable canonical curriculum; stages must be traversed in enum order."""


def stage_spec(stage: CurriculumStage | str) -> CurriculumStageSpec:
    try:
        return STAGE_SPECS[CurriculumStage(stage)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unknown curriculum stage: {stage!r}") from error


@dataclass(frozen=True)
class CurriculumConfig:
    """Explicit, serialisable gate thresholds for a training run."""

    base_learning_rate: float = 3e-4
    min_baseline_win_rate_bb_per_100: float = 0.0
    max_equity_calibration_error: float = 0.08
    require_transfer_beats_scratch: bool = True
    require_previous_checkpoint_win: bool = True

    def __post_init__(self) -> None:
        if self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be positive")
        if not 0 <= self.max_equity_calibration_error <= 1:
            raise ValueError("max_equity_calibration_error must be in [0, 1]")

    def learning_rate_for(self, stage: CurriculumStage | str) -> float:
        return self.base_learning_rate * stage_spec(stage).learning_rate_scale


@dataclass(frozen=True)
class StageEvaluation:
    """Trainer-produced, comparable result required before advancing.

    ``equity_calibration_error`` is a compatibility name for the explicitly
    protocolled expected-showdown-share ECE in automated evaluators; it must
    never be populated from ``win + 0.5 * tie`` for a multiway stage.

    ``transfer_bb_per_100`` and ``scratch_bb_per_100`` use the same fixed
    seeds, opponent pool, positions, and step budget.  This is the required
    evidence that transfer is better than starting the target stage from zero.
    """

    baseline_win_rate_bb_per_100: float
    equity_calibration_error: float
    beats_previous_checkpoint: bool
    transfer_bb_per_100: float | None = None
    scratch_bb_per_100: float | None = None
    control_set_passed: bool = True

    def passes(self, config: CurriculumConfig) -> bool:
        transfer_ok = (
            not config.require_transfer_beats_scratch
            or (self.transfer_bb_per_100 is not None and self.scratch_bb_per_100 is not None and self.transfer_bb_per_100 > self.scratch_bb_per_100)
        )
        previous_ok = not config.require_previous_checkpoint_win or self.beats_previous_checkpoint
        return (
            self.baseline_win_rate_bb_per_100 >= config.min_baseline_win_rate_bb_per_100
            and self.equity_calibration_error <= config.max_equity_calibration_error
            and previous_ok
            and transfer_ok
            and self.control_set_passed
        )


class StageTransitionError(RuntimeError):
    """Raised when a curriculum transition skips a stage or lacks evidence."""


class StageScheduler:
    """Small state machine which creates stage-constrained environments."""

    def __init__(self, config: CurriculumConfig | None = None, *, initial_stage: CurriculumStage | str = CurriculumStage.A_HEADS_UP_STARTER) -> None:
        self.config = config or CurriculumConfig()
        self._stage = CurriculumStage(initial_stage)

    @property
    def stage(self) -> CurriculumStage:
        return self._stage

    @property
    def spec(self) -> CurriculumStageSpec:
        return stage_spec(self._stage)

    @property
    def next_stage(self) -> CurriculumStage | None:
        stages = tuple(CurriculumStage)
        index = stages.index(self._stage)
        return None if index + 1 == len(stages) else stages[index + 1]

    def can_advance(self, evaluation: StageEvaluation) -> bool:
        return self.next_stage is not None and evaluation.passes(self.config)

    def advance(self, evaluation: StageEvaluation) -> CurriculumStage:
        if self.next_stage is None:
            raise StageTransitionError("the final curriculum stage cannot advance")
        if not evaluation.passes(self.config):
            raise StageTransitionError("stage evaluation did not satisfy curriculum gates")
        self._stage = self.next_stage
        return self._stage

    def make_environment(self, table_count: int = 1, *, capture_replays: bool = False, stack_variant_index: int = 0) -> BatchedHoldemEnvironment:
        """Create an executable environment constrained to the current stage."""

        return BatchedHoldemEnvironment(
            table_count,
            starting_stack=self.spec.starting_stack_chips(variant_index=stack_variant_index),
            player_count=self.spec.player_count,
            allowed_raise_actions=self.spec.allowed_raise_actions,
            capture_replays=capture_replays,
        )


@dataclass(frozen=True)
class RegressionCase:
    """One deterministic prior-stage hand used to detect forgetting."""

    case_id: str
    stage: CurriculumStage
    seed: int
    button_seat: int
    stack_variant_index: int = 0

    def state(self) -> HandState:
        spec = stage_spec(self.stage)
        return HandState(
            seed=self.seed,
            button_seat=self.button_seat,
            starting_stack=spec.starting_stack_chips(variant_index=self.stack_variant_index),
            player_count=spec.player_count,
            allowed_raise_actions=spec.allowed_raise_actions,
        )


PolicySelector = Callable[[Mapping[str, Any], Mapping[str, bool], int], Action | str]


@dataclass(frozen=True)
class RegressionResult:
    case_id: str
    action_count: int
    terminal_pnl_bb: Mapping[int, float]
    replay: Mapping[str, Any]


@dataclass(frozen=True)
class RegressionControlSet:
    """Frozen seeds/configuration for re-running earlier game distributions."""

    name: str
    stage: CurriculumStage
    cases: tuple[RegressionCase, ...]

    @classmethod
    def for_stage(cls, stage: CurriculumStage | str, *, case_count: int = 32, seed_start: int = 10_000) -> "RegressionControlSet":
        if case_count < 1:
            raise ValueError("case_count must be positive")
        resolved = CurriculumStage(stage)
        spec = stage_spec(resolved)
        return cls(
            name=f"{resolved.value}-holdout-{seed_start}-{case_count}",
            stage=resolved,
            cases=tuple(
                RegressionCase(f"{resolved.value}:{index}", resolved, seed_start + index, index % spec.player_count, index % len(spec.starting_stacks_bb))
                for index in range(case_count)
            ),
        )

    def run(self, selector: PolicySelector) -> tuple[RegressionResult, ...]:
        results: list[RegressionResult] = []
        for case in self.cases:
            state = case.state()
            actions = 0
            while not state.complete:
                assert state.actor is not None
                observation = observation_for(state, state.actor)
                action = Action(selector(observation, observation["legal_actions"], state.actor))
                if not state.legal_actions()[action]:
                    raise ValueError(f"selector chose illegal action {action.value!r} in {case.case_id}")
                state.step(action)
                actions += 1
            results.append(
                RegressionResult(
                    case.case_id,
                    actions,
                    {player.seat: (player.stack - state.starting_stack) / BIG_BLIND for player in state.players},
                    state.replay(),
                )
            )
        return tuple(results)


@dataclass(frozen=True)
class PretrainingExample:
    """Model-safe supervised sample from random or rule-based play."""

    observation: Mapping[str, Any]
    selected_action: str
    equity_target: tuple[float, float, float]
    terminal_pnl_bb: float
    stage: CurriculumStage
    # Audit fields are appended so existing positional construction remains
    # valid.  They are populated by trace-backed corpus generation.
    hand_id: int | None = None
    seed: int | None = None
    equity_samples: int | None = None
    equity_exact: bool | None = None
    label_protocol: str | None = None
    behavior_policy: str | None = None
    # Expected share of a showdown among active hands, not an awarded pot share.
    # Appended to keep all older positional construction valid.
    expected_share_target: float | None = None

    @property
    def expected_showdown_share_target(self) -> float | None:
        """Explicit semantic alias for the Stage 4 scalar label."""

        return self.expected_share_target


class TracePretrainingDataset(Sequence[PretrainingExample]):
    """Minimal dataset adapter; the future trainer may batch this directly."""

    def __init__(self, examples: Iterable[PretrainingExample]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> PretrainingExample:
        return self._examples[index]

    @classmethod
    def from_traces(cls, stage: CurriculumStage | str, traces: Iterable[HandTrace]) -> "TracePretrainingDataset":
        resolved = CurriculumStage(stage)
        examples: list[PretrainingExample] = []
        for trace in traces:
            for record in trace.as_training_records():
                target = record["equity_target"]
                pnl = record["terminal_pnl_bb"]
                if not isinstance(target, list) or len(target) != 3 or not isinstance(pnl, (int, float)):
                    raise ValueError("terminal trace has malformed equity supervision")
                samples = record.get("equity_samples")
                exact = record.get("equity_exact")
                hand_id = record.get("hand_id")
                seed = record.get("seed")
                label_protocol = record.get("label_protocol")
                expected_share = record.get("expected_showdown_share_target")
                if isinstance(expected_share, bool) or not isinstance(expected_share, (int, float)):
                    raise ValueError("terminal trace has no expected showdown-share supervision")
                examples.append(
                    PretrainingExample(
                        record["observation"],
                        str(record["selected_action"]),
                        tuple(float(value) for value in target),
                        float(pnl),
                        resolved,
                        hand_id if isinstance(hand_id, int) else None,
                        seed if isinstance(seed, int) else None,
                        samples if isinstance(samples, int) else None,
                        exact if isinstance(exact, bool) else None,
                        label_protocol if isinstance(label_protocol, str) else None,
                        None,
                        float(expected_share),
                    )
                )
        return cls(examples)

    def equity_quality(self, predictions: Iterable[Sequence[float]]) -> Mapping[str, Any]:
        targets = [example.equity_target for example in self]
        return equity_metrics(predictions, targets).as_dict()


def generate_pretraining_dataset(
    stage: CurriculumStage | str,
    hand_count: int,
    selector: PolicySelector,
    *,
    seed_start: int = 0,
    equity_samples: int = 16,
) -> TracePretrainingDataset:
    """Generate legal observations/targets without committing to a trainer.

    Pass a random or rule-based selector here for auxiliary card/backbone/equity
    pretraining.  Only the trace's public observation and labels reach the
    returned dataset; opponents' cards remain inside ``HandTrace`` temporarily.
    """

    if hand_count < 1:
        raise ValueError("hand_count must be positive")
    resolved = CurriculumStage(stage)
    spec = stage_spec(resolved)
    traces: list[HandTrace] = []
    for index in range(hand_count):
        state = HandState(
            seed=seed_start + index,
            button_seat=index % spec.player_count,
            starting_stack=spec.starting_stack_chips(variant_index=index),
            player_count=spec.player_count,
            allowed_raise_actions=spec.allowed_raise_actions,
        )
        trace = HandTrace(index, state.seed, state.button_seat, state.starting_stack, equity_samples=equity_samples)
        while not state.complete:
            assert state.actor is not None
            observation = observation_for(state, state.actor)
            action = Action(selector(observation, observation["legal_actions"], state.actor))
            if not state.legal_actions()[action]:
                raise ValueError(f"selector chose illegal action {action.value!r}")
            trace.record_action(state, action)
            state.step(action)
        trace.complete(state)
        traces.append(trace)
    return TracePretrainingDataset.from_traces(resolved, traces)


@dataclass(frozen=True)
class CheckpointTransfer:
    """Provenance recorded when continuing a model in the next stage."""

    source_path: str
    destination_path: str
    source_stage: CurriculumStage
    target_stage: CurriculumStage
    global_step: int
    learning_rate: float


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("checkpoint transfer requires PyTorch; install with `.[rl]")
    import torch

    return torch


def save_curriculum_checkpoint(
    model: PokerAgentModel,
    path: str | Path,
    *,
    stage: CurriculumStage | str,
    global_step: int,
    parent_checkpoint: str | Path | None = None,
) -> None:
    """Save a normal model checkpoint plus curriculum provenance.

    The payload remains loadable through ``PokerAgentModel.load_checkpoint``;
    it only adds fields and does not alter the model compatibility metadata.
    """

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    torch = _require_torch()
    resolved = CurriculumStage(stage)
    payload = {
        "metadata": model.checkpoint_metadata(),
        "state_dict": model.state_dict(),
        "curriculum": {
            "version": 1,
            "stage": resolved.value,
            "global_step": global_step,
            "parent_checkpoint": None if parent_checkpoint is None else str(parent_checkpoint),
            "stage_spec": {
                "player_count": stage_spec(resolved).player_count,
                "allowed_raise_actions": sorted(action.value for action in stage_spec(resolved).allowed_raise_actions),
                "starting_stacks_bb": list(stage_spec(resolved).starting_stacks_bb),
            },
        },
    }
    torch.save(payload, Path(path))


def checkpoint_curriculum_metadata(path: str | Path) -> Mapping[str, Any]:
    """Read and validate only the curriculum metadata of a checkpoint."""

    torch = _require_torch()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    curriculum = payload.get("curriculum") if isinstance(payload, Mapping) else None
    if not isinstance(curriculum, Mapping) or curriculum.get("version") != 1:
        raise ValueError("checkpoint has no compatible curriculum metadata")
    CurriculumStage(curriculum.get("stage"))
    return dict(curriculum)


def transfer_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    target_stage: CurriculumStage | str,
    config: CurriculumConfig | None = None,
    global_step: int = 0,
) -> CheckpointTransfer:
    """Copy compatible weights and record the lower target-stage learning rate."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    torch = _require_torch()
    source = Path(source_path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("invalid checkpoint payload")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("incompatible model checkpoint")
    source_info = checkpoint_curriculum_metadata(source)
    source_stage = CurriculumStage(source_info["stage"])
    target = CurriculumStage(target_stage)
    stages = tuple(CurriculumStage)
    if stages.index(target) != stages.index(source_stage) + 1:
        raise StageTransitionError("checkpoint transfer must target the immediately next curriculum stage")
    settings = config or CurriculumConfig()
    target_info = dict(source_info)
    target_info.update(
        {
            "stage": target.value,
            "global_step": global_step,
            "parent_checkpoint": str(source),
            "stage_spec": {
                "player_count": stage_spec(target).player_count,
                "allowed_raise_actions": sorted(action.value for action in stage_spec(target).allowed_raise_actions),
                "starting_stacks_bb": list(stage_spec(target).starting_stacks_bb),
            },
        }
    )
    torch.save({"metadata": dict(metadata), "state_dict": payload["state_dict"], "curriculum": target_info}, Path(destination_path))
    return CheckpointTransfer(str(source), str(destination_path), source_stage, target, global_step, settings.learning_rate_for(target))


__all__ = [
    "CheckpointTransfer",
    "CurriculumConfig",
    "CurriculumStage",
    "CurriculumStageSpec",
    "PolicySelector",
    "PretrainingExample",
    "RegressionCase",
    "RegressionControlSet",
    "RegressionResult",
    "STAGE_SPECS",
    "StageEvaluation",
    "StageScheduler",
    "StageTransitionError",
    "TracePretrainingDataset",
    "checkpoint_curriculum_metadata",
    "generate_pretraining_dataset",
    "save_curriculum_checkpoint",
    "stage_spec",
    "transfer_checkpoint",
]
