"""Acceptance tests for the staged 2/3/5-max curriculum contract."""

from __future__ import annotations

import pytest

from poker.betting import Action
from poker.curriculum import (
    CurriculumConfig,
    CurriculumStage,
    RegressionControlSet,
    STAGE_SPECS,
    StageEvaluation,
    StageScheduler,
    StageTransitionError,
    TORCH_AVAILABLE,
    checkpoint_curriculum_metadata,
    generate_pretraining_dataset,
    save_curriculum_checkpoint,
    transfer_checkpoint,
)
from poker.game_state import HandState
from poker.model import ModelConfig, PokerAgentModel
from poker.observation import observation_for
from poker.traces import rebuild_hand


def _check_or_call(_observation: dict, legal: dict[str, bool], _seat: int) -> Action:
    return Action.CHECK if legal[Action.CHECK.value] else Action.CALL


@pytest.mark.parametrize(
    ("player_count", "expected_positions", "first_actor"),
    [
        (2, {"BTN", "BB"}, 0),
        (3, {"BTN", "SB", "BB"}, 0),
        (5, {"BTN", "SB", "BB", "UTG", "CO"}, 3),
    ],
)
def test_short_handed_engine_uses_only_active_seats_and_replays(player_count: int, expected_positions: set[str], first_actor: int) -> None:
    state = HandState(seed=123, player_count=player_count)
    assert state.actor == first_actor
    assert len(state.players) == player_count
    assert set(state.positions.values()) == expected_positions
    assert len(observation_for(state, state.actor)["player_set"]) == player_count  # type: ignore[arg-type]
    while not state.complete:
        assert state.actor is not None
        state.step(_check_or_call({}, {action.value: allowed for action, allowed in state.legal_actions().items()}, state.actor))
    assert rebuild_hand(state.replay()).replay() == state.replay()


def test_stage_scheduler_enforces_evidence_and_builds_short_handed_environment() -> None:
    scheduler = StageScheduler()
    environment = scheduler.make_environment(2)
    observations = environment.reset(seeds=[1, 2])
    assert all(len(observation["player_set"]) == 2 for observation in observations)
    first_mask = environment.legal_action_masks[0]
    assert first_mask[Action.RAISE_1_3_POT.value] is False
    assert first_mask[Action.RAISE_1_2_POT.value] is True

    failed = StageEvaluation(-1.0, 0.5, False, 1.0, 2.0, False)
    assert not scheduler.can_advance(failed)
    with pytest.raises(StageTransitionError):
        scheduler.advance(failed)

    passed = StageEvaluation(1.0, 0.01, True, 3.0, 2.0, True)
    assert scheduler.advance(passed) is CurriculumStage.B_HEADS_UP_FULL
    assert scheduler.config.learning_rate_for(scheduler.stage) < scheduler.config.base_learning_rate


def test_regression_control_set_is_fixed_and_stage_correct() -> None:
    control = RegressionControlSet.for_stage(CurriculumStage.C_THREE_MAX, case_count=3, seed_start=81)
    first = control.run(_check_or_call)
    second = control.run(_check_or_call)
    assert [result.replay for result in first] == [result.replay for result in second]
    assert all(len(result.terminal_pnl_bb) == 3 for result in first)
    assert all(result.replay["player_count"] == 3 for result in first)


def test_pretraining_hook_exports_safe_labeled_traces() -> None:
    dataset = generate_pretraining_dataset(CurriculumStage.A_HEADS_UP_STARTER, 2, _check_or_call, seed_start=33, equity_samples=1)
    assert len(dataset) > 0
    example = dataset[0]
    assert example.stage is CurriculumStage.A_HEADS_UP_STARTER
    assert sum(example.equity_target) == pytest.approx(1.0)
    assert all("hole_cards" not in player for player in example.observation["players"])
    metrics = dataset.equity_quality([example.equity_target for _ in dataset])
    assert metrics["samples"] == len(dataset)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_checkpoint_transfer_keeps_weights_and_records_curriculum_provenance(tmp_path) -> None:
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    source = tmp_path / "stage-a.pt"
    destination = tmp_path / "stage-b.pt"
    save_curriculum_checkpoint(model, source, stage="A", global_step=10)
    transfer = transfer_checkpoint(source, destination, target_stage="B", global_step=20)
    assert transfer.source_stage is CurriculumStage.A_HEADS_UP_STARTER
    assert transfer.target_stage is CurriculumStage.B_HEADS_UP_FULL
    metadata = checkpoint_curriculum_metadata(destination)
    assert metadata["stage"] == "B"
    assert metadata["parent_checkpoint"] == str(source)
    restored = PokerAgentModel.load_checkpoint(destination)
    assert restored.checkpoint_metadata() == model.checkpoint_metadata()
