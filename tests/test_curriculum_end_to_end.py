"""Small native end-to-end proof for one paired adjacent transition."""

from __future__ import annotations

import pytest

from poker.betting import Action
from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.curriculum_coordinator import (
    CurriculumCoordinator,
    CurriculumCoordinatorConfig,
    EvaluationProtocol,
    OpponentSpec,
    native_paired_rung_runner,
)
from poker.curriculum_runtime import native_multiway_evaluator
from poker.model import ModelConfig, TORCH_AVAILABLE
from poker.multiway_evaluation import MultiwayEvaluationConfig
from poker.paired_rung import PairedRungConfig
from poker.train_runner import RunSettings, TrainingRunConfig, TrainingRunner
from poker.training import PPOConfig


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")


def _source_run(tmp_path, name: str, seed: int):
    ppo = PPOConfig(learning_rate=1e-3, epochs=1, minibatch_size=8, equity_samples=1)
    config = TrainingRunConfig(
        run=RunSettings(
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            seed=seed,
            iterations=1,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_iterations=1,
            checkpoint_every_seconds=None,
        ),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        ppo=ppo,
        curriculum=CurriculumConfig(
            base_learning_rate=ppo.learning_rate,
            require_transfer_beats_scratch=False,
            require_previous_checkpoint_win=False,
        ),
    )
    return TrainingRunner(config, tmp_path / name).run(install_signal_handlers=False).checkpoint_path


def test_native_paired_transition_adopts_a_resumable_full_target_checkpoint(tmp_path) -> None:
    source = _source_run(tmp_path, "source-run", 301)
    reference = _source_run(tmp_path, "reference-run", 302)
    starter_actions = (Action.RAISE_MIN, Action.RAISE_1_2_POT, Action.RAISE_POT, Action.ALL_IN)
    source_eval = MultiwayEvaluationConfig(
        player_count=2,
        deal_blocks=2,
        seed_start=8_000_000,
        allowed_raise_actions=starter_actions,
        equity_samples=1,
    )
    target_eval = MultiwayEvaluationConfig(
        player_count=2,
        deal_blocks=2,
        seed_start=8_100_000,
        equity_samples=1,
    )
    bot = (OpponentSpec("rule", {"kind": "bot", "bot": "rule"}),)
    rung_config = PairedRungConfig(
        target_stage=CurriculumStage.B_HEADS_UP_FULL,
        iterations=1,
        hands_per_iteration=1,
        table_count=1,
        base_seed=401,
        ppo=PPOConfig(learning_rate=1e-3, epochs=1, minibatch_size=8, equity_samples=1),
    )
    coordinator_config = CurriculumCoordinatorConfig(
        source_stage=CurriculumStage.A_HEADS_UP_STARTER,
        target_stage=CurriculumStage.B_HEADS_UP_FULL,
        source_protocol=EvaluationProtocol("source", bot, source_eval.as_dict()),
        target_protocols=(EvaluationProtocol("target", bot, target_eval.as_dict()),),
        paired_rung_protocol_sha256=rung_config.protocol_sha256,
        min_transfer_delta_ci95_low_bb_per_100=-1e9,
        min_target_baseline_ci95_low_bb_per_100=-1e9,
        min_source_delta_ci95_low_bb_per_100=-1e9,
        max_expected_showdown_share_ece=1.0,
        max_expected_showdown_share_mae=1.0,
    )
    coordinator = CurriculumCoordinator(
        coordinator_config,
        tmp_path / "coordinator",
        rung_runner=native_paired_rung_runner(rung_config, tmp_path / "rungs"),
        evaluator=native_multiway_evaluator(),
    )

    decision = coordinator.coordinate(source, reference)

    assert decision.accepted
    assert decision.adopted_checkpoint is not None
    adopted = TrainingRunner.resume(decision.adopted_checkpoint)
    assert adopted.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
    assert adopted.iteration == 1
    assert adopted.global_hands == 1
