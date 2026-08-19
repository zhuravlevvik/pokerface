"""Contracts for the bounded, artifact-only heads-up A -> B transition."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from poker.curriculum import CurriculumConfig, CurriculumStage, checkpoint_curriculum_metadata
from poker.curriculum_transition import CurriculumTransitionConfig, CurriculumTransitionEvaluator
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.train_runner import RunSettings, TrainingRunConfig, TrainingRunner
from poker.training import PPOConfig

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")


def _run_config(*, stage: CurriculumStage = CurriculumStage.A_HEADS_UP_STARTER) -> TrainingRunConfig:
    curriculum = CurriculumConfig(
        base_learning_rate=1e-3,
        min_baseline_win_rate_bb_per_100=-1_000_000.0,
        max_equity_calibration_error=1.0,
        require_transfer_beats_scratch=False,
        require_previous_checkpoint_win=True,
    )
    return TrainingRunConfig(
        run=RunSettings(stage=stage, seed=41, iterations=0, hands_per_iteration=1, table_count=1, checkpoint_every_seconds=None),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        ppo=PPOConfig(epochs=1, minibatch_size=8, equity_samples=1, learning_rate=1e-3),
        curriculum=curriculum,
    )


def _file_sha256(path) -> str:
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transition_config(reference) -> CurriculumTransitionConfig:
    return CurriculumTransitionConfig(
        enabled=True,
        every_iterations=1,
        hands_per_opponent=2,
        equity_samples=1,
        baseline_bots=("rule",),
        minimum_baseline_ci95_low=-1_000_000.0,
        maximum_baseline_ci95_half_width=1_000_000.0,
        minimum_prior_ci95_low=-1_000_000.0,
        reference_checkpoint=str(reference),
        reference_checkpoint_sha256=_file_sha256(reference),
        curriculum=_run_config().curriculum,
    )


def _run_context(config: TrainingRunConfig) -> dict[str, str]:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"run_config_sha256": sha256(encoded).hexdigest()}


def test_transition_freezes_evaluates_transfers_and_reloads_idempotently(tmp_path) -> None:
    config = _run_config()
    runner = TrainingRunner(config, tmp_path / "run")
    next(runner.model.parameters()).data.add_(0.01)
    source = runner.save_checkpoint(reason="candidate")
    reference = TrainingRunner(config, tmp_path / "reference").save_checkpoint(reason="reference")
    evaluator = CurriculumTransitionEvaluator(_transition_config(reference), tmp_path / "run", run_seed=config.run.seed)

    first = evaluator.evaluate_transition(
        iteration=1,
        candidate_checkpoint=source,
        reference_checkpoint=reference,
        stage=CurriculumStage.A_HEADS_UP_STARTER,
        run_context=_run_context(config),
    )

    assert first.accepted
    assert first.transfer_checkpoint_path is not None and first.transfer_checkpoint_path.exists()
    assert PokerAgentModel.load_checkpoint(first.transfer_checkpoint_path).checkpoint_metadata() == runner.model.checkpoint_metadata()
    transfer_metadata = checkpoint_curriculum_metadata(first.transfer_checkpoint_path)
    assert transfer_metadata["stage"] == "B"
    assert transfer_metadata["parent_checkpoint"] == str(first.frozen_source_path)
    assert first.frozen_source_path.exists()
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["transition_report_version"] == 2
    assert report["scalar_metric_protocol"] == "active_hands_expected_showdown_share_v1"
    assert report["suite"]["schema_version"] == "2.0"
    assert evaluator.last_evaluated_iteration == 1
    assert evaluator.last_accepted_decision is not None

    restored = CurriculumTransitionEvaluator(_transition_config(reference), tmp_path / "run", run_seed=config.run.seed)
    second = restored.evaluate_transition(
        iteration=1,
        candidate_checkpoint=source,
        reference_checkpoint=reference,
        stage=CurriculumStage.A_HEADS_UP_STARTER,
        run_context=_run_context(config),
    )
    manifest = json.loads(restored.archive_manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert second == first
    assert len(manifest["decisions"]) == 1
    assert not list((tmp_path / "run" / "curriculum-transitions").rglob(".*.tmp"))


def test_transition_rejects_wrong_stage_missing_reference_and_wrong_run_hash(tmp_path) -> None:
    config = _run_config(stage=CurriculumStage.B_HEADS_UP_FULL)
    source = TrainingRunner(config, tmp_path / "stage-b").save_checkpoint(reason="candidate")
    config_a = _run_config()
    reference = TrainingRunner(config_a, tmp_path / "reference").save_checkpoint(reason="reference")
    evaluator = CurriculumTransitionEvaluator(_transition_config(reference), tmp_path / "stage-b")

    with pytest.raises(ValueError, match="only evaluates stage A"):
        evaluator.evaluate_transition(
            iteration=1,
            candidate_checkpoint=source,
            reference_checkpoint=source,
            stage=CurriculumStage.B_HEADS_UP_FULL,
            run_context=_run_context(config),
        )
    source_a = TrainingRunner(config_a, tmp_path / "stage-a").save_checkpoint(reason="candidate")
    missing_evaluator = CurriculumTransitionEvaluator(
        replace(
            _transition_config(reference),
            reference_checkpoint=str(tmp_path / "missing.pt"),
            reference_checkpoint_sha256="0" * 64,
        ),
        tmp_path / "missing-reference",
    )
    with pytest.raises(FileNotFoundError):
        missing_evaluator.evaluate_transition(
            iteration=1,
            candidate_checkpoint=source_a,
            reference_checkpoint=None,
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            run_context=_run_context(config_a),
        )
    with pytest.raises(ValueError, match="weights are identical"):
        evaluator.evaluate_transition(
            iteration=1,
            candidate_checkpoint=source_a,
            reference_checkpoint=reference,
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            run_context=_run_context(config_a),
        )
    with pytest.raises(ValueError, match="run_config_sha256"):
        evaluator.evaluate_transition(
            iteration=1,
            candidate_checkpoint=source_a,
            reference_checkpoint=reference,
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            run_context={"run_config_sha256": "tampered"},
        )
    with pytest.raises(ValueError, match="only A -> B"):
        CurriculumTransitionConfig(source_stage=CurriculumStage.B_HEADS_UP_FULL, target_stage=CurriculumStage.C_THREE_MAX)
    with pytest.raises(ValueError, match="requires a non-empty reference_checkpoint"):
        CurriculumTransitionConfig(enabled=True)
    assert CurriculumTransitionConfig().minimum_prior_ci95_low == 0.0
    with pytest.raises(ValueError, match="pinned reference_checkpoint_sha256"):
        CurriculumTransitionConfig(enabled=True, reference_checkpoint=str(reference))


def test_manifest_reload_fails_closed_on_report_tampering(tmp_path) -> None:
    config = _run_config()
    runner = TrainingRunner(config, tmp_path / "run")
    next(runner.model.parameters()).data.add_(0.01)
    source = runner.save_checkpoint(reason="candidate")
    reference = TrainingRunner(config, tmp_path / "reference").save_checkpoint(reason="reference")
    evaluator = CurriculumTransitionEvaluator(_transition_config(reference), tmp_path / "run", run_seed=config.run.seed)
    result = evaluator.evaluate_transition(
        iteration=1,
        candidate_checkpoint=source,
        reference_checkpoint=reference,
        stage=CurriculumStage.A_HEADS_UP_STARTER,
        run_context=_run_context(config),
    )
    result.report_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="report_path is missing or hash-mismatched"):
        CurriculumTransitionEvaluator(_transition_config(reference), tmp_path / "run", run_seed=config.run.seed)
