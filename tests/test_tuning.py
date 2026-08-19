"""Contracts for deterministic, artifact-first PPO tuning."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.experiment_runner import ExperimentRunner
from poker.experiments import ExperimentConfig
from poker.promotion import PromotionConfig, PromotionEvaluator
from poker.model import ModelConfig
from poker.train_runner import LeagueConfig, PPOConfig, RunSettings, TrainingRunConfig, TrainingRunner
from poker.tuning import (
    SweepConfig,
    TuningEvidence,
    compare_tuning_evidence,
    materialize_sweep,
    publish_tuning_evaluation,
    write_hu_promotion_protocol,
    write_sweep_config,
)
from poker.tuning_cli import main as tuning_main


PROTOCOL_PATH = Path(__file__).resolve()
PROTOCOL = sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _base() -> TrainingRunConfig:
    return TrainingRunConfig(
        run=RunSettings(
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            seed=17,
            iterations=3,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_iterations=4,
        ),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        ppo=PPOConfig(learning_rate=3e-4, epochs=2, minibatch_size=8, equity_samples=1),
        curriculum=CurriculumConfig(base_learning_rate=3e-4, require_transfer_beats_scratch=False),
        league=LeagueConfig(seed=91),
    )


def _sweep(**changes) -> SweepConfig:
    values = {
        "base_config": _base(),
        "grid": {"learning_rate": (1e-4, 3e-4), "epochs": (1, 2)},
        "seeds": (31, 11),
        "max_iterations": 9,
        "evaluation_protocol_sha256": PROTOCOL,
        "evaluation_protocol_path": str(PROTOCOL_PATH),
        "code_revision": "test-revision-1",
    }
    values.update(changes)
    return SweepConfig(**values)


def _real_sweep(tmp_path, *, accept: bool = True, **changes) -> tuple[SweepConfig, PromotionConfig]:
    promotion = PromotionConfig(
        enabled=True,
        hands_per_opponent=2,
        baseline_bots=("rule",),
        equity_samples=1,
        minimum_baseline_bb_per_100=-10_000.0 if accept else 1_000_000.0,
        minimum_baseline_ci95_low=-10_000.0,
        maximum_baseline_ci95_half_width=100_000.0,
        maximum_equity_ece=1.0,
    )
    protocol_path, protocol_sha = write_hu_promotion_protocol(
        _base(), promotion, tmp_path / ("protocol.json" if accept else "protocol-reject.json")
    )
    return _sweep(
        evaluation_protocol_path=str(protocol_path),
        evaluation_protocol_sha256=protocol_sha,
        **changes,
    ), promotion


def _evidence(tmp_path, spec, promotion: PromotionConfig) -> TuningEvidence:
    experiment = ExperimentConfig(
        name=spec.trial_id,
        training=spec.config,
        max_iterations=spec.config.run.iterations,
        evaluation_protocol_path=spec.evaluation_protocol_path,
        evaluation_protocol_sha256=spec.evaluation_protocol_sha256,
        code_revision=spec.code_revision,
    )
    experiment_runner = ExperimentRunner(experiment, tmp_path / "experiments" / spec.trial_id)
    result = experiment_runner.run(install_signal_handlers=False)
    assert result.status == "completed" and result.checkpoint_path is not None
    evaluator = PromotionEvaluator(promotion, tmp_path / "promotion" / spec.trial_id, run_seed=spec.seed)
    evaluated = evaluator.evaluate_and_promote(
        iteration=result.iteration,
        candidate_checkpoint=result.checkpoint_path,
        league=experiment_runner.trainer.league,
        stage=spec.config.run.stage,
        champion_score=None,
        run_context={
            "run_config_sha256": spec.run_config_sha256,
            "evaluation_protocol_sha256": spec.evaluation_protocol_sha256,
        },
    )
    return publish_tuning_evaluation(
        spec,
        result.checkpoint_path,
        experiment_runner.ledger.manifest_path,
        evaluated.report_path,
        evaluator.archive_manifest_path,
        tmp_path / "sealed" / f"{spec.trial_id}.json",
    )


def test_expand_is_deterministic_allowlisted_and_keeps_comparison_contract() -> None:
    first = _sweep()
    reordered = _sweep(grid={"epochs": (2, 1), "learning_rate": (3e-4, 1e-4)}, seeds=(11, 31))

    trials = first.expand_trials()

    assert [item.trial_id for item in trials] == [item.trial_id for item in reordered.expand_trials()]
    assert len(trials) == 8
    assert {item.seed for item in trials} == {11, 31}
    assert all(item.config.run.iterations == 9 and item.config.run.checkpoint_every_iterations == 1 for item in trials)
    assert all(item.config.run.stage is CurriculumStage.A_HEADS_UP_STARTER for item in trials)
    assert all(item.config.run.hands_per_iteration == 1 for item in trials)
    assert all(item.config.model == _base().model and item.config.league == _base().league for item in trials)
    for trial in trials:
        assert trial.config.ppo.learning_rate == trial.config.curriculum.base_learning_rate
    assert first.as_dict()["comparability"]["stage"] == "A"


@pytest.mark.parametrize(
    ("grid", "seeds", "message"),
    [
        ({"weight_decay": (0.1,)}, (1,), "unsupported"),
        ({"epochs": (1.0,)}, (1,), "integers"),
        ({"learning_rate": (float("nan"),)}, (1,), "finite"),
        ({"epochs": (1, 1)}, (1,), "unique"),
        ({}, (1, 1), "seeds"),
    ],
)
def test_sweep_rejects_nonfinite_or_unallowlisted_grid(grid, seeds, message) -> None:
    with pytest.raises(ValueError, match=message):
        _sweep(grid=grid, seeds=seeds)


def test_materialization_is_atomic_idempotent_and_rejects_collision(tmp_path) -> None:
    sweep = _sweep(grid={"learning_rate": (1e-4,)}, seeds=(7,))

    first = materialize_sweep(sweep, tmp_path / "trials")
    second = materialize_sweep(sweep, tmp_path / "trials")

    assert [item.directory for item in first] == [item.directory for item in second]
    trial = first[0]
    config = json.loads(trial.config_path.read_text(encoding="utf-8"))
    manifest = json.loads(trial.manifest_path.read_text(encoding="utf-8"))
    experiment = json.loads(trial.experiment_config_path.read_text(encoding="utf-8"))
    assert config["run"]["iterations"] == 9
    assert config["run"]["checkpoint_every_iterations"] == 1
    assert manifest["trial_id"] == trial.spec.trial_id
    assert manifest["evaluation_protocol_sha256"] == PROTOCOL
    assert experiment["code_revision"] == "test-revision-1"
    assert experiment["training"] == config
    assert not list((tmp_path / "trials").glob(".*.tmp"))

    trial.config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collision"):
        materialize_sweep(sweep, tmp_path / "trials")

    empty_root = tmp_path / "empty-collision"
    empty_root.mkdir()
    (empty_root / trial.spec.trial_id).mkdir()
    with pytest.raises(ValueError, match="collision"):
        materialize_sweep(sweep, empty_root)


def test_materialization_recovers_its_hash_bound_partial_reservation(tmp_path, monkeypatch) -> None:
    import poker.tuning as tuning_module

    sweep = _sweep(grid={}, seeds=(7,))
    spec = sweep.expand_trials()[0]
    root = tmp_path / "trials"
    original = tuning_module.os.replace
    crashed = False

    def crash_after_config(source, destination):
        nonlocal crashed
        original(source, destination)
        if not crashed and Path(destination) == root / spec.trial_id / "config.json":
            crashed = True
            raise OSError("simulated materialization crash")

    monkeypatch.setattr(tuning_module.os, "replace", crash_after_config)
    with pytest.raises(OSError, match="materialization crash"):
        materialize_sweep(sweep, root)
    monkeypatch.setattr(tuning_module.os, "replace", original)

    recovered = materialize_sweep(sweep, root)
    assert recovered[0].manifest_path.is_file()
    assert not (root / f".{spec.trial_id}.intent").exists()


def test_comparison_requires_complete_bound_evidence_and_ranks_passing_trials(tmp_path) -> None:
    sweep, promotion = _real_sweep(tmp_path, grid={"epochs": (1, 2)}, seeds=(4,), max_iterations=1)
    one, two = sweep.expand_trials()
    evidence = (
        _evidence(tmp_path, one, promotion),
        _evidence(tmp_path, two, promotion),
    )

    report = compare_tuning_evidence(sweep, evidence)

    expected = sorted(evidence, key=lambda item: (-item.score_lower_ci, item.expected_showdown_share_ece, item.trial_id))[0]
    assert report.winner is not None and report.winner.trial.trial_id == expected.trial_id
    assert [entry.rank for entry in report.entries] == [1, 2]
    written = report.write_json(tmp_path / "reports" / "comparison.json")
    encoded = json.loads(written.read_text(encoding="utf-8"))
    assert encoded["winner_trial_id"] == two.trial_id
    assert not list(written.parent.glob(".*.tmp"))

    with pytest.raises(ValueError, match="missing"):
        compare_tuning_evidence(sweep, evidence[:1])
    bad_protocol = replace(evidence[0], evaluation_protocol_sha256="e" * 64)
    with pytest.raises(ValueError, match="protocol"):
        compare_tuning_evidence(sweep, (bad_protocol, evidence[1]))
    reused = replace(evidence[1], full_checkpoint_sha256=evidence[0].full_checkpoint_sha256)
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        compare_tuning_evidence(sweep, (evidence[0], reused))


def test_failed_evidence_is_retained_but_never_ranked(tmp_path) -> None:
    passing_sweep, passing_promotion = _real_sweep(tmp_path / "pass", grid={}, seeds=(4,), max_iterations=1)
    failed_sweep, failed_promotion = _real_sweep(tmp_path / "fail", accept=False, grid={}, seeds=(4,), max_iterations=1)
    passing = _evidence(tmp_path / "pass", passing_sweep.expand_trials()[0], passing_promotion)
    failed = _evidence(tmp_path / "fail", failed_sweep.expand_trials()[0], failed_promotion)

    assert passing.passed and not failed.passed
    report = compare_tuning_evidence(passing_sweep, (passing,))
    failed_report = compare_tuning_evidence(failed_sweep, (failed,))

    assert report.winner is not None
    assert failed_report.winner is None and failed_report.entries[0].rank is None


def test_evidence_rejects_nonfinite_metrics_and_bad_hashes() -> None:
    spec = _sweep(grid={}, seeds=(1,)).expand_trials()[0]
    with pytest.raises(ValueError, match="finite"):
        TuningEvidence(spec.trial_id, "a.pt", "b" * 64, "a.json", "c" * 64, PROTOCOL, spec.run_config_sha256, float("inf"), 0.1, 0, True)
    with pytest.raises(ValueError, match="SHA-256"):
        TuningEvidence(spec.trial_id, "a.pt", "x", "a.json", "c" * 64, PROTOCOL, spec.run_config_sha256, 0.0, 0.1, 0, True)


def test_evidence_rejects_tampered_report_and_incomplete_checkpoint(tmp_path) -> None:
    sweep, promotion = _real_sweep(tmp_path, grid={}, seeds=(1,), max_iterations=2)
    spec = sweep.expand_trials()[0]
    runner = TrainingRunner(spec.config, tmp_path / "incomplete")
    runner.iteration = 1
    incomplete = runner.save_checkpoint(reason="paused")
    with pytest.raises(ValueError, match="completed"):
        publish_tuning_evaluation(spec, incomplete, tmp_path / "missing-ledger.json", tmp_path / "missing-report.json", tmp_path / "missing-archive.json", tmp_path / "report.json")

    runner.iteration = 2
    fabricated_complete = runner.save_checkpoint(reason="fabricated-complete")
    with pytest.raises(ValueError, match="experiment ledger"):
        publish_tuning_evaluation(spec, fabricated_complete, tmp_path / "missing-ledger.json", tmp_path / "missing-report.json", tmp_path / "missing-archive.json", tmp_path / "report.json")

    evidence = _evidence(tmp_path, spec, promotion)
    evidence.evaluation_report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report SHA-256"):
        compare_tuning_evidence(sweep, (evidence,))


def test_tuning_cli_materializes_seals_and_compares_one_trial(tmp_path) -> None:
    sweep, promotion = _real_sweep(tmp_path, grid={}, seeds=(5,), max_iterations=1)
    config_path = write_sweep_config(sweep, tmp_path / "sweep.json")
    trials = tmp_path / "trials"
    assert tuning_main(["materialize", "--config", str(config_path), "--output-dir", str(trials)]) == 0
    spec = sweep.expand_trials()[0]
    assert (trials / spec.trial_id / "experiment.json").is_file()

    experiment = ExperimentConfig(
        spec.trial_id, spec.config, 1, spec.evaluation_protocol_path,
        spec.evaluation_protocol_sha256, spec.code_revision,
    )
    experiment_runner = ExperimentRunner(experiment, tmp_path / "experiment")
    completed = experiment_runner.run(install_signal_handlers=False)
    assert completed.checkpoint_path is not None
    evaluator = PromotionEvaluator(promotion, tmp_path / "promotion", run_seed=spec.seed)
    evaluated = evaluator.evaluate_and_promote(
        iteration=1,
        candidate_checkpoint=completed.checkpoint_path,
        league=experiment_runner.trainer.league,
        stage=spec.config.run.stage,
        champion_score=None,
        run_context={"run_config_sha256": spec.run_config_sha256, "evaluation_protocol_sha256": spec.evaluation_protocol_sha256},
    )
    report = tmp_path / "evaluation.json"
    assert tuning_main([
        "seal", "--config", str(config_path), "--trial-id", spec.trial_id,
        "--checkpoint", str(completed.checkpoint_path),
        "--ledger-manifest", str(experiment_runner.ledger.manifest_path),
        "--promotion-report", str(evaluated.report_path),
        "--promotion-archive-manifest", str(evaluator.archive_manifest_path),
        "--report", str(report),
    ]) == 0
    comparison = tmp_path / "comparison.json"
    assert tuning_main([
        "compare", "--config", str(config_path), "--report", str(report), "--output", str(comparison),
    ]) == 0
    assert json.loads(comparison.read_text(encoding="utf-8"))["winner_trial_id"] == spec.trial_id
