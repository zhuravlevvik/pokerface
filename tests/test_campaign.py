"""Multi-seed campaign selection and fixed evaluation RNG contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from poker.campaign import CampaignConfig, aggregate_campaign, verify_campaign_report
from poker.campaign_cli import main as campaign_main
from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.experiment_runner import ExperimentRunner
from poker.experiments import ExperimentConfig
from poker.model import ModelConfig
from poker.promotion import PromotionConfig, PromotionEvaluator
from poker.train_runner import LeagueConfig, PPOConfig, RunSettings, TrainingRunConfig
from poker.tuning import SweepConfig, publish_tuning_evaluation, write_hu_promotion_protocol, write_sweep_config


def _base() -> TrainingRunConfig:
    return TrainingRunConfig(
        run=RunSettings(
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            seed=1,
            iterations=1,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_iterations=1,
        ),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        ppo=PPOConfig(learning_rate=3e-4, epochs=1, minibatch_size=8, equity_samples=1),
        curriculum=CurriculumConfig(base_learning_rate=3e-4, require_transfer_beats_scratch=False),
        league=LeagueConfig(seed=9),
    )


def _campaign(tmp_path, *, baseline="rule"):
    promotion = PromotionConfig(
        enabled=True,
        hands_per_opponent=2,
        baseline_bots=(baseline,),
        equity_samples=1,
        calibration_bins=2,
        minimum_baseline_bb_per_100=-1e9,
        minimum_baseline_ci95_low=-1e9,
        maximum_baseline_ci95_half_width=1e9,
        maximum_equity_ece=1.0,
    )
    protocol_path, protocol_sha = write_hu_promotion_protocol(_base(), promotion, tmp_path / "protocol.json")
    sweep = SweepConfig(
        base_config=_base(),
        grid={},
        seeds=(11, 31),
        max_iterations=1,
        evaluation_protocol_sha256=protocol_sha,
        evaluation_protocol_path=str(protocol_path),
        code_revision="test-revision-campaign",
    )
    return CampaignConfig(
        sweep,
        minimum_seeds_per_variant=2,
        minimum_baseline_ci95_low=-1e9,
        maximum_expected_showdown_share_ece=1.0,
    ), promotion


def _evidence(tmp_path, config, promotion):
    result = []
    for spec in config.sweep.expand_trials():
        experiment = ExperimentConfig(
            spec.trial_id,
            spec.config,
            spec.config.run.iterations,
            spec.evaluation_protocol_path,
            spec.evaluation_protocol_sha256,
            spec.code_revision,
        )
        runner = ExperimentRunner(experiment, tmp_path / "runs" / spec.trial_id)
        completed = runner.run(install_signal_handlers=False)
        assert completed.checkpoint_path is not None
        evaluator = PromotionEvaluator(promotion, tmp_path / "evaluations" / spec.trial_id, run_seed=0)
        evaluated = evaluator.evaluate_and_promote(
            iteration=completed.iteration,
            candidate_checkpoint=completed.checkpoint_path,
            league=runner.trainer.league,
            stage=spec.config.run.stage,
            champion_score=None,
            run_context={
                "run_config_sha256": spec.run_config_sha256,
                "evaluation_protocol_sha256": spec.evaluation_protocol_sha256,
                "evaluation_run_seed": 0,
            },
        )
        result.append(publish_tuning_evaluation(
            spec,
            completed.checkpoint_path,
            runner.ledger.manifest_path,
            evaluated.report_path,
            evaluator.archive_manifest_path,
            tmp_path / "sealed" / f"{spec.trial_id}.json",
        ))
    return tuple(result)


def test_campaign_requires_complete_matrix_and_selects_only_cross_seed_variant(tmp_path) -> None:
    config, promotion = _campaign(tmp_path)
    evidence = _evidence(tmp_path, config, promotion)

    report = aggregate_campaign(config, evidence)

    assert report.winner is not None
    assert report.winner.seeds == (11, 31)
    assert report.winner.rank == 1
    assert report.winner.baselines[0].seed_count == 2
    assert report.winner.baselines[0].as_dict()["ci_method"] == "training_seed_student_t_v1"
    with pytest.raises(ValueError, match="missing trials"):
        aggregate_campaign(config, evidence[:1])

    path = report.write_json(tmp_path / "campaign.json")
    assert verify_campaign_report(config, path).winner == report.winner


def test_campaign_detects_changed_underlying_promotion_evidence(tmp_path) -> None:
    config, promotion = _campaign(tmp_path)
    evidence = _evidence(tmp_path, config, promotion)
    sealed = json.loads(evidence[0].evaluation_report_path.read_text(encoding="utf-8"))
    promotion_path = sealed["lineage"]["promotion_report_path"]
    promotion_file = Path(promotion_path)
    payload = json.loads(promotion_file.read_text(encoding="utf-8"))
    payload["suite"]["matchups"][0]["bb_per_100"] += 1.0
    promotion_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="promotion report"):
        aggregate_campaign(config, evidence)


def test_campaign_rejects_single_seed_as_selection_evidence(tmp_path) -> None:
    config, _ = _campaign(tmp_path)
    one_seed = replace(config.sweep, seeds=(11,))
    with pytest.raises(ValueError, match="more seeds"):
        CampaignConfig(one_seed, minimum_seeds_per_variant=2)


def test_protocol_pins_evaluation_rng_independently_of_training_seed(tmp_path) -> None:
    config, _ = _campaign(tmp_path, baseline="random")
    protocol = json.loads(Path(config.sweep.evaluation_protocol_path).read_text(encoding="utf-8"))
    assert protocol["promotion_protocol"]["evaluation_run_seed"] == 0


def test_campaign_cli_runs_resumes_evaluates_seals_and_aggregates(tmp_path) -> None:
    config, _ = _campaign(tmp_path)
    sweep_path = write_sweep_config(config.sweep, tmp_path / "sweep-config.json")
    config_path = tmp_path / "campaign-config.json"
    assert campaign_main([
        "init", "--sweep-config", str(sweep_path), "--output", str(config_path),
        "--minimum-seeds", "2", "--minimum-baseline-ci95-low", "-1000000000", "--maximum-ece", "1",
    ]) == 0
    trials = tmp_path / "trials"
    runs = tmp_path / "runs"
    evidence = tmp_path / "evidence"

    assert campaign_main([
        "run", "--config", str(config_path), "--trials-dir", str(trials), "--runs-dir", str(runs),
    ]) == 0
    assert campaign_main([
        "run", "--config", str(config_path), "--trials-dir", str(trials), "--runs-dir", str(runs),
    ]) == 0
    assert campaign_main(["status", "--config", str(config_path), "--runs-dir", str(runs)]) == 0
    assert campaign_main([
        "evaluate-seal", "--config", str(config_path), "--trials-dir", str(trials),
        "--runs-dir", str(runs), "--evidence-dir", str(evidence),
    ]) == 0
    assert campaign_main([
        "evaluate-seal", "--config", str(config_path), "--trials-dir", str(trials),
        "--runs-dir", str(runs), "--evidence-dir", str(evidence),
    ]) == 0
    reports = sorted(evidence.glob("*/sealed.json"))
    output = tmp_path / "campaign-report.json"
    arguments = ["aggregate", "--config", str(config_path), "--output", str(output)]
    for report in reports:
        arguments.extend(("--evidence", str(report)))
    assert campaign_main(arguments) == 0
    assert campaign_main(["verify", "--config", str(config_path), "--report", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["winner_variant_id"] is not None
