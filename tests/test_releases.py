"""Contracts for append-only, hash-verified PPO release records."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.experiment_runner import ExperimentRunner
from poker.experiments import ExperimentConfig, ExperimentLedger
from poker.model import ModelConfig, TORCH_AVAILABLE
from poker.promotion import PromotionConfig, PromotionEvaluator
from poker.releases import LineageArtifact, ReleaseRegistry, ReleaseRequest
from poker.release_cli import main as release_main
from poker.train_runner import RunSettings, TrainingRunConfig
from poker.training import PPOConfig
from poker.tuning import SweepConfig, publish_tuning_evaluation, write_hu_promotion_protocol


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")


def _sha(path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _protocol(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    promotion = PromotionConfig(
        enabled=True,
        every_iterations=1,
        hands_per_opponent=2,
        equity_samples=1,
        calibration_bins=2,
        baseline_bots=("rule",),
        historical_limit=0,
        minimum_baseline_bb_per_100=-1e9,
        minimum_baseline_ci95_low=-1e9,
        maximum_baseline_ci95_half_width=1e9,
        minimum_historical_league_score=0.0,
        minimum_historical_ci95_low=-1e9,
        maximum_equity_ece=1.0,
    )
    return (*write_hu_promotion_protocol(_training(), promotion, tmp_path / "protocol.json"), promotion)


def _training() -> TrainingRunConfig:
    ppo = PPOConfig(learning_rate=1e-3, epochs=1, minibatch_size=8, equity_samples=1)
    return TrainingRunConfig(
        run=RunSettings(
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            seed=9001,
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


def _artifacts(tmp_path):
    config = _training()
    protocol_path, protocol_sha, promotion_config = _protocol(tmp_path)
    sweep = SweepConfig(
        base_config=config,
        grid={},
        seeds=(config.run.seed,),
        max_iterations=1,
        evaluation_protocol_sha256=protocol_sha,
        evaluation_protocol_path=str(protocol_path),
        code_revision="release-test-revision",
        name="release-sweep",
    )
    trial = sweep.expand_trials()[0]
    # This tuning spec has the same resolved native TrainingRunConfig as the
    # completed experiment; it only supplies the pinned report envelope.
    assert trial.config.to_dict() == config.to_dict()
    experiment = ExperimentConfig(
        name=trial.trial_id,
        training=config,
        max_iterations=1,
        evaluation_protocol_path=str(protocol_path),
        evaluation_protocol_sha256=protocol_sha,
        code_revision="release-test-revision",
    )
    experiment_runner = ExperimentRunner(experiment, tmp_path / "trial")
    run = experiment_runner.run(install_signal_handlers=False)
    assert run.status == "completed" and run.checkpoint_path is not None
    promotion = PromotionEvaluator(
        promotion_config,
        tmp_path / "promotion",
    )
    promoted = promotion.evaluate_and_promote(
        iteration=1,
        candidate_checkpoint=run.checkpoint_path,
        league=experiment_runner.trainer.league,
        stage=config.run.stage,
        champion_score=None,
        run_context={
            "run_config_sha256": trial.run_config_sha256,
            "evaluation_protocol_sha256": protocol_sha,
        },
    )
    assert promoted.accepted
    evidence = publish_tuning_evaluation(
        trial,
        run.checkpoint_path,
        run.ledger_manifest_path,
        promoted.report_path,
        promotion.archive_manifest_path,
        tmp_path / "evaluation.json",
    )
    return experiment, run.checkpoint_path, run.ledger_manifest_path, evidence.evaluation_report_path


def _request(tmp_path, release_id: str = "release-1") -> ReleaseRequest:
    _experiment, checkpoint, ledger, report = _artifacts(tmp_path)
    lineage = tmp_path / "lineage.json"
    lineage.write_text('{"source":"stage6"}\n', encoding="utf-8")
    return ReleaseRequest(
        release_id=release_id,
        code_revision="release-test-revision",
        full_checkpoint_path=checkpoint,
        full_checkpoint_sha256=_sha(checkpoint),
        experiment_ledger_manifest_path=ledger,
        experiment_ledger_manifest_sha256=_sha(ledger),
        tuning_evaluation_report_path=report,
        tuning_evaluation_report_sha256=_sha(report),
        extra_lineage_artifacts=(LineageArtifact("lineage", lineage, _sha(lineage)),),
    )


def test_release_registers_complete_exact_lineage_and_lists_verifies(tmp_path) -> None:
    request = _request(tmp_path)
    registry = ReleaseRegistry(tmp_path / "registry")

    record = registry.register(request)

    assert record.release_id == request.release_id and record.release_path.is_file()
    assert registry.show(request.release_id) == record
    assert registry.list() == (record,)
    assert registry.verify(request.release_id) == record
    payload = json.loads(record.release_path.read_text(encoding="utf-8"))
    assert payload["verified"]["checkpoint"]["sha256"] == request.full_checkpoint_sha256
    assert payload["verified"]["experiment_ledger"]["final_checkpoint_sha256"] == request.full_checkpoint_sha256
    assert payload["verified"]["tuning_evaluation"]["evaluation_protocol_sha256"] == _sha(tmp_path / "protocol.json")


def test_release_registration_is_idempotent_and_rejects_divergent_duplicate(tmp_path) -> None:
    request = _request(tmp_path)
    registry = ReleaseRegistry(tmp_path / "registry")
    first = registry.register(request)
    second = registry.register(request)

    assert first == second
    divergent = ReleaseRequest(
        release_id=request.release_id,
        code_revision="other-revision",
        full_checkpoint_path=request.full_checkpoint_path,
        full_checkpoint_sha256=request.full_checkpoint_sha256,
        experiment_ledger_manifest_path=request.experiment_ledger_manifest_path,
        experiment_ledger_manifest_sha256=request.experiment_ledger_manifest_sha256,
        tuning_evaluation_report_path=request.tuning_evaluation_report_path,
        tuning_evaluation_report_sha256=request.tuning_evaluation_report_sha256,
    )
    with pytest.raises(ValueError, match="code_revision"):
        registry.register(divergent)


def test_release_recovers_exact_orphan_record_before_manifest_append(tmp_path) -> None:
    request = _request(tmp_path)
    registry = ReleaseRegistry(tmp_path / "registry")
    from poker.releases import _atomic_write_json

    _atomic_write_json(registry.releases_directory / f"{request.release_id}.json", registry._release_payload(request))
    # Recovery occurs in a fresh process instance, not only in the process
    # that managed to publish the immutable record before crashing.
    recovered = ReleaseRegistry(tmp_path / "registry").register(request)

    assert recovered.release_id == request.release_id
    assert ReleaseRegistry(tmp_path / "registry").list() == (recovered,)


def test_release_verify_fails_closed_on_tampered_record_or_artifact(tmp_path) -> None:
    request = _request(tmp_path)
    registry = ReleaseRegistry(tmp_path / "registry")
    record = registry.register(request)
    record.release_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash-mismatched"):
        registry.verify(request.release_id)

    # A separate registry demonstrates that external artifact hashes are
    # checked again, not merely copied into the release record.
    request = _request(tmp_path / "external")
    registry = ReleaseRegistry(tmp_path / "external-registry")
    registry.register(request)
    request.extra_lineage_artifacts[0].path.write_text('{"source":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="extra lineage artifact"):
        registry.verify(request.release_id)


def test_release_rejects_incomplete_ledger_and_checkpoint_regression(tmp_path) -> None:
    request = _request(tmp_path)
    original_ledger = json.loads(request.experiment_ledger_manifest_path.read_text(encoding="utf-8"))
    original_config = original_ledger["config"]
    incomplete_experiment = ExperimentConfig(
        name="release-trial",
        training=_training(),
        max_iterations=1,
        evaluation_protocol_path=original_config["evaluation_protocol_path"],
        evaluation_protocol_sha256=original_config["evaluation_protocol_sha256"],
        code_revision="release-test-revision",
    )
    incomplete_ledger = ExperimentLedger(tmp_path / "incomplete-trial", incomplete_experiment)
    incomplete = ReleaseRequest(
        release_id="incomplete",
        code_revision=request.code_revision,
        full_checkpoint_path=request.full_checkpoint_path,
        full_checkpoint_sha256=request.full_checkpoint_sha256,
        experiment_ledger_manifest_path=incomplete_ledger.manifest_path,
        experiment_ledger_manifest_sha256=_sha(incomplete_ledger.manifest_path),
        tuning_evaluation_report_path=request.tuning_evaluation_report_path,
        tuning_evaluation_report_sha256=request.tuning_evaluation_report_sha256,
    )
    with pytest.raises(ValueError, match="incomplete"):
        ReleaseRegistry(tmp_path / "registry").register(incomplete)

    # A fresh set of valid evidence with a report regressed to a different
    # checkpoint hash must fail the exact checkpoint binding check.
    request = _request(tmp_path / "regression")
    report = json.loads(request.tuning_evaluation_report_path.read_text(encoding="utf-8"))
    report["full_checkpoint_sha256"] = "0" * 64
    request.tuning_evaluation_report_path.write_text(json.dumps(report), encoding="utf-8")
    regressed = ReleaseRequest(
        release_id="regressed",
        code_revision=request.code_revision,
        full_checkpoint_path=request.full_checkpoint_path,
        full_checkpoint_sha256=request.full_checkpoint_sha256,
        experiment_ledger_manifest_path=request.experiment_ledger_manifest_path,
        experiment_ledger_manifest_sha256=request.experiment_ledger_manifest_sha256,
        tuning_evaluation_report_path=request.tuning_evaluation_report_path,
        tuning_evaluation_report_sha256=_sha(request.tuning_evaluation_report_path),
    )
    with pytest.raises(ValueError, match="exact checkpoint/protocol"):
        ReleaseRegistry(tmp_path / "regression-registry").register(regressed)


def test_release_cli_register_list_show_and_verify(tmp_path, capsys) -> None:
    request = _request(tmp_path)
    registry = tmp_path / "registry"
    register = [
        "register", "--registry", str(registry), "--release-id", request.release_id,
        "--code-revision", request.code_revision, "--checkpoint", str(request.full_checkpoint_path),
        "--checkpoint-sha256", request.full_checkpoint_sha256,
        "--ledger-manifest", str(request.experiment_ledger_manifest_path),
        "--ledger-manifest-sha256", request.experiment_ledger_manifest_sha256,
        "--tuning-report", str(request.tuning_evaluation_report_path),
        "--tuning-report-sha256", request.tuning_evaluation_report_sha256,
    ]
    assert release_main(register) == 0
    assert json.loads(capsys.readouterr().out)["release_id"] == request.release_id
    assert release_main(["list", "--registry", str(registry)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert release_main(["show", "--registry", str(registry), "--release-id", request.release_id]) == 0
    assert release_main(["verify", "--registry", str(registry), "--release-id", request.release_id]) == 0
