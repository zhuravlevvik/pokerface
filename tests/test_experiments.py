"""Durability contracts for the bounded Stage 6 experiment ledger."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.experiment_runner import ExperimentRunner, load_experiment_config, write_experiment_config
from poker.experiment_cli import main as experiment_main
from poker.experiments import ExperimentConfig, ExperimentLedger
from poker.model import ModelConfig, TORCH_AVAILABLE
from poker.train_runner import RunSettings, TrainingRunConfig, TrainingRunner
from poker.training import NonFiniteTrainingError, PPOConfig


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


_PROTOCOL_PATH = Path(__file__).resolve()
_PROTOCOL_SHA = sha256(_PROTOCOL_PATH.read_bytes()).hexdigest()


def _training_config(*, iterations: int = 2) -> TrainingRunConfig:
    ppo = PPOConfig(learning_rate=1e-3, epochs=1, minibatch_size=8, equity_samples=1)
    return TrainingRunConfig(
        run=RunSettings(
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            seed=811,
            iterations=iterations,
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


def _experiment(*, name: str = "pilot", iterations: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        training=_training_config(iterations=iterations),
        max_iterations=iterations,
        evaluation_protocol_path=str(_PROTOCOL_PATH),
        evaluation_protocol_sha256=_PROTOCOL_SHA,
        code_revision="test-revision-1",
    )


def _checkpoint_series(tmp_path, count: int = 2):
    runner = TrainingRunner(_training_config(iterations=count), tmp_path / "training")
    result = []
    for _ in range(count):
        metrics = runner._train_one_iteration()
        result.append(runner.save_checkpoint(reason="experiment", metrics=metrics))
    return result


def test_ledger_records_idempotent_checkpoint_and_regenerates_player_safe_jsonl(tmp_path) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    ledger = ExperimentLedger(tmp_path / "trial", _experiment(iterations=1))

    first = ledger.record_checkpoint(checkpoint)
    second = ledger.record_checkpoint(checkpoint)

    assert first == second
    assert first.iteration == first.global_hands == 1
    row = json.loads(ledger.metrics_path.read_text(encoding="utf-8"))
    assert row["iteration"] == 1
    assert "observation" not in row and "rollout" not in row and "hole_cards" not in row
    assert ledger.write_status("running").is_file()
    assert ledger.record_failure("plateau", "fixed evaluation did not improve").is_file()


def test_ledger_rejects_config_mismatch_and_divergent_duplicate_iteration(tmp_path) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    directory = tmp_path / "trial"
    ledger = ExperimentLedger(directory, _experiment(iterations=1))
    ledger.record_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="immutable trial config"):
        ExperimentLedger(directory, _experiment(name="different", iterations=1))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for tensor in payload["state_dict"].values():
        if tensor.dtype.is_floating_point:
            tensor.add_(1.0)
            break
    divergent = tmp_path / "divergent.pt"
    torch.save(payload, divergent)
    with pytest.raises(ValueError, match="divergent checkpoint"):
        ledger.record_checkpoint(divergent)


def test_ledger_fails_closed_on_tampered_event_or_checkpoint(tmp_path) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    directory = tmp_path / "trial"
    ledger = ExperimentLedger(directory, _experiment(iterations=1))
    event = ledger.record_checkpoint(checkpoint)
    event.event_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash-mismatched"):
        ExperimentLedger(directory, _experiment(iterations=1))


def test_ledger_fails_closed_on_tampered_recorded_checkpoint(tmp_path) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    directory = tmp_path / "trial"
    ledger = ExperimentLedger(directory, _experiment(iterations=1))
    ledger.record_checkpoint(checkpoint)
    checkpoint.write_bytes(b"not a checkpoint anymore")

    with pytest.raises(ValueError, match="checkpoint/provenance"):
        ExperimentLedger(directory, _experiment(iterations=1))


def test_ledger_rejects_nonfinite_update_metrics(tmp_path) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["metrics"]["total_loss"] = float("nan")
    invalid = tmp_path / "nonfinite.pt"
    torch.save(payload, invalid)

    ledger = ExperimentLedger(tmp_path / "trial", _experiment(iterations=1))
    with pytest.raises(ValueError, match="must be finite"):
        ledger.record_checkpoint(invalid)


def test_ledger_hash_chain_is_continuous_and_recover_latest_is_idempotent(tmp_path) -> None:
    first_checkpoint, second_checkpoint = _checkpoint_series(tmp_path, 2)
    ledger = ExperimentLedger(tmp_path / "trial", _experiment(iterations=2))
    first = ledger.record_checkpoint(first_checkpoint)
    second = ledger.recover_latest(second_checkpoint)
    repeated = ledger.recover_latest(second_checkpoint)

    assert second is not None and repeated == second
    assert second.previous_event_sha256 == first.event_sha256
    manifest = json.loads(ledger.manifest_path.read_text(encoding="utf-8"))
    assert [record["iteration"] for record in manifest["events"]] == [1, 2]
    assert len(ledger.metrics_path.read_text(encoding="utf-8").splitlines()) == 2


def test_ledger_recovers_atomic_event_published_before_manifest(tmp_path) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    ledger = ExperimentLedger(tmp_path / "trial", _experiment(iterations=1))
    event = ledger._checkpoint_event_data(checkpoint)
    path = ledger.events_directory / "00000001.json"
    payload = {
        "version": 1,
        "trial_id": ledger.config.trial_id,
        "config_sha256": ledger.config.config_sha256,
        "iteration": 1,
        "checkpoint": {
            "path": event["checkpoint_path"],
            "sha256": event["checkpoint_sha256"],
            "checkpoint_version": 1,
            "run_config_sha256": event["run_config_sha256"],
        },
        "progress": {"global_hands": event["global_hands"], "global_decisions": event["global_decisions"]},
        "metrics": event["metrics"],
        "previous_event_sha256": None,
    }
    # Simulate a crash after the atomic event write but before the manifest.
    from poker.experiments import _atomic_write_json

    _atomic_write_json(path, payload)

    recovered = ExperimentLedger(tmp_path / "trial", _experiment(iterations=1))
    assert recovered.last_iteration == 1


def test_ledger_retries_manifest_write_failure_in_same_process(tmp_path, monkeypatch) -> None:
    checkpoint = _checkpoint_series(tmp_path, 1)[0]
    ledger = ExperimentLedger(tmp_path / "trial", _experiment(iterations=1))
    original = ledger._write_manifest
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated manifest publication failure")
        original()

    monkeypatch.setattr(ledger, "_write_manifest", fail_once)
    with pytest.raises(OSError, match="publication failure"):
        ledger.record_checkpoint(checkpoint)
    assert ledger.last_iteration == 0

    recovered = ledger.record_checkpoint(checkpoint)
    assert recovered.iteration == ledger.last_iteration == 1


def test_experiment_config_roundtrip_and_runner_resume_are_idempotent(tmp_path) -> None:
    config = _experiment(iterations=2)
    path = write_experiment_config(config, tmp_path / "experiment.json")
    assert load_experiment_config(path) == config

    first = ExperimentRunner(config, tmp_path / "trial")
    paused = first.run(until_iteration=1, install_signal_handlers=False)
    assert paused.status == "paused"
    assert paused.iteration == 1

    resumed = ExperimentRunner(config, tmp_path / "trial")
    completed = resumed.run(install_signal_handlers=False)
    assert completed.status == "completed"
    assert completed.iteration == 2
    rows = [json.loads(line) for line in completed.metrics_path.read_text(encoding="utf-8").splitlines()]
    assert [row["iteration"] for row in rows] == [1, 2]
    assert len({row["checkpoint_sha256"] for row in rows}) == 2


def test_experiment_runner_recovers_checkpoint_published_before_observer(tmp_path) -> None:
    config = _experiment(iterations=1)
    native = TrainingRunner(config.training, tmp_path / "trial" / "training")
    native.run(install_signal_handlers=False)

    recovered = ExperimentRunner(config, tmp_path / "trial")

    assert recovered.ledger.last_iteration == 1
    assert len(recovered.ledger.metrics_path.read_text(encoding="utf-8").splitlines()) == 1


def test_experiment_runner_marks_nonfinite_failure_without_publishing(tmp_path, monkeypatch) -> None:
    runner = ExperimentRunner(_experiment(iterations=1), tmp_path / "trial")

    def fail_update():
        raise NonFiniteTrainingError("PPO total_loss is non-finite")

    monkeypatch.setattr(runner.trainer, "_train_one_iteration", fail_update)
    result = runner.run(install_signal_handlers=False)

    assert result.status == "failed_nonfinite"
    assert result.iteration == 0
    assert result.failure_path is not None and result.failure_path.is_file()
    assert not runner.trainer.latest_path.exists()
    failure = json.loads(result.failure_path.read_text(encoding="utf-8"))
    assert failure["kind"] == "nonfinite_training"


def test_experiment_cli_writes_reviewable_starter_config(tmp_path) -> None:
    destination = tmp_path / "experiment.json"

    assert experiment_main([
        "--write-default-config", str(destination),
        "--code-revision", "test-revision-1",
        "--protocol-artifact", str(tmp_path / "protocol.json"),
    ]) == 0
    config = load_experiment_config(destination)
    assert config.training.run.checkpoint_every_iterations == 1
    assert config.max_iterations == config.training.run.iterations


def test_experiment_recovers_latest_published_before_native_manifest(tmp_path, monkeypatch) -> None:
    import poker.train_runner as train_runner_module

    config = _experiment(iterations=1)
    runner = ExperimentRunner(config, tmp_path / "trial")
    metrics = runner.trainer._train_one_iteration()
    original = train_runner_module._atomic_write_text

    def crash_on_manifest(path, contents):
        if path == runner.trainer.manifest_path:
            raise OSError("simulated crash before native manifest")
        return original(path, contents)

    monkeypatch.setattr(train_runner_module, "_atomic_write_text", crash_on_manifest)
    with pytest.raises(OSError, match="native manifest"):
        runner.trainer.save_checkpoint(metrics=metrics)
    assert runner.trainer.latest_path.is_file()
    monkeypatch.setattr(train_runner_module, "_atomic_write_text", original)

    recovered = ExperimentRunner(config, tmp_path / "trial")
    assert recovered.trainer.iteration == recovered.ledger.last_iteration == 1


def test_experiment_rejects_latest_rolled_back_behind_ledger(tmp_path) -> None:
    config = _experiment(iterations=1)
    runner = ExperimentRunner(config, tmp_path / "trial")
    baseline = runner.trainer.save_checkpoint(reason="baseline").read_bytes()
    runner.run(install_signal_handlers=False)
    runner.trainer.latest_path.write_bytes(baseline)

    with pytest.raises(ValueError, match="predates"):
        ExperimentRunner(config, tmp_path / "trial")


def test_experiment_marks_immediate_second_interrupt_as_aborted(tmp_path, monkeypatch) -> None:
    runner = ExperimentRunner(_experiment(iterations=1), tmp_path / "trial")
    monkeypatch.setattr(runner.trainer, "run", lambda **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))

    result = runner.run(install_signal_handlers=False)

    assert result.status == "aborted_immediate"
    status = json.loads(runner.ledger.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "aborted_immediate"
