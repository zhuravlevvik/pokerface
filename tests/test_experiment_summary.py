"""Contracts for the read-only, player-safe experiment health summary."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.experiment_runner import ExperimentRunner
from poker.experiment_summary import (
    ExperimentHealthConfig,
    summarize_experiment,
    write_experiment_summary,
)
from poker.experiments import ExperimentConfig
from poker.model import ModelConfig, TORCH_AVAILABLE
from poker.train_runner import RunSettings, TrainingRunConfig
from poker.training import PPOConfig


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")


_PROTOCOL_PATH = Path(__file__).resolve()
_PROTOCOL_SHA = sha256(_PROTOCOL_PATH.read_bytes()).hexdigest()


def _config(*, iterations: int = 2) -> ExperimentConfig:
    ppo = PPOConfig(learning_rate=1e-3, epochs=1, minibatch_size=8, equity_samples=1)
    training = TrainingRunConfig(
        run=RunSettings(
            stage=CurriculumStage.A_HEADS_UP_STARTER,
            seed=917,
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
    return ExperimentConfig(
        name="summary-pilot",
        training=training,
        max_iterations=iterations,
        evaluation_protocol_path=str(_PROTOCOL_PATH),
        evaluation_protocol_sha256=_PROTOCOL_SHA,
        code_revision="summary-test-revision",
    )


def test_completed_summary_uses_verified_event_chain_not_metrics_projection(tmp_path) -> None:
    run_directory = tmp_path / "trial"
    result = ExperimentRunner(_config(), run_directory).run(install_signal_handlers=False)
    assert result.status == "completed"
    result.metrics_path.write_text('{"forged":true}\n', encoding="utf-8")

    summary = summarize_experiment(run_directory)

    assert summary.status == "completed"
    assert summary.completed
    assert summary.trial_id == _config().trial_id
    assert summary.config_sha256 == _config().config_sha256
    assert summary.evaluation_protocol_sha256 == _PROTOCOL_SHA
    assert summary.code_revision == "summary-test-revision"
    assert summary.iteration_range is not None
    assert summary.iteration_range.first == 1 and summary.iteration_range.last == 2
    assert summary.global_hands_range is not None and summary.global_hands_range.last == 2
    assert set(summary.metrics) == {
        "samples",
        "total_loss",
        "policy_loss",
        "value_loss",
        "equity_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "expected_showdown_share_loss",
        "gradient_norm",
    }
    # Summary construction is read-only: it neither trusts nor repairs the
    # mutable JSONL projection.
    assert result.metrics_path.read_text(encoding="utf-8") == '{"forged":true}\n'


def test_paused_summary_reports_terminal_ledger_boundary_without_claiming_completion(tmp_path) -> None:
    run_directory = tmp_path / "trial"
    result = ExperimentRunner(_config(), run_directory).run(until_iteration=1, install_signal_handlers=False)
    assert result.status == "paused"

    summary = summarize_experiment(run_directory)

    assert summary.status == "paused"
    assert not summary.completed
    assert summary.iteration_range is not None
    assert summary.iteration_range.first == summary.iteration_range.last == 1
    assert summary.global_hands_range is not None
    assert summary.global_hands_range.first == summary.global_hands_range.last == 1
    assert summary.global_decisions_range is not None
    assert summary.global_decisions_range.first == summary.global_decisions_range.last


def test_health_alerts_are_explicit_and_deterministically_ordered(tmp_path) -> None:
    run_directory = tmp_path / "trial"
    ExperimentRunner(_config(), run_directory).run(install_signal_handlers=False)
    health = ExperimentHealthConfig(
        max_abs_kl=-1.0,
        max_clip_fraction=-1.0,
        min_entropy=1e9,
        max_value_loss=-1.0,
        max_gradient_norm=-1.0,
    )

    first = summarize_experiment(run_directory, health=health)
    second = summarize_experiment(run_directory, health=health)

    assert first.alerts == second.alerts
    assert [(alert.iteration, alert.metric) for alert in first.alerts] == [
        (1, "approximate_kl"),
        (1, "clip_fraction"),
        (1, "entropy"),
        (1, "value_loss"),
        (1, "gradient_norm"),
        (2, "approximate_kl"),
        (2, "clip_fraction"),
        (2, "entropy"),
        (2, "value_loss"),
        (2, "gradient_norm"),
    ]
    assert all(alert.threshold != 0.0 for alert in first.alerts)
    assert not summarize_experiment(run_directory).alerts


def test_summary_fails_closed_on_tampered_ledger_event(tmp_path) -> None:
    run_directory = tmp_path / "trial"
    ExperimentRunner(_config(iterations=1), run_directory).run(install_signal_handlers=False)
    event = run_directory / "experiment-ledger" / "events" / "00000001.json"
    event.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash-mismatched"):
        summarize_experiment(run_directory)


def test_written_summary_is_player_safe_json(tmp_path) -> None:
    run_directory = tmp_path / "trial"
    ExperimentRunner(_config(iterations=1), run_directory).run(install_signal_handlers=False)

    destination = write_experiment_summary(summarize_experiment(run_directory), tmp_path / "health.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True).lower()

    assert payload["schema_version"] == 1
    assert payload["trial"]["id"] == _config(iterations=1).trial_id
    assert "observation" not in serialized
    assert "hole_cards" not in serialized
    assert "rollout" not in serialized
    assert "state_dict" not in serialized
    assert "optimizer_state" not in serialized
    assert "tensor" not in serialized
