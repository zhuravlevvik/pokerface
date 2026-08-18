"""End-to-end contracts for the durable Stage 1 pretraining runner."""

from __future__ import annotations

import json
import signal

import pytest

from poker.curriculum import CurriculumStage
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.pretraining import PretrainingConfig
from poker.pretraining_runner import (
    AcceptanceConfig,
    CorpusConfig,
    PretrainingRunConfig,
    PretrainingRunner,
    load_pretraining_run_config,
    write_pretraining_run_config,
)

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _config(*, epochs: int = 1) -> PretrainingRunConfig:
    return PretrainingRunConfig(
        corpus=CorpusConfig(
            train_seed_start=100,
            train_hands=2,
            holdout_seed_start=10_000,
            holdout_hands=2,
            train_equity_samples=1,
            holdout_equity_samples=1,
        ),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        training=PretrainingConfig(epochs=epochs, batch_size=64, learning_rate=1e-3, seed=77, calibration_bins=4),
        acceptance=AcceptanceConfig(
            maximum_ece=1.0,
            maximum_scalar_mae=1.0,
            minimum_brier_reduction=-10.0,
            minimum_stratum_samples=1,
            minimum_supported_strata=1,
        ),
    )


def test_config_round_trip_and_multiway_execution_guard(tmp_path) -> None:
    path = tmp_path / "pretrain.json"
    write_pretraining_run_config(_config(), path)
    assert load_pretraining_run_config(path) == _config()
    with pytest.raises(ValueError, match="heads-up"):
        CorpusConfig(stage=CurriculumStage.C_THREE_MAX)


def test_run_writes_restricted_corpus_report_resumable_checkpoint_and_inference_weights(tmp_path) -> None:
    runner = PretrainingRunner(_config(), tmp_path / "run")
    result = runner.run(install_signal_handlers=False)

    assert not result.interrupted
    assert result.epoch == 1
    assert result.checkpoint_path.exists() and runner.latest_path.exists()
    assert result.report_path.exists() and runner.corpus_path.exists()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["target_semantics"].startswith("fixed-deal")
    assert report["scalar_equity_semantics"].endswith("(heads-up only)")
    assert {"logloss", "brier_score", "expected_calibration_error"} <= set(report["aggregate"]["model"])
    assert {"mae", "rmse"} == set(report["aggregate"]["heads_up_scalar"])
    assert report["strata"]
    payload = torch.load(runner.latest_path, map_location="cpu", weights_only=True)
    assert payload["provenance"]["corpus_sha256"] == report["corpus_sha256"]
    loaded = PokerAgentModel.load_checkpoint(runner.latest_path)
    assert loaded.checkpoint_metadata() == runner.pretrainer.model.checkpoint_metadata()
    assert not list(runner.checkpoint_directory.glob(".*.tmp"))


def test_resume_matches_uninterrupted_epoch_stream_and_rejects_corpus_drift(tmp_path) -> None:
    config = _config(epochs=2)
    uninterrupted = PretrainingRunner(config, tmp_path / "uninterrupted")
    uninterrupted.run(install_signal_handlers=False)

    split = PretrainingRunner(config, tmp_path / "split")
    split.run(until_epoch=1, install_signal_handlers=False)
    resumed = PretrainingRunner.resume(split.latest_path)
    resumed.run(until_epoch=2, install_signal_handlers=False)
    for actual, expected in zip(resumed.pretrainer.model.state_dict().values(), uninterrupted.pretrainer.model.state_dict().values(), strict=True):
        assert torch.equal(actual, expected)

    lines = split.corpus_path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(json.loads(lines[0]))
    split.corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        PretrainingRunner.resume(split.latest_path)


def test_first_sigint_finishes_epoch_and_second_interrupts_immediately(tmp_path, monkeypatch) -> None:
    graceful = PretrainingRunner(_config(epochs=2), tmp_path / "graceful")
    original = graceful.pretrainer.train_epoch

    def one_interrupt(examples):
        signal.raise_signal(signal.SIGINT)
        return original(examples)

    monkeypatch.setattr(graceful.pretrainer, "train_epoch", one_interrupt)
    result = graceful.run(install_signal_handlers=True)
    assert result.interrupted and result.epoch == 1
    assert PretrainingRunner.resume(graceful.latest_path).pretrainer.epoch == 1

    immediate = PretrainingRunner(_config(epochs=2), tmp_path / "immediate")

    def two_interrupts(_examples):
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("second SIGINT must interrupt immediately")

    monkeypatch.setattr(immediate.pretrainer, "train_epoch", two_interrupts)
    with pytest.raises(KeyboardInterrupt):
        immediate.run(install_signal_handlers=True)
    assert not immediate.latest_path.exists()
