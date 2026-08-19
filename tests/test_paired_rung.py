"""Contracts for the isolated transfer-versus-scratch PPO control rung."""

from __future__ import annotations

import json
import random

import pytest

from poker.model import ModelConfig, PokerAgentModel, TORCH_AVAILABLE
from poker.paired_rung import PairedRungConfig, PairedRungRunner
from poker.training import PPOConfig


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _source(path) -> None:
    model = PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))
    model.save_checkpoint(path)


def _config(*, iterations: int = 2, hands_per_iteration: int = 1, minibatch_size: int = 8) -> PairedRungConfig:
    return PairedRungConfig(
        iterations=iterations,
        hands_per_iteration=hands_per_iteration,
        table_count=1,
        base_seed=71,
        ppo=PPOConfig(learning_rate=1e-3, epochs=2, minibatch_size=minibatch_size, equity_samples=1),
    )


def _state_dict(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state_dict"]


def test_paired_rung_freezes_native_full_and_model_artifacts_with_matched_budget(tmp_path) -> None:
    source = tmp_path / "source.pt"
    _source(source)

    result = PairedRungRunner(_config(iterations=1), tmp_path / "run", source).run()

    assert result.completed
    assert result.transfer.iteration == result.scratch.iteration == 1
    assert result.transfer.global_hands == result.scratch.global_hands == 1
    assert result.transfer.global_decisions >= 1 and result.scratch.global_decisions >= 1
    for arm in (result.transfer, result.scratch):
        assert arm.full_checkpoint_path is not None and arm.full_checkpoint_path.exists()
        assert arm.model_checkpoint_path is not None and arm.model_checkpoint_path.exists()
        # Full arm artifacts are native TrainingRunner payloads and remain
        # usable for both runner resume and normal model inference.
        from poker.train_runner import TrainingRunner

        assert TrainingRunner.resume(arm.full_checkpoint_path).global_hands == 1
        payload = torch.load(arm.full_checkpoint_path, map_location="cpu", weights_only=True)
        assert {"optimizer_state_dict", "league", "rng", "run_config", "progress"}.issubset(payload)
        assert PokerAgentModel.load_checkpoint(arm.full_checkpoint_path).checkpoint_metadata()["config"] == {
            "embedding_dim": 8,
            "hidden_dim": 16,
            "history_layers": 1,
            "player_attention_layers": 1,
            "attention_heads": 2,
            "dropout": 0.0,
        }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == result.config_sha256
    assert manifest["source_checkpoint_sha256"] == result.source_checkpoint_sha256
    assert manifest["arms"]["transfer"]["global_hands"] == 1


def test_paired_rung_resume_is_idempotent_and_matches_uninterrupted_stream(tmp_path) -> None:
    source = tmp_path / "source.pt"
    _source(source)
    config = _config(iterations=2, hands_per_iteration=4, minibatch_size=2)

    split_root = tmp_path / "split"
    first = PairedRungRunner(config, split_root, source).run(until_iteration=1)
    resumed = PairedRungRunner(config, split_root, source).run()
    repeated = PairedRungRunner(config, split_root, source).run()
    control = PairedRungRunner(config, tmp_path / "control", source).run()

    assert not first.completed
    assert resumed.completed and repeated.completed
    assert resumed.transfer.full_checkpoint_sha256 == repeated.transfer.full_checkpoint_sha256
    assert resumed.scratch.model_checkpoint_sha256 == repeated.scratch.model_checkpoint_sha256
    for resumed_arm, control_arm in ((resumed.transfer, control.transfer), (resumed.scratch, control.scratch)):
        assert resumed_arm.global_hands == control_arm.global_hands == 8
        assert resumed_arm.global_decisions == control_arm.global_decisions
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                _state_dict(resumed_arm.full_checkpoint_path).values(),
                _state_dict(control_arm.full_checkpoint_path).values(),
                strict=True,
            )
        )


def test_paired_rung_preserves_caller_rng_and_can_stop_at_an_arm_boundary(tmp_path) -> None:
    source = tmp_path / "source.pt"
    _source(source)
    random.seed(991)
    torch.manual_seed(992)
    expected_python = random.getstate()
    expected_torch = torch.get_rng_state().clone()

    runner = PairedRungRunner(_config(iterations=2), tmp_path / "run", source)
    runner.request_stop()
    stopped = runner.run()

    assert not stopped.completed
    assert stopped.transfer.iteration == 1
    assert stopped.transfer.full_checkpoint_path is not None
    assert stopped.scratch.iteration == 0
    assert stopped.scratch.full_checkpoint_path is None
    assert random.getstate() == expected_python
    assert torch.equal(torch.get_rng_state(), expected_torch)

    # The same coordinator instance clears a handled stop request; a fresh
    # process can equivalently reconstruct from the same manifest/checkpoint.
    resumed = runner.run()
    assert resumed.completed
    assert resumed.transfer.global_hands == resumed.scratch.global_hands == 2


def test_paired_rung_config_round_trip_rejects_malformed_opponents() -> None:
    config = _config(iterations=1)
    assert PairedRungConfig.from_dict(config.to_dict()) == config
    malformed = config.to_dict()
    malformed["opponents"] = ["not-an-object"]
    with pytest.raises(ValueError, match="malformed paired-rung configuration"):
        PairedRungConfig.from_dict(malformed)


def test_paired_rung_rejects_tampered_frozen_artifact_instead_of_reblessing_it(tmp_path) -> None:
    source = tmp_path / "source.pt"
    _source(source)
    config = _config(iterations=1)
    result = PairedRungRunner(config, tmp_path / "run", source).run()
    assert result.transfer.full_checkpoint_path is not None
    result.transfer.full_checkpoint_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="frozen artifact failed hash validation"):
        PairedRungRunner(config, tmp_path / "run", source)
