"""Contracts for durable PPO run state and batched current-policy inference."""

from __future__ import annotations

import json
import signal

import pytest

from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.game_state import HandState
from poker.league import default_league
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.observation import observation_for
from poker.train_runner import RunSettings, TrainingRunConfig, TrainingRunner
from poker.training import PPOConfig, PPOTrainer, UpdateMetrics

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _config(*, iterations: int = 2) -> TrainingRunConfig:
    return TrainingRunConfig(
        run=RunSettings(
            seed=43,
            iterations=iterations,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_iterations=1,
            checkpoint_every_seconds=None,
        ),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        ppo=PPOConfig(epochs=1, minibatch_size=8, equity_samples=1, learning_rate=1e-3),
        curriculum=CurriculumConfig(
            base_learning_rate=1e-3,
            require_transfer_beats_scratch=False,
            require_previous_checkpoint_win=False,
        ),
    )


def test_current_policy_batch_sampler_uses_one_forward_for_many_observations(monkeypatch) -> None:
    torch.manual_seed(7)
    model = PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))
    trainer = PPOTrainer(model, default_league(model, seed=3), PPOConfig(equity_samples=1))
    state = HandState(seed=7, player_count=2)
    observation = observation_for(state, state.actor)  # type: ignore[arg-type]
    calls: list[int] = []
    original_forward = model.forward

    def tracked_forward(observations):
        calls.append(len(observations))
        return original_forward(observations)

    monkeypatch.setattr(model, "forward", tracked_forward)
    decisions = trainer._select_current_batch((observation, observation))
    assert calls == [2]
    assert len(decisions) == 2
    assert all(observation["legal_actions"][decision[0].value] for decision in decisions)


def test_checkpoint_round_trip_is_atomic_and_model_loader_accepts_full_run(tmp_path) -> None:
    runner = TrainingRunner(_config(iterations=1), tmp_path / "run")
    result = runner.run(install_signal_handlers=False)
    assert not result.interrupted
    assert result.checkpoint_path.exists()
    assert runner.latest_path.exists()
    assert not list(runner.checkpoint_directory.glob(".*.tmp"))

    restored = TrainingRunner.resume(runner.latest_path)
    assert restored.iteration == runner.iteration == 1
    assert restored.global_hands == runner.global_hands == 1
    assert restored.global_decisions == runner.global_decisions
    assert restored.trainer._seed_counter == runner.trainer._seed_counter
    assert all(torch.equal(left, right) for left, right in zip(restored.model.state_dict().values(), runner.model.state_dict().values(), strict=True))
    # An inference/UI process only needs the normal model part of a full run.
    model_only = PokerAgentModel.load_checkpoint(runner.latest_path)
    assert model_only.checkpoint_metadata() == runner.model.checkpoint_metadata()
    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    assert manifest["latest"] == str(runner.latest_path)
    assert manifest["checkpoints"][-1]["path"] == str(result.checkpoint_path)


def test_requested_graceful_stop_writes_interrupt_checkpoint_at_safe_boundary(tmp_path) -> None:
    runner = TrainingRunner(_config(iterations=3), tmp_path / "run")
    runner.request_stop()
    result = runner.run(install_signal_handlers=False)
    assert result.interrupted
    assert result.iteration == 1
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["reason"] == "interrupt"
    assert TrainingRunner.resume(runner.latest_path).iteration == 1


def test_real_first_and_second_sigint_follow_graceful_then_immediate_semantics(tmp_path, monkeypatch) -> None:
    metrics = UpdateMetrics(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    graceful = TrainingRunner(_config(iterations=2), tmp_path / "graceful")

    def one_interrupt():
        graceful.iteration += 1
        signal.raise_signal(signal.SIGINT)
        return metrics

    monkeypatch.setattr(graceful, "_train_one_iteration", one_interrupt)
    result = graceful.run(install_signal_handlers=True)
    assert result.interrupted and result.checkpoint_path.exists()

    immediate = TrainingRunner(_config(iterations=2), tmp_path / "immediate")

    def two_interrupts():
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("the second SIGINT must interrupt immediately")

    monkeypatch.setattr(immediate, "_train_one_iteration", two_interrupts)
    with pytest.raises(KeyboardInterrupt):
        immediate.run(install_signal_handlers=True)
    assert not immediate.latest_path.exists()


def test_resume_matches_uninterrupted_training_stream(tmp_path) -> None:
    config = _config(iterations=2)
    uninterrupted = TrainingRunner(config, tmp_path / "uninterrupted")
    uninterrupted.run(install_signal_handlers=False)

    split = TrainingRunner(config, tmp_path / "split")
    split.run(until_iteration=1, install_signal_handlers=False)
    resumed = TrainingRunner.resume(split.latest_path)
    resumed.run(until_iteration=2, install_signal_handlers=False)

    assert resumed.global_hands == uninterrupted.global_hands
    assert resumed.global_decisions == uninterrupted.global_decisions
    assert resumed.trainer._seed_counter == uninterrupted.trainer._seed_counter
    assert all(torch.equal(left, right) for left, right in zip(resumed.model.state_dict().values(), uninterrupted.model.state_dict().values(), strict=True))


@pytest.mark.parametrize("stage", list(CurriculumStage))
def test_runner_applies_curriculum_learning_rate_scale(stage, tmp_path) -> None:
    config = _config(iterations=0)
    config = TrainingRunConfig(
        run=RunSettings(
            stage=stage,
            seed=config.run.seed,
            iterations=0,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_seconds=None,
        ),
        model=config.model,
        ppo=config.ppo,
        curriculum=config.curriculum,
        league=config.league,
    )
    runner = TrainingRunner(config, tmp_path / stage.value)
    expected = config.curriculum.learning_rate_for(stage)
    assert runner.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected)
