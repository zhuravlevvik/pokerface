"""Executable contracts for league self-play and PPO training primitives."""

from __future__ import annotations

import pytest

from poker.betting import Action
from poker.league import CheckpointArchive, LeagueMember, ModelPolicy, OpponentLeague, default_league
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.training import (
    NonFiniteTrainingError,
    PPOConfig,
    PPOTrainer,
    RolloutStep,
    compute_gae,
    ensure_finite_model_parameters,
    ensure_finite_optimizer_state,
    factorized_logprob_and_entropy,
)

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _model() -> PokerAgentModel:
    torch.manual_seed(123)
    return PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))


def test_league_forces_current_policy_and_rotates_its_seat() -> None:
    model = _model()
    league = default_league(model, seed=3)
    current_seats = []
    for _ in range(10):
        seating = league.sample_seating(5)
        current_seats.append(next(index for index, member in enumerate(seating) if member.name == "current"))
    assert current_seats[:5] == list(range(5))


def test_league_alternates_current_policy_seat_in_heads_up() -> None:
    model = _model()
    league = default_league(model, seed=4)
    current_seats = [
        next(index for index, member in enumerate(league.sample_seating(2)) if member.name == "current")
        for _ in range(6)
    ]
    assert current_seats == [0, 1, 0, 1, 0, 1]


def test_gae_handles_sparse_terminal_pnl_per_player_trajectory() -> None:
    first = RolloutStep(1, 0, 0, {}, 0, -1, -0.1, 1.0)
    last = RolloutStep(1, 0, 3, {}, 1, -1, -0.2, 2.0, reward=5.0, terminal=True)
    compute_gae([first, last], gamma=1.0, gae_lambda=1.0)
    assert last.advantage == pytest.approx(3.0)
    assert first.advantage == pytest.approx(4.0)
    assert first.return_ == pytest.approx(5.0)


def test_factorized_raise_logprob_includes_size_term() -> None:
    model = _model()
    from poker.game_state import HandState
    from poker.observation import observation_for

    state = HandState(seed=9)
    observation = observation_for(state, state.actor)  # type: ignore[arg-type]
    output = model([observation])
    size = int(output.bet_size_mask[0].nonzero()[0].item())
    logprob, entropy = factorized_logprob_and_entropy(output, torch.tensor([3]), torch.tensor([size]))
    expected = torch.log(output.action_probabilities[0, 3]) + torch.log(output.bet_size_probabilities[0, size])
    assert torch.allclose(logprob, expected.unsqueeze(0))
    assert torch.isfinite(entropy).all()


def test_smoke_self_play_collects_labels_and_updates_all_heads() -> None:
    model = _model()
    trainer = PPOTrainer(
        model,
        default_league(model, seed=7),
        PPOConfig(epochs=1, minibatch_size=8, equity_samples=1, learning_rate=1e-3),
    )
    # Two heads-up hands make this test quick while using the real full legal
    # discrete action domain supplied by the engine.
    rollout, metrics = trainer.train_iteration(2, table_count=2, player_count=2)
    assert rollout.hands == 2
    assert rollout.decisions > 0
    assert all(step.equity_target is not None for step in rollout.steps)
    assert all(step.expected_showdown_share_target is not None for step in rollout.steps)
    assert all(step.action_index in range(4) for step in rollout.steps)
    assert metrics.samples == rollout.decisions
    assert metrics.value_loss >= 0
    assert metrics.equity_loss >= 0
    assert metrics.expected_showdown_share_loss >= 0
    assert metrics.entropy > 0


def test_checkpoint_archive_rejects_regression_and_adds_frozen_snapshot(tmp_path) -> None:
    model = _model()
    league = OpponentLeague("current", [
        # Current must retain identity with the trainable model.
        LeagueMember(ModelPolicy("current", model), kind="current"),
    ])
    archive = CheckpointArchive(tmp_path, champion_score=1.0)
    rejected = archive.promote(model, score=0.9, league=league)
    accepted = archive.promote(model, score=1.1, league=league)
    assert not rejected.accepted and rejected.checkpoint_path is None
    assert accepted.accepted and accepted.checkpoint_path is not None and accepted.checkpoint_path.exists()
    snapshot = next(member for member in league.members if member.kind == "best")
    assert isinstance(snapshot.policy, ModelPolicy)
    assert all(not parameter.requires_grad for parameter in snapshot.policy.model.parameters())


def test_train_iteration_extends_an_empty_rollout_to_a_current_policy_decision(monkeypatch) -> None:
    model = _model()
    trainer = PPOTrainer(model, default_league(model, seed=5), PPOConfig(epochs=1, equity_samples=1))
    from poker.training import Rollout

    real_collect = trainer.collect_rollout
    calls = 0

    def empty_once(hand_count, **kwargs):
        nonlocal calls
        calls += 1
        return Rollout((), hand_count) if calls == 1 else real_collect(hand_count, **kwargs)

    monkeypatch.setattr(trainer, "collect_rollout", empty_once)
    rollout, metrics = trainer.train_iteration(1, table_count=1, player_count=2)
    assert rollout.hands == 2
    assert rollout.decisions > 0
    assert metrics.samples == rollout.decisions


def test_ppo_rejects_nan_loss_before_optimizer_step(monkeypatch) -> None:
    model = _model()
    trainer = PPOTrainer(model, default_league(model, seed=19), PPOConfig(epochs=1, minibatch_size=8, equity_samples=1))
    rollout = trainer.collect_rollout(2, table_count=2, player_count=2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    import poker.training as training_module

    monkeypatch.setattr(
        training_module.F,
        "mse_loss",
        lambda value, _target: value.sum() * float("nan"),
    )
    with pytest.raises(NonFiniteTrainingError, match="value_loss"):
        trainer.update(rollout)
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(model.parameters(), before, strict=True))


def test_ppo_rejects_infinite_gradient_norm_before_optimizer_step(monkeypatch) -> None:
    model = _model()
    trainer = PPOTrainer(model, default_league(model, seed=23), PPOConfig(epochs=1, minibatch_size=8, equity_samples=1))
    rollout = trainer.collect_rollout(2, table_count=2, player_count=2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *_args, **_kwargs: torch.tensor(float("inf")))
    with pytest.raises(NonFiniteTrainingError, match="gradient norm"):
        trainer.update(rollout)
    assert all(torch.equal(parameter, saved) for parameter, saved in zip(model.parameters(), before, strict=True))


def test_model_parameter_health_rejects_nonfinite_weights() -> None:
    model = _model()
    first = next(model.parameters())
    with torch.no_grad():
        first.view(-1)[0] = float("nan")
    with pytest.raises(NonFiniteTrainingError, match="model parameter"):
        ensure_finite_model_parameters(model)


def test_optimizer_health_rejects_nonfinite_momentum() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"].view(-1)[0] = float("nan")

    with pytest.raises(NonFiniteTrainingError, match="optimizer state"):
        ensure_finite_optimizer_state(optimizer)
