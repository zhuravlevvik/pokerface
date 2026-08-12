"""Contract tests for the optional PyTorch policy model."""

from __future__ import annotations

from copy import deepcopy

import pytest

from poker.game_state import HandState
from poker.model import ACTION_NAMES, BET_SIZE_ACTIONS, TORCH_AVAILABLE, ModelConfig, PokerAgentModel
from poker.observation import observation_for

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _decision_observation(seed: int = 42) -> dict:
    state = HandState(seed=seed)
    assert state.actor is not None
    return observation_for(state, state.actor)


def _short_handed(observation: dict, count: int) -> dict:
    result = deepcopy(observation)
    result["player_set"] = result["player_set"][:count]
    result["player_mask"] = [True] * count
    return result


def test_model_handles_two_three_and_five_player_sets_with_one_weight_set() -> None:
    torch.manual_seed(7)
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    output = model([_short_handed(_decision_observation(1), 2), _short_handed(_decision_observation(2), 3), _decision_observation(3)])

    assert output.action_logits.shape == (3, len(ACTION_NAMES))
    assert output.bet_size_logits.shape == (3, len(BET_SIZE_ACTIONS))
    assert output.value.shape == (3,)
    assert output.equity_logits.shape == (3, 3)
    assert torch.isfinite(output.action_logits).all()
    assert torch.isfinite(output.bet_size_logits).all()
    assert torch.isfinite(output.value).all()
    assert torch.isfinite(output.equity_logits).all()
    assert torch.allclose(output.action_probabilities.sum(dim=-1), torch.ones(3))
    assert torch.allclose(output.bet_size_probabilities.sum(dim=-1), torch.ones(3))
    assert torch.allclose(output.equity_probabilities.sum(dim=-1), torch.ones(3))


def test_masks_zero_illegal_probabilities_and_inference_never_selects_them() -> None:
    torch.manual_seed(8)
    observation = _decision_observation()
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    model.eval()
    output = model([observation])
    assert torch.equal(output.action_probabilities[0][~output.action_mask[0]], torch.zeros_like(output.action_probabilities[0][~output.action_mask[0]]))
    assert torch.equal(output.bet_size_probabilities[0][~output.bet_size_mask[0]], torch.zeros_like(output.bet_size_probabilities[0][~output.bet_size_mask[0]]))
    decision = model.infer(observation)
    assert observation["legal_action_mask"][decision.action]


def test_inference_is_deterministic_and_checkpoint_is_versioned(tmp_path) -> None:
    torch.manual_seed(9)
    observation = _decision_observation()
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    first = model.infer(observation)
    second = model.infer(observation)
    assert first == second
    checkpoint = tmp_path / "agent.pt"
    model.save_checkpoint(checkpoint)
    restored = PokerAgentModel.load_checkpoint(checkpoint)
    assert restored.checkpoint_metadata() == model.checkpoint_metadata()
    assert restored.infer(observation) == first
