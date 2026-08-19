"""Contract tests for the optional PyTorch policy model."""

from __future__ import annotations

from copy import deepcopy

import pytest

from poker.game_state import HandState
from poker.model import ACTION_NAMES, BET_SIZE_ACTIONS, EQUITY_HEADS, MODEL_VERSION, TORCH_AVAILABLE, ModelConfig, PokerAgentModel
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


def _history_record(*, position: str, action: str, amount_bb: float, current_bet_after_bb: float) -> dict:
    return {
        "street": "preflop",
        "street_index": 0,
        "position": position,
        "action": action,
        "amount_bb": amount_bb,
        "amount_to_pot": amount_bb / 1.5,
        "raise_to_bb": None,
        "raise_to_to_pot": None,
        "current_bet_after_bb": current_bet_after_bb,
    }


def _card_representation(model: PokerAgentModel, observation: dict) -> torch.Tensor:
    tensors = model.tensorize([observation])
    return model.card_encoder(tensors["card_ranks"], tensors["card_suits"], tensors["card_roles"], tensors["card_mask"])


def _history_representation(model: PokerAgentModel, observation: dict) -> torch.Tensor:
    tensors = model.tensorize([observation])
    return model.history_encoder(
        tensors["history_streets"],
        tensors["history_positions"],
        tensors["history_actions"],
        tensors["history_numeric"],
        tensors["history_mask"],
    )


def test_model_handles_two_three_and_five_player_sets_with_one_weight_set() -> None:
    torch.manual_seed(7)
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    output = model([_short_handed(_decision_observation(1), 2), _short_handed(_decision_observation(2), 3), _decision_observation(3)])

    assert output.action_logits.shape == (3, len(ACTION_NAMES))
    assert output.bet_size_logits.shape == (3, len(BET_SIZE_ACTIONS))
    assert output.value.shape == (3,)
    assert output.equity_logits.shape == (3, 3)
    assert output.expected_showdown_share_logit.shape == (3,)
    assert output.expected_showdown_share.shape == (3,)
    assert torch.isfinite(output.action_logits).all()
    assert torch.isfinite(output.bet_size_logits).all()
    assert torch.isfinite(output.value).all()
    assert torch.isfinite(output.equity_logits).all()
    assert torch.isfinite(output.expected_showdown_share).all()
    assert torch.allclose(output.action_probabilities.sum(dim=-1), torch.ones(3))
    assert torch.allclose(output.bet_size_probabilities.sum(dim=-1), torch.ones(3))
    assert torch.allclose(output.equity_probabilities.sum(dim=-1), torch.ones(3))
    assert torch.all((output.expected_showdown_share >= 0.0) & (output.expected_showdown_share <= 1.0))


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
    assert 0.0 <= decision.expected_showdown_share <= 1.0


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


def test_card_encoder_distinguishes_which_concrete_card_is_private_or_on_board() -> None:
    """Regression: v1 pooled rank/suit/role sums and lost this association."""

    torch.manual_seed(10)
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4)).eval()
    private_ace = _decision_observation()
    private_ace["cards"] = {
        "hole_cards": ["Ah", "Kd"],
        "board": ["Qs", "Jc", "2d"],
        "street": "flop",
        "street_index": 1,
    }
    board_ace = deepcopy(private_ace)
    board_ace["cards"] = {
        "hole_cards": ["Qs", "Kd"],
        "board": ["Ah", "Jc", "2d"],
        "street": "flop",
        "street_index": 1,
    }

    assert not torch.allclose(_card_representation(model, private_ace), _card_representation(model, board_ace))


def test_history_encoder_distinguishes_action_order() -> None:
    """Regression: an order-free Transformer treated these sequences alike."""

    torch.manual_seed(11)
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4)).eval()
    first = _decision_observation()
    first["action_history"] = [
        _history_record(position="UTG", action="raise_min", amount_bb=2.0, current_bet_after_bb=2.0),
        _history_record(position="CO", action="call", amount_bb=2.0, current_bet_after_bb=2.0),
    ]
    second = deepcopy(first)
    second["action_history"] = list(reversed(first["action_history"]))

    assert not torch.allclose(_history_representation(model, first), _history_representation(model, second))


def test_model_version_rejects_v2_checkpoint_without_expected_showdown_share_head() -> None:
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    metadata = model.checkpoint_metadata()
    metadata["model_version"] = "2.0"

    assert MODEL_VERSION == "3.0"
    assert EQUITY_HEADS == ("outcome_v1", "expected_showdown_share_v1")
    with pytest.raises(ValueError, match="incompatible checkpoint model_version"):
        model._validate_metadata(metadata)
