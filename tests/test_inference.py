"""Public inference semantics, independent of the browser transport."""

from __future__ import annotations

import pytest

from poker.inference import (
    CheckpointInferenceService,
    EXPECTED_SHOWDOWN_SHARE_PROTOCOL,
    HEURISTIC_HAND_STRENGTH_PROTOCOL,
    InferenceResponse,
    ScalarMetric,
    validate_response,
)
from poker.game_state import HandState
from poker.model import TORCH_AVAILABLE, PokerAgentModel
from poker.observation import observation_for


def test_scalar_metric_is_explicit_and_serialised_separately_from_outcomes() -> None:
    response = InferenceResponse(
        "call",
        {},
        {},
        {"win": 0.25, "tie": 0.25, "loss": 0.5},
        1.0,
        ScalarMetric("expected_showdown_share", 0.375, EXPECTED_SHOWDOWN_SHARE_PROTOCOL),
    )
    encoded = response.as_dict()

    assert encoded["equity"] == {"win": 0.25, "tie": 0.25, "loss": 0.5}
    assert "total" not in encoded["equity"]
    assert encoded["scalar_metric"] == {
        "name": "expected_showdown_share",
        "value": 0.375,
        "protocol": EXPECTED_SHOWDOWN_SHARE_PROTOCOL,
    }
    validate_response(response, {"call": True})


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="checkpoint inference needs PyTorch")
def test_checkpoint_inference_emits_learned_showdown_share_not_heads_up_total() -> None:
    state = HandState(seed=22, player_count=3)
    assert state.actor is not None
    response = CheckpointInferenceService(PokerAgentModel()).decide(observation_for(state, state.actor))

    assert set(response.equity) == {"win", "tie", "loss"}
    assert response.scalar_metric is not None
    assert response.scalar_metric.name == "expected_showdown_share"
    assert response.scalar_metric.protocol == EXPECTED_SHOWDOWN_SHARE_PROTOCOL
    assert 0.0 <= response.scalar_metric.value <= 1.0


@pytest.mark.parametrize(
    ("metric", "error"),
    [
        (ScalarMetric("heuristic_hand_strength", 1.1, HEURISTIC_HAND_STRENGTH_PROTOCOL), r"in \[0, 1\]"),
        (ScalarMetric("unknown", 0.5, "unknown_v1"), "unknown name"),
    ],
)
def test_scalar_metric_validation_rejects_untyped_or_invalid_values(metric: ScalarMetric, error: str) -> None:
    response = InferenceResponse("check", {}, {}, {"win": 0.0, "tie": 0.0, "loss": 1.0}, 0.0, metric)
    with pytest.raises(ValueError, match=error):
        validate_response(response, {"check": True})
