from __future__ import annotations

import json

import pytest

from poker.game_server import GameServer
from poker.inference import InferenceResponse, validate_response


def _events(*, mode: str = "player") -> list[dict[str, object]]:
    return GameServer().observe_hand(seed=20260812, hero_seat=0, mode=mode)  # type: ignore[arg-type]


def test_observable_hand_is_json_safe_and_player_mode_hides_opponent_cards() -> None:
    events = _events()

    assert events[0]["type"] == "hand_started"
    assert events[-1]["type"] == "hand_complete"
    assert json.dumps(events)
    for event in events:
        table = event.get("table")
        if not isinstance(table, dict):
            continue
        players = table["players"]
        assert isinstance(players, list)
        for player in players:
            assert isinstance(player, dict)
            if player["seat"] == 0:
                assert len(player["hole_cards"]) == 2
            else:
                assert "hole_cards" not in player
    # A player-safe replay remains sufficient to reconstruct the hand because
    # its seed and actions are authoritative.
    replay = events[-1]["replay"]
    assert isinstance(replay, dict)
    assert all("hole_cards" not in player for player in replay["players"])
    reproduced = GameServer().replay_hand(replay, hero_seat=0, mode="player")
    assert reproduced[-1]["table"] == events[-1]["table"]


def test_spectator_reveals_cards_only_when_hand_is_complete() -> None:
    events = _events(mode="spectator")
    in_progress = next(event for event in events if event["type"] == "hand_started")
    table = in_progress["table"]
    assert isinstance(table, dict)
    assert all("hole_cards" not in player for player in table["players"])
    complete = events[-1]["table"]
    assert isinstance(complete, dict)
    assert all(len(player["hole_cards"]) == 2 for player in complete["players"])


def test_action_events_expose_full_inference_contract() -> None:
    event = next(event for event in _events() if event["type"] == "action")
    analysis = event["analysis"]
    assert isinstance(analysis, dict)
    assert set(analysis) == {"action", "action_probabilities", "bet_size_probabilities", "equity", "value_bb"}
    assert set(analysis["equity"]) == {"win", "tie", "loss", "total"}
    assert sum(analysis["equity"][key] for key in ("win", "tie", "loss")) == pytest.approx(1.0)
    assert analysis["equity"]["total"] == pytest.approx(analysis["equity"]["win"] + 0.5 * analysis["equity"]["tie"])


def test_inference_response_rejects_illegal_action_and_bad_equity() -> None:
    illegal = InferenceResponse("fold", {}, {}, {"win": 1.0, "tie": 0.0, "loss": 0.0, "total": 1.0}, 0.0)
    with pytest.raises(ValueError, match="illegal"):
        validate_response(illegal, {"fold": False})
    inconsistent = InferenceResponse("call", {}, {}, {"win": 0.5, "tie": 0.0, "loss": 0.5, "total": 0.9}, 0.0)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_response(inconsistent, {"call": True})
