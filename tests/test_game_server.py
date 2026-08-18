from __future__ import annotations

import json

import pytest

from poker.game_server import GameServer, SeatPolicyRouter
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
    event = next(event for event in _events(mode="spectator") if event["type"] == "action")
    analysis = event["analysis"]
    assert isinstance(analysis, dict)
    assert set(analysis) == {"action", "action_probabilities", "bet_size_probabilities", "equity", "value_bb"}
    assert set(analysis["equity"]) == {"win", "tie", "loss", "total"}
    assert sum(analysis["equity"][key] for key in ("win", "tie", "loss")) == pytest.approx(1.0)
    assert analysis["equity"]["total"] == pytest.approx(analysis["equity"]["win"] + 0.5 * analysis["equity"]["tie"])


def test_player_mode_redacts_opponent_private_analysis_from_events_and_replay() -> None:
    events = _events(mode="player")
    actions = [event for event in events if event["type"] == "action"]
    assert any(event["seat"] != 0 for event in actions)
    for event in actions:
        if event["seat"] == 0:
            assert isinstance(event["analysis"], dict)
        else:
            assert event["analysis"] is None
    replay = events[-1]["replay"]
    assert isinstance(replay, dict)
    assert all(point["seat"] == 0 for point in replay["equity_points"])
    for record in replay["analyses"]:
        if record["seat"] != 0:
            assert record["analysis"] is None


def test_inference_response_rejects_illegal_action_and_bad_equity() -> None:
    illegal = InferenceResponse("fold", {}, {}, {"win": 1.0, "tie": 0.0, "loss": 0.0, "total": 1.0}, 0.0)
    with pytest.raises(ValueError, match="illegal"):
        validate_response(illegal, {"fold": False})
    inconsistent = InferenceResponse("call", {}, {}, {"win": 0.5, "tie": 0.0, "loss": 0.5, "total": 0.9}, 0.0)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_response(inconsistent, {"call": True})


def test_seat_policy_router_dispatches_each_observation_to_its_own_service() -> None:
    class Marker:
        def __init__(self, action: str) -> None:
            self.action = action
            self.calls: list[int] = []

        def decide(self, observation: object) -> InferenceResponse:
            assert isinstance(observation, dict)
            self.calls.append(observation["seat"])
            return InferenceResponse(
                self.action, {}, {}, {"win": 1.0, "tie": 0.0, "loss": 0.0, "total": 1.0}, 0.0
            )

    first, second = Marker("check"), Marker("call")
    router = SeatPolicyRouter({0: first, 1: second})
    assert router.decide({"seat": 0}).action == "check"
    assert router.decide({"seat": 1}).action == "call"
    assert first.calls == [0]
    assert second.calls == [1]


@pytest.mark.parametrize("player_count", [2, 3, 5])
def test_mixed_policy_series_is_seeded_and_exposes_policy_identity(player_count: int) -> None:
    server = GameServer()
    policies = {seat: ("bot:rule" if seat == 0 else "bot:tight") for seat in range(player_count)}
    first = server.observe_series(
        seed_start=9120,
        hands=2,
        player_count=player_count,
        hero_seat=0,
        mode="spectator",
        seat_policies=policies,
    )
    second = server.observe_series(
        seed_start=9120,
        hands=2,
        player_count=player_count,
        hero_seat=0,
        mode="spectator",
        seat_policies=policies,
    )
    assert first == second
    assert first[0]["type"] == "series_started"
    first_table = next(event["table"] for event in first if event["type"] == "hand_started")
    assert first_table["players"][0]["policy"] == {"id": "bot:rule", "name": "Rule bot", "kind": "bot"}
    assert first[-1]["summary"]["hands"] == 2
    assert set(first[-1]["summary"]["pnl"]) == set(range(player_count))


def test_checkpoint_catalog_uses_a_safe_id_instead_of_a_client_path() -> None:
    server = GameServer(checkpoint_catalog={"candidate": GameServer().inference})
    events = server.observe_hand(
        seed=99,
        player_count=2,
        hero_seat=0,
        mode="spectator",
        seat_policies={0: "checkpoint:candidate", 1: "bot:rule"},
    )
    started = events[0]["table"]
    assert started["players"][0]["policy"] == {"id": "checkpoint:candidate", "name": "candidate", "kind": "checkpoint"}
    replay = events[-1]["replay"]
    assert isinstance(replay, dict)
    imported = GameServer().replay_hand(replay, hero_seat=0, mode="spectator")
    imported_table = imported[-1]["table"]
    assert imported_table["players"][0]["policy"] == started["players"][0]["policy"]
    assert {policy["id"] for policy in server.available_policies()} >= {"checkpoint:candidate", "bot:rule"}
    with pytest.raises(ValueError, match="not in the server catalog"):
        server.observe_hand(player_count=2, seat_policies={0: "checkpoint:/tmp/not-allowed"})
