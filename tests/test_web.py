from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from poker.game_server import GameServer
from poker.web import create_app


def test_web_lists_safe_policy_ids_and_runs_mixed_series() -> None:
    server = GameServer(checkpoint_catalog={"candidate": GameServer().inference})
    client = TestClient(create_app(server, ui_defaults={"player_count": 2, "hands": 2, "seed_start": 10}))
    catalog = client.get("/api/policies")
    assert catalog.status_code == 200
    payload = catalog.json()
    assert payload["defaults"]["player_count"] == 2
    assert {item["id"] for item in payload["policies"]} >= {"bot:rule", "checkpoint:candidate"}
    response = client.post(
        "/api/hand",
        json={
            "type": "start_hand",
            "seed_start": 12,
            "hands": 2,
            "player_count": 2,
            "hero_seat": 0,
            "mode": "spectator",
            "seat_policies": {"0": "checkpoint:candidate", "1": "bot:rule"},
        },
    )
    assert response.status_code == 200
    events = response.json()["events"]
    assert events[0]["type"] == "series_started"
    assert events[-1]["type"] == "series_complete"
    unsafe = client.post(
        "/api/hand",
        json={"type": "start_hand", "player_count": 2, "seat_policies": {"0": "checkpoint:/tmp/agent.pt"}},
    )
    assert unsafe.status_code == 422
    assert "catalog" in unsafe.json()["detail"]
