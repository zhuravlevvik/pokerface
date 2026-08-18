"""Optional FastAPI/WebSocket adapter for :mod:`poker.game_server`.

Install ``pokerface[web]`` to run it.  Keeping this import optional means the
engine, training jobs and non-web tests do not take a runtime web dependency.
"""

from pathlib import Path
from typing import Any

from .game_server import GameServer


def create_app(game_server: GameServer | None = None, *, ui_defaults: dict[str, object] | None = None):
    """Build the HTTP/WebSocket app without starting an external server."""

    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, JSONResponse
    except ModuleNotFoundError as error:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("Web UI requires `pip install -e '.[web]'`.") from error

    server = game_server or GameServer()
    configured_defaults = dict(ui_defaults or {})
    app = FastAPI(title="Pokerface observer", version="0.1.0")
    static_dir = Path(__file__).with_name("static")

    def run_command(command: dict[str, Any]) -> list[dict[str, object]]:
        message_type = command.get("type", "start_hand")
        mode = command.get("mode", "player")
        hero_seat = command.get("hero_seat", 0)
        if not isinstance(mode, str) or not isinstance(hero_seat, int):
            raise ValueError("mode must be a string and hero_seat an integer")
        if message_type == "start_hand":
            seed = command.get("seed_start", command.get("seed"))
            if seed is not None and not isinstance(seed, int):
                raise ValueError("seed must be an integer or null")
            player_count = command.get("player_count", 5)
            hands = command.get("hands", 1)
            button_seat = command.get("button_seat", 0)
            if not all(isinstance(value, int) for value in (player_count, hands, button_seat)):
                raise ValueError("player_count, hands and button_seat must be integers")
            raw_policies = command.get("seat_policies", {})
            if not isinstance(raw_policies, dict):
                raise ValueError("seat_policies must be an object mapping seat to a known policy id")
            seat_policies: dict[int, str] = {}
            for raw_seat, policy_id in raw_policies.items():
                try:
                    seat = int(raw_seat)
                except (TypeError, ValueError) as error:
                    raise ValueError("seat policy keys must be integer seats") from error
                if not isinstance(policy_id, str):
                    raise ValueError("seat policy ids must be strings")
                seat_policies[seat] = policy_id
            return server.observe_series(
                seed_start=seed,
                hands=hands,
                button_seat=button_seat,
                player_count=player_count,
                hero_seat=hero_seat,
                mode=mode,  # type: ignore[arg-type]
                seat_policies=seat_policies,
            )
        if message_type == "replay":
            replay = command.get("replay")
            if not isinstance(replay, dict):
                raise ValueError("replay command needs a replay object")
            return server.replay_hand(replay, hero_seat=hero_seat, mode=mode)  # type: ignore[arg-type]
        raise ValueError("type must be 'start_hand' or 'replay'")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/static.css")
    def stylesheet():
        return FileResponse(static_dir / "static.css", media_type="text/css")

    @app.get("/app.js")
    def javascript():
        return FileResponse(static_dir / "app.js", media_type="text/javascript")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "transport": "websocket", "inference": type(server.inference).__name__}

    @app.get("/api/policies")
    def policies():
        """Expose only server-registered policy ids, never checkpoint paths."""

        return {
            "policies": server.available_policies(),
            "seat_policies": {str(seat): policy for seat, policy in server.default_seat_policies.items()},
            "defaults": configured_defaults,
        }

    @app.post("/api/hand")
    async def hand(payload: dict[str, Any]):
        try:
            return JSONResponse({"events": run_command(payload)})
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.websocket("/ws/table")
    async def table_socket(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                command = await socket.receive_json()
                if not isinstance(command, dict):
                    await socket.send_json({"type": "error", "detail": "message must be an object"})
                    continue
                try:
                    for event in run_command(command):
                        await socket.send_json(event)
                except (KeyError, TypeError, ValueError) as error:
                    await socket.send_json({"type": "error", "detail": str(error)})
        except WebSocketDisconnect:
            return

    return app
