"""Launch the safe browser viewer for checkpoint-vs-bot poker hands.

Example::

    python -m poker.watch --checkpoint hu-best=runs/hu/best.pt \\
        --seat checkpoint:hu-best --seat bot:rule --hands 3 --seed 42000

Checkpoint paths are accepted here, at the operator-controlled CLI boundary.
The browser receives only the ``checkpoint:<name>`` ids from the resulting
server catalog and can never ask the server to open an arbitrary path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .game_server import GameServer
from .web import create_app


def _checkpoint(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path or ":" in name:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH, where NAME contains no ':'")
    return name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch poker checkpoint-vs-bot hands in a browser.")
    parser.add_argument("--checkpoint", action="append", default=[], type=_checkpoint, metavar="NAME=PATH", help="register a checkpoint")
    parser.add_argument("--seat", action="append", default=[], metavar="POLICY", help="policy for consecutive seats, e.g. checkpoint:best or bot:rule")
    parser.add_argument("--players", type=int, choices=(2, 3, 5), default=2, help="initial table size")
    parser.add_argument("--hands", type=int, default=1, choices=range(1, 1001), metavar="1..1000", help="initial number of hands")
    parser.add_argument("--seed", type=int, default=None, help="initial seed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    catalog: dict[str, str] = {}
    for name, raw_path in args.checkpoint:
        if name in catalog:
            raise SystemExit(f"duplicate checkpoint name: {name}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"checkpoint does not exist or is not a file: {path}")
        catalog[name] = str(path)
    if len(args.seat) > args.players:
        raise SystemExit("more --seat values than --players")
    defaults = {seat: policy for seat, policy in enumerate(args.seat)}
    server = GameServer(checkpoint_catalog=catalog, default_seat_policies=defaults)
    # Validate ids eagerly, so a typo is reported before Uvicorn is running.
    server._seat_router(player_count=args.players, seat_policies=defaults, seed=args.seed)
    app = create_app(server, ui_defaults={"player_count": args.players, "hands": args.hands, "seed_start": args.seed})
    try:
        import uvicorn
    except ModuleNotFoundError as error:  # pragma: no cover - depends on optional web extra.
        raise SystemExit("Web viewer needs `pip install -e '.[web]'`.") from error
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()
