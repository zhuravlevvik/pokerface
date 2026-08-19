"""Small operator CLI for the append-only release registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .releases import LineageArtifact, ReleaseRecord, ReleaseRegistry, ReleaseRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register and verify hash-pinned PPO releases")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="validate evidence and append one release")
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--release-id", required=True)
    register.add_argument("--code-revision", required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--checkpoint-sha256", required=True)
    register.add_argument("--ledger-manifest", type=Path, required=True)
    register.add_argument("--ledger-manifest-sha256", required=True)
    register.add_argument("--tuning-report", type=Path, required=True)
    register.add_argument("--tuning-report-sha256", required=True)
    register.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KIND=PATH=SHA256",
        help="optional extra lineage artifact; repeatable",
    )
    for name, help_text in (("list", "list verified releases"), ("show", "show one verified release"), ("verify", "re-hash release evidence")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--registry", type=Path, required=True)
        if name != "list":
            command.add_argument("--release-id", required=name == "show")
    return parser


def _extra(value: str) -> LineageArtifact:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("--extra must use KIND=PATH=SHA256")
    return LineageArtifact(parts[0], Path(parts[1]), parts[2])


def _record_dict(record: ReleaseRecord) -> dict[str, object]:
    return {
        "release_id": record.release_id,
        "release_path": str(record.release_path),
        "release_sha256": record.release_sha256,
        "request": record.request.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = ReleaseRegistry(args.registry)
    if args.command == "register":
        record = registry.register(
            ReleaseRequest(
                args.release_id,
                args.code_revision,
                args.checkpoint,
                args.checkpoint_sha256,
                args.ledger_manifest,
                args.ledger_manifest_sha256,
                args.tuning_report,
                args.tuning_report_sha256,
                tuple(_extra(value) for value in args.extra),
            )
        )
        print(json.dumps(_record_dict(record), sort_keys=True))
        return 0
    if args.command == "list":
        print(json.dumps([_record_dict(record) for record in registry.list()], sort_keys=True))
        return 0
    result = registry.verify(args.release_id) if args.command == "verify" else registry.show(args.release_id)
    if isinstance(result, tuple):
        print(json.dumps([_record_dict(record) for record in result], sort_keys=True))
    else:
        print(json.dumps(_record_dict(result), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
