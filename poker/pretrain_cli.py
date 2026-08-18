"""Command-line entry point for reproducible Stage 1 pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pretraining_runner import (
    PretrainingRunConfig,
    PretrainingRunner,
    load_pretraining_run_config,
    write_pretraining_run_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a safe corpus and pretrain the poker equity backbone")
    parser.add_argument("--config", type=Path, help="JSON or TOML pretraining configuration")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/pretraining"), help="corpus, checkpoints and reports directory")
    parser.add_argument("--resume", nargs="?", const="latest", help="checkpoint path, or `latest` within --run-dir")
    parser.add_argument("--epochs", type=int, help="absolute target epoch for this invocation")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--write-default-config", type=Path, help="write a JSON starter config and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_default_config is not None:
        write_pretraining_run_config(PretrainingRunConfig(), args.write_default_config)
        print(f"Wrote {args.write_default_config}")
        return 0
    if args.resume:
        checkpoint = args.run_dir / "checkpoints" / "latest.pt" if args.resume == "latest" else Path(args.resume)
        runner = PretrainingRunner.resume(checkpoint, device=args.device)
    else:
        config = load_pretraining_run_config(args.config) if args.config is not None else PretrainingRunConfig()
        runner = PretrainingRunner(config, args.run_dir, device=args.device)
    result = runner.run(until_epoch=args.epochs, install_signal_handlers=True)
    status = "interrupted safely" if result.interrupted else "completed"
    acceptance = "passed" if result.acceptance_passed else "needs tuning"
    print(
        f"Pretraining {status}: epoch={result.epoch} step={result.global_step} "
        f"acceptance={acceptance} checkpoint={result.checkpoint_path} report={result.report_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
