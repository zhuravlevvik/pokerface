"""Command-line entry point for resumable PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train_runner import TrainingRunConfig, TrainingRunner, load_run_config, write_run_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a resumable poker PPO run")
    parser.add_argument("--config", type=Path, help="JSON or TOML training configuration")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"), help="directory containing checkpoints and manifest")
    parser.add_argument("--resume", nargs="?", const="latest", help="checkpoint path, or `latest` within --run-dir")
    parser.add_argument("--iterations", type=int, help="absolute target iteration (overrides config for this invocation)")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--write-default-config", type=Path, help="write a commented-free JSON starter config and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_default_config is not None:
        write_run_config(TrainingRunConfig(), args.write_default_config)
        print(f"Wrote {args.write_default_config}")
        return 0
    if args.resume:
        checkpoint = args.run_dir / "checkpoints" / "latest.pt" if args.resume == "latest" else Path(args.resume)
        runner = TrainingRunner.resume(checkpoint, device=args.device)
    else:
        config = load_run_config(args.config) if args.config is not None else TrainingRunConfig()
        runner = TrainingRunner(config, args.run_dir, device=args.device)
    result = runner.run(until_iteration=args.iterations, install_signal_handlers=True)
    status = "interrupted safely" if result.interrupted else "completed"
    print(
        f"Training {status}: iteration={result.iteration} hands={result.global_hands} "
        f"decisions={result.global_decisions} checkpoint={result.checkpoint_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
