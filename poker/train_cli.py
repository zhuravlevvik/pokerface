"""Command-line entry point for resumable PPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train_runner import TrainingRunConfig, TrainingRunner, load_run_config, write_run_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a resumable poker PPO run")
    parser.add_argument("--config", type=Path, help="JSON or TOML training configuration")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"), help="directory containing checkpoints and manifest")
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--resume", nargs="?", const="latest", help="full-run checkpoint path, or `latest` within --run-dir")
    restore.add_argument("--init-checkpoint", type=Path, help="model/pretraining checkpoint used only to initialize a fresh PPO run")
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
    config = load_run_config(args.config) if args.config is not None else None
    if args.resume:
        if config is not None and config.init_checkpoint is not None:
            _parser().error("--resume cannot be combined with config init_checkpoint; resume restores its own full run state")
        checkpoint = args.run_dir / "checkpoints" / "latest.pt" if args.resume == "latest" else Path(args.resume)
        runner = TrainingRunner.resume(checkpoint, device=args.device)
    else:
        if args.init_checkpoint is not None and config is not None and config.init_checkpoint is not None:
            _parser().error("specify init checkpoint either in config or with --init-checkpoint, not both")
        runner = TrainingRunner(config or TrainingRunConfig(), args.run_dir, device=args.device, init_checkpoint=args.init_checkpoint)
    result = runner.run(until_iteration=args.iterations, install_signal_handlers=True)
    status = "interrupted safely" if result.interrupted else "completed"
    print(
        f"Training {status}: iteration={result.iteration} hands={result.global_hands} "
        f"decisions={result.global_decisions} checkpoint={result.checkpoint_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
