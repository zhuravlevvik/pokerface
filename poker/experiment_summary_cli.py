"""CLI for a read-only, player-safe experiment ledger health summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment_summary import ExperimentHealthConfig, summarize_experiment, write_experiment_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a verified experiment ledger without reading rollout data")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-abs-kl", type=float)
    parser.add_argument("--max-clip-fraction", type=float)
    parser.add_argument("--min-entropy", type=float)
    parser.add_argument("--max-value-loss", type=float)
    parser.add_argument("--max-gradient-norm", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    health = ExperimentHealthConfig(
        max_abs_kl=args.max_abs_kl,
        max_clip_fraction=args.max_clip_fraction,
        min_entropy=args.min_entropy,
        max_value_loss=args.max_value_loss,
        max_gradient_norm=args.max_gradient_norm,
    )
    summary = summarize_experiment(args.run_dir, health=health)
    write_experiment_summary(summary, args.output)
    print(
        f"Experiment {summary.status}: trial={summary.trial_id} "
        f"completed={summary.completed} alerts={len(summary.alerts)} output={args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
