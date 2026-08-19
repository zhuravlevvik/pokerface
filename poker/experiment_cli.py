"""CLI for one resumable, ledger-backed PPO experiment trial."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .experiment_runner import ExperimentRunner, load_experiment_config, write_experiment_config
from .experiments import ExperimentConfig
from .train_runner import TrainingRunConfig
from .promotion import PromotionConfig
from .tuning import write_hu_promotion_protocol


def _default_config(code_revision: str, protocol_artifact: Path) -> ExperimentConfig:
    training = TrainingRunConfig()
    training = replace(training, run=replace(training.run, checkpoint_every_iterations=1))
    protocol_path, protocol_sha = write_hu_promotion_protocol(training, PromotionConfig(enabled=True), protocol_artifact)
    return ExperimentConfig(
        name="hu-stage-a-pilot",
        training=training,
        max_iterations=training.run.iterations,
        evaluation_protocol_path=str(protocol_path),
        evaluation_protocol_sha256=protocol_sha,
        code_revision=code_revision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume one immutable PPO experiment trial")
    parser.add_argument("--config", type=Path, help="experiment JSON configuration")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/experiments/default"))
    parser.add_argument("--until-iteration", type=int, help="pause at this absolute iteration boundary")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--write-default-config", type=Path, help="write a starter experiment config and exit")
    parser.add_argument("--code-revision", help="immutable source/build revision used with --write-default-config")
    parser.add_argument("--protocol-artifact", type=Path, help="path to create the preregistered HU evaluation protocol")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.write_default_config is not None:
        if not args.code_revision or args.protocol_artifact is None:
            parser.error("--write-default-config requires --code-revision and --protocol-artifact")
        write_experiment_config(_default_config(args.code_revision, args.protocol_artifact), args.write_default_config)
        print(f"Wrote {args.write_default_config} and pinned protocol {args.protocol_artifact}")
        return 0
    if args.config is None:
        parser.error("--config is required")
    runner = ExperimentRunner(load_experiment_config(args.config), args.run_dir, device=args.device)
    result = runner.run(until_iteration=args.until_iteration, install_signal_handlers=True)
    print(
        f"Experiment {result.status}: trial={result.trial_id} iteration={result.iteration} "
        f"hands={result.global_hands} decisions={result.global_decisions} metrics={result.metrics_path}"
    )
    if result.status == "aborted_immediate":
        return 130
    return 2 if result.status == "failed_nonfinite" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
