"""CLI for deterministic sweep materialization and evidence comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment_runner import load_experiment_config
from .train_runner import TrainingRunConfig
from .promotion import PromotionConfig, PromotionEvaluator
from .train_runner import TrainingRunner
from .tuning import (
    SweepConfig,
    TuningEvidence,
    compare_tuning_evidence,
    load_sweep_config,
    materialize_sweep,
    publish_tuning_evaluation,
    single_experiment_trial,
    write_hu_promotion_protocol,
    write_sweep_config,
)


def _default_sweep(code_revision: str, protocol_artifact: Path) -> SweepConfig:
    training = TrainingRunConfig()
    protocol_path, protocol_sha = write_hu_promotion_protocol(training, PromotionConfig(enabled=True), protocol_artifact)
    return SweepConfig(
        base_config=training,
        grid={"learning_rate": (1e-4, 3e-4), "entropy_coefficient": (0.005, 0.01)},
        seeds=(11, 31),
        max_iterations=10,
        evaluation_protocol_sha256=protocol_sha,
        evaluation_protocol_path=str(protocol_path),
        code_revision=code_revision,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize and compare immutable PPO tuning trials")
    parser.add_argument("--write-default-config", type=Path, help="write a starter sweep config and exit")
    parser.add_argument("--code-revision", help="immutable source/build revision used with --write-default-config")
    parser.add_argument("--protocol-artifact", type=Path, help="path to create the preregistered HU evaluation protocol")
    subparsers = parser.add_subparsers(dest="command")

    materialize = subparsers.add_parser("materialize", help="write one experiment config per trial")
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="run the preregistered HU promotion suite")
    evaluate_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_source.add_argument("--config", type=Path, help="immutable sweep configuration")
    evaluate_source.add_argument("--experiment-config", type=Path, help="standalone immutable experiment configuration")
    evaluate.add_argument("--trial-id", help="trial id within --config; not used with --experiment-config")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)

    seal = subparsers.add_parser("seal", help="bind fixed-evaluation metrics to a final checkpoint")
    seal_source = seal.add_mutually_exclusive_group(required=True)
    seal_source.add_argument("--config", type=Path, help="immutable sweep configuration")
    seal_source.add_argument("--experiment-config", type=Path, help="standalone immutable experiment configuration")
    seal.add_argument("--trial-id", help="trial id within --config; not used with --experiment-config")
    seal.add_argument("--checkpoint", type=Path, required=True)
    seal.add_argument("--ledger-manifest", type=Path, required=True)
    seal.add_argument("--promotion-report", type=Path, required=True)
    seal.add_argument("--promotion-archive-manifest", type=Path, required=True)
    seal.add_argument("--report", type=Path, required=True)

    compare = subparsers.add_parser("compare", help="verify and rank one report per trial")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--report", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def _trial(config: SweepConfig, trial_id: str):
    matches = [trial for trial in config.expand_trials() if trial.trial_id == trial_id]
    if len(matches) != 1:
        raise ValueError(f"unknown tuning trial_id {trial_id!r}")
    return matches[0]


def _command_trial(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """Resolve the one permitted source of evaluation/sealing lineage."""

    if args.config is not None:
        if not args.trial_id:
            parser.error("--config requires --trial-id")
        return _trial(load_sweep_config(args.config), args.trial_id)
    if args.trial_id is not None:
        parser.error("--trial-id may only be used with --config")
    return single_experiment_trial(load_experiment_config(args.experiment_config))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.write_default_config is not None:
        if not args.code_revision or args.protocol_artifact is None:
            parser.error("--write-default-config requires --code-revision and --protocol-artifact")
        write_sweep_config(_default_sweep(args.code_revision, args.protocol_artifact), args.write_default_config)
        print(f"Wrote {args.write_default_config} and pinned protocol {args.protocol_artifact}")
        return 0
    if args.command is None:
        parser.error("choose materialize, seal or compare")
    if args.command == "materialize":
        config = load_sweep_config(args.config)
        trials = materialize_sweep(config, args.output_dir)
        print(f"Materialized {len(trials)} trials under {args.output_dir}")
        return 0
    if args.command == "evaluate":
        trial = _command_trial(args, parser)
        protocol = json.loads(Path(trial.evaluation_protocol_path).read_text(encoding="utf-8"))
        promotion_data = protocol.get("promotion_config") if isinstance(protocol, dict) else None
        if not isinstance(promotion_data, dict):
            raise ValueError("evaluation protocol artifact has no promotion_config")
        promotion_protocol = protocol.get("promotion_protocol")
        evaluation_run_seed = (
            promotion_protocol.get("evaluation_run_seed")
            if isinstance(promotion_protocol, dict)
            else None
        )
        if isinstance(evaluation_run_seed, bool) or not isinstance(evaluation_run_seed, int) or evaluation_run_seed < 0:
            raise ValueError("evaluation protocol artifact has no valid fixed evaluation_run_seed")
        runner = TrainingRunner.resume(args.checkpoint)
        if runner.config.to_dict() != trial.config.to_dict() or runner.iteration != trial.config.run.iterations:
            raise ValueError("evaluation checkpoint does not match the completed tuning trial")
        evaluator = PromotionEvaluator(PromotionConfig(**promotion_data), args.output_dir, run_seed=evaluation_run_seed)
        result = evaluator.evaluate_and_promote(
            iteration=runner.iteration,
            candidate_checkpoint=args.checkpoint,
            league=runner.league,
            stage=runner.scheduler.stage,
            champion_score=None,
            run_context={
                "run_config_sha256": trial.run_config_sha256,
                "evaluation_protocol_sha256": trial.evaluation_protocol_sha256,
                "evaluation_run_seed": evaluation_run_seed,
            },
        )
        print(f"Evaluation {'accepted' if result.accepted else 'rejected'}: report={result.report_path} archive={evaluator.archive_manifest_path}")
        return 0
    if args.command == "seal":
        evidence = publish_tuning_evaluation(
            _command_trial(args, parser),
            args.checkpoint,
            args.ledger_manifest,
            args.promotion_report,
            args.promotion_archive_manifest,
            args.report,
        )
        print(f"Sealed {evidence.trial_id}: {evidence.evaluation_report_path}")
        return 0
    reports: list[TuningEvidence] = []
    config = load_sweep_config(args.config)
    trials = {trial.trial_id: trial for trial in config.expand_trials()}
    for path in args.report:
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial_id = payload.get("trial_id") if isinstance(payload, dict) else None
        if trial_id not in trials:
            raise ValueError(f"report contains unknown trial_id {trial_id!r}")
        reports.append(TuningEvidence.from_artifacts(trials[trial_id], payload["full_checkpoint_path"], path))
    result = compare_tuning_evidence(config, reports)
    result.write_json(args.output)
    print(f"Compared {len(reports)} trials; winner={None if result.winner is None else result.winner.trial.trial_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
