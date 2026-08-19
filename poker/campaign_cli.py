"""CLI for complete-matrix, multi-seed PPO campaign selection."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .campaign import (
    CampaignConfig,
    aggregate_campaign,
    load_campaign_config,
    verify_campaign_report,
    write_campaign_config,
)
from .experiment_runner import ExperimentRunner, load_experiment_config
from .experiment_summary import summarize_experiment
from .promotion import PromotionConfig, PromotionEvaluator
from .tuning import TuningEvidence, load_sweep_config, materialize_sweep, publish_tuning_evaluation


def _load_config(path: Path) -> CampaignConfig:
    return load_campaign_config(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate immutable PPO trials across training seeds")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init", help="pin campaign gates around an immutable sweep")
    initialize.add_argument("--sweep-config", type=Path, required=True)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--name", default="stage-a-campaign")
    initialize.add_argument("--minimum-seeds", type=int, default=2)
    initialize.add_argument("--minimum-baseline-ci95-low", type=float, default=0.0)
    initialize.add_argument("--maximum-ece", type=float, default=0.10)
    run = subparsers.add_parser("run", help="materialize and run/resume every trial in the campaign")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--trials-dir", type=Path, required=True)
    run.add_argument("--runs-dir", type=Path, required=True)
    run.add_argument("--device")
    evaluate = subparsers.add_parser("evaluate-seal", help="evaluate and seal every completed campaign trial")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--trials-dir", type=Path, required=True)
    evaluate.add_argument("--runs-dir", type=Path, required=True)
    evaluate.add_argument("--evidence-dir", type=Path, required=True)
    evaluate.add_argument("--device")
    status = subparsers.add_parser("status", help="show verified status for every materialized trial")
    status.add_argument("--config", type=Path, required=True)
    status.add_argument("--runs-dir", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate", help="verify the complete matrix and publish selection")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--evidence", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="revalidate a published campaign report")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        config = CampaignConfig(
            sweep=load_sweep_config(args.sweep_config),
            minimum_seeds_per_variant=args.minimum_seeds,
            minimum_baseline_ci95_low=args.minimum_baseline_ci95_low,
            maximum_expected_showdown_share_ece=args.maximum_ece,
            name=args.name,
        )
        write_campaign_config(config, args.output)
        print(f"Wrote {args.output}; campaign={config.campaign_id}")
        return 0
    config = _load_config(args.config)
    if args.command == "run":
        materialized = materialize_sweep(config.sweep, args.trials_dir)
        for trial in materialized:
            experiment = load_experiment_config(trial.experiment_config_path)
            result = ExperimentRunner(experiment, args.runs_dir / trial.spec.trial_id, device=args.device).run(
                install_signal_handlers=True
            )
            print(f"{trial.spec.trial_id}: {result.status} iteration={result.iteration}")
            if result.status in {"manual_interrupt", "aborted_immediate"}:
                return 130
            if result.status != "completed":
                return 2
        return 0
    if args.command == "status":
        incomplete = False
        for trial in config.sweep.expand_trials():
            run_directory = args.runs_dir / trial.trial_id
            if not run_directory.is_dir():
                print(f"{trial.trial_id}: missing")
                incomplete = True
                continue
            summary = summarize_experiment(run_directory)
            if (
                summary.trial_name != trial.trial_id
                or dict(summary.training_config) != trial.config.to_dict()
                or summary.evaluation_protocol_sha256 != trial.evaluation_protocol_sha256
                or summary.code_revision != trial.code_revision
            ):
                raise ValueError(f"campaign run identity does not match trial: {trial.trial_id}")
            print(
                f"{trial.trial_id}: {summary.status} completed={summary.completed} "
                f"alerts={len(summary.alerts)}"
            )
            incomplete = incomplete or not summary.completed
        return 2 if incomplete else 0
    if args.command == "evaluate-seal":
        materialized = materialize_sweep(config.sweep, args.trials_dir)
        protocol = json.loads(Path(config.sweep.evaluation_protocol_path).read_text(encoding="utf-8"))
        promotion_data = protocol.get("promotion_config") if isinstance(protocol, dict) else None
        promotion_protocol = protocol.get("promotion_protocol") if isinstance(protocol, dict) else None
        evaluation_run_seed = (
            promotion_protocol.get("evaluation_run_seed")
            if isinstance(promotion_protocol, dict)
            else None
        )
        if not isinstance(promotion_data, dict):
            raise ValueError("campaign protocol artifact has no promotion_config")
        if isinstance(evaluation_run_seed, bool) or not isinstance(evaluation_run_seed, int) or evaluation_run_seed < 0:
            raise ValueError("campaign protocol artifact has no valid fixed evaluation_run_seed")
        for trial in materialized:
            experiment = load_experiment_config(trial.experiment_config_path)
            run_directory = args.runs_dir / trial.spec.trial_id
            summary = summarize_experiment(run_directory)
            if (
                summary.trial_name != trial.spec.trial_id
                or dict(summary.training_config) != trial.spec.config.to_dict()
                or summary.evaluation_protocol_sha256 != trial.spec.evaluation_protocol_sha256
                or summary.code_revision != trial.spec.code_revision
            ):
                raise ValueError(f"campaign run identity does not match trial: {trial.spec.trial_id}")
            if not summary.completed:
                raise ValueError(f"campaign trial is not complete: {trial.spec.trial_id}")
            runner = ExperimentRunner(experiment, run_directory, device=args.device)
            event = runner.ledger.last_event
            if event is None or event.iteration != experiment.max_iterations:
                raise ValueError(f"campaign trial has no terminal checkpoint: {trial.spec.trial_id}")
            destination = args.evidence_dir / trial.spec.trial_id / "sealed.json"
            if destination.is_file():
                existing = TuningEvidence.from_artifacts(trial.spec, event.checkpoint_path, destination)
                print(
                    f"{trial.spec.trial_id}: existing "
                    f"{'accepted' if existing.passed else 'rejected'} evidence={destination}"
                )
                continue
            evaluation_directory = args.evidence_dir / trial.spec.trial_id / "promotion"
            evaluator = PromotionEvaluator(PromotionConfig(**promotion_data), evaluation_directory, run_seed=evaluation_run_seed)
            evaluated = evaluator.evaluate_and_promote(
                iteration=event.iteration,
                candidate_checkpoint=event.checkpoint_path,
                league=runner.trainer.league,
                stage=runner.trainer.scheduler.stage,
                champion_score=None,
                run_context={
                    "run_config_sha256": trial.spec.run_config_sha256,
                    "evaluation_protocol_sha256": trial.spec.evaluation_protocol_sha256,
                    "evaluation_run_seed": evaluation_run_seed,
                },
            )
            publish_tuning_evaluation(
                trial.spec,
                event.checkpoint_path,
                runner.ledger.manifest_path,
                evaluated.report_path,
                evaluator.archive_manifest_path,
                destination,
            )
            print(f"{trial.spec.trial_id}: {'accepted' if evaluated.accepted else 'rejected'} evidence={destination}")
        return 0
    if args.command == "verify":
        report = verify_campaign_report(config, args.report)
        print(f"Verified {report.config.campaign_id}; winner={None if report.winner is None else report.winner.variant_id}")
        return 0
    evidence: list[TuningEvidence] = []
    for path in args.evidence:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid tuning evidence: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError("tuning evidence must be an object")
        evidence.append(TuningEvidence.from_dict({
            "trial_id": payload.get("trial_id"),
            "full_checkpoint_path": payload.get("full_checkpoint_path"),
            "full_checkpoint_sha256": payload.get("full_checkpoint_sha256"),
            "evaluation_report_path": str(path.resolve()),
            "evaluation_report_sha256": sha256(path.read_bytes()).hexdigest(),
            "evaluation_protocol_sha256": payload.get("evaluation_protocol_sha256"),
            "run_config_sha256": payload.get("run_config_sha256"),
            "score_lower_ci": (payload.get("metrics") or {}).get("score_lower_ci") if isinstance(payload.get("metrics"), dict) else None,
            "expected_showdown_share_ece": (payload.get("metrics") or {}).get("expected_showdown_share_ece") if isinstance(payload.get("metrics"), dict) else None,
            "illegal_action_count": (payload.get("metrics") or {}).get("illegal_action_count") if isinstance(payload.get("metrics"), dict) else None,
            "passed": (payload.get("decision") or {}).get("passed") if isinstance(payload.get("decision"), dict) else None,
        }))
    report = aggregate_campaign(config, evidence)
    report.write_json(args.output)
    print(f"Aggregated {len(evidence)} trials; winner={None if report.winner is None else report.winner.variant_id}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
