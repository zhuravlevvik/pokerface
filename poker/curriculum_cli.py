"""CLI for one durable paired curriculum transition."""

from __future__ import annotations

import argparse
from pathlib import Path

from .curriculum_coordinator import CurriculumCoordinator, native_paired_rung_runner
from .curriculum_runtime import (
    default_curriculum_job_config,
    load_curriculum_job_config,
    native_multiway_evaluator,
    write_curriculum_job_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume a paired transfer-vs-scratch curriculum gate")
    parser.add_argument("--config", type=Path, help="JSON paired-curriculum job configuration")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/curriculum"), help="durable job directory")
    parser.add_argument("--source-checkpoint", type=Path, help="full checkpoint to transfer and evaluate")
    parser.add_argument("--reference-checkpoint", type=Path, help="distinct full source-stage regression checkpoint")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--write-default-config", type=Path, help="write a B-to-C starter JSON config and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.write_default_config is not None:
        write_curriculum_job_config(default_curriculum_job_config(), args.write_default_config)
        print(f"Wrote {args.write_default_config}")
        return 0
    if args.config is None or args.source_checkpoint is None or args.reference_checkpoint is None:
        parser.error("--config, --source-checkpoint and --reference-checkpoint are required")
    job = load_curriculum_job_config(args.config)
    coordinator = CurriculumCoordinator(
        job.coordinator,
        args.run_dir / "coordinator",
        rung_runner=native_paired_rung_runner(
            job.paired_rung,
            args.run_dir / "paired-rungs",
            device=args.device,
            install_signal_handlers=True,
        ),
        evaluator=native_multiway_evaluator(),
    )
    try:
        decision = coordinator.coordinate(args.source_checkpoint, args.reference_checkpoint)
    except (InterruptedError, KeyboardInterrupt):
        print("Curriculum run interrupted safely; repeat the same command to resume.")
        return 130
    status = "accepted" if decision.accepted else "rejected"
    print(f"Transition {status}: report={decision.report_path} adopted={decision.adopted_checkpoint}")
    if decision.reasons:
        print("Reasons: " + "; ".join(decision.reasons))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
