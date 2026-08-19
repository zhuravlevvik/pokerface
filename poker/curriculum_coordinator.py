"""Durable, paired evidence for one adjacent curriculum transition.

This module is intentionally independent from :mod:`poker.train_runner`.
It coordinates *completed*, resumable full-run checkpoints produced by a
paired transfer/scratch rung and evaluates them with common-deal multiway
suites.  The runner and evaluator are injected so the orchestration contract
is useful before (and after) either implementation changes.

The public report deliberately contains only checkpoint provenance, fixed
protocol descriptions, aggregate evaluation reports, and gate results.  It
never serialises observations, private cards, action histories, or replay
payloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from .betting import Action, RAISE_ACTIONS
from .curriculum import CurriculumStage, stage_spec
from .multiway_evaluation import (
    MultiwayEvaluationReport,
    PairedMultiwayEvaluation,
    pair_multiway_reports,
)
from .rules import BIG_BLIND


COORDINATOR_MANIFEST_VERSION = 1
COORDINATOR_INTENT_VERSION = 1
COORDINATOR_REPORT_VERSION = 1
EXPECTED_SHOWDOWN_SHARE_PROTOCOL = "active_hands_expected_showdown_share_v1"


@dataclass(frozen=True)
class OpponentSpec:
    """Durable identity of one fixed non-candidate opponent slot.

    Construction of the policy remains with the evaluator callback.  Keeping
    the identity/provenance separate means coordinator artifacts stay
    serialisable and cannot accidentally persist a live bot or player state.
    """

    identity: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("opponent identity must be a non-empty string")
        _json_safe_mapping(self.provenance, "opponent provenance")

    def as_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "provenance": _json_normalize(self.provenance)}


@dataclass(frozen=True)
class EvaluationProtocol:
    """One fixed suite that must pass independently.

    ``protocol`` is an evaluator-owned JSON-safe description (normally a
    ``MultiwayEvaluationConfig.as_dict()``).  The coordinator hashes it into
    its intent, while the returned report's own protocol SHA detects a caller
    that evaluated a different suite.
    """

    name: str
    opponents: tuple[OpponentSpec, ...]
    protocol: Mapping[str, object]
    required_expected_showdown_share_strata: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("evaluation protocol name must be non-empty")
        if not self.opponents:
            raise ValueError("evaluation protocol needs explicit opponent slots")
        identities = [item.identity for item in self.opponents]
        if len(identities) != len(set(identities)):
            raise ValueError("evaluation protocol opponent identities must be unique")
        _json_safe_mapping(self.protocol, "evaluation protocol")
        if any(not isinstance(item, str) or not item for item in self.required_expected_showdown_share_strata):
            raise ValueError("required expected-showdown-share strata must be non-empty strings")
        if len(set(self.required_expected_showdown_share_strata)) != len(self.required_expected_showdown_share_strata):
            raise ValueError("required expected-showdown-share strata must be unique")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "opponents": [item.as_dict() for item in self.opponents],
            "protocol": _json_normalize(self.protocol),
            "required_expected_showdown_share_strata": list(self.required_expected_showdown_share_strata),
        }


@dataclass(frozen=True)
class CurriculumCoordinatorConfig:
    """Immutable evidence contract for exactly one adjacent transition."""

    source_stage: CurriculumStage
    target_stage: CurriculumStage
    source_protocol: EvaluationProtocol
    target_protocols: tuple[EvaluationProtocol, ...]
    paired_rung_protocol_sha256: str
    min_transfer_delta_ci95_low_bb_per_100: float = 0.0
    min_target_baseline_ci95_low_bb_per_100: float = 0.0
    min_source_delta_ci95_low_bb_per_100: float = 0.0
    max_expected_showdown_share_ece: float = 0.08
    max_expected_showdown_share_mae: float = 0.20
    min_required_stratum_samples: int = 1
    max_illegal_actions: int = 0

    def __post_init__(self) -> None:
        source, target = CurriculumStage(self.source_stage), CurriculumStage(self.target_stage)
        stages = tuple(CurriculumStage)
        if stages.index(target) != stages.index(source) + 1:
            raise ValueError("curriculum coordinator only permits one adjacent source -> target transition")
        if not self.target_protocols:
            raise ValueError("at least one target evaluation protocol is required")
        if (
            not isinstance(self.paired_rung_protocol_sha256, str)
            or len(self.paired_rung_protocol_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.paired_rung_protocol_sha256)
        ):
            raise ValueError("paired_rung_protocol_sha256 must be a lowercase SHA-256 digest")
        names = [item.name for item in self.target_protocols]
        if len(names) != len(set(names)):
            raise ValueError("target evaluation protocol names must be unique")
        for value, name in (
            (self.max_expected_showdown_share_ece, "max_expected_showdown_share_ece"),
            (self.max_expected_showdown_share_mae, "max_expected_showdown_share_mae"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_required_stratum_samples < 1:
            raise ValueError("min_required_stratum_samples must be positive")
        if self.max_illegal_actions < 0:
            raise ValueError("max_illegal_actions must be non-negative")
        _validate_protocol_for_stage(self.source_protocol, source)
        for protocol in self.target_protocols:
            _validate_protocol_for_stage(protocol, target)
        required_stacks = set(stage_spec(target).starting_stacks_bb)
        configured_stacks = {
            int(protocol.protocol["starting_stack"]) // BIG_BLIND for protocol in self.target_protocols
        }
        if configured_stacks != required_stacks or len(self.target_protocols) != len(required_stacks):
            raise ValueError("target evaluation protocols must cover every and only target-stage stack variant")
        object.__setattr__(self, "source_stage", source)
        object.__setattr__(self, "target_stage", target)

    def as_dict(self) -> dict[str, object]:
        return {
            "source_stage": self.source_stage.value,
            "target_stage": self.target_stage.value,
            "source_protocol": self.source_protocol.as_dict(),
            "target_protocols": [item.as_dict() for item in self.target_protocols],
            "paired_rung_protocol_sha256": self.paired_rung_protocol_sha256,
            "min_transfer_delta_ci95_low_bb_per_100": self.min_transfer_delta_ci95_low_bb_per_100,
            "min_target_baseline_ci95_low_bb_per_100": self.min_target_baseline_ci95_low_bb_per_100,
            "min_source_delta_ci95_low_bb_per_100": self.min_source_delta_ci95_low_bb_per_100,
            "max_expected_showdown_share_ece": self.max_expected_showdown_share_ece,
            "max_expected_showdown_share_mae": self.max_expected_showdown_share_mae,
            "min_required_stratum_samples": self.min_required_stratum_samples,
            "max_illegal_actions": self.max_illegal_actions,
            "expected_showdown_share_protocol": EXPECTED_SHOWDOWN_SHARE_PROTOCOL,
        }


@dataclass(frozen=True)
class TransitionIntent:
    """The immutable durable plan published before training/evaluation starts."""

    decision_key: str
    config_sha256: str
    source_checkpoint: Path
    source_checkpoint_sha256: str
    reference_checkpoint: Path
    reference_checkpoint_sha256: str
    source_provenance: Mapping[str, object]
    reference_provenance: Mapping[str, object]
    path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "version": COORDINATOR_INTENT_VERSION,
            "decision_key": self.decision_key,
            "config_sha256": self.config_sha256,
            "source": {
                "checkpoint": str(self.source_checkpoint),
                "sha256": self.source_checkpoint_sha256,
                "provenance": dict(self.source_provenance),
            },
            "reference": {
                "checkpoint": str(self.reference_checkpoint),
                "sha256": self.reference_checkpoint_sha256,
                "provenance": dict(self.reference_provenance),
            },
        }


@dataclass(frozen=True)
class PairedRungRequest:
    """Input to an injected native paired-rung implementation."""

    intent: TransitionIntent
    source_checkpoint: Path
    reference_checkpoint: Path
    source_stage: CurriculumStage
    target_stage: CurriculumStage


@dataclass(frozen=True)
class PairedRungArms:
    """Completed native full-run checkpoints for transfer and scratch arms."""

    transfer_checkpoint: Path
    scratch_checkpoint: Path
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        _json_safe_mapping(self.provenance, "paired rung provenance")


@dataclass(frozen=True)
class EvaluationRequest:
    """Input to the injected player-safe multiway evaluator."""

    checkpoint: Path
    stage: CurriculumStage
    protocol: EvaluationProtocol
    role: str


class PairedRungRunner(Protocol):
    def __call__(self, request: PairedRungRequest) -> PairedRungArms: ...


class CheckpointEvaluator(Protocol):
    def __call__(self, request: EvaluationRequest) -> MultiwayEvaluationReport: ...


@dataclass(frozen=True)
class CurriculumTransitionDecision:
    decision_key: str
    accepted: bool
    reasons: tuple[str, ...]
    intent_path: Path
    report_path: Path
    adopted_checkpoint: Path | None


def native_paired_rung_runner(
    rung_config: object,
    run_directory: str | Path,
    *,
    device: str | None = None,
    install_signal_handlers: bool = False,
    checkpoint_loader: Callable[[Path], Mapping[str, object]] | None = None,
) -> PairedRungRunner:
    """Adapt :mod:`poker.paired_rung` without coupling the coordinator API.

    The returned callback uses an intent-keyed native rung directory.  Its
    result must reach the configured paired boundary; partial/interrupted
    rungs are deliberately not admissible transition evidence.
    """

    # Kept local so importing the coordinator does not make the optional RL
    # stack mandatory for callback-injection tests and report inspection.
    from .paired_rung import PairedRungConfig, PairedRungRunner

    if not isinstance(rung_config, PairedRungConfig):
        raise TypeError("rung_config must be PairedRungConfig")
    loader = checkpoint_loader or _load_training_checkpoint
    root = Path(run_directory)

    def run(request: PairedRungRequest) -> PairedRungArms:
        if rung_config.target_stage is not request.target_stage:
            raise ValueError("native paired-rung target stage disagrees with transition intent")
        result = PairedRungRunner(
            rung_config,
            root / request.intent.decision_key,
            request.source_checkpoint,
            device=device,
        ).run(install_signal_handlers=install_signal_handlers)
        if not result.completed or result.iteration != rung_config.iterations:
            raise InterruptedError(
                "paired rung stopped safely at a full-checkpoint boundary; rerun the same curriculum command to resume"
            )
        transfer, scratch = result.transfer.full_checkpoint_path, result.scratch.full_checkpoint_path
        if transfer is None or scratch is None:
            raise RuntimeError("native paired rung did not publish both full arm checkpoints")
        transfer_payload, scratch_payload = loader(transfer), loader(scratch)
        transfer_config, scratch_config = transfer_payload.get("run_config"), scratch_payload.get("run_config")
        if not isinstance(transfer_config, Mapping) or not isinstance(scratch_config, Mapping):
            raise ValueError("native paired rung full checkpoints lack run configuration")
        return PairedRungArms(
            transfer,
            scratch,
            {
                "target_stage": rung_config.target_stage.value,
                "protocol_sha256": result.config_sha256,
                "source_checkpoint_sha256": result.source_checkpoint_sha256,
                "source_run_config_sha256": result.source_run_config_sha256,
                "budget": {
                    "iterations": rung_config.iterations,
                    "hands_per_iteration": rung_config.hands_per_iteration,
                    "table_count": rung_config.table_count,
                },
                "transfer_run_config_sha256": _canonical_sha256(dict(transfer_config)),
                "scratch_run_config_sha256": _canonical_sha256(dict(scratch_config)),
                "native_manifest_path": str(result.manifest_path),
                "native_manifest_sha256": _file_sha256(result.manifest_path),
            },
        )

    return run


class CurriculumCoordinator:
    """Idempotently coordinate one source-stage checkpoint to its next stage."""

    def __init__(
        self,
        config: CurriculumCoordinatorConfig,
        directory: str | Path,
        *,
        rung_runner: PairedRungRunner,
        evaluator: CheckpointEvaluator,
        checkpoint_loader: Callable[[Path], Mapping[str, object]] | None = None,
    ) -> None:
        if not callable(rung_runner) or not callable(evaluator):
            raise TypeError("rung_runner and evaluator must be callable")
        self.config = config
        self.config_sha256 = _canonical_sha256(config.as_dict())
        self.directory = Path(directory)
        self.rung_runner = rung_runner
        self.evaluator = evaluator
        self.checkpoint_loader = checkpoint_loader or _load_training_checkpoint
        self.intent_directory = self.directory / "intents"
        self.input_directory = self.directory / "inputs"
        self.arm_directory = self.directory / "arms"
        self.evaluation_directory = self.directory / "evaluations"
        self.report_directory = self.directory / "reports"
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = self._load_manifest()

    def coordinate(self, source_checkpoint: str | Path, reference_checkpoint: str | Path) -> CurriculumTransitionDecision:
        """Run/recover an immutable paired transition decision.

        An intent is atomically published before the rung callback is invoked.
        A retry with the same source/reference/config returns the previous
        verified decision; an interrupted intent simply resumes its callback
        work.  A rejected decision has no adoption checkpoint.
        """

        source = Path(source_checkpoint)
        reference = Path(reference_checkpoint)
        source_provenance = self._validate_full_checkpoint(source, self.config.source_stage, "source")
        reference_provenance = self._validate_full_checkpoint(reference, self.config.source_stage, "reference")
        if source_provenance["model_metadata_sha256"] != reference_provenance["model_metadata_sha256"]:
            raise ValueError("source and reference checkpoints have incompatible model architecture/metadata")
        if source_provenance["model_state_sha256"] == reference_provenance["model_state_sha256"]:
            raise ValueError("source and reference checkpoints must contain distinct model weights")
        source_sha, reference_sha = _file_sha256(source), _file_sha256(reference)
        if source_sha == reference_sha:
            raise ValueError("source and reference checkpoints must be distinct immutable artifacts")
        config_sha = self.config_sha256
        key = _canonical_sha256(
            {
                "source_stage": self.config.source_stage.value,
                "target_stage": self.config.target_stage.value,
                "source_sha256": source_sha,
                "reference_sha256": reference_sha,
                "config_sha256": config_sha,
            }
        )
        existing = self._existing_decision(key)
        if existing is not None:
            return existing

        source = self._freeze_input(key, "source", source, source_sha)
        reference = self._freeze_input(key, "reference", reference, reference_sha)
        source_provenance = self._validate_full_checkpoint(source, self.config.source_stage, "frozen source")
        reference_provenance = self._validate_full_checkpoint(reference, self.config.source_stage, "frozen reference")
        if source_provenance["model_metadata_sha256"] != reference_provenance["model_metadata_sha256"]:
            raise ValueError("frozen source and reference have incompatible model architecture/metadata")
        if source_provenance["model_state_sha256"] == reference_provenance["model_state_sha256"]:
            raise ValueError("frozen source and reference must contain distinct model weights")

        intent_path = self.intent_directory / f"{key}.json"
        intent = TransitionIntent(
            key,
            config_sha,
            source,
            source_sha,
            reference,
            reference_sha,
            source_provenance,
            reference_provenance,
            intent_path,
        )
        self._write_or_validate_intent(intent)

        arms = self.rung_runner(
            PairedRungRequest(intent, source, reference, self.config.source_stage, self.config.target_stage)
        )
        if not isinstance(arms, PairedRungArms):
            raise TypeError("rung_runner must return PairedRungArms")
        transfer_origin = Path(arms.transfer_checkpoint)
        scratch_origin = Path(arms.scratch_checkpoint)
        transfer_provenance = self._validate_full_checkpoint(transfer_origin, self.config.target_stage, "transfer arm")
        scratch_provenance = self._validate_full_checkpoint(scratch_origin, self.config.target_stage, "scratch arm")
        self._validate_rung_provenance(
            arms.provenance,
            transfer_origin,
            transfer_provenance,
            scratch_origin,
            scratch_provenance,
            intent,
        )
        if any(item["model_metadata_sha256"] != source_provenance["model_metadata_sha256"] for item in (transfer_provenance, scratch_provenance)):
            raise ValueError("paired rung arms have incompatible model architecture/metadata")
        if transfer_provenance["model_state_sha256"] == scratch_provenance["model_state_sha256"]:
            raise ValueError("paired rung transfer and scratch arms must contain distinct model weights")
        transfer = self._freeze_arm(key, "transfer", transfer_origin)
        scratch = self._freeze_arm(key, "scratch", scratch_origin)
        # A frozen full runner checkpoint, rather than model-only weights, is
        # the only object which may be adopted after acceptance.
        self._validate_full_checkpoint(transfer, self.config.target_stage, "frozen transfer arm")
        self._validate_full_checkpoint(scratch, self.config.target_stage, "frozen scratch arm")

        target_rows: list[dict[str, object]] = []
        reasons: list[str] = []
        for protocol in self.config.target_protocols:
            transfer_report = self._evaluate_and_write(key, "target_transfer", transfer, self.config.target_stage, protocol)
            scratch_report = self._evaluate_and_write(key, "target_scratch", scratch, self.config.target_stage, protocol)
            paired = pair_multiway_reports(transfer_report[0], scratch_report[0])
            suite_reasons = self._target_gate_reasons(protocol, transfer_report[0], scratch_report[0], paired)
            reasons.extend(f"target[{protocol.name}]: {reason}" for reason in suite_reasons)
            target_rows.append(
                {
                    "protocol": protocol.as_dict(),
                    "transfer": transfer_report[1],
                    "scratch": scratch_report[1],
                    "paired": paired.as_dict(),
                    "paired_sha256": _canonical_sha256(paired.as_dict()),
                    "gate_reasons": suite_reasons,
                }
            )

        source_candidate = self._evaluate_and_write(key, "source_candidate", source, self.config.source_stage, self.config.source_protocol)
        source_reference = self._evaluate_and_write(key, "source_reference", reference, self.config.source_stage, self.config.source_protocol)
        source_paired = pair_multiway_reports(source_candidate[0], source_reference[0])
        source_reasons = self._source_gate_reasons(source_candidate[0], source_reference[0], source_paired)
        reasons.extend(f"source: {reason}" for reason in source_reasons)
        accepted = not reasons

        report_path = self.report_directory / f"{key}.json"
        report = {
            "version": COORDINATOR_REPORT_VERSION,
            "decision_key": key,
            "config": self.config.as_dict(),
            "config_sha256": config_sha,
            "intent_path": str(intent_path),
            "intent_sha256": _file_sha256(intent_path),
            "source": _checkpoint_record(source, source_sha, source_provenance),
            "reference": _checkpoint_record(reference, reference_sha, reference_provenance),
            "arms": {
                "rung_provenance": dict(arms.provenance),
                "transfer_origin": _checkpoint_record(transfer_origin, _file_sha256(transfer_origin), transfer_provenance),
                "scratch_origin": _checkpoint_record(scratch_origin, _file_sha256(scratch_origin), scratch_provenance),
                "transfer": _checkpoint_record(transfer, _file_sha256(transfer), self._validate_full_checkpoint(transfer, self.config.target_stage, "transfer")),
                "scratch": _checkpoint_record(scratch, _file_sha256(scratch), self._validate_full_checkpoint(scratch, self.config.target_stage, "scratch")),
            },
            "target_evaluations": target_rows,
            "source_evaluation": {
                "protocol": self.config.source_protocol.as_dict(),
                "candidate": source_candidate[1],
                "reference": source_reference[1],
                "paired": source_paired.as_dict(),
                "paired_sha256": _canonical_sha256(source_paired.as_dict()),
                "gate_reasons": source_reasons,
            },
            "decision": {
                "accepted": accepted,
                "reasons": reasons,
                "adopted_checkpoint": str(transfer) if accepted else None,
                "adopted_checkpoint_sha256": _file_sha256(transfer) if accepted else None,
            },
        }
        _atomic_write_json(report_path, report)
        record = {
            "decision_key": key,
            "intent_path": str(intent_path),
            "intent_sha256": _file_sha256(intent_path),
            "report_path": str(report_path),
            "report_sha256": _file_sha256(report_path),
            "accepted": accepted,
            "adopted_checkpoint": str(transfer) if accepted else None,
            "adopted_checkpoint_sha256": _file_sha256(transfer) if accepted else None,
        }
        self._record(record)
        return CurriculumTransitionDecision(key, accepted, tuple(reasons), intent_path, report_path, transfer if accepted else None)

    def _target_gate_reasons(
        self,
        protocol: EvaluationProtocol,
        transfer: MultiwayEvaluationReport,
        scratch: MultiwayEvaluationReport,
        paired: PairedMultiwayEvaluation,
    ) -> list[str]:
        reasons = self._report_contract_reasons(transfer, protocol, "transfer")
        reasons.extend(self._report_contract_reasons(scratch, protocol, "scratch"))
        if paired.delta_bb_per_100_ci95_low < self.config.min_transfer_delta_ci95_low_bb_per_100:
            reasons.append("transfer-minus-scratch paired CI lower bound is below its configured floor")
        if transfer.bb_per_100_ci95_low < self.config.min_target_baseline_ci95_low_bb_per_100:
            reasons.append("target baseline CI lower bound is below its configured floor")
        return reasons

    def _source_gate_reasons(
        self,
        candidate: MultiwayEvaluationReport,
        reference: MultiwayEvaluationReport,
        paired: PairedMultiwayEvaluation,
    ) -> list[str]:
        reasons = self._report_contract_reasons(candidate, self.config.source_protocol, "candidate")
        reasons.extend(self._report_contract_reasons(reference, self.config.source_protocol, "reference"))
        if paired.delta_bb_per_100_ci95_low < self.config.min_source_delta_ci95_low_bb_per_100:
            reasons.append("source candidate-minus-reference paired CI lower bound is below its configured regression floor")
        return reasons

    def _report_contract_reasons(
        self, report: MultiwayEvaluationReport, protocol: EvaluationProtocol, label: str
    ) -> list[str]:
        reasons: list[str] = []
        expected_player_count = stage_spec(self.config.source_stage if label in {"candidate", "reference"} else self.config.target_stage).player_count
        if report.config.player_count != expected_player_count:
            reasons.append(f"{label} report player count does not match stage")
        if report.config.as_dict() != dict(protocol.protocol):
            reasons.append(f"{label} report config does not match frozen protocol")
        if tuple(report.opponent_slots) != tuple(item.identity for item in protocol.opponents):
            reasons.append(f"{label} report opponent slots do not match frozen protocol")
        diagnostics = report.model_diagnostics
        if diagnostics.illegal_action_count > self.config.max_illegal_actions:
            reasons.append(f"{label} report exceeds illegal-action limit")
        metric = diagnostics.expected_showdown_share
        if metric is None:
            reasons.append(f"{label} report has no expected-showdown-share diagnostics")
            return reasons
        if metric.expected_calibration_error > self.config.max_expected_showdown_share_ece:
            reasons.append(f"{label} expected-showdown-share ECE exceeds limit")
        if metric.mean_absolute_error > self.config.max_expected_showdown_share_mae:
            reasons.append(f"{label} expected-showdown-share MAE exceeds limit")
        for stratum in protocol.required_expected_showdown_share_strata:
            support = diagnostics.expected_showdown_share_support.get(stratum, 0)
            if support < self.config.min_required_stratum_samples:
                reasons.append(f"{label} required stratum {stratum!r} has insufficient support")
                continue
            stratum_metric = diagnostics.expected_showdown_share_by_stratum.get(stratum)
            if stratum_metric is None:
                reasons.append(f"{label} required stratum {stratum!r} has no calibration metric")
            elif stratum_metric.expected_calibration_error > self.config.max_expected_showdown_share_ece:
                reasons.append(f"{label} required stratum {stratum!r} ECE exceeds limit")
            elif stratum_metric.mean_absolute_error > self.config.max_expected_showdown_share_mae:
                reasons.append(f"{label} required stratum {stratum!r} MAE exceeds limit")
        return reasons

    def _evaluate_and_write(
        self, key: str, role: str, checkpoint: Path, stage: CurriculumStage, protocol: EvaluationProtocol
    ) -> tuple[MultiwayEvaluationReport, dict[str, object]]:
        report = self.evaluator(EvaluationRequest(checkpoint, stage, protocol, role))
        if not isinstance(report, MultiwayEvaluationReport):
            raise TypeError("evaluator must return MultiwayEvaluationReport")
        path = self.evaluation_directory / f"{key}_{role}_{_safe_component(protocol.name)}_{_canonical_sha256(protocol.as_dict())[:12]}.json"
        payload = report.as_dict()
        if path.exists():
            if _load_json(path) != payload:
                raise ValueError("evaluation artifact already exists with divergent immutable contents")
        else:
            report.write_json(path)
        # The report's own schema is designed as a player-safe aggregate.  Do
        # not accept arbitrary callback metadata in place of it.
        return report, {"path": str(path), "sha256": _file_sha256(path), "protocol_sha256": report.protocol_sha256, "report": payload}

    def _validate_rung_provenance(
        self,
        provenance: Mapping[str, object],
        transfer_path: Path,
        transfer: Mapping[str, object],
        scratch_path: Path,
        scratch: Mapping[str, object],
        intent: TransitionIntent,
    ) -> None:
        """Require completed paired-rung budget/protocol evidence, fail closed."""

        required = (
            "target_stage",
            "protocol_sha256",
            "source_checkpoint_sha256",
            "source_run_config_sha256",
            "budget",
            "transfer_run_config_sha256",
            "scratch_run_config_sha256",
            "native_manifest_path",
            "native_manifest_sha256",
        )
        if any(key not in provenance for key in required):
            raise ValueError("paired rung provenance lacks mandatory budget/protocol fields")
        if provenance["target_stage"] != self.config.target_stage.value:
            raise ValueError("paired rung provenance target stage does not match coordinator target")
        if provenance["source_checkpoint_sha256"] != intent.source_checkpoint_sha256:
            raise ValueError("paired rung source checkpoint does not match transition intent")
        if provenance["source_run_config_sha256"] != intent.source_provenance["run_config_sha256"]:
            raise ValueError("paired rung source run configuration does not match transition intent")
        protocol_sha = provenance["protocol_sha256"]
        if protocol_sha != self.config.paired_rung_protocol_sha256:
            raise ValueError("paired rung provenance does not match the configured rung protocol SHA-256")
        budget = provenance["budget"]
        if not isinstance(budget, Mapping) or any(
            not isinstance(budget.get(key), int) or isinstance(budget.get(key), bool) or budget.get(key, 0) < 1
            for key in ("iterations", "hands_per_iteration", "table_count")
        ):
            raise ValueError("paired rung provenance has invalid completed budget")
        manifest_path = provenance.get("native_manifest_path")
        manifest_sha = provenance.get("native_manifest_sha256")
        if (
            not isinstance(manifest_path, str)
            or not isinstance(manifest_sha, str)
            or not Path(manifest_path).is_file()
            or _file_sha256(Path(manifest_path)) != manifest_sha
        ):
            raise ValueError("paired rung native manifest is missing or hash-mismatched")
        native_manifest = _load_json(Path(manifest_path))
        if (
            native_manifest.get("config_sha256") != protocol_sha
            or native_manifest.get("source_checkpoint_sha256") != intent.source_checkpoint_sha256
            or native_manifest.get("completed") is not True
        ):
            raise ValueError("paired rung native manifest provenance does not match transition intent")
        native_config = native_manifest.get("config")
        if not isinstance(native_config, Mapping) or _canonical_sha256(dict(native_config)) != protocol_sha:
            raise ValueError("paired rung native manifest config does not match its pinned hash")
        for field in ("iterations", "hands_per_iteration", "table_count"):
            if native_config.get(field) != budget.get(field):
                raise ValueError("paired rung native manifest budget does not match callback provenance")
        native_arms = native_manifest.get("arms")
        if not isinstance(native_arms, Mapping):
            raise ValueError("paired rung native manifest has malformed arm records")
        for label, path, checkpoint in (
            ("transfer", transfer_path, transfer),
            ("scratch", scratch_path, scratch),
        ):
            record = native_arms.get(label)
            if (
                not isinstance(record, Mapping)
                or record.get("full_checkpoint") != str(path)
                or record.get("full_checkpoint_sha256") != _file_sha256(path)
                or record.get("iteration") != checkpoint["iteration"]
                or record.get("global_hands") != checkpoint["global_hands"]
                or record.get("global_decisions") != checkpoint["global_decisions"]
            ):
                raise ValueError(f"paired rung native manifest does not bind the returned {label} checkpoint")
        for label, checkpoint in (("transfer", transfer), ("scratch", scratch)):
            if provenance.get(f"{label}_run_config_sha256") != checkpoint["run_config_sha256"]:
                raise ValueError(f"paired rung {label} config SHA does not match its full checkpoint")
            iteration = checkpoint.get("iteration")
            hands = checkpoint.get("global_hands")
            if (
                iteration != budget["iterations"]
                or hands != budget["iterations"] * budget["hands_per_iteration"]
                or checkpoint.get("hands_per_iteration") != budget["hands_per_iteration"]
                or checkpoint.get("table_count") != budget["table_count"]
            ):
                raise ValueError(f"paired rung {label} full checkpoint does not satisfy the exact completed budget")

    def _freeze_arm(self, key: str, label: str, source: Path) -> Path:
        source_sha = _file_sha256(source)
        destination = self.arm_directory / key / label / "checkpoints" / f"paired_rung_{source_sha[:12]}.pt"
        if destination.exists():
            if _file_sha256(destination) != source_sha:
                raise ValueError("existing frozen rung arm hash does not match its source")
            return destination
        _atomic_copy(source, destination)
        if _file_sha256(source) != source_sha or _file_sha256(destination) != source_sha:
            raise ValueError("rung arm changed while being frozen")
        return destination

    def _freeze_input(self, key: str, label: str, source: Path, expected_sha: str) -> Path:
        destination = self.input_directory / key / f"{label}-{expected_sha[:12]}.pt"
        if destination.exists():
            if _file_sha256(destination) != expected_sha:
                raise ValueError(f"existing frozen {label} checkpoint is hash-mismatched")
            return destination
        _atomic_copy(source, destination)
        if _file_sha256(source) != expected_sha or _file_sha256(destination) != expected_sha:
            raise ValueError(f"{label} checkpoint changed while being frozen")
        return destination

    def _validate_full_checkpoint(self, path: Path, stage: CurriculumStage, label: str) -> dict[str, object]:
        if not path.is_file():
            raise ValueError(f"{label} checkpoint is missing: {path}")
        payload = self.checkpoint_loader(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label} checkpoint loader returned a non-mapping")
        required = ("checkpoint_version", "metadata", "state_dict", "optimizer_state_dict", "run_config", "curriculum", "progress", "league", "rng")
        if any(key not in payload for key in required):
            raise ValueError(f"{label} must be a native full TrainingRunner checkpoint")
        from .train_runner import CHECKPOINT_VERSION

        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError(f"{label} has an incompatible TrainingRunner checkpoint version")
        metadata, state_dict, curriculum, run_config, progress = (
            payload.get("metadata"), payload.get("state_dict"), payload.get("curriculum"), payload.get("run_config"), payload.get("progress")
        )
        if not all(isinstance(item, Mapping) for item in (metadata, state_dict, curriculum, run_config, progress)):
            raise ValueError(f"{label} full checkpoint has malformed provenance")
        if curriculum.get("stage") != stage.value:
            raise ValueError(f"{label} checkpoint stage must be {stage.value}")
        model_version = metadata.get("model_version")
        model_config = metadata.get("config")
        run_settings = run_config.get("run")
        if not isinstance(model_version, str) or not isinstance(model_config, Mapping):
            raise ValueError(f"{label} checkpoint lacks model provenance")
        if not isinstance(run_settings, Mapping):
            raise ValueError(f"{label} checkpoint lacks native run settings")
        return {
            "checkpoint_version": payload.get("checkpoint_version"),
            "stage": stage.value,
            "model_version": model_version,
            "model_metadata_sha256": _canonical_sha256(dict(metadata)),
            "model_state_sha256": _model_state_sha256(state_dict),
            "run_config_sha256": _canonical_sha256(dict(run_config)),
            "iteration": progress.get("iteration"),
            "global_hands": progress.get("global_hands"),
            "global_decisions": progress.get("global_decisions"),
            "hands_per_iteration": run_settings.get("hands_per_iteration"),
            "table_count": run_settings.get("table_count"),
        }

    def _write_or_validate_intent(self, intent: TransitionIntent) -> None:
        expected = intent.as_dict()
        if intent.path.exists():
            actual = _load_json(intent.path)
            if actual != expected:
                raise ValueError("existing transition intent disagrees with requested immutable provenance")
            return
        _atomic_write_json(intent.path, expected)

    def _existing_decision(self, key: str) -> CurriculumTransitionDecision | None:
        records = self.manifest["decisions"]
        assert isinstance(records, list)
        record = next((item for item in records if item.get("decision_key") == key), None)
        if record is None:
            return None
        self._validate_record(record)
        report = _load_json(Path(str(record["report_path"])))
        decision = report.get("decision")
        if not isinstance(decision, Mapping):
            raise ValueError("coordinator report has malformed decision")
        accepted = bool(decision.get("accepted"))
        adopted_value = decision.get("adopted_checkpoint")
        adopted = Path(str(adopted_value)) if accepted and isinstance(adopted_value, str) else None
        return CurriculumTransitionDecision(
            key,
            accepted,
            tuple(str(item) for item in decision.get("reasons", ())),
            Path(str(record["intent_path"])),
            Path(str(record["report_path"])),
            adopted,
        )

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {
                "version": COORDINATOR_MANIFEST_VERSION,
                "config_sha256": self.config_sha256,
                "decisions": [],
            }
        manifest = _load_json(self.manifest_path)
        if (
            manifest.get("version") != COORDINATOR_MANIFEST_VERSION
            or manifest.get("config_sha256") != self.config_sha256
            or not isinstance(manifest.get("decisions"), list)
        ):
            raise ValueError("incompatible or malformed curriculum coordinator manifest")
        for record in manifest["decisions"]:
            if not isinstance(record, Mapping):
                raise ValueError("coordinator manifest contains malformed decision")
            self._validate_record(record)
        return manifest

    def _validate_record(self, record: Mapping[str, object]) -> None:
        for path_key, hash_key in (("intent_path", "intent_sha256"), ("report_path", "report_sha256")):
            value, digest = record.get(path_key), record.get(hash_key)
            path = Path(str(value))
            if not isinstance(value, str) or not isinstance(digest, str) or not path.is_file() or _file_sha256(path) != digest:
                raise ValueError(f"coordinator manifest {path_key} is missing or hash-mismatched")
        report = _load_json(Path(str(record["report_path"])))
        if (
            report.get("version") != COORDINATOR_REPORT_VERSION
            or report.get("config_sha256") != self.config_sha256
            or report.get("decision_key") != record.get("decision_key")
            or report.get("intent_sha256") != record.get("intent_sha256")
        ):
            raise ValueError("coordinator report and manifest provenance disagree")
        intent = _load_json(Path(str(record["intent_path"])))
        if (
            intent.get("version") != COORDINATOR_INTENT_VERSION
            or intent.get("decision_key") != record.get("decision_key")
            or intent.get("config_sha256") != self.config_sha256
        ):
            raise ValueError("coordinator intent and manifest provenance disagree")
        for label in ("source", "reference"):
            intent_item, report_item = intent.get(label), report.get(label)
            if not isinstance(intent_item, Mapping) or not isinstance(report_item, Mapping):
                raise ValueError("coordinator intent/report checkpoint provenance is malformed")
            if intent_item.get("checkpoint") != report_item.get("path") or intent_item.get("sha256") != report_item.get("sha256"):
                raise ValueError("coordinator intent and report checkpoint provenance disagree")
        decision = report.get("decision")
        if not isinstance(decision, Mapping) or bool(decision.get("accepted")) != bool(record.get("accepted")):
            raise ValueError("coordinator report and manifest decision disagree")
        expected_paths = _artifact_paths(report)
        for path, digest in expected_paths:
            if not path.is_file() or _file_sha256(path) != digest:
                raise ValueError("coordinator artifact is missing or hash-mismatched")
        _validate_embedded_evaluation_reports(report)
        accepted = bool(record.get("accepted"))
        adopted, adopted_sha = record.get("adopted_checkpoint"), record.get("adopted_checkpoint_sha256")
        if accepted:
            if not isinstance(adopted, str) or not isinstance(adopted_sha, str) or _file_sha256(Path(adopted)) != adopted_sha:
                raise ValueError("accepted coordinator adoption checkpoint is missing or hash-mismatched")
            if decision.get("adopted_checkpoint") != adopted or decision.get("adopted_checkpoint_sha256") != adopted_sha:
                raise ValueError("accepted coordinator adoption does not match report")
        elif adopted is not None or adopted_sha is not None or decision.get("adopted_checkpoint") is not None:
            raise ValueError("rejected coordinator decision must never publish an adoption checkpoint")

    def _record(self, record: Mapping[str, object]) -> None:
        decisions = self.manifest["decisions"]
        assert isinstance(decisions, list)
        existing = next((item for item in decisions if item.get("decision_key") == record["decision_key"]), None)
        if existing is not None:
            if dict(existing) != dict(record):
                raise ValueError("retry produced a different immutable coordinator decision")
            return
        decisions.append(dict(record))
        _atomic_write_json(self.manifest_path, self.manifest)


def _checkpoint_record(path: Path, digest: str, provenance: Mapping[str, object]) -> dict[str, object]:
    return {"path": str(path), "sha256": digest, "provenance": dict(provenance)}


def _validate_protocol_for_stage(protocol: EvaluationProtocol, stage: CurriculumStage) -> None:
    spec = stage_spec(stage)
    value = protocol.protocol
    try:
        player_count = int(value["player_count"])
        starting_stack = int(value["starting_stack"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation protocol lacks stage game settings") from error
    if player_count != spec.player_count or len(protocol.opponents) != player_count - 1:
        raise ValueError("evaluation protocol player/opponent count does not match its curriculum stage")
    allowed_value = value.get("allowed_raise_actions")
    if allowed_value is None:
        allowed = RAISE_ACTIONS | {Action.ALL_IN}
    elif isinstance(allowed_value, Sequence) and not isinstance(allowed_value, (str, bytes)):
        try:
            allowed = frozenset(Action(item) for item in allowed_value)
        except ValueError as error:
            raise ValueError("evaluation protocol has invalid raise actions") from error
    else:
        raise ValueError("evaluation protocol has invalid raise actions")
    if frozenset(allowed) != spec.allowed_raise_actions:
        raise ValueError("evaluation protocol action abstraction does not match its curriculum stage")
    if starting_stack % BIG_BLIND or starting_stack // BIG_BLIND not in spec.starting_stacks_bb:
        raise ValueError("evaluation protocol starting stack does not match its curriculum stage")
    configured_strata = value.get("required_expected_showdown_share_strata", ())
    if tuple(configured_strata) != protocol.required_expected_showdown_share_strata:
        raise ValueError("evaluation protocol required strata disagree with its evaluator config")


def _artifact_paths(report: Mapping[str, object]) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    intent_path, intent_sha = report.get("intent_path"), report.get("intent_sha256")
    if not isinstance(intent_path, str) or not isinstance(intent_sha, str):
        raise ValueError("coordinator report has malformed intent reference")
    result.append((Path(intent_path), intent_sha))
    for label in ("source", "reference"):
        item = report.get(label)
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("coordinator report has malformed frozen input artifact")
        result.append((Path(str(item["path"])), str(item["sha256"])))
    arms = report.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("coordinator report has malformed arms")
    for label in ("transfer_origin", "scratch_origin", "transfer", "scratch"):
        item = arms.get(label)
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("coordinator report has malformed arm artifact")
        result.append((Path(str(item["path"])), str(item["sha256"])))
    rung_provenance = arms.get("rung_provenance")
    if not isinstance(rung_provenance, Mapping):
        raise ValueError("coordinator report has malformed paired-rung provenance")
    native_path, native_sha = rung_provenance.get("native_manifest_path"), rung_provenance.get("native_manifest_sha256")
    if not isinstance(native_path, str) or not isinstance(native_sha, str):
        raise ValueError("coordinator report has malformed native paired-rung manifest")
    result.append((Path(native_path), native_sha))
    target_rows = report.get("target_evaluations")
    source_row = report.get("source_evaluation")
    if not isinstance(target_rows, list) or not isinstance(source_row, Mapping):
        raise ValueError("coordinator report has malformed evaluation artifacts")
    for row in [*target_rows, source_row]:
        if not isinstance(row, Mapping):
            raise ValueError("coordinator report has malformed evaluation row")
        labels = ("transfer", "scratch") if row is not source_row else ("candidate", "reference")
        for label in labels:
            item = row.get(label)
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                raise ValueError("coordinator report has malformed evaluation artifact")
            result.append((Path(str(item["path"])), str(item["sha256"])))
    return result


def _validate_embedded_evaluation_reports(report: Mapping[str, object]) -> None:
    """Bind each report-summary embedded in the decision to its JSON artifact."""

    rows = report.get("target_evaluations")
    source = report.get("source_evaluation")
    assert isinstance(rows, list) and isinstance(source, Mapping)  # validated by _artifact_paths.
    for row in [*rows, source]:
        assert isinstance(row, Mapping)
        labels = ("candidate", "reference") if row is source else ("transfer", "scratch")
        for label in labels:
            item = row[label]
            assert isinstance(item, Mapping)
            path = Path(str(item["path"]))
            embedded = item.get("report")
            if not isinstance(embedded, Mapping) or _load_json(path) != dict(embedded):
                raise ValueError("coordinator embedded evaluation report disagrees with evaluation artifact")


def _load_training_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - non-RL install.
        raise RuntimeError("PyTorch is required to validate full TrainingRunner checkpoints") from error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def _model_state_sha256(state_dict: Mapping[str, object]) -> str:
    """Hash model tensors independently from optimizer/RNG checkpoint state."""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - non-RL install.
        raise RuntimeError("PyTorch is required to hash full TrainingRunner checkpoints") from error
    buffer = BytesIO()
    torch.save(dict(sorted(state_dict.items())), buffer, _use_new_zipfile_serialization=False)
    return sha256(buffer.getvalue()).hexdigest()


def _json_safe_mapping(value: Mapping[str, object], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        json.dumps(dict(value), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-safe") from error


def _json_normalize(value: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical JSON shape (notably tuples become arrays)."""

    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    """Return a conservative filename component; retain a hash beside it."""

    rendered = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return rendered[:80] or "protocol"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON coordinator artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON coordinator artifact must be an object: {path}")
    return value


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        _publish_file(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        _publish_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_file(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    try:
        descriptor = os.open(destination.parent, os.O_RDONLY)
    except OSError:  # pragma: no cover - filesystem-specific fallback.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "COORDINATOR_INTENT_VERSION",
    "COORDINATOR_MANIFEST_VERSION",
    "COORDINATOR_REPORT_VERSION",
    "CheckpointEvaluator",
    "CurriculumCoordinator",
    "CurriculumCoordinatorConfig",
    "CurriculumTransitionDecision",
    "EvaluationProtocol",
    "EvaluationRequest",
    "OpponentSpec",
    "PairedRungArms",
    "PairedRungRequest",
    "PairedRungRunner",
    "TransitionIntent",
    "native_paired_rung_runner",
]
