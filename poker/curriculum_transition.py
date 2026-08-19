"""Durable evidence artifacts for the bounded heads-up A -> B transition.

The class in this module deliberately does not mutate a live
``TrainingRunner``.  A runner can ask whether a transition is due and submit a
completed full-run checkpoint; this layer then freezes the source, evaluates
it, records immutable evidence, and (only on acceptance) publishes a
model-only B-stage transfer artifact.  Keeping those side effects outside the
runner makes retries and crash recovery auditable and idempotent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .bots import AggroBot, CallingStationBot, RandomBot, RuleBot, TightBot
from .curriculum import CurriculumConfig, CurriculumStage, StageEvaluation, checkpoint_curriculum_metadata, stage_spec
from .evaluation import EvaluationConfig, EvaluationSuiteReport, MatchupReport, evaluate_suite
from .league import ModelPolicy, OpponentLeague
from .model import TORCH_AVAILABLE, PokerAgentModel

if TORCH_AVAILABLE:
    import torch


TRANSITION_MANIFEST_VERSION = 2
TRANSITION_REPORT_VERSION = 2
HU_BASELINES = ("rule", "tight", "aggro", "calling_station", "random")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for curriculum transition evidence; install the project with `.[rl]`.")


@dataclass(frozen=True)
class CurriculumTransitionConfig:
    """Opt-in, serialisable fixed protocol for automatic A -> B evidence.

    ``curriculum`` is kept explicit because ``StageEvaluation`` is the source
    of truth for acceptance, including its intentionally strict optional
    transfer-vs-scratch requirement.  This bounded layer does not fabricate a
    scratch control result; callers that require one receive a rejected
    evidence record until such an experiment supplies it.
    """

    enabled: bool = False
    every_iterations: int = 5
    evaluate_on_complete: bool = True
    hands_per_opponent: int = 200
    seed_start: int = 3_000_000
    equity_samples: int = 32
    calibration_bins: int = 10
    baseline_bots: tuple[str, ...] = HU_BASELINES
    minimum_baseline_ci95_low: float = -10_000.0
    maximum_baseline_ci95_half_width: float = 10_000.0
    minimum_prior_ci95_low: float = 0.0
    reference_checkpoint: str | None = None
    reference_checkpoint_sha256: str | None = None
    reset_optimizer: bool = True
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    source_stage: CurriculumStage = CurriculumStage.A_HEADS_UP_STARTER
    target_stage: CurriculumStage = CurriculumStage.B_HEADS_UP_FULL

    def __post_init__(self) -> None:
        if self.source_stage is not CurriculumStage.A_HEADS_UP_STARTER or self.target_stage is not CurriculumStage.B_HEADS_UP_FULL:
            raise ValueError("automatic curriculum transition currently supports only A -> B")
        if self.every_iterations < 1:
            raise ValueError("every_iterations must be positive")
        if self.hands_per_opponent < 2 or self.hands_per_opponent % 2:
            raise ValueError("hands_per_opponent must be an even heads-up position-rotation budget")
        if self.equity_samples < 1 or self.calibration_bins < 1:
            raise ValueError("equity_samples and calibration_bins must be positive")
        if not self.baseline_bots or any(name not in HU_BASELINES for name in self.baseline_bots):
            raise ValueError(f"baseline_bots must be a non-empty subset of {HU_BASELINES}")
        if len(self.baseline_bots) != len(set(self.baseline_bots)):
            raise ValueError("baseline_bots must be unique")
        if self.maximum_baseline_ci95_half_width <= 0:
            raise ValueError("maximum_baseline_ci95_half_width must be positive")
        if any(
            not isfinite(value)
            for value in (
                self.minimum_baseline_ci95_low,
                self.maximum_baseline_ci95_half_width,
                self.minimum_prior_ci95_low,
            )
        ):
            raise ValueError("curriculum transition gates must be finite")
        if self.enabled and (not isinstance(self.reference_checkpoint, str) or not self.reference_checkpoint.strip()):
            raise ValueError("enabled A -> B transition requires a non-empty reference_checkpoint")
        if self.enabled and (
            not isinstance(self.reference_checkpoint_sha256, str)
            or len(self.reference_checkpoint_sha256) != 64
            or self.reference_checkpoint_sha256 != self.reference_checkpoint_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.reference_checkpoint_sha256.lower())
        ):
            raise ValueError("enabled A -> B transition requires a pinned reference_checkpoint_sha256")


@dataclass(frozen=True)
class TransitionEvidence:
    """A serialisable mapping of one fixed suite onto curriculum gates."""

    stage_evaluation: StageEvaluation
    baseline_bb_per_100: float
    max_equity_ece: float
    prior_ci95_low: float
    sanity_passed: bool
    illegal_action_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_evaluation": asdict(self.stage_evaluation),
            "baseline_bb_per_100": self.baseline_bb_per_100,
            "max_equity_ece": self.max_equity_ece,
            "prior_ci95_low": self.prior_ci95_low,
            "sanity_passed": self.sanity_passed,
            "illegal_action_count": self.illegal_action_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionEvidence":
        stage = value.get("stage_evaluation")
        if not isinstance(stage, Mapping):
            raise ValueError("transition evidence has no stage evaluation")
        try:
            evaluation = StageEvaluation(**dict(stage))
            return cls(
                evaluation,
                float(value["baseline_bb_per_100"]),
                float(value["max_equity_ece"]),
                float(value["prior_ci95_low"]),
                bool(value["sanity_passed"]),
                int(value["illegal_action_count"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("transition evidence is malformed") from error


@dataclass(frozen=True)
class TransitionEvaluation:
    """One idempotent A -> B decision plus its immutable artifacts."""

    iteration: int
    accepted: bool
    reasons: tuple[str, ...]
    evidence: TransitionEvidence
    report_path: Path
    transfer_checkpoint_path: Path | None
    frozen_source_path: Path

    @property
    def checkpoint_path(self) -> Path | None:
        """Compatibility-friendly alias for the accepted transfer artifact."""

        return self.transfer_checkpoint_path

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "evidence": self.evidence.as_dict(),
            "report_path": str(self.report_path),
            "transfer_checkpoint_path": None if self.transfer_checkpoint_path is None else str(self.transfer_checkpoint_path),
            "frozen_source_path": str(self.frozen_source_path),
        }


class CurriculumTransitionEvaluator:
    """Freeze, evaluate and transfer A-stage candidates without live mutation."""

    def __init__(self, config: CurriculumTransitionConfig, run_directory: str | Path, *, run_seed: int = 0) -> None:
        self.config = config
        self.run_directory = Path(run_directory)
        self.run_seed = run_seed
        self.directory = self.run_directory / "curriculum-transitions"
        self.candidate_directory = self.directory / "candidates"
        self.report_directory = self.directory / "reports"
        self.transfer_directory = self.directory / "transfers"
        for directory in (self.candidate_directory, self.report_directory, self.transfer_directory):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = self._load_manifest()

    @property
    def archive_manifest_path(self) -> Path:
        """Manifest alias matching the promotion evaluator integration shape."""

        return self.manifest_path

    @property
    def last_evaluated_iteration(self) -> int:
        decisions = self.manifest.get("decisions", [])
        if not isinstance(decisions, list):
            raise ValueError("transition manifest decisions are malformed")
        return max((int(item["iteration"]) for item in decisions), default=-1)

    @property
    def last_accepted_decision(self) -> Mapping[str, Any] | None:
        decisions = self.manifest.get("decisions", [])
        if not isinstance(decisions, list):
            raise ValueError("transition manifest decisions are malformed")
        accepted = [item for item in decisions if isinstance(item, Mapping) and item.get("accepted")]
        return None if not accepted else dict(accepted[-1])

    def should_evaluate(self, iteration: int, *, completing: bool = False, last_evaluation_iteration: int = -1) -> bool:
        """Use the same bounded trigger semantics as periodic promotion."""

        if not self.config.enabled or iteration <= 0 or iteration == last_evaluation_iteration:
            return False
        return iteration % self.config.every_iterations == 0 or (completing and self.config.evaluate_on_complete)

    def evaluate_transition(
        self,
        *,
        iteration: int,
        candidate_checkpoint: str | Path,
        reference_checkpoint: str | Path | None = None,
        stage: CurriculumStage,
        run_context: Mapping[str, Any] | None = None,
        league: OpponentLeague | None = None,
    ) -> TransitionEvaluation:
        """Evaluate a frozen A-stage full checkpoint and maybe publish B weights.

        ``reference_checkpoint`` is preferred because it makes the previous-model
        regression gate explicit.  For runner integration, an archive-backed
        ``best`` or ``historical`` model may instead be resolved from ``league``.
        """

        _require_torch()
        if iteration < 1:
            raise ValueError("transition iteration must be positive")
        if stage is not self.config.source_stage:
            raise ValueError(f"automatic curriculum transition only evaluates stage {self.config.source_stage.value}")
        source_path = Path(candidate_checkpoint)
        source_stage, source_sha, source_run_hash = self._source_provenance(source_path)
        if source_stage is not self.config.source_stage:
            raise ValueError(f"transition source checkpoint must be stage {self.config.source_stage.value}")
        supplied_context = dict(run_context or {})
        claimed_hash = supplied_context.get("run_config_sha256")
        if claimed_hash is not None and claimed_hash != source_run_hash:
            raise ValueError("run_context run_config_sha256 does not match source full checkpoint")
        supplied_context["run_config_sha256"] = source_run_hash
        parent = self._resolve_parent(reference_checkpoint or self.config.reference_checkpoint, league)
        parent_sha = _file_sha256(parent)
        if parent_sha == source_sha:
            raise ValueError("transition candidate and previous-stage reference checkpoint must differ")
        candidate_model = PokerAgentModel.load_checkpoint(source_path, map_location="cpu")
        parent_model = PokerAgentModel.load_checkpoint(parent, map_location="cpu")
        if candidate_model.checkpoint_metadata() != parent_model.checkpoint_metadata():
            raise ValueError("previous-stage reference model is incompatible with the transition candidate")
        if all(
            torch.equal(candidate_model.state_dict()[name], parent_model.state_dict()[name])
            for name in candidate_model.state_dict()
        ):
            raise ValueError("transition candidate weights are identical to the previous-stage reference")
        candidate = self._freeze_source(source_path, source_sha, iteration)
        key = self._decision_key(source_sha)
        existing = self._existing_decision(key, source_path, source_sha, source_run_hash, candidate)
        if existing is not None:
            return existing

        suite = self._evaluate_suite(candidate, parent)
        evidence, reasons = self._evidence(suite)
        accepted = not reasons and evidence.stage_evaluation.passes(self.config.curriculum)
        if not evidence.stage_evaluation.passes(self.config.curriculum):
            reasons.append("StageEvaluation did not satisfy configured curriculum gates")
        report_path = self.report_directory / f"transition_A_to_B_{iteration:08d}_{source_sha[:12]}.json"
        transfer_path: Path | None = None
        if accepted:
            transfer_path = self.transfer_directory / f"transfer_A_to_B_{iteration:08d}_{source_sha[:12]}.pt"
            self._publish_transfer(candidate, transfer_path, global_step=self._source_global_decisions(source_path))
        report = {
            "transition_report_version": TRANSITION_REPORT_VERSION,
            "scalar_metric_protocol": "active_hands_expected_showdown_share_v1",
            "decision_key": key,
            "iteration": iteration,
            "source": {
                "full_checkpoint": str(source_path),
                "full_checkpoint_sha256": source_sha,
                "frozen_checkpoint": str(candidate),
                "frozen_checkpoint_sha256": _file_sha256(candidate),
                "stage": self.config.source_stage.value,
                "run_config_sha256": source_run_hash,
                "run_context": supplied_context,
            },
            "parent": {"checkpoint": str(parent), "checkpoint_sha256": parent_sha},
            "target_stage": self.config.target_stage.value,
            "transition_config": _config_dict(self.config),
            "transition_config_sha256": _canonical_sha256(_config_dict(self.config)),
            "suite": suite.as_dict(),
            "evidence": evidence.as_dict(),
            "decision": {
                "accepted": accepted,
                "reasons": reasons,
                "transfer_checkpoint": None if transfer_path is None else str(transfer_path),
                "transfer_checkpoint_sha256": None if transfer_path is None else _file_sha256(transfer_path),
            },
        }
        _atomic_write_json(report_path, report)
        record = {
            "decision_key": key,
            "iteration": iteration,
            "source_full_checkpoint": str(source_path),
            "source_full_checkpoint_sha256": source_sha,
            "run_config_sha256": source_run_hash,
            "frozen_source_checkpoint": str(candidate),
            "frozen_source_checkpoint_sha256": _file_sha256(candidate),
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": parent_sha,
            "accepted": accepted,
            "report_path": str(report_path),
            "report_sha256": _file_sha256(report_path),
            "transfer_checkpoint": None if transfer_path is None else str(transfer_path),
            "transfer_checkpoint_sha256": None if transfer_path is None else _file_sha256(transfer_path),
        }
        self._record(record)
        return TransitionEvaluation(iteration, accepted, tuple(reasons), evidence, report_path, transfer_path, candidate)

    def _evaluate_suite(self, candidate: Path, parent: Path) -> EvaluationSuiteReport:
        spec = stage_spec(self.config.source_stage)
        seed = self.run_seed + self.config.seed_start
        factories = {
            "rule": lambda: RuleBot(),
            "tight": lambda: TightBot(),
            "aggro": lambda: AggroBot(seed=seed),
            "calling_station": lambda: CallingStationBot(seed=seed),
            "random": lambda: RandomBot(seed=seed),
        }
        opponents: dict[str, Any] = {f"baseline:{name}": factories[name]() for name in self.config.baseline_bots}
        opponents["prior:checkpoint"] = ModelPolicy.from_checkpoint("prior:checkpoint", parent)
        protocol = EvaluationConfig(
            hands_per_opponent=self.config.hands_per_opponent,
            seed_start=self.config.seed_start,
            starting_stack=spec.starting_stack_chips(),
            player_count=2,
            allowed_raise_actions=tuple(sorted(spec.allowed_raise_actions, key=lambda action: action.value)),
            equity_samples=self.config.equity_samples,
            calibration_bins=self.config.calibration_bins,
            paired_position_seeds=True,
        )
        candidate_id = candidate.stem
        return evaluate_suite(candidate_id, ModelPolicy.from_checkpoint(candidate_id, candidate), opponents, config=protocol)

    def _evidence(self, suite: EvaluationSuiteReport) -> tuple[TransitionEvidence, list[str]]:
        baselines = tuple(item for item in suite.matchups if item.opponent.startswith("baseline:"))
        prior = tuple(item for item in suite.matchups if item.opponent == "prior:checkpoint")
        if not baselines or len(prior) != 1:
            raise ValueError("transition evaluation lacks required baseline or prior matchup")
        baseline_score = _aggregate_bb_per_100(baselines)
        prior_match = prior[0]
        showdown_share = [item.expected_showdown_share for item in suite.matchups]
        if any(item is None for item in showdown_share):
            raise ValueError("transition evaluation has no expected-showdown-share diagnostics")
        max_ece = max(
            item.expected_calibration_error for item in showdown_share if item is not None
        )
        illegal_count = sum(item.model_diagnostics.illegal_action_count for item in suite.matchups)
        sanity_passed = all(item.legal and item.finite for item in suite.sanity_checks)
        previous_ok = prior_match.bb_per_100_ci95_low >= self.config.minimum_prior_ci95_low
        control_passed = sanity_passed and illegal_count == 0
        stage_evaluation = StageEvaluation(
            baseline_win_rate_bb_per_100=baseline_score,
            equity_calibration_error=max_ece,
            beats_previous_checkpoint=previous_ok,
            transfer_bb_per_100=None,
            scratch_bb_per_100=None,
            control_set_passed=control_passed,
        )
        reasons: list[str] = []
        if any(item.bb_per_100_ci95_low < self.config.minimum_baseline_ci95_low for item in baselines):
            reasons.append("a baseline matchup confidence bound is below the configured floor")
        if any((item.bb_per_100_ci95_high - item.bb_per_100_ci95_low) / 2.0 > self.config.maximum_baseline_ci95_half_width for item in baselines):
            reasons.append("a baseline matchup confidence interval is wider than the configured ceiling")
        if not previous_ok:
            reasons.append("prior checkpoint lower confidence bound is below the configured floor")
        if not sanity_passed:
            reasons.append("one or more fixed sanity scenarios failed")
        if illegal_count:
            reasons.append("candidate selected an illegal or masked action")
        return TransitionEvidence(stage_evaluation, baseline_score, max_ece, prior_match.bb_per_100_ci95_low, sanity_passed, illegal_count), reasons

    def _source_provenance(self, path: Path) -> tuple[CurriculumStage, str, str]:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("run_config"), Mapping) or not isinstance(payload.get("progress"), Mapping):
            raise ValueError("transition source must be a full training checkpoint")
        # This validates model metadata and the stage attached to the full run.
        PokerAgentModel.load_checkpoint(path, map_location="cpu")
        curriculum = payload.get("curriculum")
        stage = curriculum.get("stage") if isinstance(curriculum, Mapping) else None
        try:
            source_stage = CurriculumStage(stage)
        except (TypeError, ValueError) as error:
            raise ValueError("transition source full checkpoint has invalid curriculum stage") from error
        return source_stage, _file_sha256(path), _canonical_sha256(payload["run_config"])

    def _source_global_decisions(self, path: Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        progress = payload.get("progress") if isinstance(payload, Mapping) else None
        value = progress.get("global_decisions") if isinstance(progress, Mapping) else None
        if not isinstance(value, int) or value < 0:
            raise ValueError("source full checkpoint has invalid global decision count")
        return value

    def _resolve_parent(self, parent_checkpoint: str | Path | None, league: OpponentLeague | None) -> Path:
        if parent_checkpoint is not None:
            path = Path(parent_checkpoint)
            if not path.is_file():
                raise FileNotFoundError(path)
            configured = self.config.reference_checkpoint
            if configured is not None and path.resolve() != Path(configured).resolve():
                raise ValueError("reference checkpoint differs from the path pinned in transition config")
            expected_sha = self.config.reference_checkpoint_sha256
            if expected_sha is not None and _file_sha256(path) != expected_sha:
                raise ValueError("reference checkpoint does not match reference_checkpoint_sha256")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            curriculum = payload.get("curriculum") if isinstance(payload, Mapping) else None
            if (
                not isinstance(payload, Mapping)
                or not isinstance(payload.get("run_config"), Mapping)
                or not isinstance(payload.get("progress"), Mapping)
                or not isinstance(curriculum, Mapping)
                or curriculum.get("stage") != self.config.source_stage.value
            ):
                raise ValueError("reference checkpoint must be a full Stage A training checkpoint")
            PokerAgentModel.load_checkpoint(path, map_location="cpu")
            return path
        if league is not None:
            for member in reversed(league.members):
                if member.kind in {"best", "historical"} and isinstance(member.policy, ModelPolicy) and member.policy.checkpoint_path is not None:
                    path = member.policy.checkpoint_path
                    if path.is_file():
                        PokerAgentModel.load_checkpoint(path, map_location="cpu")
                        return path
        raise ValueError("A -> B transition requires parent_checkpoint or an archive-backed prior model in league")

    def _freeze_source(self, source: Path, source_sha: str, iteration: int) -> Path:
        destination = self.candidate_directory / f"candidate_A_to_B_{iteration:08d}_{source_sha[:12]}.pt"
        if destination.exists():
            if _file_sha256(destination) != source_sha:
                raise ValueError("frozen transition source checkpoint hash mismatch")
            return destination
        _atomic_copy(source, destination)
        if _file_sha256(source) != source_sha or _file_sha256(destination) != source_sha:
            raise ValueError("source full checkpoint changed while being frozen")
        return destination

    def _publish_transfer(self, source: Path, destination: Path, *, global_step: int) -> None:
        if destination.exists():
            self._validate_transfer(destination, source)
            return
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            model = PokerAgentModel.load_checkpoint(source, map_location="cpu")
            target = self.config.target_stage
            torch.save(
                {
                    "metadata": model.checkpoint_metadata(),
                    "state_dict": model.state_dict(),
                    "curriculum": {
                        "version": 1,
                        "stage": target.value,
                        "global_step": global_step,
                        "parent_checkpoint": str(source),
                        "stage_spec": {
                            "player_count": stage_spec(target).player_count,
                            "allowed_raise_actions": sorted(action.value for action in stage_spec(target).allowed_raise_actions),
                            "starting_stacks_bb": list(stage_spec(target).starting_stacks_bb),
                        },
                    },
                },
                temporary,
            )
            _publish_file(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._validate_transfer(destination, source)

    def _validate_transfer(self, path: Path, source: Path) -> None:
        info = checkpoint_curriculum_metadata(path)
        if CurriculumStage(info["stage"]) is not self.config.target_stage or info.get("parent_checkpoint") != str(source):
            raise ValueError("transition transfer artifact provenance does not match frozen source")
        PokerAgentModel.load_checkpoint(path, map_location="cpu")

    def _decision_key(self, source_sha: str) -> str:
        return f"{self.config.source_stage.value}->{self.config.target_stage.value}:{source_sha}"

    def _existing_decision(self, key: str, source: Path, source_sha: str, run_hash: str, frozen: Path) -> TransitionEvaluation | None:
        decisions = self.manifest["decisions"]
        assert isinstance(decisions, list)
        record = next((item for item in decisions if item.get("decision_key") == key), None)
        if record is None:
            return None
        if not isinstance(record, Mapping):
            raise ValueError("transition manifest has malformed decision")
        required = {
            "source_full_checkpoint": str(source),
            "source_full_checkpoint_sha256": source_sha,
            "run_config_sha256": run_hash,
            "frozen_source_checkpoint": str(frozen),
        }
        if any(record.get(name) != value for name, value in required.items()):
            raise ValueError("transition retry provenance does not match existing decision")
        report_path = Path(str(record.get("report_path")))
        if not report_path.is_file() or record.get("report_sha256") != _file_sha256(report_path):
            raise ValueError("transition report is missing or hash-mismatched")
        report = _load_json(report_path)
        if report.get("decision_key") != key or report.get("source", {}).get("full_checkpoint_sha256") != source_sha:
            raise ValueError("transition report does not match manifest provenance")
        evidence_data = report.get("evidence")
        decision = report.get("decision")
        if not isinstance(evidence_data, Mapping) or not isinstance(decision, Mapping):
            raise ValueError("transition report has malformed evidence or decision")
        evidence = TransitionEvidence.from_dict(evidence_data)
        accepted = bool(decision.get("accepted"))
        if accepted != bool(record.get("accepted")):
            raise ValueError("transition report and manifest disagree on acceptance")
        transfer_value = record.get("transfer_checkpoint")
        transfer = None if transfer_value is None else Path(str(transfer_value))
        if accepted:
            if transfer is None or not transfer.is_file() or record.get("transfer_checkpoint_sha256") != _file_sha256(transfer):
                raise ValueError("accepted transition transfer artifact is missing or hash-mismatched")
            self._validate_transfer(transfer, frozen)
        elif transfer is not None:
            raise ValueError("rejected transition unexpectedly has a transfer artifact")
        return TransitionEvaluation(
            int(record["iteration"]), accepted, tuple(str(item) for item in decision.get("reasons", [])), evidence, report_path, transfer, frozen
        )

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": TRANSITION_MANIFEST_VERSION, "decisions": []}
        value = _load_json(self.manifest_path)
        if value.get("version") != TRANSITION_MANIFEST_VERSION or not isinstance(value.get("decisions"), list):
            raise ValueError("incompatible or malformed curriculum transition manifest")
        self._validate_manifest(value)
        return value

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Fail closed before a resumed runner can trust transition artifacts."""

        decisions = manifest.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError("transition manifest decisions are malformed")
        expected_config_sha = _canonical_sha256(_config_dict(self.config))
        seen: set[str] = set()
        for record in decisions:
            if not isinstance(record, Mapping):
                raise ValueError("transition manifest contains a malformed decision")
            key = record.get("decision_key")
            iteration = record.get("iteration")
            if not isinstance(key, str) or not key or key in seen or not isinstance(iteration, int) or iteration < 1:
                raise ValueError("transition manifest has invalid decision identity")
            seen.add(key)
            required_paths = (
                ("source_full_checkpoint", "source_full_checkpoint_sha256"),
                ("frozen_source_checkpoint", "frozen_source_checkpoint_sha256"),
                ("parent_checkpoint", "parent_checkpoint_sha256"),
                ("report_path", "report_sha256"),
            )
            for path_key, hash_key in required_paths:
                path_value, hash_value = record.get(path_key), record.get(hash_key)
                path = Path(str(path_value))
                if not isinstance(path_value, str) or not isinstance(hash_value, str) or not path.is_file() or _file_sha256(path) != hash_value:
                    raise ValueError(f"transition manifest {path_key} is missing or hash-mismatched")
            source = Path(str(record["source_full_checkpoint"]))
            frozen = Path(str(record["frozen_source_checkpoint"]))
            report = _load_json(Path(str(record["report_path"])))
            report_source = report.get("source")
            report_parent = report.get("parent")
            report_decision = report.get("decision")
            if not isinstance(report_source, Mapping) or not isinstance(report_parent, Mapping) or not isinstance(report_decision, Mapping):
                raise ValueError("transition report has malformed source, parent or decision")
            if (
                report.get("decision_key") != key
                or report.get("iteration") != iteration
                or report.get("transition_config_sha256") != expected_config_sha
                or report_source.get("full_checkpoint") != str(source)
                or report_source.get("full_checkpoint_sha256") != record.get("source_full_checkpoint_sha256")
                or report_source.get("frozen_checkpoint") != str(frozen)
                or report_source.get("frozen_checkpoint_sha256") != record.get("frozen_source_checkpoint_sha256")
                or report_source.get("run_config_sha256") != record.get("run_config_sha256")
                or report_parent.get("checkpoint") != record.get("parent_checkpoint")
                or report_parent.get("checkpoint_sha256") != record.get("parent_checkpoint_sha256")
                or bool(report_decision.get("accepted")) != bool(record.get("accepted"))
                or report_decision.get("transfer_checkpoint") != record.get("transfer_checkpoint")
                or report_decision.get("transfer_checkpoint_sha256") != record.get("transfer_checkpoint_sha256")
            ):
                raise ValueError("transition manifest and report provenance disagree")
            accepted = bool(record.get("accepted"))
            transfer_value = record.get("transfer_checkpoint")
            transfer_sha = record.get("transfer_checkpoint_sha256")
            if accepted:
                transfer = Path(str(transfer_value))
                if not isinstance(transfer_value, str) or not isinstance(transfer_sha, str) or not transfer.is_file() or _file_sha256(transfer) != transfer_sha:
                    raise ValueError("accepted transition transfer artifact is missing or hash-mismatched")
                self._validate_transfer(transfer, frozen)
            elif transfer_value is not None or transfer_sha is not None:
                raise ValueError("rejected transition unexpectedly has a transfer artifact")

    def _record(self, record: Mapping[str, Any]) -> None:
        decisions = self.manifest["decisions"]
        assert isinstance(decisions, list)
        existing = next((item for item in decisions if item.get("decision_key") == record["decision_key"]), None)
        if existing is not None:
            if dict(existing) != dict(record):
                raise ValueError("transition retry produced a different immutable decision")
            return
        decisions.append(dict(record))
        _atomic_write_json(self.manifest_path, self.manifest)


def _aggregate_bb_per_100(reports: Sequence[MatchupReport]) -> float:
    hands = sum(item.hands for item in reports)
    if not hands:
        raise ValueError("transition evaluation has no baseline hands")
    return sum(item.pnl_bb for item in reports) / hands * 100.0


def _config_dict(config: CurriculumTransitionConfig) -> dict[str, Any]:
    result = asdict(config)
    result["source_stage"] = config.source_stage.value
    result["target_stage"] = config.target_stage.value
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # pragma: no cover - filesystem-specific fallback.
        pass


__all__ = [
    "CurriculumTransitionConfig",
    "CurriculumTransitionEvaluator",
    "TRANSITION_MANIFEST_VERSION",
    "TRANSITION_REPORT_VERSION",
    "TransitionEvaluation",
    "TransitionEvidence",
]
