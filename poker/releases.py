"""Append-only, hash-verified release registry for completed PPO trials.

The registry deliberately promotes no model and starts no training.  It
records the narrow lineage needed to make a native full checkpoint releasable:
one completed experiment ledger and one passing, pinned tuning evaluation.
All referenced files are re-opened on registration and verification; a path
alone is never treated as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .experiments import ExperimentConfig, ExperimentLedger
from .model import TORCH_AVAILABLE
from .promotion import PROMOTION_REPORT_VERSION, PromotionConfig
from .train_runner import CHECKPOINT_VERSION, TrainingRunConfig
from .tuning import (
    TUNING_EVALUATION_KIND,
    TUNING_PROTOCOL_KIND,
    TUNING_SCHEMA_VERSION,
    hu_promotion_protocol_payload,
)

if TORCH_AVAILABLE:
    import torch


RELEASE_REGISTRY_VERSION = 1
RELEASE_KIND = "poker_ppo_release_v1"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return str(value)


def _safe_component(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value in {".", ".."}
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value)
    ):
        raise ValueError(f"{label} must use only letters, digits, '.', '_' or '-' and be non-empty")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path, label: str = "artifact") -> str:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(path: str | Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve()
    expected = _require_sha256(expected_sha256, f"{label} expected_sha256")
    if _file_sha256(resolved, label) != expected:
        raise ValueError(f"{label} SHA-256 does not match the requested artifact")
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    contents = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # pragma: no cover - filesystem-specific fallback.
        pass


@dataclass(frozen=True)
class LineageArtifact:
    """An additional immutable, hash-pinned release input."""

    kind: str
    path: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _safe_component(self.kind, "lineage artifact kind"))
        object.__setattr__(self, "path", Path(self.path).resolve())
        _require_sha256(self.expected_sha256, "lineage artifact expected_sha256")

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "path": str(self.path), "expected_sha256": self.expected_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LineageArtifact":
        try:
            return cls(str(value["kind"]), Path(str(value["path"])), str(value["expected_sha256"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed extra lineage artifact") from error


@dataclass(frozen=True)
class ReleaseRequest:
    """Complete immutable release intent; all paths must be hash-pinned."""

    release_id: str
    code_revision: str
    full_checkpoint_path: Path
    full_checkpoint_sha256: str
    experiment_ledger_manifest_path: Path
    experiment_ledger_manifest_sha256: str
    tuning_evaluation_report_path: Path
    tuning_evaluation_report_sha256: str
    extra_lineage_artifacts: Sequence[LineageArtifact] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_id", _safe_component(self.release_id, "release_id"))
        if not isinstance(self.code_revision, str) or not self.code_revision.strip() or len(self.code_revision) > 512:
            raise ValueError("code_revision must be explicit, non-empty and at most 512 characters")
        object.__setattr__(self, "code_revision", self.code_revision.strip())
        for field_name in (
            "full_checkpoint_path",
            "experiment_ledger_manifest_path",
            "tuning_evaluation_report_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).resolve())
        for field_name in (
            "full_checkpoint_sha256",
            "experiment_ledger_manifest_sha256",
            "tuning_evaluation_report_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if not isinstance(self.extra_lineage_artifacts, Sequence) or isinstance(self.extra_lineage_artifacts, (str, bytes)):
            raise ValueError("extra_lineage_artifacts must be a sequence of LineageArtifact")
        extras = tuple(self.extra_lineage_artifacts)
        if any(not isinstance(item, LineageArtifact) for item in extras):
            raise ValueError("extra_lineage_artifacts must contain LineageArtifact values")
        canonical_extras = tuple(sorted(extras, key=lambda item: (item.kind, str(item.path), item.expected_sha256)))
        if len({(item.kind, item.path) for item in canonical_extras}) != len(canonical_extras):
            raise ValueError("extra_lineage_artifacts cannot repeat a kind/path pair")
        object.__setattr__(self, "extra_lineage_artifacts", canonical_extras)

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "code_revision": self.code_revision,
            "full_checkpoint": {"path": str(self.full_checkpoint_path), "expected_sha256": self.full_checkpoint_sha256},
            "experiment_ledger_manifest": {
                "path": str(self.experiment_ledger_manifest_path),
                "expected_sha256": self.experiment_ledger_manifest_sha256,
            },
            "tuning_evaluation_report": {
                "path": str(self.tuning_evaluation_report_path),
                "expected_sha256": self.tuning_evaluation_report_sha256,
            },
            "extra_lineage_artifacts": [item.as_dict() for item in self.extra_lineage_artifacts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseRequest":
        try:
            checkpoint = value["full_checkpoint"]
            ledger = value["experiment_ledger_manifest"]
            report = value["tuning_evaluation_report"]
            extras = value.get("extra_lineage_artifacts", ())
            if not all(isinstance(item, Mapping) for item in (checkpoint, ledger, report)):
                raise ValueError("release request artifact bindings must be objects")
            if not isinstance(extras, Sequence) or isinstance(extras, (str, bytes)):
                raise ValueError("release request extra artifacts must be an array")
            if any(not isinstance(item, Mapping) for item in extras):
                raise ValueError("release request extra artifacts must be objects")
            return cls(
                str(value["release_id"]),
                str(value["code_revision"]),
                Path(str(checkpoint["path"])),
                str(checkpoint["expected_sha256"]),
                Path(str(ledger["path"])),
                str(ledger["expected_sha256"]),
                Path(str(report["path"])),
                str(report["expected_sha256"]),
                tuple(LineageArtifact.from_dict(item) for item in extras if isinstance(item, Mapping)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed release request") from error


@dataclass(frozen=True)
class ReleaseRecord:
    release_id: str
    release_path: Path
    release_sha256: str
    request: ReleaseRequest


class ReleaseRegistry:
    """Immutable release records plus an atomically replaced append manifest."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.releases_directory = self.directory / "releases"
        if self.manifest_path.exists():
            self._manifest = _load_json(self.manifest_path, "release registry manifest")
        else:
            self._manifest: dict[str, Any] = {"version": RELEASE_REGISTRY_VERSION, "releases": []}
            _atomic_write_json(self.manifest_path, self._manifest)
        # A process can die after publishing an immutable release JSON but
        # before appending its manifest record.  Defer that single-file crash
        # window to ``register(request)`` where the requested evidence can be
        # compared byte-for-byte; list/show/verify remain fail-closed.
        self._verify_registry(allow_unlisted_orphans=True)

    @property
    def manifest_path(self) -> Path:
        return self.directory / "release-manifest.json"

    def register(self, request: ReleaseRequest) -> ReleaseRecord:
        """Validate evidence and atomically append one immutable release."""

        if not isinstance(request, ReleaseRequest):
            raise TypeError("request must be a ReleaseRequest")
        path = self._release_path(request.release_id)
        existing = self._record_by_id(request.release_id)
        if existing is None and path.exists():
            # The only recoverable two-file crash window is an immutable
            # release JSON published before the append manifest.  Validate the
            # exact request first; every other unlisted file is fail-closed.
            payload = self._release_payload(request)
            if _load_json(path, "orphan release record") != payload:
                raise ValueError("release record path already exists with divergent evidence")
            self._verify_registry(allowed_orphan=path)
            record = {"release_id": request.release_id, "path": str(path), "sha256": _file_sha256(path, "release record")}
            self._append_record(record)
            return self._record_from_manifest(record, verify=True)
        self._verify_registry()
        payload = self._release_payload(request)
        if existing is not None:
            record = self._record_from_manifest(existing, verify=True)
            existing_payload = _load_json(record.release_path, "release record")
            if existing_payload != payload:
                raise ValueError("release_id already exists with divergent immutable evidence")
            return record
        _atomic_write_json(path, payload)
        record = {"release_id": request.release_id, "path": str(path), "sha256": _file_sha256(path, "release record")}
        self._append_record(record)
        return self._record_from_manifest(record, verify=True)

    def _append_record(self, record: Mapping[str, Any]) -> None:
        # Do not mutate the live in-memory manifest until its atomic durable
        # replacement succeeds.  A failed fsync/replace cannot make a retry
        # believe an append was committed (manifest append rollback).
        next_manifest = dict(self._manifest)
        next_manifest["releases"] = [*self._records(), dict(record)]
        self._write_manifest(next_manifest)
        self._manifest = next_manifest

    def register_release(self, request: ReleaseRequest) -> ReleaseRecord:
        """Compatibility spelling for callers that prefer an explicit verb."""

        return self.register(request)

    def list(self) -> tuple[ReleaseRecord, ...]:
        self._verify_registry()
        return tuple(self._record_from_manifest(record, verify=True) for record in self._records())

    def list_releases(self) -> tuple[ReleaseRecord, ...]:
        return self.list()

    def show(self, release_id: str) -> ReleaseRecord:
        self._verify_registry()
        record = self._record_by_id(_safe_component(release_id, "release_id"))
        if record is None:
            raise KeyError(f"unknown release_id: {release_id}")
        return self._record_from_manifest(record, verify=True)

    def verify(self, release_id: str | None = None) -> ReleaseRecord | tuple[ReleaseRecord, ...]:
        """Re-hash the registry and all external lineage artifacts."""

        self._verify_registry()
        return self.list() if release_id is None else self.show(release_id)

    def _release_payload(self, request: ReleaseRequest) -> dict[str, Any]:
        checkpoint = self._validate_checkpoint(request)
        checkpoint_run_config = checkpoint.pop("_run_config")
        ledger = self._validate_ledger(request, checkpoint_run_config)
        tuning = self._validate_tuning_report(
            request,
            checkpoint_run_config,
            checkpoint["run_config_sha256"],
            ledger["evaluation_protocol_sha256"],
            ledger["evaluation_protocol_path"],
            ledger["name"],
            request.experiment_ledger_manifest_path,
            request.experiment_ledger_manifest_sha256,
        )
        for artifact in request.extra_lineage_artifacts:
            _verified_path(artifact.path, artifact.expected_sha256, f"extra lineage artifact {artifact.kind!r}")
        request_data = request.as_dict()
        return {
            "version": RELEASE_REGISTRY_VERSION,
            "kind": RELEASE_KIND,
            "release_id": request.release_id,
            "request": request_data,
            "request_sha256": _canonical_sha256(request_data),
            "verified": {"checkpoint": checkpoint, "experiment_ledger": ledger, "tuning_evaluation": tuning},
        }

    def _validate_checkpoint(self, request: ReleaseRequest) -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to verify release checkpoints; install the project with `.[rl].")
        checkpoint_path = _verified_path(request.full_checkpoint_path, request.full_checkpoint_sha256, "full checkpoint")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = ("metadata", "state_dict", "optimizer_state_dict", "run_config", "curriculum", "progress", "league", "rng")
        if not isinstance(payload, Mapping) or payload.get("checkpoint_version") != CHECKPOINT_VERSION or any(key not in payload for key in required):
            raise ValueError("release requires a compatible native full TrainingRunner checkpoint")
        if not all(isinstance(payload.get(key), Mapping) for key in required):
            raise ValueError("release full checkpoint has malformed native state")
        run_config = payload["run_config"]
        try:
            normalized = TrainingRunConfig.from_dict(run_config).to_dict()
        except ValueError as error:
            raise ValueError("release full checkpoint has invalid run configuration") from error
        if normalized != dict(run_config):
            raise ValueError("release full checkpoint run configuration is not canonical")
        progress = payload["progress"]
        iteration = progress.get("iteration")
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 1:
            raise ValueError("release full checkpoint has invalid completed iteration")
        return {
            "path": str(checkpoint_path),
            "sha256": request.full_checkpoint_sha256,
            "checkpoint_version": CHECKPOINT_VERSION,
            "run_config_sha256": _canonical_sha256(dict(run_config)),
            "iteration": iteration,
            "_run_config": dict(run_config),
        }

    def _validate_ledger(self, request: ReleaseRequest, checkpoint_run_config: Mapping[str, Any]) -> dict[str, Any]:
        manifest_path = _verified_path(
            request.experiment_ledger_manifest_path,
            request.experiment_ledger_manifest_sha256,
            "experiment ledger manifest",
        )
        if manifest_path.name != "experiment-manifest.json" or manifest_path.parent.name != "experiment-ledger":
            raise ValueError("experiment ledger manifest must be the native experiment-ledger/experiment-manifest.json")
        manifest = _load_json(manifest_path, "experiment ledger manifest")
        config_data = manifest.get("config")
        if not isinstance(config_data, Mapping):
            raise ValueError("experiment ledger manifest has no ExperimentConfig")
        experiment = ExperimentConfig.from_dict(config_data)
        if experiment.code_revision != request.code_revision:
            raise ValueError("experiment ledger code_revision does not match release request")
        if experiment.training.to_dict() != dict(checkpoint_run_config):
            raise ValueError("experiment ledger TrainingRunConfig does not match full checkpoint")
        ledger = ExperimentLedger(manifest_path.parent.parent, experiment)
        if ledger.manifest_path.resolve() != manifest_path or _file_sha256(manifest_path, "experiment ledger manifest") != request.experiment_ledger_manifest_sha256:
            raise ValueError("experiment ledger manifest changed during release verification")
        if ledger.last_iteration != experiment.max_iterations:
            raise ValueError("experiment ledger is incomplete through its declared max_iterations")
        records = ledger._records()
        if not records:
            raise ValueError("experiment ledger is missing its final event")
        final = ledger._event_from_record(records[-1])
        if final.checkpoint_sha256 != request.full_checkpoint_sha256:
            raise ValueError("experiment ledger final event checkpoint SHA-256 does not match release checkpoint")
        status = _load_json(ledger.status_path, "experiment status")
        if (
            status.get("version") != 1
            or status.get("trial_id") != experiment.trial_id
            or status.get("config_sha256") != experiment.config_sha256
            or status.get("state") != "completed"
            or status.get("last_iteration") != final.iteration
            or status.get("last_event_sha256") != final.event_sha256
        ):
            raise ValueError("experiment ledger has no completed terminal status")
        return {
            "manifest_path": str(manifest_path),
            "manifest_sha256": request.experiment_ledger_manifest_sha256,
            "trial_id": experiment.trial_id,
            "name": experiment.name,
            "config_sha256": experiment.config_sha256,
            "evaluation_protocol_sha256": experiment.evaluation_protocol_sha256,
            "evaluation_protocol_path": experiment.evaluation_protocol_path,
            "max_iterations": experiment.max_iterations,
            "final_event_sha256": final.event_sha256,
            "final_checkpoint_sha256": final.checkpoint_sha256,
        }

    def _validate_tuning_report(
        self,
        request: ReleaseRequest,
        checkpoint_run_config: Mapping[str, Any],
        run_config_sha256: str,
        evaluation_protocol_sha256: str,
        evaluation_protocol_path: str,
        experiment_name: str,
        ledger_manifest_path: Path,
        ledger_manifest_sha256: str,
    ) -> dict[str, Any]:
        report_path = _verified_path(
            request.tuning_evaluation_report_path,
            request.tuning_evaluation_report_sha256,
            "tuning evaluation report",
        )
        report = _load_json(report_path, "tuning evaluation report")
        metrics, decision = report.get("metrics"), report.get("decision")
        if (
            report.get("schema_version") != TUNING_SCHEMA_VERSION
            or report.get("kind") != TUNING_EVALUATION_KIND
            or report.get("full_checkpoint_path") != str(request.full_checkpoint_path)
            or report.get("full_checkpoint_sha256") != request.full_checkpoint_sha256
            or report.get("run_config_sha256") != run_config_sha256
            or report.get("evaluation_protocol_sha256") != evaluation_protocol_sha256
            or not isinstance(metrics, Mapping)
            or not isinstance(decision, Mapping)
            or decision.get("passed") is not True
            or metrics.get("illegal_action_count") != 0
            or isinstance(metrics.get("illegal_action_count"), bool)
        ):
            raise ValueError("tuning evaluation report is not a passing exact checkpoint/protocol binding")
        trial_id = report.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("tuning evaluation report has invalid trial_id")
        _safe_component(trial_id, "tuning evaluation trial_id")
        if trial_id != experiment_name:
            raise ValueError("tuning evaluation report trial_id does not match ExperimentConfig.name")
        lineage = report.get("lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("tuning evaluation report lacks mandatory ledger/promotion lineage wrapper")
        promotion_binding = lineage.get("source_promotion", lineage.get("promotion", lineage))
        if not isinstance(promotion_binding, Mapping):
            raise ValueError("tuning evaluation report has malformed promotion lineage wrapper")
        if (
            lineage.get("experiment_ledger_manifest_path") != str(ledger_manifest_path)
            or lineage.get("experiment_ledger_manifest_sha256") != ledger_manifest_sha256
            or lineage.get("evaluation_protocol_path") != evaluation_protocol_path
            or lineage.get("evaluation_protocol_sha256") != evaluation_protocol_sha256
        ):
            raise ValueError("tuning evaluation report ledger binding does not match release evidence")
        promotion_report_path, promotion_report_sha = self._lineage_artifact(
            promotion_binding,
            "promotion_report",
            "source promotion report",
        )
        promotion_archive_path, promotion_archive_sha = self._lineage_artifact(
            promotion_binding,
            "promotion_archive_manifest",
            "source promotion archive manifest",
        )
        derived_score, derived_ece = self._validate_promotion_lineage(
            promotion_report_path,
            promotion_report_sha,
            promotion_archive_path,
            promotion_archive_sha,
            request,
            checkpoint_run_config,
            run_config_sha256,
            evaluation_protocol_path,
            evaluation_protocol_sha256,
        )
        for field_name in ("score_lower_ci", "expected_showdown_share_ece"):
            value = metrics.get(field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"tuning evaluation report {field_name} must be finite")
        if not 0.0 <= float(metrics["expected_showdown_share_ece"]) <= 1.0:
            raise ValueError("tuning evaluation report expected_showdown_share_ece must be in [0, 1]")
        if float(metrics["score_lower_ci"]) != derived_score or float(metrics["expected_showdown_share_ece"]) != derived_ece:
            raise ValueError("tuning evaluation report metrics diverge from source promotion evidence")
        return {
            "path": str(report_path),
            "sha256": request.tuning_evaluation_report_sha256,
            "trial_id": trial_id,
            "run_config_sha256": run_config_sha256,
            "evaluation_protocol_sha256": evaluation_protocol_sha256,
            "promotion_report_path": str(promotion_report_path),
            "promotion_report_sha256": promotion_report_sha,
            "promotion_archive_manifest_path": str(promotion_archive_path),
            "promotion_archive_manifest_sha256": promotion_archive_sha,
        }

    @staticmethod
    def _lineage_artifact(binding: Mapping[str, Any], prefix: str, label: str) -> tuple[Path, str]:
        nested = binding.get(prefix)
        if isinstance(nested, Mapping):
            path, expected_sha = nested.get("path"), nested.get("sha256")
        else:
            path, expected_sha = binding.get(f"{prefix}_path"), binding.get(f"{prefix}_sha256")
        if not isinstance(path, str) or not isinstance(expected_sha, str):
            raise ValueError(f"tuning evaluation report lacks {label} hash binding")
        return _verified_path(path, expected_sha, label), expected_sha

    @staticmethod
    def _validate_promotion_lineage(
        report_path: Path,
        report_sha256: str,
        archive_path: Path,
        archive_sha256: str,
        request: ReleaseRequest,
        checkpoint_run_config: Mapping[str, Any],
        run_config_sha256: str,
        evaluation_protocol_path: str,
        evaluation_protocol_sha256: str,
    ) -> tuple[float, float]:
        report = _load_json(report_path, "source promotion report")
        candidate, decision, suite = report.get("candidate"), report.get("decision"), report.get("suite")
        protocol_artifact_path = _verified_path(
            evaluation_protocol_path,
            evaluation_protocol_sha256,
            "evaluation protocol artifact",
        )
        protocol_artifact = _load_json(protocol_artifact_path, "evaluation protocol artifact")
        promotion_config, expected_protocol = protocol_artifact.get("promotion_config"), protocol_artifact.get("promotion_protocol")
        run_context = candidate.get("run_context") if isinstance(candidate, Mapping) else None
        try:
            canonical_protocol = hu_promotion_protocol_payload(
                TrainingRunConfig.from_dict(checkpoint_run_config),
                PromotionConfig(**dict(promotion_config)) if isinstance(promotion_config, Mapping) else PromotionConfig(),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("source promotion protocol is invalid for release training config") from error
        if (
            report.get("promotion_report_version") != PROMOTION_REPORT_VERSION
            or protocol_artifact.get("schema_version") != TUNING_SCHEMA_VERSION
            or protocol_artifact.get("kind") != TUNING_PROTOCOL_KIND
            or protocol_artifact.get("evaluator") != f"hu_promotion_report_v{PROMOTION_REPORT_VERSION}"
            or not isinstance(promotion_config, Mapping)
            or not isinstance(expected_protocol, Mapping)
            or protocol_artifact.get("promotion_protocol_sha256") != _canonical_sha256(dict(expected_protocol))
            or _canonical_sha256(protocol_artifact) != _canonical_sha256(canonical_protocol)
            or not isinstance(candidate, Mapping)
            or not isinstance(decision, Mapping)
            or not isinstance(suite, Mapping)
            or candidate.get("source_full_checkpoint") != str(request.full_checkpoint_path)
            or candidate.get("source_full_checkpoint_sha256") != request.full_checkpoint_sha256
            or not isinstance(run_context, Mapping)
            or run_context.get("run_config_sha256") != run_config_sha256
            or run_context.get("evaluation_protocol_sha256") != evaluation_protocol_sha256
            or report.get("promotion_config") != dict(promotion_config)
            or report.get("protocol") != dict(expected_protocol)
            or report.get("protocol_sha256") != protocol_artifact.get("promotion_protocol_sha256")
            or decision.get("accepted") is not True
            or decision.get("reasons") != []
        ):
            raise ValueError("source promotion report is not an accepted exact full-checkpoint evaluation")
        score = decision.get("baseline_score_bb_per_100")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(float(score)):
            raise ValueError("source promotion report baseline score must be finite")
        matchups = suite.get("matchups")
        if not isinstance(matchups, list) or not matchups:
            raise ValueError("source promotion report has no matchup diagnostics")
        baselines = []
        for matchup in matchups:
            diagnostics = matchup.get("model_diagnostics") if isinstance(matchup, Mapping) else None
            share = matchup.get("expected_showdown_share") if isinstance(matchup, Mapping) else None
            if not isinstance(diagnostics, Mapping) or diagnostics.get("illegal_action_count") != 0:
                raise ValueError("source promotion report has illegal actions")
            if isinstance(matchup, Mapping) and str(matchup.get("opponent", "")).startswith("baseline:"):
                baselines.append(matchup)
            if share is not None:
                ece = share.get("expected_calibration_error") if isinstance(share, Mapping) else None
                if isinstance(ece, bool) or not isinstance(ece, (int, float)) or not isfinite(float(ece)) or not 0.0 <= float(ece) <= 1.0:
                    raise ValueError("source promotion report expected-showdown-share ECE must be finite and bounded")
        if not baselines or any(not isinstance(item.get("expected_showdown_share"), Mapping) for item in baselines):
            raise ValueError("source promotion report lacks baseline expected-showdown-share diagnostics")
        try:
            raw_scores = [item["bb_per_100_ci95_low"] for item in baselines]
            raw_eces = [item["expected_showdown_share"]["expected_calibration_error"] for item in baselines]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (*raw_scores, *raw_eces)):
                raise ValueError("boolean/non-numeric evaluation metric")
            derived_score = min(float(value) for value in raw_scores)
            derived_ece = max(float(value) for value in raw_eces)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("source promotion report lacks derived tuning metrics") from error
        candidate_path, candidate_sha = candidate.get("path"), candidate.get("sha256")
        if not isinstance(candidate_path, str) or not isinstance(candidate_sha, str) or _file_sha256(Path(candidate_path).resolve(), "promotion candidate") != candidate_sha:
            raise ValueError("source promotion report candidate checkpoint is missing or hash-mismatched")
        archive = _load_json(archive_path, "source promotion archive manifest")
        decisions = archive.get("decisions")
        if archive.get("version") != PROMOTION_REPORT_VERSION or not isinstance(decisions, list):
            raise ValueError("source promotion archive manifest has incompatible schema")
        expected_report = report_path.resolve()
        if not any(
            isinstance(item, Mapping)
            and Path(str(item.get("report_path"))).resolve() == expected_report
            and item.get("report_sha256") == report_sha256
            and item.get("source_full_checkpoint_sha256") == request.full_checkpoint_sha256
            and item.get("accepted") is True
            for item in decisions
        ):
            raise ValueError("source promotion archive does not bind the accepted report and release checkpoint")
        return derived_score, derived_ece

    def _verify_registry(
        self,
        *,
        allowed_orphan: Path | None = None,
        allow_unlisted_orphans: bool = False,
    ) -> None:
        if self._manifest.get("version") != RELEASE_REGISTRY_VERSION or not isinstance(self._manifest.get("releases"), list):
            raise ValueError("release registry manifest has incompatible schema")
        seen: set[str] = set()
        listed_paths: set[Path] = set()
        for record in self._records():
            if not isinstance(record, Mapping):
                raise ValueError("release registry manifest has malformed record")
            release_id = record.get("release_id")
            if not isinstance(release_id, str) or release_id in seen:
                raise ValueError("release registry manifest has duplicate or invalid release_id")
            _safe_component(release_id, "release_id")
            seen.add(release_id)
            listed_paths.add(self._record_from_manifest(record, verify=True).release_path)
        actual_paths = {path.resolve() for path in self.releases_directory.glob("*.json")} if self.releases_directory.exists() else set()
        allowed = set() if allowed_orphan is None else {allowed_orphan.resolve()}
        if not allow_unlisted_orphans and (actual_paths != listed_paths | allowed or allowed & listed_paths):
            raise ValueError("release registry manifest rollback or unlisted immutable release artifact detected")

    def _record_from_manifest(self, record: Mapping[str, Any], *, verify: bool) -> ReleaseRecord:
        release_id = _safe_component(record.get("release_id"), "release_id")
        path = Path(str(record.get("path"))).resolve()
        expected_path = self._release_path(release_id)
        expected_sha = _require_sha256(record.get("sha256"), "release record sha256")
        if path != expected_path or _file_sha256(path, "release record") != expected_sha:
            raise ValueError("release record is missing or hash-mismatched")
        payload = _load_json(path, "release record")
        if (
            payload.get("version") != RELEASE_REGISTRY_VERSION
            or payload.get("kind") != RELEASE_KIND
            or payload.get("release_id") != release_id
            or not isinstance(payload.get("request"), Mapping)
            or payload.get("request_sha256") != _canonical_sha256(payload["request"])
            or not isinstance(payload.get("verified"), Mapping)
        ):
            raise ValueError("release record has incompatible schema")
        request = ReleaseRequest.from_dict(payload["request"])
        if request.release_id != release_id:
            raise ValueError("release record request does not match its release_id")
        if verify and self._release_payload(request) != payload:
            raise ValueError("release record provenance diverges from verified artifacts")
        return ReleaseRecord(release_id, path, expected_sha, request)

    def _records(self) -> list[dict[str, Any]]:
        records = self._manifest["releases"]
        assert isinstance(records, list)
        return records

    def _record_by_id(self, release_id: str) -> dict[str, Any] | None:
        return next((record for record in self._records() if record.get("release_id") == release_id), None)

    def _release_path(self, release_id: str) -> Path:
        return self.releases_directory / f"{release_id}.json"

    def _write_manifest(self, manifest: Mapping[str, Any] | None = None) -> None:
        _atomic_write_json(self.manifest_path, self._manifest if manifest is None else manifest)


__all__ = [
    "LineageArtifact",
    "RELEASE_KIND",
    "RELEASE_REGISTRY_VERSION",
    "ReleaseRecord",
    "ReleaseRegistry",
    "ReleaseRequest",
]
