"""Hash-verified, player-safe experiment ledger for long PPO trials.

The ledger intentionally does not own a rollout loop or tune a live model.  A
caller records an already-published native :class:`TrainingRunner` checkpoint
at a safe iteration boundary.  This keeps experiment tracking auditable and
allows later monitors/tuners to consume immutable metrics without receiving
observations, action histories, cards, or rollouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .model import TORCH_AVAILABLE
from .train_runner import CHECKPOINT_VERSION, TrainingRunConfig
from .training import UpdateMetrics

if TORCH_AVAILABLE:
    import torch


EXPERIMENT_LEDGER_VERSION = 1
"""On-disk schema for experiment manifests and checkpoint events."""


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for experiment ledgers; install the project with `.[rl]`.")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable contract for one bounded PPO trial.

    ``code_revision`` is deliberately explicit rather than inferred from a
    mutable checkout.  An operator must pin the source revision (for example,
    a commit SHA or immutable build identifier) before comparing trials.
    """

    name: str
    training: TrainingRunConfig
    max_iterations: int
    evaluation_protocol_path: str
    evaluation_protocol_sha256: str
    code_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if (
            not isinstance(self.code_revision, str)
            or not self.code_revision.strip()
            or self.code_revision.upper().startswith("REPLACE_")
        ):
            raise ValueError("code_revision must be explicit and non-empty")
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if self.max_iterations != self.training.run.iterations:
            raise ValueError("max_iterations must exactly match training.run.iterations")
        if self.training.run.checkpoint_every_iterations != 1:
            raise ValueError("experiment trials require checkpoint_every_iterations=1")
        if self.training.promotion.enabled or self.training.transition.enabled:
            raise ValueError("experiment trials cannot enable promotion or legacy curriculum transition")
        if not isinstance(self.evaluation_protocol_path, str) or not self.evaluation_protocol_path.strip():
            raise ValueError("evaluation_protocol_path must be explicit and non-empty")
        protocol_path = Path(self.evaluation_protocol_path).resolve()
        if not protocol_path.is_file():
            raise ValueError(f"evaluation protocol artifact is missing: {protocol_path}")
        object.__setattr__(self, "evaluation_protocol_path", str(protocol_path))
        digest = self.evaluation_protocol_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest.lower() != digest
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("evaluation_protocol_sha256 must be a lowercase SHA-256 digest")
        if _file_sha256(protocol_path) != digest:
            raise ValueError("evaluation protocol artifact SHA-256 does not match configuration")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "name": self.name,
            "training": self.training.to_dict(),
            "max_iterations": self.max_iterations,
            "evaluation_protocol_path": self.evaluation_protocol_path,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "code_revision": self.code_revision,
        }
        # ``TrainingRunConfig`` contains tuple-valued league fields.  The
        # ledger persists JSON, so expose the exact JSON-normalized shape for
        # both manifest equality and hashing.
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        training = value.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("experiment training config must be an object")
        try:
            return cls(
                name=value["name"],
                training=TrainingRunConfig.from_dict(training),
                max_iterations=value["max_iterations"],
                evaluation_protocol_path=value["evaluation_protocol_path"],
                evaluation_protocol_sha256=value["evaluation_protocol_sha256"],
                code_revision=value["code_revision"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed experiment configuration") from error

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def trial_id(self) -> str:
        """Stable human-readable identity, safe for a directory component."""

        prefix = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in self.name).strip("_")
        return f"{prefix[:48] or 'trial'}-{self.config_sha256[:16]}"


@dataclass(frozen=True)
class ExperimentEvent:
    """One immutable, player-safe metrics record linked to a full checkpoint."""

    iteration: int
    checkpoint_path: Path
    checkpoint_sha256: str
    event_path: Path
    event_sha256: str
    previous_event_sha256: str | None
    global_hands: int
    global_decisions: int
    metrics: Mapping[str, Any]


class ExperimentLedger:
    """Append a hash chain of full-checkpoint metrics and regenerate JSONL.

    Each final event references exactly one trainer iteration.  Event files
    are immutable; ``metrics.jsonl`` is a convenience projection regenerated
    from the verified chain and must never be treated as the source of truth.
    """

    def __init__(self, run_directory: str | Path, config: ExperimentConfig) -> None:
        _require_torch()
        self.run_directory = Path(run_directory)
        self.config = config
        self.directory = self.run_directory / "experiment-ledger"
        self.events_directory = self.directory / "events"
        self._manifest: dict[str, Any]
        if self.manifest_path.exists():
            self._manifest = _load_json(self.manifest_path, "experiment manifest")
            self._validate_manifest()
            self._recover_orphan_events()
        else:
            self._manifest = {
                "version": EXPERIMENT_LEDGER_VERSION,
                "trial_id": config.trial_id,
                "config": config.to_dict(),
                "config_sha256": config.config_sha256,
                "events": [],
            }
            _atomic_write_json(self.manifest_path, self._manifest)
        self._regenerate_metrics_jsonl()

    @property
    def manifest_path(self) -> Path:
        return self.directory / "experiment-manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.jsonl"

    @property
    def status_path(self) -> Path:
        return self.directory / "status.json"

    @property
    def failure_path(self) -> Path:
        return self.directory / "failure.json"

    @property
    def last_iteration(self) -> int:
        records = self._records()
        return 0 if not records else int(records[-1]["iteration"])

    @property
    def last_event(self) -> ExperimentEvent | None:
        records = self._records()
        return None if not records else self._event_from_record(records[-1])

    def record_checkpoint(self, checkpoint_path: str | Path) -> ExperimentEvent:
        """Record one native full checkpoint or return its existing event.

        The checkpoint is loaded with ``weights_only=True`` and its full run
        configuration, version, progress, SHA-256 and finite update metrics
        are all checked before any event is published.
        """

        event_data = self._checkpoint_event_data(Path(checkpoint_path))
        iteration = event_data["iteration"]
        existing = self._record_for_iteration(iteration)
        if existing is not None:
            if existing.get("checkpoint_sha256") != event_data["checkpoint_sha256"]:
                raise ValueError(f"experiment ledger already has a divergent checkpoint for iteration {iteration}")
            self._validate_event_record(existing, expected_previous=self._previous_event_hash_before(iteration))
            return self._event_from_record(existing)
        if iteration <= self.last_iteration:
            raise ValueError("experiment ledger iterations must be strictly increasing")
        event_path = self.events_directory / f"{iteration:08d}.json"
        if event_path.exists():
            # A final event can exist only after a crash between its atomic
            # publication and manifest update; recover it through the same
            # validation path instead of silently replacing it.
            self._recover_orphan_events()
            existing = self._record_for_iteration(iteration)
            if existing is None:
                raise ValueError("orphan experiment event is not a recoverable chain continuation")
            if existing.get("checkpoint_sha256") != event_data["checkpoint_sha256"]:
                raise ValueError(f"experiment ledger already has a divergent checkpoint for iteration {iteration}")
            return self._event_from_record(existing)
        previous = self._last_event_hash()
        event_payload = {
            "version": EXPERIMENT_LEDGER_VERSION,
            "trial_id": self.config.trial_id,
            "config_sha256": self.config.config_sha256,
            "iteration": iteration,
            "checkpoint": {
                "path": event_data["checkpoint_path"],
                "sha256": event_data["checkpoint_sha256"],
                "checkpoint_version": CHECKPOINT_VERSION,
                "run_config_sha256": event_data["run_config_sha256"],
            },
            "progress": {
                "global_hands": event_data["global_hands"],
                "global_decisions": event_data["global_decisions"],
            },
            "metrics": event_data["metrics"],
            "previous_event_sha256": previous,
        }
        _atomic_write_json(event_path, event_payload)
        record = {
            "iteration": iteration,
            "event_path": str(event_path),
            "event_sha256": _file_sha256(event_path),
            "checkpoint_path": event_data["checkpoint_path"],
            "checkpoint_sha256": event_data["checkpoint_sha256"],
        }
        self._records().append(record)
        try:
            self._write_manifest()
        except BaseException:
            # Keep the in-memory view aligned with the durable manifest.  The
            # already-published event is a recoverable orphan on retry.
            self._records().pop()
            raise
        self._regenerate_metrics_jsonl()
        return self._event_from_record(record)

    def recover_latest(self, checkpoint_path: str | Path) -> ExperimentEvent | None:
        """Idempotently add a published latest checkpoint when ledger lags.

        It never invents missed data: if the supplied checkpoint is older than
        the chain tip it fails, and if it is the recorded tip it returns that
        immutable event unchanged.
        """

        event_data = self._checkpoint_event_data(Path(checkpoint_path))
        iteration = int(event_data["iteration"])
        if iteration < self.last_iteration:
            raise ValueError("latest checkpoint predates the experiment ledger")
        if iteration == self.last_iteration:
            record = self._record_for_iteration(iteration)
            assert record is not None
            if record.get("checkpoint_sha256") != event_data["checkpoint_sha256"]:
                raise ValueError("latest checkpoint diverges from the experiment ledger")
            return self._event_from_record(record)
        return self.record_checkpoint(checkpoint_path)

    def write_status(self, state: str) -> Path:
        """Publish a minimal liveness snapshot without arbitrary run payloads."""

        if not state or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in state):
            raise ValueError("status state must use lowercase letters, digits, '_' or '-'")
        payload = {
            "version": EXPERIMENT_LEDGER_VERSION,
            "trial_id": self.config.trial_id,
            "config_sha256": self.config.config_sha256,
            "state": state,
            "last_iteration": self.last_iteration,
            "last_event_sha256": self._last_event_hash(),
        }
        _atomic_write_json(self.status_path, payload)
        return self.status_path

    def record_failure(self, kind: str, message: str) -> Path:
        """Publish a bounded failure marker; no traceback/rollout is stored."""

        if not kind or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in kind):
            raise ValueError("failure kind must use lowercase letters, digits, '_' or '-'")
        if not message or len(message) > 1_024:
            raise ValueError("failure message must be between 1 and 1024 characters")
        payload = {
            "version": EXPERIMENT_LEDGER_VERSION,
            "trial_id": self.config.trial_id,
            "config_sha256": self.config.config_sha256,
            "kind": kind,
            "message": message,
            "last_iteration": self.last_iteration,
            "last_event_sha256": self._last_event_hash(),
        }
        _atomic_write_json(self.failure_path, payload)
        return self.failure_path

    def _checkpoint_event_data(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError(f"experiment checkpoint is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("experiment event requires a compatible full TrainingRunner checkpoint")
        run_config, progress = payload.get("run_config"), payload.get("progress")
        if not isinstance(run_config, Mapping) or not isinstance(progress, Mapping):
            raise ValueError("experiment checkpoint has malformed run config/progress")
        if _canonical_sha256(dict(run_config)) != _canonical_sha256(self.config.training.to_dict()):
            raise ValueError("experiment checkpoint run configuration does not match trial config")
        iteration = progress.get("iteration")
        hands, decisions = progress.get("global_hands"), progress.get("global_decisions")
        if (
            not isinstance(iteration, int)
            or not 1 <= iteration <= self.config.max_iterations
            or not isinstance(hands, int)
            or hands < 0
            or not isinstance(decisions, int)
            or decisions < 0
        ):
            raise ValueError("experiment checkpoint has invalid bounded progress")
        metrics = _metrics_from_payload(payload)
        return {
            "iteration": iteration,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": _file_sha256(path),
            "run_config_sha256": _canonical_sha256(dict(run_config)),
            "global_hands": hands,
            "global_decisions": decisions,
            "metrics": metrics,
        }

    def _validate_manifest(self) -> None:
        manifest = self._manifest
        if (
            manifest.get("version") != EXPERIMENT_LEDGER_VERSION
            or manifest.get("trial_id") != self.config.trial_id
            or manifest.get("config_sha256") != self.config.config_sha256
            or manifest.get("config") != self.config.to_dict()
            or not isinstance(manifest.get("events"), list)
        ):
            raise ValueError("experiment manifest does not match immutable trial config")
        previous: str | None = None
        previous_iteration = 0
        for record in self._records():
            iteration = record.get("iteration")
            if not isinstance(iteration, int) or iteration <= previous_iteration:
                raise ValueError("experiment manifest events are not strictly ordered")
            self._validate_event_record(record, expected_previous=previous)
            previous = str(record["event_sha256"])
            previous_iteration = iteration

    def _validate_event_record(self, record: Mapping[str, Any], *, expected_previous: str | None) -> None:
        required = ("iteration", "event_path", "event_sha256", "checkpoint_path", "checkpoint_sha256")
        if any(key not in record for key in required):
            raise ValueError("experiment manifest event record is incomplete")
        iteration = record["iteration"]
        if not isinstance(iteration, int) or not 1 <= iteration <= self.config.max_iterations:
            raise ValueError("experiment manifest event iteration is invalid")
        event_path = Path(str(record["event_path"]))
        checkpoint_path = Path(str(record["checkpoint_path"]))
        if (
            event_path != self.events_directory / f"{iteration:08d}.json"
            or not event_path.is_file()
            or not isinstance(record["event_sha256"], str)
            or _file_sha256(event_path) != record["event_sha256"]
        ):
            raise ValueError("experiment event artifact is missing or hash-mismatched")
        event = _load_json(event_path, "experiment event")
        checkpoint = event.get("checkpoint")
        progress = event.get("progress")
        if (
            event.get("version") != EXPERIMENT_LEDGER_VERSION
            or event.get("trial_id") != self.config.trial_id
            or event.get("config_sha256") != self.config.config_sha256
            or event.get("iteration") != iteration
            or event.get("previous_event_sha256") != expected_previous
            or not isinstance(checkpoint, Mapping)
            or not isinstance(progress, Mapping)
            or checkpoint.get("path") != str(checkpoint_path)
            or checkpoint.get("sha256") != record["checkpoint_sha256"]
            or not checkpoint_path.is_file()
            or _file_sha256(checkpoint_path) != record["checkpoint_sha256"]
        ):
            raise ValueError("experiment event does not bind its checkpoint/provenance")
        actual = self._checkpoint_event_data(checkpoint_path)
        if (
            actual["iteration"] != iteration
            or actual["checkpoint_sha256"] != record["checkpoint_sha256"]
            or checkpoint.get("run_config_sha256") != actual["run_config_sha256"]
            or progress.get("global_hands") != actual["global_hands"]
            or progress.get("global_decisions") != actual["global_decisions"]
            or event.get("metrics") != actual["metrics"]
        ):
            raise ValueError("experiment event checkpoint metrics/progress diverge")

    def _recover_orphan_events(self) -> None:
        """Finish a safe crash window: event file published before manifest."""

        self.events_directory.mkdir(parents=True, exist_ok=True)
        listed = {Path(str(record["event_path"])) for record in self._records()}
        orphans = sorted(path for path in self.events_directory.glob("*.json") if path not in listed)
        if not orphans:
            return
        if len(orphans) != 1:
            raise ValueError("experiment ledger has multiple unrecorded event artifacts")
        path = orphans[0]
        event = _load_json(path, "orphan experiment event")
        iteration = event.get("iteration")
        if not isinstance(iteration, int) or iteration <= self.last_iteration or path != self.events_directory / f"{iteration:08d}.json":
            raise ValueError("orphan experiment event is not a recoverable continuation")
        checkpoint = event.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("path"), str) or not isinstance(checkpoint.get("sha256"), str):
            raise ValueError("orphan experiment event lacks checkpoint provenance")
        record = {
            "iteration": iteration,
            "event_path": str(path),
            "event_sha256": _file_sha256(path),
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
        }
        self._validate_event_record(record, expected_previous=self._last_event_hash())
        self._records().append(record)
        try:
            self._write_manifest()
        except BaseException:
            self._records().pop()
            raise

    def _regenerate_metrics_jsonl(self) -> None:
        lines: list[str] = []
        previous: str | None = None
        for record in self._records():
            self._validate_event_record(record, expected_previous=previous)
            event = _load_json(Path(str(record["event_path"])), "experiment event")
            lines.append(json.dumps(_metrics_row(event), sort_keys=True, separators=(",", ":"), allow_nan=False))
            previous = str(record["event_sha256"])
        _atomic_write_text(self.metrics_path, "\n".join(lines) + ("\n" if lines else ""))

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self._manifest)

    def _records(self) -> list[dict[str, Any]]:
        records = self._manifest.get("events")
        if not isinstance(records, list):  # Guarded by _validate_manifest.
            raise ValueError("experiment manifest events are malformed")
        return records

    def _record_for_iteration(self, iteration: int) -> dict[str, Any] | None:
        return next((record for record in self._records() if record.get("iteration") == iteration), None)

    def _last_event_hash(self) -> str | None:
        records = self._records()
        return None if not records else str(records[-1]["event_sha256"])

    def _previous_event_hash_before(self, iteration: int) -> str | None:
        previous: str | None = None
        for record in self._records():
            if record.get("iteration") == iteration:
                return previous
            previous = str(record["event_sha256"])
        return previous

    def _event_from_record(self, record: Mapping[str, Any]) -> ExperimentEvent:
        event_path = Path(str(record["event_path"]))
        event = _load_json(event_path, "experiment event")
        checkpoint = event["checkpoint"]
        progress = event["progress"]
        assert isinstance(checkpoint, Mapping) and isinstance(progress, Mapping)
        return ExperimentEvent(
            iteration=int(record["iteration"]),
            checkpoint_path=Path(str(record["checkpoint_path"])),
            checkpoint_sha256=str(record["checkpoint_sha256"]),
            event_path=event_path,
            event_sha256=str(record["event_sha256"]),
            previous_event_sha256=event.get("previous_event_sha256"),
            global_hands=int(progress["global_hands"]),
            global_decisions=int(progress["global_decisions"]),
            metrics=dict(event["metrics"]),
        )


def _metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload.get("metrics")
    if candidate is None:
        manifest = payload.get("manifest")
        candidate = manifest.get("last_metrics") if isinstance(manifest, Mapping) else None
    if not isinstance(candidate, Mapping):
        raise ValueError("experiment checkpoint has no completed UpdateMetrics")
    expected = {field.name for field in fields(UpdateMetrics)}
    if set(candidate) != expected:
        raise ValueError("experiment checkpoint UpdateMetrics schema is incompatible")
    result: dict[str, Any] = {}
    for field in fields(UpdateMetrics):
        value = candidate[field.name]
        if field.name == "samples":
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("experiment checkpoint metrics.samples must be a positive integer")
            result[field.name] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"experiment checkpoint metric {field.name!r} must be finite")
            result[field.name] = float(value)
    return result


def _metrics_row(event: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = event["checkpoint"]
    progress = event["progress"]
    assert isinstance(checkpoint, Mapping) and isinstance(progress, Mapping)
    return {
        "trial_id": event["trial_id"],
        "config_sha256": event["config_sha256"],
        "iteration": event["iteration"],
        "checkpoint_sha256": checkpoint["sha256"],
        "global_hands": progress["global_hands"],
        "global_decisions": progress["global_decisions"],
        **dict(event["metrics"]),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:  # pragma: no cover - filesystem-specific fallback.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "EXPERIMENT_LEDGER_VERSION",
    "ExperimentConfig",
    "ExperimentEvent",
    "ExperimentLedger",
]
