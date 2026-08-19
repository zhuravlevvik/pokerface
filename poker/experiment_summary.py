"""Read-only, player-safe health summaries for experiment ledgers.

The experiment ledger is the source of truth for an experiment's learning
curve.  This module deliberately replays its verified event chain instead of
reading ``metrics.jsonl``: that file is merely a replaceable convenience
projection.  It never opens checkpoint tensors for output and never repairs
or rewrites ledger artifacts while producing a summary.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

from .experiments import EXPERIMENT_LEDGER_VERSION, ExperimentConfig, ExperimentLedger
from .training import UpdateMetrics


EXPERIMENT_SUMMARY_VERSION = 1
"""Schema version for the player-safe experiment summary JSON."""


_STATUS_PATTERN = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True)
class ExperimentHealthConfig:
    """Optional, explicit health thresholds for a learning curve.

    Every threshold defaults to ``None``.  Thus a summary never silently
    becomes a promotion/release gate; callers choose every alerting policy.
    """

    max_abs_kl: float | None = None
    max_clip_fraction: float | None = None
    min_entropy: float | None = None
    max_value_loss: float | None = None
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value))
            ):
                raise ValueError(f"health threshold {field.name} must be a finite number or None")

    def to_dict(self) -> dict[str, float | None]:
        return {field.name: None if (value := getattr(self, field.name)) is None else float(value) for field in fields(self)}


@dataclass(frozen=True)
class MetricRange:
    """Minimum, maximum and terminal value for one public scalar metric."""

    minimum: float | int
    maximum: float | int
    last: float | int

    def to_dict(self) -> dict[str, float | int]:
        return {"min": self.minimum, "max": self.maximum, "last": self.last}


@dataclass(frozen=True)
class CounterRange:
    """First and last cumulative counter values represented by the ledger."""

    first: int
    last: int

    def to_dict(self) -> dict[str, int]:
        return {"first": self.first, "last": self.last}


@dataclass(frozen=True)
class ExperimentHealthAlert:
    """One threshold crossing, ordered by ledger iteration and rule order."""

    iteration: int
    metric: str
    value: float
    threshold: float
    rule: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "iteration": self.iteration,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "rule": self.rule,
        }


@dataclass(frozen=True)
class ExperimentSummary:
    """Verified, tensor-free summary of one immutable PPO experiment."""

    trial_id: str
    trial_name: str
    config_sha256: str
    training_config: Mapping[str, Any]
    max_iterations: int
    evaluation_protocol_sha256: str
    code_revision: str
    status: str
    completed: bool
    iteration_range: CounterRange | None
    global_hands_range: CounterRange | None
    global_decisions_range: CounterRange | None
    metrics: Mapping[str, MetricRange]
    health: ExperimentHealthConfig
    alerts: tuple[ExperimentHealthAlert, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a player-safe JSON envelope with no checkpoint payloads."""

        return {
            "schema_version": EXPERIMENT_SUMMARY_VERSION,
            "trial": {
                "id": self.trial_id,
                "name": self.trial_name,
                "config_sha256": self.config_sha256,
            },
            "config": {
                "max_iterations": self.max_iterations,
                "training": dict(self.training_config),
            },
            "evaluation_protocol": {"sha256": self.evaluation_protocol_sha256},
            "code_revision": self.code_revision,
            "status": self.status,
            "completed": self.completed,
            "iteration_range": None if self.iteration_range is None else self.iteration_range.to_dict(),
            "counter_range": {
                "global_hands": None if self.global_hands_range is None else self.global_hands_range.to_dict(),
                "global_decisions": None if self.global_decisions_range is None else self.global_decisions_range.to_dict(),
            },
            "metrics": {name: values.to_dict() for name, values in self.metrics.items()},
            "health": {"thresholds": self.health.to_dict(), "alerts": [alert.to_dict() for alert in self.alerts]},
        }


def summarize_experiment(
    run_directory: str | Path,
    *,
    health: ExperimentHealthConfig | None = None,
) -> ExperimentSummary:
    """Revalidate a ledger and derive a tensor-free, deterministic summary.

    The construction below intentionally bypasses ``ExperimentLedger.__init__``:
    its normal operational constructor may recover an orphan event and
    regenerate ``metrics.jsonl``.  A health command must be read-only, so it
    invokes the ledger's same manifest/event validators on an in-memory view.
    """

    directory = Path(run_directory).resolve()
    config, ledger = _open_verified_ledger_read_only(directory)
    status = _verified_status(ledger, config)
    events = tuple(ledger._event_from_record(record) for record in ledger._records())
    metric_names = tuple(field.name for field in fields(UpdateMetrics))
    metrics = _metric_ranges(events, metric_names)
    selected_health = ExperimentHealthConfig() if health is None else health
    if not isinstance(selected_health, ExperimentHealthConfig):
        raise TypeError("health must be an ExperimentHealthConfig or None")
    alerts = _health_alerts(events, selected_health)
    iterations = tuple(event.iteration for event in events)
    contiguous = iterations == tuple(range(1, config.max_iterations + 1))
    completed = status == "completed" and contiguous
    return ExperimentSummary(
        trial_id=config.trial_id,
        trial_name=config.name,
        config_sha256=config.config_sha256,
        training_config=config.training.to_dict(),
        max_iterations=config.max_iterations,
        evaluation_protocol_sha256=config.evaluation_protocol_sha256,
        code_revision=config.code_revision,
        status=status,
        completed=completed,
        iteration_range=None if not events else CounterRange(iterations[0], iterations[-1]),
        global_hands_range=None if not events else CounterRange(events[0].global_hands, events[-1].global_hands),
        global_decisions_range=None if not events else CounterRange(events[0].global_decisions, events[-1].global_decisions),
        metrics=metrics,
        health=selected_health,
        alerts=alerts,
    )


def write_experiment_summary(summary: ExperimentSummary, path: str | Path) -> Path:
    """Atomically publish a JSON summary supplied by :func:`summarize_experiment`."""

    if not isinstance(summary, ExperimentSummary):
        raise TypeError("summary must be an ExperimentSummary")
    destination = Path(path)
    _atomic_write_json(destination, summary.to_dict())
    return destination


def _open_verified_ledger_read_only(run_directory: Path) -> tuple[ExperimentConfig, ExperimentLedger]:
    manifest_path = run_directory / "experiment-ledger" / "experiment-manifest.json"
    manifest = _load_json(manifest_path, "experiment ledger manifest")
    config_value = manifest.get("config")
    if not isinstance(config_value, Mapping):
        raise ValueError("experiment ledger manifest has no immutable ExperimentConfig")
    config = ExperimentConfig.from_dict(config_value)

    # Configure only the state needed by ExperimentLedger's validators.  Do
    # not call __init__: summary collection must not create, repair or rewrite
    # any file in the experiment directory.
    ledger = object.__new__(ExperimentLedger)
    ledger.run_directory = run_directory
    ledger.config = config
    ledger.directory = manifest_path.parent
    ledger.events_directory = ledger.directory / "events"
    ledger._manifest = manifest
    ledger._validate_manifest()
    return config, ledger


def _verified_status(ledger: ExperimentLedger, config: ExperimentConfig) -> str:
    status = _load_json(ledger.status_path, "experiment status")
    event = ledger.last_event
    state = status.get("state")
    if (
        status.get("version") != EXPERIMENT_LEDGER_VERSION
        or status.get("trial_id") != config.trial_id
        or status.get("config_sha256") != config.config_sha256
        or not isinstance(state, str)
        or _STATUS_PATTERN.fullmatch(state) is None
        or status.get("last_iteration") != ledger.last_iteration
        or status.get("last_event_sha256") != (None if event is None else event.event_sha256)
    ):
        raise ValueError("experiment status does not match the verified ledger identity")
    return state


def _metric_ranges(events: tuple[Any, ...], metric_names: tuple[str, ...]) -> dict[str, MetricRange]:
    if not events:
        return {}
    result: dict[str, MetricRange] = {}
    for name in metric_names:
        values = tuple(event.metrics[name] for event in events)
        result[name] = MetricRange(min(values), max(values), values[-1])
    return result


def _health_alerts(events: tuple[Any, ...], health: ExperimentHealthConfig) -> tuple[ExperimentHealthAlert, ...]:
    rules = (
        ("max_abs_kl", "approximate_kl", "abs(value) > max_abs_kl", lambda value, threshold: abs(value) > threshold),
        ("max_clip_fraction", "clip_fraction", "value > max_clip_fraction", lambda value, threshold: value > threshold),
        ("min_entropy", "entropy", "value < min_entropy", lambda value, threshold: value < threshold),
        ("max_value_loss", "value_loss", "value > max_value_loss", lambda value, threshold: value > threshold),
        ("max_gradient_norm", "gradient_norm", "value > max_gradient_norm", lambda value, threshold: value > threshold),
    )
    alerts: list[ExperimentHealthAlert] = []
    for event in events:
        for threshold_name, metric, rule, crossed in rules:
            threshold = getattr(health, threshold_name)
            if threshold is None:
                continue
            value = float(event.metrics[metric])
            if crossed(value, float(threshold)):
                alerts.append(ExperimentHealthAlert(event.iteration, metric, value, float(threshold), rule))
    return tuple(alerts)


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
    "EXPERIMENT_SUMMARY_VERSION",
    "CounterRange",
    "ExperimentHealthAlert",
    "ExperimentHealthConfig",
    "ExperimentSummary",
    "MetricRange",
    "summarize_experiment",
    "write_experiment_summary",
]
