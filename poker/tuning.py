"""Deterministic, artifact-first PPO tuning specifications.

This module deliberately plans and compares experiments without starting a
trainer.  A sweep materialises one immutable, reviewable configuration per
trial; an operator may then start those trial directories independently.  The
comparison layer accepts only evidence bound to those exact trial identifiers
and to one pinned evaluation protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from types import MappingProxyType
from uuid import uuid4

from .curriculum import stage_spec
from .experiments import ExperimentConfig, ExperimentLedger
from .model import TORCH_AVAILABLE
from .promotion import PROMOTION_REPORT_VERSION, PromotionConfig, PromotionEvaluator
from .train_runner import CHECKPOINT_VERSION, TrainingRunConfig

if TORCH_AVAILABLE:
    import torch


TUNING_SCHEMA_VERSION = "1.0"
TUNING_TRIAL_KIND = "poker_ppo_tuning_trial_v1"
TUNING_COMPARISON_KIND = "poker_ppo_tuning_comparison_v1"
TUNING_EVALUATION_KIND = "poker_ppo_tuning_evaluation_v1"
TUNING_PROTOCOL_KIND = "poker_ppo_hu_promotion_protocol_v1"

ALLOWED_PPO_GRID_FIELDS = frozenset(
    {
        "learning_rate",
        "entropy_coefficient",
        "clip_ratio",
        "gae_lambda",
        "value_coefficient",
        "equity_coefficient",
        "expected_showdown_share_coefficient",
        "epochs",
        "minibatch_size",
    }
)
_INTEGER_GRID_FIELDS = frozenset({"epochs", "minibatch_size"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return str(value)


def _safe_name(value: str, label: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError(f"{label} must use only letters, digits, '.', '_' or '-'")
    return value


def hu_promotion_protocol_payload(training: TrainingRunConfig, promotion: PromotionConfig) -> dict[str, object]:
    """Canonical preregistration artifact for one standalone HU evaluation."""

    spec = stage_spec(training.run.stage)
    if spec.player_count != 2:
        raise ValueError("the initial tuning evidence protocol supports only heads-up stages A/B")
    protocol = {
        "stage": training.run.stage.value,
        "outcome_protocol": "fixed_deal_virtual_showdown_outcome_v1",
        "scalar_metric_protocol": "active_hands_expected_showdown_share_v1",
        "player_count": 2,
        "starting_stack": spec.starting_stack_chips(),
        "allowed_raise_actions": [
            action.value for action in sorted(spec.allowed_raise_actions, key=lambda action: action.value)
        ],
        "paired_position_seeds": True,
        "ci_method": "paired_position_seed_block_normal_v1",
        "seed_start": promotion.seed_start,
        "evaluation_run_seed": 0,
        "seed_blocks_per_opponent": promotion.hands_per_opponent // 2,
    }
    return {
        "schema_version": TUNING_SCHEMA_VERSION,
        "kind": TUNING_PROTOCOL_KIND,
        "evaluator": f"hu_promotion_report_v{PROMOTION_REPORT_VERSION}",
        "promotion_config": asdict(promotion),
        "promotion_protocol": protocol,
        "promotion_protocol_sha256": _sha256(protocol),
    }


def write_hu_promotion_protocol(
    training: TrainingRunConfig,
    promotion: PromotionConfig,
    path: str | Path,
) -> tuple[Path, str]:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(hu_promotion_protocol_payload(training, promotion)) + b"\n"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("evaluation protocol artifact already exists with different contents")
    else:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            _write_file(temporary, payload)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
    return destination, _file_sha256(destination)


@dataclass(frozen=True)
class TrialSpec:
    """One fully resolved, immutable training trial."""

    trial_id: str
    sweep_id: str
    seed: int
    ppo_overrides: Mapping[str, int | float]
    config: TrainingRunConfig
    evaluation_protocol_sha256: str
    evaluation_protocol_path: str
    base_config_sha256: str
    code_revision: str

    @property
    def run_config_sha256(self) -> str:
        return _sha256(self.config.to_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TUNING_SCHEMA_VERSION,
            "kind": TUNING_TRIAL_KIND,
            "trial_id": self.trial_id,
            "sweep_id": self.sweep_id,
            "seed": self.seed,
            "ppo_overrides": dict(sorted(self.ppo_overrides.items())),
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "evaluation_protocol_path": self.evaluation_protocol_path,
            "base_config_sha256": self.base_config_sha256,
            "code_revision": self.code_revision,
            "run_config_sha256": self.run_config_sha256,
            "config": self.config.to_dict(),
        }


@dataclass(frozen=True)
class SweepConfig:
    """Finite allowlisted PPO grid applied to one fixed training contract.

    The only per-trial changes are a value from ``grid`` and ``run.seed``.
    Stage, hand budget, model and league consequently remain identical.  A
    learning-rate override is mirrored into ``curriculum.base_learning_rate``
    to retain :class:`TrainingRunConfig`'s shared-LR invariant.
    """

    base_config: TrainingRunConfig
    grid: Mapping[str, Sequence[int | float]]
    seeds: Sequence[int]
    max_iterations: int
    evaluation_protocol_sha256: str
    evaluation_protocol_path: str
    code_revision: str
    name: str = "ppo-tuning"

    def __post_init__(self) -> None:
        _safe_name(self.name, "sweep name")
        _require_sha256(self.evaluation_protocol_sha256, "evaluation_protocol_sha256")
        if not isinstance(self.evaluation_protocol_path, str) or not self.evaluation_protocol_path.strip():
            raise ValueError("evaluation_protocol_path must be explicit and non-empty")
        protocol_path = Path(self.evaluation_protocol_path).resolve()
        if _file_sha256(protocol_path) != self.evaluation_protocol_sha256:
            raise ValueError("evaluation protocol artifact SHA-256 does not match configuration")
        object.__setattr__(self, "evaluation_protocol_path", str(protocol_path))
        if (
            not isinstance(self.code_revision, str)
            or not self.code_revision.strip()
            or self.code_revision.upper().startswith("REPLACE_")
        ):
            raise ValueError("code_revision must be explicit and non-empty")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.grid, Mapping):
            raise ValueError("grid must be a mapping of allowlisted PPO fields")
        if any(not isinstance(field, str) for field in self.grid):
            raise ValueError("grid field names must be strings")
        unexpected = set(self.grid) - ALLOWED_PPO_GRID_FIELDS
        if unexpected:
            raise ValueError(f"grid contains unsupported PPO fields: {sorted(unexpected)!r}")
        normalized_grid: dict[str, tuple[int | float, ...]] = {}
        for field_name in sorted(self.grid):
            raw_values = self.grid[field_name]
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)) or not raw_values:
                raise ValueError(f"grid[{field_name!r}] must be a non-empty sequence")
            values: list[int | float] = []
            for value in raw_values:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                    raise ValueError(f"grid[{field_name!r}] values must be finite numbers")
                if field_name in _INTEGER_GRID_FIELDS:
                    if not isinstance(value, int):
                        raise ValueError(f"grid[{field_name!r}] values must be integers")
                    normalized: int | float = value
                else:
                    normalized = float(value)
                values.append(normalized)
            if len(set(values)) != len(values):
                raise ValueError(f"grid[{field_name!r}] values must be unique")
            # Constructing a PPOConfig is the authoritative range check.
            for value in values:
                replace(self.base_config.ppo, **{field_name: value})
                if field_name == "learning_rate":
                    # PPO permits zero as a generic primitive, but a real
                    # TrainingRunConfig deliberately forbids a zero base LR.
                    replace(self.base_config.curriculum, base_learning_rate=float(value))
            normalized_grid[field_name] = tuple(sorted(values))
        if not isinstance(self.seeds, Sequence) or isinstance(self.seeds, (str, bytes)) or not self.seeds:
            raise ValueError("seeds must be a non-empty sequence of unique integers")
        normalized_seeds: list[int] = []
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seeds must be non-negative integers")
            normalized_seeds.append(seed)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "grid", MappingProxyType({field: normalized_grid[field] for field in sorted(normalized_grid)}))
        object.__setattr__(self, "seeds", tuple(sorted(normalized_seeds)))

    @property
    def base_config_sha256(self) -> str:
        return _sha256(self.base_config.to_dict())

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TUNING_SCHEMA_VERSION,
            "name": self.name,
            "base_config": self.base_config.to_dict(),
            "base_config_sha256": self.base_config_sha256,
            "grid": {field: list(values) for field, values in sorted(self.grid.items())},
            "seeds": list(self.seeds),
            "max_iterations": self.max_iterations,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "evaluation_protocol_path": self.evaluation_protocol_path,
            "code_revision": self.code_revision,
            "comparability": {
                "stage": self.base_config.run.stage.value,
                "hands_per_iteration": self.base_config.run.hands_per_iteration,
                "table_count": self.base_config.run.table_count,
                "model_sha256": _sha256(asdict(self.base_config.model)),
                "league_sha256": _sha256(asdict(self.base_config.league)),
            },
        }

    @property
    def sweep_id(self) -> str:
        return f"sweep-{_sha256(self.as_dict())[:16]}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepConfig":
        base = value.get("base_config")
        grid = value.get("grid")
        seeds = value.get("seeds")
        if not isinstance(base, Mapping) or not isinstance(grid, Mapping) or not isinstance(seeds, Sequence):
            raise ValueError("malformed sweep configuration")
        result = cls(
            base_config=TrainingRunConfig.from_dict(base),
            grid=dict(grid),
            seeds=tuple(seeds),
            max_iterations=value.get("max_iterations"),
            evaluation_protocol_sha256=value.get("evaluation_protocol_sha256"),
            evaluation_protocol_path=value.get("evaluation_protocol_path"),
            code_revision=value.get("code_revision"),
            name=value.get("name", "ppo-tuning"),
        )
        if _canonical_json(dict(value)) != _canonical_json(result.as_dict()):
            raise ValueError("sweep configuration derived hashes or comparability fields do not match")
        return result

    def expand_trials(self) -> tuple[TrialSpec, ...]:
        """Return a stable Cartesian product, independent of input ordering."""

        fields = tuple(sorted(self.grid))
        value_sets = tuple(self.grid[field] for field in fields)
        combinations = product(*value_sets) if fields else ((),)
        result: list[TrialSpec] = []
        for values in combinations:
            overrides = dict(zip(fields, values, strict=True))
            ppo = replace(self.base_config.ppo, **overrides)
            curriculum = self.base_config.curriculum
            if "learning_rate" in overrides:
                curriculum = replace(curriculum, base_learning_rate=float(overrides["learning_rate"]))
            for seed in self.seeds:
                run = replace(
                    self.base_config.run,
                    seed=seed,
                    iterations=self.max_iterations,
                    checkpoint_every_iterations=1,
                )
                config = replace(self.base_config, run=run, ppo=ppo, curriculum=curriculum)
                # This should be guaranteed by the construction above; leave
                # it explicit so future grid extensions cannot silently break
                # the shared-LR invariant.
                if config.ppo.learning_rate != config.curriculum.base_learning_rate:
                    raise RuntimeError("materialized trial violates the PPO/curriculum learning-rate invariant")
                trial_input = {
                    "sweep_id": self.sweep_id,
                    "seed": seed,
                    "ppo_overrides": dict(sorted(overrides.items())),
                    "run_config_sha256": _sha256(config.to_dict()),
                }
                trial_id = f"trial-{_sha256(trial_input)[:16]}"
                result.append(
                    TrialSpec(
                        trial_id,
                        self.sweep_id,
                        seed,
                        MappingProxyType(dict(sorted(overrides.items()))),
                        config,
                        self.evaluation_protocol_sha256,
                        self.evaluation_protocol_path,
                        self.base_config_sha256,
                        self.code_revision,
                    )
                )
        return tuple(result)


def expand_sweep(config: SweepConfig) -> tuple[TrialSpec, ...]:
    """Convenience wrapper for callers that prefer a function API."""

    return config.expand_trials()


def single_experiment_trial(experiment: ExperimentConfig) -> TrialSpec:
    """Adapt one immutable experiment contract for the evidence workflow.

    A standalone experiment is deliberately *not* a one-item sweep: it has no
    mutable grid and therefore cannot be submitted to ``compare``.  It can,
    however, use exactly the same checkpoint, ledger and fixed-promotion
    validation as a materialized tuning trial.

    ``ExperimentLedger`` persists ``ExperimentConfig.name`` as the public
    tuning identity expected by the sealed report validator.  Keep that value
    as ``trial_id`` rather than using ``ExperimentConfig.trial_id`` (the
    ledger's internal hash-suffixed identifier).
    """

    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    return TrialSpec(
        trial_id=experiment.name,
        sweep_id=f"standalone-{experiment.config_sha256[:16]}",
        seed=experiment.training.run.seed,
        ppo_overrides=MappingProxyType({}),
        config=experiment.training,
        evaluation_protocol_sha256=experiment.evaluation_protocol_sha256,
        evaluation_protocol_path=experiment.evaluation_protocol_path,
        base_config_sha256=_sha256(experiment.training.to_dict()),
        code_revision=experiment.code_revision,
    )


def load_sweep_config(path: str | Path) -> SweepConfig:
    return SweepConfig.from_dict(_load_json(Path(path), "tuning sweep config"))


def write_sweep_config(config: SweepConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(config.as_dict()) + b"\n"
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        _write_file(temporary, payload)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


@dataclass(frozen=True)
class MaterializedTrial:
    """Paths to the immutable operator inputs for one planned trial."""

    spec: TrialSpec
    directory: Path
    config_path: Path
    experiment_config_path: Path
    manifest_path: Path


def _write_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - some network filesystems reject this.
        pass
    finally:
        os.close(descriptor)


def _trial_payloads(spec: TrialSpec) -> tuple[bytes, bytes, bytes]:
    config_payload = _canonical_json(spec.config.to_dict()) + b"\n"
    experiment = ExperimentConfig(
        name=spec.trial_id,
        training=spec.config,
        max_iterations=spec.config.run.iterations,
        evaluation_protocol_sha256=spec.evaluation_protocol_sha256,
        evaluation_protocol_path=spec.evaluation_protocol_path,
        code_revision=spec.code_revision,
    )
    experiment_payload = _canonical_json(experiment.to_dict()) + b"\n"
    manifest = spec.as_dict()
    manifest["config_file"] = "config.json"
    manifest["config_file_sha256"] = sha256(config_payload).hexdigest()
    manifest["experiment_config_file"] = "experiment.json"
    manifest["experiment_config_file_sha256"] = sha256(experiment_payload).hexdigest()
    return config_payload, experiment_payload, _canonical_json(manifest) + b"\n"


def _existing_trial(
    spec: TrialSpec,
    directory: Path,
    config_payload: bytes,
    experiment_payload: bytes,
    manifest_payload: bytes,
) -> MaterializedTrial:
    config_path = directory / "config.json"
    experiment_path = directory / "experiment.json"
    manifest_path = directory / "manifest.json"
    if not directory.is_dir() or not config_path.is_file() or not experiment_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"trial directory collision for {spec.trial_id}")
    if (
        config_path.read_bytes() != config_payload
        or experiment_path.read_bytes() != experiment_payload
        or manifest_path.read_bytes() != manifest_payload
    ):
        raise ValueError(f"trial directory collision for {spec.trial_id}")
    return MaterializedTrial(spec, directory, config_path, experiment_path, manifest_path)


def materialize_sweep(config: SweepConfig, root: str | Path) -> tuple[MaterializedTrial, ...]:
    """Atomically materialise deterministic configs/manifests under trial ids.

    Existing byte-identical artifacts are accepted as an idempotent retry;
    every other pre-existing path is a collision and is never overwritten.
    """

    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(destination_root)
    materialized: list[MaterializedTrial] = []
    for spec in config.expand_trials():
        directory = destination_root / spec.trial_id
        config_payload, experiment_payload, manifest_payload = _trial_payloads(spec)
        intent = destination_root / f".{spec.trial_id}.intent"
        intent_payload = _canonical_json(
            {
                "trial_id": spec.trial_id,
                "config_sha256": sha256(config_payload).hexdigest(),
                "experiment_sha256": sha256(experiment_payload).hexdigest(),
                "manifest_sha256": sha256(manifest_payload).hexdigest(),
            }
        ) + b"\n"
        if directory.exists() and (directory / "manifest.json").is_file():
            result = _existing_trial(spec, directory, config_payload, experiment_payload, manifest_payload)
            if intent.exists():
                if intent.read_bytes() != intent_payload:
                    raise ValueError(f"trial reservation collision for {spec.trial_id}")
                intent.unlink()
                _fsync_directory(destination_root)
            materialized.append(result)
            continue
        if directory.exists() and not intent.exists():
            raise ValueError(f"trial directory collision for {spec.trial_id}")
        temporary = destination_root / f".{spec.trial_id}.{uuid4().hex}.tmp"
        created_intent = False
        try:
            if intent.exists():
                if intent.read_bytes() != intent_payload:
                    raise ValueError(f"trial reservation collision for {spec.trial_id}")
            else:
                try:
                    with intent.open("xb") as stream:
                        stream.write(intent_payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    created_intent = True
                    _fsync_directory(destination_root)
                except FileExistsError:
                    if intent.read_bytes() != intent_payload:
                        raise ValueError(f"trial reservation collision for {spec.trial_id}")
            if not directory.exists():
                try:
                    directory.mkdir()
                except FileExistsError:
                    if created_intent:
                        intent.unlink()
                    raise ValueError(f"trial directory collision for {spec.trial_id}") from None
            temporary.mkdir()
            config_path = temporary / "config.json"
            experiment_path = temporary / "experiment.json"
            manifest_path = temporary / "manifest.json"
            _write_file(config_path, config_payload)
            _write_file(experiment_path, experiment_payload)
            _write_file(manifest_path, manifest_payload)
            _fsync_directory(temporary)
            for source, target, expected in (
                (config_path, directory / "config.json", config_payload),
                (experiment_path, directory / "experiment.json", experiment_payload),
            ):
                if target.exists():
                    if target.read_bytes() != expected:
                        raise ValueError(f"trial directory collision for {spec.trial_id}")
                    source.unlink()
                else:
                    os.replace(source, target)
            # Manifest is the commit marker and is published last.
            target_manifest = directory / "manifest.json"
            if target_manifest.exists():
                if target_manifest.read_bytes() != manifest_payload:
                    raise ValueError(f"trial directory collision for {spec.trial_id}")
                manifest_path.unlink()
            else:
                os.replace(manifest_path, target_manifest)
            _fsync_directory(directory)
            if intent.exists():
                intent.unlink()
            _fsync_directory(destination_root)
            materialized.append(
                MaterializedTrial(
                    spec,
                    directory,
                    directory / "config.json",
                    directory / "experiment.json",
                    directory / "manifest.json",
                )
            )
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
    return tuple(materialized)


@dataclass(frozen=True)
class TuningEvidence:
    """Minimal standardized, player-safe evidence for exactly one trial."""

    trial_id: str
    full_checkpoint_path: Path
    full_checkpoint_sha256: str
    evaluation_report_path: Path
    evaluation_report_sha256: str
    evaluation_protocol_sha256: str
    run_config_sha256: str
    score_lower_ci: float
    expected_showdown_share_ece: float
    illegal_action_count: int
    passed: bool

    def __post_init__(self) -> None:
        _safe_name(self.trial_id, "trial_id")
        object.__setattr__(self, "full_checkpoint_path", Path(self.full_checkpoint_path))
        object.__setattr__(self, "evaluation_report_path", Path(self.evaluation_report_path))
        _require_sha256(self.full_checkpoint_sha256, "full_checkpoint_sha256")
        _require_sha256(self.evaluation_report_sha256, "evaluation_report_sha256")
        _require_sha256(self.evaluation_protocol_sha256, "evaluation_protocol_sha256")
        _require_sha256(self.run_config_sha256, "run_config_sha256")
        for value, label in (
            (self.score_lower_ci, "score_lower_ci"),
            (self.expected_showdown_share_ece, "expected_showdown_share_ece"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{label} must be finite")
        if not 0.0 <= self.expected_showdown_share_ece <= 1.0:
            raise ValueError("expected_showdown_share_ece must be in [0, 1]")
        if isinstance(self.illegal_action_count, bool) or not isinstance(self.illegal_action_count, int) or self.illegal_action_count < 0:
            raise ValueError("illegal_action_count must be a non-negative integer")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")

    def as_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "full_checkpoint_path": str(self.full_checkpoint_path),
            "full_checkpoint_sha256": self.full_checkpoint_sha256,
            "evaluation_report_path": str(self.evaluation_report_path),
            "evaluation_report_sha256": self.evaluation_report_sha256,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "run_config_sha256": self.run_config_sha256,
            "score_lower_ci": self.score_lower_ci,
            "expected_showdown_share_ece": self.expected_showdown_share_ece,
            "illegal_action_count": self.illegal_action_count,
            "passed": self.passed,
        }

    @classmethod
    def from_artifacts(
        cls,
        trial: TrialSpec,
        full_checkpoint_path: str | Path,
        evaluation_report_path: str | Path,
    ) -> "TuningEvidence":
        checkpoint_path = Path(full_checkpoint_path).resolve()
        report_path = Path(evaluation_report_path).resolve()
        checkpoint_sha = _file_sha256(checkpoint_path)
        report_sha = _file_sha256(report_path)
        report = _load_json(report_path, "tuning evaluation report")
        decision = report.get("decision")
        metrics = report.get("metrics")
        if not isinstance(decision, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("tuning evaluation report lacks decision/metrics")
        evidence = cls(
            trial_id=str(report.get("trial_id")),
            full_checkpoint_path=checkpoint_path,
            full_checkpoint_sha256=checkpoint_sha,
            evaluation_report_path=report_path,
            evaluation_report_sha256=report_sha,
            evaluation_protocol_sha256=str(report.get("evaluation_protocol_sha256")),
            run_config_sha256=str(report.get("run_config_sha256")),
            score_lower_ci=metrics.get("score_lower_ci"),
            expected_showdown_share_ece=metrics.get("expected_showdown_share_ece"),
            illegal_action_count=metrics.get("illegal_action_count"),
            passed=decision.get("passed"),
        )
        _validate_evidence_artifacts(trial, evidence)
        return evidence

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TuningEvidence":
        try:
            return cls(
                trial_id=value["trial_id"],
                full_checkpoint_path=Path(str(value["full_checkpoint_path"])),
                full_checkpoint_sha256=value["full_checkpoint_sha256"],
                evaluation_report_path=Path(str(value["evaluation_report_path"])),
                evaluation_report_sha256=value["evaluation_report_sha256"],
                evaluation_protocol_sha256=value["evaluation_protocol_sha256"],
                run_config_sha256=value["run_config_sha256"],
                score_lower_ci=value["score_lower_ci"],
                expected_showdown_share_ece=value["expected_showdown_share_ece"],
                illegal_action_count=value["illegal_action_count"],
                passed=value["passed"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed tuning evidence") from error


@dataclass(frozen=True)
class TuningComparisonEntry:
    trial: TrialSpec
    evidence: TuningEvidence
    rank: int | None

    def as_dict(self) -> dict[str, object]:
        return {"trial": self.trial.as_dict(), "evidence": self.evidence.as_dict(), "rank": self.rank}


@dataclass(frozen=True)
class TuningComparisonReport:
    sweep_id: str
    evaluation_protocol_sha256: str
    entries: tuple[TuningComparisonEntry, ...]

    @property
    def winner(self) -> TuningComparisonEntry | None:
        return next((entry for entry in self.entries if entry.rank == 1), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TUNING_SCHEMA_VERSION,
            "kind": TUNING_COMPARISON_KIND,
            "sweep_id": self.sweep_id,
            "evaluation_protocol_sha256": self.evaluation_protocol_sha256,
            "winner_trial_id": None if self.winner is None else self.winner.trial.trial_id,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_json(self.as_dict()) + b"\n"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            _write_file(temporary, payload)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


def publish_tuning_evaluation(
    trial: TrialSpec,
    full_checkpoint_path: str | Path,
    experiment_ledger_manifest_path: str | Path,
    promotion_report_path: str | Path,
    promotion_archive_manifest_path: str | Path,
    report_path: str | Path,
) -> TuningEvidence:
    """Seal a completed ledger plus verified HU promotion evidence.

    No metric or pass/fail flag is accepted from the caller.  They are derived
    from the immutable PromotionEvaluator report after its archive, source
    checkpoint, preregistered protocol and completed ledger are revalidated.
    """

    checkpoint = Path(full_checkpoint_path).resolve()
    _validate_trial_checkpoint(trial, checkpoint)
    ledger = _validate_completed_ledger(trial, checkpoint, Path(experiment_ledger_manifest_path).resolve())
    promotion = _validate_promotion_evidence(
        trial,
        checkpoint,
        Path(promotion_report_path).resolve(),
        Path(promotion_archive_manifest_path).resolve(),
    )
    payload = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "kind": TUNING_EVALUATION_KIND,
        "trial_id": trial.trial_id,
        "full_checkpoint_path": str(checkpoint),
        "full_checkpoint_sha256": _file_sha256(checkpoint),
        "run_config_sha256": trial.run_config_sha256,
        "evaluation_protocol_sha256": trial.evaluation_protocol_sha256,
        "lineage": {
            "experiment_ledger_manifest_path": str(ledger["manifest_path"]),
            "experiment_ledger_manifest_sha256": ledger["manifest_sha256"],
            "promotion_report_path": str(promotion["report_path"]),
            "promotion_report_sha256": promotion["report_sha256"],
            "promotion_archive_manifest_path": str(promotion["archive_manifest_path"]),
            "promotion_archive_manifest_sha256": promotion["archive_manifest_sha256"],
            "evaluation_protocol_path": trial.evaluation_protocol_path,
            "evaluation_protocol_sha256": trial.evaluation_protocol_sha256,
        },
        "metrics": dict(promotion["metrics"]),
        "decision": {"passed": promotion["passed"]},
    }
    destination = Path(report_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload) + b"\n"
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError("tuning evaluation report already exists with different evidence")
    else:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            _write_file(temporary, encoded)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
    return TuningEvidence.from_artifacts(trial, checkpoint, destination)


def _validate_trial_checkpoint(trial: TrialSpec, path: Path) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to verify tuning checkpoints; install the project with `.[rl]`.")
    if not path.is_file():
        raise ValueError(f"tuning full checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("tuning evidence requires a compatible full TrainingRunner checkpoint")
    run_config = payload.get("run_config")
    progress = payload.get("progress")
    if not isinstance(run_config, Mapping) or not isinstance(progress, Mapping):
        raise ValueError("tuning checkpoint has malformed run config/progress")
    if _sha256(dict(run_config)) != trial.run_config_sha256:
        raise ValueError("tuning checkpoint run config does not match its trial")
    if progress.get("iteration") != trial.config.run.iterations:
        raise ValueError("tuning checkpoint has not completed the declared trial budget")


def _validate_completed_ledger(trial: TrialSpec, checkpoint: Path, manifest_path: Path) -> dict[str, object]:
    if manifest_path.name != "experiment-manifest.json" or manifest_path.parent.name != "experiment-ledger":
        raise ValueError("tuning evidence requires a native experiment ledger manifest")
    manifest_sha = _file_sha256(manifest_path)
    manifest = _load_json(manifest_path, "experiment ledger manifest")
    config_data = manifest.get("config")
    if not isinstance(config_data, Mapping):
        raise ValueError("experiment ledger manifest lacks immutable config")
    experiment = ExperimentConfig.from_dict(config_data)
    if (
        experiment.name != trial.trial_id
        or experiment.training.to_dict() != trial.config.to_dict()
        or experiment.code_revision != trial.code_revision
        or experiment.evaluation_protocol_path != trial.evaluation_protocol_path
        or experiment.evaluation_protocol_sha256 != trial.evaluation_protocol_sha256
        or experiment.max_iterations != trial.config.run.iterations
    ):
        raise ValueError("experiment ledger does not belong to the tuning trial")
    ledger = ExperimentLedger(manifest_path.parent.parent, experiment)
    if ledger.manifest_path.resolve() != manifest_path or _file_sha256(manifest_path) != manifest_sha:
        raise ValueError("experiment ledger changed during tuning evidence validation")
    event = ledger.last_event
    if event is None or event.iteration != experiment.max_iterations:
        raise ValueError("experiment ledger is not complete through the declared trial budget")
    if event.checkpoint_path.resolve() != checkpoint or event.checkpoint_sha256 != _file_sha256(checkpoint):
        raise ValueError("experiment ledger final event does not bind the tuning checkpoint")
    status = _load_json(ledger.status_path, "experiment status")
    if (
        status.get("version") != 1
        or status.get("trial_id") != experiment.trial_id
        or status.get("config_sha256") != experiment.config_sha256
        or status.get("state") != "completed"
        or status.get("last_iteration") != event.iteration
        or status.get("last_event_sha256") != event.event_sha256
    ):
        raise ValueError("experiment ledger does not have a completed terminal status")
    return {"manifest_path": manifest_path, "manifest_sha256": manifest_sha}


def _validate_promotion_evidence(
    trial: TrialSpec,
    checkpoint: Path,
    report_path: Path,
    archive_manifest_path: Path,
) -> dict[str, object]:
    protocol_artifact = _load_json(Path(trial.evaluation_protocol_path), "evaluation protocol artifact")
    promotion_config_data = protocol_artifact.get("promotion_config")
    expected_protocol = protocol_artifact.get("promotion_protocol")
    if (
        protocol_artifact.get("schema_version") != TUNING_SCHEMA_VERSION
        or protocol_artifact.get("kind") != TUNING_PROTOCOL_KIND
        or protocol_artifact.get("evaluator") != f"hu_promotion_report_v{PROMOTION_REPORT_VERSION}"
        or not isinstance(promotion_config_data, Mapping)
        or not isinstance(expected_protocol, Mapping)
        or protocol_artifact.get("promotion_protocol_sha256") != _sha256(dict(expected_protocol))
    ):
        raise ValueError("evaluation protocol artifact is incompatible")
    try:
        promotion_config = PromotionConfig(**dict(promotion_config_data))
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation protocol has invalid promotion config") from error
    if _canonical_json(protocol_artifact) != _canonical_json(hu_promotion_protocol_payload(trial.config, promotion_config)):
        raise ValueError("evaluation protocol is not canonical for the tuning trial stage/config")
    report = _load_json(report_path, "promotion evaluation report")
    candidate = report.get("candidate")
    decision = report.get("decision")
    suite = report.get("suite")
    run_context = candidate.get("run_context") if isinstance(candidate, Mapping) else None
    expected_run_seed = expected_protocol.get("evaluation_run_seed")
    if (
        report.get("promotion_report_version") != PROMOTION_REPORT_VERSION
        or _canonical_json(report.get("promotion_config")) != _canonical_json(dict(promotion_config_data))
        or report.get("protocol") != dict(expected_protocol)
        or report.get("protocol_sha256") != protocol_artifact["promotion_protocol_sha256"]
        or not isinstance(candidate, Mapping)
        or candidate.get("source_full_checkpoint_sha256") != _file_sha256(checkpoint)
        or not isinstance(run_context, Mapping)
        or run_context.get("run_config_sha256") != trial.run_config_sha256
        or run_context.get("evaluation_protocol_sha256") != trial.evaluation_protocol_sha256
        or run_context.get("evaluation_run_seed") != expected_run_seed
        or not isinstance(decision, Mapping)
        or not isinstance(decision.get("accepted"), bool)
        or not isinstance(decision.get("reasons"), list)
        or (decision.get("accepted") and decision.get("reasons") != [])
        or (not decision.get("accepted") and not decision.get("reasons"))
        or not isinstance(suite, Mapping)
        or suite.get("schema_version") != "2.0"
    ):
        raise ValueError("promotion report does not match the completed trial and preregistered protocol")
    if isinstance(expected_run_seed, bool) or not isinstance(expected_run_seed, int) or expected_run_seed < 0:
        raise ValueError("evaluation protocol has an invalid fixed evaluation RNG seed")
    evaluator = PromotionEvaluator(promotion_config, report_path.parent.parent, run_seed=expected_run_seed)
    if evaluator.archive_manifest_path.resolve() != archive_manifest_path or not archive_manifest_path.is_file():
        raise ValueError("promotion archive manifest path does not match its evaluator")
    report_sha = _file_sha256(report_path)
    archive_sha = _file_sha256(archive_manifest_path)
    decisions = evaluator.archive_manifest.get("decisions")
    if not isinstance(decisions, list) or not any(
        isinstance(item, Mapping)
        and Path(str(item.get("report_path"))).resolve() == report_path
        and item.get("report_sha256") == report_sha
        and item.get("source_full_checkpoint_sha256") == _file_sha256(checkpoint)
        and item.get("accepted") is decision.get("accepted")
        for item in decisions
    ):
        raise ValueError("promotion archive does not contain the accepted tuning report")
    matchups = suite.get("matchups")
    if not isinstance(matchups, list):
        raise ValueError("promotion suite has malformed matchups")
    opponents = report.get("opponents")
    if not isinstance(opponents, list):
        raise ValueError("promotion report has malformed opponent registry")
    expected_bot_seed = expected_run_seed + promotion_config.seed_start
    for opponent in opponents:
        if not isinstance(opponent, Mapping) or not str(opponent.get("name", "")).startswith("baseline:"):
            continue
        baseline_name = str(opponent["name"]).split(":", 1)[1]
        expected_seed = None if baseline_name in {"rule", "tight"} else expected_bot_seed
        if opponent.get("seed") != expected_seed:
            raise ValueError("promotion baseline RNG does not match the preregistered fixed evaluation seed")
    baselines = [item for item in matchups if isinstance(item, Mapping) and str(item.get("opponent", "")).startswith("baseline:")]
    if not baselines:
        raise ValueError("promotion suite has no baseline evidence")
    try:
        raw_lower = [item["bb_per_100_ci95_low"] for item in baselines]
        raw_ece = [item["expected_showdown_share"]["expected_calibration_error"] for item in baselines]
        raw_illegal = [item["model_diagnostics"]["illegal_action_count"] for item in baselines]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (*raw_lower, *raw_ece)):
            raise ValueError("boolean/non-numeric evaluation metric")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_illegal):
            raise ValueError("boolean/non-integer illegal-action count")
        lower_ci = min(float(value) for value in raw_lower)
        illegal = sum(raw_illegal)
        ece = max(float(value) for value in raw_ece)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("promotion suite lacks required tuning metrics") from error
    if not isfinite(lower_ci) or not isfinite(ece) or not 0.0 <= ece <= 1.0 or illegal != 0:
        raise ValueError("promotion suite tuning metrics are invalid or contain illegal actions")
    return {
        "report_path": report_path,
        "report_sha256": report_sha,
        "archive_manifest_path": archive_manifest_path,
        "archive_manifest_sha256": archive_sha,
        "metrics": {
            "score_lower_ci": lower_ci,
            "expected_showdown_share_ece": ece,
            "illegal_action_count": illegal,
        },
        "passed": bool(decision["accepted"]),
    }


def _validate_evidence_artifacts(trial: TrialSpec, evidence: TuningEvidence) -> None:
    _validate_trial_checkpoint(trial, evidence.full_checkpoint_path)
    if _file_sha256(evidence.full_checkpoint_path) != evidence.full_checkpoint_sha256:
        raise ValueError("tuning checkpoint SHA-256 does not match evidence")
    if evidence.run_config_sha256 != trial.run_config_sha256:
        raise ValueError("tuning evidence run config does not match its trial")
    if evidence.evaluation_protocol_sha256 != trial.evaluation_protocol_sha256:
        raise ValueError("tuning evidence evaluation protocol does not match its trial")
    if not evidence.evaluation_report_path.is_file():
        raise ValueError(f"tuning evaluation report is missing: {evidence.evaluation_report_path}")
    if _file_sha256(evidence.evaluation_report_path) != evidence.evaluation_report_sha256:
        raise ValueError("tuning evaluation report SHA-256 does not match evidence")
    report = _load_json(evidence.evaluation_report_path, "tuning evaluation report")
    metrics = report.get("metrics")
    decision = report.get("decision")
    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("tuning evaluation report lacks verified lineage")
    ledger = _validate_completed_ledger(
        trial,
        evidence.full_checkpoint_path,
        Path(str(lineage.get("experiment_ledger_manifest_path"))).resolve(),
    )
    promotion = _validate_promotion_evidence(
        trial,
        evidence.full_checkpoint_path,
        Path(str(lineage.get("promotion_report_path"))).resolve(),
        Path(str(lineage.get("promotion_archive_manifest_path"))).resolve(),
    )
    if (
        report.get("schema_version") != TUNING_SCHEMA_VERSION
        or report.get("kind") != TUNING_EVALUATION_KIND
        or report.get("trial_id") != trial.trial_id
        or report.get("full_checkpoint_path") != str(evidence.full_checkpoint_path)
        or report.get("full_checkpoint_sha256") != evidence.full_checkpoint_sha256
        or report.get("run_config_sha256") != trial.run_config_sha256
        or report.get("evaluation_protocol_sha256") != trial.evaluation_protocol_sha256
        or lineage.get("experiment_ledger_manifest_sha256") != ledger["manifest_sha256"]
        or lineage.get("promotion_report_sha256") != promotion["report_sha256"]
        or lineage.get("promotion_archive_manifest_sha256") != promotion["archive_manifest_sha256"]
        or lineage.get("evaluation_protocol_path") != trial.evaluation_protocol_path
        or lineage.get("evaluation_protocol_sha256") != trial.evaluation_protocol_sha256
        or not isinstance(metrics, Mapping)
        or not isinstance(decision, Mapping)
        or metrics.get("score_lower_ci") != evidence.score_lower_ci
        or metrics.get("expected_showdown_share_ece") != evidence.expected_showdown_share_ece
        or metrics.get("illegal_action_count") != evidence.illegal_action_count
        or decision.get("passed") is not evidence.passed
        or dict(metrics) != promotion["metrics"]
        or decision.get("passed") is not promotion["passed"]
    ):
        raise ValueError("tuning evaluation report does not bind its checkpoint and metrics")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path}")
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


def compare_tuning_evidence(config: SweepConfig, evidence: Sequence[TuningEvidence]) -> TuningComparisonReport:
    """Fail closed unless one valid evidence record exists for every trial.

    Passing trials are sorted by descending lower confidence bound, then by
    ascending expected-showdown-share ECE, then by stable trial id.  Failed
    trials remain in the report but have no rank.
    """

    expected = {spec.trial_id: spec for spec in config.expand_trials()}
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise ValueError("evidence must be a sequence of TuningEvidence")
    by_trial: dict[str, TuningEvidence] = {}
    checkpoint_hashes: set[str] = set()
    report_hashes: set[str] = set()
    for item in evidence:
        if not isinstance(item, TuningEvidence):
            raise ValueError("evidence items must be TuningEvidence")
        if item.trial_id not in expected:
            raise ValueError(f"evidence contains unknown trial_id {item.trial_id!r}")
        if item.trial_id in by_trial:
            raise ValueError(f"evidence contains duplicate trial_id {item.trial_id!r}")
        if item.evaluation_protocol_sha256 != config.evaluation_protocol_sha256:
            raise ValueError("evidence evaluation protocol does not match the sweep")
        _validate_evidence_artifacts(expected[item.trial_id], item)
        if item.full_checkpoint_sha256 in checkpoint_hashes:
            raise ValueError("evidence reuses a full checkpoint SHA-256 across trials")
        if item.evaluation_report_sha256 in report_hashes:
            raise ValueError("evidence reuses an evaluation report SHA-256 across trials")
        checkpoint_hashes.add(item.full_checkpoint_sha256)
        report_hashes.add(item.evaluation_report_sha256)
        by_trial[item.trial_id] = item
    missing = sorted(set(expected) - set(by_trial))
    if missing:
        raise ValueError(f"evidence is missing trials: {missing!r}")
    passing = sorted(
        (item for item in by_trial.values() if item.passed),
        key=lambda item: (-float(item.score_lower_ci), float(item.expected_showdown_share_ece), item.trial_id),
    )
    ranks = {item.trial_id: index for index, item in enumerate(passing, start=1)}
    entries = tuple(
        TuningComparisonEntry(expected[trial_id], by_trial[trial_id], ranks.get(trial_id))
        for trial_id in sorted(expected, key=lambda value: (ranks.get(value, 1_000_000), value))
    )
    return TuningComparisonReport(config.sweep_id, config.evaluation_protocol_sha256, entries)


__all__ = [
    "ALLOWED_PPO_GRID_FIELDS",
    "MaterializedTrial",
    "SweepConfig",
    "TUNING_COMPARISON_KIND",
    "TUNING_EVALUATION_KIND",
    "TUNING_PROTOCOL_KIND",
    "TUNING_SCHEMA_VERSION",
    "TUNING_TRIAL_KIND",
    "TrialSpec",
    "TuningComparisonEntry",
    "TuningComparisonReport",
    "TuningEvidence",
    "compare_tuning_evidence",
    "expand_sweep",
    "hu_promotion_protocol_payload",
    "load_sweep_config",
    "materialize_sweep",
    "publish_tuning_evaluation",
    "single_experiment_trial",
    "write_hu_promotion_protocol",
    "write_sweep_config",
]
