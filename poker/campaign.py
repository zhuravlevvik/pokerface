"""Fail-closed multi-seed selection for immutable PPO tuning trials.

The single-trial tuning layer deliberately cannot decide that a hyperparameter
variant is good from one lucky seed.  This module groups the complete sweep
matrix by PPO overrides, revalidates every sealed trial, and computes a
Student-t confidence interval across training seeds for every baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
import os
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .tuning import SweepConfig, TrialSpec, TuningEvidence, compare_tuning_evidence


CAMPAIGN_SCHEMA_VERSION = "1.0"
CAMPAIGN_REPORT_KIND = "poker_ppo_multiseed_campaign_v1"
CAMPAIGN_CI_METHOD = "training_seed_student_t_v1"

# Two-sided 95% Student-t critical values.  Values above 30 use the normal
# approximation; campaign seed counts are expected to be small in practice.
_T95 = (
    0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
    2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
    2.042,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"campaign artifact is missing: {path}")
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value in {".", ".."}
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value)
    ):
        raise ValueError(f"{label} must be a safe non-empty name")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("campaign report already exists with different contents")
        return path
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
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
    return path


@dataclass(frozen=True)
class CampaignConfig:
    """Immutable selection policy around one already-preregistered sweep."""

    sweep: SweepConfig
    minimum_seeds_per_variant: int = 2
    minimum_baseline_ci95_low: float = 0.0
    maximum_expected_showdown_share_ece: float = 0.10
    name: str = "stage-a-campaign"

    def __post_init__(self) -> None:
        _safe_name(self.name, "campaign name")
        if (
            isinstance(self.minimum_seeds_per_variant, bool)
            or not isinstance(self.minimum_seeds_per_variant, int)
            or self.minimum_seeds_per_variant < 2
        ):
            raise ValueError("minimum_seeds_per_variant must be an integer of at least two")
        if self.minimum_seeds_per_variant > len(self.sweep.seeds):
            raise ValueError("campaign requires more seeds than the sweep contains")
        for value, label in (
            (self.minimum_baseline_ci95_low, "minimum_baseline_ci95_low"),
            (self.maximum_expected_showdown_share_ece, "maximum_expected_showdown_share_ece"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{label} must be finite")
        if not 0.0 <= self.maximum_expected_showdown_share_ece <= 1.0:
            raise ValueError("maximum_expected_showdown_share_ece must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "name": self.name,
            "sweep": self.sweep.as_dict(),
            "sweep_id": self.sweep.sweep_id,
            "minimum_seeds_per_variant": self.minimum_seeds_per_variant,
            "minimum_baseline_ci95_low": float(self.minimum_baseline_ci95_low),
            "maximum_expected_showdown_share_ece": float(self.maximum_expected_showdown_share_ece),
            "ci_method": CAMPAIGN_CI_METHOD,
        }

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    @property
    def campaign_id(self) -> str:
        return f"campaign-{self.config_sha256[:16]}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignConfig":
        sweep = value.get("sweep")
        if not isinstance(sweep, Mapping):
            raise ValueError("campaign config lacks a sweep")
        result = cls(
            sweep=SweepConfig.from_dict(sweep),
            minimum_seeds_per_variant=value.get("minimum_seeds_per_variant"),
            minimum_baseline_ci95_low=value.get("minimum_baseline_ci95_low"),
            maximum_expected_showdown_share_ece=value.get("maximum_expected_showdown_share_ece"),
            name=value.get("name", "stage-a-campaign"),
        )
        if _canonical_bytes(dict(value)) != _canonical_bytes(result.as_dict()):
            raise ValueError("campaign config derived fields do not match")
        return result


@dataclass(frozen=True)
class BaselineSeedAggregate:
    baseline: str
    seed_count: int
    mean_bb_per_100: float
    standard_error: float
    ci95_low: float
    ci95_high: float

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "seed_count": self.seed_count,
            "mean_bb_per_100": self.mean_bb_per_100,
            "standard_error": self.standard_error,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "ci_method": CAMPAIGN_CI_METHOD,
        }


@dataclass(frozen=True)
class CampaignVariant:
    variant_id: str
    ppo_overrides: Mapping[str, int | float]
    trial_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    baselines: tuple[BaselineSeedAggregate, ...]
    maximum_expected_showdown_share_ece: float
    passed: bool
    reasons: tuple[str, ...]
    rank: int | None = None

    @property
    def worst_baseline_ci95_low(self) -> float:
        return min(item.ci95_low for item in self.baselines)

    @property
    def mean_baseline_score(self) -> float:
        return mean(item.mean_bb_per_100 for item in self.baselines)

    def as_dict(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "ppo_overrides": dict(self.ppo_overrides),
            "trial_ids": list(self.trial_ids),
            "seeds": list(self.seeds),
            "baselines": [item.as_dict() for item in self.baselines],
            "maximum_expected_showdown_share_ece": self.maximum_expected_showdown_share_ece,
            "worst_baseline_ci95_low": self.worst_baseline_ci95_low,
            "mean_baseline_score": self.mean_baseline_score,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class CampaignReport:
    config: CampaignConfig
    evidence: tuple[TuningEvidence, ...]
    variants: tuple[CampaignVariant, ...]

    @property
    def winner(self) -> CampaignVariant | None:
        return next((variant for variant in self.variants if variant.rank == 1), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "kind": CAMPAIGN_REPORT_KIND,
            "campaign_id": self.config.campaign_id,
            "campaign_config_sha256": self.config.config_sha256,
            "sweep_id": self.config.sweep.sweep_id,
            "evaluation_protocol_sha256": self.config.sweep.evaluation_protocol_sha256,
            "winner_variant_id": None if self.winner is None else self.winner.variant_id,
            "winner_trial_ids": [] if self.winner is None else list(self.winner.trial_ids),
            "evidence": [item.as_dict() for item in sorted(self.evidence, key=lambda item: item.trial_id)],
            "variants": [item.as_dict() for item in self.variants],
        }

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write(Path(path), self.as_dict())


def _student_interval(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) < 2:
        raise ValueError("a cross-seed confidence interval requires at least two seeds")
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    standard_error = sqrt(variance / len(values))
    degrees = len(values) - 1
    critical = _T95[degrees] if degrees < len(_T95) else 1.96
    margin = critical * standard_error
    return center, standard_error, center - margin, center + margin


def _variant_key(trial: TrialSpec) -> str:
    return _canonical_sha256(dict(sorted(trial.ppo_overrides.items())))


def _baseline_scores(evidence: TuningEvidence) -> dict[str, float]:
    sealed = json.loads(evidence.evaluation_report_path.read_text(encoding="utf-8"))
    lineage = sealed.get("lineage") if isinstance(sealed, Mapping) else None
    if not isinstance(lineage, Mapping):
        raise ValueError("sealed tuning evidence lacks promotion lineage")
    promotion_path = Path(str(lineage.get("promotion_report_path"))).resolve()
    expected_sha = lineage.get("promotion_report_sha256")
    if _file_sha256(promotion_path) != expected_sha:
        raise ValueError("campaign promotion report changed after tuning evidence was sealed")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    suite = promotion.get("suite") if isinstance(promotion, Mapping) else None
    matchups = suite.get("matchups") if isinstance(suite, Mapping) else None
    if not isinstance(matchups, list):
        raise ValueError("campaign promotion report has malformed matchups")
    scores: dict[str, float] = {}
    for matchup in matchups:
        if not isinstance(matchup, Mapping):
            continue
        opponent = matchup.get("opponent")
        if not isinstance(opponent, str) or not opponent.startswith("baseline:"):
            continue
        value = matchup.get("bb_per_100")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("campaign baseline score must be finite")
        if opponent in scores:
            raise ValueError("campaign promotion report repeats a baseline")
        scores[opponent] = float(value)
    if not scores:
        raise ValueError("campaign promotion report has no baseline score")
    return scores


def aggregate_campaign(config: CampaignConfig, evidence: Sequence[TuningEvidence]) -> CampaignReport:
    """Verify a complete matrix and rank variants only from cross-seed gates."""

    comparison = compare_tuning_evidence(config.sweep, evidence)
    verified = {entry.evidence.trial_id: entry.evidence for entry in comparison.entries}
    groups: dict[str, list[TrialSpec]] = {}
    for trial in config.sweep.expand_trials():
        groups.setdefault(_variant_key(trial), []).append(trial)

    candidates: list[CampaignVariant] = []
    for key in sorted(groups):
        trials = sorted(groups[key], key=lambda trial: trial.seed)
        overrides = MappingProxyType(dict(sorted(trials[0].ppo_overrides.items())))
        if any(dict(trial.ppo_overrides) != dict(overrides) for trial in trials):
            raise ValueError("campaign variant grouping mixed PPO overrides")
        if tuple(trial.seed for trial in trials) != tuple(config.sweep.seeds):
            raise ValueError("campaign variant does not contain the exact preregistered seed set")
        score_maps = [_baseline_scores(verified[trial.trial_id]) for trial in trials]
        baseline_names = set(score_maps[0])
        if any(set(item) != baseline_names for item in score_maps[1:]):
            raise ValueError("campaign trials were not evaluated against the same baseline roster")
        aggregates = tuple(
            BaselineSeedAggregate(name, len(trials), *_student_interval([scores[name] for scores in score_maps]))
            for name in sorted(baseline_names)
        )
        max_ece = max(verified[trial.trial_id].expected_showdown_share_ece for trial in trials)
        reasons: list[str] = []
        if len(trials) < config.minimum_seeds_per_variant:
            reasons.append("insufficient training-seed support")
        if any(not verified[trial.trial_id].passed for trial in trials):
            reasons.append("one or more seed trials failed their preregistered gate")
        if any(item.ci95_low < config.minimum_baseline_ci95_low for item in aggregates):
            reasons.append("a cross-seed baseline confidence bound is below the campaign floor")
        if max_ece > config.maximum_expected_showdown_share_ece:
            reasons.append("a seed trial exceeds the campaign calibration ceiling")
        candidates.append(
            CampaignVariant(
                variant_id=f"variant-{key[:16]}",
                ppo_overrides=overrides,
                trial_ids=tuple(trial.trial_id for trial in trials),
                seeds=tuple(trial.seed for trial in trials),
                baselines=aggregates,
                maximum_expected_showdown_share_ece=max_ece,
                passed=not reasons,
                reasons=tuple(reasons),
            )
        )

    passing = sorted(
        (item for item in candidates if item.passed),
        key=lambda item: (-item.worst_baseline_ci95_low, -item.mean_baseline_score, item.maximum_expected_showdown_share_ece, item.variant_id),
    )
    ranks = {item.variant_id: index for index, item in enumerate(passing, start=1)}
    ranked = tuple(
        CampaignVariant(
            item.variant_id,
            item.ppo_overrides,
            item.trial_ids,
            item.seeds,
            item.baselines,
            item.maximum_expected_showdown_share_ece,
            item.passed,
            item.reasons,
            ranks.get(item.variant_id),
        )
        for item in sorted(candidates, key=lambda value: (ranks.get(value.variant_id, 1_000_000), value.variant_id))
    )
    return CampaignReport(config, tuple(verified[key] for key in sorted(verified)), ranked)


def write_campaign_config(config: CampaignConfig, path: str | Path) -> Path:
    if not isinstance(config, CampaignConfig):
        raise TypeError("config must be a CampaignConfig")
    return _atomic_write(Path(path), config.as_dict())


def load_campaign_config(path: str | Path) -> CampaignConfig:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid campaign config: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("campaign config must be an object")
    return CampaignConfig.from_dict(payload)


def verify_campaign_report(config: CampaignConfig, path: str | Path) -> CampaignReport:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid campaign report: {source}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("evidence"), list):
        raise ValueError("campaign report is malformed")
    evidence = tuple(TuningEvidence.from_dict(item) for item in payload["evidence"] if isinstance(item, Mapping))
    if len(evidence) != len(payload["evidence"]):
        raise ValueError("campaign report contains malformed evidence")
    recomputed = aggregate_campaign(config, evidence)
    if _canonical_bytes(payload) != _canonical_bytes(recomputed.as_dict()):
        raise ValueError("campaign report does not match its verified evidence/config")
    return recomputed


__all__ = [
    "BaselineSeedAggregate",
    "CAMPAIGN_CI_METHOD",
    "CAMPAIGN_REPORT_KIND",
    "CAMPAIGN_SCHEMA_VERSION",
    "CampaignConfig",
    "CampaignReport",
    "CampaignVariant",
    "aggregate_campaign",
    "load_campaign_config",
    "verify_campaign_report",
    "write_campaign_config",
]
