"""Fixed-suite evaluation and immutable promotion for trained candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .bots import AggroBot, CallingStationBot, RandomBot, RuleBot, TightBot
from .curriculum import CurriculumStage, stage_spec
from .evaluation import EvaluationConfig, EvaluationSuiteReport, MatchupReport, evaluate_suite
from .league import LeagueMember, ModelPolicy, OpponentLeague
from .model import TORCH_AVAILABLE, PokerAgentModel

if TORCH_AVAILABLE:
    import torch


PROMOTION_REPORT_VERSION = 2
PROMOTION_BASELINES = ("rule", "tight", "aggro", "calling_station", "random")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for checkpoint promotion; install the project with `.[rl]`.")


@dataclass(frozen=True)
class PromotionConfig:
    """Stable protocol and explicit gates for periodic model promotion."""

    enabled: bool = False
    every_iterations: int = 5
    evaluate_on_complete: bool = True
    hands_per_opponent: int = 200
    seed_start: int = 2_000_000
    equity_samples: int = 32
    calibration_bins: int = 10
    baseline_bots: tuple[str, ...] = PROMOTION_BASELINES
    historical_limit: int = 4
    league_historical_limit: int = 8
    historical_weight: float = 1.0
    minimum_baseline_bb_per_100: float = 0.0
    minimum_baseline_ci95_low: float = -10_000.0
    maximum_baseline_ci95_half_width: float = 10_000.0
    minimum_historical_league_score: float = 0.45
    minimum_historical_ci95_low: float = -10_000.0
    maximum_equity_ece: float = 0.08
    minimum_champion_improvement: float = 0.0

    def __post_init__(self) -> None:
        if self.every_iterations < 1:
            raise ValueError("every_iterations must be positive")
        if self.hands_per_opponent < 2 or self.hands_per_opponent % 2:
            raise ValueError("hands_per_opponent must be an even heads-up position-rotation budget")
        if self.equity_samples < 1 or self.calibration_bins < 1:
            raise ValueError("equity_samples and calibration_bins must be positive")
        if not self.baseline_bots or any(name not in PROMOTION_BASELINES for name in self.baseline_bots):
            raise ValueError(f"baseline_bots must be a non-empty subset of {PROMOTION_BASELINES}")
        if len(self.baseline_bots) != len(set(self.baseline_bots)):
            raise ValueError("baseline_bots must be unique")
        if self.historical_limit < 0 or self.league_historical_limit < 1 or self.historical_weight <= 0:
            raise ValueError("historical_limit must be non-negative, league_historical_limit positive, and historical_weight positive")
        if not 0.0 <= self.minimum_historical_league_score <= 1.0:
            raise ValueError("minimum_historical_league_score must be in [0, 1]")
        if self.maximum_baseline_ci95_half_width <= 0:
            raise ValueError("maximum_baseline_ci95_half_width must be positive")
        if not 0.0 <= self.maximum_equity_ece <= 1.0:
            raise ValueError("maximum_equity_ece must be in [0, 1]")
        numeric_gates = (
            self.minimum_baseline_bb_per_100,
            self.minimum_baseline_ci95_low,
            self.maximum_baseline_ci95_half_width,
            self.minimum_historical_league_score,
            self.minimum_historical_ci95_low,
            self.maximum_equity_ece,
            self.minimum_champion_improvement,
        )
        if any(not isfinite(value) for value in numeric_gates):
            raise ValueError("promotion gates must be finite")
        if self.minimum_champion_improvement < 0:
            raise ValueError("minimum_champion_improvement must be non-negative")


@dataclass(frozen=True)
class PromotionEvaluation:
    iteration: int
    accepted: bool
    baseline_score_bb_per_100: float
    previous_champion_score: float | None
    reasons: tuple[str, ...]
    report_path: Path
    checkpoint_path: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "accepted": self.accepted,
            "baseline_score_bb_per_100": self.baseline_score_bb_per_100,
            "previous_champion_score": self.previous_champion_score,
            "reasons": list(self.reasons),
            "report_path": str(self.report_path),
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
        }


class PromotionEvaluator:
    """Evaluate one in-memory policy and archive only accepted snapshots."""

    def __init__(self, config: PromotionConfig, run_directory: str | Path, *, run_seed: int = 0) -> None:
        self.config = config
        self.run_directory = Path(run_directory)
        self.report_directory = self.run_directory / "evaluations"
        self.candidate_directory = self.run_directory / "candidates"
        self.archive_directory = self.run_directory / "archive"
        self.report_directory.mkdir(parents=True, exist_ok=True)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        self.run_seed = run_seed
        self.archive_manifest_path = self.archive_directory / "manifest.json"
        self.archive_manifest = self._load_archive_manifest()

    @property
    def champion_score(self) -> float | None:
        champion = self.archive_manifest.get("champion")
        return float(champion["baseline_score_bb_per_100"]) if isinstance(champion, Mapping) else None

    @property
    def last_evaluated_iteration(self) -> int:
        decisions = self.archive_manifest.get("decisions", [])
        return max((int(item["iteration"]) for item in decisions), default=-1)

    def synchronize_league(self, league: OpponentLeague) -> None:
        """Restore manifest-backed promoted policies after a crash/resume."""

        promoted = self.archive_manifest.get("promoted", [])
        if not isinstance(promoted, list):
            raise ValueError("malformed promoted archive entries")
        promoted = promoted[-self.config.league_historical_limit :]
        retained_names = {Path(str(entry["checkpoint_path"])).stem for entry in promoted}
        league.members[:] = [
            member
            for member in league.members
            if member.kind not in {"best", "historical"} or member.policy.name in retained_names
        ]
        existing = {member.policy.name: member for member in league.members}
        for entry in promoted:
            path = Path(str(entry["checkpoint_path"]))
            name = path.stem
            if name in existing:
                policy = existing[name].policy
                if not isinstance(policy, ModelPolicy) or policy.checkpoint_path != path:
                    raise ValueError(f"league member {name!r} disagrees with promotion archive")
                continue
            league.add(
                LeagueMember(
                    ModelPolicy.from_checkpoint(name, path),
                    weight=self.config.historical_weight,
                    kind="best",
                )
            )
            existing[name] = league.members[-1]

    def should_evaluate(self, iteration: int, *, completing: bool = False, last_evaluation_iteration: int = -1) -> bool:
        if not self.config.enabled or iteration == last_evaluation_iteration:
            return False
        return iteration > 0 and (iteration % self.config.every_iterations == 0 or (completing and self.config.evaluate_on_complete))

    def evaluate_and_promote(
        self,
        *,
        iteration: int,
        candidate_checkpoint: str | Path,
        league: OpponentLeague,
        stage: CurriculumStage,
        champion_score: float | None,
        run_context: Mapping[str, Any] | None = None,
    ) -> PromotionEvaluation:
        _require_torch()
        spec = stage_spec(stage)
        if spec.player_count != 2:
            raise ValueError("the current promotion protocol is heads-up only")
        manifest_champion = self.champion_score
        if champion_score is not None and manifest_champion is not None and champion_score != manifest_champion:
            raise ValueError("full-run checkpoint champion score disagrees with archive manifest")
        previous_champion = manifest_champion if manifest_champion is not None else champion_score
        candidate_path, candidate_sha, source_sha = self._freeze_candidate(iteration, Path(candidate_checkpoint))
        opponents, opponent_registry = self._opponents(league)
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
        candidate_id = candidate_path.stem
        suite = evaluate_suite(
            candidate_id,
            ModelPolicy.from_checkpoint(candidate_id, candidate_path),
            opponents,
            config=protocol,
        )
        baseline_reports = tuple(item for item in suite.matchups if item.opponent.startswith("baseline:"))
        historical_reports = tuple(item for item in suite.matchups if item.opponent.startswith("historical:"))
        baseline_score = _aggregate_bb_per_100(baseline_reports)
        reasons = self._gate_reasons(suite, baseline_reports, historical_reports, baseline_score, previous_champion)
        accepted = not reasons
        checkpoint_path = None
        if accepted:
            checkpoint_path = self.archive_directory / f"promoted_{iteration:08d}_{candidate_sha[:12]}.pt"
            if checkpoint_path.exists():
                if _file_sha256(checkpoint_path) != candidate_sha:
                    raise ValueError(f"immutable promotion checkpoint hash mismatch: {checkpoint_path}")
            else:
                _atomic_copy(candidate_path, checkpoint_path)
        report_path = self.report_directory / f"evaluation_{candidate_id}.json"
        protocol = {
            "stage": stage.value,
            "outcome_protocol": "fixed_deal_virtual_showdown_outcome_v1",
            "scalar_metric_protocol": "active_hands_expected_showdown_share_v1",
            "player_count": spec.player_count,
            "starting_stack": spec.starting_stack_chips(),
            "allowed_raise_actions": [action.value for action in sorted(spec.allowed_raise_actions, key=lambda action: action.value)],
            "paired_position_seeds": True,
            "ci_method": "paired_position_seed_block_normal_v1",
            "seed_start": self.config.seed_start,
            "seed_blocks_per_opponent": self.config.hands_per_opponent // 2,
        }
        report = {
            "promotion_report_version": PROMOTION_REPORT_VERSION,
            "iteration": iteration,
            "promotion_config": asdict(self.config),
            "protocol": protocol,
            "protocol_sha256": _canonical_sha256(protocol),
            "candidate": {
                "id": candidate_id,
                "path": str(candidate_path),
                "sha256": candidate_sha,
                "source_full_checkpoint": str(candidate_checkpoint),
                "source_full_checkpoint_sha256": source_sha,
                "run_context": dict(run_context or {}),
            },
            "opponents": opponent_registry,
            "suite": suite.as_dict(),
            "decision": {
                "accepted": accepted,
                "baseline_score_bb_per_100": baseline_score,
                "previous_champion_score": previous_champion,
                "reasons": list(reasons),
                "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            },
        }
        _atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        report_sha = _file_sha256(report_path)
        decision_record = {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "accepted": accepted,
            "baseline_score_bb_per_100": baseline_score,
            "candidate_sha256": candidate_sha,
            "source_full_checkpoint_sha256": source_sha,
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "protocol_sha256": report["protocol_sha256"],
        }
        self._record_decision(decision_record)
        if accepted:
            self.synchronize_league(league)
        return PromotionEvaluation(iteration, accepted, baseline_score, previous_champion, tuple(reasons), report_path, checkpoint_path)

    def _opponents(self, league: OpponentLeague) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        seed = self.run_seed + self.config.seed_start
        factories = {
            "rule": lambda: RuleBot(),
            "tight": lambda: TightBot(),
            "aggro": lambda: AggroBot(seed=seed),
            "calling_station": lambda: CallingStationBot(seed=seed),
            "random": lambda: RandomBot(seed=seed),
        }
        opponents: dict[str, Any] = {f"baseline:{name}": factories[name]() for name in self.config.baseline_bots}
        registry: list[dict[str, Any]] = [
            {
                "name": f"baseline:{name}",
                "kind": "baseline",
                "implementation": type(opponents[f"baseline:{name}"]).__name__,
                "seed": None if name in {"rule", "tight"} else seed,
                "rng_protocol": "per-deal-role-seed-v1",
            }
            for name in self.config.baseline_bots
        ]
        historical = [
            member
            for member in league.members
            if member.kind in {"best", "historical"} and isinstance(member.policy, ModelPolicy)
        ]
        if self.config.historical_limit:
            historical = historical[-self.config.historical_limit :]
        else:
            historical = []
        for member in historical:
            opponents[f"historical:{member.policy.name}"] = member.policy
            path = member.policy.checkpoint_path
            registry.append(
                {
                    "name": f"historical:{member.policy.name}",
                    "kind": member.kind,
                    "checkpoint_path": None if path is None else str(path),
                    "checkpoint_sha256": None if path is None else _file_sha256(path),
                }
            )
        return opponents, registry

    def _freeze_candidate(self, iteration: int, source: Path) -> tuple[Path, str, str]:
        source_sha = _file_sha256(source)
        existing = tuple(self.candidate_directory.glob(f"candidate_{iteration:08d}_*.pt"))
        if len(existing) > 1:
            raise ValueError(f"multiple immutable candidates exist for iteration {iteration}")
        if existing:
            candidate = existing[0]
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            provenance = payload.get("candidate") if isinstance(payload, Mapping) else None
            if not isinstance(provenance, Mapping) or provenance.get("source_full_checkpoint_sha256") != source_sha:
                raise ValueError("existing immutable candidate came from a different full checkpoint")
            return candidate, _file_sha256(candidate), source_sha
        model = PokerAgentModel.load_checkpoint(source, map_location="cpu")
        temporary = self.candidate_directory / f".candidate_{iteration:08d}.{uuid4().hex}.tmp"
        try:
            torch.save(
                {
                    "metadata": model.checkpoint_metadata(),
                    "state_dict": model.state_dict(),
                    "candidate": {"iteration": iteration, "source_full_checkpoint_sha256": source_sha},
                },
                temporary,
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            candidate_sha = _file_sha256(temporary)
            candidate = self.candidate_directory / f"candidate_{iteration:08d}_{candidate_sha[:12]}.pt"
            os.replace(temporary, candidate)
            _fsync_directory(candidate.parent)
            return candidate, candidate_sha, source_sha
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load_archive_manifest(self) -> dict[str, Any]:
        if not self.archive_manifest_path.exists():
            return {"version": PROMOTION_REPORT_VERSION, "decisions": [], "promoted": [], "champion": None}
        payload = json.loads(self.archive_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("version") != PROMOTION_REPORT_VERSION:
            raise ValueError("incompatible promotion archive manifest")
        decisions, promoted = payload.get("decisions"), payload.get("promoted")
        if not isinstance(decisions, list) or not isinstance(promoted, list):
            raise ValueError("malformed promotion archive manifest")
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("malformed promotion decision entry")
            report_path = Path(str(decision.get("report_path")))
            if not report_path.exists() or _file_sha256(report_path) != decision.get("report_sha256"):
                raise ValueError("promotion report is missing or does not match archive manifest")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            candidate = report.get("candidate") if isinstance(report, Mapping) else None
            report_decision = report.get("decision") if isinstance(report, Mapping) else None
            if not isinstance(candidate, Mapping) or not isinstance(report_decision, Mapping):
                raise ValueError("promotion report lacks candidate/decision provenance")
            candidate_path = Path(str(candidate.get("path")))
            source_path = Path(str(candidate.get("source_full_checkpoint")))
            if (
                candidate.get("id") != decision.get("candidate_id")
                or candidate.get("sha256") != decision.get("candidate_sha256")
                or candidate.get("source_full_checkpoint_sha256") != decision.get("source_full_checkpoint_sha256")
                or not candidate_path.exists()
                or _file_sha256(candidate_path) != decision.get("candidate_sha256")
                or not source_path.exists()
                or _file_sha256(source_path) != decision.get("source_full_checkpoint_sha256")
            ):
                raise ValueError("promotion candidate/source provenance does not match archive manifest")
            if (
                report.get("iteration") != decision.get("iteration")
                or report.get("protocol_sha256") != decision.get("protocol_sha256")
                or report_decision.get("accepted") != decision.get("accepted")
                or report_decision.get("baseline_score_bb_per_100") != decision.get("baseline_score_bb_per_100")
                or report_decision.get("checkpoint_path") != decision.get("checkpoint_path")
            ):
                raise ValueError("promotion decision fields do not match its immutable report")
        for entry in promoted:
            if not isinstance(entry, Mapping):
                raise ValueError("malformed promoted archive entry")
            path = Path(str(entry.get("checkpoint_path")))
            if not path.exists() or _file_sha256(path) != entry.get("candidate_sha256"):
                raise ValueError("promoted checkpoint is missing or does not match archive manifest")
        champion = payload.get("champion")
        if champion is not None and (not isinstance(champion, Mapping) or champion not in promoted):
            raise ValueError("promotion archive champion is not a promoted decision")
        return dict(payload)

    def _record_decision(self, record: Mapping[str, Any]) -> None:
        decisions = self.archive_manifest["decisions"]
        promoted = self.archive_manifest["promoted"]
        assert isinstance(decisions, list) and isinstance(promoted, list)
        existing = next((item for item in decisions if item.get("candidate_id") == record["candidate_id"]), None)
        if existing is not None:
            if dict(existing) != dict(record):
                raise ValueError("promotion retry produced a different immutable decision")
            return
        decisions.append(dict(record))
        if record["accepted"]:
            promoted.append(dict(record))
            self.archive_manifest["champion"] = dict(record)
        _atomic_write_text(self.archive_manifest_path, json.dumps(self.archive_manifest, indent=2, sort_keys=True) + "\n")

    def _gate_reasons(
        self,
        suite: EvaluationSuiteReport,
        baselines: Sequence[MatchupReport],
        historical: Sequence[MatchupReport],
        baseline_score: float,
        champion_score: float | None,
    ) -> list[str]:
        reasons: list[str] = []
        if baseline_score < self.config.minimum_baseline_bb_per_100:
            reasons.append("baseline BB/100 is below the configured floor")
        if any(item.bb_per_100_ci95_low < self.config.minimum_baseline_ci95_low for item in baselines):
            reasons.append("a baseline matchup confidence bound is below the configured floor")
        if any(
            (item.bb_per_100_ci95_high - item.bb_per_100_ci95_low) / 2.0 > self.config.maximum_baseline_ci95_half_width
            for item in baselines
        ):
            reasons.append("a baseline matchup confidence interval is wider than the configured ceiling")
        showdown_share = [item.expected_showdown_share for item in baselines]
        if any(
            item is None or item.expected_calibration_error > self.config.maximum_equity_ece
            for item in showdown_share
        ):
            reasons.append("heads-up expected-showdown-share calibration exceeds the configured ECE ceiling")
        if any(item.model_diagnostics.illegal_action_count for item in suite.matchups):
            reasons.append("candidate selected an illegal or masked action")
        if any(not item.legal or not item.finite for item in suite.sanity_checks):
            reasons.append("one or more fixed sanity scenarios failed")
        if historical:
            historical_score = sum(item.league_score for item in historical) / len(historical)
            if historical_score < self.config.minimum_historical_league_score:
                reasons.append("historical regression score is below the configured floor")
            if any(item.bb_per_100_ci95_low < self.config.minimum_historical_ci95_low for item in historical):
                reasons.append("a historical matchup confidence bound is below the regression floor")
        if champion_score is not None and baseline_score <= champion_score + self.config.minimum_champion_improvement:
            reasons.append("fixed-baseline score did not improve on the champion")
        return reasons


def _aggregate_bb_per_100(reports: Sequence[MatchupReport]) -> float:
    hands = sum(item.hands for item in reports)
    if not hands:
        raise ValueError("promotion evaluation has no fixed-baseline hands")
    return sum(item.pnl_bb for item in reports) / hands * 100.0


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        _publish_file(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        _publish_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_file(temporary: Path, destination: Path) -> None:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # pragma: no cover - filesystem dependent.
        pass


__all__ = [
    "PROMOTION_BASELINES",
    "PROMOTION_REPORT_VERSION",
    "PromotionConfig",
    "PromotionEvaluation",
    "PromotionEvaluator",
]
