"""Executable adapters and JSON configuration for paired curriculum gates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .betting import Action
from .bots import AggroBot, CallingStationBot, RandomBot, RuleBot, TightBot
from .curriculum import CurriculumStage
from .curriculum_coordinator import (
    CheckpointEvaluator,
    CurriculumCoordinatorConfig,
    EvaluationProtocol,
    EvaluationRequest,
    OpponentSpec,
)
from .league import ModelPolicy
from .multiway_evaluation import MultiwayEvaluationConfig, OpponentSeat, evaluate_multiway_suite
from .paired_rung import PairedRungConfig


@dataclass(frozen=True)
class CurriculumJobConfig:
    """Complete serialisable configuration for one adjacent paired gate."""

    coordinator: CurriculumCoordinatorConfig
    paired_rung: PairedRungConfig

    def __post_init__(self) -> None:
        if self.paired_rung.target_stage is not self.coordinator.target_stage:
            raise ValueError("paired-rung target stage must match coordinator target stage")
        if self.paired_rung.protocol_sha256 != self.coordinator.paired_rung_protocol_sha256:
            raise ValueError("paired-rung protocol hash must match coordinator evidence contract")

    def as_dict(self) -> dict[str, object]:
        return {"coordinator": self.coordinator.as_dict(), "paired_rung": self.paired_rung.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CurriculumJobConfig":
        coordinator = value.get("coordinator")
        rung = value.get("paired_rung")
        if not isinstance(coordinator, Mapping) or not isinstance(rung, Mapping):
            raise ValueError("curriculum job needs coordinator and paired_rung objects")
        return cls(_coordinator_from_dict(coordinator), PairedRungConfig.from_dict(rung))


def native_multiway_evaluator() -> CheckpointEvaluator:
    """Resolve only pinned checkpoints or allow-listed bots for evaluation."""

    def evaluate(request: EvaluationRequest):
        config = _multiway_config_from_dict(request.protocol.protocol)
        candidate = ModelPolicy.from_checkpoint(request.role, request.checkpoint)
        opponents = tuple(_opponent_seat(spec) for spec in request.protocol.opponents)
        return evaluate_multiway_suite(request.role, candidate, opponents, config=config)

    return evaluate


def load_curriculum_job_config(path: str | Path) -> CurriculumJobConfig:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid curriculum job config: {source}") from error
    if not isinstance(value, Mapping):
        raise ValueError("curriculum job config must be a JSON object")
    return CurriculumJobConfig.from_dict(value)


def write_curriculum_job_config(config: CurriculumJobConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def default_curriculum_job_config() -> CurriculumJobConfig:
    """Return a small B -> C starter contract; budgets should be tuned upward."""

    source_config = MultiwayEvaluationConfig(
        player_count=2,
        deal_blocks=32,
        seed_start=6_000_000,
        required_expected_showdown_share_strata=("street=preflop|active_players=2",),
    )
    target_config = MultiwayEvaluationConfig(
        player_count=3,
        deal_blocks=32,
        seed_start=7_000_000,
        required_expected_showdown_share_strata=(
            "street=preflop|active_players=3",
            "street=flop|active_players=3",
        ),
    )
    source = EvaluationProtocol(
        "source-hu-100bb",
        (OpponentSpec("rule", {"kind": "bot", "bot": "rule"}),),
        source_config.as_dict(),
        source_config.required_expected_showdown_share_strata,
    )
    target = EvaluationProtocol(
        "target-3max-100bb",
        (
            OpponentSpec("rule", {"kind": "bot", "bot": "rule"}),
            OpponentSpec("aggro", {"kind": "bot", "bot": "aggro"}),
        ),
        target_config.as_dict(),
        target_config.required_expected_showdown_share_strata,
    )
    rung = PairedRungConfig(target_stage=CurriculumStage.C_THREE_MAX)
    coordinator = CurriculumCoordinatorConfig(
        CurriculumStage.B_HEADS_UP_FULL,
        CurriculumStage.C_THREE_MAX,
        source,
        (target,),
        rung.protocol_sha256,
    )
    return CurriculumJobConfig(coordinator, rung)


def _coordinator_from_dict(value: Mapping[str, Any]) -> CurriculumCoordinatorConfig:
    source = value.get("source_protocol")
    targets = value.get("target_protocols")
    if not isinstance(source, Mapping) or not isinstance(targets, list) or not targets:
        raise ValueError("coordinator needs source_protocol and non-empty target_protocols")
    try:
        return CurriculumCoordinatorConfig(
            source_stage=CurriculumStage(str(value["source_stage"])),
            target_stage=CurriculumStage(str(value["target_stage"])),
            source_protocol=_evaluation_protocol_from_dict(source),
            target_protocols=tuple(_evaluation_protocol_from_dict(item) for item in targets),
            paired_rung_protocol_sha256=str(value["paired_rung_protocol_sha256"]),
            min_transfer_delta_ci95_low_bb_per_100=float(value.get("min_transfer_delta_ci95_low_bb_per_100", 0.0)),
            min_target_baseline_ci95_low_bb_per_100=float(value.get("min_target_baseline_ci95_low_bb_per_100", 0.0)),
            min_source_delta_ci95_low_bb_per_100=float(value.get("min_source_delta_ci95_low_bb_per_100", 0.0)),
            max_expected_showdown_share_ece=float(value.get("max_expected_showdown_share_ece", 0.08)),
            max_expected_showdown_share_mae=float(value.get("max_expected_showdown_share_mae", 0.20)),
            min_required_stratum_samples=int(value.get("min_required_stratum_samples", 1)),
            max_illegal_actions=int(value.get("max_illegal_actions", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed curriculum coordinator configuration") from error


def _evaluation_protocol_from_dict(value: object) -> EvaluationProtocol:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation protocol must be an object")
    opponents = value.get("opponents")
    protocol = value.get("protocol")
    strata = value.get("required_expected_showdown_share_strata", [])
    if not isinstance(opponents, list) or not isinstance(protocol, Mapping) or not isinstance(strata, list):
        raise ValueError("malformed evaluation protocol")
    parsed: list[OpponentSpec] = []
    for item in opponents:
        if not isinstance(item, Mapping) or not isinstance(item.get("provenance"), Mapping):
            raise ValueError("malformed opponent specification")
        parsed.append(OpponentSpec(str(item.get("identity", "")), dict(item["provenance"])))
    return EvaluationProtocol(str(value.get("name", "")), tuple(parsed), dict(protocol), tuple(str(item) for item in strata))


def _multiway_config_from_dict(value: Mapping[str, object]) -> MultiwayEvaluationConfig:
    allowed = value.get("allowed_raise_actions")
    if allowed is not None and (
        not isinstance(allowed, Sequence)
        or isinstance(allowed, (str, bytes))
        or not all(isinstance(item, (str, Action)) for item in allowed)
    ):
        raise ValueError("allowed_raise_actions must be null or an array of action names")
    strata = value.get("required_expected_showdown_share_strata", [])
    if not isinstance(strata, Sequence) or isinstance(strata, (str, bytes)) or not all(isinstance(item, str) for item in strata):
        raise ValueError("required_expected_showdown_share_strata must be an array")
    try:
        return MultiwayEvaluationConfig(
            player_count=int(value["player_count"]),
            deal_blocks=int(value["deal_blocks"]),
            seed_start=int(value["seed_start"]),
            starting_stack=int(value["starting_stack"]),
            allowed_raise_actions=None if allowed is None else tuple(Action(item) for item in allowed),
            equity_samples=int(value["equity_samples"]),
            calibration_bins=int(value["calibration_bins"]),
            required_expected_showdown_share_strata=tuple(str(item) for item in strata),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed multiway evaluation configuration") from error


def _opponent_seat(spec: OpponentSpec) -> OpponentSeat:
    provenance = dict(spec.provenance)
    kind = provenance.get("kind")
    if kind == "bot":
        bot_name = provenance.get("bot")
        factories = {
            "rule": RuleBot,
            "random": RandomBot,
            "tight": TightBot,
            "aggro": AggroBot,
            "calling_station": CallingStationBot,
        }
        factory = factories.get(bot_name)
        if factory is None:
            raise ValueError(f"unsupported fixed bot opponent: {bot_name!r}")
        return OpponentSeat(spec.identity, factory)
    if kind == "checkpoint":
        path_value, expected_sha = provenance.get("path"), provenance.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("checkpoint opponent requires path and pinned SHA-256")
        path = Path(path_value)
        if not path.is_file() or _file_sha256(path) != expected_sha:
            raise ValueError(f"checkpoint opponent {spec.identity!r} is missing or hash-mismatched")
        policy = ModelPolicy.from_checkpoint(spec.identity, path)
        return OpponentSeat(spec.identity, lambda: policy)
    raise ValueError(f"unsupported opponent kind: {kind!r}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CurriculumJobConfig",
    "default_curriculum_job_config",
    "load_curriculum_job_config",
    "native_multiway_evaluator",
    "write_curriculum_job_config",
]
