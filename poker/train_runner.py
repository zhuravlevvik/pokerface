"""Resumable, stage-aware PPO training runner.

The runner owns process-level training state that does not belong in
``PPOTrainer``: run configuration, curriculum selection, counters and durable
checkpoints.  It only checkpoints after a complete PPO update, so a saved run
can always be resumed without trying to reconstruct a half-consumed rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
import json
import os
import random
import signal
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .bots import AggroBot, CallingStationBot, RandomBot, RuleBot, TightBot
from .curriculum import CurriculumConfig, CurriculumStage, StageScheduler, stage_spec
from .league import BotPolicy, LeagueMember, ModelPolicy, OpponentLeague
from .model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from .promotion import PromotionConfig, PromotionEvaluation, PromotionEvaluator
from .training import PPOConfig, PPOTrainer, UpdateMetrics

if TORCH_AVAILABLE:
    import torch


CHECKPOINT_VERSION = 1


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training; install the project with `.[rl]`.")


@dataclass(frozen=True)
class LeagueMemberConfig:
    """One serialisable non-historical member of the initial opponent league."""

    name: str
    kind: str
    weight: float
    bot: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("league member name must not be empty")
        if self.weight <= 0:
            raise ValueError("league member weight must be positive")
        if self.kind not in {"current", "baseline", "counter"}:
            raise ValueError(f"unsupported configured league member kind: {self.kind!r}")
        if self.kind == "current" and self.bot is not None:
            raise ValueError("the current policy cannot name a bot")
        if self.kind != "current" and self.bot not in {"rule", "random", "tight", "aggro", "calling_station"}:
            raise ValueError("configured non-current league members need a known bot")


def _default_member_configs() -> tuple[LeagueMemberConfig, ...]:
    return (
        LeagueMemberConfig("current", "current", 3.0),
        LeagueMemberConfig("rule", "baseline", 1.0, "rule"),
        LeagueMemberConfig("random", "baseline", 0.35, "random"),
        LeagueMemberConfig("counter_tight", "counter", 0.75, "tight"),
        LeagueMemberConfig("counter_aggro", "counter", 0.75, "aggro"),
    )


@dataclass(frozen=True)
class LeagueConfig:
    """Reproducible initial league setup, independent from saved snapshots."""

    seed: int | None = None
    members: tuple[LeagueMemberConfig, ...] = field(default_factory=_default_member_configs)

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("league configuration needs at least one member")
        current = [member for member in self.members if member.kind == "current"]
        if len(current) != 1:
            raise ValueError("league configuration needs exactly one current member")
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("league member names must be unique")


@dataclass(frozen=True)
class RunSettings:
    """Scheduling knobs for one process invocation or resumable run."""

    stage: CurriculumStage = CurriculumStage.A_HEADS_UP_STARTER
    seed: int = 0
    iterations: int = 10
    hands_per_iteration: int = 128
    table_count: int = 8
    checkpoint_every_iterations: int = 1
    checkpoint_every_decisions: int | None = None
    checkpoint_every_seconds: float | None = 900.0

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.hands_per_iteration < 1 or self.table_count < 1:
            raise ValueError("iterations must be non-negative; hand and table counts must be positive")
        if self.checkpoint_every_iterations < 1:
            raise ValueError("checkpoint_every_iterations must be positive")
        if self.checkpoint_every_decisions is not None and self.checkpoint_every_decisions < 1:
            raise ValueError("checkpoint_every_decisions must be positive or null")
        if self.checkpoint_every_seconds is not None and self.checkpoint_every_seconds <= 0:
            raise ValueError("checkpoint_every_seconds must be positive or null")


@dataclass(frozen=True)
class TrainingRunConfig:
    """Fully serialisable input configuration for a PPO training run."""

    run: RunSettings = field(default_factory=RunSettings)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    # This is deliberately model-only initialization, never a continuation of
    # optimizer, RNG, league, or rollout state from another run.
    init_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.ppo.learning_rate != self.curriculum.base_learning_rate:
            raise ValueError(
                "ppo.learning_rate and curriculum.base_learning_rate must match; "
                "curriculum stage scaling is applied to this shared base"
            )
        if self.init_checkpoint is not None and (not isinstance(self.init_checkpoint, str) or not self.init_checkpoint.strip()):
            raise ValueError("init_checkpoint must be a non-empty path string or null")
        if self.promotion.enabled and stage_spec(self.run.stage).player_count != 2:
            raise ValueError("the current promotion protocol can only be enabled for heads-up stages A/B")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run"]["stage"] = self.run.stage.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingRunConfig":
        def mapping(name: str) -> dict[str, Any]:
            value = data.get(name, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            return dict(value)

        run_data = mapping("run")
        if "stage" in run_data:
            run_data["stage"] = CurriculumStage(run_data["stage"])
        league_data = mapping("league")
        members_data = league_data.get("members")
        if members_data is not None:
            if not isinstance(members_data, Sequence) or isinstance(members_data, (str, bytes)):
                raise ValueError("league.members must be an array")
            league_data["members"] = tuple(LeagueMemberConfig(**dict(member)) for member in members_data if isinstance(member, Mapping))
            if len(league_data["members"]) != len(members_data):
                raise ValueError("each league member must be an object")
        promotion_data = mapping("promotion")
        baseline_bots = promotion_data.get("baseline_bots")
        if baseline_bots is not None:
            if not isinstance(baseline_bots, Sequence) or isinstance(baseline_bots, (str, bytes)):
                raise ValueError("promotion.baseline_bots must be an array")
            promotion_data["baseline_bots"] = tuple(baseline_bots)
        return cls(
            run=RunSettings(**run_data),
            model=ModelConfig(**mapping("model")),
            ppo=PPOConfig(**mapping("ppo")),
            curriculum=CurriculumConfig(**mapping("curriculum")),
            league=LeagueConfig(**league_data),
            promotion=PromotionConfig(**promotion_data),
            init_checkpoint=data.get("init_checkpoint"),
        )


def load_run_config(path: str | Path) -> TrainingRunConfig:
    """Load a JSON or TOML run configuration without optional dependencies."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".toml":
        import tomllib

        data = tomllib.loads(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("training configuration must be an object")
    return TrainingRunConfig.from_dict(data)


def write_run_config(config: TrainingRunConfig, path: str | Path) -> None:
    """Write a reviewable JSON config.  Intended for creating a first run."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(destination, json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
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


def _fsync_file(path: Path) -> None:
    """Flush one completed temporary artifact before publishing it."""

    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist a rename where the platform/filesystem supports directory fsync."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform-specific fallback.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - some network filesystems reject it.
        pass
    finally:
        os.close(descriptor)


def _bot_from_name(name: str, seed: int | None):
    factories: dict[str, Callable[[int | None], object]] = {
        "rule": lambda _seed: RuleBot(),
        "random": lambda value: RandomBot(seed=value),
        "tight": lambda _seed: TightBot(),
        "aggro": lambda value: AggroBot(seed=value),
        "calling_station": lambda value: CallingStationBot(seed=value),
    }
    try:
        return factories[name](seed)
    except KeyError as error:
        raise ValueError(f"unknown bot: {name!r}") from error


def build_league(config: LeagueConfig, model: PokerAgentModel, *, fallback_seed: int | None = None) -> OpponentLeague:
    """Build the configured league, deriving its seed from the run if omitted."""

    league_seed = config.seed if config.seed is not None else fallback_seed
    members: list[LeagueMember] = []
    current_name = next(member.name for member in config.members if member.kind == "current")
    for item in config.members:
        if item.kind == "current":
            policy = ModelPolicy(item.name, model)
        else:
            policy = BotPolicy(item.name, _bot_from_name(item.bot or "", item.seed if item.seed is not None else league_seed))
        members.append(LeagueMember(policy, weight=item.weight, kind=item.kind))
    return OpponentLeague(current_name=current_name, members=members, seed=league_seed)


def _bot_state(bot: object) -> dict[str, Any]:
    state: dict[str, Any] = {"type": type(bot).__name__}
    rng = getattr(bot, "_rng", None)
    if rng is not None:
        state["rng_state"] = rng.getstate()
    return state


def _league_state(league: OpponentLeague) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for member in league.members:
        policy = member.policy
        policy_state: dict[str, Any]
        if isinstance(policy, ModelPolicy):
            policy_state = {
                "type": "current" if policy.name == league.current_name else "model",
                "checkpoint_path": None if policy.checkpoint_path is None else str(policy.checkpoint_path),
            }
        elif isinstance(policy, BotPolicy):
            policy_state = {"type": "bot", "bot": _bot_state(policy.bot)}
        else:  # pragma: no cover - protocol extensions must explicitly add persistence support.
            raise TypeError(f"cannot checkpoint unsupported league policy {type(policy).__name__}")
        members.append({"name": policy.name, "weight": member.weight, "kind": member.kind, "policy": policy_state})
    return {
        "current_name": league.current_name,
        "members": members,
        "seed": league.seed,
        "rng_state": league._rng.getstate(),
        "hand_index": league._hand_index,
    }


def _restore_bot(state: Mapping[str, Any]):
    type_name = state.get("type")
    bot_names = {
        "RuleBot": "rule",
        "RandomBot": "random",
        "TightBot": "tight",
        "AggroBot": "aggro",
        "CallingStationBot": "calling_station",
    }
    if type_name not in bot_names:
        raise ValueError(f"unsupported saved bot type: {type_name!r}")
    bot = _bot_from_name(bot_names[type_name], None)
    rng_state = state.get("rng_state")
    if rng_state is not None:
        rng = getattr(bot, "_rng", None)
        if rng is None:
            raise ValueError(f"saved random state for deterministic bot {type_name}")
        rng.setstate(rng_state)
    return bot


def _restore_league(state: Mapping[str, Any], current_model: PokerAgentModel) -> OpponentLeague:
    members_payload = state.get("members")
    if not isinstance(members_payload, Sequence):
        raise ValueError("checkpoint league has no members")
    current_name = state.get("current_name")
    if not isinstance(current_name, str):
        raise ValueError("checkpoint league has no current policy name")
    members: list[LeagueMember] = []
    for item in members_payload:
        if not isinstance(item, Mapping) or not isinstance(item.get("policy"), Mapping):
            raise ValueError("malformed checkpoint league member")
        name, kind, weight = item.get("name"), item.get("kind"), item.get("weight")
        if not isinstance(name, str) or not isinstance(kind, str) or not isinstance(weight, (int, float)):
            raise ValueError("malformed checkpoint league member metadata")
        policy_data = item["policy"]
        policy_type = policy_data.get("type")
        if policy_type == "current":
            policy = ModelPolicy(name, current_model)
        elif policy_type == "model":
            checkpoint_path = policy_data.get("checkpoint_path")
            if not isinstance(checkpoint_path, str):
                raise ValueError(f"saved league model {name!r} has no checkpoint path")
            try:
                policy = ModelPolicy.from_checkpoint(name, checkpoint_path)
            except (OSError, ValueError) as error:
                raise ValueError(f"saved league model {name!r} is missing or incompatible: {checkpoint_path}") from error
        elif policy_type == "bot":
            bot_data = policy_data.get("bot")
            if not isinstance(bot_data, Mapping):
                raise ValueError(f"malformed bot state for {name!r}")
            policy = BotPolicy(name, _restore_bot(bot_data))
        else:
            raise ValueError(f"unsupported saved league policy type: {policy_type!r}")
        members.append(LeagueMember(policy, float(weight), kind))
    league = OpponentLeague(current_name, members, seed=state.get("seed") if isinstance(state.get("seed"), int) else None)
    rng_state = state.get("rng_state")
    if rng_state is not None:
        league._rng.setstate(rng_state)
    hand_index = state.get("hand_index")
    if not isinstance(hand_index, int) or hand_index < 0:
        raise ValueError("checkpoint league has invalid hand index")
    league._hand_index = hand_index
    return league


def _rng_state() -> dict[str, Any]:
    _require_torch()
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    _require_torch()
    python_state = state.get("python")
    torch_state = state.get("torch")
    if python_state is None or torch_state is None:
        raise ValueError("checkpoint has incomplete random-number-generator state")
    random.setstate(python_state)
    torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint was saved with CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True)
class TrainingRunResult:
    iteration: int
    global_decisions: int
    global_hands: int
    interrupted: bool
    checkpoint_path: Path


class TrainingRunner:
    """Durable owner of a PPO run and its safe checkpoint boundaries."""

    def __init__(
        self,
        config: TrainingRunConfig,
        run_directory: str | Path,
        *,
        device: str | None = None,
        init_checkpoint: str | Path | None = None,
    ) -> None:
        _require_torch()
        self.run_directory = Path(run_directory)
        self.checkpoint_directory = self.run_directory / "checkpoints"
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        configured_init = config.init_checkpoint
        if init_checkpoint is not None and configured_init is not None and Path(init_checkpoint) != Path(configured_init):
            raise ValueError("init checkpoint was specified both in config and constructor with different paths")
        self.init_checkpoint = Path(init_checkpoint) if init_checkpoint is not None else (Path(configured_init) if configured_init is not None else None)
        if self.init_checkpoint is not None and config.init_checkpoint is None:
            config = replace(config, init_checkpoint=str(self.init_checkpoint))
        self.config = config
        self._seed_everything(config.run.seed)
        self.scheduler = StageScheduler(config.curriculum, initial_stage=config.run.stage)
        self.model = PokerAgentModel(config.model)
        if self.init_checkpoint is not None:
            self._load_initial_model_weights(self.init_checkpoint)
            # Loading a source model constructs modules and may consume global
            # RNG.  A fresh PPO run must begin from its own configured stream.
            self._seed_everything(config.run.seed)
        self.league = build_league(config.league, self.model, fallback_seed=config.run.seed)
        self.trainer = PPOTrainer(self.model, self.league, config.ppo, device=device)
        self._apply_stage_learning_rate()
        self.trainer._seed_counter = config.run.seed
        self.iteration = 0
        self.global_decisions = 0
        self.global_hands = 0
        self.best_score: float | None = None
        self.last_evaluation_iteration = -1
        self.pending_promotion_iteration: int | None = None
        self.promotion_evaluator = (
            PromotionEvaluator(config.promotion, self.run_directory, run_seed=config.run.seed)
            if config.promotion.enabled
            else None
        )
        if self.promotion_evaluator is not None and self.promotion_evaluator.last_evaluated_iteration >= 0:
            raise ValueError("run directory already contains promotion state; use --resume")
        self.manifest: dict[str, Any] = {"version": 1, "checkpoints": []}
        if self.init_checkpoint is not None:
            self.manifest["initialization"] = {
                "kind": "model_weights_only",
                "checkpoint": str(self.init_checkpoint),
                "checkpoint_sha256": _file_sha256(self.init_checkpoint),
                "metadata": self.model.checkpoint_metadata(),
            }
        if self.promotion_evaluator is not None:
            self.manifest["promotion"] = {
                "enabled": True,
                "archive_manifest": str(self.promotion_evaluator.archive_manifest_path),
                "evaluations": [],
            }
        self._stop_requested = False
        self._last_checkpoint_time = time.monotonic()
        self._last_checkpoint_decisions = 0

    def _apply_stage_learning_rate(self) -> float:
        """Apply the curriculum multiplier to PPO's configured base LR."""

        effective = self.scheduler.config.learning_rate_for(self.scheduler.stage)
        for group in self.trainer.optimizer.param_groups:
            group["lr"] = effective
        return effective

    def _load_initial_model_weights(self, path: Path) -> None:
        """Load only compatible network weights for a new independent run."""

        source = PokerAgentModel.load_checkpoint(path, map_location="cpu")
        expected = self.model.checkpoint_metadata()
        actual = source.checkpoint_metadata()
        # Model metadata includes architecture config.  Checking the complete
        # contract prevents a partial state-dict load or silently changing the
        # target run's declared observation/action-space interface.
        if actual != expected:
            raise ValueError("init checkpoint model metadata/config does not match the new PPO run")
        self.model.load_state_dict(source.state_dict(), strict=True)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_directory / "latest.pt"

    @property
    def manifest_path(self) -> Path:
        return self.run_directory / "manifest.json"

    def request_stop(self) -> None:
        """Ask the run loop to stop after its current complete PPO update."""

        self._stop_requested = True

    def _checkpoint_payload(self, *, reason: str, metrics: UpdateMetrics | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "metadata": self.model.checkpoint_metadata(),
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "run_config": self.config.to_dict(),
            "curriculum": {"stage": self.scheduler.stage.value, "config": asdict(self.scheduler.config)},
            "progress": {
                "iteration": self.iteration,
                "global_decisions": self.global_decisions,
                "global_hands": self.global_hands,
                "seed_counter": self.trainer._seed_counter,
            },
            "league": _league_state(self.league),
            "rng": _rng_state(),
            "best_score": self.best_score,
            "promotion_state": {
                "last_evaluation_iteration": self.last_evaluation_iteration,
                "pending_promotion_iteration": self.pending_promotion_iteration,
                "archive_manifest": None
                if self.promotion_evaluator is None
                else str(self.promotion_evaluator.archive_manifest_path),
                "archive_manifest_sha256": None
                if self.promotion_evaluator is None or not self.promotion_evaluator.archive_manifest_path.exists()
                else _file_sha256(self.promotion_evaluator.archive_manifest_path),
            },
            "effective_learning_rate": self.trainer.optimizer.param_groups[0]["lr"],
            "manifest": self.manifest,
            "checkpoint_control": {"last_checkpoint_decisions": self._last_checkpoint_decisions},
            "reason": reason,
        }
        if metrics is not None:
            payload["metrics"] = asdict(metrics)
        return payload

    def save_checkpoint(self, *, reason: str = "periodic", metrics: UpdateMetrics | None = None) -> Path:
        """Atomically publish an immutable checkpoint and atomically refresh latest."""

        tag = f"{reason}_{self.iteration:08d}.pt"
        path = self.checkpoint_directory / tag
        records = self.manifest.setdefault("checkpoints", [])
        if not isinstance(records, list):
            raise ValueError("run manifest checkpoint list is corrupted")
        record = {"path": str(path), "iteration": self.iteration, "reason": reason}
        if not records or records[-1] != record:
            records.append(record)
        self.manifest["latest"] = str(self.latest_path)
        self.manifest["updated_at"] = time.time()
        self._last_checkpoint_decisions = self.global_decisions
        payload = self._checkpoint_payload(reason=reason, metrics=metrics)
        _atomic_torch_save(payload, path)
        # ``latest`` is a full independently-written file instead of a mutable
        # symlink: this works across platforms and readers never observe a
        # partially-written checkpoint.
        _atomic_torch_save(payload, self.latest_path)
        _atomic_write_text(self.manifest_path, json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
        self._last_checkpoint_time = time.monotonic()
        return path

    @classmethod
    def resume(cls, path: str | Path, *, device: str | None = None) -> "TrainingRunner":
        """Restore a complete checkpoint after the original run's safe boundary."""

        _require_torch()
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("not a compatible resumable training checkpoint")
        config_data = payload.get("run_config")
        progress = payload.get("progress")
        curriculum = payload.get("curriculum")
        league_state = payload.get("league")
        rng = payload.get("rng")
        if not all(isinstance(value, Mapping) for value in (config_data, progress, curriculum, league_state, rng)):
            raise ValueError("checkpoint has incomplete run state")
        config = TrainingRunConfig.from_dict(config_data)
        instance = cls.__new__(cls)
        instance.config = config
        instance.run_directory = checkpoint_path.parent.parent
        instance.checkpoint_directory = instance.run_directory / "checkpoints"
        instance.device = device
        instance.init_checkpoint = Path(config.init_checkpoint) if config.init_checkpoint is not None else None
        curriculum_config_data = curriculum.get("config")
        stage = curriculum.get("stage")
        if not isinstance(curriculum_config_data, Mapping) or not isinstance(stage, str):
            raise ValueError("checkpoint has invalid curriculum state")
        instance.scheduler = StageScheduler(CurriculumConfig(**dict(curriculum_config_data)), initial_stage=CurriculumStage(stage))
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("config"), Mapping):
            raise ValueError("checkpoint has no model metadata")
        PokerAgentModel._validate_metadata(metadata)
        instance.model = PokerAgentModel(ModelConfig(**dict(metadata["config"])))
        instance.model.load_state_dict(payload["state_dict"])
        instance.league = _restore_league(league_state, instance.model)
        instance.trainer = PPOTrainer(instance.model, instance.league, config.ppo, device=device)
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint has no optimizer state")
        instance.trainer.optimizer.load_state_dict(optimizer_state)
        instance._apply_stage_learning_rate()
        for name in ("iteration", "global_decisions", "global_hands", "seed_counter"):
            if not isinstance(progress.get(name), int) or progress[name] < 0:
                raise ValueError(f"checkpoint has invalid progress field {name!r}")
        instance.iteration = progress["iteration"]
        instance.global_decisions = progress["global_decisions"]
        instance.global_hands = progress["global_hands"]
        instance.trainer._seed_counter = progress["seed_counter"]
        best_score = payload.get("best_score")
        if best_score is not None and not isinstance(best_score, (int, float)):
            raise ValueError("checkpoint has invalid best score")
        instance.best_score = None if best_score is None else float(best_score)
        promotion_state = payload.get("promotion_state", {})
        if not isinstance(promotion_state, Mapping):
            raise ValueError("checkpoint has invalid promotion state")
        saved_last_evaluation = promotion_state.get("last_evaluation_iteration", -1)
        if not isinstance(saved_last_evaluation, int) or saved_last_evaluation < -1:
            raise ValueError("checkpoint has invalid last evaluation iteration")
        pending_promotion = promotion_state.get("pending_promotion_iteration")
        if pending_promotion is not None and (
            not isinstance(pending_promotion, int)
            or pending_promotion < 1
            or pending_promotion != instance.iteration
        ):
            raise ValueError("checkpoint has invalid pending promotion iteration")
        instance.promotion_evaluator = (
            PromotionEvaluator(config.promotion, instance.run_directory, run_seed=config.run.seed)
            if config.promotion.enabled
            else None
        )
        if instance.promotion_evaluator is None:
            instance.last_evaluation_iteration = saved_last_evaluation
            instance.pending_promotion_iteration = pending_promotion
        else:
            archive_last = instance.promotion_evaluator.last_evaluated_iteration
            if archive_last > instance.iteration:
                raise ValueError("promotion archive is ahead of the restored training iteration")
            saved_archive_sha = promotion_state.get("archive_manifest_sha256")
            if archive_last == saved_last_evaluation and saved_archive_sha is not None:
                if not isinstance(saved_archive_sha, str) or _file_sha256(instance.promotion_evaluator.archive_manifest_path) != saved_archive_sha:
                    raise ValueError("promotion archive manifest does not match full-run checkpoint")
            if archive_last < saved_last_evaluation:
                raise ValueError("promotion archive is behind the restored full-run checkpoint")
            if archive_last > saved_last_evaluation and pending_promotion != archive_last:
                raise ValueError("promotion archive is ahead without a matching durable promotion intent")
            archive_decisions = instance.promotion_evaluator.archive_manifest.get("decisions", [])
            if not isinstance(archive_decisions, list):
                raise ValueError("promotion archive decisions are malformed")
            for archive_decision in archive_decisions:
                if not isinstance(archive_decision, Mapping) or not isinstance(archive_decision.get("iteration"), int):
                    raise ValueError("promotion archive contains an invalid decision iteration")
                instance._validate_recovered_promotion(archive_decision["iteration"])
            instance.promotion_evaluator.synchronize_league(instance.league)
            instance.last_evaluation_iteration = max(saved_last_evaluation, archive_last)
            instance.pending_promotion_iteration = (
                None
                if pending_promotion is not None and instance.last_evaluation_iteration >= pending_promotion
                else pending_promotion
            )
            if instance.promotion_evaluator.champion_score is not None:
                instance.best_score = instance.promotion_evaluator.champion_score
        manifest = payload.get("manifest", {"version": 1, "checkpoints": []})
        if not isinstance(manifest, Mapping):
            raise ValueError("checkpoint has invalid manifest")
        instance.manifest = dict(manifest)
        if instance.promotion_evaluator is not None:
            section = instance.manifest.setdefault(
                "promotion",
                {"enabled": True, "archive_manifest": str(instance.promotion_evaluator.archive_manifest_path), "evaluations": []},
            )
            if not isinstance(section, dict):
                raise ValueError("checkpoint manifest has invalid promotion section")
            section["evaluations"] = list(instance.promotion_evaluator.archive_manifest.get("decisions", []))
        instance._stop_requested = False
        instance._last_checkpoint_time = time.monotonic()
        control = payload.get("checkpoint_control", {})
        if not isinstance(control, Mapping):
            raise ValueError("checkpoint has invalid checkpoint control state")
        last_checkpoint_decisions = control.get("last_checkpoint_decisions", instance.global_decisions)
        if not isinstance(last_checkpoint_decisions, int) or not 0 <= last_checkpoint_decisions <= instance.global_decisions:
            raise ValueError("checkpoint has invalid last checkpoint decision count")
        instance._last_checkpoint_decisions = last_checkpoint_decisions
        # Construction of model/snapshots itself consumes RNG.  Restore last,
        # then the next rollout/update is bit-for-bit on the original stream.
        _restore_rng_state(rng)
        return instance

    def _should_checkpoint(self) -> bool:
        if self.iteration % self.config.run.checkpoint_every_iterations == 0:
            return True
        decisions = self.config.run.checkpoint_every_decisions
        if decisions is not None and self.global_decisions - self._last_checkpoint_decisions >= decisions:
            return True
        seconds = self.config.run.checkpoint_every_seconds
        return seconds is not None and time.monotonic() - self._last_checkpoint_time >= seconds

    def _run_promotion(self, metrics: UpdateMetrics | None = None) -> PromotionEvaluation:
        if self.promotion_evaluator is None:
            raise RuntimeError("promotion is not enabled for this run")
        # Publish a frozen full-run source before evaluation.  If a process
        # died after this boundary, the same immutable source is reused.
        self.pending_promotion_iteration = self.iteration
        source = self.checkpoint_directory / f"candidate_{self.iteration:08d}.pt"
        if not source.exists():
            source = self.save_checkpoint(reason="candidate", metrics=metrics)
        rng = _rng_state()
        try:
            result = self.promotion_evaluator.evaluate_and_promote(
                iteration=self.iteration,
                candidate_checkpoint=source,
                league=self.league,
                stage=self.scheduler.stage,
                champion_score=self.best_score,
                run_context={
                    "iteration": self.iteration,
                    "global_decisions": self.global_decisions,
                    "global_hands": self.global_hands,
                    "run_config_sha256": _canonical_sha256(self.config.to_dict()),
                },
            )
        finally:
            # Loading frozen evaluation/archive models constructs modules;
            # promotion must not perturb the learner's next stochastic rollout.
            _restore_rng_state(rng)
        self.last_evaluation_iteration = self.iteration
        self.pending_promotion_iteration = None
        if result.accepted:
            self.best_score = result.baseline_score_bb_per_100
        section = self.manifest.setdefault("promotion", {"enabled": True, "evaluations": []})
        if not isinstance(section, dict) or not isinstance(section.setdefault("evaluations", []), list):
            raise ValueError("run manifest promotion section is corrupted")
        archive_decisions = self.promotion_evaluator.archive_manifest.get("decisions", [])
        if not isinstance(archive_decisions, list) or not archive_decisions:
            raise ValueError("promotion completed without an archive decision record")
        record = dict(archive_decisions[-1])
        evaluations = section["evaluations"]
        if not evaluations or evaluations[-1] != record:
            evaluations.append(record)
        self.save_checkpoint(reason="promotion", metrics=metrics)
        return result

    def _validate_recovered_promotion(self, iteration: int) -> None:
        if self.promotion_evaluator is None:
            raise RuntimeError("promotion recovery requires an evaluator")
        decisions = self.promotion_evaluator.archive_manifest.get("decisions", [])
        if not isinstance(decisions, list):
            raise ValueError("promotion archive decisions are malformed")
        decision = next((item for item in decisions if item.get("iteration") == iteration), None)
        if not isinstance(decision, Mapping):
            raise ValueError("promotion archive lacks the expected iteration decision")
        source = self.checkpoint_directory / f"candidate_{iteration:08d}.pt"
        if not source.exists() or decision.get("source_full_checkpoint_sha256") != _file_sha256(source):
            raise ValueError("promotion decision does not match this run's frozen full checkpoint")
        report_path = Path(str(decision.get("report_path")))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = report.get("candidate") if isinstance(report, Mapping) else None
        run_context = candidate.get("run_context") if isinstance(candidate, Mapping) else None
        if not isinstance(run_context, Mapping) or run_context.get("run_config_sha256") != _canonical_sha256(self.config.to_dict()):
            raise ValueError("promotion report does not match this run configuration")

    def _train_one_iteration(self) -> UpdateMetrics:
        spec = self.scheduler.spec
        rollout, metrics = self.trainer.train_iteration(
            self.config.run.hands_per_iteration,
            table_count=self.config.run.table_count,
            player_count=spec.player_count,
            starting_stack=spec.starting_stack_chips(variant_index=self.iteration),
            allowed_raise_actions=spec.allowed_raise_actions,
            should_stop=lambda: self._stop_requested,
        )
        self.iteration += 1
        self.global_decisions += rollout.decisions
        self.global_hands += rollout.hands
        self.manifest["last_metrics"] = asdict(metrics)
        return metrics

    def run(self, *, until_iteration: int | None = None, install_signal_handlers: bool = True) -> TrainingRunResult:
        """Train to the target iteration and save only complete-update states.

        The first SIGINT requests an orderly checkpoint at the next safe
        boundary.  A second SIGINT raises ``KeyboardInterrupt`` immediately;
        no potentially half-updated state is written by this method.
        """

        target = self.config.run.iterations if until_iteration is None else until_iteration
        if target < self.iteration:
            raise ValueError("target iteration precedes the restored run")
        old_handler = None
        signal_count = 0

        def on_interrupt(_signum: int, _frame: Any) -> None:
            nonlocal signal_count
            signal_count += 1
            if signal_count == 1:
                self.request_stop()
            else:
                raise KeyboardInterrupt

        if install_signal_handlers:
            old_handler = signal.signal(signal.SIGINT, on_interrupt)
        try:
            if self.pending_promotion_iteration is not None:
                self._run_promotion()
                if self._stop_requested:
                    path = self.save_checkpoint(reason="interrupt")
                    return TrainingRunResult(self.iteration, self.global_decisions, self.global_hands, True, path)
            while self.iteration < target:
                metrics = self._train_one_iteration()
                if self._stop_requested:
                    path = self.save_checkpoint(reason="interrupt", metrics=metrics)
                    return TrainingRunResult(self.iteration, self.global_decisions, self.global_hands, True, path)
                if self.promotion_evaluator is not None and self.promotion_evaluator.should_evaluate(
                    self.iteration,
                    last_evaluation_iteration=self.last_evaluation_iteration,
                ):
                    self._run_promotion(metrics)
                    if self._stop_requested:
                        path = self.save_checkpoint(reason="interrupt", metrics=metrics)
                        return TrainingRunResult(self.iteration, self.global_decisions, self.global_hands, True, path)
                if self._should_checkpoint():
                    self.save_checkpoint(metrics=metrics)
            if self.promotion_evaluator is not None and self.promotion_evaluator.should_evaluate(
                self.iteration,
                completing=True,
                last_evaluation_iteration=self.last_evaluation_iteration,
            ):
                self._run_promotion()
                if self._stop_requested:
                    path = self.save_checkpoint(reason="interrupt")
                    return TrainingRunResult(self.iteration, self.global_decisions, self.global_hands, True, path)
            path = self.save_checkpoint(reason="complete")
            return TrainingRunResult(self.iteration, self.global_decisions, self.global_hands, False, path)
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)


__all__ = [
    "CHECKPOINT_VERSION",
    "LeagueConfig",
    "LeagueMemberConfig",
    "RunSettings",
    "TrainingRunConfig",
    "TrainingRunResult",
    "TrainingRunner",
    "build_league",
    "load_run_config",
    "write_run_config",
]
