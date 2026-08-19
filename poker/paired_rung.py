"""Durable, matched transfer-versus-scratch PPO training controls.

This module deliberately coordinates two *independent* normal PPO runs.  It
does not borrow a caller's league or random-number stream: both arms receive
the same serialisable stage/opponent protocol and start their environment
stream at the same seed.  Their native :class:`TrainingRunner` checkpoints
retain optimiser, league and RNG state, while this layer adds immutable
full/model snapshots and an auditable paired manifest.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import signal
import shutil
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from .curriculum import CurriculumStage
from .model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from .training import PPOConfig

if TORCH_AVAILABLE:
    import torch


PAIRED_RUNG_MANIFEST_VERSION = 1
"""Version of the durable paired-rung manifest/checkpoint envelope."""

_BOT_NAMES = frozenset({"rule", "random", "tight", "aggro", "calling_station"})
_MEMBER_KINDS = frozenset({"baseline", "counter"})


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for paired PPO rungs; install the project with `.[rl]`.")


@dataclass(frozen=True)
class PairedRungOpponentConfig:
    """One fixed non-learning opponent in the matched arm protocol."""

    name: str
    bot: str
    weight: float = 1.0
    kind: str = "baseline"
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("paired-rung opponent name must not be empty")
        if self.bot not in _BOT_NAMES:
            raise ValueError(f"unknown paired-rung bot: {self.bot!r}")
        if self.kind not in _MEMBER_KINDS:
            raise ValueError(f"unsupported paired-rung opponent kind: {self.kind!r}")
        if self.weight <= 0:
            raise ValueError("paired-rung opponent weight must be positive")


def _default_opponents() -> tuple[PairedRungOpponentConfig, ...]:
    return (
        PairedRungOpponentConfig("rule", "rule", 1.0),
        PairedRungOpponentConfig("random", "random", 0.35),
        PairedRungOpponentConfig("counter_tight", "tight", 0.75, "counter"),
        PairedRungOpponentConfig("counter_aggro", "aggro", 0.75, "counter"),
    )


@dataclass(frozen=True)
class PairedRungConfig:
    """Serializable matched budget and target-stage protocol.

    ``ppo.learning_rate`` is the unscaled base value.  Each arm uses the
    normal curriculum stage scale through ``TrainingRunner``.  The source
    architecture comes from the supplied source checkpoint, so a scratch arm
    is a true same-architecture control rather than a separately configured
    network.
    """

    target_stage: CurriculumStage = CurriculumStage.C_THREE_MAX
    iterations: int = 10
    hands_per_iteration: int = 128
    table_count: int = 8
    base_seed: int = 0
    ppo: PPOConfig = field(default_factory=PPOConfig)
    current_weight: float = 3.0
    opponents: tuple[PairedRungOpponentConfig, ...] = field(default_factory=_default_opponents)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "target_stage", CurriculumStage(self.target_stage))
        except ValueError as error:
            raise ValueError(f"unknown target curriculum stage: {self.target_stage!r}") from error
        if self.iterations < 1 or self.hands_per_iteration < 1 or self.table_count < 1:
            raise ValueError("iterations, hands_per_iteration and table_count must be positive")
        if not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer")
        if self.current_weight <= 0:
            raise ValueError("current_weight must be positive")
        if not self.opponents:
            raise ValueError("paired rung needs at least one fixed opponent")
        names = [item.name for item in self.opponents]
        if len(names) != len(set(names)) or "current" in names:
            raise ValueError("paired-rung opponent names must be unique and may not be 'current'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_stage": self.target_stage.value,
            "iterations": self.iterations,
            "hands_per_iteration": self.hands_per_iteration,
            "table_count": self.table_count,
            "base_seed": self.base_seed,
            "ppo": asdict(self.ppo),
            "current_weight": self.current_weight,
            "opponents": [asdict(item) for item in self.opponents],
        }

    @property
    def protocol_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairedRungConfig":
        opponents = value.get("opponents")
        ppo = value.get("ppo")
        if not isinstance(opponents, Sequence) or isinstance(opponents, (str, bytes)):
            raise ValueError("paired-rung opponents must be an array")
        if not isinstance(ppo, Mapping):
            raise ValueError("paired-rung ppo must be an object")
        if any(not isinstance(item, Mapping) for item in opponents):
            raise ValueError("malformed paired-rung configuration")
        try:
            return cls(
                target_stage=CurriculumStage(str(value["target_stage"])),
                iterations=int(value["iterations"]),
                hands_per_iteration=int(value["hands_per_iteration"]),
                table_count=int(value["table_count"]),
                base_seed=int(value["base_seed"]),
                ppo=PPOConfig(**dict(ppo)),
                current_weight=float(value.get("current_weight", 3.0)),
                opponents=tuple(PairedRungOpponentConfig(**dict(item)) for item in opponents),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed paired-rung configuration") from error


@dataclass(frozen=True)
class PairedRungArmResult:
    """Frozen result of one arm, with counters independent of BB outcomes."""

    name: str
    iteration: int
    global_hands: int
    global_decisions: int
    full_checkpoint_path: Path | None
    model_checkpoint_path: Path | None
    full_checkpoint_sha256: str | None
    model_checkpoint_sha256: str | None


@dataclass(frozen=True)
class PairedRungResult:
    """One recoverable paired-control state for an evaluator/coordinator."""

    transfer: PairedRungArmResult
    scratch: PairedRungArmResult
    completed: bool
    config_sha256: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    source_run_config_sha256: str | None
    manifest_path: Path

    @property
    def iteration(self) -> int:
        """Largest completed paired boundary (the lower arm iteration)."""

        return min(self.transfer.iteration, self.scratch.iteration)


class PairedRungRunner:
    """Run or resume a matched transfer/scratch control without caller mutation."""

    def __init__(
        self,
        config: PairedRungConfig,
        run_directory: str | Path,
        source_checkpoint: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        _require_torch()
        self.config = config
        self.run_directory = Path(run_directory)
        self.directory = self.run_directory / "paired-rung"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.source_input_path = Path(source_checkpoint)
        if not self.source_input_path.is_file():
            raise FileNotFoundError(f"paired-rung source checkpoint does not exist: {self.source_input_path}")
        # This validates that either a model-only or a full source actually
        # matches the live model contract before any arm directory is made.
        with _preserved_global_rng():
            source = PokerAgentModel.load_checkpoint(self.source_input_path, map_location="cpu")
        self.model_config: ModelConfig = source.config
        self.source_sha256 = _file_sha256(self.source_input_path)
        self.source_run_config_sha256 = _source_run_config_sha256(self.source_input_path)
        self.source_path = self.directory / "sources" / f"source-{self.source_sha256}.pt"
        _freeze_copy(self.source_input_path, self.source_path, expected_sha256=self.source_sha256)
        self.config_sha256 = self.config.protocol_sha256
        self._stop_requested = False
        self._ensure_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def request_stop(self) -> None:
        """Finish the active arm update, persist it, then return incomplete."""

        self._stop_requested = True

    def run(self, *, until_iteration: int | None = None, install_signal_handlers: bool = False) -> PairedRungResult:
        """Train each arm to the same requested target iteration.

        Native runner checkpoints are made after every iteration.  A crash
        between arms is therefore safe: a subsequent call advances only the
        lagging arm to the same boundary.  All Python/Torch/CUDA global RNG
        state is restored before this method returns, including on failure.
        """

        target = self.config.iterations if until_iteration is None else until_iteration
        if not isinstance(target, int) or not 1 <= target <= self.config.iterations:
            raise ValueError("until_iteration must be between one and configured iterations")
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
            with _preserved_global_rng():
                transfer = self._load_arm("transfer")
                scratch = self._load_arm("scratch")
                for iteration in range(min(transfer.runner.iteration, scratch.runner.iteration) + 1, target + 1):
                    if transfer.runner.iteration < iteration:
                        transfer, interrupted = self._advance_arm(
                            "transfer", transfer, iteration
                        )
                        if interrupted or self._stop_requested:
                            self._write_manifest(transfer, scratch)
                            result = self._result(transfer, scratch, completed=False)
                            self._stop_requested = False
                            return result
                    if scratch.runner.iteration < iteration:
                        scratch, interrupted = self._advance_arm(
                            "scratch", scratch, iteration
                        )
                        if interrupted or self._stop_requested:
                            self._write_manifest(transfer, scratch)
                            result = self._result(transfer, scratch, completed=False)
                            self._stop_requested = False
                            return result
                    self._write_manifest(transfer, scratch)
                completed = transfer.runner.iteration == scratch.runner.iteration == self.config.iterations
                self._write_manifest(transfer, scratch)
                return self._result(transfer, scratch, completed=completed)
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)

    def _load_arm(self, name: str) -> "_ArmState":
        latest = self._native_arm_directory(name) / "checkpoints" / "latest.pt"
        if latest.is_file():
            arm = self._resume_arm(name, latest)
            # A process may have died after TrainingRunner atomically
            # checkpointed an arm but before this coordinator froze its paired
            # artifacts.  Reconstruct those immutable copies without replaying
            # an update.
            if arm.runner.iteration:
                return self._freeze_arm_artifacts(arm, latest)
            return arm
        return self._new_arm(name)

    def _new_arm(self, name: str) -> "_ArmState":
        from .curriculum import CurriculumConfig
        from .train_runner import TrainingRunConfig, TrainingRunner

        init_checkpoint = str(self.source_path) if name == "transfer" else None
        runner = TrainingRunner(
            TrainingRunConfig(
                run=self._run_settings(),
                model=self.model_config,
                ppo=self.config.ppo,
                curriculum=CurriculumConfig(
                    base_learning_rate=self.config.ppo.learning_rate,
                    require_transfer_beats_scratch=False,
                    require_previous_checkpoint_win=False,
                ),
                league=self._league_config(),
                init_checkpoint=init_checkpoint,
                init_checkpoint_sha256=self.source_sha256 if name == "transfer" else None,
                init_checkpoint_kind="model_weights_only" if name == "transfer" else None,
            ),
            self._native_arm_directory(name),
            device=self.device,
        )
        # A transfer run already reseeds after loading its weights; resetting
        # here gives scratch the same rollout/update RNG start without changing
        # its separately seeded parameter initialisation.
        TrainingRunner._seed_everything(self.config.base_seed)
        return _ArmState(name, runner, rng_state=_capture_rng_state())

    def _resume_arm(self, name: str, path: Path) -> "_ArmState":
        from .train_runner import TrainingRunner

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"paired-rung {name} checkpoint is malformed")
        expected = self._native_config_dict(name)
        actual = payload.get("run_config")
        if not isinstance(actual, Mapping) or _canonical_sha256(dict(actual)) != _canonical_sha256(expected):
            raise ValueError(f"paired-rung {name} checkpoint does not match its frozen protocol")
        runner = TrainingRunner.resume(path, device=self.device)
        self._validate_native_arm(name, runner)
        return _ArmState(name, runner, rng_state=_capture_rng_state())

    def _advance_arm(
        self,
        name: str,
        arm: "_ArmState",
        target_iteration: int,
    ) -> tuple["_ArmState", bool]:
        before_hands = arm.runner.global_hands
        _restore_rng_state(arm.rng_state)
        # The paired layer, rather than the inner runner, owns SIGINT.  The
        # first stop request must retain the configured whole-iteration hand
        # budget, then return at this durable boundary.
        result = arm.runner.run(until_iteration=target_iteration, install_signal_handlers=False)
        arm.rng_state = _capture_rng_state()
        if arm.runner.iteration != target_iteration:
            raise RuntimeError(f"paired-rung {name} stopped before a complete iteration boundary")
        if arm.runner.global_hands - before_hands != self.config.hands_per_iteration:
            raise RuntimeError("paired rung requires an exact hand budget; PPO consumed an unexpected extra rollout")
        return self._freeze_arm_artifacts(arm, result.checkpoint_path), result.interrupted

    def _freeze_arm_artifacts(self, arm: "_ArmState", native_checkpoint: Path) -> "_ArmState":
        frozen_full = self._frozen_full_path(arm.name, arm.runner.iteration)
        recorded = self._recorded_arm(arm.name)
        if recorded is not None and recorded.get("iteration") == arm.runner.iteration:
            self._validate_recorded_artifact(arm.name, recorded)
            return _ArmState(
                arm.name,
                arm.runner,
                frozen_full,
                self._frozen_model_path(arm.name, arm.runner.iteration),
                arm.rng_state,
            )
        # ``latest.pt`` is independently serialised by TrainingRunner, so it
        # may have different bytes from the immutable checkpoint first frozen
        # at this same boundary.  Once published, retain the frozen original.
        existed = frozen_full.exists()
        _freeze_copy(native_checkpoint, frozen_full, verify_existing=False)
        if existed:
            self._validate_unrecorded_frozen_full(arm, frozen_full)
        frozen_model = self._frozen_model_path(arm.name, arm.runner.iteration)
        if frozen_model.exists():
            self._validate_unrecorded_frozen_pair(arm, frozen_full, frozen_model)
        else:
            _atomic_torch_save(
                {
                    "metadata": arm.runner.model.checkpoint_metadata(),
                    "state_dict": arm.runner.model.state_dict(),
                    "paired_rung": self._artifact_lineage(arm.name, arm.runner, frozen_full),
                },
                frozen_model,
            )
        return _ArmState(arm.name, arm.runner, frozen_full, frozen_model, arm.rng_state)

    def _run_settings(self):
        # Lazy imports keep this module usable as a serialisable config layer
        # without train_runner importing it back.
        from .train_runner import RunSettings

        return RunSettings(
            stage=self.config.target_stage,
            seed=self.config.base_seed,
            iterations=self.config.iterations,
            hands_per_iteration=self.config.hands_per_iteration,
            table_count=self.config.table_count,
            checkpoint_every_iterations=1,
            checkpoint_every_seconds=None,
        )

    def _league_config(self):
        from .train_runner import LeagueConfig, LeagueMemberConfig

        members = [LeagueMemberConfig("current", "current", self.config.current_weight)]
        members.extend(
            LeagueMemberConfig(item.name, item.kind, item.weight, item.bot, self.config.base_seed + item.seed_offset)
            for item in self.config.opponents
        )
        return LeagueConfig(seed=self.config.base_seed, members=tuple(members))

    def _native_config_dict(self, name: str) -> dict[str, Any]:
        from .curriculum import CurriculumConfig
        from .train_runner import TrainingRunConfig

        return TrainingRunConfig(
            run=self._run_settings(),
            model=self.model_config,
            ppo=self.config.ppo,
            curriculum=CurriculumConfig(
                base_learning_rate=self.config.ppo.learning_rate,
                require_transfer_beats_scratch=False,
                require_previous_checkpoint_win=False,
            ),
            league=self._league_config(),
            init_checkpoint=str(self.source_path) if name == "transfer" else None,
            init_checkpoint_sha256=self.source_sha256 if name == "transfer" else None,
            init_checkpoint_kind="model_weights_only" if name == "transfer" else None,
        ).to_dict()

    def _native_arm_directory(self, name: str) -> Path:
        return self.directory / "arms" / name / "native-run"

    def _frozen_full_path(self, name: str, iteration: int) -> Path:
        return self.directory / "arms" / name / "frozen" / f"full-{iteration:08d}.pt"

    def _frozen_model_path(self, name: str, iteration: int) -> Path:
        return self.directory / "arms" / name / "frozen" / f"model-{iteration:08d}.pt"

    def _artifact_lineage(self, name: str, runner: Any, full_path: Path) -> dict[str, Any]:
        return {
            "version": PAIRED_RUNG_MANIFEST_VERSION,
            "arm": name,
            "config_sha256": self.config_sha256,
            "source_checkpoint": str(self.source_path),
            "source_checkpoint_sha256": self.source_sha256,
            "source_run_config_sha256": self.source_run_config_sha256,
            "native_full_checkpoint": str(full_path),
            "native_full_checkpoint_sha256": _file_sha256(full_path),
            "progress": {
                "iteration": runner.iteration,
                "global_hands": runner.global_hands,
                "global_decisions": runner.global_decisions,
            },
        }

    def _result(self, transfer: "_ArmState", scratch: "_ArmState", *, completed: bool) -> PairedRungResult:
        return PairedRungResult(
            transfer=self._arm_result(transfer),
            scratch=self._arm_result(scratch),
            completed=completed,
            config_sha256=self.config_sha256,
            source_checkpoint_path=self.source_path,
            source_checkpoint_sha256=self.source_sha256,
            source_run_config_sha256=self.source_run_config_sha256,
            manifest_path=self.manifest_path,
        )

    def _arm_result(self, arm: "_ArmState") -> PairedRungArmResult:
        if arm.runner.iteration == 0:
            return PairedRungArmResult(
                name=arm.name,
                iteration=0,
                global_hands=0,
                global_decisions=0,
                full_checkpoint_path=None,
                model_checkpoint_path=None,
                full_checkpoint_sha256=None,
                model_checkpoint_sha256=None,
            )
        full = arm.frozen_full_path or self._frozen_full_path(arm.name, arm.runner.iteration)
        model = arm.frozen_model_path or self._frozen_model_path(arm.name, arm.runner.iteration)
        if not full.is_file() or not model.is_file():
            raise RuntimeError(f"paired-rung {arm.name} has no frozen artifact at iteration {arm.runner.iteration}")
        return PairedRungArmResult(
            name=arm.name,
            iteration=arm.runner.iteration,
            global_hands=arm.runner.global_hands,
            global_decisions=arm.runner.global_decisions,
            full_checkpoint_path=full,
            model_checkpoint_path=model,
            full_checkpoint_sha256=_file_sha256(full),
            model_checkpoint_sha256=_file_sha256(model),
        )

    def _ensure_manifest(self) -> None:
        if self.manifest_path.exists():
            manifest = _load_json(self.manifest_path)
            if (
                manifest.get("version") != PAIRED_RUNG_MANIFEST_VERSION
                or manifest.get("config_sha256") != self.config_sha256
                or manifest.get("source_checkpoint_sha256") != self.source_sha256
            ):
                raise ValueError("paired-rung manifest does not match config/source checkpoint")
            if manifest.get("source_checkpoint") != str(self.source_path) or not self.source_path.is_file() or _file_sha256(self.source_path) != self.source_sha256:
                raise ValueError("paired-rung frozen source checkpoint does not match manifest")
            self._validate_manifest_records(manifest)
            self._manifest = manifest
            return
        self._manifest = {
            "version": PAIRED_RUNG_MANIFEST_VERSION,
            "config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "source_checkpoint": str(self.source_path),
            "source_checkpoint_sha256": self.source_sha256,
            "source_run_config_sha256": self.source_run_config_sha256,
            "arms": {},
            "completed": False,
        }
        _atomic_write_json(self.manifest_path, self._manifest)

    def _write_manifest(self, transfer: "_ArmState", scratch: "_ArmState") -> None:
        arms: dict[str, Any] = {}
        for arm in (transfer, scratch):
            if arm.runner.iteration == 0:
                continue
            result = self._arm_result(arm)
            arms[arm.name] = {
                "iteration": result.iteration,
                "global_hands": result.global_hands,
                "global_decisions": result.global_decisions,
                "full_checkpoint": str(result.full_checkpoint_path),
                "full_checkpoint_sha256": result.full_checkpoint_sha256,
                "model_checkpoint": str(result.model_checkpoint_path),
                "model_checkpoint_sha256": result.model_checkpoint_sha256,
            }
        self._manifest = {
                "version": PAIRED_RUNG_MANIFEST_VERSION,
                "config": self.config.to_dict(),
                "config_sha256": self.config_sha256,
                "source_checkpoint": str(self.source_path),
                "source_checkpoint_sha256": self.source_sha256,
                "source_run_config_sha256": self.source_run_config_sha256,
                "arms": arms,
                "completed": (
                    transfer.runner.iteration == scratch.runner.iteration == self.config.iterations
                ),
        }
        _atomic_write_json(self.manifest_path, self._manifest)

    def validate_manifest(self) -> None:
        """Fail closed if a persisted paired artifact or counter was altered."""

        manifest = _load_json(self.manifest_path)
        self._validate_manifest_records(manifest)

    def _recorded_arm(self, name: str) -> Mapping[str, Any] | None:
        arms = self._manifest.get("arms", {})
        if not isinstance(arms, Mapping):  # Already guarded on manifest load.
            raise ValueError("paired-rung manifest arms are malformed")
        value = arms.get(name)
        return value if isinstance(value, Mapping) else None

    def _validate_manifest_records(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("config") != self.config.to_dict() or manifest.get("source_run_config_sha256") != self.source_run_config_sha256:
            raise ValueError("paired-rung manifest config/provenance does not match current protocol")
        arms = manifest.get("arms")
        if not isinstance(arms, Mapping) or any(name not in {"transfer", "scratch"} for name in arms):
            raise ValueError("paired-rung manifest arms are malformed")
        for name, record in arms.items():
            if not isinstance(record, Mapping):
                raise ValueError("paired-rung manifest arm record is malformed")
            self._validate_recorded_artifact(str(name), record)
        completed = manifest.get("completed")
        if not isinstance(completed, bool):
            raise ValueError("paired-rung manifest completed flag is malformed")
        if completed:
            if set(arms) != {"transfer", "scratch"} or any(record.get("iteration") != self.config.iterations for record in arms.values() if isinstance(record, Mapping)):
                raise ValueError("completed paired-rung manifest lacks both final arm artifacts")

    def _validate_recorded_artifact(self, name: str, record: Mapping[str, Any]) -> None:
        iteration = record.get("iteration")
        hands = record.get("global_hands")
        decisions = record.get("global_decisions")
        if (
            not isinstance(iteration, int)
            or not 1 <= iteration <= self.config.iterations
            or not isinstance(hands, int)
            or hands != iteration * self.config.hands_per_iteration
            or not isinstance(decisions, int)
            or decisions < 0
        ):
            raise ValueError(f"paired-rung {name} manifest counters are malformed")
        for path_key, hash_key in (("full_checkpoint", "full_checkpoint_sha256"), ("model_checkpoint", "model_checkpoint_sha256")):
            path_value, expected_hash = record.get(path_key), record.get(hash_key)
            if not isinstance(path_value, str) or not isinstance(expected_hash, str):
                raise ValueError(f"paired-rung {name} manifest has incomplete artifact hashes")
            path = Path(path_value)
            expected_path = self._frozen_full_path(name, iteration) if path_key == "full_checkpoint" else self._frozen_model_path(name, iteration)
            if path != expected_path or not path.is_file() or _file_sha256(path) != expected_hash:
                raise ValueError(f"paired-rung {name} frozen artifact failed hash validation")
        full_payload = torch.load(self._frozen_full_path(name, iteration), map_location="cpu", weights_only=True)
        if not isinstance(full_payload, Mapping) or _canonical_sha256(dict(full_payload.get("run_config", {}))) != _canonical_sha256(self._native_config_dict(name)):
            raise ValueError(f"paired-rung {name} frozen full artifact has the wrong native protocol")
        progress = full_payload.get("progress")
        if not isinstance(progress, Mapping) or progress.get("iteration") != iteration or progress.get("global_hands") != hands or progress.get("global_decisions") != decisions:
            raise ValueError(f"paired-rung {name} frozen full artifact counters do not match manifest")
        model_payload = torch.load(self._frozen_model_path(name, iteration), map_location="cpu", weights_only=True)
        lineage = model_payload.get("paired_rung") if isinstance(model_payload, Mapping) else None
        if not isinstance(lineage, Mapping) or lineage.get("config_sha256") != self.config_sha256 or lineage.get("arm") != name:
            raise ValueError(f"paired-rung {name} frozen model artifact has invalid lineage")

    def _validate_native_arm(self, name: str, runner: Any) -> None:
        if (
            runner.scheduler.stage is not self.config.target_stage
            or not 0 <= runner.iteration <= self.config.iterations
            or runner.global_hands != runner.iteration * self.config.hands_per_iteration
            or runner.global_decisions < 0
        ):
            raise ValueError(f"paired-rung {name} native run has invalid target-stage progress")
        recorded = self._recorded_arm(name)
        if recorded is not None and runner.iteration < recorded.get("iteration", -1):
            raise ValueError(f"paired-rung {name} native run is behind its frozen artifact")

    def _validate_unrecorded_frozen_pair(self, arm: "_ArmState", full: Path, model: Path) -> None:
        """Recover only the narrow crash window before a manifest publish."""

        self._validate_unrecorded_frozen_full(arm, full)
        model_payload = torch.load(model, map_location="cpu", weights_only=True)
        lineage = model_payload.get("paired_rung") if isinstance(model_payload, Mapping) else None
        if not isinstance(lineage, Mapping) or lineage.get("config_sha256") != self.config_sha256 or lineage.get("arm") != arm.name:
            raise ValueError("unrecorded paired-rung frozen model artifact is not recoverable")

    def _validate_unrecorded_frozen_full(self, arm: "_ArmState", full: Path) -> None:
        """Validate an artifact left by a crash before its manifest entry."""

        payload = torch.load(full, map_location="cpu", weights_only=True)
        progress = payload.get("progress") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or _canonical_sha256(dict(payload.get("run_config", {}))) != _canonical_sha256(self._native_config_dict(arm.name))
            or not isinstance(progress, Mapping)
            or progress.get("iteration") != arm.runner.iteration
            or progress.get("global_hands") != arm.runner.global_hands
        ):
            raise ValueError("unrecorded paired-rung frozen full artifact is not recoverable")


# Short form for applications that prefer a noun over the coordinator name.
PairedRung = PairedRungRunner


@dataclass
class _ArmState:
    name: str
    runner: Any
    frozen_full_path: Path | None = None
    frozen_model_path: Path | None = None
    rng_state: dict[str, Any] = field(default_factory=dict)


@contextmanager
def _preserved_global_rng() -> Iterator[None]:
    _require_torch()
    state = _capture_rng_state()
    try:
        yield
    finally:
        _restore_rng_state(state)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["cuda"] = tuple(item.clone() for item in torch.cuda.get_rng_state_all())
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(list(state["cuda"]))


def _source_run_config_sha256(path: Path) -> str | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint is malformed")
    run_config = payload.get("run_config")
    return _canonical_sha256(dict(run_config)) if isinstance(run_config, Mapping) else None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze_copy(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    verify_existing: bool = True,
) -> None:
    expected = expected_sha256 or _file_sha256(source)
    if destination.exists():
        if verify_existing and _file_sha256(destination) != expected:
            raise ValueError(f"immutable paired-rung artifact hash mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
        _publish_file(temporary, destination)
        if _file_sha256(destination) != expected:
            raise RuntimeError("atomic paired-rung artifact copy did not match source hash")
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        _publish_file(temporary, path)
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid paired-rung manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("paired-rung manifest must be an object")
    return value


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
    "PAIRED_RUNG_MANIFEST_VERSION",
    "PairedRung",
    "PairedRungArmResult",
    "PairedRungConfig",
    "PairedRungOpponentConfig",
    "PairedRungResult",
    "PairedRungRunner",
]
