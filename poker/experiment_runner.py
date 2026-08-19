"""Resumable single-trial runner backed by the immutable experiment ledger."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from .experiments import ExperimentConfig, ExperimentLedger, _atomic_write_json
from .train_runner import TrainingRunResult, TrainingRunner
from .training import NonFiniteTrainingError, UpdateMetrics


@dataclass(frozen=True)
class ExperimentRunResult:
    trial_id: str
    status: str
    iteration: int
    global_hands: int
    global_decisions: int
    checkpoint_path: Path | None
    ledger_manifest_path: Path
    metrics_path: Path
    failure_path: Path | None = None


class ExperimentRunner:
    """Run exactly one immutable tuning trial and journal every boundary."""

    def __init__(self, config: ExperimentConfig, run_directory: str | Path, *, device: str | None = None) -> None:
        self.config = config
        self.run_directory = Path(run_directory)
        self.training_directory = self.run_directory / "training"
        self.device = device
        self.ledger = ExperimentLedger(self.run_directory, config)
        latest = self.training_directory / "checkpoints" / "latest.pt"
        if latest.is_file():
            self.trainer = TrainingRunner.resume(latest, device=device)
            if self.trainer.config.to_dict() != config.training.to_dict():
                raise ValueError("resumed trainer config does not match immutable experiment config")
            self._recover_published_boundary()
            # Continue from the immutable artifact actually bound by the
            # ledger, never from mutable latest.pt.
            last_event = self.ledger.last_event
            if last_event is not None:
                self.trainer = TrainingRunner.resume(last_event.checkpoint_path, device=device)
        else:
            if self.ledger.last_iteration:
                raise ValueError("experiment ledger exists but native training checkpoint is missing")
            self.trainer = TrainingRunner(config.training, self.training_directory, device=device)

    def run(
        self,
        *,
        until_iteration: int | None = None,
        install_signal_handlers: bool = True,
    ) -> ExperimentRunResult:
        target = self.config.max_iterations if until_iteration is None else until_iteration
        if not isinstance(target, int) or isinstance(target, bool) or not self.trainer.iteration <= target <= self.config.max_iterations:
            raise ValueError("until_iteration must be between current progress and the experiment budget")
        self.ledger.write_status("running")
        try:
            result = self.trainer.run(
                until_iteration=target,
                install_signal_handlers=install_signal_handlers,
                checkpoint_observer=self._observe_checkpoint,
            )
        except NonFiniteTrainingError as error:
            failure = self.ledger.record_failure("nonfinite_training", str(error))
            self.ledger.write_status("failed_nonfinite")
            return self._result("failed_nonfinite", None, failure)
        except KeyboardInterrupt:
            # The second Ctrl+C intentionally publishes no checkpoint because
            # the update may be partial.  Make that state explicit instead of
            # leaving a stale `running` marker.
            self.ledger.write_status("aborted_immediate")
            return self._result("aborted_immediate", None, None)
        status = "manual_interrupt" if result.interrupted else (
            "completed" if result.iteration == self.config.max_iterations else "paused"
        )
        self.ledger.write_status(status)
        return self._result(status, result, None)

    def _observe_checkpoint(self, path: Path, metrics: UpdateMetrics | None) -> None:
        # Iteration-zero/pending-control checkpoints have no PPO metric and are
        # not part of a tuning learning curve.
        if metrics is None:
            return
        self.ledger.record_checkpoint(path)

    def _recover_published_boundary(self) -> None:
        if self.trainer.iteration < self.ledger.last_iteration:
            raise ValueError("mutable latest checkpoint predates the experiment ledger")
        if self.trainer.iteration == self.ledger.last_iteration:
            return
        if self.trainer.iteration != self.ledger.last_iteration + 1:
            raise ValueError("experiment ledger is missing more than one completed iteration")
        # The full latest checkpoint embeds the manifest as it existed before
        # publication.  This survives the crash window where latest.pt was
        # durable but training/manifest.json was not yet replaced.
        records = self.trainer.manifest.get("checkpoints")
        if not isinstance(records, list):
            raise ValueError("native training manifest has malformed checkpoint records")
        candidates = [
            item for item in records
            if isinstance(item, Mapping) and item.get("iteration") == self.trainer.iteration
        ]
        if not candidates or not isinstance(candidates[-1].get("path"), str):
            raise ValueError("ledger lags training progress but no immutable recovery checkpoint exists")
        self.ledger.recover_latest(Path(str(candidates[-1]["path"])))

    def _result(
        self,
        status: str,
        training_result: TrainingRunResult | None,
        failure_path: Path | None,
    ) -> ExperimentRunResult:
        checkpoint = None if training_result is None else training_result.checkpoint_path
        return ExperimentRunResult(
            self.config.trial_id,
            status,
            self.trainer.iteration,
            self.trainer.global_hands,
            self.trainer.global_decisions,
            checkpoint,
            self.ledger.manifest_path,
            self.ledger.metrics_path,
            failure_path,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid experiment config: {source}") from error
    if not isinstance(value, Mapping):
        raise ValueError("experiment config must be a JSON object")
    return ExperimentConfig.from_dict(value)


def write_experiment_config(config: ExperimentConfig, path: str | Path) -> Path:
    destination = Path(path)
    _atomic_write_json(destination, config.to_dict())
    return destination


__all__ = [
    "ExperimentRunResult",
    "ExperimentRunner",
    "load_experiment_config",
    "write_experiment_config",
]
