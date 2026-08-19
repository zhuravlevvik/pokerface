"""Durable orchestration for Stage 1 equity/backbone pretraining.

The runner joins corpus generation and the reusable minibatch trainer while
keeping their contracts independently testable.  Checkpoints are written only
after complete epochs, so Ctrl+C resume never has to reconstruct a partially
applied optimizer step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .curriculum import CurriculumStage, stage_spec
from .equity import equity_metrics, expected_showdown_share_metrics
from .model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from .pretraining import EquityBackbonePretrainer, PretrainingConfig, PretrainingMetrics
from .pretraining_data import (
    BASELINE_BOT_NAMES,
    PretrainingCorpus,
    SeedRange,
    generate_pretraining_corpus,
    load_pretraining_corpus,
    write_pretraining_corpus,
)
from .traces import EQUITY_LABEL_PROTOCOL

if TORCH_AVAILABLE:
    import torch


PRETRAINING_RUN_VERSION = 2


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for pretraining; install the project with `.[rl]`.")


@dataclass(frozen=True)
class CorpusConfig:
    """Reproducible, hand-disjoint corpus generation settings."""

    stage: CurriculumStage = CurriculumStage.A_HEADS_UP_STARTER
    train_seed_start: int = 0
    train_hands: int = 2_000
    holdout_seed_start: int = 1_000_000
    holdout_hands: int = 500
    train_equity_samples: int = 16
    holdout_equity_samples: int = 128
    bot_mix: tuple[str, ...] = BASELINE_BOT_NAMES
    balanced_samples_per_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.train_hands < 1 or self.holdout_hands < 1:
            raise ValueError("train_hands and holdout_hands must be positive")
        if self.train_equity_samples < 1 or self.holdout_equity_samples < 1:
            raise ValueError("equity sample counts must be positive")
        if self.balanced_samples_per_epoch is not None and self.balanced_samples_per_epoch < 1:
            raise ValueError("balanced_samples_per_epoch must be positive or null")
        train = SeedRange(self.train_seed_start, self.train_hands)
        holdout = SeedRange(self.holdout_seed_start, self.holdout_hands)
        if train.overlaps(holdout):
            raise ValueError("train and holdout seed ranges must be disjoint")
        if not self.bot_mix or any(name not in BASELINE_BOT_NAMES for name in self.bot_mix):
            raise ValueError("bot_mix contains an unsupported baseline bot")
        # Stage 1 is deliberately a small heads-up warm-up.  The explicit
        # expected-showdown-share label is multiway-correct, but widening the
        # corpus belongs to later curriculum stages rather than this runner.
        if stage_spec(self.stage).player_count != 2:
            raise ValueError("Stage 1 pretraining runner currently supports heads-up stages A/B only")


@dataclass(frozen=True)
class AcceptanceConfig:
    """Report gates; a failed gate is evidence, not a training-process error."""

    maximum_ece: float = 0.08
    maximum_scalar_mae: float = 0.08
    minimum_brier_reduction: float = 0.0
    minimum_stratum_samples: int = 100
    minimum_supported_strata: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_ece <= 1.0:
            raise ValueError("maximum_ece must be in [0, 1]")
        if not 0.0 <= self.maximum_scalar_mae <= 1.0:
            raise ValueError("maximum_scalar_mae must be in [0, 1]")
        if self.minimum_brier_reduction > 1.0:
            raise ValueError("minimum_brier_reduction must not exceed 1")
        if self.minimum_stratum_samples < 1:
            raise ValueError("minimum_stratum_samples must be positive")
        if self.minimum_supported_strata < 1:
            raise ValueError("minimum_supported_strata must be positive")


@dataclass(frozen=True)
class PretrainingRunConfig:
    """Fully serialisable Stage 1 experiment contract."""

    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: PretrainingConfig = field(default_factory=lambda: PretrainingConfig(epochs=10))
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    checkpoint_every_epochs: int = 1

    def __post_init__(self) -> None:
        if self.checkpoint_every_epochs < 1:
            raise ValueError("checkpoint_every_epochs must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["corpus"]["stage"] = self.corpus.stage.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainingRunConfig":
        def section(name: str) -> dict[str, Any]:
            value = data.get(name, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            return dict(value)

        corpus = section("corpus")
        if "stage" in corpus:
            corpus["stage"] = CurriculumStage(corpus["stage"])
        if "bot_mix" in corpus:
            names = corpus["bot_mix"]
            if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
                raise ValueError("corpus.bot_mix must be an array")
            corpus["bot_mix"] = tuple(names)
        checkpoint_every = data.get("checkpoint_every_epochs", 1)
        if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int):
            raise ValueError("checkpoint_every_epochs must be an integer")
        return cls(
            corpus=CorpusConfig(**corpus),
            model=ModelConfig(**section("model")),
            training=PretrainingConfig(**section("training")),
            acceptance=AcceptanceConfig(**section("acceptance")),
            checkpoint_every_epochs=checkpoint_every,
        )


def load_pretraining_run_config(path: str | Path) -> PretrainingRunConfig:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".toml":
        import tomllib

        data = tomllib.loads(text)
    else:
        data = json.loads(text)
    if not isinstance(data, Mapping):
        raise ValueError("pretraining configuration must be an object")
    return PretrainingRunConfig.from_dict(data)


def write_pretraining_run_config(config: PretrainingRunConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(destination, json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        except OSError:  # pragma: no cover - filesystem dependent.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _corpus_matches_config(corpus: PretrainingCorpus, config: CorpusConfig) -> bool:
    return (
        corpus.stage == config.stage
        and corpus.train_seed_range == SeedRange(config.train_seed_start, config.train_hands)
        and corpus.holdout_seed_range == SeedRange(config.holdout_seed_start, config.holdout_hands)
        and corpus.bot_mix == config.bot_mix
        and corpus.train_equity_samples == config.train_equity_samples
        and corpus.holdout_equity_samples == config.holdout_equity_samples
    )


def _group_key(example: Any) -> str:
    observation = example.observation
    cards, table, players = observation["cards"], observation["table"], observation["player_set"]
    return f"{cards['street']}|players={len(players)}|active_opponents={int(table['active_player_count']) - 1}"


def _target_rows(records: Sequence[Any]) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in record.example.equity_target) for record in records]


def _mean_target(records: Sequence[Any]) -> tuple[float, float, float]:
    targets = _target_rows(records)
    if not targets:
        raise ValueError("cannot calculate an empirical prior from an empty split")
    return tuple(sum(row[index] for row in targets) / len(targets) for index in range(3))  # type: ignore[return-value]


def _share_targets(records: Sequence[Any]) -> list[float]:
    return [float(record.example.expected_showdown_share_target) for record in records]


def _mean_share(records: Sequence[Any]) -> float:
    targets = _share_targets(records)
    if not targets:
        raise ValueError("cannot calculate an empirical prior from an empty split")
    return sum(targets) / len(targets)


def _brier_reduction(model: Any, baseline: Any) -> float:
    return 1.0 - model.brier_score / max(baseline.brier_score, 1e-12)


def _label_quality(records: Sequence[Any]) -> dict[str, Any]:
    samples = [record.example.equity_samples for record in records]
    if not samples or any(not isinstance(value, int) for value in samples):
        raise ValueError("corpus records lack equity sample provenance")
    exact = sum(record.example.equity_exact is True for record in records)
    return {
        "decisions": len(records),
        "exact_decisions": exact,
        "monte_carlo_decisions": len(records) - exact,
        "minimum_samples": min(samples),
        "maximum_samples": max(samples),
    }


@dataclass(frozen=True)
class PretrainingRunResult:
    epoch: int
    global_step: int
    interrupted: bool
    checkpoint_path: Path
    report_path: Path
    acceptance_passed: bool


class PretrainingRunner:
    """Own one reproducible corpus, trainer, report stream and checkpoint set."""

    def __init__(self, config: PretrainingRunConfig, run_directory: str | Path, *, device: str | None = None) -> None:
        _require_torch()
        self.config = config
        self.run_directory = Path(run_directory)
        self.checkpoint_directory = self.run_directory / "checkpoints"
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.corpus_path = self.run_directory / "corpus.jsonl"
        self.report_path = self.run_directory / "report.json"
        self.manifest_path = self.run_directory / "manifest.json"
        self._seed_everything(config.training.seed)
        self.corpus = self._load_or_generate_corpus()
        self.corpus_sha256 = _file_hash(self.corpus_path)
        self.config_sha256 = _canonical_hash(config.to_dict())
        training = config.training
        if device is not None:
            training = PretrainingConfig(**{**asdict(training), "device": device})
        provenance = self._provenance()
        self.pretrainer = EquityBackbonePretrainer(PokerAgentModel(config.model), training, provenance=provenance)
        self.manifest: dict[str, Any] = {"version": PRETRAINING_RUN_VERSION, "checkpoints": []}
        self._stop_requested = False

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_or_generate_corpus(self) -> PretrainingCorpus:
        settings = self.config.corpus
        if self.corpus_path.exists():
            corpus = load_pretraining_corpus(self.corpus_path)
            if not _corpus_matches_config(corpus, settings):
                raise ValueError("existing corpus.jsonl does not match the requested run configuration")
            return corpus
        corpus = generate_pretraining_corpus(
            settings.stage,
            train_seed_range=SeedRange(settings.train_seed_start, settings.train_hands),
            holdout_seed_range=SeedRange(settings.holdout_seed_start, settings.holdout_hands),
            bot_mix=settings.bot_mix,
            train_equity_samples=settings.train_equity_samples,
            holdout_equity_samples=settings.holdout_equity_samples,
        )
        write_pretraining_corpus(corpus, self.corpus_path)
        return corpus

    def _provenance(self) -> dict[str, str]:
        return {
            "run_version": str(PRETRAINING_RUN_VERSION),
            "run_config_json": json.dumps(self.config.to_dict(), sort_keys=True, separators=(",", ":")),
            "run_config_sha256": self.config_sha256,
            "corpus_sha256": self.corpus_sha256,
            "equity_label_protocol": EQUITY_LABEL_PROTOCOL,
        }

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_directory / "latest.pt"

    def request_stop(self) -> None:
        self._stop_requested = True

    def _training_examples(self) -> tuple[Any, ...]:
        settings = self.config.corpus
        indices = self.corpus.balanced_indices(
            "train",
            count=settings.balanced_samples_per_epoch,
            seed=self.config.training.seed + self.pretrainer.epoch,
        )
        return tuple(self.corpus.train[index].example for index in indices)

    def build_report(self, metrics: PretrainingMetrics | None = None) -> dict[str, Any]:
        """Evaluate against train-derived empirical priors and safe strata."""

        holdout = self.corpus.holdout_dataset
        validation = self.pretrainer.evaluate(holdout, breakdowns={"context": _group_key})
        train_groups: dict[str, list[Any]] = {}
        holdout_groups: dict[str, list[Any]] = {}
        for record in self.corpus.train:
            train_groups.setdefault(_group_key(record.example), []).append(record)
        for record in self.corpus.holdout:
            holdout_groups.setdefault(_group_key(record.example), []).append(record)

        overall_prior = _mean_target(self.corpus.train)
        overall_targets = _target_rows(self.corpus.holdout)
        overall_baseline = equity_metrics([overall_prior] * len(overall_targets), overall_targets, bins=self.config.training.calibration_bins)
        aggregate_reduction = _brier_reduction(validation.equity, overall_baseline)
        overall_share_targets = _share_targets(self.corpus.holdout)
        overall_share_prior = _mean_share(self.corpus.train)
        aggregate_baseline_share = expected_showdown_share_metrics(
            [overall_share_prior] * len(overall_share_targets),
            overall_share_targets,
            bins=self.config.training.calibration_bins,
        )
        aggregate_share_reduction = _brier_reduction(
            validation.expected_showdown_share,
            aggregate_baseline_share,
        )
        strata: dict[str, Any] = {}
        supported_passes: list[bool] = []
        for key, records in sorted(holdout_groups.items()):
            prior = _mean_target(train_groups.get(key, self.corpus.train))
            targets = _target_rows(records)
            baseline = equity_metrics([prior] * len(targets), targets, bins=self.config.training.calibration_bins)
            model_metrics = validation.breakdowns["context"][key]
            reduction = _brier_reduction(model_metrics, baseline)
            share_targets = _share_targets(records)
            share_prior = _mean_share(train_groups.get(key, self.corpus.train))
            share_metrics = validation.expected_showdown_share_breakdowns["context"][key]
            share_baseline = expected_showdown_share_metrics(
                [share_prior] * len(share_targets),
                share_targets,
                bins=self.config.training.calibration_bins,
            )
            share_reduction = _brier_reduction(share_metrics, share_baseline)
            supported = len(records) >= self.config.acceptance.minimum_stratum_samples
            if supported:
                supported_passes.append(
                    share_reduction >= self.config.acceptance.minimum_brier_reduction
                    and share_metrics.expected_calibration_error <= self.config.acceptance.maximum_ece
                    and share_metrics.mean_absolute_error <= self.config.acceptance.maximum_scalar_mae
                )
            strata[key] = {
                "support": len(records),
                "supported": supported,
                "outcome_model": model_metrics.as_dict(),
                "expected_showdown_share": share_metrics.as_dict(),
                "outcome_empirical_prior": baseline.as_dict(),
                "expected_showdown_share_empirical_prior": share_baseline.as_dict(),
                "outcome_brier_reduction": reduction,
                "expected_showdown_share_brier_reduction": share_reduction,
            }

        acceptance_passed = (
            validation.expected_showdown_share.expected_calibration_error <= self.config.acceptance.maximum_ece
            and validation.expected_showdown_share.mean_absolute_error <= self.config.acceptance.maximum_scalar_mae
            and aggregate_share_reduction >= self.config.acceptance.minimum_brier_reduction
            and len(supported_passes) >= self.config.acceptance.minimum_supported_strata
            and all(supported_passes)
        )
        report: dict[str, Any] = {
            "version": PRETRAINING_RUN_VERSION,
            "stage": self.config.corpus.stage.value,
            "player_count": stage_spec(self.config.corpus.stage).player_count,
            "epoch": self.pretrainer.epoch,
            "global_step": self.pretrainer.global_step,
            "corpus_sha256": self.corpus_sha256,
            "run_config_sha256": self.config_sha256,
            "label_protocol": EQUITY_LABEL_PROTOCOL,
            "target_semantics": "fixed-deal virtual showdown; not public-range equity",
            "scalar_metric": {
                "name": "expected_showdown_share",
                "protocol": "active_hands_expected_showdown_share_v1",
                "semantics": "mean fractional share among active best hands at fixed-deal virtual showdown",
            },
            "label_quality": {
                "train": _label_quality(self.corpus.train),
                "holdout": _label_quality(self.corpus.holdout),
            },
            "training_metrics": None if metrics is None else asdict(metrics),
            "aggregate": {
                "outcome_model": validation.equity.as_dict(),
                "expected_showdown_share": validation.expected_showdown_share.as_dict(),
                "outcome_empirical_prior": overall_baseline.as_dict(),
                "expected_showdown_share_empirical_prior": aggregate_baseline_share.as_dict(),
                "outcome_brier_reduction": aggregate_reduction,
                "expected_showdown_share_brier_reduction": aggregate_share_reduction,
            },
            "strata": strata,
            "acceptance": {
                "passed": acceptance_passed,
                "maximum_ece": self.config.acceptance.maximum_ece,
                "maximum_scalar_mae": self.config.acceptance.maximum_scalar_mae,
                "minimum_brier_reduction": self.config.acceptance.minimum_brier_reduction,
                "minimum_stratum_samples": self.config.acceptance.minimum_stratum_samples,
                "minimum_supported_strata": self.config.acceptance.minimum_supported_strata,
                "supported_strata": len(supported_passes),
                "insufficient_evidence": len(supported_passes) < self.config.acceptance.minimum_supported_strata,
            },
        }
        _atomic_write_text(self.report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report

    def save_checkpoint(self, *, reason: str, metrics: PretrainingMetrics | None = None) -> Path:
        tag = f"{reason}_{self.pretrainer.epoch:08d}.pt"
        path = self.checkpoint_directory / tag
        self.pretrainer.provenance = self._provenance()
        self.pretrainer.save_checkpoint(path)
        self.pretrainer.save_checkpoint(self.latest_path)
        records = self.manifest.setdefault("checkpoints", [])
        if not isinstance(records, list):
            raise ValueError("pretraining manifest checkpoint list is corrupted")
        record = {"path": str(path), "epoch": self.pretrainer.epoch, "reason": reason}
        if not records or records[-1] != record:
            records.append(record)
        self.manifest.update(
            {
                "latest": str(self.latest_path),
                "report": str(self.report_path),
                "corpus": str(self.corpus_path),
                "updated_at": time.time(),
                "last_metrics": None if metrics is None else asdict(metrics),
            }
        )
        _atomic_write_text(self.manifest_path, json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def resume(cls, path: str | Path, *, device: str | None = None) -> "PretrainingRunner":
        _require_torch()
        checkpoint = Path(path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("pretraining_checkpoint_version") is None:
            raise ValueError("not a compatible pretraining checkpoint")
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or not isinstance(provenance.get("run_config_json"), str):
            raise ValueError("checkpoint has no Stage 1 run provenance")
        config_data = json.loads(provenance["run_config_json"])
        if not isinstance(config_data, Mapping):
            raise ValueError("checkpoint Stage 1 run configuration is invalid")
        config = PretrainingRunConfig.from_dict(config_data)
        instance = cls.__new__(cls)
        instance.config = config
        instance.run_directory = checkpoint.parent.parent
        instance.checkpoint_directory = instance.run_directory / "checkpoints"
        instance.corpus_path = instance.run_directory / "corpus.jsonl"
        instance.report_path = instance.run_directory / "report.json"
        instance.manifest_path = instance.run_directory / "manifest.json"
        instance.corpus = load_pretraining_corpus(instance.corpus_path)
        if not _corpus_matches_config(instance.corpus, config.corpus):
            raise ValueError("resume corpus does not match checkpoint run configuration")
        instance.corpus_sha256 = _file_hash(instance.corpus_path)
        instance.config_sha256 = _canonical_hash(config.to_dict())
        if instance.corpus_sha256 != provenance.get("corpus_sha256") or instance.config_sha256 != provenance.get("run_config_sha256"):
            raise ValueError("checkpoint provenance does not match local corpus/config")
        instance.pretrainer = EquityBackbonePretrainer.load_checkpoint(checkpoint, device=device)
        if instance.manifest_path.exists():
            manifest = json.loads(instance.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise ValueError("pretraining manifest must be an object")
            instance.manifest = dict(manifest)
        else:
            instance.manifest = {"version": PRETRAINING_RUN_VERSION, "checkpoints": []}
        instance._stop_requested = False
        return instance

    def run(self, *, until_epoch: int | None = None, install_signal_handlers: bool = True) -> PretrainingRunResult:
        target = self.config.training.epochs if until_epoch is None else until_epoch
        if target < self.pretrainer.epoch:
            raise ValueError("target epoch precedes the restored run")
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
            last_metrics: PretrainingMetrics | None = None
            while self.pretrainer.epoch < target:
                last_metrics = self.pretrainer.train_epoch(self._training_examples())
                report = self.build_report(last_metrics)
                if self._stop_requested:
                    path = self.save_checkpoint(reason="interrupt", metrics=last_metrics)
                    return PretrainingRunResult(
                        self.pretrainer.epoch,
                        self.pretrainer.global_step,
                        True,
                        path,
                        self.report_path,
                        bool(report["acceptance"]["passed"]),
                    )
                if self.pretrainer.epoch % self.config.checkpoint_every_epochs == 0:
                    self.save_checkpoint(reason="periodic", metrics=last_metrics)
            report = self.build_report(last_metrics)
            path = self.save_checkpoint(reason="complete", metrics=last_metrics)
            return PretrainingRunResult(
                self.pretrainer.epoch,
                self.pretrainer.global_step,
                False,
                path,
                self.report_path,
                bool(report["acceptance"]["passed"]),
            )
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)


__all__ = [
    "AcceptanceConfig",
    "CorpusConfig",
    "PRETRAINING_RUN_VERSION",
    "PretrainingRunConfig",
    "PretrainingRunResult",
    "PretrainingRunner",
    "load_pretraining_run_config",
    "write_pretraining_run_config",
]
