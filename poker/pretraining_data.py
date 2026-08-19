"""Inspectable, restricted reproducibility corpora for poker pretraining.

This module deliberately sits between the trace generator and a future
pretraining loop.  Model inputs remain player-safe and private equity snapshots
never leave ``generate_pretraining_dataset``.  The file itself is restricted:
its audit seeds intentionally reproduce the whole deal and must not be exposed
to a policy or browser client.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .betting import Action
from .bots import AggroBot, CallingStationBot, RandomBot, RuleBot, TightBot
from .curriculum import CurriculumStage, PretrainingExample, TracePretrainingDataset, generate_pretraining_dataset
from .observation import OBSERVATION_VERSION
from .traces import EQUITY_LABEL_PROTOCOL


PRETRAINING_CORPUS_SCHEMA_VERSION = "2.0"
"""Version of the on-disk JSONL corpus contract."""

BASELINE_BOT_NAMES = ("rule", "tight", "aggro", "calling_station", "random")
DEFAULT_EQUITY_BUCKETS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_PRETRAINING_STAGE = CurriculumStage.A_HEADS_UP_STARTER
"""Stage 1 defaults to heads-up; multiway labels are analysis-only for now."""


@dataclass(frozen=True)
class SeedRange:
    """A half-open, explicitly named collection of deterministic hand seeds."""

    start: int
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("seed range count must be positive")

    @property
    def stop(self) -> int:
        return self.start + self.count

    def overlaps(self, other: "SeedRange") -> bool:
        return self.start < other.stop and other.start < self.stop

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "stop": self.stop, "count": self.count}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeedRange":
        start, count = data.get("start"), data.get("count")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("seed range must contain integer start and count")
        result = cls(start=start, count=count)
        stop = data.get("stop")
        if stop is not None and stop != result.stop:
            raise ValueError("seed range stop does not match start + count")
        return result


@dataclass(frozen=True, order=True)
class PretrainingStratum:
    """The sampling cell for one decision, derived only from safe data."""

    street: str
    active_player_count: int
    hero_position: str
    equity_bucket: str
    action: str

    def key(self) -> str:
        return "|".join((self.street, str(self.active_player_count), self.hero_position, self.equity_bucket, self.action))

    def as_dict(self) -> dict[str, Any]:
        return {
            "street": self.street,
            "active_player_count": self.active_player_count,
            "hero_position": self.hero_position,
            "equity_bucket": self.equity_bucket,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainingStratum":
        street, active, position, bucket, action = (
            data.get("street"),
            data.get("active_player_count"),
            data.get("hero_position"),
            data.get("equity_bucket"),
            data.get("action"),
        )
        if not all(isinstance(value, str) for value in (street, position, bucket, action)) or isinstance(active, bool) or not isinstance(active, int):
            raise ValueError("malformed pretraining stratum")
        return cls(street, active, position, bucket, action)


@dataclass(frozen=True)
class CorpusRecord:
    """One serialisable training record with safe observation and audit fields."""

    example: PretrainingExample
    stratum: PretrainingStratum
    behavior_policy: str = "baseline:unknown-v1"


@dataclass(frozen=True)
class PretrainingCorpus:
    """Disjoint train/holdout corpora plus enough metadata to audit generation."""

    stage: CurriculumStage
    train_seed_range: SeedRange
    holdout_seed_range: SeedRange
    bot_mix: tuple[str, ...]
    train_equity_samples: int
    holdout_equity_samples: int
    train: tuple[CorpusRecord, ...]
    holdout: tuple[CorpusRecord, ...]

    def __post_init__(self) -> None:
        if self.train_seed_range.overlaps(self.holdout_seed_range):
            raise ValueError("train and holdout seed ranges must be disjoint")
        if not self.bot_mix or any(name not in BASELINE_BOT_NAMES for name in self.bot_mix):
            raise ValueError("bot_mix must contain supported baseline bot names")
        if self.train_equity_samples < 1 or self.holdout_equity_samples < 1:
            raise ValueError("equity sample counts must be positive")
        _validate_split_records(
            self.train,
            "train",
            self.train_seed_range,
            self.train_equity_samples,
            self.stage,
            self.bot_mix,
        )
        _validate_split_records(
            self.holdout,
            "holdout",
            self.holdout_seed_range,
            self.holdout_equity_samples,
            self.stage,
            self.bot_mix,
        )

    @property
    def equity_samples(self) -> int | None:
        """Compatibility convenience; ``None`` intentionally exposes split drift."""

        return self.train_equity_samples if self.train_equity_samples == self.holdout_equity_samples else None

    @property
    def train_dataset(self) -> TracePretrainingDataset:
        return TracePretrainingDataset(record.example for record in self.train)

    @property
    def holdout_dataset(self) -> TracePretrainingDataset:
        return TracePretrainingDataset(record.example for record in self.holdout)

    def summary(self) -> dict[str, dict[str, int]]:
        return {"train": strata_summary(self.train), "holdout": strata_summary(self.holdout)}

    def balanced_indices(self, split: str, *, count: int | None = None, seed: int = 0) -> tuple[int, ...]:
        return balanced_indices(self.train if split == "train" else self.holdout if split == "holdout" else _bad_split(split), count=count, seed=seed)


def _bad_split(split: str) -> tuple[CorpusRecord, ...]:
    raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")


def _validate_split_records(
    records: Sequence[CorpusRecord],
    split: str,
    seed_range: SeedRange,
    requested_samples: int,
    stage: CurriculumStage,
    bot_mix: Sequence[str],
) -> None:
    if not records:
        raise ValueError(f"{split} split must contain at least one decision")
    expected_seeds = set(range(seed_range.start, seed_range.stop))
    observed_seeds: set[int] = set()
    mixture = BaselineBotMixture(bot_mix)
    for record in records:
        example = record.example
        if example.stage != stage or example.seed not in expected_seeds:
            raise ValueError(f"{split} row seed/stage is outside declared split metadata")
        if example.hand_id != example.seed - seed_range.start:
            raise ValueError(f"{split} row hand_id does not match its declared seed range")
        observed_seeds.add(example.seed)
        target = example.equity_target
        if (
            len(target) != 3
            or any(not isfinite(value) or value < 0.0 for value in target)
            or abs(sum(target) - 1.0) > 1e-8
        ):
            raise ValueError(f"{split} row has an invalid equity target")
        if (
            isinstance(example.equity_samples, bool)
            or not isinstance(example.equity_samples, int)
            or example.equity_samples < 1
            or not isinstance(example.equity_exact, bool)
            or example.label_protocol != EQUITY_LABEL_PROTOCOL
        ):
            raise ValueError(f"{split} row has invalid equity label provenance")
        expected_share = example.expected_showdown_share_target
        if (
            isinstance(expected_share, bool)
            or not isinstance(expected_share, (int, float))
            or not isfinite(float(expected_share))
            or not 0.0 <= float(expected_share) <= 1.0
        ):
            raise ValueError(f"{split} row has an invalid expected showdown-share target")
        if (not example.equity_exact and example.equity_samples != requested_samples) or (
            example.equity_exact and example.equity_samples > requested_samples
        ):
            raise ValueError(f"{split} row equity sample count contradicts split metadata")
        if record.stratum != stratum_for(example):
            raise ValueError(f"{split} row has inconsistent stratum metadata")
        expected_policy = mixture.policy_id_for(example.observation)
        if record.behavior_policy != expected_policy or example.behavior_policy != expected_policy:
            raise ValueError(f"{split} row has inconsistent behavior-policy provenance")
    if observed_seeds != expected_seeds:
        raise ValueError(f"{split} split does not cover every declared hand seed")


class BaselineBotMixture:
    """A deterministic mixture whose decisions use only the acting player's view.

    A bot identity is stable for a player's hand (cards + position), while its
    seeded stochastic choices may still depend on public action history.  New
    bot objects are cheap and avoid carrying RNG state across corpus examples.
    """

    def __init__(self, bot_mix: Sequence[str] = BASELINE_BOT_NAMES) -> None:
        names = tuple(str(name) for name in bot_mix)
        if not names or any(name not in BASELINE_BOT_NAMES for name in names):
            raise ValueError(f"bot_mix must be a non-empty subset of {BASELINE_BOT_NAMES}")
        self.bot_mix = names

    def select_action(self, observation: Mapping[str, Any], legal_actions: Mapping[str, bool], _seat: int) -> Action:
        bot_name = self.bot_name_for(observation)
        # The full observation is still player-safe and makes random baselines
        # deterministic at each public decision point.
        bot_seed = _stable_number(observation) % (2**63)
        if bot_name == "rule":
            return RuleBot().select_action(observation, legal_actions)
        if bot_name == "tight":
            return TightBot().select_action(observation, legal_actions)
        if bot_name == "aggro":
            return AggroBot(seed=bot_seed).select_action(observation, legal_actions)
        if bot_name == "calling_station":
            return CallingStationBot(seed=bot_seed).select_action(observation, legal_actions)
        return RandomBot(seed=bot_seed).select_action(observation, legal_actions)

    def bot_name_for(self, observation: Mapping[str, Any]) -> str:
        """Return the stable policy identity for this player's current hand."""

        cards = observation.get("cards")
        hero = observation.get("hero")
        player_set = observation.get("player_set")
        if not isinstance(cards, Mapping) or not isinstance(hero, Mapping) or not isinstance(player_set, list):
            raise ValueError("observation lacks fields required for baseline policy identity")
        # Board, stack, commitment and action history evolve within a hand.
        # These fields do not, and all are visible to the acting player.
        identity = _stable_number(
            {
                "hole_cards": cards.get("hole_cards"),
                "hero_seat": hero.get("seat"),
                "hero_position": hero.get("position"),
                "table_player_count": len(player_set),
            }
        )
        return self.bot_mix[identity % len(self.bot_mix)]

    def policy_id_for(self, observation: Mapping[str, Any]) -> str:
        return f"baseline:{self.bot_name_for(observation)}:v1"


def _stable_number(value: Any) -> int:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big")


def equity_bucket(expected_showdown_share: float, *, boundaries: Sequence[float] = DEFAULT_EQUITY_BUCKETS) -> str:
    """Return a stable bucket for expected showdown share only.

    The scalar is averaged across active showdown hands and is explicitly not
    a current-pot or side-pot share.  Buckets are left-inclusive and the final
    bucket includes one.
    """

    if isinstance(expected_showdown_share, bool) or not isinstance(expected_showdown_share, (int, float)):
        raise ValueError("expected showdown share must be numeric")
    if len(boundaries) < 2 or boundaries[0] != 0.0 or boundaries[-1] != 1.0 or any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("equity bucket boundaries must increase from 0.0 to 1.0")
    value = float(expected_showdown_share)
    if not 0.0 <= value <= 1.0:
        raise ValueError("expected showdown share must be in [0, 1]")
    for lower, upper in zip(boundaries, boundaries[1:]):
        if value < upper or upper == 1.0:
            return f"{lower:.1f}-{upper:.1f}"
    raise AssertionError("unreachable equity bucket")


def stratum_for(example: PretrainingExample) -> PretrainingStratum:
    """Derive sampling metadata from a model-safe example only."""

    observation = example.observation
    cards, hero, table = observation.get("cards"), observation.get("hero"), observation.get("table")
    if not isinstance(cards, Mapping) or not isinstance(hero, Mapping) or not isinstance(table, Mapping):
        raise ValueError("pretraining example has malformed canonical observation")
    street, position, active = cards.get("street"), hero.get("position"), table.get("active_player_count")
    if not isinstance(street, str) or not isinstance(position, str) or isinstance(active, bool) or not isinstance(active, int):
        raise ValueError("pretraining example lacks safe stratum fields")
    expected_share = example.expected_showdown_share_target
    if expected_share is None:
        raise ValueError("pretraining example has no expected showdown-share target")
    return PretrainingStratum(street, active, position, equity_bucket(expected_share), example.selected_action)


def generate_pretraining_corpus(
    stage: CurriculumStage | str = DEFAULT_PRETRAINING_STAGE,
    *,
    train_seed_range: SeedRange,
    holdout_seed_range: SeedRange,
    bot_mix: Sequence[str] = BASELINE_BOT_NAMES,
    train_equity_samples: int = 16,
    holdout_equity_samples: int = 128,
    equity_samples: int | None = None,
) -> PretrainingCorpus:
    """Generate a deterministic baseline-bot corpus with an honest holdout.

    This intentionally delegates all hand execution, trace construction and
    player-safe projection to :func:`generate_pretraining_dataset`.
    """

    resolved = CurriculumStage(stage)
    if equity_samples is not None:
        if equity_samples < 1:
            raise ValueError("equity_samples must be positive")
        train_equity_samples = equity_samples
        holdout_equity_samples = equity_samples
    if train_seed_range.overlaps(holdout_seed_range):
        raise ValueError("train and holdout seed ranges must be disjoint")
    mixture = BaselineBotMixture(bot_mix)
    train_dataset = generate_pretraining_dataset(
        resolved, train_seed_range.count, mixture.select_action, seed_start=train_seed_range.start, equity_samples=train_equity_samples
    )
    holdout_dataset = generate_pretraining_dataset(
        resolved, holdout_seed_range.count, mixture.select_action, seed_start=holdout_seed_range.start, equity_samples=holdout_equity_samples
    )
    return PretrainingCorpus(
        stage=resolved,
        train_seed_range=train_seed_range,
        holdout_seed_range=holdout_seed_range,
        bot_mix=mixture.bot_mix,
        train_equity_samples=train_equity_samples,
        holdout_equity_samples=holdout_equity_samples,
        train=_records_with_policy(train_dataset, mixture),
        holdout=_records_with_policy(holdout_dataset, mixture),
    )


def strata_summary(records: Iterable[CorpusRecord]) -> dict[str, int]:
    """Return sorted stratum counts suitable for experiment logs and manifests."""

    counts = Counter(record.stratum.key() for record in records)
    return dict(sorted(counts.items()))


def _records_with_policy(dataset: TracePretrainingDataset, mixture: BaselineBotMixture) -> tuple[CorpusRecord, ...]:
    records: list[CorpusRecord] = []
    for example in dataset:
        policy = mixture.policy_id_for(example.observation)
        audited = replace(example, behavior_policy=policy)
        records.append(CorpusRecord(audited, stratum_for(audited), policy))
    return tuple(records)


def balanced_indices(records: Sequence[CorpusRecord], *, count: int | None = None, seed: int = 0) -> tuple[int, ...]:
    """Round-robin deterministic indices across strata.

    With the default count every original example appears exactly once, but
    rare strata are interleaved early rather than being buried under common
    preflop/check records.  With an explicit count each stratum cycles
    independently, producing an actually balanced oversampled stream.
    """

    if not records:
        return ()
    requested = len(records) if count is None else count
    if requested < 0:
        raise ValueError("count must be non-negative")
    grouped: dict[PretrainingStratum, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.stratum].append(index)
    rng = Random(seed)
    group_keys = sorted(grouped)
    for key in group_keys:
        rng.shuffle(grouped[key])
    rng.shuffle(group_keys)
    offsets = {key: 0 for key in group_keys}
    result: list[int] = []
    if count is not None:
        while len(result) < requested:
            for key in group_keys:
                values = grouped[key]
                result.append(values[offsets[key] % len(values)])
                offsets[key] += 1
                if len(result) == requested:
                    break
        return tuple(result)
    while len(result) < requested:
        emitted = False
        for key in group_keys:
            values = grouped[key]
            if offsets[key] == len(values):
                continue
            result.append(values[offsets[key]])
            offsets[key] += 1
            emitted = True
            if len(result) == requested:
                break
        if not emitted:
            # Begin a new deterministic epoch only after every original row
            # has appeared once.  This keeps a default pass lossless.
            offsets = {key: 0 for key in group_keys}
    return tuple(result)


def write_pretraining_corpus(corpus: PretrainingCorpus, path: str | Path) -> None:
    """Atomically write restricted JSONL with safe model inputs and audit seeds."""

    _validate_split_records(
        corpus.train,
        "train",
        corpus.train_seed_range,
        corpus.train_equity_samples,
        corpus.stage,
        corpus.bot_mix,
    )
    _validate_split_records(
        corpus.holdout,
        "holdout",
        corpus.holdout_seed_range,
        corpus.holdout_equity_samples,
        corpus.stage,
        corpus.bot_mix,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_metadata_record(corpus), sort_keys=True, separators=(",", ":"))]
    for split, records in (("train", corpus.train), ("holdout", corpus.holdout)):
        lines.extend(json.dumps(_example_record(split, record), sort_keys=True, separators=(",", ":")) for record in records)
    _atomic_write_text(destination, "\n".join(lines) + "\n")


def load_pretraining_corpus(path: str | Path) -> PretrainingCorpus:
    """Load and validate a corpus written by :func:`write_pretraining_corpus`."""

    source = Path(path)
    try:
        raw_lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read pretraining corpus: {source}") from error
    if not raw_lines:
        raise ValueError("pretraining corpus is empty")
    try:
        metadata = json.loads(raw_lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("pretraining corpus metadata is invalid JSON") from error
    if not isinstance(metadata, Mapping):
        raise ValueError("pretraining corpus metadata must be an object")
    stage, train_range, holdout_range, bot_mix, train_samples, holdout_samples = _parse_metadata(metadata)
    records: dict[str, list[CorpusRecord]] = {"train": [], "holdout": []}
    for line_number, line in enumerate(raw_lines[1:], start=2):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"pretraining corpus line {line_number} is invalid JSON") from error
        if not isinstance(item, Mapping):
            raise ValueError(f"pretraining corpus line {line_number} must be an object")
        split, record = _parse_example_record(item, stage)
        records[split].append(record)
    return PretrainingCorpus(
        stage, train_range, holdout_range, bot_mix, train_samples, holdout_samples, tuple(records["train"]), tuple(records["holdout"])
    )


def _metadata_record(corpus: PretrainingCorpus) -> dict[str, Any]:
    return {
        "record_type": "metadata",
        "schema_version": PRETRAINING_CORPUS_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_VERSION,
        "stage": corpus.stage.value,
        "generator": "baseline-bot-mixture-v1",
        "bot_mix": list(corpus.bot_mix),
        "train_equity_samples": corpus.train_equity_samples,
        "holdout_equity_samples": corpus.holdout_equity_samples,
        "equity_label_protocol": EQUITY_LABEL_PROTOCOL,
        "splits": {"train": corpus.train_seed_range.as_dict(), "holdout": corpus.holdout_seed_range.as_dict()},
        "summary": corpus.summary(),
    }


def _example_record(split: str, record: CorpusRecord) -> dict[str, Any]:
    _validate_player_safe_example(record.example)
    details = _safe_row_details(record.example)
    return {
        "record_type": "example",
        "schema_version": PRETRAINING_CORPUS_SCHEMA_VERSION,
        "split": split,
        "stage": record.example.stage.value,
        "hand_id": record.example.hand_id,
        "hand_seed": record.example.seed,
        "behavior_policy": record.behavior_policy,
        "street": details["street"],
        "table_player_count": details["table_player_count"],
        "active_opponent_count": details["active_opponent_count"],
        "observation": record.example.observation,
        "selected_action": record.example.selected_action,
        "equity_target": list(record.example.equity_target),
        "expected_showdown_share_target": record.example.expected_showdown_share_target,
        "terminal_pnl_bb": record.example.terminal_pnl_bb,
        "equity_samples": record.example.equity_samples,
        "equity_exact": record.example.equity_exact,
        "label_protocol": record.example.label_protocol,
        "stratum": record.stratum.as_dict(),
    }


def _parse_metadata(data: Mapping[str, Any]) -> tuple[CurriculumStage, SeedRange, SeedRange, tuple[str, ...], int, int]:
    schema_version = data.get("schema_version")
    if schema_version == "1.0":
        raise ValueError("pretraining corpus schema v1 is unsupported: regenerate labels with expected showdown share")
    if data.get("record_type") != "metadata" or schema_version != PRETRAINING_CORPUS_SCHEMA_VERSION:
        raise ValueError("incompatible pretraining corpus metadata")
    if data.get("observation_schema_version") != OBSERVATION_VERSION:
        raise ValueError("incompatible observation schema")
    if data.get("equity_label_protocol") == "fixed_deal_virtual_showdown_v1":
        raise ValueError("pretraining corpus label protocol v1 is unsupported: expected showdown-share labels are required")
    if data.get("equity_label_protocol") != EQUITY_LABEL_PROTOCOL:
        raise ValueError("incompatible equity label protocol")
    try:
        stage = CurriculumStage(data.get("stage"))
    except ValueError as error:
        raise ValueError("invalid corpus stage") from error
    splits, names = data.get("splits"), data.get("bot_mix")
    train_samples, holdout_samples = data.get("train_equity_samples"), data.get("holdout_equity_samples")
    if not isinstance(splits, Mapping) or not isinstance(splits.get("train"), Mapping) or not isinstance(splits.get("holdout"), Mapping):
        raise ValueError("corpus metadata lacks split seed ranges")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names) or not names:
        raise ValueError("corpus metadata has invalid bot mix")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (train_samples, holdout_samples)):
        raise ValueError("corpus metadata has invalid split equity sample counts")
    return stage, SeedRange.from_dict(splits["train"]), SeedRange.from_dict(splits["holdout"]), tuple(names), train_samples, holdout_samples


def _parse_example_record(data: Mapping[str, Any], stage: CurriculumStage) -> tuple[str, CorpusRecord]:
    if data.get("record_type") != "example" or data.get("schema_version") != PRETRAINING_CORPUS_SCHEMA_VERSION:
        raise ValueError("incompatible pretraining example record")
    split, encoded_stage, observation = data.get("split"), data.get("stage"), data.get("observation")
    action, target, pnl, encoded_stratum = data.get("selected_action"), data.get("equity_target"), data.get("terminal_pnl_bb"), data.get("stratum")
    expected_share = data.get("expected_showdown_share_target")
    hand_id, seed, policy = data.get("hand_id"), data.get("hand_seed"), data.get("behavior_policy")
    samples, exact, protocol = data.get("equity_samples"), data.get("equity_exact"), data.get("label_protocol")
    if split not in {"train", "holdout"} or encoded_stage != stage.value or not isinstance(observation, Mapping) or not isinstance(action, str):
        raise ValueError("malformed pretraining example record")
    if not isinstance(target, list) or len(target) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in target):
        raise ValueError("pretraining example has invalid equity target")
    if any(not isfinite(float(value)) or float(value) < 0.0 for value in target) or abs(sum(float(value) for value in target) - 1.0) > 1e-8:
        raise ValueError("pretraining example has invalid equity target")
    if isinstance(expected_share, bool) or not isinstance(expected_share, (int, float)) or not isfinite(float(expected_share)) or not 0.0 <= float(expected_share) <= 1.0:
        raise ValueError("pretraining example has invalid expected showdown-share target")
    if isinstance(pnl, bool) or not isinstance(pnl, (int, float)) or not isinstance(encoded_stratum, Mapping):
        raise ValueError("pretraining example has invalid labels")
    if isinstance(hand_id, bool) or not isinstance(hand_id, int) or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("pretraining example lacks audit hand identifiers")
    if not isinstance(policy, str) or not policy.startswith("baseline:") or isinstance(samples, bool) or not isinstance(samples, int) or samples < 1 or not isinstance(exact, bool) or protocol != EQUITY_LABEL_PROTOCOL:
        raise ValueError("pretraining example has invalid label provenance")
    example = PretrainingExample(
        dict(observation), action, tuple(float(value) for value in target), float(pnl), stage,
        hand_id, seed, samples, exact, protocol, policy, float(expected_share),
    )
    _validate_player_safe_example(example)
    stratum = PretrainingStratum.from_dict(encoded_stratum)
    if stratum != stratum_for(example):
        raise ValueError("pretraining example stratum does not match its safe fields")
    details = _safe_row_details(example)
    if data.get("street") != details["street"] or data.get("table_player_count") != details["table_player_count"] or data.get("active_opponent_count") != details["active_opponent_count"]:
        raise ValueError("pretraining example safe context does not match observation")
    return split, CorpusRecord(example, stratum, policy)


def _safe_row_details(example: PretrainingExample) -> dict[str, int | str]:
    """Auditable context derived from the safe observation, never a snapshot."""

    observation = example.observation
    cards, players, table = observation.get("cards"), observation.get("player_set"), observation.get("table")
    if not isinstance(cards, Mapping) or not isinstance(players, list) or not isinstance(table, Mapping):
        raise ValueError("example lacks safe context fields")
    street, active = cards.get("street"), table.get("active_player_count")
    if not isinstance(street, str) or isinstance(active, bool) or not isinstance(active, int):
        raise ValueError("example has malformed safe context")
    return {"street": street, "table_player_count": len(players), "active_opponent_count": max(0, active - 1)}


def _validate_player_safe_example(example: PretrainingExample) -> None:
    """Reject common accidental leaks before a corpus is published or consumed."""

    observation = example.observation
    if observation.get("schema_version") != OBSERVATION_VERSION:
        raise ValueError("example has incompatible observation schema")
    if "seed" in observation or "deck" in observation or "remaining_deck" in observation:
        raise ValueError("example observation contains private deal information")
    cards = observation.get("cards")
    players = observation.get("players")
    if not isinstance(cards, Mapping) or not isinstance(cards.get("hole_cards"), list) or not isinstance(players, list):
        raise ValueError("example lacks canonical player-safe observation fields")
    for player in players:
        if not isinstance(player, Mapping) or any(key in player for key in ("hole_cards", "cards", "deck", "remaining_deck")):
            raise ValueError("example observation contains opponent private cards")
    player_set = observation.get("player_set")
    if not isinstance(player_set, list) or any(not isinstance(player, Mapping) or "hole_cards" in player for player in player_set):
        raise ValueError("example observation contains player-set private cards")


def _atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent.
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - network filesystems can reject it.
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "BASELINE_BOT_NAMES",
    "DEFAULT_PRETRAINING_STAGE",
    "DEFAULT_EQUITY_BUCKETS",
    "PRETRAINING_CORPUS_SCHEMA_VERSION",
    "BaselineBotMixture",
    "CorpusRecord",
    "PretrainingCorpus",
    "PretrainingStratum",
    "SeedRange",
    "balanced_indices",
    "equity_bucket",
    "generate_pretraining_corpus",
    "load_pretraining_corpus",
    "strata_summary",
    "stratum_for",
    "write_pretraining_corpus",
]
