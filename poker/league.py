"""Opponent league and durable checkpoint archive for self-play training.

The league deliberately has no optimiser dependency.  A training loop asks it
for a seat assignment for a hand, and the league supplies current, frozen and
baseline opponents in a position-rotated order.  Frozen policies are loaded
from checkpoints instead of sharing mutable parameters with the learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Callable, Mapping, Protocol

from .betting import Action
from .bots import AggroBot, PokerBot, RandomBot, RuleBot, TightBot
from .model import TORCH_AVAILABLE, PokerAgentModel


class LeaguePolicy(Protocol):
    """A non-learning action source used in one seat of a sampled hand."""

    name: str

    def select_action(self, observation: Mapping[str, object], legal_actions: Mapping[str, bool]) -> Action:
        """Select one legal engine action."""


@dataclass
class BotPolicy:
    """Adapter which gives an existing baseline bot a stable league name."""

    name: str
    bot: PokerBot

    def select_action(self, observation: Mapping[str, object], legal_actions: Mapping[str, bool]) -> Action:
        return self.bot.select_action(observation, legal_actions)


@dataclass
class ModelPolicy:
    """Frozen or current neural policy; sampling is supplied by the trainer.

    ``select_action`` is intentionally deterministic, which makes this class
    usable for an archive or an evaluator without coupling it to PPO's random
    sampling.  The trainer recognises the current member by its name and
    samples its factorised policy distribution itself.
    """

    name: str
    model: PokerAgentModel
    checkpoint_path: Path | None = None

    def select_action(self, observation: Mapping[str, object], legal_actions: Mapping[str, bool]) -> Action:
        return Action(self.model.infer(observation).action)

    @classmethod
    def from_checkpoint(cls, name: str, path: str | Path) -> "ModelPolicy":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load a model league member")
        model = PokerAgentModel.load_checkpoint(path)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(name=name, model=model, checkpoint_path=Path(path))


@dataclass(frozen=True)
class LeagueMember:
    """A weighted member of an opponent mixture."""

    policy: LeaguePolicy
    weight: float = 1.0
    kind: str = "baseline"

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("league member weight must be positive")
        if self.kind not in {"current", "historical", "best", "baseline", "counter"}:
            raise ValueError(f"unsupported league member kind: {self.kind!r}")


@dataclass
class OpponentLeague:
    """Weighted policy population with deterministic seat rotation.

    The current policy is always present at least once in a sampled hand.  The
    remaining seats are independent weighted samples from the opponent league.
    Keeping exactly one current-policy seat makes the position rotation exact
    and avoids accidentally turning a league hand back into pure self-play.
    """

    current_name: str
    members: list[LeagueMember] = field(default_factory=list)
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)
    _hand_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)
        if not self.members:
            raise ValueError("league needs at least one member")
        self._validate_members()

    @property
    def current(self) -> ModelPolicy:
        for member in self.members:
            if member.policy.name == self.current_name:
                if not isinstance(member.policy, ModelPolicy):
                    raise TypeError("current league member must be a ModelPolicy")
                return member.policy
        raise RuntimeError("league has no current policy")

    def _validate_members(self) -> None:
        names = [member.policy.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("league member names must be unique")
        if self.current_name not in names:
            raise ValueError("league must contain its current policy")
        if not isinstance(self.current, ModelPolicy):
            raise TypeError("current league member must be a ModelPolicy")

    def add(self, member: LeagueMember) -> None:
        if any(existing.policy.name == member.policy.name for existing in self.members):
            raise ValueError(f"league already contains {member.policy.name!r}")
        self.members.append(member)

    def sample_seating(self, player_count: int) -> tuple[LeaguePolicy, ...]:
        """Return policies indexed by seat, rotating the forced current seat.

        For each complete ``player_count`` consecutive hands the forced current
        policy visits every seat exactly once.  Other members are independently
        sampled directly into their final seats.
        """

        if player_count < 2:
            raise ValueError("player_count must be at least two")
        current = self.current
        sampled = [self._weighted_sample(exclude_name=self.current_name).policy for _ in range(player_count)]
        forced_seat = self._hand_index % player_count
        sampled[forced_seat] = current
        seating = tuple(sampled)
        if seating[forced_seat].name != self.current_name:
            raise RuntimeError("sampled table lost the current policy")
        self._hand_index += 1
        return seating

    def _weighted_sample(self, *, exclude_name: str | None = None) -> LeagueMember:
        candidates = [member for member in self.members if member.policy.name != exclude_name]
        if not candidates:
            # A one-member league is still a useful smoke-test configuration;
            # it intentionally degrades to regular shared-policy self-play.
            candidates = list(self.members)
        total = sum(member.weight for member in candidates)
        threshold = self._rng.random() * total
        running = 0.0
        for member in candidates:
            running += member.weight
            if threshold <= running:
                return member
        return candidates[-1]  # Floating point fall-back.


def default_league(current_model: PokerAgentModel, *, seed: int | None = None) -> OpponentLeague:
    """Create a useful initial mixture including contrasting counter-bots."""

    return OpponentLeague(
        current_name="current",
        seed=seed,
        members=[
            LeagueMember(ModelPolicy("current", current_model), weight=3.0, kind="current"),
            LeagueMember(BotPolicy("rule", RuleBot()), weight=1.0, kind="baseline"),
            LeagueMember(BotPolicy("random", RandomBot(seed=seed)), weight=0.35, kind="baseline"),
            # Tight and aggro policies deliberately expose exploitability at
            # opposite ends of the style spectrum.
            LeagueMember(BotPolicy("counter_tight", TightBot()), weight=0.75, kind="counter"),
            LeagueMember(BotPolicy("counter_aggro", AggroBot(seed=seed)), weight=0.75, kind="counter"),
        ],
    )


@dataclass(frozen=True)
class PromotionResult:
    accepted: bool
    score: float
    checkpoint_path: Path | None
    reason: str


class CheckpointArchive:
    """Save candidates and promote non-regressing models into the league.

    The supplied evaluator is intentionally a callback: stage 10 owns rich
    tournament and calibration evaluation.  This archive only implements the
    stage-09 gate -- compare a scalar fixed-suite score to the current champion
    and keep accepted immutable snapshots available as league opponents.
    """

    def __init__(self, directory: str | Path, *, champion_score: float | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.champion_score = champion_score
        self._sequence = 0

    def promote(
        self,
        model: PokerAgentModel,
        *,
        score: float,
        league: OpponentLeague | None = None,
        name_prefix: str = "historical",
    ) -> PromotionResult:
        if self.champion_score is not None and score < self.champion_score:
            return PromotionResult(False, score, None, "score regressed versus champion")
        path = self.directory / f"{name_prefix}_{self._sequence:06d}.pt"
        self._sequence += 1
        model.save_checkpoint(path)
        self.champion_score = score
        if league is not None:
            member_name = path.stem
            league.add(LeagueMember(ModelPolicy.from_checkpoint(member_name, path), weight=1.0, kind="best"))
        return PromotionResult(True, score, path, "promoted")


__all__ = [
    "BotPolicy",
    "CheckpointArchive",
    "LeagueMember",
    "LeaguePolicy",
    "ModelPolicy",
    "OpponentLeague",
    "PromotionResult",
    "default_league",
]
