"""Betting-domain types and side-pot construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE_MIN = "raise_min"
    RAISE_1_3_POT = "raise_1_3_pot"
    RAISE_1_2_POT = "raise_1_2_pot"
    RAISE_3_4_POT = "raise_3_4_pot"
    RAISE_POT = "raise_pot"
    RAISE_1_5_POT = "raise_1_5_pot"
    ALL_IN = "all_in"


RAISE_ACTIONS = frozenset(
    {
        Action.RAISE_MIN,
        Action.RAISE_1_3_POT,
        Action.RAISE_1_2_POT,
        Action.RAISE_3_4_POT,
        Action.RAISE_POT,
        Action.RAISE_1_5_POT,
    }
)


@dataclass
class PlayerState:
    seat: int
    stack: int
    committed_total: int = 0
    committed_street: int = 0
    folded: bool = False
    all_in: bool = False

    @property
    def active(self) -> bool:
        return not self.folded


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: tuple[int, ...]
    level: int


def build_pots(players: Iterable[PlayerState]) -> list[Pot]:
    """Build main and side pots from total contributions.

    Folded players count towards the size but are deliberately excluded from
    eligibility. An empty eligible set indicates an invalid terminal state.
    """

    players = tuple(players)
    levels = sorted({player.committed_total for player in players if player.committed_total > 0})
    previous = 0
    pots: list[Pot] = []
    for level in levels:
        contributors = [player for player in players if player.committed_total >= level]
        amount = (level - previous) * len(contributors)
        eligible = tuple(player.seat for player in contributors if not player.folded)
        if amount:
            pots.append(Pot(amount=amount, eligible=eligible, level=level))
        previous = level
    return pots
