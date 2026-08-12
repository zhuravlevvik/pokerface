"""Pure Python seven-card Hold'em evaluator with total ordering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .cards import Card

HAND_NAMES = (
    "high_card",
    "one_pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
)


@dataclass(frozen=True, order=True)
class HandRank:
    """Comparable rank, ordered from weakest to strongest."""

    category: int
    tiebreak: tuple[int, ...]

    @property
    def name(self) -> str:
        return HAND_NAMES[self.category]


def _straight_high(ranks: Iterable[int]) -> int | None:
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 4, -1):
        if all(rank in unique for rank in range(high - 4, high + 1)):
            return high
    return None


def evaluate_five(cards: Iterable[Card]) -> HandRank:
    cards = tuple(cards)
    if len(cards) != 5 or len(set(cards)) != 5:
        raise ValueError("evaluate_five requires exactly five distinct cards")
    ranks = [card.rank for card in cards]
    counts = Counter(ranks)
    count_groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({card.suit for card in cards}) == 1
    straight_high = _straight_high(ranks)
    if flush and straight_high is not None:
        return HandRank(8, (straight_high,))
    if count_groups[0][0] == 4:
        quad = count_groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad)
        return HandRank(7, (quad, kicker))
    if [group[0] for group in count_groups] == [3, 2]:
        return HandRank(6, (count_groups[0][1], count_groups[1][1]))
    if flush:
        return HandRank(5, tuple(sorted(ranks, reverse=True)))
    if straight_high is not None:
        return HandRank(4, (straight_high,))
    if count_groups[0][0] == 3:
        trip = count_groups[0][1]
        kickers = tuple(sorted((rank for rank in ranks if rank != trip), reverse=True))
        return HandRank(3, (trip, *kickers))
    if [group[0] for group in count_groups] == [2, 2, 1]:
        pairs = sorted((rank for count, rank in count_groups if count == 2), reverse=True)
        kicker = next(rank for count, rank in count_groups if count == 1)
        return HandRank(2, (*pairs, kicker))
    if count_groups[0][0] == 2:
        pair = count_groups[0][1]
        kickers = tuple(sorted((rank for rank in ranks if rank != pair), reverse=True))
        return HandRank(1, (pair, *kickers))
    return HandRank(0, tuple(sorted(ranks, reverse=True)))


def evaluate(cards: Iterable[Card]) -> HandRank:
    """Return the best five-card rank from five to seven distinct cards."""

    cards = tuple(cards)
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError("evaluate requires five to seven distinct cards")
    return max(evaluate_five(combo) for combo in combinations(cards, 5))
