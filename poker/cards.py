"""Card primitives and reproducible deck construction."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

RANKS = tuple(range(2, 15))
SUITS = ("c", "d", "h", "s")
RANK_SYMBOLS = {**{rank: str(rank) for rank in range(2, 10)}, 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}


@dataclass(frozen=True, order=True)
class Card:
    """An immutable standard playing card. Ranks use 2..14, where 14 is ace."""

    rank: int
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANKS or self.suit not in SUITS:
            raise ValueError(f"invalid card: {self.rank!r}{self.suit!r}")

    def __str__(self) -> str:
        return f"{RANK_SYMBOLS[self.rank]}{self.suit}"

    @classmethod
    def parse(cls, token: str) -> "Card":
        if len(token) != 2:
            raise ValueError(f"invalid card token {token!r}")
        rank_token, suit = token[0].upper(), token[1].lower()
        ranks = {symbol: rank for rank, symbol in RANK_SYMBOLS.items()}
        if rank_token not in ranks:
            raise ValueError(f"invalid card token {token!r}")
        return cls(ranks[rank_token], suit)


class Deck:
    """A shuffled 52-card deck whose order is fully determined by ``seed``."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        Random(seed).shuffle(self._cards)

    @property
    def remaining(self) -> int:
        return len(self._cards)

    def deal(self, count: int = 1) -> list[Card]:
        if count < 0 or count > len(self._cards):
            raise ValueError("cannot deal requested number of cards")
        result = self._cards[:count]
        del self._cards[:count]
        return result

    def snapshot(self) -> tuple[Card, ...]:
        return tuple(self._cards)
