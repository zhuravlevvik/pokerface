"""Versioned, player-safe observations shared by training and inference.

``HandState`` deliberately contains complete information, including every
player's hole cards and the undealt portion of the deck.  This module is the
only supported projection from that state to a policy input.  It must therefore
remain free of opponent cards, future board cards and any other oracle data.

The canonical, model-ready fields are ``cards``, ``hero``, ``table``,
``player_set``, ``player_mask``, ``action_history`` and
``legal_action_mask``.  A small set of flat aliases is retained while the
pre-RL baseline bots are migrated; new model code should use canonical fields.
All canonical money features are expressed in big blinds and, when useful, as
a ratio of the current pot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf
from typing import Iterable, Literal, Mapping, TypedDict

from .betting import Action
from .game_state import ActionRecord, HandState, Street
from .rules import BIG_BLIND


OBSERVATION_VERSION = "1.0"
"""Version of the serialisable policy-input contract."""

Position = Literal["BTN", "SB", "BB", "UTG", "CO"]
ActionName = Literal[
    "fold",
    "check",
    "call",
    "raise_min",
    "raise_1_3_pot",
    "raise_1_2_pot",
    "raise_3_4_pot",
    "raise_pot",
    "raise_1_5_pot",
    "all_in",
]


class CardFeatures(TypedDict):
    """Cards known to the hero at the current decision point."""

    hole_cards: list[str]
    board: list[str]
    street: str
    street_index: int


class HeroFeatures(TypedDict):
    """Hero-specific features, normalised for the policy network."""

    seat: int
    position: Position
    stack_bb: float
    stack_to_pot: float
    committed_street_bb: float
    committed_street_to_pot: float
    committed_total_bb: float
    committed_total_to_pot: float
    to_call_bb: float
    to_call_to_pot: float
    min_raise_to_bb: float
    min_raise_to_pot: float


class TableFeatures(TypedDict):
    """Public table-wide features at the current point in the hand."""

    pot_bb: float
    current_bet_bb: float
    last_full_raise_bb: float
    active_player_count: int
    actionable_player_count: int


class PlayerFeatures(TypedDict):
    """One element of the unordered player feature set.

    ``seat`` is an engine/audit identifier retained for compatibility.  A
    model should use ``position`` and the set mask instead; it must not infer a
    fixed player slot from the sequence index.
    """

    seat: int
    position: Position
    is_hero: bool
    stack_bb: float
    stack_to_pot: float
    committed_street_bb: float
    committed_street_to_pot: float
    committed_total_bb: float
    committed_total_to_pot: float
    folded: bool
    all_in: bool
    active: bool
    last_action: ActionName | None
    last_action_amount_bb: float
    last_action_amount_to_pot: float


class ActionHistoryFeatures(TypedDict):
    """One public action, with its amount measured before that action."""

    street: str
    street_index: int
    position: Position
    action: ActionName
    amount_bb: float
    amount_to_pot: float
    raise_to_bb: float | None
    raise_to_to_pot: float | None
    current_bet_after_bb: float


class Observation(TypedDict):
    """JSON-serialisable policy observation, stable at ``schema_version`` 1.0."""

    schema_version: str
    cards: CardFeatures
    hero: HeroFeatures
    table: TableFeatures
    player_set: list[PlayerFeatures]
    player_mask: list[bool]
    action_history: list[ActionHistoryFeatures]
    legal_action_mask: dict[str, bool]

    # Compatibility aliases used by the existing diagnostic bots and traces.
    seat: int
    street: str
    hole_cards: list[str]
    board: list[str]
    pot: int
    current_bet: int
    to_call: int
    actor: int | None
    players: list[dict[str, object]]
    legal_actions: dict[str, bool]


_STREET_INDEX = {
    Street.PREFLOP: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
    Street.SHOWDOWN: 4,
    Street.COMPLETE: 5,
}


def _bb(chips: int) -> float:
    return chips / BIG_BLIND


def _pot_ratio(chips: int, pot: int) -> float:
    """Return a finite amount/pot feature even before any voluntary action."""

    return chips / max(pot, BIG_BLIND)


def _action_mask(state: HandState, seat: int) -> dict[str, bool]:
    return {action.value: allowed for action, allowed in state.legal_actions(seat).items()}


def _history_features(state: HandState) -> tuple[list[ActionHistoryFeatures], dict[int, ActionHistoryFeatures]]:
    """Encode public records and return the latest action for each seat.

    The state records the public preceding pot at action time.  Keeping it on
    the immutable action record avoids a retrospective reconstruction error
    when an uncalled wager is returned at the end of an earlier street.
    """

    history: list[ActionHistoryFeatures] = []
    last_by_seat: dict[int, ActionHistoryFeatures] = {}
    for record in state.action_history:
        item = _record_features(record, state, record.pot_before)
        history.append(item)
        last_by_seat[record.seat] = item
    return history, last_by_seat


def _record_features(record: ActionRecord, state: HandState, pot_before: int) -> ActionHistoryFeatures:
    raise_to = record.raise_to
    return {
        "street": record.street.value,
        "street_index": _STREET_INDEX[record.street],
        "position": state.positions[record.seat],  # type: ignore[typeddict-item]
        "action": record.action.value,  # type: ignore[typeddict-item]
        "amount_bb": _bb(record.amount),
        "amount_to_pot": _pot_ratio(record.amount, pot_before),
        "raise_to_bb": None if raise_to is None else _bb(raise_to),
        "raise_to_to_pot": None if raise_to is None else _pot_ratio(raise_to, pot_before),
        "current_bet_after_bb": _bb(record.current_bet_after),
    }


def observation_for(state: HandState, seat: int) -> Observation:
    """Return the legal, versioned observation for ``seat``.

    The caller can request a view for any existing seat (not just the actor),
    which is useful for evaluation and replay.  Only the current actor receives
    enabled actions; non-actors get an all-false action mask from the engine.
    """

    if not 0 <= seat < len(state.players):
        raise ValueError(f"seat must be in 0..{len(state.players) - 1}")
    hero = state.player(seat)
    pot = state.pot
    history, last_by_seat = _history_features(state)
    hero_to_call = state.to_call(seat)
    min_raise = state.current_bet + state.last_full_raise
    cards: CardFeatures = {
        "hole_cards": [str(card) for card in state.hole_cards[seat]],
        "board": [str(card) for card in state.board],
        "street": state.street.value,
        "street_index": _STREET_INDEX[state.street],
    }
    hero_features: HeroFeatures = {
        "seat": seat,
        "position": state.positions[seat],  # type: ignore[typeddict-item]
        "stack_bb": _bb(hero.stack),
        "stack_to_pot": _pot_ratio(hero.stack, pot),
        "committed_street_bb": _bb(hero.committed_street),
        "committed_street_to_pot": _pot_ratio(hero.committed_street, pot),
        "committed_total_bb": _bb(hero.committed_total),
        "committed_total_to_pot": _pot_ratio(hero.committed_total, pot),
        "to_call_bb": _bb(hero_to_call),
        "to_call_to_pot": _pot_ratio(hero_to_call, pot),
        "min_raise_to_bb": _bb(min_raise),
        "min_raise_to_pot": _pot_ratio(min_raise, pot),
    }
    player_set: list[PlayerFeatures] = []
    legacy_players: list[dict[str, object]] = []
    for other in state.players:
        last_action = last_by_seat.get(other.seat)
        player_set.append(
            {
                "seat": other.seat,
                "position": state.positions[other.seat],  # type: ignore[typeddict-item]
                "is_hero": other.seat == seat,
                "stack_bb": _bb(other.stack),
                "stack_to_pot": _pot_ratio(other.stack, pot),
                "committed_street_bb": _bb(other.committed_street),
                "committed_street_to_pot": _pot_ratio(other.committed_street, pot),
                "committed_total_bb": _bb(other.committed_total),
                "committed_total_to_pot": _pot_ratio(other.committed_total, pot),
                "folded": other.folded,
                "all_in": other.all_in,
                "active": not other.folded,
                "last_action": None if last_action is None else last_action["action"],
                "last_action_amount_bb": 0.0 if last_action is None else last_action["amount_bb"],
                "last_action_amount_to_pot": 0.0 if last_action is None else last_action["amount_to_pot"],
            }
        )
        # Do not add hole cards here: this legacy view is deliberately safe.
        legacy_players.append(
            {
                "seat": other.seat,
                "position": state.positions[other.seat],
                "stack": other.stack,
                "committed_street": other.committed_street,
                "committed_total": other.committed_total,
                "folded": other.folded,
                "all_in": other.all_in,
            }
        )
    mask = _action_mask(state, seat)
    table: TableFeatures = {
        "pot_bb": _bb(pot),
        "current_bet_bb": _bb(state.current_bet),
        "last_full_raise_bb": _bb(state.last_full_raise),
        "active_player_count": sum(not player.folded for player in state.players),
        "actionable_player_count": sum(not player.folded and not player.all_in for player in state.players),
    }
    return {
        "schema_version": OBSERVATION_VERSION,
        "cards": cards,
        "hero": hero_features,
        "table": table,
        "player_set": player_set,
        # A future short-handed table represents empty seats with False.  The
        # current 5-max engine occupies all five places, even after a fold.
        "player_mask": [True] * len(player_set),
        "action_history": history,
        "legal_action_mask": mask,
        # Compatibility aliases.  These are raw chips for existing bots only.
        "seat": seat,
        "street": state.street.value,
        "hole_cards": list(cards["hole_cards"]),
        "board": list(cards["board"]),
        "pot": pot,
        "current_bet": state.current_bet,
        "to_call": hero_to_call,
        "actor": state.actor,
        "players": legacy_players,
        "legal_actions": dict(mask),
    }


@dataclass
class _ScalarSummary:
    """Online mean/range accumulator used to inspect feature normalisation."""

    count: int = 0
    mean: float = 0.0
    minimum: float = inf
    maximum: float = -inf

    def add(self, value: float) -> None:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "min": self.minimum, "max": self.maximum}


@dataclass
class ObservationFeatureStatistics:
    """Aggregate normalised feature ranges without retaining private state.

    Training code may call :meth:`update` for each emitted observation, then
    call :meth:`log` at an epoch boundary.  The aggregate contains only scalar
    network features; no card identities, per-hand data or hidden cards are
    logged.
    """

    _features: dict[str, _ScalarSummary] = field(default_factory=dict)

    def update(self, observation: Mapping[str, object]) -> None:
        if observation.get("schema_version") != OBSERVATION_VERSION:
            raise ValueError("observation schema version is unsupported")
        hero = observation["hero"]
        table = observation["table"]
        player_set = observation["player_set"]
        if not isinstance(hero, Mapping) or not isinstance(table, Mapping) or not isinstance(player_set, list):
            raise ValueError("malformed canonical observation")
        for name in (
            "stack_bb",
            "stack_to_pot",
            "committed_street_bb",
            "committed_street_to_pot",
            "committed_total_bb",
            "committed_total_to_pot",
            "to_call_bb",
            "to_call_to_pot",
            "min_raise_to_bb",
            "min_raise_to_pot",
        ):
            self._add(f"hero.{name}", hero[name])
        for name in ("pot_bb", "current_bet_bb", "last_full_raise_bb", "active_player_count", "actionable_player_count"):
            self._add(f"table.{name}", table[name])
        for player in player_set:
            if not isinstance(player, Mapping):
                raise ValueError("player_set must contain mappings")
            for name in (
                "stack_bb",
                "stack_to_pot",
                "committed_street_bb",
                "committed_street_to_pot",
                "committed_total_bb",
                "committed_total_to_pot",
                "last_action_amount_bb",
                "last_action_amount_to_pot",
            ):
                self._add(f"player_set.{name}", player[name])

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        """Return a JSON-serialisable aggregate suitable for experiment logs."""

        return {name: summary.as_dict() for name, summary in sorted(self._features.items())}

    def log(self, logger: logging.Logger | None = None) -> dict[str, dict[str, float | int]]:
        """Log and return aggregate feature statistics at an explicit boundary."""

        result = self.snapshot()
        (logger or logging.getLogger(__name__)).info("observation feature statistics: %s", result)
        return result

    def _add(self, name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        self._features.setdefault(name, _ScalarSummary()).add(float(value))


def collect_feature_statistics(observations: Iterable[Mapping[str, object]]) -> ObservationFeatureStatistics:
    """Convenience helper for a batch of train or inference observations."""

    statistics = ObservationFeatureStatistics()
    for observation in observations:
        statistics.update(observation)
    return statistics
