"""Small, inspectable baseline poker policies.

The bots in this module are deliberately not learning agents.  They consume
only the player-safe observation produced by :mod:`poker.observation` and
always return an action allowed by the supplied action mask.  They are useful
for engine tests, data collection and fixed evaluation opponents.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Mapping, Protocol

from .betting import Action
from .cards import Card
from .evaluator import evaluate


ActionMask = Mapping[str, bool]


class PokerBot(Protocol):
    """Minimal policy interface shared by all non-learning opponents."""

    def select_action(self, observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> Action:
        """Choose exactly one currently legal discrete action."""


def _mask(observation: Mapping[str, object], legal_actions: ActionMask | None) -> ActionMask:
    if legal_actions is not None:
        return legal_actions
    candidate = observation.get("legal_actions")
    if not isinstance(candidate, Mapping):
        raise ValueError("observation must contain a legal_actions mask")
    return candidate  # type: ignore[return-value]


def legal_choices(observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> tuple[Action, ...]:
    """Return legal actions in a stable order, or fail on a malformed state."""

    mask = _mask(observation, legal_actions)
    choices = tuple(action for action in Action if bool(mask.get(action.value, False)))
    if not choices:
        raise ValueError("bot received an observation with no legal actions")
    return choices


def _first_available(choices: tuple[Action, ...], *preferences: Action) -> Action:
    for action in preferences:
        if action in choices:
            return action
    return choices[0]


def _raise_choice(choices: tuple[Action, ...], *, large: bool = False) -> Action | None:
    ordered = (
        (Action.RAISE_POT, Action.RAISE_3_4_POT, Action.RAISE_1_2_POT, Action.RAISE_MIN, Action.ALL_IN)
        if large
        else (Action.RAISE_1_2_POT, Action.RAISE_1_3_POT, Action.RAISE_MIN, Action.RAISE_3_4_POT, Action.ALL_IN)
    )
    return next((action for action in ordered if action in choices), None)


def _hole_cards(observation: Mapping[str, object]) -> tuple[Card, Card]:
    tokens = observation.get("hole_cards")
    if not isinstance(tokens, list) or len(tokens) != 2 or not all(isinstance(token, str) for token in tokens):
        raise ValueError("observation must contain exactly two hole-card strings")
    return Card.parse(tokens[0]), Card.parse(tokens[1])


def preflop_strength(observation: Mapping[str, object]) -> float:
    """Cheap 0..1 starting-hand score used by the non-learning policies.

    It intentionally is a coarse heuristic, not an equity calculator.  Pairs,
    high cards, suitedness and connectedness are rewarded; weak offsuit hands
    remain near zero.
    """

    first, second = _hole_cards(observation)
    high, low = sorted((first.rank, second.rank), reverse=True)
    if high == low:
        return min(1.0, 0.48 + (high - 2) * 0.043)
    score = (high - 2) / 12 * 0.42 + (low - 2) / 12 * 0.18
    if first.suit == second.suit:
        score += 0.10
    gap = high - low
    if gap == 1:
        score += 0.11
    elif gap == 2:
        score += 0.05
    if high >= 13:
        score += 0.12
    elif high >= 11:
        score += 0.05
    return min(1.0, score)


def hand_strength(observation: Mapping[str, object]) -> float:
    """Return a simple showdown-strength proxy from public and own cards."""

    board_tokens = observation.get("board")
    if not isinstance(board_tokens, list) or not all(isinstance(token, str) for token in board_tokens):
        raise ValueError("observation board must be a list of card strings")
    if not board_tokens:
        return preflop_strength(observation)
    cards = (*_hole_cards(observation), *(Card.parse(token) for token in board_tokens))
    rank = evaluate(cards)
    # The category is the most important signal; kickers distinguish hands in
    # the same category without pretending to know opponents' hidden cards.
    category = rank.category / 8
    kicker = sum(value / 14 ** (index + 1) for index, value in enumerate(rank.tiebreak)) * 0.12
    return min(1.0, 0.12 + category * 0.80 + kicker)


class RandomBot:
    """Uniformly samples one legal action using an instance-local RNG."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = Random(seed)

    def select_action(self, observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> Action:
        choices = legal_choices(observation, legal_actions)
        return self._rng.choice(choices)


class TightBot:
    """Conservative policy: narrow entry range and few speculative bluffs."""

    def select_action(self, observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> Action:
        choices = legal_choices(observation, legal_actions)
        score = hand_strength(observation)
        to_call = int(observation["to_call"])
        street = str(observation["street"])
        if street == "preflop":
            if score >= 0.73:
                raise_action = _raise_choice(choices)
                return raise_action or _first_available(choices, Action.CALL, Action.CHECK)
            if score >= 0.53:
                return _first_available(choices, Action.CALL, Action.CHECK, Action.FOLD)
            return _first_available(choices, Action.FOLD, Action.CHECK, Action.CALL)
        if score >= 0.56:
            raise_action = _raise_choice(choices)
            if score >= 0.76 and raise_action is not None:
                return raise_action
            return _first_available(choices, Action.CALL, Action.CHECK, Action.FOLD)
        if to_call == 0:
            return _first_available(choices, Action.CHECK, Action.FOLD)
        return _first_available(choices, Action.FOLD, Action.CALL)


class AggroBot:
    """Pressure-oriented policy that prefers betting and raising when possible."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = Random(seed)

    def select_action(self, observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> Action:
        choices = legal_choices(observation, legal_actions)
        score = hand_strength(observation)
        raise_action = _raise_choice(choices, large=score >= 0.55)
        # A small amount of seeded randomness prevents this bot from becoming
        # an identical action script while keeping experiments reproducible.
        if raise_action is not None and (score >= 0.34 or self._rng.random() < 0.62):
            return raise_action
        return _first_available(choices, Action.CALL, Action.CHECK, Action.FOLD)


class CallingStationBot:
    """Loose passive opponent that almost always continues after entering."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = Random(seed)

    def select_action(self, observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> Action:
        choices = legal_choices(observation, legal_actions)
        score = hand_strength(observation)
        street = str(observation["street"])
        if street == "preflop" and score < 0.20 and Action.FOLD in choices:
            return Action.FOLD
        raise_action = _raise_choice(choices)
        if raise_action is not None and score >= 0.82 and self._rng.random() < 0.22:
            return raise_action
        return _first_available(choices, Action.CALL, Action.CHECK, Action.FOLD)


class RuleBot:
    """A deterministic, interpretable pot-odds and hand-strength baseline."""

    def select_action(self, observation: Mapping[str, object], legal_actions: ActionMask | None = None) -> Action:
        choices = legal_choices(observation, legal_actions)
        strength = hand_strength(observation)
        to_call = int(observation["to_call"])
        pot = max(1, int(observation["pot"]))
        call_fraction = to_call / (pot + to_call)
        street = str(observation["street"])
        position = next(
            str(player["position"])
            for player in observation["players"]  # type: ignore[index,union-attr]
            if player["seat"] == observation["seat"]  # type: ignore[index]
        )
        position_bonus = 0.05 if position in {"BTN", "CO"} else 0.0
        threshold = call_fraction + (0.07 if street == "preflop" else 0.03) - position_bonus
        raise_action = _raise_choice(choices, large=strength >= 0.74)
        if raise_action is not None and strength >= max(0.64, threshold + 0.20):
            return raise_action
        if to_call == 0:
            if raise_action is not None and strength >= 0.52:
                return raise_action
            return _first_available(choices, Action.CHECK, Action.CALL)
        if strength >= threshold:
            return _first_available(choices, Action.CALL, Action.CHECK)
        return _first_available(choices, Action.FOLD, Action.CALL)


@dataclass(frozen=True)
class BotStatistics:
    """Characteristic action rates calculated across a bot's dealt hands."""

    hands: int
    vpip: float
    pfr: float
    three_bet: float
    fold_to_raise: float
    aggression_factor: float
