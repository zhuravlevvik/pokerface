"""Reproducible baseline-bot tournament and behaviour statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .betting import Action, RAISE_ACTIONS
from .bots import BotStatistics, PokerBot
from .game_state import HandState
from .observation import observation_for
from .rules import BIG_BLIND, SEAT_COUNT


@dataclass(frozen=True)
class TournamentResult:
    """Aggregate chip results and action tendencies from rotated 5-max hands."""

    hands: int
    pnl_bb: Mapping[str, float]
    statistics: Mapping[str, BotStatistics]


@dataclass
class _Counters:
    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    three_bets: int = 0
    faced_preflop_raise: int = 0
    folded_to_preflop_raise: int = 0
    aggressive: int = 0
    calls: int = 0


def run_tournament(
    bots: Mapping[str, PokerBot],
    hand_count: int,
    *,
    seed_start: int = 0,
    starting_stack: int = 10_000,
) -> TournamentResult:
    """Play a position-rotated tournament between exactly five named bots.

    Seat assignments rotate one step every hand, giving every entrant each
    table position equally often in every complete group of five hands.
    ``seed_start`` controls only the dealt cards; stochastic bots remain
    reproducible when constructed with explicit seeds.
    """

    if len(bots) != SEAT_COUNT:
        raise ValueError("a 5-max tournament requires exactly five bots")
    if hand_count < 1:
        raise ValueError("hand_count must be positive")
    names = tuple(bots)
    counters = {name: _Counters() for name in names}
    chip_delta = {name: 0 for name in names}
    for hand_index in range(hand_count):
        seat_names = tuple(names[(seat - hand_index) % SEAT_COUNT] for seat in range(SEAT_COUNT))
        for name in names:
            counters[name].hands += 1
        state = HandState(seed=seed_start + hand_index, starting_stack=starting_stack)
        voluntary_players: set[int] = set()
        pfr_players: set[int] = set()
        three_bet_players: set[int] = set()
        preflop_raises = 0
        while not state.complete:
            seat = state.actor
            assert seat is not None
            name = seat_names[seat]
            observation = observation_for(state, seat)
            action = bots[name].select_action(observation, observation["legal_actions"])
            if not state.legal_actions(seat)[action]:
                raise ValueError(f"bot {name!r} selected illegal action {action.value}")
            if action in RAISE_ACTIONS or action == Action.ALL_IN:
                counters[name].aggressive += 1
            elif action == Action.CALL:
                counters[name].calls += 1
            if state.street.value == "preflop":
                to_call = state.to_call(seat)
                if to_call > 0 and preflop_raises > 0:
                    counters[name].faced_preflop_raise += 1
                    if action == Action.FOLD:
                        counters[name].folded_to_preflop_raise += 1
                if action == Action.CALL or action in RAISE_ACTIONS or action == Action.ALL_IN:
                    # Posting a blind is not voluntary; calling/raising is.
                    if not (state.player(seat).committed_street > 0 and action == Action.CALL and to_call == 0):
                        voluntary_players.add(seat)
                if action in RAISE_ACTIONS or action == Action.ALL_IN:
                    if preflop_raises >= 1:
                        three_bet_players.add(seat)
                    pfr_players.add(seat)
                    preflop_raises += 1
            state.step(action)
        for seat in voluntary_players:
            counters[seat_names[seat]].vpip_hands += 1
        for seat in pfr_players:
            counters[seat_names[seat]].pfr_hands += 1
        for seat in three_bet_players:
            counters[seat_names[seat]].three_bets += 1
        for seat, player in enumerate(state.players):
            chip_delta[seat_names[seat]] += player.stack - starting_stack
    return TournamentResult(
        hands=hand_count,
        pnl_bb={name: amount / BIG_BLIND for name, amount in chip_delta.items()},
        statistics={name: _statistics(counter) for name, counter in counters.items()},
    )


def _statistics(counter: _Counters) -> BotStatistics:
    denominator = max(1, counter.hands)
    faced_raise = max(1, counter.faced_preflop_raise)
    return BotStatistics(
        hands=counter.hands,
        vpip=counter.vpip_hands / denominator,
        pfr=counter.pfr_hands / denominator,
        three_bet=counter.three_bets / denominator,
        fold_to_raise=counter.folded_to_preflop_raise / faced_raise,
        aggression_factor=counter.aggressive / max(1, counter.calls),
    )
