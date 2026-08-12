from __future__ import annotations

import logging
from typing import Any

from poker.betting import Action
from poker.environment import HoldemEnvironment
from poker.game_state import HandState
from poker.observation import OBSERVATION_VERSION, ObservationFeatureStatistics, observation_for


def _all_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_all_strings(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_all_strings(item) for item in value)) if value else set()
    return set()


def test_observation_hides_other_players_hole_cards() -> None:
    state = HandState(seed=9)
    observation = observation_for(state, state.actor)
    assert "hole_cards" in observation
    assert all("hole_cards" not in player for player in observation["players"])


def test_canonical_observation_is_versioned_normalized_and_player_set_based() -> None:
    state = HandState(seed=9)
    assert state.actor is not None
    observation = observation_for(state, state.actor)

    assert observation["schema_version"] == OBSERVATION_VERSION
    assert observation["cards"]["hole_cards"] == observation["hole_cards"]
    assert observation["table"]["pot_bb"] == 1.5
    assert observation["hero"]["to_call_bb"] == 1.0
    assert len(observation["player_set"]) == 5
    assert observation["player_mask"] == [True] * 5
    assert observation["legal_action_mask"] == observation["legal_actions"]
    assert all("stack_bb" in player and "last_action" in player for player in observation["player_set"])


def test_opponent_and_future_cards_never_affect_hero_observation() -> None:
    state = HandState(seed=12)
    assert state.actor is not None
    hero = state.actor
    before = observation_for(state, hero)
    opponent_seats = [seat for seat in range(5) if seat != hero]
    first, second = opponent_seats[:2]
    future_cards = [str(card) for card in state.deck.snapshot()]
    opponent_cards = [str(card) for seat in opponent_seats for card in state.hole_cards[seat]]

    # This mutation models an alternative private deal.  The hero's legal
    # information and its model input must be byte-for-byte unchanged.
    state.hole_cards[first], state.hole_cards[second] = state.hole_cards[second], state.hole_cards[first]
    after = observation_for(state, hero)

    assert after == before
    emitted_strings = _all_strings(after)
    assert not emitted_strings.intersection(opponent_cards)
    assert not emitted_strings.intersection(future_cards)


def test_legal_action_mask_exactly_matches_engine_and_blocks_prohibited_actions() -> None:
    state = HandState(seed=1)
    assert state.actor is not None
    observation = observation_for(state, state.actor)
    assert observation["legal_action_mask"] == {
        action.value: allowed for action, allowed in state.legal_actions().items()
    }
    assert observation["legal_action_mask"][Action.CHECK.value] is False
    assert observation["legal_action_mask"][Action.FOLD.value] is True
    assert observation["legal_action_mask"][Action.CALL.value] is True

    # A view for a non-actor may be inspected but must never be allowed to act.
    non_actor = (state.actor + 1) % 5
    assert not any(observation_for(state, non_actor)["legal_action_mask"].values())

    # Once the four callers have responded, the BB can check but cannot call
    # or fold into a zero price.
    for _ in range(4):
        state.step(Action.CALL)
    assert state.actor is not None
    bb_observation = observation_for(state, state.actor)
    assert bb_observation["legal_action_mask"][Action.CHECK.value] is True
    assert bb_observation["legal_action_mask"][Action.CALL.value] is False
    assert bb_observation["legal_action_mask"][Action.FOLD.value] is False


def test_action_history_uses_pre_action_pot_normalization_and_statistics_are_loggable(caplog) -> None:
    state = HandState(seed=2)
    state.step(Action.CALL)
    assert state.actor is not None
    observation = observation_for(state, state.actor)
    first_action = observation["action_history"][0]
    assert first_action["amount_bb"] == 1.0
    assert first_action["amount_to_pot"] == 1.0 / 1.5
    assert first_action["position"] == "UTG"

    statistics = ObservationFeatureStatistics()
    statistics.update(observation)
    with caplog.at_level(logging.INFO):
        snapshot = statistics.log()
    assert snapshot["hero.stack_bb"]["count"] == 1
    assert "observation feature statistics" in caplog.text


def test_environment_runs_one_hand() -> None:
    environment = HoldemEnvironment()
    observation = environment.reset(seed=7)
    while True:
        action = Action.CHECK if observation["legal_actions"]["check"] else Action.CALL
        observation, done, info = environment.step(action)
        if done:
            assert info["replay"]["street"] == "complete"
            break
