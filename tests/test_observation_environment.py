from poker.betting import Action
from poker.environment import HoldemEnvironment
from poker.game_state import HandState
from poker.observation import observation_for


def test_observation_hides_other_players_hole_cards() -> None:
    state = HandState(seed=9)
    observation = observation_for(state, state.actor)
    assert "hole_cards" in observation
    assert all("hole_cards" not in player for player in observation["players"])


def test_environment_runs_one_hand() -> None:
    environment = HoldemEnvironment()
    observation = environment.reset(seed=7)
    while True:
        action = Action.CHECK if observation["legal_actions"]["check"] else Action.CALL
        observation, done, info = environment.step(action)
        if done:
            assert info["replay"]["street"] == "complete"
            break
