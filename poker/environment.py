"""Small dependency-free environment adapter for the future training stage."""

from __future__ import annotations

from .betting import Action
from .game_state import HandState
from .observation import observation_for


class HoldemEnvironment:
    """Single-hand API; no trainer, bot, UI or vectorisation lives here."""

    def __init__(self, *, button_seat: int = 0, starting_stack: int = 10_000, player_count: int = 5, allowed_raise_actions: frozenset[Action] | None = None) -> None:
        self.button_seat = button_seat
        self.starting_stack = starting_stack
        self.player_count = player_count
        self.allowed_raise_actions = allowed_raise_actions
        self.state: HandState | None = None

    def reset(self, *, seed: int | None = None) -> dict:
        self.state = HandState(seed=seed, button_seat=self.button_seat, starting_stack=self.starting_stack, player_count=self.player_count, allowed_raise_actions=self.allowed_raise_actions)
        return observation_for(self.state, self.state.actor)  # type: ignore[arg-type]

    def step(self, action: Action | str) -> tuple[dict | None, bool, dict]:
        if self.state is None:
            raise RuntimeError("call reset before step")
        self.state.step(action)
        done = self.state.complete
        observation = None if done else observation_for(self.state, self.state.actor)  # type: ignore[arg-type]
        return observation, done, {"replay": self.state.replay() if done else None}
