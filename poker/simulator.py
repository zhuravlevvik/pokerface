"""Vector-style, UI-free simulator for independent Hold'em hands.

This module deliberately owns no policy, bot, neural network, or trainer.
Callers provide one legal action per live table.  That keeps environment
throughput and trace collection independently testable before RL is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence, Union

from .betting import Action
from .game_state import HandState
from .observation import observation_for
from .traces import HandTrace


@dataclass(frozen=True)
class BatchStep:
    """The result of advancing each live table by at most one action."""

    observations: tuple[dict[str, Any] | None, ...]
    legal_action_masks: tuple[dict[str, bool], ...]
    terminal: tuple[bool, ...]
    rewards: tuple[dict[int, float], ...]
    infos: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SimulationBenchmark:
    hands: int
    decisions: int
    seconds: float

    @property
    def hands_per_second(self) -> float:
        return self.hands / self.seconds if self.seconds else float("inf")


class BatchedHoldemEnvironment:
    """A deterministic batch of independent single-hand environments.

    ``step`` requires an action for each nonterminal table and ``None`` for a
    terminal one.  A hand trace is retained for every hand; full replays are
    retained only when ``capture_replays`` is enabled.
    """

    def __init__(
        self,
        table_count: int,
        *,
        button_seat: int = 0,
        starting_stack: int = 10_000,
        player_count: int = 5,
        allowed_raise_actions: frozenset[Action] | None = None,
        capture_replays: bool = False,
        equity_samples: int = 16,
    ) -> None:
        if table_count < 1:
            raise ValueError("table_count must be positive")
        if equity_samples < 1:
            raise ValueError("equity_samples must be positive")
        self.table_count = table_count
        self.button_seat = button_seat
        self.starting_stack = starting_stack
        self.player_count = player_count
        self.allowed_raise_actions = allowed_raise_actions
        self.capture_replays = capture_replays
        self.equity_samples = equity_samples
        self.states: list[HandState] = []
        self._traces: list[HandTrace] = []
        self._hand_counter = 0
        self._rewards: list[dict[int, float]] = [self._zero_rewards() for _ in range(table_count)]

    @property
    def terminal(self) -> tuple[bool, ...]:
        return tuple(state.complete for state in self.states)

    @property
    def rewards(self) -> tuple[dict[int, float], ...]:
        return tuple(dict(reward) for reward in self._rewards)

    @property
    def traces(self) -> tuple[HandTrace, ...]:
        """Training traces for the currently dealt hands.

        Callers may use the public decision order to join an on-policy sample
        to its terminal virtual-showdown label.  The trace's private snapshots
        remain encapsulated and are never part of player observations.
        """

        return tuple(self._traces)

    @property
    def legal_action_masks(self) -> tuple[dict[str, bool], ...]:
        return tuple(self._mask(state) for state in self.states)

    def reset(self, *, seeds: Sequence[int | None] | None = None) -> tuple[dict[str, Any], ...]:
        """Deal one new reproducible hand per table and return actor views."""

        if seeds is None:
            seeds = [None] * self.table_count
        if len(seeds) != self.table_count:
            raise ValueError("seeds length must match table_count")
        self.states = []
        self._traces = []
        self._rewards = [self._zero_rewards() for _ in range(self.table_count)]
        observations: list[dict[str, Any]] = []
        for seed in seeds:
            state = HandState(seed=seed, button_seat=self.button_seat, starting_stack=self.starting_stack, player_count=self.player_count, allowed_raise_actions=self.allowed_raise_actions)
            trace = HandTrace(
                hand_id=self._hand_counter,
                seed=seed,
                button_seat=self.button_seat,
                starting_stack=self.starting_stack,
                equity_samples=self.equity_samples,
            )
            self._hand_counter += 1
            self.states.append(state)
            self._traces.append(trace)
            observations.append(observation_for(state, state.actor))  # type: ignore[arg-type]
        return tuple(observations)

    def step(self, actions: Sequence[Action | str | None]) -> BatchStep:
        """Advance every active table once and return batched agent-facing data."""

        if len(self.states) != self.table_count:
            raise RuntimeError("call reset before step")
        if len(actions) != self.table_count:
            raise ValueError("actions length must match table_count")
        observations: list[dict[str, Any] | None] = []
        masks: list[dict[str, bool]] = []
        terminals: list[bool] = []
        infos: list[dict[str, Any]] = []
        for index, (state, trace, action) in enumerate(zip(self.states, self._traces, actions)):
            if state.complete:
                if action is not None:
                    raise ValueError("terminal tables require a None action")
                observations.append(None)
                masks.append(self._mask(state))
                terminals.append(True)
                infos.append({"hand_id": trace.hand_id, "trace": None, "replay": None})
                continue
            if action is None:
                raise ValueError("live tables require an action")
            trace.record_action(state, action)
            state.step(action)
            if state.complete:
                trace.complete(state)
                self._rewards[index] = dict(trace.terminal_pnl_bb or {})
                replay = state.replay() if self.capture_replays else None
                observations.append(None)
                masks.append(self._mask(state))
                terminals.append(True)
                infos.append({"hand_id": trace.hand_id, "trace": trace, "replay": replay})
            else:
                observations.append(observation_for(state, state.actor))  # type: ignore[arg-type]
                masks.append(self._mask(state))
                terminals.append(False)
                infos.append({"hand_id": trace.hand_id, "trace": None, "replay": None})
        return BatchStep(
            observations=tuple(observations),
            legal_action_masks=tuple(masks),
            terminal=tuple(terminals),
            rewards=self.rewards,
            infos=tuple(infos),
        )

    def _zero_rewards(self) -> dict[int, float]:
        return {seat: 0.0 for seat in range(self.player_count)}

    @staticmethod
    def _mask(state: HandState) -> dict[str, bool]:
        if state.complete:
            return {action.value: False for action in Action}
        return {action.value: allowed for action, allowed in state.legal_actions().items()}


ActionSelector = Callable[[dict[str, Any], dict[str, bool], int], Union[Action, str]]


def benchmark_hands(
    hand_count: int,
    action_selector: ActionSelector,
    *,
    batch_size: int = 1,
    seed_start: int = 0,
) -> SimulationBenchmark:
    """Measure hand-generation throughput with a caller-supplied action source.

    The selector receives a player-safe observation, its legal mask and the
    table index.  It is not a poker bot supplied by this module.
    """

    if hand_count < 1:
        raise ValueError("hand_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    started = perf_counter()
    decisions = 0
    completed = 0
    next_seed = seed_start
    while completed < hand_count:
        count = min(batch_size, hand_count - completed)
        environment = BatchedHoldemEnvironment(count)
        observations = environment.reset(seeds=range(next_seed, next_seed + count))
        next_seed += count
        active = [True] * count
        while any(active):
            actions: list[Action | str | None] = []
            for table, observation in enumerate(observations):
                if active[table]:
                    actions.append(action_selector(observation, environment.legal_action_masks[table], table))
                    decisions += 1
                else:
                    actions.append(None)
            result = environment.step(actions)
            observations = result.observations
            active = [not done for done in result.terminal]
        completed += count
    return SimulationBenchmark(hands=hand_count, decisions=decisions, seconds=perf_counter() - started)
