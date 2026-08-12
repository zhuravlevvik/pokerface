"""Training-safe hand traces and deterministic replay helpers.

The public ``observation`` in a decision trace is the only state intended for
an agent.  A completed replay is retained separately: it may contain every
player's hole cards because it is an audit/training artifact generated after a
hand ends, never a model input during a hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .betting import Action
from .equity import EquitySnapshot, EquityTarget, capture_equity_snapshot, generate_equity_target
from .game_state import HandState
from .observation import observation_for
from .rules import BIG_BLIND


def _action_log(state: HandState) -> list[dict[str, Any]]:
    return [record.as_dict() for record in state.action_history]


@dataclass
class DecisionTrace:
    """One agent-visible decision point, captured immediately before its move."""

    hand_id: int
    seed: int | None
    hero_seat: int
    street: str
    observation: dict[str, Any]
    legal_action_mask: dict[str, bool]
    selected_action: str
    action_log: list[dict[str, Any]]
    pot: int
    stacks: list[int]
    active_players: list[int]
    terminal_pnl_bb: float | None = None
    equity_snapshot_reference: str | None = None
    equity_target: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy suitable for a compact dataset."""

        return {
            "hand_id": self.hand_id,
            "seed": self.seed,
            "hero_seat": self.hero_seat,
            "street": self.street,
            "observation": self.observation,
            "legal_action_mask": self.legal_action_mask,
            "selected_action": self.selected_action,
            "action_log": self.action_log,
            "pot": self.pot,
            "stacks": self.stacks,
            "active_players": self.active_players,
            "terminal_pnl_bb": self.terminal_pnl_bb,
            "equity_snapshot_reference": self.equity_snapshot_reference,
            "equity_target": None if self.equity_target is None else list(self.equity_target),
        }


@dataclass
class HandTrace:
    """All decision records for one hand, completed with terminal PnLs."""

    hand_id: int
    seed: int | None
    button_seat: int
    starting_stack: int
    decisions: list[DecisionTrace] = field(default_factory=list)
    terminal_pnl_bb: dict[int, float] | None = None
    equity_samples: int = 16
    _equity_snapshots: dict[str, EquitySnapshot] = field(default_factory=dict, init=False, repr=False)

    def record_action(self, state: HandState, action: Action | str) -> None:
        """Capture the current actor's legal view before applying ``action``."""

        if state.actor is None:
            raise RuntimeError("cannot trace an action without an actor")
        normalized = Action(action)
        seat = state.actor
        observation = observation_for(state, seat)
        reference = f"{self.hand_id}:{len(self.decisions)}"
        self._equity_snapshots[reference] = capture_equity_snapshot(state, seat, reference=reference)
        self.decisions.append(
            DecisionTrace(
                hand_id=self.hand_id,
                seed=self.seed,
                hero_seat=seat,
                street=state.street.value,
                observation=observation,
                legal_action_mask=dict(observation["legal_actions"]),
                selected_action=normalized.value,
                action_log=_action_log(state),
                pot=state.pot,
                stacks=[player.stack for player in state.players],
                active_players=[player.seat for player in state.players if not player.folded],
                equity_snapshot_reference=reference,
            )
        )

    def complete(self, state: HandState) -> None:
        if not state.complete:
            raise RuntimeError("cannot complete a trace for a live hand")
        pnl = {
            player.seat: (player.stack - self.starting_stack) / BIG_BLIND
            for player in state.players
        }
        self.terminal_pnl_bb = pnl
        for decision in self.decisions:
            decision.terminal_pnl_bb = pnl[decision.hero_seat]
            reference = decision.equity_snapshot_reference
            if reference is None:  # Defensive: every trace created here has one.
                raise RuntimeError("decision has no equity snapshot")
            target: EquityTarget = generate_equity_target(self._equity_snapshots[reference], samples=self.equity_samples)
            decision.equity_target = target.as_list()

    def as_training_records(self) -> list[dict[str, Any]]:
        """Return model-safe records, including labels but never hidden cards."""

        if self.terminal_pnl_bb is None:
            raise RuntimeError("hand trace is not terminal")
        return [decision.as_dict() for decision in self.decisions]


def rebuild_hand(replay: dict[str, Any]) -> HandState:
    """Rebuild a terminal hand from its replay action log and verify its result.

    This intentionally ignores exposed hole cards in ``replay``.  The seed and
    action sequence are the authoritative inputs, so replay validation also
    catches accidental nondeterminism in the game state machine.
    """

    state = HandState(
        seed=replay["seed"],
        button_seat=replay["button_seat"],
        starting_stack=replay.get("starting_stack", 10_000),
        player_count=replay.get("player_count", 5),
        allowed_raise_actions=None
        if replay.get("allowed_raise_actions") is None
        else frozenset(Action(value) for value in replay["allowed_raise_actions"]),
    )
    for record in replay["actions"]:
        if state.actor != record["seat"]:
            raise ValueError("replay action order does not match the hand state")
        state.step(record["action"])
    if not state.complete:
        raise ValueError("replay does not contain a completed hand")
    if state.replay() != replay:
        raise ValueError("replay result does not match deterministic reconstruction")
    return state
