"""Checkpoint-backed policy inference isolated from the game server.

This module deliberately has no HTTP/WebSocket dependency.  The same fixed
checkpoint service is usable by the browser server, a CLI replay inspector, or
an offline evaluator.  ``HeuristicInferenceService`` is only a runnable
fallback for a fresh checkout; it is not an additional training algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .betting import Action
from .bots import RuleBot, hand_strength
from .model import BET_SIZE_ACTIONS, PokerAgentModel


@dataclass(frozen=True)
class InferenceResponse:
    """JSON-safe model explanation attached to one selected action."""

    action: str
    action_probabilities: dict[str, float]
    bet_size_probabilities: dict[str, float]
    equity: dict[str, float]
    value_bb: float

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_probabilities": dict(self.action_probabilities),
            "bet_size_probabilities": dict(self.bet_size_probabilities),
            "equity": dict(self.equity),
            "value_bb": self.value_bb,
        }


class DecisionService(Protocol):
    """Small boundary between a game and a fixed policy implementation."""

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        """Choose one currently legal engine action and return its explanation."""


class CheckpointInferenceService:
    """Inference-only wrapper around a :class:`PokerAgentModel` checkpoint."""

    def __init__(self, model: PokerAgentModel) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str) -> "CheckpointInferenceService":
        return cls(PokerAgentModel.load_checkpoint(path))

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        decision = self.model.infer(observation)
        equity = dict(decision.equity)
        equity["total"] = equity["win"] + 0.5 * equity["tie"]
        return InferenceResponse(
            action=decision.action,
            action_probabilities=dict(decision.action_probabilities),
            bet_size_probabilities=dict(decision.bet_size_probabilities),
            equity=equity,
            value_bb=decision.value_bb,
        )


class HeuristicInferenceService:
    """Deterministic UI fallback when no trained checkpoint is supplied.

    It preserves the response contract so UI and game-server work can be
    developed before training produces its first checkpoint.  The returned
    equity is explicitly a cheap hand-strength proxy, not Monte-Carlo equity.
    """

    def __init__(self) -> None:
        self.bot = RuleBot()

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        selected = self.bot.select_action(observation)
        action_kind = "raise" if selected.value in BET_SIZE_ACTIONS else selected.value
        action_probabilities = {name: 0.0 for name in ("fold", "check", "call", "raise")}
        action_probabilities[action_kind] = 1.0
        bet_size_probabilities = {name: 0.0 for name in BET_SIZE_ACTIONS}
        if selected.value in bet_size_probabilities:
            bet_size_probabilities[selected.value] = 1.0
        strength = hand_strength(observation)
        # This deliberately has no tie estimate; a trained equity head replaces
        # it when a checkpoint is configured.
        equity = {"win": strength, "tie": 0.0, "loss": 1.0 - strength, "total": strength}
        return InferenceResponse(
            action=selected.value,
            action_probabilities=action_probabilities,
            bet_size_probabilities=bet_size_probabilities,
            equity=equity,
            value_bb=(strength - 0.5) * float(observation["pot"]) / 100,
        )


def validate_response(response: InferenceResponse, legal_actions: Mapping[str, bool]) -> None:
    """Reject a policy response that cannot safely be applied to the engine."""

    try:
        action = Action(response.action)
    except ValueError as error:
        raise ValueError(f"inference returned unknown action {response.action!r}") from error
    if not legal_actions.get(action.value, False):
        raise ValueError(f"inference returned illegal action {action.value!r}")
    for name in ("win", "tie", "loss", "total"):
        if name not in response.equity:
            raise ValueError(f"inference equity has no {name!r}")
    if abs(sum(response.equity[name] for name in ("win", "tie", "loss")) - 1.0) > 1e-5:
        raise ValueError("inference equity probabilities must sum to one")
    expected_total = response.equity["win"] + 0.5 * response.equity["tie"]
    if abs(response.equity["total"] - expected_total) > 1e-5:
        raise ValueError("inference equity total is inconsistent")
