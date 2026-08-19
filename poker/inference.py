"""Checkpoint-backed policy inference isolated from the game server.

This module deliberately has no HTTP/WebSocket dependency.  The same fixed
checkpoint service is usable by the browser server, a CLI replay inspector, or
an offline evaluator.  ``HeuristicInferenceService`` is only a runnable
fallback for a fresh checkout; it is not an additional training algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol

from .betting import Action
from .bots import AggroBot, CallingStationBot, PokerBot, RandomBot, RuleBot, TightBot, hand_strength
from .model import BET_SIZE_ACTIONS, PokerAgentModel


EXPECTED_SHOWDOWN_SHARE_PROTOCOL = "active_hands_expected_showdown_share_v1"
"""Expected equal-pot share at showdown among the currently active hands."""

HEURISTIC_HAND_STRENGTH_PROTOCOL = "heuristic_hand_strength_v1"
"""Cheap visible-card strength proxy; deliberately not model equity."""


@dataclass(frozen=True)
class ScalarMetric:
    """One explicitly named scalar emitted beside the outcome distribution.

    A scalar must always state what it means.  In particular, multiway
    ``win/tie/loss`` cannot honestly be collapsed to ``win + 0.5 * tie``.
    """

    name: str
    value: float
    protocol: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "protocol": self.protocol}


@dataclass(frozen=True)
class InferenceResponse:
    """JSON-safe policy explanation attached to one selected action.

    ``equity`` remains the categorical ``win/tie/loss`` outcome head for
    compatibility.  Any displayed scalar is carried separately in
    :attr:`scalar_metric`, with a protocol that prevents a caller from
    mistaking a heuristic or a heads-up convention for multiway pot share.
    """

    action: str
    action_probabilities: dict[str, float]
    bet_size_probabilities: dict[str, float]
    equity: dict[str, float]
    value_bb: float
    scalar_metric: ScalarMetric | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_probabilities": dict(self.action_probabilities),
            "bet_size_probabilities": dict(self.bet_size_probabilities),
            "equity": dict(self.equity),
            "value_bb": self.value_bb,
            "scalar_metric": None if self.scalar_metric is None else self.scalar_metric.as_dict(),
        }


class DecisionService(Protocol):
    """Small boundary between a game and a fixed policy implementation."""

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        """Choose one currently legal engine action and return its explanation."""


@dataclass(frozen=True)
class PolicyIdentity:
    """Stable, presentation-safe identity for a policy at a table seat.

    ``policy_id`` is deliberately an identifier, rather than a filesystem
    location.  The web client only ever sends this value back to the server;
    checkpoint paths stay in the server-side catalog.
    """

    policy_id: str
    name: str
    kind: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.policy_id, "name": self.name, "kind": self.kind}


def decision_identity(service: DecisionService) -> PolicyIdentity:
    """Return a useful safe label for any decision service.

    Third-party services are supported too, so existing ``GameServer`` users
    do not have to wrap their service merely to use the observer.
    """

    identity = getattr(service, "identity", None)
    if isinstance(identity, PolicyIdentity):
        return identity
    name = type(service).__name__
    return PolicyIdentity(policy_id=f"service:{name}", name=name, kind="service")


class IdentifiedDecisionService:
    """Attach a safe catalog identity to an otherwise ordinary service."""

    def __init__(self, service: DecisionService, identity: PolicyIdentity) -> None:
        self.service = service
        self.identity = identity

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        return self.service.decide(observation)


class CheckpointInferenceService:
    """Inference-only wrapper around a :class:`PokerAgentModel` checkpoint."""

    def __init__(self, model: PokerAgentModel, *, policy_id: str = "checkpoint", name: str | None = None) -> None:
        self.model = model
        self.model.eval()
        self.identity = PolicyIdentity(policy_id=policy_id, name=name or policy_id, kind="checkpoint")

    @classmethod
    def from_checkpoint(
        cls, path: str, *, policy_id: str = "checkpoint", name: str | None = None
    ) -> "CheckpointInferenceService":
        return cls(PokerAgentModel.load_checkpoint(path), policy_id=policy_id, name=name)

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        decision = self.model.infer(observation)
        return InferenceResponse(
            action=decision.action,
            action_probabilities=dict(decision.action_probabilities),
            bet_size_probabilities=dict(decision.bet_size_probabilities),
            equity=dict(decision.equity),
            value_bb=decision.value_bb,
            scalar_metric=ScalarMetric(
                name="expected_showdown_share",
                value=float(decision.expected_showdown_share),
                protocol=EXPECTED_SHOWDOWN_SHARE_PROTOCOL,
            ),
        )


class BotInferenceService:
    """Adapt a baseline :class:`PokerBot` to the normal inference contract."""

    def __init__(self, bot: PokerBot, *, policy_id: str, name: str) -> None:
        self.bot = bot
        self.identity = PolicyIdentity(policy_id=policy_id, name=name, kind="bot")

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        selected = self.bot.select_action(observation)
        action_kind = "raise" if selected.value in BET_SIZE_ACTIONS else selected.value
        action_probabilities = {name: 0.0 for name in ("fold", "check", "call", "raise")}
        action_probabilities[action_kind] = 1.0
        bet_size_probabilities = {name: 0.0 for name in BET_SIZE_ACTIONS}
        if selected.value in bet_size_probabilities:
            bet_size_probabilities[selected.value] = 1.0
        strength = hand_strength(observation)
        return InferenceResponse(
            action=selected.value,
            action_probabilities=action_probabilities,
            bet_size_probabilities=bet_size_probabilities,
            equity={"win": strength, "tie": 0.0, "loss": 1.0 - strength},
            value_bb=(strength - 0.5) * float(observation["pot"]) / 100,
            scalar_metric=ScalarMetric(
                name="heuristic_hand_strength",
                value=strength,
                protocol=HEURISTIC_HAND_STRENGTH_PROTOCOL,
            ),
        )


class HeuristicInferenceService(BotInferenceService):
    """Deterministic UI fallback when no trained checkpoint is supplied.

    It preserves the response contract so UI and game-server work can be
    developed before training produces its first checkpoint.  The returned
    equity is explicitly a cheap hand-strength proxy, not Monte-Carlo equity.
    """

    def __init__(self) -> None:
        super().__init__(RuleBot(), policy_id="bot:rule", name="Rule bot")


_BOT_TYPES: dict[str, tuple[str, type[PokerBot]]] = {
    "random": ("Random bot", RandomBot),
    "tight": ("Tight bot", TightBot),
    "aggro": ("Aggro bot", AggroBot),
    "calling_station": ("Calling station", CallingStationBot),
    "rule": ("Rule bot", RuleBot),
}


def baseline_policy(policy_id: str, *, seed: int | None = None) -> BotInferenceService:
    """Build one fresh baseline service from a safe ``bot:<name>`` id."""

    if not policy_id.startswith("bot:"):
        raise ValueError("baseline policy id must start with 'bot:'")
    bot_key = policy_id.removeprefix("bot:")
    try:
        name, bot_type = _BOT_TYPES[bot_key]
    except KeyError as error:
        raise ValueError(f"unknown baseline policy {policy_id!r}") from error
    # Only stochastic bots accept a seed.  Keeping RuleBot/TightBot's simple
    # constructor also makes custom baseline implementations easy to add.
    try:
        bot = bot_type(seed=seed)  # type: ignore[call-arg]
    except TypeError:
        bot = bot_type()
    return BotInferenceService(bot, policy_id=policy_id, name=name)


def baseline_policy_catalog() -> list[dict[str, str]]:
    """Return presentation-safe options for UI and CLI policy selectors."""

    return [PolicyIdentity(f"bot:{key}", name, "bot").as_dict() for key, (name, _) in _BOT_TYPES.items()]


def validate_response(response: InferenceResponse, legal_actions: Mapping[str, bool]) -> None:
    """Reject a policy response that cannot safely be applied to the engine."""

    try:
        action = Action(response.action)
    except ValueError as error:
        raise ValueError(f"inference returned unknown action {response.action!r}") from error
    if not legal_actions.get(action.value, False):
        raise ValueError(f"inference returned illegal action {action.value!r}")
    for name in ("win", "tie", "loss"):
        if name not in response.equity:
            raise ValueError(f"inference equity has no {name!r}")
    probabilities = tuple(float(response.equity[name]) for name in ("win", "tie", "loss"))
    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("inference equity probabilities must be finite values in [0, 1]")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError("inference equity probabilities must sum to one")
    metric = response.scalar_metric
    if metric is None:
        return
    validate_scalar_metric(metric)


def validate_scalar_metric(metric: ScalarMetric) -> None:
    """Validate a scalar independently when loading a saved browser replay."""

    if metric.name not in {"expected_showdown_share", "heuristic_hand_strength"}:
        raise ValueError(f"inference scalar metric has unknown name {metric.name!r}")
    expected_protocol = (
        EXPECTED_SHOWDOWN_SHARE_PROTOCOL
        if metric.name == "expected_showdown_share"
        else HEURISTIC_HAND_STRENGTH_PROTOCOL
    )
    if metric.protocol != expected_protocol:
        raise ValueError(f"inference scalar metric has incompatible protocol {metric.protocol!r}")
    if not isfinite(metric.value) or not 0.0 <= metric.value <= 1.0:
        raise ValueError("inference scalar metric must be finite and in [0, 1]")
