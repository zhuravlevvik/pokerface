"""Observable single-hand game server, independent of the web transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from .betting import Action
from .game_state import HandState
from .inference import (
    CheckpointInferenceService,
    DecisionService,
    HeuristicInferenceService,
    IdentifiedDecisionService,
    InferenceResponse,
    PolicyIdentity,
    baseline_policy,
    baseline_policy_catalog,
    decision_identity,
    validate_response,
)
from .observation import observation_for
from .traces import rebuild_hand

ViewerMode = Literal["player", "spectator"]
# ``DecisionService`` is a runtime-checkable structural protocol only at call
# sites.  Do not form ``str | DecisionService`` here: on some supported Python
# versions a Protocol cannot participate in a runtime union expression.
CheckpointSource = object


def _validate_viewer(mode: str, hero_seat: int, state: HandState) -> ViewerMode:
    if mode not in {"player", "spectator"}:
        raise ValueError("mode must be 'player' or 'spectator'")
    if not 0 <= hero_seat < state.player_count:
        raise ValueError(f"hero_seat must be in 0..{state.player_count - 1}")
    return mode  # type: ignore[return-value]


@dataclass(frozen=True)
class EquityPoint:
    street: str
    seat: int
    equity: float
    action_index: int

    def as_dict(self) -> dict[str, object]:
        return {"street": self.street, "seat": self.seat, "equity": self.equity, "action_index": self.action_index}


@dataclass
class ObservableHand:
    """Mutable delivery state for one hand and one configured policy service."""

    state: HandState
    inference: DecisionService
    equity_points: list[EquityPoint] = field(default_factory=list)
    analyses: list[dict[str, object]] = field(default_factory=list)

    def policy_for_seat(self, seat: int) -> PolicyIdentity:
        if isinstance(self.inference, SeatPolicyRouter):
            return self.inference.identity_for(seat)
        return decision_identity(self.inference)

    def browser_replay(self, *, reveal_hole_cards: bool, hero_seat: int) -> dict[str, object]:
        """Engine replay plus UI-only explanations needed for an exact replay."""

        replay = self.state.replay(reveal_hole_cards=reveal_hole_cards)
        replay["equity_points"] = [
            point.as_dict() for point in self.equity_points if reveal_hole_cards or point.seat == hero_seat
        ]
        replay["analyses"] = list(self.analyses)
        replay["policies"] = {str(seat): self.policy_for_seat(seat).as_dict() for seat in range(self.state.player_count)}
        return replay

    def table_view(self, *, mode: ViewerMode, hero_seat: int) -> dict[str, object]:
        _validate_viewer(mode, hero_seat, self.state)
        reveal_all = mode == "spectator" and self.state.complete
        players: list[dict[str, object]] = []
        for player in self.state.players:
            item: dict[str, object] = {
                "seat": player.seat,
                "position": self.state.positions[player.seat],
                "stack": player.stack,
                "committed_street": player.committed_street,
                "folded": player.folded,
                "all_in": player.all_in,
                "payout": self.state.payouts[player.seat],
                "is_hero": player.seat == hero_seat,
                "policy": self.policy_for_seat(player.seat).as_dict(),
                "pnl": player.stack - self.state.starting_stack,
            }
            if reveal_all or (mode == "player" and player.seat == hero_seat):
                item["hole_cards"] = [str(card) for card in self.state.hole_cards[player.seat]]
            players.append(item)
        return {
            "street": self.state.street.value,
            "board": [str(card) for card in self.state.board],
            "pot": self.state.pot,
            "button_seat": self.state.button_seat,
            "current_actor": self.state.actor,
            "players": players,
            "action_history": [record.as_dict() for record in self.state.action_history],
            "complete": self.state.complete,
            "equity_points": [point.as_dict() for point in self.equity_points if point.seat == hero_seat],
        }

    def play(self, *, mode: ViewerMode, hero_seat: int) -> Iterator[dict[str, object]]:
        """Yield fully serialisable updates before/after every policy decision."""

        _validate_viewer(mode, hero_seat, self.state)
        yield {"type": "hand_started", "table": self.table_view(mode=mode, hero_seat=hero_seat)}
        while not self.state.complete:
            if self.state.actor is None:
                raise RuntimeError("live hand has no actor")
            seat = self.state.actor
            observation = observation_for(self.state, seat)
            response = self.inference.decide(observation)
            validate_response(response, observation["legal_actions"])
            policy = self.policy_for_seat(seat)
            visible_analysis = response.as_dict() if mode == "spectator" or seat == hero_seat else None
            self.equity_points.append(
                EquityPoint(
                    street=self.state.street.value,
                    seat=seat,
                    equity=response.equity["total"],
                    action_index=len(self.state.action_history),
                )
            )
            self.analyses.append({"seat": seat, "policy": policy.as_dict(), "analysis": visible_analysis})
            self.state.step(Action(response.action))
            yield {
                "type": "action",
                "seat": seat,
                "selected_action": response.action,
                "analysis": visible_analysis,
                "policy": policy.as_dict(),
                "table": self.table_view(mode=mode, hero_seat=hero_seat),
            }
        yield {
            "type": "hand_complete",
            "table": self.table_view(mode=mode, hero_seat=hero_seat),
            "replay": self.browser_replay(reveal_hole_cards=mode == "spectator", hero_seat=hero_seat),
        }


class SeatPolicyRouter:
    """Dispatch a decision by the acting seat, without leaking other policies."""

    def __init__(self, services: Mapping[int, DecisionService]) -> None:
        if not services:
            raise ValueError("seat policy router needs at least one service")
        self.services = dict(services)

    def decide(self, observation: Mapping[str, object]) -> InferenceResponse:
        seat = observation.get("seat")
        if not isinstance(seat, int):
            raise ValueError("observation must contain an integer seat")
        try:
            return self.services[seat].decide(observation)
        except KeyError as error:
            raise ValueError(f"no policy configured for seat {seat}") from error

    def identity_for(self, seat: int) -> PolicyIdentity:
        try:
            return decision_identity(self.services[seat])
        except KeyError as error:
            raise ValueError(f"no policy configured for seat {seat}") from error


class GameServer:
    """Creates observable games; no trainer or checkpoint mutation lives here."""

    def __init__(
        self,
        inference: DecisionService | None = None,
        *,
        checkpoint_catalog: Mapping[str, CheckpointSource] | None = None,
        default_seat_policies: Mapping[int, str] | None = None,
    ) -> None:
        self.inference = inference or HeuristicInferenceService()
        self._checkpoint_catalog = self._validate_checkpoint_catalog(checkpoint_catalog or {})
        self._checkpoint_services: dict[str, DecisionService] = {}
        self.default_seat_policies = dict(default_seat_policies or {})

    @staticmethod
    def _validate_checkpoint_catalog(catalog: Mapping[str, object]) -> dict[str, CheckpointSource]:
        result: dict[str, CheckpointSource] = {}
        for key, source in catalog.items():
            if not key or ":" in key or not isinstance(source, (str, Path)) and not hasattr(source, "decide"):
                raise ValueError("checkpoint catalog keys must be non-empty ids without ':'")
            result[key] = str(source) if isinstance(source, Path) else source
        return result

    def available_policies(self) -> list[dict[str, str]]:
        """Policies selectable by a client; checkpoint paths never leave this process."""

        policies = baseline_policy_catalog()
        policies.extend(
            PolicyIdentity(policy_id=f"checkpoint:{key}", name=key, kind="checkpoint").as_dict()
            for key in sorted(self._checkpoint_catalog)
        )
        return policies

    def _policy_from_id(self, policy_id: str, *, seed: int | None) -> DecisionService:
        if policy_id.startswith("bot:"):
            return baseline_policy(policy_id, seed=seed)
        if policy_id.startswith("checkpoint:"):
            key = policy_id.removeprefix("checkpoint:")
            if key not in self._checkpoint_catalog:
                raise ValueError(f"checkpoint policy {policy_id!r} is not in the server catalog")
            if key not in self._checkpoint_services:
                source = self._checkpoint_catalog[key]
                self._checkpoint_services[key] = (
                    IdentifiedDecisionService(source, PolicyIdentity(policy_id=policy_id, name=key, kind="checkpoint"))
                    if not isinstance(source, str)
                    else CheckpointInferenceService.from_checkpoint(source, policy_id=policy_id, name=key)
                )
            return self._checkpoint_services[key]
        raise ValueError(f"unknown policy id {policy_id!r}")

    def _seat_router(
        self, *, player_count: int, seat_policies: Mapping[int, str] | None, seed: int | None
    ) -> SeatPolicyRouter:
        requested = dict(self.default_seat_policies)
        if seat_policies is not None:
            requested.update(seat_policies)
        invalid_seats = sorted(seat for seat in requested if not 0 <= seat < player_count)
        if invalid_seats:
            raise ValueError(f"seat policies contain invalid seats: {invalid_seats}")
        services: dict[int, DecisionService] = {}
        for seat in range(player_count):
            policy_id = requested.get(seat)
            # A server constructed with one inference service retains its old
            # behavior unless a seat explicitly selects a catalog policy.
            services[seat] = self.inference if policy_id is None else self._policy_from_id(
                policy_id, seed=None if seed is None else seed * 31 + seat
            )
        return SeatPolicyRouter(services)

    def _replay_router(self, replay: Mapping[str, Any], *, player_count: int, seed: int | None) -> SeatPolicyRouter:
        """Restore saved labels without resolving any checkpoint path on replay."""

        raw_policies = replay.get("policies")
        if raw_policies is None:
            return self._seat_router(player_count=player_count, seat_policies=None, seed=seed)
        if not isinstance(raw_policies, Mapping):
            raise ValueError("replay policies must be an object")
        services: dict[int, DecisionService] = {}
        for seat in range(player_count):
            raw_identity = raw_policies.get(str(seat))
            if not isinstance(raw_identity, Mapping):
                raise ValueError(f"replay has no policy identity for seat {seat}")
            try:
                identity = PolicyIdentity(
                    policy_id=str(raw_identity["id"]), name=str(raw_identity["name"]), kind=str(raw_identity["kind"])
                )
            except KeyError as error:
                raise ValueError("replay policy identity must contain id, name and kind") from error
            services[seat] = IdentifiedDecisionService(self.inference, identity)
        return SeatPolicyRouter(services)

    def observe_hand(
        self,
        *,
        seed: int | None = None,
        button_seat: int = 0,
        starting_stack: int = 10_000,
        player_count: int = 5,
        hero_seat: int = 0,
        mode: ViewerMode = "player",
        seat_policies: Mapping[int, str] | None = None,
    ) -> list[dict[str, object]]:
        state = HandState(seed=seed, button_seat=button_seat, starting_stack=starting_stack, player_count=player_count)
        router = self._seat_router(player_count=player_count, seat_policies=seat_policies, seed=seed)
        return list(ObservableHand(state, router).play(mode=mode, hero_seat=hero_seat))

    def observe_series(
        self,
        *,
        seed_start: int | None = None,
        hands: int = 1,
        button_seat: int = 0,
        starting_stack: int = 10_000,
        player_count: int = 5,
        hero_seat: int = 0,
        mode: ViewerMode = "player",
        seat_policies: Mapping[int, str] | None = None,
    ) -> list[dict[str, object]]:
        """Play independent deterministic hands and include a running PnL summary.

        A watch series intentionally resets stacks for every hand: it is a
        visual inspection tool, not a cash-game session simulator.  PnL is
        nevertheless accumulated from each hand's terminal stack delta.
        """

        if not isinstance(hands, int) or not 1 <= hands <= 1_000:
            raise ValueError("hands must be an integer in 1..1000")
        _validate_viewer(mode, hero_seat, HandState(seed=seed_start, button_seat=button_seat, starting_stack=starting_stack, player_count=player_count))
        totals = {seat: 0 for seat in range(player_count)}
        policies = self._seat_router(player_count=player_count, seat_policies=seat_policies, seed=seed_start)
        events: list[dict[str, object]] = [
            {
                "type": "series_started",
                "series": {
                    "hands": hands,
                    "seed_start": seed_start,
                    "policies": {str(seat): policies.identity_for(seat).as_dict() for seat in range(player_count)},
                },
            }
        ]
        for hand_index in range(hands):
            seed = None if seed_start is None else seed_start + hand_index
            state = HandState(
                seed=seed,
                button_seat=(button_seat + hand_index) % player_count,
                starting_stack=starting_stack,
                player_count=player_count,
            )
            # Build services for every hand so seeded random baselines have a
            # reproducible, hand-local RNG rather than mutable shared state.
            router = self._seat_router(player_count=player_count, seat_policies=seat_policies, seed=seed)
            hand_events = list(ObservableHand(state, router).play(mode=mode, hero_seat=hero_seat))
            for event in hand_events:
                event["hand_index"] = hand_index
            complete = hand_events[-1]["table"]
            if not isinstance(complete, dict):  # pragma: no cover - internal invariant.
                raise RuntimeError("completed hand has no table")
            for player in complete["players"]:  # type: ignore[index]
                totals[int(player["seat"])] += int(player["pnl"])
            hand_events[-1]["series_pnl"] = dict(totals)
            events.extend(hand_events)
        events.append(
            {
                "type": "series_complete",
                "summary": {
                    "hands": hands,
                    "pnl": totals,
                    "policies": {str(seat): policies.identity_for(seat).as_dict() for seat in range(player_count)},
                },
            }
        )
        return events

    def replay_hand(self, replay: dict[str, Any], *, hero_seat: int = 0, mode: ViewerMode = "player") -> list[dict[str, object]]:
        """Validate an imported replay then deterministically reconstruct its UI events."""

        engine_replay = {key: value for key, value in replay.items() if key not in {"equity_points", "analyses", "policies"}}
        rebuild_hand(engine_replay)
        state = HandState(
            seed=engine_replay["seed"],
            button_seat=engine_replay["button_seat"],
            starting_stack=engine_replay.get("starting_stack", 10_000),
            player_count=engine_replay.get("player_count", 5),
            allowed_raise_actions=None
            if engine_replay.get("allowed_raise_actions") is None
            else frozenset(Action(value) for value in engine_replay["allowed_raise_actions"]),
        )
        mode = _validate_viewer(mode, hero_seat, state)
        observable = ObservableHand(state, self._replay_router(replay, player_count=state.player_count, seed=state.seed))
        raw_points = replay.get("equity_points", [])
        raw_analyses = replay.get("analyses", [])
        if not isinstance(raw_points, list) or not isinstance(raw_analyses, list):
            raise ValueError("replay UI metadata must be lists")
        points_by_action: dict[int, list[EquityPoint]] = {}
        for item in raw_points:
            if not isinstance(item, dict):
                raise ValueError("replay equity point must be an object")
            point = EquityPoint(
                street=str(item["street"]), seat=int(item["seat"]), equity=float(item["equity"]), action_index=int(item["action_index"])
            )
            points_by_action.setdefault(point.action_index, []).append(point)
        events: list[dict[str, object]] = [{"type": "hand_started", "table": observable.table_view(mode=mode, hero_seat=hero_seat)}]
        for index, record in enumerate(engine_replay["actions"]):
            if state.actor != record["seat"]:
                raise ValueError("replay action order does not match the hand state")
            observable.equity_points.extend(points_by_action.get(index, []))
            state.step(record["action"])
            analysis = raw_analyses[index].get("analysis") if index < len(raw_analyses) and isinstance(raw_analyses[index], dict) else None
            events.append(
                {
                    "type": "action",
                    "seat": record["seat"],
                    "selected_action": record["action"],
                    "analysis": analysis,
                    "policy": observable.policy_for_seat(record["seat"]).as_dict(),
                    "table": observable.table_view(mode=mode, hero_seat=hero_seat),
                }
            )
        events.append({"type": "hand_complete", "table": observable.table_view(mode=mode, hero_seat=hero_seat), "replay": replay})
        return events
