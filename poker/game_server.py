"""Observable single-hand game server, independent of the web transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from .betting import Action
from .game_state import HandState
from .inference import DecisionService, HeuristicInferenceService, InferenceResponse, validate_response
from .observation import observation_for
from .traces import rebuild_hand

ViewerMode = Literal["player", "spectator"]


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

    def browser_replay(self, *, reveal_hole_cards: bool) -> dict[str, object]:
        """Engine replay plus UI-only explanations needed for an exact replay."""

        replay = self.state.replay(reveal_hole_cards=reveal_hole_cards)
        replay["equity_points"] = [point.as_dict() for point in self.equity_points]
        replay["analyses"] = list(self.analyses)
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
            self.equity_points.append(
                EquityPoint(
                    street=self.state.street.value,
                    seat=seat,
                    equity=response.equity["total"],
                    action_index=len(self.state.action_history),
                )
            )
            self.analyses.append({"seat": seat, "analysis": response.as_dict()})
            self.state.step(Action(response.action))
            yield {
                "type": "action",
                "seat": seat,
                "selected_action": response.action,
                "analysis": response.as_dict(),
                "table": self.table_view(mode=mode, hero_seat=hero_seat),
            }
        yield {
            "type": "hand_complete",
            "table": self.table_view(mode=mode, hero_seat=hero_seat),
            "replay": self.browser_replay(reveal_hole_cards=mode == "spectator"),
        }


class GameServer:
    """Creates observable games; no trainer or checkpoint mutation lives here."""

    def __init__(self, inference: DecisionService | None = None) -> None:
        self.inference = inference or HeuristicInferenceService()

    def observe_hand(
        self,
        *,
        seed: int | None = None,
        button_seat: int = 0,
        starting_stack: int = 10_000,
        player_count: int = 5,
        hero_seat: int = 0,
        mode: ViewerMode = "player",
    ) -> list[dict[str, object]]:
        state = HandState(seed=seed, button_seat=button_seat, starting_stack=starting_stack, player_count=player_count)
        return list(ObservableHand(state, self.inference).play(mode=mode, hero_seat=hero_seat))

    def replay_hand(self, replay: dict[str, Any], *, hero_seat: int = 0, mode: ViewerMode = "player") -> list[dict[str, object]]:
        """Validate an imported replay then deterministically reconstruct its UI events."""

        engine_replay = {key: value for key, value in replay.items() if key not in {"equity_points", "analyses"}}
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
        observable = ObservableHand(state, self.inference)
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
                    "table": observable.table_view(mode=mode, hero_seat=hero_seat),
                }
            )
        events.append({"type": "hand_complete", "table": observable.table_view(mode=mode, hero_seat=hero_seat), "replay": replay})
        return events
