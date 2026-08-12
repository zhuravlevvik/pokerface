"""Player-safe projection of a hand state (never exposes opponent hole cards)."""

from __future__ import annotations

from typing import Any

from .game_state import HandState


def observation_for(state: HandState, seat: int) -> dict[str, Any]:
    player = state.player(seat)
    return {
        "seat": seat,
        "street": state.street.value,
        "hole_cards": [str(card) for card in state.hole_cards[seat]],
        "board": [str(card) for card in state.board],
        "pot": state.pot,
        "current_bet": state.current_bet,
        "to_call": state.to_call(seat),
        "actor": state.actor,
        "players": [
            {
                "seat": other.seat,
                "position": state.positions[other.seat],
                "stack": other.stack,
                "committed_street": other.committed_street,
                "committed_total": other.committed_total,
                "folded": other.folded,
                "all_in": other.all_in,
            }
            for other in state.players
        ],
        "legal_actions": {action.value: allowed for action, allowed in state.legal_actions(seat).items()},
    }
