"""Pure rule helpers for seats, action order and discrete raise sizing."""

from __future__ import annotations

from fractions import Fraction

from .betting import Action

SEAT_COUNT = 5
CHIPS_PER_BB = 100
SMALL_BLIND = 50
BIG_BLIND = 100

_RAISE_FACTORS = {
    Action.RAISE_1_3_POT: Fraction(1, 3),
    Action.RAISE_1_2_POT: Fraction(1, 2),
    Action.RAISE_3_4_POT: Fraction(3, 4),
    Action.RAISE_POT: Fraction(1, 1),
    Action.RAISE_1_5_POT: Fraction(3, 2),
}


def clockwise_after(seat: int, count: int = SEAT_COUNT) -> int:
    return (seat + 1) % count


def _validate_player_count(player_count: int) -> None:
    if not 2 <= player_count <= SEAT_COUNT:
        raise ValueError(f"player_count must be in 2..{SEAT_COUNT}")


def positions(button_seat: int, *, player_count: int = SEAT_COUNT) -> dict[int, str]:
    """Return position names for a contiguous 2--5 player cash table.

    Heads-up intentionally follows the standard convention where BTN posts the
    small blind and acts first preflop.  Seats are renumbered ``0..N-1`` for a
    short-handed hand; no invisible 5-max placeholders participate in rules.
    """

    _validate_player_count(player_count)
    if not 0 <= button_seat < player_count:
        raise ValueError(f"button seat must be in 0..{player_count - 1}")
    names = {
        2: ("BTN", "BB"),
        3: ("BTN", "SB", "BB"),
        4: ("BTN", "SB", "BB", "UTG"),
        5: ("BTN", "SB", "BB", "UTG", "CO"),
    }[player_count]
    return {(button_seat + offset) % player_count: name for offset, name in enumerate(names)}


def preflop_order(button_seat: int, *, player_count: int = SEAT_COUNT) -> tuple[int, ...]:
    _validate_player_count(player_count)
    if not 0 <= button_seat < player_count:
        raise ValueError(f"button seat must be in 0..{player_count - 1}")
    # BTN is first in heads-up; otherwise action starts to the BB's left.
    start_offset = 0 if player_count == 2 else 3
    return tuple((button_seat + start_offset + offset) % player_count for offset in range(player_count))


def postflop_order(button_seat: int, *, player_count: int = SEAT_COUNT) -> tuple[int, ...]:
    _validate_player_count(player_count)
    if not 0 <= button_seat < player_count:
        raise ValueError(f"button seat must be in 0..{player_count - 1}")
    return tuple((button_seat + offset) % player_count for offset in range(1, player_count + 1))


def round_half_up(value: Fraction) -> int:
    """Round a non-negative rational to integer chips, ties away from zero."""

    if value < 0:
        raise ValueError("raise sizing is non-negative")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def raise_to_for(action: Action, *, current_bet: int, to_call: int, pot_before: int, last_full_raise: int) -> int:
    """Translate a discrete sizing action into final street contribution."""

    if action == Action.RAISE_MIN:
        return current_bet + last_full_raise
    try:
        factor = _RAISE_FACTORS[action]
    except KeyError as exc:
        raise ValueError(f"{action.value} is not a sized raise") from exc
    desired = current_bet + round_half_up(factor * (pot_before + to_call))
    return max(desired, current_bet + last_full_raise)
