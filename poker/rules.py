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


def positions(button_seat: int) -> dict[int, str]:
    if not 0 <= button_seat < SEAT_COUNT:
        raise ValueError("button seat must be in 0..4")
    names = ("BTN", "SB", "BB", "UTG", "CO")
    return {(button_seat + offset) % SEAT_COUNT: name for offset, name in enumerate(names)}


def preflop_order(button_seat: int) -> tuple[int, ...]:
    return tuple((button_seat + offset) % SEAT_COUNT for offset in (3, 4, 0, 1, 2))


def postflop_order(button_seat: int) -> tuple[int, ...]:
    return tuple((button_seat + offset) % SEAT_COUNT for offset in (1, 2, 3, 4, 0))


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
