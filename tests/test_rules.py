from fractions import Fraction

from poker.betting import Action
from poker.rules import BIG_BLIND, positions, postflop_order, preflop_order, raise_to_for, round_half_up


def test_positions_and_action_order_rotate_from_button() -> None:
    assert positions(3) == {3: "BTN", 4: "SB", 0: "BB", 1: "UTG", 2: "CO"}
    assert preflop_order(3) == (1, 2, 3, 4, 0)
    assert postflop_order(3) == (4, 0, 1, 2, 3)


def test_discrete_raise_sizes_and_half_up_rounding() -> None:
    assert round_half_up(Fraction(1, 2)) == 1
    assert round_half_up(Fraction(3, 2)) == 2
    common = dict(current_bet=100, to_call=50, pot_before=250, last_full_raise=100)
    assert raise_to_for(Action.RAISE_MIN, **common) == 200
    assert raise_to_for(Action.RAISE_1_3_POT, **common) == 200
    assert raise_to_for(Action.RAISE_1_2_POT, **common) == 250
    assert raise_to_for(Action.RAISE_3_4_POT, **common) == 325
    assert raise_to_for(Action.RAISE_POT, **common) == 400
    assert raise_to_for(Action.RAISE_1_5_POT, **common) == 550
    assert BIG_BLIND == 100
