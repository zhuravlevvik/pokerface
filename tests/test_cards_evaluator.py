from poker.cards import Card, Deck
from poker.evaluator import evaluate, evaluate_five


def cards(tokens: str):
    return [Card.parse(token) for token in tokens.split()]


def test_deck_seed_is_reproducible_and_has_all_cards() -> None:
    first = Deck(42).deal(52)
    second = Deck(42).deal(52)
    assert first == second
    assert len(set(first)) == 52


def test_every_hand_category_is_recognised() -> None:
    cases = {
        "high_card": "As Kd 9h 6c 3s",
        "one_pair": "As Ad 9h 6c 3s",
        "two_pair": "As Ad 9h 9c 3s",
        "three_of_a_kind": "As Ad Ah 9c 3s",
        "straight": "9s 8d 7h 6c 5s",
        "flush": "As Js 9s 6s 3s",
        "full_house": "As Ad Ah 9c 9s",
        "four_of_a_kind": "As Ad Ah Ac 9s",
        "straight_flush": "9s 8s 7s 6s 5s",
    }
    assert {evaluate_five(cards(source)).name for source in cases.values()} == set(cases)
    for expected, source in cases.items():
        assert evaluate_five(cards(source)).name == expected


def test_evaluator_handles_wheel_kickers_and_best_of_seven() -> None:
    assert evaluate_five(cards("As 2d 3h 4c 5s")).tiebreak == (5,)
    assert evaluate_five(cards("As Ad Kd 9h 6c")) > evaluate_five(cards("Ks Kd Qd 9h 6c"))
    assert evaluate(cards("As Ad Kh Qc Jd Ts 2c")).name == "straight"
    assert evaluate(cards("As Ad Kh Qc Jd Ts 2c")).tiebreak == (14,)


def test_tie_breakers_within_each_hand_class() -> None:
    stronger = [
        ("As Kd 9h 6c 3s", "Ks Qd 9h 6c 3s"),  # high card
        ("As Ad Kh 6c 3s", "As Ad Qh 6c 3s"),  # pair kicker
        ("As Ad Kh Kc 3s", "As Ad Qh Qc Ks"),  # two pair
        ("As Ad Ah Kc 3s", "Ks Kd Kh Qc 3s"),  # trips
        ("Ts 9d 8h 7c 6s", "9s 8d 7h 6c 5s"),  # straight
        ("As Js 9s 6s 3s", "Ks Js 9s 6s 3s"),  # flush
        ("As Ad Ah Ks Kd", "Ks Kd Kh As Ad"),  # full house
        ("As Ad Ah Ac Ks", "Ks Kd Kh Kc As"),  # quads
        ("Ts 9s 8s 7s 6s", "9s 8s 7s 6s 5s"),  # straight flush
    ]
    for winner, loser in stronger:
        assert evaluate_five(cards(winner)) > evaluate_five(cards(loser))
