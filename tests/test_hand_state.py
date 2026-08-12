from poker.betting import Action, PlayerState, build_pots
from poker.game_state import HandState, Street
from poker.cards import Card


def play_to_terminal_with_checks_or_calls(state: HandState) -> HandState:
    while not state.complete:
        mask = state.legal_actions()
        state.step(Action.CHECK if mask[Action.CHECK] else Action.CALL)
    return state


def test_same_seed_and_actions_produce_identical_replay() -> None:
    first = play_to_terminal_with_checks_or_calls(HandState(seed=123, button_seat=2))
    second = play_to_terminal_with_checks_or_calls(HandState(seed=123, button_seat=2))
    assert first.replay() == second.replay()
    assert len(first.board) == 5
    assert sum(player.stack for player in first.players) == 50_000


def test_preflop_and_postflop_action_order() -> None:
    state = HandState(seed=1, button_seat=0)
    assert state.actor == 3  # UTG
    for expected in (3, 4, 0, 1):
        assert state.actor == expected
        state.step(Action.CALL)
    assert state.actor == 2  # BB may check when unraised
    state.step(Action.CHECK)
    assert state.street == Street.FLOP
    for expected in (1, 2, 3, 4, 0):
        assert state.actor == expected
        state.step(Action.CHECK)
    assert state.street == Street.TURN


def test_action_mask_and_min_raise() -> None:
    state = HandState(seed=2)
    mask = state.legal_actions()
    assert mask[Action.CALL] and mask[Action.FOLD]
    assert not mask[Action.CHECK]
    state.step(Action.RAISE_MIN)
    assert state.current_bet == 200
    assert state.last_full_raise == 100
    mask = state.legal_actions()
    assert mask[Action.CALL] and mask[Action.FOLD] and mask[Action.RAISE_MIN]
    assert not mask[Action.CHECK]


def test_short_all_in_raise_does_not_reopen_prior_actor() -> None:
    state = HandState(seed=3, starting_stack=300)
    # UTG raises to 200; CO calls. BTN has only 300 total and can short raise to 300.
    state.step(Action.RAISE_MIN)  # seat 3
    state.step(Action.CALL)  # seat 4
    state.step(Action.ALL_IN)  # BTN seat 0: target 300, short raise versus last full 100
    assert state.actor == 1
    state.step(Action.CALL)  # SB
    state.step(Action.CALL)  # BB
    assert state.actor == 3  # UTG already acted after the last full raise
    mask = state.legal_actions()
    assert mask[Action.CALL] and mask[Action.FOLD]
    assert not mask[Action.RAISE_MIN]


def test_uncontested_hand_returns_uncalled_bet_then_awards_pot() -> None:
    state = HandState(seed=4)
    state.step(Action.RAISE_MIN)  # UTG invests 200 total
    state.step(Action.FOLD)
    state.step(Action.FOLD)
    state.step(Action.FOLD)
    state.step(Action.FOLD)
    assert state.complete
    assert state.player(3).stack == 10_150  # 10000 - 200 + returned 100 + blinds
    assert state.returned_chips == [{"seat": 3, "amount": 100}]
    assert sum(player.stack for player in state.players) == 50_000


def test_side_pot_layers_include_folded_contributions() -> None:
    players = [
        PlayerState(0, 0, committed_total=100),
        PlayerState(1, 0, committed_total=200),
        PlayerState(2, 0, committed_total=200, folded=True),
        PlayerState(3, 0, committed_total=300),
        PlayerState(4, 0, committed_total=300),
    ]
    assert [(pot.amount, pot.eligible, pot.level) for pot in build_pots(players)] == [
        # main: 5 * 100, then 4 * 100 and 2 * 100
        (500, (0, 1, 3, 4), 100),
        (400, (1, 3, 4), 200),
        (200, (3, 4), 300),
    ]


def _set_contribution(player: PlayerState, amount: int, *, folded: bool = False) -> None:
    player.stack = 10_000 - amount
    player.committed_total = amount
    player.committed_street = amount
    player.folded = folded
    player.all_in = False


def test_showdown_distributes_multiple_side_pots() -> None:
    state = HandState(seed=5)
    for player, contribution, folded in zip(state.players, (100, 200, 200, 300, 300), (False, False, True, False, False)):
        _set_contribution(player, contribution, folded=folded)
    state.board = [Card.parse(token) for token in "Ah Kd 2c 3d 4h".split()]
    state.hole_cards = {
        0: tuple(Card.parse(token) for token in "As Ad".split()),
        1: tuple(Card.parse(token) for token in "Kh Ks".split()),
        2: tuple(Card.parse(token) for token in "5c 6c".split()),
        3: tuple(Card.parse(token) for token in "Qh Qs".split()),
        4: tuple(Card.parse(token) for token in "Jh Js".split()),
    }
    state._showdown()
    assert state.payouts == {0: 500, 1: 400, 2: 0, 3: 200, 4: 0}
    assert sum(player.stack for player in state.players) == 50_000


def test_split_pot_odd_chip_goes_clockwise_from_button() -> None:
    state = HandState(seed=6)
    _set_contribution(state.player(0), 1)
    _set_contribution(state.player(1), 1)
    _set_contribution(state.player(2), 1, folded=True)
    for seat in (3, 4):
        _set_contribution(state.player(seat), 0, folded=True)
    state.board = [Card.parse(token) for token in "As Ks Qs Js Ts".split()]
    state.hole_cards = {seat: tuple(Card.parse(token) for token in pair.split()) for seat, pair in {
        0: "2c 3c", 1: "4c 5c", 2: "6c 7c", 3: "8c 9c", 4: "2d 3d"
    }.items()}
    state._showdown()
    assert state.payouts[0] == 1
    assert state.payouts[1] == 2  # seat 1 is first clockwise winner after BTN=0


def test_random_legal_hands_preserve_cards_and_chips() -> None:
    for seed in range(10_000):
        state = HandState(seed=seed)
        while not state.complete:
            mask = state.legal_actions()
            # Deterministic varied legal choice, deliberately including raises/folds.
            legal = [action for action, allowed in mask.items() if allowed]
            state.step(legal[(seed + len(state.action_history)) % len(legal)])
        assert sum(player.stack for player in state.players) == 50_000
        visible = [card for hole in state.hole_cards.values() for card in hole] + state.board
        assert len(visible) == len(set(visible))
