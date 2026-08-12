"""Authoritative deterministic state machine for a single 5-max NLHE hand."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .betting import Action, PlayerState, RAISE_ACTIONS, Pot, build_pots
from .cards import Card, Deck
from .evaluator import HandRank, evaluate
from .rules import BIG_BLIND, SEAT_COUNT, SMALL_BLIND, positions, postflop_order, preflop_order, raise_to_for


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ActionRecord:
    street: Street
    seat: int
    action: Action
    pot_before: int
    amount: int
    raise_to: int | None
    current_bet_after: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "street": self.street.value,
            "seat": self.seat,
            "action": self.action.value,
            "pot_before": self.pot_before,
            "amount": self.amount,
            "raise_to": self.raise_to,
            "current_bet_after": self.current_bet_after,
        }


@dataclass
class HandState:
    """One complete 2--5 player cash-game hand; money is integer chips."""

    seed: int | None = None
    button_seat: int = 0
    starting_stack: int = 10_000
    player_count: int = SEAT_COUNT
    allowed_raise_actions: frozenset[Action] | None = None
    deck: Deck = field(init=False)
    players: list[PlayerState] = field(init=False)
    hole_cards: dict[int, tuple[Card, Card]] = field(init=False)
    board: list[Card] = field(default_factory=list, init=False)
    street: Street = field(default=Street.PREFLOP, init=False)
    current_bet: int = field(default=BIG_BLIND, init=False)
    last_full_raise: int = field(default=BIG_BLIND, init=False)
    actor: int | None = field(init=False)
    action_history: list[ActionRecord] = field(default_factory=list, init=False)
    payouts: dict[int, int] = field(default_factory=dict, init=False)
    returned_chips: list[dict[str, int]] = field(default_factory=list, init=False)
    _awaiting: list[int] = field(default_factory=list, init=False)
    _raise_right: dict[int, bool] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 2 <= self.player_count <= SEAT_COUNT:
            raise ValueError(f"player_count must be in 2..{SEAT_COUNT}")
        if not 0 <= self.button_seat < self.player_count:
            raise ValueError(f"button seat must be in 0..{self.player_count - 1}")
        if self.starting_stack < BIG_BLIND:
            raise ValueError("starting stack must cover the big blind")
        if self.allowed_raise_actions is not None and not self.allowed_raise_actions.issubset(RAISE_ACTIONS | {Action.ALL_IN}):
            raise ValueError("allowed_raise_actions contains a non-raise action")
        self.deck = Deck(self.seed)
        self.players = [PlayerState(seat=seat, stack=self.starting_stack) for seat in range(self.player_count)]
        self.hole_cards = {seat: tuple(self.deck.deal(2)) for seat in range(self.player_count)}
        self.payouts = {seat: 0 for seat in range(self.player_count)}
        if self.player_count == 2:
            self._commit(self.button_seat, SMALL_BLIND)
            self._commit((self.button_seat + 1) % self.player_count, BIG_BLIND)
        else:
            self._commit((self.button_seat + 1) % self.player_count, SMALL_BLIND)
            self._commit((self.button_seat + 2) % self.player_count, BIG_BLIND)
        self._start_betting_round(preflop_order(self.button_seat, player_count=self.player_count), reset_bet=False)
        self._assert_invariants()

    @property
    def pot(self) -> int:
        return sum(player.committed_total for player in self.players)

    @property
    def positions(self) -> dict[int, str]:
        return positions(self.button_seat, player_count=self.player_count)

    @property
    def complete(self) -> bool:
        return self.street == Street.COMPLETE

    def player(self, seat: int) -> PlayerState:
        return self.players[seat]

    def to_call(self, seat: int | None = None) -> int:
        seat = self.actor if seat is None else seat
        if seat is None:
            return 0
        return max(0, self.current_bet - self.player(seat).committed_street)

    def legal_actions(self, seat: int | None = None) -> dict[Action, bool]:
        """Return a complete action mask for the current actor only."""

        seat = self.actor if seat is None else seat
        result = {action: False for action in Action}
        if self.complete or seat is None or seat != self.actor:
            return result
        player = self.player(seat)
        if player.folded or player.all_in:
            return result
        to_call = self.to_call(seat)
        result[Action.FOLD] = to_call > 0
        result[Action.CHECK] = to_call == 0
        result[Action.CALL] = to_call > 0
        all_in_to = player.committed_street + player.stack
        result[Action.ALL_IN] = (self.allowed_raise_actions is None or Action.ALL_IN in self.allowed_raise_actions) and player.stack > to_call and all_in_to > self.current_bet
        if not self._raise_right[seat]:
            return result
        permitted = RAISE_ACTIONS if self.allowed_raise_actions is None else RAISE_ACTIONS.intersection(self.allowed_raise_actions)
        for action in permitted:
            target = self.raise_to(action, seat)
            contribution = target - player.committed_street
            # Exact stack sized bets use the all-in id, avoiding duplicate actions.
            result[action] = contribution < player.stack and target > self.current_bet
        return result

    def raise_to(self, action: Action, seat: int | None = None) -> int:
        seat = self.actor if seat is None else seat
        if seat is None:
            raise RuntimeError("hand has no actor")
        return raise_to_for(
            action,
            current_bet=self.current_bet,
            to_call=self.to_call(seat),
            pot_before=self.pot,
            last_full_raise=self.last_full_raise,
        )

    def step(self, action: Action | str) -> None:
        """Apply one legal discrete action by the current actor."""

        if isinstance(action, str):
            action = Action(action)
        if self.actor is None or self.complete:
            raise RuntimeError("hand is already complete")
        if not self.legal_actions()[action]:
            raise ValueError(f"illegal action {action.value} for seat {self.actor}")
        seat = self.actor
        player = self.player(seat)
        old_current = self.current_bet
        pot_before = self.pot
        amount = 0
        target: int | None = None
        if action == Action.FOLD:
            player.folded = True
        elif action == Action.CHECK:
            pass
        elif action == Action.CALL:
            amount = min(self.to_call(seat), player.stack)
            self._commit(seat, amount)
        else:
            target = player.committed_street + player.stack if action == Action.ALL_IN else self.raise_to(action, seat)
            amount = target - player.committed_street
            self._commit(seat, amount)
            self.current_bet = target
        self._raise_right[seat] = False
        self.action_history.append(ActionRecord(self.street, seat, action, pot_before, amount, target, self.current_bet))
        self._after_action(seat, old_current)
        self._assert_invariants()

    def _after_action(self, seat: int, old_current: int) -> None:
        live = self._live_seats()
        if len(live) == 1:
            self._refund_uncalled_street_bet()
            self._award_uncontested(live[0])
            return
        raised = self.current_bet > old_current
        full_raise = raised and self.current_bet - old_current >= self.last_full_raise
        if full_raise:
            self.last_full_raise = self.current_bet - old_current
            self._raise_right = {candidate: candidate != seat and self._can_act(candidate) for candidate in range(self.player_count)}
        # A bet/raise requires every undercalled player to respond. A short
        # all-in intentionally does not reopen their raise rights.
        if raised:
            self._awaiting = [candidate for candidate in self._ordered_after(seat) if self._can_act(candidate) and self.player(candidate).committed_street < self.current_bet]
        else:
            self._awaiting = [candidate for candidate in self._awaiting if candidate != seat and self._can_act(candidate)]
        self._select_next_or_advance()

    def _start_betting_round(self, order: tuple[int, ...], *, reset_bet: bool) -> None:
        if reset_bet:
            for player in self.players:
                player.committed_street = 0
            self.current_bet = 0
            self.last_full_raise = BIG_BLIND
        self._raise_right = {seat: self._can_act(seat) for seat in range(self.player_count)}
        self._awaiting = [seat for seat in order if self._can_act(seat)]
        self._select_next_or_advance()

    def _select_next_or_advance(self) -> None:
        while self._awaiting and not self._can_act(self._awaiting[0]):
            self._awaiting.pop(0)
        if self._awaiting:
            self.actor = self._awaiting[0]
            return
        self.actor = None
        self._finish_street()

    def _finish_street(self) -> None:
        self._refund_uncalled_street_bet()
        live = self._live_seats()
        if len(live) == 1:
            self._award_uncontested(live[0])
            return
        if all(self.player(seat).all_in for seat in live):
            self._runout_and_showdown()
            return
        if self.street == Street.PREFLOP:
            self.street = Street.FLOP
            self.board.extend(self.deck.deal(3))
        elif self.street == Street.FLOP:
            self.street = Street.TURN
            self.board.extend(self.deck.deal(1))
        elif self.street == Street.TURN:
            self.street = Street.RIVER
            self.board.extend(self.deck.deal(1))
        elif self.street == Street.RIVER:
            self._showdown()
            return
        else:
            raise RuntimeError(f"cannot finish street {self.street}")
        self._start_betting_round(postflop_order(self.button_seat, player_count=self.player_count), reset_bet=True)

    def _runout_and_showdown(self) -> None:
        if self.street == Street.PREFLOP:
            self.board.extend(self.deck.deal(3))
            self.street = Street.FLOP
        if self.street == Street.FLOP:
            self.board.extend(self.deck.deal(1))
            self.street = Street.TURN
        if self.street == Street.TURN:
            self.board.extend(self.deck.deal(1))
            self.street = Street.RIVER
        self._showdown()

    def _showdown(self) -> None:
        self._refund_uncalled_street_bet()
        self.street = Street.SHOWDOWN
        pots = build_pots(self.players)
        ranks: dict[int, HandRank] = {seat: evaluate((*self.hole_cards[seat], *self.board)) for seat in self._live_seats()}
        for pot in pots:
            if not pot.eligible:
                raise AssertionError("a showdown pot must have an eligible player")
            best = max(ranks[seat] for seat in pot.eligible)
            winners = [seat for seat in pot.eligible if ranks[seat] == best]
            share, remainder = divmod(pot.amount, len(winners))
            for winner in winners:
                self.player(winner).stack += share
                self.payouts[winner] += share
            for winner in self._clockwise_from_button(winners)[:remainder]:
                self.player(winner).stack += 1
                self.payouts[winner] += 1
        self._clear_commitments()
        self.street = Street.COMPLETE
        self.actor = None

    def _award_uncontested(self, winner: int) -> None:
        amount = self.pot
        self.player(winner).stack += amount
        self.payouts[winner] += amount
        self._clear_commitments()
        self.street = Street.COMPLETE
        self.actor = None

    def _refund_uncalled_street_bet(self) -> None:
        contributions = sorted((player.committed_street, player.seat) for player in self.players)
        highest, seat = contributions[-1]
        second = contributions[-2][0]
        if highest > second:
            refund = highest - second
            player = self.player(seat)
            player.stack += refund
            player.committed_street -= refund
            player.committed_total -= refund
            player.all_in = player.stack == 0
            self.current_bet = max(other.committed_street for other in self.players)
            self.returned_chips.append({"seat": seat, "amount": refund})

    def _commit(self, seat: int, amount: int) -> None:
        player = self.player(seat)
        if amount < 0 or amount > player.stack:
            raise ValueError("invalid chip commitment")
        player.stack -= amount
        player.committed_total += amount
        player.committed_street += amount
        player.all_in = player.stack == 0

    def _clear_commitments(self) -> None:
        for player in self.players:
            player.committed_total = 0
            player.committed_street = 0

    def _live_seats(self) -> list[int]:
        return [player.seat for player in self.players if not player.folded]

    def _can_act(self, seat: int) -> bool:
        player = self.player(seat)
        return not player.folded and not player.all_in

    def _ordered_after(self, seat: int) -> list[int]:
        return [((seat + offset) % self.player_count) for offset in range(1, self.player_count)]

    def _clockwise_from_button(self, candidates: list[int]) -> list[int]:
        candidate_set = set(candidates)
        return [((self.button_seat + offset) % self.player_count) for offset in range(1, self.player_count + 1) if ((self.button_seat + offset) % self.player_count) in candidate_set]

    def _assert_invariants(self) -> None:
        all_cards = [card for cards in self.hole_cards.values() for card in cards] + self.board + list(self.deck.snapshot())
        assert len(all_cards) == 52 and len(set(all_cards)) == 52, "cards must partition the deck"
        assert all(player.stack >= 0 and player.committed_total >= 0 and player.committed_street >= 0 for player in self.players)
        if self.complete:
            assert sum(player.stack for player in self.players) == self.starting_stack * self.player_count
        else:
            assert sum(player.stack for player in self.players) + self.pot == self.starting_stack * self.player_count

    def replay(self, *, reveal_hole_cards: bool = True) -> dict[str, Any]:
        """Stable, JSON-serialisable audit record for this hand."""

        players = []
        for player in self.players:
            item: dict[str, Any] = {
                "seat": player.seat,
                "position": self.positions[player.seat],
                "stack": player.stack,
                "committed_total": player.committed_total,
                "folded": player.folded,
                "all_in": player.all_in,
                "payout": self.payouts[player.seat],
            }
            if reveal_hole_cards:
                item["hole_cards"] = [str(card) for card in self.hole_cards[player.seat]]
            players.append(item)
        return {
            "seed": self.seed,
            "button_seat": self.button_seat,
            "starting_stack": self.starting_stack,
            "player_count": self.player_count,
            "allowed_raise_actions": None if self.allowed_raise_actions is None else sorted(action.value for action in self.allowed_raise_actions),
            "street": self.street.value,
            "board": [str(card) for card in self.board],
            "players": players,
            "actions": [record.as_dict() for record in self.action_history],
            "returned_chips": list(self.returned_chips),
            "pot": self.pot,
        }
