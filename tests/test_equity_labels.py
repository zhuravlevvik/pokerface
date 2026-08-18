"""Tests for virtual-showdown supervision and equity-head diagnostics."""

from __future__ import annotations

import json

import pytest

from poker.cards import Card, Deck
from poker.equity import EquitySnapshot, EquityTarget, capture_equity_snapshot, equity_cross_entropy, equity_metrics, generate_equity_target
from poker.game_state import HandState
from poker.simulator import BatchedHoldemEnvironment


def _cards(*tokens: str) -> tuple[Card, ...]:
    return tuple(Card.parse(token) for token in tokens)


def _river_snapshot(hero: tuple[str, str], opponent: tuple[str, str], board: tuple[str, ...]) -> EquitySnapshot:
    return EquitySnapshot(
        hero_seat=0,
        hero_hole_cards=_cards(*hero),  # type: ignore[arg-type]
        opponent_hole_cards=(_cards(*opponent),),  # type: ignore[arg-type]
        board=_cards(*board),
        remaining_deck=(),
        source_seed=17,
        reference="river",
    )


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (_river_snapshot(("Th", "3d"), ("As", "Ad"), ("Ah", "Kh", "Qh", "Jh", "2c")), (1.0, 0.0, 0.0)),
        (_river_snapshot(("2c", "3d"), ("As", "Ad"), ("Ah", "Kh", "Qh", "Jh", "Th")), (0.0, 1.0, 0.0)),
        (_river_snapshot(("3d", "4s"), ("Th", "9h"), ("Ah", "Kh", "Qh", "Jh", "2c")), (0.0, 0.0, 1.0)),
    ],
)
def test_river_target_is_exact_for_win_tie_and_loss(snapshot: EquitySnapshot, expected: tuple[float, float, float]) -> None:
    target = generate_equity_target(snapshot, samples=32)

    assert target.exact is True
    assert target.samples == 1
    assert target.probabilities == expected


def test_pre_river_target_is_soft_reproducible_and_uses_runouts() -> None:
    state = HandState(seed=31)
    snapshot = capture_equity_snapshot(state, state.actor or 0, reference="preflop-decision")

    first = generate_equity_target(snapshot, samples=16)
    second = generate_equity_target(snapshot, samples=16)

    assert first == second
    assert first.exact is False
    assert first.samples == 16
    assert sum(first.probabilities) == pytest.approx(1.0)
    assert all(value * 16 == pytest.approx(round(value * 16)) for value in first.probabilities)


def test_turn_target_enumerates_exactly_when_budget_covers_all_rivers() -> None:
    known = _cards("Ah", "Ad", "Kc", "Kd", "2c", "3d", "4h", "5s")
    remaining = tuple(card for suit in ("c", "d", "h", "s") for rank in range(2, 15) if (card := Card(rank, suit)) not in known)
    snapshot = EquitySnapshot(
        hero_seat=0,
        hero_hole_cards=known[:2],  # type: ignore[arg-type]
        opponent_hole_cards=(known[2:4],),  # type: ignore[arg-type]
        board=known[4:],
        remaining_deck=remaining,
    )

    target = generate_equity_target(snapshot, samples=len(remaining))

    assert target.exact is True
    assert target.samples == len(remaining) == 44
    assert sum(target.probabilities) == pytest.approx(1.0)


def test_training_trace_exposes_label_but_never_private_label_context() -> None:
    environment = BatchedHoldemEnvironment(1)
    observations = environment.reset(seeds=[44])
    while not all(environment.terminal):
        actions = [
            None if done else ("check" if mask["check"] else "call")
            for done, mask in zip(environment.terminal, environment.legal_action_masks, strict=True)
        ]
        result = environment.step(actions)
        observations = tuple(item for item in result.observations if item is not None)
    trace = result.infos[0]["trace"]
    assert trace is not None
    records = trace.as_training_records()
    encoded = json.dumps(records)

    assert all(record["equity_target"] is not None for record in records)
    assert all(sum(record["equity_target"]) == pytest.approx(1.0) for record in records)
    assert "remaining_deck" not in encoded
    assert "opponent_hole_cards" not in encoded
    for record, decision in zip(records, trace.decisions, strict=True):
        assert decision.equity_snapshot_reference is not None
        snapshot = trace._equity_snapshots[decision.equity_snapshot_reference]
        record_json = json.dumps(record)
        for cards in snapshot.opponent_hole_cards:
            for card in cards:
                assert str(card) not in record_json
    for decision in trace.decisions:
        assert decision.observation["cards"]["hole_cards"]
        assert all("hole_cards" not in player for player in decision.observation["players"])


def test_equity_metrics_reports_perfectly_calibrated_holdout() -> None:
    predictions = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    targets = [EquityTarget(*row, samples=1, exact=True) for row in predictions]

    report = equity_metrics(predictions, targets, bins=5)

    assert report.samples == 3
    assert report.logloss == pytest.approx(0.0)
    assert report.brier_score == pytest.approx(0.0)
    assert report.expected_calibration_error == pytest.approx(0.0)
    assert sum(bin.count for bin in report.calibration) == 3
    assert report.as_dict()["samples"] == 3


def test_equity_metrics_uses_displayed_win_plus_half_tie_for_calibration() -> None:
    report = equity_metrics([(0.4, 0.6, 0.0)], [(0.4, 0.6, 0.0)], bins=10)

    assert report.expected_calibration_error == pytest.approx(0.0)
    populated = next(item for item in report.calibration if item.count)
    assert populated.mean_prediction == pytest.approx(0.7)
    assert populated.mean_target == pytest.approx(0.7)


def test_equity_cross_entropy_accepts_soft_targets_when_torch_is_installed() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.25, 0.25]])

    loss = equity_cross_entropy(logits, targets)
    loss.backward()

    assert loss.item() == pytest.approx(1.0986123)
    assert logits.grad is not None


def test_snapshot_rejects_duplicate_private_cards() -> None:
    ace_hearts = Card.parse("Ah")
    with pytest.raises(ValueError, match="distinct"):
        EquitySnapshot(
            hero_seat=0,
            hero_hole_cards=(ace_hearts, Card.parse("Kd")),
            opponent_hole_cards=((ace_hearts, Card.parse("Qs")),),
            board=(),
            remaining_deck=Deck(1).snapshot(),
        )
