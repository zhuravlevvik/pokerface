"""Contracts for the reusable common-deal 2/3/5-max evaluation suite."""

from __future__ import annotations

import json

import pytest

from poker.bots import AggroBot, RandomBot, RuleBot, TightBot
from poker.league import ModelPolicy
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.multiway_evaluation import (
    MULTIWAY_EVALUATION_PROTOCOL,
    PAIRED_BLOCK_CI_METHOD,
    MultiwayEvaluationConfig,
    OpponentSeat,
    evaluate_multiway_suite,
    pair_multiway_reports,
)


def _opponents(player_count: int) -> tuple[OpponentSeat, ...]:
    factories = (
        OpponentSeat("baseline:rule", RuleBot),
        OpponentSeat("baseline:tight", TightBot),
        OpponentSeat("baseline:aggro", AggroBot),
        OpponentSeat("baseline:random", RandomBot),
    )
    return factories[: player_count - 1]


@pytest.mark.parametrize("player_count", (2, 3, 5))
def test_common_deal_suite_rotates_candidate_through_all_seats(player_count: int) -> None:
    report = evaluate_multiway_suite(
        "candidate",
        RandomBot(seed=17),
        _opponents(player_count),
        config=MultiwayEvaluationConfig(player_count=player_count, deal_blocks=2, seed_start=700, equity_samples=1),
    )

    assert report.hands == player_count * 2
    assert report.seed_blocks == 2
    assert len(report.block_pnl_bb) == 2
    assert sum(report.block_pnl_bb) * player_count == pytest.approx(report.pnl_bb)
    assert set(report.position_hands.values()) == {2}
    assert report.bb_per_100 == pytest.approx(report.pnl_bb / report.hands * 100)
    assert report.bb_per_100_ci95_low <= report.bb_per_100 <= report.bb_per_100_ci95_high
    assert report.model_diagnostics.decisions == 0
    encoded = report.as_dict()
    assert encoded["protocol"]["name"] == MULTIWAY_EVALUATION_PROTOCOL
    assert encoded["protocol"]["ci_method"] == PAIRED_BLOCK_CI_METHOD
    assert encoded["opponent_slots"] == [item.identity for item in _opponents(player_count)]
    assert set(encoded["seat_assignments"]) == {str(seat) for seat in range(player_count)}
    assert all(list(assignment.values()).count("candidate") == 1 for assignment in encoded["seat_assignments"].values())
    if player_count == 3:
        assert encoded["seat_assignments"]["1"] == {
            "0": "baseline:tight",
            "1": "candidate",
            "2": "baseline:rule",
        }
    json.dumps(encoded, sort_keys=True)


def test_heterogeneous_factories_are_called_per_explicit_non_candidate_slot() -> None:
    calls: list[str] = []

    def factory(name: str, bot_type):
        def create():
            calls.append(name)
            return bot_type()

        return create

    opponents = (
        OpponentSeat("seat-slot:rule", factory("rule", RuleBot)),
        OpponentSeat("seat-slot:aggro", factory("aggro", AggroBot)),
    )
    evaluate_multiway_suite(
        "candidate",
        TightBot(),
        opponents,
        config=MultiwayEvaluationConfig(player_count=3, deal_blocks=2, seed_start=810, equity_samples=1),
    )

    assert calls.count("rule") == 6
    assert calls.count("aggro") == 6


def test_policy_rng_isolation_and_protocol_hash_make_reports_reproducible(tmp_path) -> None:
    candidate = RandomBot(seed=91)
    shared_opponent = RandomBot(seed=92)
    candidate_state = candidate._rng.getstate()
    opponent_state = shared_opponent._rng.getstate()
    opponents = (OpponentSeat("baseline:shared-random", lambda: shared_opponent),)
    config = MultiwayEvaluationConfig(player_count=2, deal_blocks=3, seed_start=900, equity_samples=1)

    first = evaluate_multiway_suite("candidate", candidate, opponents, config=config)
    second = evaluate_multiway_suite("candidate", candidate, opponents, config=config)

    assert candidate._rng.getstate() == candidate_state
    assert shared_opponent._rng.getstate() == opponent_state
    assert first.as_dict() == second.as_dict()
    written = first.write_json(tmp_path / "reports" / "multiway.json")
    assert json.loads(written.read_text(encoding="utf-8"))["protocol_sha256"] == first.protocol_sha256


def test_suite_rejects_missing_or_duplicate_explicit_opponent_seats() -> None:
    config = MultiwayEvaluationConfig(player_count=3, deal_blocks=2, equity_samples=1)
    with pytest.raises(ValueError, match="every non-candidate seat"):
        evaluate_multiway_suite("candidate", TightBot(), (OpponentSeat("rule", RuleBot),), config=config)
    with pytest.raises(ValueError, match="unique"):
        evaluate_multiway_suite(
            "candidate",
            TightBot(),
            (OpponentSeat("same", RuleBot), OpponentSeat("same", TightBot)),
            config=config,
        )


def test_paired_transfer_scratch_helper_uses_common_deal_block_deltas() -> None:
    config = MultiwayEvaluationConfig(player_count=3, deal_blocks=3, seed_start=1_100, equity_samples=1)
    transfer = evaluate_multiway_suite("transfer", TightBot(), _opponents(3), config=config)
    scratch = evaluate_multiway_suite("scratch", RuleBot(), _opponents(3), config=config)
    paired = pair_multiway_reports(transfer, scratch)

    assert transfer.protocol_sha256 == scratch.protocol_sha256
    assert paired.seed_blocks == 3
    assert paired.delta_block_pnl_bb == tuple(
        left - right for left, right in zip(transfer.block_pnl_bb, scratch.block_pnl_bb, strict=True)
    )
    assert paired.delta_bb_per_100_ci95_low <= paired.delta_bb_per_100 <= paired.delta_bb_per_100_ci95_high
    assert paired.as_dict()["protocol_sha256"] == paired.protocol_sha256


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install the project with .[rl]")
def test_model_diagnostics_include_expected_showdown_share_for_multiway() -> None:
    import torch

    torch.manual_seed(12)
    model = PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))
    report = evaluate_multiway_suite(
        "candidate:model",
        ModelPolicy("candidate:model", model),
        _opponents(3),
        config=MultiwayEvaluationConfig(
            player_count=3,
            deal_blocks=2,
            seed_start=1000,
            equity_samples=1,
            calibration_bins=4,
            required_expected_showdown_share_strata=(
                "street=preflop|active_players=3",
                "street=impossible|active_players=3",
            ),
        ),
    )

    diagnostics = report.model_diagnostics
    assert diagnostics.decisions > 0
    assert diagnostics.illegal_action_count == 0
    assert diagnostics.equity is not None
    assert diagnostics.expected_showdown_share is not None
    assert diagnostics.expected_showdown_share.samples == diagnostics.decisions
    assert len(diagnostics.expected_showdown_share.calibration) == 4
    assert diagnostics.expected_showdown_share_support["street=preflop|active_players=3"] > 0
    assert diagnostics.expected_showdown_share_by_stratum["street=preflop|active_players=3"].samples > 0
    assert diagnostics.expected_showdown_share_support["street=impossible|active_players=3"] == 0
    assert "street=impossible|active_players=3" not in diagnostics.expected_showdown_share_by_stratum
