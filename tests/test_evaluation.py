"""Contracts for the fixed, promotion-safe evaluation suite."""

from __future__ import annotations

import json

import pytest

from poker.bots import AggroBot, RandomBot, RuleBot, TightBot
from poker.evaluation import EvaluationConfig, evaluate_suite
from poker.league import ModelPolicy
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel


def test_suite_uses_identical_fixed_deals_and_exact_position_rotation(tmp_path) -> None:
    config = EvaluationConfig(hands_per_opponent=10, seed_start=500, equity_samples=1)
    first = evaluate_suite("candidate", RandomBot(seed=17), {"tight": TightBot(), "rule": RuleBot()}, config=config)
    second = evaluate_suite("candidate", RandomBot(seed=17), {"tight": TightBot(), "rule": RuleBot()}, config=config)

    assert first.as_dict() == second.as_dict()
    assert len(first.matchups) == 2
    for matchup in first.matchups:
        assert matchup.position_hands == {"BTN": 2, "SB": 2, "BB": 2, "UTG": 2, "CO": 2}
        assert matchup.statistics.hands == 10
        assert matchup.bb_per_100 == pytest.approx(matchup.pnl_bb * 10)
        assert matchup.league_score == pytest.approx(matchup.win_rate + 0.5 * matchup.tie_rate)
    output = first.write_json(tmp_path / "reports" / "candidate.json")
    encoded = json.loads(output.read_text())
    assert encoded["schema_version"] == "1.0"
    assert 0.0 <= encoded["aggregate_league_score"] <= 1.0
    assert encoded["matchups"][0]["statistics"]["vpip"] >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")
def test_model_suite_reports_all_required_diagnostics_and_sanity_checks() -> None:
    import torch

    torch.manual_seed(3)
    model = PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))
    report = evaluate_suite(
        "candidate",
        ModelPolicy("candidate", model),
        {"aggro": AggroBot(seed=4)},
        config=EvaluationConfig(hands_per_opponent=5, seed_start=900, equity_samples=1, calibration_bins=5),
    )

    matchup = report.matchups[0]
    assert matchup.model_diagnostics.decisions > 0
    assert matchup.model_diagnostics.policy_entropy is not None
    assert matchup.model_diagnostics.value_mae_bb is not None
    assert matchup.model_diagnostics.masked_action_rate == 0.0
    assert matchup.equity is not None
    assert matchup.equity.samples == matchup.model_diagnostics.decisions
    assert len(matchup.equity.calibration) == 5
    assert {scenario.name for scenario in report.sanity_checks} == {
        "nuts_river",
        "lost_river",
        "flush_draw",
        "favourable_pot_odds",
        "unfavourable_pot_odds",
        "short_stack_all_in",
    }
    assert all(item.legal and item.finite for item in report.sanity_checks)
