"""Contracts for the fixed, promotion-safe evaluation suite."""

from __future__ import annotations

import json

import pytest

from poker.betting import Action
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
    assert encoded["schema_version"] == "1.2"
    assert 0.0 <= encoded["aggregate_league_score"] <= 1.0
    assert encoded["matchups"][0]["statistics"]["vpip"] >= 0
    assert encoded["matchups"][0]["bb_per_100_ci95_low"] <= encoded["matchups"][0]["bb_per_100"]
    assert encoded["matchups"][0]["bb_per_100_ci95_high"] >= encoded["matchups"][0]["bb_per_100"]


@pytest.mark.parametrize(
    ("player_count", "hands", "expected_positions"),
    [
        (2, 6, {"BTN": 3, "BB": 3}),
        (3, 6, {"BTN": 2, "SB": 2, "BB": 2}),
        (5, 5, {"BTN": 1, "SB": 1, "BB": 1, "UTG": 1, "CO": 1}),
    ],
)
def test_suite_is_stage_aware_and_rotates_every_short_handed_position(
    player_count: int,
    hands: int,
    expected_positions: dict[str, int],
) -> None:
    report = evaluate_suite(
        "candidate",
        TightBot(),
        {"rule": RuleBot()},
        config=EvaluationConfig(
            hands_per_opponent=hands,
            seed_start=1_500,
            equity_samples=1,
            player_count=player_count,
        ),
    )
    matchup = report.matchups[0]
    assert matchup.position_hands == expected_positions
    assert matchup.hands == hands


def test_suite_rejects_unbalanced_position_budget() -> None:
    with pytest.raises(ValueError, match="multiple of player_count"):
        EvaluationConfig(hands_per_opponent=5, player_count=3)


def test_suite_serializes_and_enforces_stage_raise_abstraction(tmp_path) -> None:
    allowed = (Action.RAISE_MIN, Action.RAISE_1_2_POT, Action.RAISE_POT, Action.ALL_IN)
    report = evaluate_suite(
        "candidate",
        TightBot(),
        {"rule": RuleBot()},
        config=EvaluationConfig(
            hands_per_opponent=4,
            seed_start=1_600,
            equity_samples=1,
            player_count=2,
            allowed_raise_actions=allowed,
        ),
    )
    encoded = json.loads(report.write_json(tmp_path / "hu-stage-a.json").read_text())
    assert encoded["config"]["allowed_raise_actions"] == [action.value for action in allowed]


def test_paired_position_seeds_repeat_each_deal_and_report_block_ci(monkeypatch) -> None:
    import poker.evaluation as evaluation

    seen: list[int | None] = []
    original = evaluation.HandState

    class TrackingHandState(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("seed"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(evaluation, "HandState", TrackingHandState)
    report = evaluate_suite(
        "candidate",
        TightBot(),
        {"rule": RuleBot()},
        config=EvaluationConfig(hands_per_opponent=6, player_count=2, seed_start=7_000, equity_samples=1, paired_position_seeds=True),
    )

    assert seen[:6] == [7_000, 7_000, 7_001, 7_001, 7_002, 7_002]
    matchup = report.matchups[0]
    assert matchup.seed_blocks == 3
    assert matchup.ci_method == "paired_position_seed_block_normal_v1"


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
