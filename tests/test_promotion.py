"""Contracts for fixed-suite HU promotion and its durable archive."""

from __future__ import annotations

import json

import pytest

from poker.curriculum import CurriculumStage
from poker.league import default_league
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.promotion import PromotionConfig, PromotionEvaluator

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _model() -> PokerAgentModel:
    torch.manual_seed(19)
    return PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))


def _config(**overrides) -> PromotionConfig:
    values = {
        "enabled": True,
        "every_iterations": 2,
        "hands_per_opponent": 4,
        "equity_samples": 1,
        "calibration_bins": 4,
        "baseline_bots": ("rule",),
        "historical_limit": 0,
        "minimum_baseline_bb_per_100": -1e9,
        "minimum_baseline_ci95_low": -1e9,
        "maximum_baseline_ci95_half_width": 1e9,
        "minimum_historical_league_score": 0.0,
        "minimum_historical_ci95_low": -1e9,
        "maximum_equity_ece": 1.0,
    }
    values.update(overrides)
    return PromotionConfig(**values)


def test_schedule_is_iteration_based_and_does_not_repeat_completed_evaluation(tmp_path) -> None:
    evaluator = PromotionEvaluator(_config(), tmp_path)
    assert not evaluator.should_evaluate(1)
    assert evaluator.should_evaluate(2)
    assert not evaluator.should_evaluate(2, last_evaluation_iteration=2)
    assert evaluator.should_evaluate(3, completing=True)


def test_promotion_freezes_candidate_writes_auditable_report_and_restores_manifest(tmp_path) -> None:
    model = _model()
    source = tmp_path / "full-checkpoint.pt"
    model.save_checkpoint(source)
    league = default_league(model, seed=5)
    evaluator = PromotionEvaluator(_config(minimum_champion_improvement=0.1), tmp_path / "run", run_seed=7)

    first = evaluator.evaluate_and_promote(
        iteration=2,
        candidate_checkpoint=source,
        league=league,
        stage=CurriculumStage.A_HEADS_UP_STARTER,
        champion_score=None,
        run_context={"global_decisions": 12},
    )

    assert first.accepted and first.checkpoint_path is not None and first.checkpoint_path.exists()
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["promotion_report_version"] == 3
    assert report["protocol"]["paired_position_seeds"] is True
    assert report["protocol"]["scalar_metric_protocol"] == "active_hands_expected_showdown_share_v1"
    assert report["suite"]["schema_version"] == "2.0"
    assert report["candidate"]["source_full_checkpoint_sha256"]
    assert report["suite"]["matchups"][0]["seed_blocks"] == 2
    assert report["decision"]["checkpoint_path"] == str(first.checkpoint_path)
    assert any(member.kind == "best" for member in league.members)
    restored = PromotionEvaluator(evaluator.config, evaluator.run_directory, run_seed=7)
    assert restored.champion_score == first.baseline_score_bb_per_100

    second = restored.evaluate_and_promote(
        iteration=4,
        candidate_checkpoint=source,
        league=league,
        stage=CurriculumStage.A_HEADS_UP_STARTER,
        champion_score=first.baseline_score_bb_per_100,
    )
    assert not second.accepted
    assert second.checkpoint_path is None
    assert "did not improve" in " ".join(second.reasons)
    manifest = json.loads(restored.archive_manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 3
    assert len(manifest["decisions"]) == 2
    assert len(manifest["promoted"]) == 1


def test_archive_manifest_detects_report_tampering_and_multiway_is_rejected(tmp_path) -> None:
    model = _model()
    source = tmp_path / "candidate.pt"
    model.save_checkpoint(source)
    evaluator = PromotionEvaluator(_config(), tmp_path / "run")
    league = default_league(model, seed=3)
    result = evaluator.evaluate_and_promote(
        iteration=2,
        candidate_checkpoint=source,
        league=league,
        stage=CurriculumStage.A_HEADS_UP_STARTER,
        champion_score=None,
    )
    result.report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report is missing or does not match"):
        PromotionEvaluator(evaluator.config, evaluator.run_directory)

    clean = PromotionEvaluator(_config(), tmp_path / "multiway")
    with pytest.raises(ValueError, match="heads-up only"):
        clean.evaluate_and_promote(
            iteration=2,
            candidate_checkpoint=source,
            league=league,
            stage=CurriculumStage.C_THREE_MAX,
            champion_score=None,
        )
