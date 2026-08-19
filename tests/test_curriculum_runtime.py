"""Tests for the executable paired-curriculum configuration layer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from poker.curriculum import CurriculumStage
from poker.curriculum_cli import main
from poker.curriculum_coordinator import EvaluationProtocol, EvaluationRequest, OpponentSpec
from poker.curriculum_runtime import (
    CurriculumJobConfig,
    default_curriculum_job_config,
    load_curriculum_job_config,
    native_multiway_evaluator,
    write_curriculum_job_config,
)
from poker.model import PokerAgentModel
from poker.multiway_evaluation import MultiwayEvaluationConfig


def test_curriculum_job_config_round_trip(tmp_path: Path) -> None:
    expected = default_curriculum_job_config()
    path = write_curriculum_job_config(expected, tmp_path / "job.json")

    actual = load_curriculum_job_config(path)

    assert actual.as_dict() == expected.as_dict()
    assert actual.coordinator.source_stage is CurriculumStage.B_HEADS_UP_FULL
    assert actual.coordinator.target_stage is CurriculumStage.C_THREE_MAX
    assert actual.paired_rung.target_stage is CurriculumStage.C_THREE_MAX
    assert len(actual.coordinator.target_protocols[0].opponents) == 2


def test_job_rejects_rung_for_different_target_stage() -> None:
    job = default_curriculum_job_config()

    with pytest.raises(ValueError, match="target stage"):
        CurriculumJobConfig(job.coordinator, replace(job.paired_rung, target_stage=CurriculumStage.D_FIVE_MAX_FIXED))


def test_cli_writes_starter_config(tmp_path: Path) -> None:
    destination = tmp_path / "starter.json"

    assert main(["--write-default-config", str(destination)]) == 0
    assert load_curriculum_job_config(destination).coordinator.target_stage is CurriculumStage.C_THREE_MAX


def test_cli_reports_safe_interruption_as_resumable(tmp_path: Path, monkeypatch, capsys) -> None:
    config = write_curriculum_job_config(default_curriculum_job_config(), tmp_path / "job.json")

    class InterruptedCoordinator:
        def __init__(self, *args, **kwargs):
            pass

        def coordinate(self, *args, **kwargs):
            raise InterruptedError

    monkeypatch.setattr("poker.curriculum_cli.CurriculumCoordinator", InterruptedCoordinator)

    assert main([
        "--config", str(config),
        "--run-dir", str(tmp_path / "run"),
        "--source-checkpoint", str(tmp_path / "source.pt"),
        "--reference-checkpoint", str(tmp_path / "reference.pt"),
    ]) == 130
    assert "repeat the same command" in capsys.readouterr().out


def test_native_evaluator_runs_allowlisted_bot_and_rejects_unpinned_checkpoint(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.pt"
    PokerAgentModel().save_checkpoint(candidate)
    config = MultiwayEvaluationConfig(player_count=2, deal_blocks=2, equity_samples=1)
    bot_protocol = EvaluationProtocol(
        "hu-rule",
        (OpponentSpec("rule", {"kind": "bot", "bot": "rule"}),),
        config.as_dict(),
    )

    evaluator = native_multiway_evaluator()
    report = evaluator(EvaluationRequest(candidate, CurriculumStage.B_HEADS_UP_FULL, bot_protocol, "candidate"))

    assert report.hands == 4
    assert report.opponent_slots == ("rule",)

    bad_protocol = EvaluationProtocol(
        "hu-checkpoint",
        (OpponentSpec("frozen", {"kind": "checkpoint", "path": str(candidate), "sha256": "0" * 64}),),
        config.as_dict(),
    )
    with pytest.raises(ValueError, match="hash-mismatched"):
        evaluator(EvaluationRequest(candidate, CurriculumStage.B_HEADS_UP_FULL, bad_protocol, "candidate"))
