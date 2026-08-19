"""Tests for durable paired-rung curriculum coordination."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from poker.betting import Action
from poker.curriculum import CurriculumStage
from poker.curriculum_coordinator import (
    CurriculumCoordinator,
    CurriculumCoordinatorConfig,
    EvaluationProtocol,
    OpponentSpec,
    PairedRungArms,
)
from poker.equity import ExpectedShowdownShareMetrics
from poker.multiway_evaluation import MultiwayEvaluationConfig, MultiwayEvaluationReport, MultiwayModelDiagnostics


_FAKE_RUNG_CONFIG = {
    "target_stage": "B",
    "iterations": 4,
    "hands_per_iteration": 1,
    "table_count": 1,
}
_FAKE_RUNG_SHA = sha256(
    json.dumps(_FAKE_RUNG_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _full_checkpoint(stage: CurriculumStage, state_marker: str = "model") -> dict[str, object]:
    """Small loader fixture with the native TrainingRunner provenance shape."""

    return {
        "checkpoint_version": 1,
        "metadata": {"model_version": "3.0", "config": {"hidden_dim": 8}},
        "state_dict": {"marker": state_marker},
        "optimizer_state_dict": {},
        "run_config": {
            "run": {
                "stage": stage.value,
                "hands_per_iteration": 1,
                "table_count": 1,
            }
        },
        "curriculum": {"stage": stage.value, "config": {}},
        "progress": {"iteration": 4, "global_hands": 4, "global_decisions": 40},
        "league": {},
        "rng": {},
    }


def _metric(*, ece: float = 0.01, mae: float = 0.02) -> ExpectedShowdownShareMetrics:
    return ExpectedShowdownShareMetrics(4, 0.1, 0.1, mae, mae, ece, ())


def _report(
    name: str,
    blocks: tuple[float, float],
    *,
    ece: float = 0.01,
    stage: CurriculumStage = CurriculumStage.B_HEADS_UP_FULL,
) -> MultiwayEvaluationReport:
    allowed = None
    if stage is CurriculumStage.A_HEADS_UP_STARTER:
        allowed = (Action.RAISE_MIN, Action.RAISE_1_2_POT, Action.RAISE_POT, Action.ALL_IN)
    config = MultiwayEvaluationConfig(
        player_count=2,
        deal_blocks=2,
        seed_start=17,
        equity_samples=1,
        allowed_raise_actions=allowed,
        required_expected_showdown_share_strata=("street=preflop|active_players=2",),
    )
    average = sum(blocks) / len(blocks)
    bb = average * 100.0
    diagnostics = MultiwayModelDiagnostics(
        decisions=4,
        policy_entropy=0.3,
        value_mae_bb=0.1,
        value_rmse_bb=0.1,
        masked_action_rate=0.0,
        illegal_action_count=0,
        equity=None,
        expected_showdown_share=_metric(ece=ece),
        expected_showdown_share_by_stratum={"street=preflop|active_players=2": _metric(ece=ece)},
        expected_showdown_share_support={"street=preflop|active_players=2": 4},
    )
    return MultiwayEvaluationReport(
        candidate=name,
        config=config,
        opponent_slots=("bot:rule",),
        hands=4,
        seed_blocks=2,
        pnl_bb=sum(blocks) * 2,
        block_pnl_bb=blocks,
        bb_per_100=bb,
        bb_per_100_standard_error=0.0,
        bb_per_100_ci95_low=bb,
        bb_per_100_ci95_high=bb,
        position_hands={"BTN": 2, "BB": 2},
        pnl_by_position_bb={"BTN": sum(blocks), "BB": sum(blocks)},
        model_diagnostics=diagnostics,
    )


def _protocol(
    name: str = "fixed-100bb", *, stage: CurriculumStage = CurriculumStage.B_HEADS_UP_FULL
) -> EvaluationProtocol:
    allowed = None
    if stage is CurriculumStage.A_HEADS_UP_STARTER:
        allowed = (Action.RAISE_MIN, Action.RAISE_1_2_POT, Action.RAISE_POT, Action.ALL_IN)
    return EvaluationProtocol(
        name,
        (OpponentSpec("bot:rule", {"kind": "rule"}),),
        MultiwayEvaluationConfig(
            player_count=2,
            deal_blocks=2,
            seed_start=17,
            equity_samples=1,
            allowed_raise_actions=allowed,
            required_expected_showdown_share_strata=("street=preflop|active_players=2",),
        ).as_dict(),
        ("street=preflop|active_players=2",),
    )


def _config() -> CurriculumCoordinatorConfig:
    source = _protocol("source-fixed", stage=CurriculumStage.A_HEADS_UP_STARTER)
    return CurriculumCoordinatorConfig(
        CurriculumStage.A_HEADS_UP_STARTER,
        CurriculumStage.B_HEADS_UP_FULL,
        source,
        (_protocol(),),
        _FAKE_RUNG_SHA,
        min_target_baseline_ci95_low_bb_per_100=-100.0,
    )


def _rung_provenance(directory: Path, request, transfer: Path, scratch: Path) -> dict[str, object]:
    run_config = {"run": {"stage": "B", "hands_per_iteration": 1, "table_count": 1}}
    digest = sha256(json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = directory / "native-manifest.json"
    manifest.write_text(json.dumps({
        "config": _FAKE_RUNG_CONFIG,
        "config_sha256": _FAKE_RUNG_SHA,
        "source_checkpoint_sha256": request.intent.source_checkpoint_sha256,
        "completed": True,
        "arms": {
            label: {
                "full_checkpoint": str(path),
                "full_checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
                "iteration": 4,
                "global_hands": 4,
                "global_decisions": 40,
            }
            for label, path in (("transfer", transfer), ("scratch", scratch))
        },
    }))
    return {
        "target_stage": "B",
        "protocol_sha256": _FAKE_RUNG_SHA,
        "source_checkpoint_sha256": request.intent.source_checkpoint_sha256,
        "source_run_config_sha256": request.intent.source_provenance["run_config_sha256"],
        "budget": {"iterations": 4, "hands_per_iteration": 1, "table_count": 1},
        "transfer_run_config_sha256": digest,
        "scratch_run_config_sha256": digest,
        "native_manifest_path": str(manifest),
        "native_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
    }


def test_coordinator_writes_intent_first_resumes_idempotently_and_adopts_full_transfer_arm(tmp_path) -> None:
    source, reference = tmp_path / "source.pt", tmp_path / "reference.pt"
    transfer, scratch = tmp_path / "transfer.pt", tmp_path / "scratch.pt"
    for path in (source, reference, transfer, scratch):
        path.write_bytes(path.name.encode())
    calls = {"rung": 0, "evaluation": 0}

    def loader(path: Path):
        return _full_checkpoint(CurriculumStage.A_HEADS_UP_STARTER if path.name.startswith(("source", "reference")) else CurriculumStage.B_HEADS_UP_FULL, path.name)

    def rung(request):
        calls["rung"] += 1
        assert request.intent.path.is_file()  # durable before mutable work
        return PairedRungArms(transfer, scratch, _rung_provenance(tmp_path, request, transfer, scratch))

    def evaluator(request):
        calls["evaluation"] += 1
        if request.role == "target_transfer":
            return _report("transfer", (2.0, 2.0))
        if request.role == "target_scratch":
            return _report("scratch", (0.0, 0.0))
        if request.role == "source_candidate":
            return _report("source", (1.0, 1.0), stage=CurriculumStage.A_HEADS_UP_STARTER)
        return _report("reference", (0.0, 0.0), stage=CurriculumStage.A_HEADS_UP_STARTER)

    coordinator = CurriculumCoordinator(_config(), tmp_path / "coord", rung_runner=rung, evaluator=evaluator, checkpoint_loader=loader)
    first = coordinator.coordinate(source, reference)

    assert first.accepted
    assert first.adopted_checkpoint is not None
    assert first.adopted_checkpoint.is_file()
    assert calls == {"rung": 1, "evaluation": 4}
    report = first.report_path.read_text(encoding="utf-8")
    assert "private_cards" not in report.lower()
    assert "adopted_checkpoint" in report
    assert "/inputs/" in report

    resumed = CurriculumCoordinator(_config(), tmp_path / "coord", rung_runner=rung, evaluator=evaluator, checkpoint_loader=loader)
    second = resumed.coordinate(source, reference)
    assert second == first
    assert calls == {"rung": 1, "evaluation": 4}


def test_coordinator_rejects_bad_scalar_calibration_and_never_adopts(tmp_path) -> None:
    source, reference = tmp_path / "source.pt", tmp_path / "reference.pt"
    transfer, scratch = tmp_path / "transfer.pt", tmp_path / "scratch.pt"
    for path in (source, reference, transfer, scratch):
        path.write_bytes(path.name.encode())

    def loader(path: Path):
        return _full_checkpoint(CurriculumStage.A_HEADS_UP_STARTER if path.name.startswith(("source", "reference")) else CurriculumStage.B_HEADS_UP_FULL, path.name)

    def evaluator(request):
        if request.role == "target_transfer":
            return _report("transfer", (2.0, 2.0), ece=0.5)
        if request.role == "target_scratch":
            return _report("scratch", (0.0, 0.0))
        if request.role == "source_candidate":
            return _report("source", (1.0, 1.0), stage=CurriculumStage.A_HEADS_UP_STARTER)
        return _report("reference", (0.0, 0.0), stage=CurriculumStage.A_HEADS_UP_STARTER)

    decision = CurriculumCoordinator(
        _config(), tmp_path / "coord", rung_runner=lambda request: PairedRungArms(transfer, scratch, _rung_provenance(tmp_path, request, transfer, scratch)), evaluator=evaluator, checkpoint_loader=loader
    ).coordinate(source, reference)

    assert not decision.accepted
    assert decision.adopted_checkpoint is None
    assert any("ECE" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda value: value.update(protocol_sha256="b" * 64), "rung protocol"),
        (lambda value: value["budget"].update(iterations=3), "budget"),
    ),
)
def test_coordinator_rejects_unpinned_rung_protocol_or_inexact_budget(tmp_path, mutate, message) -> None:
    source, reference = tmp_path / "source.pt", tmp_path / "reference.pt"
    transfer, scratch = tmp_path / "transfer.pt", tmp_path / "scratch.pt"
    for path in (source, reference, transfer, scratch):
        path.write_bytes(path.name.encode())

    def loader(path: Path):
        stage = CurriculumStage.A_HEADS_UP_STARTER if path.name.startswith(("source", "reference")) else CurriculumStage.B_HEADS_UP_FULL
        return _full_checkpoint(stage, path.name)

    def rung(request):
        provenance = _rung_provenance(tmp_path, request, transfer, scratch)
        mutate(provenance)
        return PairedRungArms(transfer, scratch, provenance)

    with pytest.raises(ValueError, match=message):
        CurriculumCoordinator(
            _config(),
            tmp_path / "coord",
            rung_runner=rung,
            evaluator=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
            checkpoint_loader=loader,
        ).coordinate(source, reference)


def test_coordinator_manifest_fails_closed_when_evaluation_artifact_is_tampered(tmp_path) -> None:
    source, reference = tmp_path / "source.pt", tmp_path / "reference.pt"
    transfer, scratch = tmp_path / "transfer.pt", tmp_path / "scratch.pt"
    for path in (source, reference, transfer, scratch):
        path.write_bytes(path.name.encode())

    def loader(path: Path):
        return _full_checkpoint(CurriculumStage.A_HEADS_UP_STARTER if path.name.startswith(("source", "reference")) else CurriculumStage.B_HEADS_UP_FULL, path.name)

    def evaluator(request):
        values = {"target_transfer": (2.0, 2.0), "target_scratch": (0.0, 0.0), "source_candidate": (1.0, 1.0), "source_reference": (0.0, 0.0)}
        stage = CurriculumStage.A_HEADS_UP_STARTER if request.role.startswith("source") else CurriculumStage.B_HEADS_UP_FULL
        return _report(request.role, values[request.role], stage=stage)

    directory = tmp_path / "coord"
    decision = CurriculumCoordinator(
        _config(), directory, rung_runner=lambda request: PairedRungArms(transfer, scratch, _rung_provenance(tmp_path, request, transfer, scratch)), evaluator=evaluator, checkpoint_loader=loader
    ).coordinate(source, reference)
    report = __import__("json").loads(decision.report_path.read_text())
    evaluation = Path(report["target_evaluations"][0]["transfer"]["path"])
    evaluation.write_text("{}\n")

    with pytest.raises(ValueError, match="hash-mismatched"):
        CurriculumCoordinator(_config(), directory, rung_runner=lambda request: PairedRungArms(transfer, scratch, _rung_provenance(tmp_path, request, transfer, scratch)), evaluator=evaluator, checkpoint_loader=loader)


def test_coordinator_manifest_rejects_job_config_drift(tmp_path) -> None:
    directory = tmp_path / "coord"
    directory.mkdir()
    config = _config()
    (directory / "manifest.json").write_text(json.dumps({
        "version": 1,
        "config_sha256": "0" * 64,
        "decisions": [],
    }))

    with pytest.raises(ValueError, match="manifest"):
        CurriculumCoordinator(
            config,
            directory,
            rung_runner=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
            evaluator=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
            checkpoint_loader=lambda _: _full_checkpoint(CurriculumStage.A_HEADS_UP_STARTER),
        )


def test_config_rejects_skipped_stage() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="adjacent"):
        CurriculumCoordinatorConfig(
            CurriculumStage.A_HEADS_UP_STARTER,
            CurriculumStage.C_THREE_MAX,
            protocol,
            (protocol,),
            "a" * 64,
        )


def test_config_requires_stage_action_abstraction_and_every_stage_e_stack() -> None:
    with pytest.raises(ValueError, match="action abstraction"):
        CurriculumCoordinatorConfig(
            CurriculumStage.A_HEADS_UP_STARTER,
            CurriculumStage.B_HEADS_UP_FULL,
            _protocol("wrong-source"),
            (_protocol("target"),),
            "a" * 64,
        )

    opponents = tuple(OpponentSpec(f"bot-{index}", {"kind": "bot", "bot": "rule"}) for index in range(4))

    def five(name: str, stack_bb: int) -> EvaluationProtocol:
        config = MultiwayEvaluationConfig(player_count=5, deal_blocks=2, starting_stack=stack_bb * 100)
        return EvaluationProtocol(name, opponents, config.as_dict())

    with pytest.raises(ValueError, match="every and only"):
        CurriculumCoordinatorConfig(
            CurriculumStage.D_FIVE_MAX_FIXED,
            CurriculumStage.E_FIVE_MAX_EXPANDED,
            five("source", 100),
            (five("target-50", 50),),
            "a" * 64,
        )

    config = CurriculumCoordinatorConfig(
        CurriculumStage.D_FIVE_MAX_FIXED,
        CurriculumStage.E_FIVE_MAX_EXPANDED,
        five("source", 100),
        (five("target-50", 50), five("target-100", 100), five("target-200", 200)),
        "a" * 64,
    )
    assert len(config.target_protocols) == 3


def test_coordinator_rejects_distinct_files_with_identical_model_weights(tmp_path) -> None:
    source, reference = tmp_path / "source.pt", tmp_path / "reference.pt"
    source.write_bytes(b"source envelope")
    reference.write_bytes(b"reference envelope")

    coordinator = CurriculumCoordinator(
        _config(),
        tmp_path / "coord",
        rung_runner=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
        evaluator=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
        checkpoint_loader=lambda _: _full_checkpoint(CurriculumStage.A_HEADS_UP_STARTER, "same weights"),
    )

    with pytest.raises(ValueError, match="distinct model weights"):
        coordinator.coordinate(source, reference)


def test_gate_reasons_fail_closed_on_protocol_drift_and_stratified_calibration_support(tmp_path) -> None:
    coordinator = CurriculumCoordinator(
        _config(),
        tmp_path / "coord",
        rung_runner=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
        evaluator=lambda _: (_ for _ in ()).throw(AssertionError("not invoked")),
        checkpoint_loader=lambda _: _full_checkpoint(CurriculumStage.B_HEADS_UP_FULL),
    )
    protocol = _config().target_protocols[0]
    drifted = replace(_report("transfer", (1.0, 1.0)), config=MultiwayEvaluationConfig(
        player_count=2,
        deal_blocks=2,
        seed_start=18,
        equity_samples=1,
        required_expected_showdown_share_strata=("street=preflop|active_players=2",),
    ))
    assert any("config does not match" in reason for reason in coordinator._report_contract_reasons(drifted, protocol, "transfer"))

    high_stratum = _report("transfer", (1.0, 1.0), ece=0.5)
    assert any("required stratum" in reason and "ECE" in reason for reason in coordinator._report_contract_reasons(high_stratum, protocol, "transfer"))

    missing_support = replace(
        _report("transfer", (1.0, 1.0)),
        model_diagnostics=replace(
            _report("transfer", (1.0, 1.0)).model_diagnostics,
            expected_showdown_share_support={},
            expected_showdown_share_by_stratum={},
        ),
    )
    assert any("insufficient support" in reason for reason in coordinator._report_contract_reasons(missing_support, protocol, "transfer"))
