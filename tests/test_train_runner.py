"""Contracts for durable PPO run state and batched current-policy inference."""

from __future__ import annotations

import json
import signal

import pytest
import poker.train_runner as train_runner_module

from poker.curriculum import CurriculumConfig, CurriculumStage
from poker.curriculum_transition import CurriculumTransitionConfig
from poker.game_state import HandState
from poker.league import default_league
from poker.model import ModelConfig, TORCH_AVAILABLE, PokerAgentModel
from poker.observation import observation_for
from poker.pretraining import EquityBackbonePretrainer, PretrainingConfig
from poker.promotion import PromotionConfig
from poker.train_runner import RunSettings, TrainingRunConfig, TrainingRunner
from poker.training import PPOConfig, PPOTrainer, UpdateMetrics

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _config(*, iterations: int = 2) -> TrainingRunConfig:
    return TrainingRunConfig(
        run=RunSettings(
            seed=43,
            iterations=iterations,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_iterations=1,
            checkpoint_every_seconds=None,
        ),
        model=ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2),
        ppo=PPOConfig(epochs=1, minibatch_size=8, equity_samples=1, learning_rate=1e-3),
        curriculum=CurriculumConfig(
            base_learning_rate=1e-3,
            require_transfer_beats_scratch=False,
            require_previous_checkpoint_win=False,
        ),
    )


def _promotion_config(*, accept: bool = True) -> PromotionConfig:
    return PromotionConfig(
        enabled=True,
        every_iterations=1,
        hands_per_opponent=2,
        equity_samples=1,
        calibration_bins=4,
        baseline_bots=("rule",),
        historical_limit=1,
        minimum_baseline_bb_per_100=-1e9 if accept else 1e9,
        minimum_baseline_ci95_low=-1e9,
        maximum_baseline_ci95_half_width=1e9,
        minimum_historical_league_score=0.0,
        minimum_historical_ci95_low=-1e9,
        maximum_equity_ece=1.0,
    )


def _with_promotion(config: TrainingRunConfig, promotion: PromotionConfig) -> TrainingRunConfig:
    return TrainingRunConfig(
        run=config.run,
        model=config.model,
        ppo=config.ppo,
        curriculum=config.curriculum,
        league=config.league,
        promotion=promotion,
        transition=config.transition,
        init_checkpoint=config.init_checkpoint,
    )


def _with_transition(config: TrainingRunConfig, reference_checkpoint, *, accept: bool = True) -> TrainingRunConfig:
    curriculum = CurriculumConfig(
        base_learning_rate=config.curriculum.base_learning_rate,
        min_baseline_win_rate_bb_per_100=-1e9 if accept else 1e9,
        max_equity_calibration_error=1.0,
        require_transfer_beats_scratch=False,
        require_previous_checkpoint_win=True,
    )
    transition = CurriculumTransitionConfig(
        enabled=True,
        every_iterations=1,
        hands_per_opponent=2,
        equity_samples=1,
        calibration_bins=4,
        baseline_bots=("rule",),
        minimum_baseline_ci95_low=-1e9,
        maximum_baseline_ci95_half_width=1e9,
        minimum_prior_ci95_low=-1e9,
        reference_checkpoint=str(reference_checkpoint),
        reference_checkpoint_sha256=train_runner_module._file_sha256(reference_checkpoint),
        curriculum=curriculum,
    )
    return TrainingRunConfig(
        run=config.run,
        model=config.model,
        ppo=config.ppo,
        curriculum=curriculum,
        league=config.league,
        promotion=config.promotion,
        transition=transition,
        init_checkpoint=config.init_checkpoint,
    )


def test_current_policy_batch_sampler_uses_one_forward_for_many_observations(monkeypatch) -> None:
    torch.manual_seed(7)
    model = PokerAgentModel(ModelConfig(embedding_dim=8, hidden_dim=16, history_layers=1, attention_heads=2))
    trainer = PPOTrainer(model, default_league(model, seed=3), PPOConfig(equity_samples=1))
    state = HandState(seed=7, player_count=2)
    observation = observation_for(state, state.actor)  # type: ignore[arg-type]
    calls: list[int] = []
    original_forward = model.forward

    def tracked_forward(observations):
        calls.append(len(observations))
        return original_forward(observations)

    monkeypatch.setattr(model, "forward", tracked_forward)
    decisions = trainer._select_current_batch((observation, observation))
    assert calls == [2]
    assert len(decisions) == 2
    assert all(observation["legal_actions"][decision[0].value] for decision in decisions)


def test_checkpoint_round_trip_is_atomic_and_model_loader_accepts_full_run(tmp_path) -> None:
    runner = TrainingRunner(_config(iterations=1), tmp_path / "run")
    result = runner.run(install_signal_handlers=False)
    assert not result.interrupted
    assert result.checkpoint_path.exists()
    assert runner.latest_path.exists()
    assert not list(runner.checkpoint_directory.glob(".*.tmp"))

    restored = TrainingRunner.resume(runner.latest_path)
    assert restored.iteration == runner.iteration == 1
    assert restored.global_hands == runner.global_hands == 1
    assert restored.global_decisions == runner.global_decisions
    assert restored.trainer._seed_counter == runner.trainer._seed_counter
    assert all(torch.equal(left, right) for left, right in zip(restored.model.state_dict().values(), runner.model.state_dict().values(), strict=True))
    # An inference/UI process only needs the normal model part of a full run.
    model_only = PokerAgentModel.load_checkpoint(runner.latest_path)
    assert model_only.checkpoint_metadata() == runner.model.checkpoint_metadata()
    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    assert manifest["latest"] == str(runner.latest_path)
    assert manifest["checkpoints"][-1]["path"] == str(result.checkpoint_path)


@pytest.mark.parametrize("source_kind", ("ordinary", "pretraining"))
def test_fresh_runner_can_warm_start_weights_without_restoring_run_state(tmp_path, source_kind) -> None:
    config = _config(iterations=0)
    source = PokerAgentModel(config.model)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    checkpoint = tmp_path / f"{source_kind}.pt"
    if source_kind == "ordinary":
        source.save_checkpoint(checkpoint)
    else:
        EquityBackbonePretrainer(source, PretrainingConfig(batch_size=2)).save_checkpoint(checkpoint)

    runner = TrainingRunner(config, tmp_path / f"warm-{source_kind}", init_checkpoint=checkpoint)

    assert runner.iteration == runner.global_decisions == runner.global_hands == 0
    assert runner.trainer._seed_counter == config.run.seed
    assert not runner.trainer.optimizer.state
    assert all(torch.equal(actual, expected) for actual, expected in zip(runner.model.state_dict().values(), source.state_dict().values(), strict=True))
    assert runner.manifest["initialization"]["kind"] == "model_weights_only"
    assert runner.manifest["initialization"]["checkpoint"] == str(checkpoint)


def test_warm_start_rejects_incompatible_metadata_and_resume_rejects_model_checkpoint(tmp_path) -> None:
    config = _config(iterations=0)
    source = PokerAgentModel(config.model)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["metadata"]["model_version"] = "bad-version"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)

    with pytest.raises(ValueError, match="incompatible checkpoint model_version"):
        TrainingRunner(config, tmp_path / "bad", init_checkpoint=incompatible)
    incompatible_config = TrainingRunConfig(
        run=config.run,
        model=ModelConfig(embedding_dim=8, hidden_dim=24, history_layers=1, attention_heads=2),
        ppo=config.ppo,
        curriculum=config.curriculum,
        league=config.league,
    )
    with pytest.raises(ValueError, match="metadata/config does not match"):
        TrainingRunner(incompatible_config, tmp_path / "wrong-architecture", init_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="not a compatible resumable training checkpoint"):
        TrainingRunner.resume(checkpoint)


def test_requested_graceful_stop_writes_interrupt_checkpoint_at_safe_boundary(tmp_path) -> None:
    runner = TrainingRunner(_config(iterations=3), tmp_path / "run")
    runner.request_stop()
    result = runner.run(install_signal_handlers=False)
    assert result.interrupted
    assert result.iteration == 1
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["reason"] == "interrupt"
    assert TrainingRunner.resume(runner.latest_path).iteration == 1


def test_real_first_and_second_sigint_follow_graceful_then_immediate_semantics(tmp_path, monkeypatch) -> None:
    metrics = UpdateMetrics(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    graceful = TrainingRunner(_config(iterations=2), tmp_path / "graceful")

    def one_interrupt():
        graceful.iteration += 1
        signal.raise_signal(signal.SIGINT)
        return metrics

    monkeypatch.setattr(graceful, "_train_one_iteration", one_interrupt)
    result = graceful.run(install_signal_handlers=True)
    assert result.interrupted and result.checkpoint_path.exists()

    immediate = TrainingRunner(_config(iterations=2), tmp_path / "immediate")

    def two_interrupts():
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        raise AssertionError("the second SIGINT must interrupt immediately")

    monkeypatch.setattr(immediate, "_train_one_iteration", two_interrupts)
    with pytest.raises(KeyboardInterrupt):
        immediate.run(install_signal_handlers=True)
    assert not immediate.latest_path.exists()


def test_resume_matches_uninterrupted_training_stream(tmp_path) -> None:
    config = _config(iterations=2)
    uninterrupted = TrainingRunner(config, tmp_path / "uninterrupted")
    uninterrupted.run(install_signal_handlers=False)

    split = TrainingRunner(config, tmp_path / "split")
    split.run(until_iteration=1, install_signal_handlers=False)
    resumed = TrainingRunner.resume(split.latest_path)
    resumed.run(until_iteration=2, install_signal_handlers=False)

    assert resumed.global_hands == uninterrupted.global_hands
    assert resumed.global_decisions == uninterrupted.global_decisions
    assert resumed.trainer._seed_counter == uninterrupted.trainer._seed_counter
    assert all(torch.equal(left, right) for left, right in zip(resumed.model.state_dict().values(), uninterrupted.model.state_dict().values(), strict=True))


def test_runner_evaluates_promotes_and_restores_archive_state(tmp_path) -> None:
    config = _with_promotion(_config(iterations=1), _promotion_config())
    runner = TrainingRunner(config, tmp_path / "promoted")

    result = runner.run(install_signal_handlers=False)

    assert runner.last_evaluation_iteration == result.iteration == 1
    assert runner.best_score is not None
    assert runner.promotion_evaluator is not None
    assert runner.promotion_evaluator.archive_manifest_path.exists()
    assert len([member for member in runner.league.members if member.kind == "best"]) == 1
    restored = TrainingRunner.resume(runner.latest_path)
    assert restored.last_evaluation_iteration == 1
    assert restored.best_score == runner.best_score
    assert len([member for member in restored.league.members if member.kind == "best"]) == 1
    payload = torch.load(runner.latest_path, map_location="cpu", weights_only=True)
    assert payload["promotion_state"]["last_evaluation_iteration"] == 1
    assert payload["promotion_state"]["archive_manifest_sha256"]


def test_rejected_evaluation_does_not_perturb_training_rng_or_weights(tmp_path) -> None:
    base = _config(iterations=2)
    control = TrainingRunner(base, tmp_path / "control")
    control.run(install_signal_handlers=False)

    evaluated = TrainingRunner(_with_promotion(base, _promotion_config(accept=False)), tmp_path / "evaluated")
    evaluated.run(install_signal_handlers=False)

    assert evaluated.global_hands == control.global_hands
    assert evaluated.global_decisions == control.global_decisions
    assert evaluated.trainer._seed_counter == control.trainer._seed_counter
    assert not [member for member in evaluated.league.members if member.kind == "best"]
    assert all(torch.equal(left, right) for left, right in zip(evaluated.model.state_dict().values(), control.model.state_dict().values(), strict=True))


def test_resume_recovers_manifest_decision_after_crash_before_full_checkpoint(tmp_path) -> None:
    config = _with_promotion(_config(iterations=1), _promotion_config())
    runner = TrainingRunner(config, tmp_path / "recovery")
    metrics = runner._train_one_iteration()
    runner.pending_promotion_iteration = runner.iteration
    source = runner.save_checkpoint(reason="candidate", metrics=metrics)
    assert runner.promotion_evaluator is not None
    decision = runner.promotion_evaluator.evaluate_and_promote(
        iteration=runner.iteration,
        candidate_checkpoint=source,
        league=runner.league,
        stage=runner.scheduler.stage,
        champion_score=None,
        run_context={"run_config_sha256": train_runner_module._canonical_sha256(config.to_dict())},
    )
    assert decision.accepted

    restored = TrainingRunner.resume(source)

    assert restored.last_evaluation_iteration == 1
    assert restored.best_score == decision.baseline_score_bb_per_100
    assert len([member for member in restored.league.members if member.kind == "best"]) == 1


def test_resume_finishes_pending_promotion_before_any_next_rollout(tmp_path, monkeypatch) -> None:
    config = _with_promotion(_config(iterations=1), _promotion_config())
    runner = TrainingRunner(config, tmp_path / "pending")
    metrics = runner._train_one_iteration()
    runner.pending_promotion_iteration = runner.iteration
    source = runner.save_checkpoint(reason="candidate", metrics=metrics)

    restored = TrainingRunner.resume(source)

    monkeypatch.setattr(restored, "_train_one_iteration", lambda: (_ for _ in ()).throw(AssertionError("next rollout started before pending promotion")))
    result = restored.run(until_iteration=1, install_signal_handlers=False)
    assert not result.interrupted
    assert restored.last_evaluation_iteration == 1
    assert restored.pending_promotion_iteration is None


def test_resume_recovers_report_and_archive_copy_written_before_manifest(tmp_path, monkeypatch) -> None:
    config = _with_promotion(_config(iterations=1), _promotion_config())
    runner = TrainingRunner(config, tmp_path / "orphan-artifacts")
    metrics = runner._train_one_iteration()
    assert runner.promotion_evaluator is not None

    def crash_before_manifest(_record):
        raise RuntimeError("simulated crash before archive manifest")

    monkeypatch.setattr(runner.promotion_evaluator, "_record_decision", crash_before_manifest)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner._run_promotion(metrics)
    source = runner.checkpoint_directory / "candidate_00000001.pt"
    assert source.exists()
    assert list((runner.run_directory / "evaluations").glob("*.json"))
    assert list((runner.run_directory / "archive").glob("promoted_*.pt"))
    assert not runner.promotion_evaluator.archive_manifest_path.exists()

    restored = TrainingRunner.resume(source)
    result = restored.run(until_iteration=1, install_signal_handlers=False)
    assert not result.interrupted
    assert restored.last_evaluation_iteration == 1
    assert restored.promotion_evaluator is not None and restored.promotion_evaluator.archive_manifest_path.exists()


def test_stop_requested_during_promotion_does_not_run_an_extra_update(tmp_path, monkeypatch) -> None:
    config = _with_promotion(_config(iterations=2), _promotion_config())
    runner = TrainingRunner(config, tmp_path / "stop-during-eval")
    original_train = runner._train_one_iteration
    original_promotion = runner._run_promotion
    updates = 0

    def tracked_train():
        nonlocal updates
        updates += 1
        return original_train()

    def stop_after_promotion(metrics=None):
        result = original_promotion(metrics)
        runner.request_stop()
        return result

    monkeypatch.setattr(runner, "_train_one_iteration", tracked_train)
    monkeypatch.setattr(runner, "_run_promotion", stop_after_promotion)
    result = runner.run(install_signal_handlers=False)

    assert result.interrupted
    assert result.iteration == updates == 1


def test_runner_transitions_a_to_b_resets_optimizer_and_resumes_target_stage(tmp_path) -> None:
    base = _config(iterations=1)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    config = _with_transition(base, reference)
    assert TrainingRunConfig.from_dict(config.to_dict()) == config
    runner = TrainingRunner(config, tmp_path / "transitioned")

    result = runner.run(install_signal_handlers=False)

    assert not result.interrupted
    assert runner.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
    assert runner.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(config.curriculum.learning_rate_for("B"))
    assert not runner.trainer.optimizer.state
    assert runner.last_transition_evaluation_iteration == 1
    assert runner.pending_transition_iteration is None
    assert runner.transition_evaluator is not None
    accepted = runner.transition_evaluator.last_accepted_decision
    assert accepted is not None
    transfer = accepted["transfer_checkpoint"]
    assert transfer is not None and PokerAgentModel.load_checkpoint(transfer).checkpoint_metadata() == runner.model.checkpoint_metadata()
    payload = torch.load(runner.latest_path, map_location="cpu", weights_only=True)
    assert payload["curriculum"]["stage"] == "B"
    assert payload["curriculum_transition_state"]["last_evaluation_iteration"] == 1
    restored = TrainingRunner.resume(runner.latest_path)
    assert restored.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
    assert restored.pending_transition_iteration is None
    assert not restored.trainer.optimizer.state


def test_rejected_curriculum_transition_keeps_stage_optimizer_and_rng_stream(tmp_path) -> None:
    base = _config(iterations=2)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    rejected_config = _with_transition(base, reference, accept=False)
    rejected = TrainingRunner(rejected_config, tmp_path / "rejected")
    rejected.run(install_signal_handlers=False)

    assert rejected.scheduler.stage is CurriculumStage.A_HEADS_UP_STARTER
    assert rejected.trainer.optimizer.state
    assert rejected.last_transition_evaluation_iteration == 2
    assert rejected.transition_evaluator is not None
    assert rejected.transition_evaluator.last_accepted_decision is None
    assert not list((rejected.run_directory / "curriculum-transitions" / "transfers").glob("*.pt"))


def test_resume_finishes_pending_transition_before_next_rollout(tmp_path, monkeypatch) -> None:
    base = _config(iterations=1)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    config = _with_transition(base, reference)
    runner = TrainingRunner(config, tmp_path / "pending-transition")
    metrics = runner._train_one_iteration()
    runner.pending_transition_iteration = runner.iteration
    source = runner.save_checkpoint(reason="transition_candidate", metrics=metrics)

    restored = TrainingRunner.resume(source)
    monkeypatch.setattr(restored, "_train_one_iteration", lambda: (_ for _ in ()).throw(AssertionError("rollout started before transition recovery")))
    result = restored.run(until_iteration=1, install_signal_handlers=False)

    assert not result.interrupted
    assert restored.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
    assert restored.pending_transition_iteration is None


def test_resume_recovers_transition_manifest_ahead_of_pending_source(tmp_path) -> None:
    base = _config(iterations=1)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    config = _with_transition(base, reference)
    runner = TrainingRunner(config, tmp_path / "manifest-ahead")
    metrics = runner._train_one_iteration()
    runner.pending_transition_iteration = runner.iteration
    source = runner.save_checkpoint(reason="transition_candidate", metrics=metrics)
    assert runner.transition_evaluator is not None
    decision = runner.transition_evaluator.evaluate_transition(
        iteration=runner.iteration,
        candidate_checkpoint=source,
        reference_checkpoint=reference,
        stage=runner.scheduler.stage,
        run_context={"run_config_sha256": train_runner_module._canonical_sha256(config.to_dict())},
    )
    assert decision.accepted

    restored = TrainingRunner.resume(source)
    restored.run(until_iteration=1, install_signal_handlers=False)

    assert restored.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
    assert restored.pending_transition_iteration is None


def test_resume_processes_due_transition_before_next_rollout(tmp_path, monkeypatch) -> None:
    base = _config(iterations=2)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    config = _with_transition(base, reference)
    runner = TrainingRunner(config, tmp_path / "due-after-interrupt")
    runner.request_stop()
    interrupted = runner.run(install_signal_handlers=False)
    assert interrupted.interrupted and runner.scheduler.stage is CurriculumStage.A_HEADS_UP_STARTER

    restored = TrainingRunner.resume(runner.latest_path)
    original_train = restored._train_one_iteration

    def assert_transition_first():
        assert restored.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
        return original_train()

    monkeypatch.setattr(restored, "_train_one_iteration", assert_transition_first)
    restored.run(until_iteration=2, install_signal_handlers=False)
    assert restored.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL


def test_resume_after_transition_matches_uninterrupted_target_stage_stream(tmp_path) -> None:
    base = _config(iterations=2)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    config = _with_transition(base, reference)

    uninterrupted = TrainingRunner(config, tmp_path / "transition-uninterrupted")
    uninterrupted.run(install_signal_handlers=False)

    split = TrainingRunner(config, tmp_path / "transition-split")
    split.run(until_iteration=1, install_signal_handlers=False)
    resumed = TrainingRunner.resume(split.latest_path)
    resumed.run(until_iteration=2, install_signal_handlers=False)

    assert resumed.scheduler.stage is uninterrupted.scheduler.stage is CurriculumStage.B_HEADS_UP_FULL
    assert resumed.trainer._seed_counter == uninterrupted.trainer._seed_counter
    assert resumed.global_decisions == uninterrupted.global_decisions
    assert all(
        torch.equal(left, right)
        for left, right in zip(resumed.model.state_dict().values(), uninterrupted.model.state_dict().values(), strict=True)
    )


def test_transition_config_rejects_unsafe_or_unsupported_automation(tmp_path) -> None:
    base = _config(iterations=0)
    reference = TrainingRunner(base, tmp_path / "reference-run").save_checkpoint(reason="reference")
    transition = _with_transition(base, reference).transition
    with pytest.raises(ValueError, match="cannot be enabled in the same run"):
        TrainingRunConfig(
            run=base.run,
            model=base.model,
            ppo=base.ppo,
            curriculum=transition.curriculum,
            league=base.league,
            promotion=_promotion_config(),
            transition=transition,
        )
    strict_curriculum = CurriculumConfig(base_learning_rate=base.ppo.learning_rate, require_transfer_beats_scratch=True)
    with pytest.raises(ValueError, match="cannot require transfer-vs-scratch"):
        TrainingRunConfig(
            run=base.run,
            model=base.model,
            ppo=base.ppo,
            curriculum=strict_curriculum,
            league=base.league,
            transition=CurriculumTransitionConfig(
                enabled=True,
                reference_checkpoint=str(reference),
                reference_checkpoint_sha256=train_runner_module._file_sha256(reference),
                curriculum=strict_curriculum,
            ),
        )
    with pytest.raises(ValueError, match="only A -> B"):
        CurriculumTransitionConfig(
            source_stage=CurriculumStage.C_THREE_MAX,
            target_stage=CurriculumStage.D_FIVE_MAX_FIXED,
        )


@pytest.mark.parametrize("stage", list(CurriculumStage))
def test_runner_applies_curriculum_learning_rate_scale(stage, tmp_path) -> None:
    config = _config(iterations=0)
    config = TrainingRunConfig(
        run=RunSettings(
            stage=stage,
            seed=config.run.seed,
            iterations=0,
            hands_per_iteration=1,
            table_count=1,
            checkpoint_every_seconds=None,
        ),
        model=config.model,
        ppo=config.ppo,
        curriculum=config.curriculum,
        league=config.league,
    )
    runner = TrainingRunner(config, tmp_path / stage.value)
    expected = config.curriculum.learning_rate_for(stage)
    assert runner.trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected)
