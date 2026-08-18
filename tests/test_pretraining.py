"""Contracts for reusable equity/backbone supervised warm-up."""

from __future__ import annotations

from copy import deepcopy

import pytest

from poker.game_state import HandState
from poker.model import ACTION_NAMES, BET_SIZE_ACTIONS, TORCH_AVAILABLE, ModelConfig, PokerAgentModel
from poker.observation import observation_for
from poker.pretraining import EquityBackbonePretrainer, PretrainingConfig, factorized_behavior_cloning_loss

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch
    from torch.nn import functional as F


def _model() -> PokerAgentModel:
    return PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))


def _examples(count: int = 6):
    rows = []
    for seed in range(count):
        state = HandState(seed=100 + seed, player_count=2)
        assert state.actor is not None
        # Initial decisions have a legal min-raise, so these rows also exercise
        # factorized type + conditional-sizing behaviour cloning.
        rows.append(
            type(
                "Example",
                (),
                {
                    "observation": observation_for(state, state.actor),
                    "selected_action": BET_SIZE_ACTIONS[seed % len(BET_SIZE_ACTIONS)],
                    "equity_target": (0.2 + 0.1 * (seed % 3), 0.1, 0.7 - 0.1 * (seed % 3)),
                    "terminal_pnl_bb": float(seed - 3),
                },
            )()
        )
    return rows


def test_equity_pretraining_loss_is_finite_and_changes_parameters() -> None:
    torch.manual_seed(1)
    model = _model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trainer = EquityBackbonePretrainer(model, PretrainingConfig(batch_size=2, learning_rate=1e-3, seed=12))

    metrics = trainer.train_epoch(_examples())

    assert metrics.samples == 6
    assert all(torch.isfinite(torch.tensor(value)) for value in (metrics.total_loss, metrics.equity_loss, metrics.behavior_cloning_loss, metrics.value_warmup_loss))
    assert metrics.behavior_cloning_loss == 0.0
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_each_epoch_builds_one_deterministic_permutation() -> None:
    trainer = EquityBackbonePretrainer(_model(), PretrainingConfig(batch_size=2, seed=12))
    calls = 0
    original = trainer._indices_for_epoch

    def counted(size: int):
        nonlocal calls
        calls += 1
        return original(size)

    trainer._indices_for_epoch = counted  # type: ignore[method-assign]
    trainer.train_epoch(_examples(5))

    assert calls == 1


def test_factorized_behavior_cloning_includes_size_only_for_raises() -> None:
    torch.manual_seed(2)
    model = _model().eval()
    rows = _examples(2)
    output = model([row.observation for row in rows])
    actions = torch.tensor([ACTION_NAMES.index("raise"), ACTION_NAMES.index("call")])
    sizes = torch.tensor([BET_SIZE_ACTIONS.index(BET_SIZE_ACTIONS[0]), -1])

    actual = factorized_behavior_cloning_loss(output, actions, sizes)
    action_log_probs = F.log_softmax(output.action_logits, dim=-1)
    size_log_probs = F.log_softmax(output.bet_size_logits, dim=-1)
    expected = -(
        action_log_probs[0, actions[0]]
        + size_log_probs[0, sizes[0]]
        + action_log_probs[1, actions[1]]
    ) / 2

    assert actual == pytest.approx(float(expected.item()))


def test_checkpoint_resume_replays_the_next_deterministic_epoch(tmp_path) -> None:
    torch.manual_seed(3)
    original = _model()
    rows = _examples(7)
    config = PretrainingConfig(batch_size=3, learning_rate=1e-3, seed=91, behavior_cloning_coefficient=0.05, value_warmup_coefficient=0.1)

    uninterrupted_model = _model()
    uninterrupted_model.load_state_dict(deepcopy(original.state_dict()))
    uninterrupted = EquityBackbonePretrainer(uninterrupted_model, config)
    uninterrupted.fit(rows, epochs=2)

    interrupted_model = _model()
    interrupted_model.load_state_dict(deepcopy(original.state_dict()))
    interrupted = EquityBackbonePretrainer(interrupted_model, config)
    first = interrupted.train_epoch(rows)
    checkpoint = tmp_path / "pretraining.pt"
    interrupted.save_checkpoint(checkpoint)
    resumed = EquityBackbonePretrainer.load_checkpoint(checkpoint)
    second = resumed.train_epoch(rows)

    assert (first.epoch, second.epoch, second.global_step) == (1, 2, uninterrupted.global_step)
    for actual, expected in zip(resumed.model.state_dict().values(), uninterrupted.model.state_dict().values(), strict=True):
        assert torch.equal(actual, expected)


def test_pretraining_checkpoint_is_loadable_by_inference_model_loader(tmp_path) -> None:
    torch.manual_seed(4)
    trainer = EquityBackbonePretrainer(_model(), PretrainingConfig(batch_size=2))
    trainer.train_epoch(_examples())
    checkpoint = trainer.save_checkpoint(tmp_path / "warmup.pt")

    loaded = PokerAgentModel.load_checkpoint(checkpoint)

    assert loaded.checkpoint_metadata() == trainer.model.checkpoint_metadata()
    for actual, expected in zip(loaded.state_dict().values(), trainer.model.state_dict().values(), strict=True):
        assert torch.equal(actual, expected)


def test_checkpoint_roundtrips_provenance_and_allows_cpu_override_for_cuda_metadata(tmp_path) -> None:
    trainer = EquityBackbonePretrainer(
        _model(),
        PretrainingConfig(batch_size=2),
        provenance={"corpus_sha256": "abc123", "run_config_sha256": "def456"},
    )
    trainer.train_epoch(_examples())
    checkpoint = trainer.save_checkpoint(tmp_path / "warmup.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Simulate an artifact emitted by a GPU run.  Loading it with an explicit
    # CPU target must not attempt CUDA allocation from source metadata.
    payload["pretraining_config"]["device"] = "cuda"
    cuda_source = tmp_path / "cuda-source.pt"
    torch.save(payload, cuda_source)

    restored = EquityBackbonePretrainer.load_checkpoint(cuda_source, device="cpu")

    assert restored.config.device == "cpu"
    assert restored.provenance == {"corpus_sha256": "abc123", "run_config_sha256": "def456"}


def test_validation_reports_equity_quality_and_hook_breakdowns() -> None:
    torch.manual_seed(5)
    trainer = EquityBackbonePretrainer(_model(), PretrainingConfig(batch_size=2, calibration_bins=4))
    rows = _examples()

    report = trainer.evaluate(rows, breakdowns={"action": lambda example: example.selected_action})

    assert report.equity.samples == len(rows)
    assert report.equity.logloss >= 0.0
    assert report.equity.brier_score >= 0.0
    assert report.equity.expected_calibration_error >= 0.0
    assert set(report.breakdowns["action"]) == {row.selected_action for row in rows}
