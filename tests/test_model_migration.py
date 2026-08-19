"""Explicit v2 -> v3 expected-showdown-share checkpoint migration tests."""

from __future__ import annotations

import pytest

from poker.model import MODEL_VERSION, TORCH_AVAILABLE, ModelConfig, PokerAgentModel
from poker.model_migration import MIGRATION_VERSION, migrate_v2_checkpoint

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed; install with .[rl]")

if TORCH_AVAILABLE:
    import torch


def _write_v2_fixture(path) -> tuple[PokerAgentModel, dict]:
    model = PokerAgentModel(ModelConfig(embedding_dim=16, hidden_dim=32, history_layers=1, attention_heads=4))
    payload = {"metadata": model.checkpoint_metadata(), "state_dict": model.state_dict()}
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["model_version"] = "2.0"
    payload["metadata"].pop("equity_heads")
    payload["state_dict"] = {
        name: value.detach().clone()
        for name, value in payload["state_dict"].items()
        if not name.startswith("expected_showdown_share_head.")
    }
    torch.save(payload, path)
    return model, payload


def test_normal_loader_rejects_v2_but_explicit_migration_preserves_matching_weights(tmp_path) -> None:
    torch.manual_seed(21)
    source = tmp_path / "v2.pt"
    _, v2 = _write_v2_fixture(source)

    with pytest.raises(ValueError, match="incompatible checkpoint model_version"):
        PokerAgentModel.load_checkpoint(source)

    destination = migrate_v2_checkpoint(source, tmp_path / "v3.pt", expected_showdown_share_init_seed=77)
    restored = PokerAgentModel.load_checkpoint(destination)
    payload = torch.load(destination, map_location="cpu", weights_only=True)

    assert restored.checkpoint_metadata()["model_version"] == MODEL_VERSION == "3.0"
    assert payload["lineage"]["migration_version"] == MIGRATION_VERSION
    assert payload["lineage"]["source_model_version"] == "2.0"
    assert payload["lineage"]["initialized_parameters"] == [
        "expected_showdown_share_head.bias",
        "expected_showdown_share_head.weight",
    ]
    for name, value in v2["state_dict"].items():
        assert torch.equal(restored.state_dict()[name], value)


def test_v2_migration_initializes_only_new_head_deterministically(tmp_path) -> None:
    source = tmp_path / "v2.pt"
    _write_v2_fixture(source)

    first = PokerAgentModel.load_checkpoint(migrate_v2_checkpoint(source, tmp_path / "first.pt", expected_showdown_share_init_seed=3))
    second = PokerAgentModel.load_checkpoint(migrate_v2_checkpoint(source, tmp_path / "second.pt", expected_showdown_share_init_seed=3))

    assert torch.equal(first.expected_showdown_share_head.weight, second.expected_showdown_share_head.weight)
    assert torch.equal(first.expected_showdown_share_head.bias, second.expected_showdown_share_head.bias)
