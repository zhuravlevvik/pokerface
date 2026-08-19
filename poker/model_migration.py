"""Explicit, auditable migration from model v2 to v3 expected-share head.

Normal checkpoint loading intentionally rejects v2 because v3 has a new
learnable head.  This helper is the opt-in warm-start path: it copies every
matching v2 tensor exactly and deterministically initialises only the missing
expected-showdown-share head.  It never attempts to carry optimizer/RNG/run
state, so it writes a model-only v3 artifact.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .model import ACTION_NAMES, BET_SIZE_ACTIONS, EQUITY_OUTCOMES, TORCH_AVAILABLE, ModelConfig, PokerAgentModel
from .observation import OBSERVATION_VERSION

if TORCH_AVAILABLE:
    import torch


V2_MODEL_VERSION = "2.0"
MIGRATION_VERSION = "v2_to_v3_expected_showdown_share_v1"


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for poker.model_migration; install the project with `.[rl]`.")


def migrate_v2_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    expected_showdown_share_init_seed: int = 0,
) -> Path:
    """Create a v3 model-only checkpoint from an explicit compatible v2 input."""

    _require_torch()
    source = Path(source_path)
    destination = Path(destination_path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("v2 migration source is not a checkpoint object")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("v2 migration source has no model metadata or state dict")
    _validate_v2_metadata(metadata)
    config_data = metadata.get("config")
    if not isinstance(config_data, Mapping):
        raise ValueError("v2 migration source has no model config")
    missing = {"expected_showdown_share_head.weight", "expected_showdown_share_head.bias"}
    # Construction and the sole new-head initialisation are isolated from the
    # caller's stochastic training stream.
    with torch.random.fork_rng(devices=[]):
        model = PokerAgentModel(ModelConfig(**dict(config_data)))
        current = model.state_dict()
        source_keys, current_keys = set(state_dict), set(current)
        if source_keys != current_keys - missing:
            raise ValueError("v2 migration source state dict does not match the v2-compatible v3 architecture")
        for name, value in state_dict.items():
            if not hasattr(value, "shape") or value.shape != current[name].shape:
                raise ValueError(f"v2 migration tensor {name!r} has incompatible shape")
            current[name] = value
        model.load_state_dict(current, strict=True)
        torch.manual_seed(expected_showdown_share_init_seed)
        model.expected_showdown_share_head.reset_parameters()
    result: dict[str, Any] = {
        "metadata": model.checkpoint_metadata(),
        "state_dict": model.state_dict(),
        "lineage": {
            "migration_version": MIGRATION_VERSION,
            "source_path": str(source),
            "source_sha256": _file_sha256(source),
            "source_model_version": V2_MODEL_VERSION,
            "expected_showdown_share_init_seed": expected_showdown_share_init_seed,
            "initialized_parameters": sorted(missing),
        },
    }
    _atomic_torch_save(result, destination)
    return destination


def _validate_v2_metadata(metadata: Mapping[str, object]) -> None:
    expected = {
        "model_version": V2_MODEL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_space": list(ACTION_NAMES),
        "bet_size_actions": list(BET_SIZE_ACTIONS),
        "equity_outcomes": list(EQUITY_OUTCOMES),
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"incompatible v2 checkpoint {name}: {metadata.get(name)!r}")
    if "equity_heads" in metadata:
        raise ValueError("v2 migration source unexpectedly declares v3 equity heads")


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:  # pragma: no cover - filesystem-specific fallback.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["MIGRATION_VERSION", "V2_MODEL_VERSION", "migrate_v2_checkpoint"]
