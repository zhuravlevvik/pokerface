"""Neural policy model for the versioned poker observation contract.

The model intentionally accepts only :mod:`poker.observation` dictionaries.
It has no access to opponent hole cards, the undealt deck, or any training
labels.  PyTorch is an optional dependency because the deterministic engine
and its test suite remain useful on machines used only for replay/UI work.
Install the ``rl`` extra to construct a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .betting import Action
from .observation import OBSERVATION_VERSION

try:  # Keep importing the game engine possible without its ML extra.
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError:  # pragma: no cover - exercised on non-RL installs.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


TORCH_AVAILABLE = torch is not None
"""Whether the optional PyTorch dependency is available in this environment."""

MODEL_VERSION = "1.0"
ACTION_NAMES = ("fold", "check", "call", "raise")
BET_SIZE_ACTIONS = (
    Action.RAISE_MIN.value,
    Action.RAISE_1_3_POT.value,
    Action.RAISE_1_2_POT.value,
    Action.RAISE_3_4_POT.value,
    Action.RAISE_POT.value,
    Action.RAISE_1_5_POT.value,
    Action.ALL_IN.value,
)
EQUITY_OUTCOMES = ("win", "tie", "loss")

_RANK_TO_ID = {rank: index + 1 for index, rank in enumerate("23456789TJQKA")}
_SUIT_TO_ID = {suit: index + 1 for index, suit in enumerate("cdhs")}
_POSITION_TO_ID = {position: index for index, position in enumerate(("BTN", "SB", "BB", "UTG", "CO"))}
_STREET_TO_ID = {street: index for index, street in enumerate(("preflop", "flop", "turn", "river", "showdown", "complete"))}
_HISTORY_ACTION_TO_ID = {action.value: index + 1 for index, action in enumerate(Action)}


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for poker.model; install the project with `.[rl]`.")


@dataclass(frozen=True)
class ModelConfig:
    """Small, deliberately configurable network suitable for MVP self-play."""

    embedding_dim: int = 32
    hidden_dim: int = 128
    history_layers: int = 3
    player_attention_layers: int = 1
    attention_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if self.history_layers <= 0 or self.player_attention_layers <= 0:
            raise ValueError("encoder layer counts must be positive")
        if self.attention_heads <= 0 or self.embedding_dim % self.attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class ModelOutput:
    """Batched output of :class:`PokerAgentModel`.

    ``action_probabilities`` and ``bet_size_probabilities`` are normalized
    only over their legal mask.  The masks are returned to make callers avoid
    accidentally sampling a raw, masked logit.
    """

    action_logits: Tensor
    action_probabilities: Tensor
    bet_size_logits: Tensor
    bet_size_probabilities: Tensor
    value: Tensor
    equity_logits: Tensor
    equity_probabilities: Tensor
    action_mask: Tensor
    bet_size_mask: Tensor


@dataclass(frozen=True)
class InferenceDecision:
    """Deterministic, engine-ready output for one acting player."""

    action: str
    action_probabilities: dict[str, float]
    bet_size_probabilities: dict[str, float]
    value_bb: float
    equity: dict[str, float]


if TORCH_AVAILABLE:

    class CardEncoder(nn.Module):
        """Embeds known cards by rank, suit and private/public role."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            dim = config.embedding_dim
            self.rank_embedding = nn.Embedding(14, dim, padding_idx=0)
            self.suit_embedding = nn.Embedding(5, dim, padding_idx=0)
            self.role_embedding = nn.Embedding(3, dim, padding_idx=0)
            self.projection = nn.Sequential(nn.Linear(dim, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim))

        def forward(self, ranks: Tensor, suits: Tensor, roles: Tensor, mask: Tensor) -> Tensor:
            encoded = self.rank_embedding(ranks) + self.suit_embedding(suits) + self.role_embedding(roles)
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.projection(pooled)


    class PlayerSetEncoder(nn.Module):
        """Permutation-invariant attention encoder over present table seats."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            dim = config.embedding_dim
            self.position_embedding = nn.Embedding(len(_POSITION_TO_ID), dim)
            self.last_action_embedding = nn.Embedding(len(_HISTORY_ACTION_TO_ID) + 1, dim)
            # Stack/commitment ratios, state flags and last-action amount.
            self.numeric_projection = nn.Sequential(nn.Linear(11, dim), nn.GELU(), nn.LayerNorm(dim))
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=config.attention_heads,
                dim_feedforward=dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.player_attention_layers)
            self.projection = nn.Sequential(nn.Linear(dim, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim))

        def forward(self, positions: Tensor, last_actions: Tensor, numeric: Tensor, mask: Tensor) -> Tensor:
            tokens = self.position_embedding(positions) + self.last_action_embedding(last_actions) + self.numeric_projection(numeric)
            encoded = self.encoder(tokens, src_key_padding_mask=~mask)
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.projection(pooled)


    class HistoryEncoder(nn.Module):
        """Small Transformer over public action-history tokens."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            dim = config.embedding_dim
            self.street_embedding = nn.Embedding(len(_STREET_TO_ID), dim)
            self.position_embedding = nn.Embedding(len(_POSITION_TO_ID), dim)
            self.action_embedding = nn.Embedding(len(_HISTORY_ACTION_TO_ID) + 1, dim)
            self.numeric_projection = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.LayerNorm(dim))
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=config.attention_heads,
                dim_feedforward=dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.history_layers)
            self.projection = nn.Sequential(nn.Linear(dim, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim))

        def forward(self, streets: Tensor, positions: Tensor, actions: Tensor, numeric: Tensor, mask: Tensor) -> Tensor:
            tokens = self.street_embedding(streets) + self.position_embedding(positions) + self.action_embedding(actions)
            tokens = tokens + self.numeric_projection(numeric)
            batch = tokens.shape[0]
            cls = self.cls.expand(batch, -1, -1)
            # The prepended CLS token means even a preflop observation with no
            # public actions has a valid Transformer sequence.
            encoded = self.encoder(torch.cat((cls, tokens), dim=1), src_key_padding_mask=torch.cat((torch.zeros((batch, 1), dtype=torch.bool, device=mask.device), ~mask), dim=1))
            return self.projection(encoded[:, 0])


    class PokerAgentModel(nn.Module):
        """Policy/value/equity model over legal player observations only."""

        observation_version = OBSERVATION_VERSION
        action_names = ACTION_NAMES
        bet_size_actions = BET_SIZE_ACTIONS

        def __init__(self, config: ModelConfig | None = None) -> None:
            super().__init__()
            self.config = config or ModelConfig()
            self.card_encoder = CardEncoder(self.config)
            self.player_set_encoder = PlayerSetEncoder(self.config)
            self.history_encoder = HistoryEncoder(self.config)
            # Hero has ten monetary features and table contributes five.
            self.global_projection = nn.Sequential(nn.Linear(15, self.config.hidden_dim), nn.GELU(), nn.LayerNorm(self.config.hidden_dim))
            fused = self.config.hidden_dim * 4
            self.backbone = nn.Sequential(
                nn.Linear(fused, self.config.hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(self.config.hidden_dim * 2),
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                nn.GELU(),
            )
            self.action_head = nn.Linear(self.config.hidden_dim, len(ACTION_NAMES))
            self.bet_size_head = nn.Linear(self.config.hidden_dim, len(BET_SIZE_ACTIONS))
            self.value_head = nn.Linear(self.config.hidden_dim, 1)
            self.equity_head = nn.Linear(self.config.hidden_dim, len(EQUITY_OUTCOMES))

        def forward(self, observations: Sequence[Mapping[str, object]]) -> ModelOutput:
            """Run one batched forward pass for active decision observations.

            Every observation must have at least one legal engine action.  This
            rejects non-actor views instead of silently producing a fictitious
            action distribution for a player who cannot act.
            """

            tensors = self.tensorize(observations, device=self._device())
            cards = self.card_encoder(tensors["card_ranks"], tensors["card_suits"], tensors["card_roles"], tensors["card_mask"])
            players = self.player_set_encoder(tensors["player_positions"], tensors["player_last_actions"], tensors["player_numeric"], tensors["player_mask"])
            history = self.history_encoder(tensors["history_streets"], tensors["history_positions"], tensors["history_actions"], tensors["history_numeric"], tensors["history_mask"])
            global_features = self.global_projection(tensors["global_numeric"])
            hidden = self.backbone(torch.cat((cards, players, history, global_features), dim=-1))
            raw_action_logits = self.action_head(hidden)
            raw_bet_logits = self.bet_size_head(hidden)
            action_logits = self._mask_logits(raw_action_logits, tensors["action_mask"])
            bet_size_logits = self._mask_logits(raw_bet_logits, tensors["bet_size_mask"], require_legal=False)
            equity_logits = self.equity_head(hidden)
            return ModelOutput(
                action_logits=action_logits,
                action_probabilities=torch.softmax(action_logits, dim=-1),
                bet_size_logits=bet_size_logits,
                bet_size_probabilities=self._masked_probabilities(bet_size_logits, tensors["bet_size_mask"]),
                value=self.value_head(hidden).squeeze(-1),
                equity_logits=equity_logits,
                equity_probabilities=torch.softmax(equity_logits, dim=-1),
                action_mask=tensors["action_mask"],
                bet_size_mask=tensors["bet_size_mask"],
            )

        @torch.no_grad()
        def infer(self, observation: Mapping[str, object]) -> InferenceDecision:
            """Return a deterministic argmax action plus UI-friendly heads."""

            was_training = self.training
            self.eval()
            output = self((observation,))
            if was_training:
                self.train()
            action_index = int(output.action_probabilities[0].argmax().item())
            action = ACTION_NAMES[action_index]
            bet_probabilities = {name: float(output.bet_size_probabilities[0, index].item()) for index, name in enumerate(BET_SIZE_ACTIONS)}
            if action == "raise":
                action = BET_SIZE_ACTIONS[int(output.bet_size_probabilities[0].argmax().item())]
            return InferenceDecision(
                action=action,
                action_probabilities={name: float(output.action_probabilities[0, index].item()) for index, name in enumerate(ACTION_NAMES)},
                bet_size_probabilities=bet_probabilities,
                value_bb=float(output.value[0].item()),
                equity={name: float(output.equity_probabilities[0, index].item()) for index, name in enumerate(EQUITY_OUTCOMES)},
            )

        def checkpoint_metadata(self) -> dict[str, object]:
            """Return compatibility metadata stored beside every state dict."""

            return {
                "model_version": MODEL_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "action_space": list(ACTION_NAMES),
                "bet_size_actions": list(BET_SIZE_ACTIONS),
                "equity_outcomes": list(EQUITY_OUTCOMES),
                "config": asdict(self.config),
            }

        def save_checkpoint(self, path: str | Path) -> None:
            torch.save({"metadata": self.checkpoint_metadata(), "state_dict": self.state_dict()}, Path(path))

        @classmethod
        def load_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "PokerAgentModel":
            checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
            metadata = checkpoint.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError("checkpoint has no metadata")
            cls._validate_metadata(metadata)
            config_data = metadata.get("config")
            if not isinstance(config_data, Mapping):
                raise ValueError("checkpoint has no model config")
            model = cls(ModelConfig(**dict(config_data)))
            model.load_state_dict(checkpoint["state_dict"])
            return model

        @staticmethod
        def _validate_metadata(metadata: Mapping[str, object]) -> None:
            expected = {
                "model_version": MODEL_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "action_space": list(ACTION_NAMES),
                "bet_size_actions": list(BET_SIZE_ACTIONS),
                "equity_outcomes": list(EQUITY_OUTCOMES),
            }
            for name, value in expected.items():
                if metadata.get(name) != value:
                    raise ValueError(f"incompatible checkpoint {name}: {metadata.get(name)!r}")

        @staticmethod
        def _mask_logits(logits: Tensor, mask: Tensor, *, require_legal: bool = True) -> Tensor:
            if require_legal and not bool(mask.any(dim=-1).all()):
                raise ValueError("each acting observation needs at least one legal action and raise sizing")
            return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

        @staticmethod
        def _masked_probabilities(logits: Tensor, mask: Tensor) -> Tensor:
            """Normalize only legal entries; no legal sizing means all zeros.

            A player may be unable to raise (for example, all-in).  In that
            case a bet-size distribution is intentionally absent rather than
            inventing an allowed size merely to make a softmax well-defined.
            """

            probabilities = torch.softmax(logits, dim=-1) * mask.to(logits.dtype)
            has_legal = mask.any(dim=-1, keepdim=True)
            return torch.where(has_legal, probabilities, torch.zeros_like(probabilities))

        def _device(self) -> torch.device:
            return next(self.parameters()).device

        @staticmethod
        def tensorize(observations: Sequence[Mapping[str, object]], *, device: torch.device | None = None) -> dict[str, Tensor]:
            """Convert variable-length canonical observations to padded tensors."""

            if not observations:
                raise ValueError("observations must not be empty")
            for observation in observations:
                if observation.get("schema_version") != OBSERVATION_VERSION:
                    raise ValueError("unsupported observation schema version")
            max_players = max(len(_as_list(observation, "player_set")) for observation in observations)
            max_history = max(1, max(len(_as_list(observation, "action_history")) for observation in observations))
            batch = len(observations)
            result: dict[str, Tensor] = {
                "card_ranks": torch.zeros((batch, 7), dtype=torch.long, device=device),
                "card_suits": torch.zeros((batch, 7), dtype=torch.long, device=device),
                "card_roles": torch.zeros((batch, 7), dtype=torch.long, device=device),
                "card_mask": torch.zeros((batch, 7), dtype=torch.bool, device=device),
                "player_positions": torch.zeros((batch, max_players), dtype=torch.long, device=device),
                "player_last_actions": torch.zeros((batch, max_players), dtype=torch.long, device=device),
                "player_numeric": torch.zeros((batch, max_players, 11), dtype=torch.float32, device=device),
                "player_mask": torch.zeros((batch, max_players), dtype=torch.bool, device=device),
                "history_streets": torch.zeros((batch, max_history), dtype=torch.long, device=device),
                "history_positions": torch.zeros((batch, max_history), dtype=torch.long, device=device),
                "history_actions": torch.zeros((batch, max_history), dtype=torch.long, device=device),
                "history_numeric": torch.zeros((batch, max_history, 5), dtype=torch.float32, device=device),
                "history_mask": torch.zeros((batch, max_history), dtype=torch.bool, device=device),
                "global_numeric": torch.zeros((batch, 15), dtype=torch.float32, device=device),
                "action_mask": torch.zeros((batch, len(ACTION_NAMES)), dtype=torch.bool, device=device),
                "bet_size_mask": torch.zeros((batch, len(BET_SIZE_ACTIONS)), dtype=torch.bool, device=device),
            }
            for batch_index, observation in enumerate(observations):
                PokerAgentModel._fill_tensors(result, batch_index, observation)
            return result

        @staticmethod
        def _fill_tensors(result: dict[str, Tensor], batch_index: int, observation: Mapping[str, object]) -> None:
            cards = _as_mapping(observation, "cards")
            card_values = [(card, 1) for card in _as_list(cards, "hole_cards")] + [(card, 2) for card in _as_list(cards, "board")]
            if len(card_values) > 7:
                raise ValueError("observation contains more than seven known cards")
            for card_index, (card, role) in enumerate(card_values):
                rank, suit = _card_ids(card)
                result["card_ranks"][batch_index, card_index] = rank
                result["card_suits"][batch_index, card_index] = suit
                result["card_roles"][batch_index, card_index] = role
                result["card_mask"][batch_index, card_index] = True

            players = _as_list(observation, "player_set")
            supplied_mask = _as_list(observation, "player_mask")
            if len(players) != len(supplied_mask):
                raise ValueError("player_mask must match player_set length")
            for player_index, (player, present) in enumerate(zip(players, supplied_mask, strict=True)):
                if not isinstance(present, bool):
                    raise ValueError("player_mask must contain bools")
                if not present:
                    continue
                player_data = _mapping_value(player, "player_set item")
                result["player_positions"][batch_index, player_index] = _position_id(player_data["position"])
                result["player_last_actions"][batch_index, player_index] = _history_action_id(player_data.get("last_action"), allow_none=True)
                result["player_numeric"][batch_index, player_index] = torch.tensor(
                    [
                        _number(player_data, "stack_bb"),
                        _number(player_data, "stack_to_pot"),
                        _number(player_data, "committed_street_bb"),
                        _number(player_data, "committed_street_to_pot"),
                        _number(player_data, "committed_total_bb"),
                        _number(player_data, "committed_total_to_pot"),
                        float(_bool(player_data, "is_hero")),
                        float(_bool(player_data, "folded")),
                        float(_bool(player_data, "all_in")),
                        float(_bool(player_data, "active")),
                        _number(player_data, "last_action_amount_to_pot"),
                    ],
                    dtype=torch.float32,
                    device=result["player_numeric"].device,
                )
                result["player_mask"][batch_index, player_index] = True
            if not bool(result["player_mask"][batch_index].any()):
                raise ValueError("observation must contain one present player")

            history = _as_list(observation, "action_history")
            for history_index, record in enumerate(history):
                record_data = _mapping_value(record, "action_history item")
                result["history_streets"][batch_index, history_index] = _street_id(record_data["street"])
                result["history_positions"][batch_index, history_index] = _position_id(record_data["position"])
                result["history_actions"][batch_index, history_index] = _history_action_id(record_data["action"])
                result["history_numeric"][batch_index, history_index] = torch.tensor(
                    [
                        _number(record_data, "amount_bb"),
                        _number(record_data, "amount_to_pot"),
                        _optional_number(record_data, "raise_to_bb"),
                        _optional_number(record_data, "raise_to_to_pot"),
                        _number(record_data, "current_bet_after_bb"),
                    ],
                    dtype=torch.float32,
                    device=result["history_numeric"].device,
                )
                result["history_mask"][batch_index, history_index] = True

            hero = _as_mapping(observation, "hero")
            table = _as_mapping(observation, "table")
            result["global_numeric"][batch_index] = torch.tensor(
                [
                    *[_number(hero, name) for name in ("stack_bb", "stack_to_pot", "committed_street_bb", "committed_street_to_pot", "committed_total_bb", "committed_total_to_pot", "to_call_bb", "to_call_to_pot", "min_raise_to_bb", "min_raise_to_pot")],
                    *[_number(table, name) for name in ("pot_bb", "current_bet_bb", "last_full_raise_bb", "active_player_count", "actionable_player_count")],
                ],
                dtype=torch.float32,
                device=result["global_numeric"].device,
            )
            legal = _as_mapping(observation, "legal_action_mask")
            for index, action in enumerate(ACTION_NAMES[:3]):
                result["action_mask"][batch_index, index] = _legal(legal, action)
            raise_mask = [_legal(legal, action) for action in BET_SIZE_ACTIONS]
            result["action_mask"][batch_index, 3] = any(raise_mask)
            result["bet_size_mask"][batch_index] = torch.tensor(raise_mask, dtype=torch.bool, device=result["bet_size_mask"].device)

else:

    class _UnavailableTorchComponent:  # pragma: no cover - trivial optional dependency path.
        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_torch()

    CardEncoder = _UnavailableTorchComponent
    PlayerSetEncoder = _UnavailableTorchComponent
    HistoryEncoder = _UnavailableTorchComponent

    class PokerAgentModel:  # pragma: no cover - trivial optional-dependency path.
        """Placeholder that reports the missing optional RL dependency."""

        def __init__(self, config: ModelConfig | None = None) -> None:
            _require_torch()


def _as_mapping(parent: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _mapping_value(parent.get(name), name)


def _mapping_value(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_list(parent: Mapping[str, object], name: str) -> list[object]:
    value = parent.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _number(mapping: Mapping[str, object], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _optional_number(mapping: Mapping[str, object], name: str) -> float:
    value = mapping.get(name)
    return 0.0 if value is None else _number(mapping, name)


def _bool(mapping: Mapping[str, object], name: str) -> bool:
    value = mapping.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _card_ids(value: object) -> tuple[int, int]:
    if not isinstance(value, str) or len(value) != 2 or value[0] not in _RANK_TO_ID or value[1] not in _SUIT_TO_ID:
        raise ValueError(f"invalid card encoding: {value!r}")
    return _RANK_TO_ID[value[0]], _SUIT_TO_ID[value[1]]


def _position_id(value: object) -> int:
    if not isinstance(value, str) or value not in _POSITION_TO_ID:
        raise ValueError(f"invalid position: {value!r}")
    return _POSITION_TO_ID[value]


def _street_id(value: object) -> int:
    if not isinstance(value, str) or value not in _STREET_TO_ID:
        raise ValueError(f"invalid street: {value!r}")
    return _STREET_TO_ID[value]


def _history_action_id(value: object, *, allow_none: bool = False) -> int:
    if value is None and allow_none:
        return 0
    if not isinstance(value, str) or value not in _HISTORY_ACTION_TO_ID:
        raise ValueError(f"invalid action: {value!r}")
    return _HISTORY_ACTION_TO_ID[value]


def _legal(legal_actions: Mapping[str, object], action: str) -> bool:
    value = legal_actions.get(action)
    if not isinstance(value, bool):
        raise ValueError(f"legal_action_mask[{action!r}] must be bool")
    return value


__all__ = [
    "ACTION_NAMES",
    "BET_SIZE_ACTIONS",
    "EQUITY_OUTCOMES",
    "MODEL_VERSION",
    "TORCH_AVAILABLE",
    "CardEncoder",
    "HistoryEncoder",
    "InferenceDecision",
    "ModelConfig",
    "ModelOutput",
    "PokerAgentModel",
    "PlayerSetEncoder",
]
