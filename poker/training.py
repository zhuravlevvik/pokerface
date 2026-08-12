"""Executable on-policy PPO/GAE primitives for discrete Hold'em self-play.

This module is deliberately small enough to inspect.  It trains one shared
network playing any number of seats; an :class:`OpponentLeague` supplies
frozen snapshots and baselines for the other seats.  No hidden opponent cards
are passed to the policy: virtual-showdown labels arrive only from completed
``HandTrace`` training records.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .betting import Action
from .league import ModelPolicy, OpponentLeague
from .model import ACTION_NAMES, BET_SIZE_ACTIONS, TORCH_AVAILABLE, PokerAgentModel
from .simulator import BatchedHoldemEnvironment

if TORCH_AVAILABLE:
    import torch
    from torch import Tensor
    from torch.nn import functional as F


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for poker.training; install the project with `.[rl]`.")


@dataclass(frozen=True)
class PPOConfig:
    """Conservative hyperparameters for the first self-play implementation."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    equity_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    epochs: int = 2
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    equity_samples: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
        if min(self.value_coefficient, self.equity_coefficient, self.entropy_coefficient, self.learning_rate) < 0:
            raise ValueError("PPO coefficients and learning_rate must be non-negative")
        if self.epochs < 1 or self.minibatch_size < 1 or self.equity_samples < 1:
            raise ValueError("epochs, minibatch_size and equity_samples must be positive")


@dataclass
class RolloutStep:
    """One current-policy decision with immutable old-policy statistics."""

    hand_id: int
    seat: int
    order: int
    observation: dict[str, Any]
    action_index: int
    bet_size_index: int  # -1 when the selected action is not raise.
    old_log_probability: float
    old_value: float
    equity_target: tuple[float, float, float] | None = None
    reward: float = 0.0
    terminal: bool = False
    advantage: float = 0.0
    return_: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.action_index < len(ACTION_NAMES):
            raise ValueError("invalid action index")
        if self.action_index == 3 and not 0 <= self.bet_size_index < len(BET_SIZE_ACTIONS):
            raise ValueError("raise action needs a legal size index")
        if self.action_index != 3 and self.bet_size_index != -1:
            raise ValueError("only raises may carry a sizing index")
        if not isfinite(self.old_log_probability) or not isfinite(self.old_value):
            raise ValueError("old policy statistics must be finite")


@dataclass(frozen=True)
class Rollout:
    steps: tuple[RolloutStep, ...]
    hands: int

    @property
    def decisions(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class UpdateMetrics:
    samples: int
    total_loss: float
    policy_loss: float
    value_loss: float
    equity_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


def factorized_logprob_and_entropy(output, action_indices: Tensor, bet_size_indices: Tensor) -> tuple[Tensor, Tensor]:
    """Return log P(action,size) and entropy for the full legal action space.

    A raise is one action type plus a conditional sizing choice.  Its PPO
    ratio must therefore include *both* log probabilities; non-raises carry no
    fictional sizing term.  ``output`` is the model's already legal-masked
    output, making illegal choices impossible in both collection and updates.
    """

    _require_torch()
    action_log_probs = F.log_softmax(output.action_logits, dim=-1)
    action_logprob = action_log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    action_entropy = -(output.action_probabilities * action_log_probs).sum(dim=-1)
    is_raise = action_indices == 3
    size_logprob = torch.zeros_like(action_logprob)
    # Entropy is a property of the *policy*, not of the sampled action.  It
    # therefore includes a legal raise-sizing distribution even for rollout
    # rows which happened to check/call/fold.
    all_size_log_probs = F.log_softmax(output.bet_size_logits, dim=-1)
    conditional_size_entropy = -(output.bet_size_probabilities * all_size_log_probs).sum(dim=-1)
    conditional_size_entropy = torch.where(output.bet_size_mask.any(dim=-1), conditional_size_entropy, torch.zeros_like(conditional_size_entropy))
    if bool(is_raise.any()):
        selected_sizes = bet_size_indices[is_raise]
        if bool((selected_sizes < 0).any()):
            raise ValueError("raised rollout samples need a sizing action")
        size_logprob[is_raise] = all_size_log_probs[is_raise].gather(1, selected_sizes.unsqueeze(1)).squeeze(1)
    # Entropy of the joint distribution: H(type) + P(raise) * H(size|raise).
    joint_entropy = action_entropy + output.action_probabilities[:, 3] * conditional_size_entropy
    return action_logprob + size_logprob, joint_entropy


def compute_gae(steps: Sequence[RolloutStep], *, gamma: float, gae_lambda: float) -> None:
    """Populate GAE for one player trajectory ordered by own decision time."""

    if not steps:
        return
    advantage = 0.0
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        if step.terminal:
            next_value = 0.0
            next_advantage = 0.0
        else:
            if index + 1 >= len(steps):
                raise ValueError("non-terminal rollout trajectory has no next decision")
            next_value = steps[index + 1].old_value
            next_advantage = advantage
        delta = step.reward + gamma * next_value - step.old_value
        advantage = delta + gamma * gae_lambda * next_advantage
        step.advantage = advantage
        step.return_ = advantage + step.old_value


class PPOTrainer:
    """Collect on-policy league rollouts and update the shared neural agent."""

    def __init__(self, model: PokerAgentModel, league: OpponentLeague, config: PPOConfig | None = None, *, device: str | None = None) -> None:
        _require_torch()
        if league.current.model is not model:
            raise ValueError("league current policy must reference the trainable model")
        self.model = model
        self.league = league
        self.config = config or PPOConfig()
        if device is not None:
            self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self._seed_counter = 0

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _select_current(self, observation: Mapping[str, object]) -> tuple[Action, int, int, float, float]:
        was_training = self.model.training
        self.model.eval()
        output = self.model([observation])
        action_index = int(torch.multinomial(output.action_probabilities[0], 1).item())
        bet_size_index = -1
        log_probability = torch.log(output.action_probabilities[0, action_index])
        if action_index == 3:
            if not bool(output.bet_size_mask[0].any()):
                raise RuntimeError("model sampled raise without a legal raise size")
            bet_size_index = int(torch.multinomial(output.bet_size_probabilities[0], 1).item())
            log_probability = log_probability + torch.log(output.bet_size_probabilities[0, bet_size_index])
            action = Action(BET_SIZE_ACTIONS[bet_size_index])
        else:
            action = Action(ACTION_NAMES[action_index])
        if was_training:
            self.model.train()
        return action, action_index, bet_size_index, float(log_probability.item()), float(output.value[0].item())

    def collect_rollout(
        self,
        hand_count: int,
        *,
        table_count: int = 8,
        player_count: int = 5,
        starting_stack: int = 10_000,
        allowed_raise_actions: frozenset[Action] | None = None,
    ) -> Rollout:
        """Play exactly ``hand_count`` hands and return current-policy samples.

        PnL is delayed until each hand terminates.  We then turn each seat's
        own sequence of decisions into a sparse-reward trajectory and attach
        the trace's virtual-showdown target.  Opponent decisions are never
        added to PPO data unless that opponent is the current shared network.
        """

        if hand_count < 1 or table_count < 1:
            raise ValueError("hand_count and table_count must be positive")
        all_steps: list[RolloutStep] = []
        completed_hands = 0
        while completed_hands < hand_count:
            batch_size = min(table_count, hand_count - completed_hands)
            environment = BatchedHoldemEnvironment(
                batch_size,
                player_count=player_count,
                starting_stack=starting_stack,
                allowed_raise_actions=allowed_raise_actions,
                equity_samples=self.config.equity_samples,
            )
            observations = environment.reset(seeds=range(self._seed_counter, self._seed_counter + batch_size))
            self._seed_counter += batch_size
            seatings = [self.league.sample_seating(player_count) for _ in range(batch_size)]
            per_table: list[list[RolloutStep]] = [[] for _ in range(batch_size)]
            active = [True] * batch_size
            while any(active):
                actions: list[Action | None] = []
                for table, observation in enumerate(observations):
                    if not active[table]:
                        actions.append(None)
                        continue
                    if observation is None:
                        raise RuntimeError("live table has no actor observation")
                    seat = environment.states[table].actor
                    if seat is None:
                        raise RuntimeError("live table has no actor")
                    policy = seatings[table][seat]
                    legal = observation["legal_actions"]
                    if not isinstance(legal, Mapping):
                        raise ValueError("observation has malformed legal action mask")
                    if policy.name == self.league.current_name:
                        action, action_index, size_index, old_logprob, old_value = self._select_current(observation)
                        per_table[table].append(
                            RolloutStep(
                                hand_id=environment.traces[table].hand_id,
                                seat=seat,
                                order=len(environment.traces[table].decisions),
                                observation=dict(observation),
                                action_index=action_index,
                                bet_size_index=size_index,
                                old_log_probability=old_logprob,
                                old_value=old_value,
                            )
                        )
                    else:
                        action = policy.select_action(observation, legal)  # type: ignore[arg-type]
                    if not bool(legal.get(action.value, False)):
                        raise ValueError(f"league member {policy.name!r} selected illegal action {action.value!r}")
                    actions.append(action)
                result = environment.step(actions)
                observations = result.observations
                for table, done in enumerate(result.terminal):
                    if not done or not active[table]:
                        continue
                    active[table] = False
                    trace = result.infos[table]["trace"]
                    if trace is None:
                        raise RuntimeError("terminal table did not return a hand trace")
                    trace_by_order = {index: decision for index, decision in enumerate(trace.decisions)}
                    for step in per_table[table]:
                        decision = trace_by_order[step.order]
                        target = decision.equity_target
                        if target is None or len(target) != 3:
                            raise RuntimeError("terminal current-policy trace has no equity target")
                        step.equity_target = (float(target[0]), float(target[1]), float(target[2]))
                    self._finish_hand(per_table[table], result.rewards[table])
                    all_steps.extend(per_table[table])
            completed_hands += batch_size
        return Rollout(tuple(all_steps), completed_hands)

    def _finish_hand(self, steps: list[RolloutStep], rewards: Mapping[int, float]) -> None:
        by_seat: dict[int, list[RolloutStep]] = {}
        for step in steps:
            by_seat.setdefault(step.seat, []).append(step)
        for seat_steps in by_seat.values():
            seat_steps.sort(key=lambda item: item.order)
            last = seat_steps[-1]
            last.reward = float(rewards[last.seat])
            last.terminal = True
            compute_gae(seat_steps, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)

    def update(self, rollout: Rollout) -> UpdateMetrics:
        """Run PPO minibatches against the fixed old rollout policy."""

        if not rollout.steps:
            raise ValueError("rollout contains no current-policy decisions")
        if any(step.equity_target is None for step in rollout.steps):
            raise ValueError("rollout must be terminal and carry equity targets")
        self.model.train()
        advantages = torch.tensor([step.advantage for step in rollout.steps], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        returns = torch.tensor([step.return_ for step in rollout.steps], dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor([step.old_log_probability for step in rollout.steps], dtype=torch.float32, device=self.device)
        actions = torch.tensor([step.action_index for step in rollout.steps], dtype=torch.long, device=self.device)
        sizes = torch.tensor([step.bet_size_index for step in rollout.steps], dtype=torch.long, device=self.device)
        targets = torch.tensor([step.equity_target for step in rollout.steps], dtype=torch.float32, device=self.device)
        totals = {"total": 0.0, "policy": 0.0, "value": 0.0, "equity": 0.0, "entropy": 0.0, "kl": 0.0, "clip": 0.0}
        updates = 0
        count = len(rollout.steps)
        for _ in range(self.config.epochs):
            for indices in torch.randperm(count, device=self.device).split(self.config.minibatch_size):
                observation_batch = [rollout.steps[int(index)].observation for index in indices.cpu().tolist()]
                output = self.model(observation_batch)
                new_logprobs, entropy = factorized_logprob_and_entropy(output, actions[indices], sizes[indices])
                ratio = torch.exp(new_logprobs - old_logprobs[indices])
                surrogate_a = ratio * advantages[indices]
                surrogate_b = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[indices]
                policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
                value_loss = F.mse_loss(output.value, returns[indices])
                equity_loss = -(targets[indices] * F.log_softmax(output.equity_logits, dim=-1)).sum(dim=-1).mean()
                total_loss = policy_loss + self.config.value_coefficient * value_loss + self.config.equity_coefficient * equity_loss - self.config.entropy_coefficient * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    totals["total"] += float(total_loss.item())
                    totals["policy"] += float(policy_loss.item())
                    totals["value"] += float(value_loss.item())
                    totals["equity"] += float(equity_loss.item())
                    totals["entropy"] += float(entropy.mean().item())
                    totals["kl"] += float((old_logprobs[indices] - new_logprobs).mean().item())
                    totals["clip"] += float(((ratio - 1.0).abs() > self.config.clip_ratio).float().mean().item())
                updates += 1
        return UpdateMetrics(
            samples=count,
            total_loss=totals["total"] / updates,
            policy_loss=totals["policy"] / updates,
            value_loss=totals["value"] / updates,
            equity_loss=totals["equity"] / updates,
            entropy=totals["entropy"] / updates,
            approximate_kl=totals["kl"] / updates,
            clip_fraction=totals["clip"] / updates,
        )

    def train_iteration(self, hand_count: int, **rollout_kwargs: Any) -> tuple[Rollout, UpdateMetrics]:
        rollout = self.collect_rollout(hand_count, **rollout_kwargs)
        return rollout, self.update(rollout)


__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "Rollout",
    "RolloutStep",
    "UpdateMetrics",
    "compute_gae",
    "factorized_logprob_and_entropy",
]
