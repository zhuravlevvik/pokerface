from poker.betting import Action
from poker.simulator import BatchedHoldemEnvironment, benchmark_hands
from poker.traces import rebuild_hand


def _check_or_call(_observation: dict, mask: dict[str, bool], _table: int) -> Action:
    return Action.CHECK if mask[Action.CHECK.value] else Action.CALL


def _play_batch_to_terminal(environment: BatchedHoldemEnvironment, observations: tuple[dict, ...]):
    result = None
    while not all(environment.terminal):
        actions = [
            None if done else _check_or_call(observation, environment.legal_action_masks[index], index)
            for index, (done, observation) in enumerate(zip(environment.terminal, observations))
        ]
        result = environment.step(actions)
        observations = result.observations
    return result


def test_batched_simulator_is_reproducible_and_returns_terminal_rewards() -> None:
    first = BatchedHoldemEnvironment(2, capture_replays=True)
    second = BatchedHoldemEnvironment(2, capture_replays=True)
    first_result = _play_batch_to_terminal(first, first.reset(seeds=[11, 12]))
    second_result = _play_batch_to_terminal(second, second.reset(seeds=[11, 12]))

    assert first_result is not None
    assert first_result.terminal == (True, True)
    assert first_result.rewards == second_result.rewards
    assert sum(first_result.rewards[0].values()) == 0
    assert [info["replay"] for info in first_result.infos] == [info["replay"] for info in second_result.infos]


def test_trace_captures_pre_action_safe_observation_and_terminal_pnl() -> None:
    environment = BatchedHoldemEnvironment(1, capture_replays=True)
    result = _play_batch_to_terminal(environment, environment.reset(seeds=[19]))
    trace = result.infos[0]["trace"]

    assert trace is not None
    assert trace.decisions
    first = trace.decisions[0]
    assert first.selected_action == Action.CALL.value
    assert first.action_log == []
    assert first.hero_seat == 3
    assert first.legal_action_mask == first.observation["legal_actions"]
    assert all("hole_cards" not in player for player in first.observation["players"])
    assert first.terminal_pnl_bb == trace.terminal_pnl_bb[first.hero_seat]
    assert first.expected_showdown_share_target is not None
    assert 0.0 <= first.expected_showdown_share_target <= 1.0
    assert len(trace.as_training_records()) == len(trace.decisions)


def test_selected_replay_rebuilds_exact_terminal_hand() -> None:
    environment = BatchedHoldemEnvironment(1, capture_replays=True)
    result = _play_batch_to_terminal(environment, environment.reset(seeds=[23]))
    replay = result.infos[0]["replay"]
    rebuilt = rebuild_hand(replay)

    assert rebuilt.replay() == replay


def test_benchmark_generates_multiple_hands_without_ui_or_builtin_bot() -> None:
    benchmark = benchmark_hands(12, _check_or_call, batch_size=3, seed_start=100)

    assert benchmark.hands == 12
    assert benchmark.decisions > 0
    assert benchmark.seconds >= 0
    assert benchmark.hands_per_second > 0
