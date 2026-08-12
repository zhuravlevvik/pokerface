from __future__ import annotations

from poker.bots import AggroBot, CallingStationBot, RandomBot, RuleBot, TightBot
from poker.game_state import HandState
from poker.observation import observation_for
from poker.tournament import run_tournament


def _play_bot_only(bot, *, seed: int) -> None:
    """Use one policy for every seat and assert each selected action is legal."""

    state = HandState(seed=seed)
    while not state.complete:
        seat = state.actor
        assert seat is not None
        observation = observation_for(state, seat)
        action = bot.select_action(observation, observation["legal_actions"])
        assert state.legal_actions(seat)[action]
        state.step(action)


def test_every_baseline_bot_only_selects_legal_actions() -> None:
    for index, bot in enumerate((RandomBot(seed=1), TightBot(), AggroBot(seed=2), CallingStationBot(seed=3), RuleBot())):
        for seed in range(index * 8, index * 8 + 8):
            _play_bot_only(bot, seed=seed)


def test_seeded_tournament_is_reproducible() -> None:
    def entrants():
        return {
            "random": RandomBot(seed=10),
            "aggro": AggroBot(seed=11),
            "station": CallingStationBot(seed=12),
            "tight": TightBot(),
            "rule": RuleBot(),
        }

    first = run_tournament(entrants(), 40, seed_start=300)
    second = run_tournament(entrants(), 40, seed_start=300)
    assert first == second


def test_baseline_statistics_show_intended_play_styles() -> None:
    result = run_tournament(
        {
            "tight": TightBot(),
            "aggro": AggroBot(seed=20),
            "station": CallingStationBot(seed=21),
            "random": RandomBot(seed=22),
            "rule": RuleBot(),
        },
        150,
        seed_start=600,
    )
    tight = result.statistics["tight"]
    aggro = result.statistics["aggro"]
    station = result.statistics["station"]
    assert tight.vpip < station.vpip
    assert tight.pfr < aggro.pfr
    assert station.aggression_factor < aggro.aggression_factor
    assert 0 <= aggro.three_bet <= 1


def test_rule_bot_beats_random_pool_on_fixed_sufficient_sample() -> None:
    result = run_tournament(
        {
            "rule": RuleBot(),
            "random_1": RandomBot(seed=31),
            "random_2": RandomBot(seed=32),
            "random_3": RandomBot(seed=33),
            "random_4": RandomBot(seed=34),
        },
        400,
        seed_start=900,
    )
    assert result.pnl_bb["rule"] > 0
    assert result.pnl_bb["rule"] > max(
        result.pnl_bb[name] for name in result.pnl_bb if name.startswith("random_")
    )
