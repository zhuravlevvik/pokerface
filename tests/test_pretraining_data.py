"""Acceptance tests for the inspectable Stage 1 pretraining corpus."""

from __future__ import annotations

import json

import pytest

from poker.curriculum import CurriculumStage
from poker.pretraining_data import (
    PRETRAINING_CORPUS_SCHEMA_VERSION,
    SeedRange,
    balanced_indices,
    equity_bucket,
    generate_pretraining_corpus,
    load_pretraining_corpus,
    write_pretraining_corpus,
)


def _corpus():
    return generate_pretraining_corpus(
        CurriculumStage.A_HEADS_UP_STARTER,
        train_seed_range=SeedRange(100, 5),
        holdout_seed_range=SeedRange(10_000, 3),
        equity_samples=1,
    )


def test_generation_is_reproducible_and_splits_by_disjoint_hand_seed_ranges() -> None:
    first = _corpus()
    second = _corpus()
    assert first == second
    assert not first.train_seed_range.overlaps(first.holdout_seed_range)
    assert {record.example.seed for record in first.train} <= set(range(100, 105))
    assert {record.example.seed for record in first.holdout} <= set(range(10_000, 10_003))
    assert {record.example.seed for record in first.train}.isdisjoint({record.example.seed for record in first.holdout})
    assert all(record.example.hand_id is not None for record in (*first.train, *first.holdout))


def test_baseline_policy_identity_is_stable_for_one_player_hand() -> None:
    corpus = _corpus()
    policies_by_player_hand: dict[tuple[int | None, int], set[str]] = {}
    for record in corpus.train:
        seat = record.example.observation["hero"]["seat"]
        assert isinstance(seat, int)
        policies_by_player_hand.setdefault((record.example.seed, seat), set()).add(record.behavior_policy)
    assert policies_by_player_hand
    assert all(len(policies) == 1 for policies in policies_by_player_hand.values())


def test_jsonl_round_trip_is_atomic_format_with_auditable_metadata(tmp_path) -> None:
    corpus = _corpus()
    path = tmp_path / "stage-a.jsonl"
    write_pretraining_corpus(corpus, path)
    assert load_pretraining_corpus(path) == corpus
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["schema_version"] == PRETRAINING_CORPUS_SCHEMA_VERSION
    assert rows[0]["equity_label_protocol"] == "fixed_deal_virtual_showdown_v1"
    assert rows[0]["train_equity_samples"] == 1
    assert rows[0]["holdout_equity_samples"] == 1
    example = rows[1]
    assert {"hand_id", "hand_seed", "behavior_policy", "street", "table_player_count", "active_opponent_count", "equity_samples", "equity_exact", "label_protocol"} <= set(example)


def test_serialized_rows_do_not_contain_opponent_hole_cards_or_private_deck(tmp_path) -> None:
    path = tmp_path / "safe.jsonl"
    write_pretraining_corpus(_corpus(), path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    for row in rows:
        observation = row["observation"]
        # Hero cards are intentional policy input; players/sets carry no cards.
        assert len(observation["cards"]["hole_cards"]) == 2
        assert all("hole_cards" not in player and "deck" not in player for player in observation["players"])
        assert all("hole_cards" not in player and "deck" not in player for player in observation["player_set"])
        encoded = json.dumps(row, sort_keys=True)
        assert "opponent_hole_cards" not in encoded
        assert "remaining_deck" not in encoded
        assert "equity_snapshot" not in encoded


def test_strata_summary_and_balanced_indices_preserve_rare_cells() -> None:
    corpus = _corpus()
    summary = corpus.summary()["train"]
    assert summary
    first = balanced_indices(corpus.train, seed=77)
    second = balanced_indices(corpus.train, seed=77)
    assert first == second
    assert set(first) == set(range(len(corpus.train)))
    assert sum(summary.values()) == len(corpus.train)
    # All requested sampling positions are deterministic and legal indices.
    sampled = balanced_indices(corpus.train, count=len(corpus.train) + 7, seed=3)
    assert len(sampled) == len(corpus.train) + 7
    assert all(0 <= index < len(corpus.train) for index in sampled)


def test_overlapping_seed_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        generate_pretraining_corpus("A", train_seed_range=SeedRange(1, 3), holdout_seed_range=SeedRange(3, 2), equity_samples=1)


def test_holdout_can_use_less_noisy_equity_labels_than_train() -> None:
    corpus = generate_pretraining_corpus(
        train_seed_range=SeedRange(20, 1),
        holdout_seed_range=SeedRange(40, 1),
        train_equity_samples=1,
        holdout_equity_samples=8,
    )
    assert corpus.stage is CurriculumStage.A_HEADS_UP_STARTER
    assert corpus.train_equity_samples == 1
    assert corpus.holdout_equity_samples == 8
    assert corpus.equity_samples is None


def test_multiway_bucket_uses_win_probability_not_heads_up_tie_score() -> None:
    # In a three-way hand a tie is not treated as a fixed half-pot share.
    assert equity_bucket((0.0, 1.0, 0.0), active_player_count=2) == "0.4-0.6"
    assert equity_bucket((0.0, 1.0, 0.0), active_player_count=3) == "0.0-0.2"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "holdout", "outside declared split"),
        ("equity_samples", 2, "sample count contradicts"),
        ("equity_target", [-0.1, 0.1, 1.0], "invalid equity target"),
    ],
)
def test_loader_rejects_split_contamination_and_false_label_provenance(tmp_path, field, value, message) -> None:
    path = tmp_path / "tampered.jsonl"
    write_pretraining_corpus(_corpus(), path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1][field] = value
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_pretraining_corpus(path)
