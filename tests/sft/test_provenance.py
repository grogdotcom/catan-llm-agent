"""Provenance and sampling tests (§5, §7)."""

import json

from catan_llm.llm.sft.provenance import (
    annotate_record_with_provenance,
    derive_trajectory_id,
    sample_winning_trajectories,
    seat_index_for_color,
)


def test_derive_trajectory_id_stable():
    a = derive_trajectory_id(12345, ["RED", "BLUE", "ORANGE", "WHITE"])
    b = derive_trajectory_id(12345, ["RED", "BLUE", "ORANGE", "WHITE"])
    assert a == b
    c = derive_trajectory_id(12346, ["RED", "BLUE", "ORANGE", "WHITE"])
    assert a != c


def test_seat_index_for_color():
    order = ["RED", "BLUE", "ORANGE", "WHITE"]
    assert seat_index_for_color("BLUE", order) == 1
    assert seat_index_for_color("WHITE", order) == 3


def test_annotate_record():
    rec = {"prompt": "hello", "completion": "2", "color": "BLUE", "winner": "BLUE", "game_id": 5}
    out = annotate_record_with_provenance(
        rec,
        trajectory_id="seed-123-abc",
        game_seed=123,
        game_end_turn=72,
        seat_index=1,
        trajectory_index=3,
    )
    assert out["trajectory_id"] == "seed-123-abc"
    assert out["game_seed"] == 123
    assert out["game_end_turn"] == 72
    assert out["seat_index"] == 1
    assert out["trajectory_index"] == 3
    assert out["color"] == "BLUE"
    # original not mutated
    assert "trajectory_id" not in rec


def test_sample_winning_trajectories_stratified():
    # Create 400 trajectories: 100 per seat
    trajs = []
    for seat in range(4):
        color = ["RED", "BLUE", "ORANGE", "WHITE"][seat]
        for i in range(100):
            trajs.append(
                {
                    "trajectory_id": f"t-{seat}-{i}",
                    "winner": color,
                    "seat_order": ["RED", "BLUE", "ORANGE", "WHITE"],
                    "winner_seat_index": seat,
                }
            )
    sampled = sample_winning_trajectories(trajs, sample_size=500, seed=42)
    # We only have 400, so sample should be 400 (capped by population)
    assert len(sampled) == 400
    # Check stratification when enough: sample 8 from 400 -> ~2 per seat
    sampled2 = sample_winning_trajectories(trajs, sample_size=8, seed=42)
    from collections import Counter

    by_seat = Counter(t["winner_seat_index"] for t in sampled2)
    # Should be roughly even: each seat 2
    assert all(v == 2 for v in by_seat.values())

    # Deterministic
    sampled3 = sample_winning_trajectories(trajs, sample_size=8, seed=42)
    assert [t["trajectory_id"] for t in sampled2] == [t["trajectory_id"] for t in sampled3]
    sampled4 = sample_winning_trajectories(trajs, sample_size=8, seed=43)
    assert [t["trajectory_id"] for t in sampled2] != [t["trajectory_id"] for t in sampled4]


def test_sample_target_500():
    trajs = []
    for seat in range(4):
        color = ["RED", "BLUE", "ORANGE", "WHITE"][seat]
        for i in range(200):
            trajs.append(
                {"trajectory_id": f"t-{seat}-{i}", "winner": color, "seat_order": ["RED", "BLUE", "ORANGE", "WHITE"]}
            )
    sampled = sample_winning_trajectories(trajs, sample_size=500, seed=123)
    assert len(sampled) == 500
    from collections import Counter

    by_seat = Counter(json.loads(t["seat_order_json"]) if "seat_order_json" in t else t["winner"] for t in sampled)
    # Winner distribution: count per color
    from collections import Counter as C

    by_color = C(t["winner"] for t in sampled)
    # Should be ~125 per seat (800 total, sample 500)
    for color in ["RED", "BLUE", "ORANGE", "WHITE"]:
        assert 100 <= by_color[color] <= 150
