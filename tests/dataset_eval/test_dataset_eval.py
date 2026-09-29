"""Dataset exporter + evaluation metrics/reporting."""

from catan_llm.llm.dataset import build_sft_record, prompt_with_strategy, split_summary
from catan_llm.llm.evaluation import acceptance, accuracy, rejection_reasons, run_report, summarize


def test_prompt_with_strategy_inserts():
    p = prompt_with_strategy("x\n\n[RECENT TURNS]\nhist", "Plan")
    assert "[CURRENT STRATEGY]\nPlan\n\n[RECENT TURNS]" in p


def test_build_sft_record_shape():
    cp = {
        "checkpoint_id": "run-x-0",
        "run_id": "run",
        "trajectory_id": "t",
        "seat_index": 2,
        "checkpoint_index": 1,
        "checkpoint_count": 4,
        "band": 0,
        "normalized_progress": 0.25,
        "strategy_in": "Go ore",
        "strategy_source_checkpoint": None,
        "opportunity_id": "opp1",
    }
    out = {
        "raw_output_text": " thinkingt response<strategy>s</strategy><action>2</action>",
        "predicted_action": 2,
        "strategy_out": "s",
        "think_text": "t",
        "validation_status": "accepted",
    }
    opp = {"engine_completion": "2", "color": "RED", "turn": 5, "phase": "PLAY_TURN", "game_id": 1, "trajectory_index": 3}
    traj = {"game_end_turn": 72, "game_seed": 9, "winner": "RED"}
    rec = build_sft_record(
        checkpoint=cp,
        output=out,
        opportunity=opp,
        trajectory=traj,
        prompt=prompt_with_strategy("board\n\n[RECENT TURNS]\nhist", "Go ore"),
        split="train",
        split_seed=42,
        run_id="run",
    )
    assert rec["strategy_in"] == "Go ore"
    assert rec["strategy_out"] == "s"
    assert rec["trajectory_id"] == "t"
    assert rec["split"] == "train"
    assert rec["game_end_turn"] == 72
    assert rec["completion"].startswith(" t") is False or "<strategy>" in rec["completion"]
    assert "[CURRENT STRATEGY]\nGo ore" in rec["prompt"]


def test_split_summary():
    assert split_summary([{"split": "train"}, {"split": "validation"}])["train"] == 1
    assert split_summary([])["total"] == 0


def test_metrics_aggregation():
    results = [
        {"correct": True, "valid": True, "phase": "PLAY_TURN", "expected_index": 1, "predicted_index": 1},
        {"correct": False, "valid": True, "phase": "PLAY_TURN", "expected_index": 2, "predicted_index": 3},
        {"correct": False, "valid": False, "phase": "MOVE_ROBBER", "expected_index": 1, "predicted_index": None},
    ]
    assert accuracy(results) == 1 / 3
    assert accuracy([]) == 0.0
    assert summarize(results)["by_phase"]["PLAY_TURN"]["total"] == 2
    assert summarize(results)["by_color"] == {"(none)": summarize(results)["by_color"]["(none)"]}


def test_rejection_reporting():
    outputs = [
        {"validation_status": "accepted", "phase": "PLAY_TURN"},
        {"validation_status": "rejected", "rejection_reason": "missing_think", "phase": "PLAY_TURN"},
        {"validation_status": "rejected", "rejection_reason": "wrong_alpha_beta_action", "phase": "MOVE_ROBBER"},
    ]
    assert acceptance(outputs) == {"accepted": 1, "rejected": 2, "total": 3}
    assert rejection_reasons(outputs)["missing_think"] == 1
    rep = run_report(outputs)
    assert rep["by_phase"]["PLAY_TURN"] == 1
    assert rep["by_checkpoint"] == {"(none)": 1}
