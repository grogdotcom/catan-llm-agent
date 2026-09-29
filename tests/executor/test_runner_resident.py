"""Resident runner tests for executor generic."""

import json
import tempfile

from catan_llm.executor.runner import RunExecutor
from catan_llm.executor.spec import RunSpec, RetryPolicy
from catan_llm.executor.store import RunStore
from catan_llm.llm.sft.provenance import derive_trajectory_id


def _traj(gid=0):
    seat_order = ["RED", "BLUE", "ORANGE", "WHITE"]
    tid = derive_trajectory_id(1000 + gid, seat_order)
    return {"trajectory_id": tid, "game_seed": 1000+gid, "game_id": gid, "game_end_turn": 72, "winner": "RED", "seat_order": seat_order, "seat_order_json": json.dumps(seat_order), "source_hash": ""}

def _rec(traj_id, seat_idx=0, turn=5, color="RED"):
    return {"trajectory_id": traj_id, "seat_index": seat_idx, "color": color, "turn": turn, "phase": "PLAY_TURN", "game_id": 0, "trajectory_index": 0, "decision_id": 0, "completion": "1", "engine_completion": "1", "num_moves": 3, "prompt": "[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]\nblah\n\n[PLAYABLE MOVES]\n1. a\n2. b\n3. c", "normalized_progress": turn/72.0}


def test_spec_to_run_config():
    spec = RunSpec(run_id="run-x", model="gpt-5.6-luna", transport="batch", retry_policy=RetryPolicy(max_retries=2))
    cfg = spec.to_run_config()
    assert cfg["run_id"] == "run-x"
    assert cfg["model"] == "gpt-5.6-luna"
    assert cfg["transport"] == "batch"


def test_run_executor_status_cancel_requeue(tmp_path=None):
    store = RunStore(":memory:")
    run_id = store.create_run({"run_id": "run-resident-1", "model": "gpt-5.6-luna"})
    traj = _traj(0)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5,15,30]):
        store.upsert_decision_opportunity(_rec(traj["trajectory_id"], 0, turn, color="RED"))
    # register generic checkpoint
    opp_id = store.list_opportunities(traj["trajectory_id"])[0]["opportunity_id"]
    store.register_checkpoints(run_id, [{
        "checkpoint_id": f"{run_id}-{traj['trajectory_id']}-0-1",
        "trajectory_id": traj["trajectory_id"],
        "seat_index": 0,
        "checkpoint_index": 1,
        "checkpoint_count": 1,
        "opportunity_id": opp_id,
        "strategy_in": "bootstrap",
        "strategy_source_checkpoint": None,
        "status": "pending",
        "skip_reason": None,
    }])
    executor = RunExecutor(store, run_id)
    status = executor.status()
    assert status["run"]["run_id"] == run_id
    assert status["total_checkpoints"] >= 1
    executor.cancel()
    assert store.get_run(run_id)["status"] == "cancelled"
    cps = store.list_checkpoints(run_id)
    if cps:
        store.conn.execute("UPDATE checkpoints SET status='rejected' WHERE checkpoint_id=?", (cps[0]["checkpoint_id"],))
        store.conn.commit()
        n = executor.requeue(only_rejected=True)
        assert n >= 1
    store.close()


def test_run_executor_run_is_resume(tmp_path=None):
    store = RunStore(":memory:")
    run_id = store.create_run({})
    traj = _traj(1)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5,15]):
        store.upsert_decision_opportunity(_rec(traj["trajectory_id"], 0, turn, color="RED"))
    opp_id = store.list_opportunities(traj["trajectory_id"])[0]["opportunity_id"]
    store.register_checkpoints(run_id, [{
        "checkpoint_id": f"{run_id}-{traj['trajectory_id']}-0-1",
        "trajectory_id": traj["trajectory_id"],
        "seat_index": 0,
        "checkpoint_index": 1,
        "checkpoint_count": 1,
        "opportunity_id": opp_id,
        "strategy_in": "s",
        "strategy_source_checkpoint": None,
        "status": "pending",
        "skip_reason": None,
    }])
    executor = RunExecutor(store, run_id)
    executor.run()
    assert store.get_run(run_id) is not None
    store.close()
