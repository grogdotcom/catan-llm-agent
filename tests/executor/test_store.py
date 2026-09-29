"""RunStore (executor) generic tests: durable state, generic chaining, batches."""

import json
import tempfile
from pathlib import Path

from catan_llm.executor.store import RunStore
from catan_llm.llm.sft.provenance import derive_trajectory_id

MidgameRunStore = RunStore

def _make_trajectory(game_id, winner, game_end_turn=72, seed=1000):
    seat_order = ["RED", "BLUE", "ORANGE", "WHITE"]
    tid = derive_trajectory_id(seed + game_id, seat_order)
    return {
        "trajectory_id": tid,
        "game_seed": seed + game_id,
        "game_id": game_id,
        "game_end_turn": game_end_turn,
        "winner": winner,
        "seat_order": seat_order,
        "seat_order_json": json.dumps(seat_order),
        "source_hash": "",
    }

def _make_record(traj_id, seat_idx, turn, phase, game_id, traj_idx=0, completion="2", num_moves=3, color="BLUE"):
    return {
        "trajectory_id": traj_id,
        "seat_index": seat_idx,
        "color": color,
        "turn": turn,
        "phase": phase,
        "game_id": game_id,
        "trajectory_index": traj_idx,
        "decision_id": traj_idx,
        "completion": completion,
        "engine_completion": completion,
        "num_moves": num_moves,
        "prompt": f"[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]\nturn {turn}\n\n[PLAYABLE MOVES]\n1. a\n2. b\n3. c",
        "chosen_label": "a",
        "normalized_progress": turn / 72.0,
    }

def _register_generic(store, run_id, traj_id, seat_idx, count, strategy_ins):
    opps = store.list_opportunities(traj_id)
    # map turn sorted
    opps_sorted = sorted(opps, key=lambda o: o["turn"])
    cps = []
    for idx in range(1, count+1):
        opp = opps_sorted[idx-1] if idx-1 < len(opps_sorted) else opps_sorted[0]
        cps.append({
            "checkpoint_id": f"{run_id}-{traj_id}-{seat_idx}-{idx}",
            "trajectory_id": traj_id,
            "seat_index": seat_idx,
            "checkpoint_index": idx,
            "checkpoint_count": count,
            "opportunity_id": opp["opportunity_id"],
            "strategy_in": strategy_ins[idx-1] if idx-1 < len(strategy_ins) else None,
            "strategy_source_checkpoint": None if idx==1 else f"{run_id}-{traj_id}-{seat_idx}-{idx-1}",
            "status": "pending" if strategy_ins[idx-1] else "skipped",
            "skip_reason": None if strategy_ins[idx-1] else "missing_bootstrap_strategy",
        })
    # but mark skipped accordingly: if strategy_in is None and we want skipped
    for cp in cps:
        if cp["strategy_in"] is None:
            cp["status"] = "skipped"
            cp["skip_reason"] = "missing_bootstrap_strategy" if cp["checkpoint_index"]==1 else "broken_strategy_lineage"
    store.register_checkpoints(run_id, cps)
    return cps

def test_store_create_run_and_trajectory():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"sample_seed": 42, "model": "gpt-5.6-luna"})
    assert store.get_run(run_id) is not None
    traj = _make_trajectory(0, "BLUE")
    store.upsert_trajectory(traj)
    assert len(store.list_trajectories()) == 1
    store.close()

def test_register_generic_checkpoints():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"sample_seed": 1})
    traj = _make_trajectory(0, "BLUE", game_end_turn=60)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 25, 35, 45, 55]):
        rec = _make_record(traj["trajectory_id"], 1, turn, "PLAY_TURN", 0, traj_idx=i, completion="1", color="BLUE")
        store.upsert_decision_opportunity(rec)
    _register_generic(store, run_id, traj["trajectory_id"], 1, 4, ["bootstrap strat", None, None, None])
    cps = store.list_checkpoints(run_id)
    # Should have 4
    assert len(cps) == 4
    first = [c for c in cps if c["checkpoint_index"] == 1][0]
    assert first["strategy_in"] == "bootstrap strat"
    assert first["status"] == "pending"
    store.close()

def test_strategy_chaining_bootstrap_and_next():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "RED", game_end_turn=72)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30, 45, 60]):
        rec = _make_record(traj["trajectory_id"], 0, turn, "PLAY_TURN", 0, traj_idx=i, completion="2", color="RED")
        store.upsert_decision_opportunity(rec)
    _register_generic(store, run_id, traj["trajectory_id"], 0, 3, ["init strat", None, None])
    cps = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
    assert cps[0]["strategy_in"] == "init strat"
    assert cps[1]["status"] == "skipped"
    assert cps[1]["skip_reason"] == "broken_strategy_lineage"
    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    assert len(chunk_ids) == 1
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "res.jsonl"
        cp_id = cps[0]["checkpoint_id"]
        body = {
            "output_text": "<think>ok</think><strategy>next strat</strategy><action>2</action>",
            "output": [{"type": "message", "content": [{"text": "<think>ok</think><strategy>next strat</strategy><action>2</action>"}]}],
        }
        line = {"custom_id": cp_id, "response": {"status_code": 200, "body": body}}
        with open(result_path, "w") as f:
            f.write(json.dumps(line) + "\n")
        res = store.import_batch_results(chunk_ids[0], str(result_path))
        assert res["accepted"] == 1
        nxt = store.advance_checkpoint(run_id)
        assert nxt == 1
        cps2 = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
        second = [c for c in cps2 if c["checkpoint_index"] == 2][0]
        assert second["strategy_in"] == "next strat"
        assert second["status"] == "pending"
    store.close()

def test_invalid_chain_does_not_reuse_stale():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(1, "BLUE", game_end_turn=72)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30, 45, 60]):
        rec = _make_record(traj["trajectory_id"], 1, turn, "PLAY_TURN", 1, traj_idx=i, completion="2", color="BLUE")
        store.upsert_decision_opportunity(rec)
    _register_generic(store, run_id, traj["trajectory_id"], 1, 3, ["init", None, None])
    cps = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
    first_id = cps[0]["checkpoint_id"]
    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "res.jsonl"
        body = {"output_text": "<think>t</think><strategy>s</strategy><action>999</action>"}
        line = {"custom_id": first_id, "response": {"status_code": 200, "body": body}}
        with open(result_path, "w") as f:
            f.write(json.dumps(line) + "\n")
        res = store.import_batch_results(chunk_ids[0], str(result_path))
        assert res["rejected"] == 1
        store.advance_checkpoint(run_id)
        cps2 = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
        second = [c for c in cps2 if c["checkpoint_index"] == 2][0]
        assert second["status"] == "skipped"
        assert second["skip_reason"] == "broken_strategy_lineage"
        assert second["strategy_in"] is None
    store.close()

def test_missing_bootstrap_marks_skipped():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(2, "ORANGE", game_end_turn=72)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30]):
        rec = _make_record(traj["trajectory_id"], 2, turn, "PLAY_TURN", 2, traj_idx=i, completion="1", color="ORANGE")
        store.upsert_decision_opportunity(rec)
    _register_generic(store, run_id, traj["trajectory_id"], 2, 2, [None, None])
    cps = store.list_checkpoints(run_id)
    assert all(c["status"] == "skipped" for c in cps)
    assert all(c["skip_reason"] in ("missing_bootstrap_strategy", "broken_strategy_lineage") for c in cps)
    store.close()

def test_batch_chunks_persist_and_resume_no_duplicates():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"chunk_size": 100})
    traj = _make_trajectory(0, "RED")
    store.upsert_trajectory(traj)
    for i in range(5):
        rec = _make_record(traj["trajectory_id"], 0, turn=5 + i * 10, phase="PLAY_TURN", game_id=0, traj_idx=i, color="RED")
        store.upsert_decision_opportunity(rec)
    _register_generic(store, run_id, traj["trajectory_id"], 0, 3, ["s", None, None])
    # Need 3 pending but only first pending so chunk count 1
    # Adjust to have 3 pending for test
    store.conn.execute("UPDATE checkpoints SET status='pending', strategy_in='s' WHERE checkpoint_index>1")
    store.conn.commit()
    ids1 = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=100, request_dir=tempfile.mkdtemp())
    assert len(ids1) == 1
    ids2 = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=100, request_dir=tempfile.mkdtemp())
    assert ids1 == ids2
    store.record_batch_submission(ids1[0], "batch_123")
    cur = store.conn.execute("SELECT openai_batch_id FROM batch_chunks WHERE chunk_id=?", (ids1[0],))
    assert cur.fetchone()["openai_batch_id"] == "batch_123"
    ids3 = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=100, request_dir=tempfile.mkdtemp())
    cur2 = store.conn.execute("SELECT openai_batch_id FROM batch_chunks WHERE chunk_id=?", (ids1[0],))
    assert cur2.fetchone()["openai_batch_id"] == "batch_123"
    store.close()

def test_not_advance_until_validated():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "BLUE", game_end_turn=72)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30, 45, 60]):
        rec = _make_record(traj["trajectory_id"], 1, turn, "PLAY_TURN", 0, traj_idx=i, completion="1", color="BLUE")
        store.upsert_decision_opportunity(rec)
    _register_generic(store, run_id, traj["trajectory_id"], 1, 3, ["s", None, None])
    # Make second pending too
    store.conn.execute("UPDATE checkpoints SET status='pending', strategy_in='s2' WHERE checkpoint_index=2")
    store.conn.commit()
    store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    res = store.advance_checkpoint(run_id)
    assert res is None
    assert store.get_run(run_id)["current_checkpoint"] == 0
    store.close()

def test_chunk_size_approx_100():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"chunk_size": 100})
    phase2_needed = []
    for gid in range(50):
        traj = _make_trajectory(gid, "RED", game_end_turn=90, seed=2000 + gid)
        store.upsert_trajectory(traj)
        for i, turn in enumerate([5, 15, 30, 45, 60, 75]):
            rec = _make_record(traj["trajectory_id"], 0, turn, "PLAY_TURN", gid, traj_idx=i, color="RED")
            store.upsert_decision_opportunity(rec)
        # register 1 pending per traj for epoch1
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
    cps1 = store.list_checkpoints(run_id, checkpoint_index=1)
    pending1 = [c for c in cps1 if c["status"] == "pending"]
    assert len(pending1) == 50
    ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=20)
    assert len(ids) == 3
    for cid in ids:
        cur = store.conn.execute("SELECT request_path FROM batch_chunks WHERE chunk_id=?", (cid,))
        path = cur.fetchone()["request_path"]
        lines = Path(path).read_text().splitlines()
        assert len(lines) <= 20
    store.close()

def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        store = MidgameRunStore(db_path)
        run_id = store.create_run({"sample_seed": 999})
        traj = _make_trajectory(0, "WHITE", game_end_turn=80)
        store.upsert_trajectory(traj)
        store.close()
        store2 = MidgameRunStore(db_path)
        assert store2.get_run(run_id) is not None
        assert len(store2.list_trajectories()) == 1
        store2.close()

def test_transaction_atomicity_create_run():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"sample_seed": 1})
    try:
        store.create_run({"run_id": run_id, "sample_seed": 2})
        assert False, "should have raised"
    except Exception:
        pass
    assert store.get_run(run_id)["sample_seed"] == 1
    store.close()

def test_register_checkpoints_upsert():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "BLUE")
    store.upsert_trajectory(traj)
    rec = _make_record(traj["trajectory_id"], 1, 5, "PLAY_TURN", 0, color="BLUE")
    store.upsert_decision_opportunity(rec)
    opp_id = store.list_opportunities()[0]["opportunity_id"]
    cp = {
        "checkpoint_id": f"{run_id}-{traj['trajectory_id']}-1-1",
        "trajectory_id": traj["trajectory_id"],
        "seat_index": 1,
        "checkpoint_index": 1,
        "checkpoint_count": 1,
        "opportunity_id": opp_id,
        "strategy_in": "init",
        "strategy_source_checkpoint": None,
        "status": "pending",
        "skip_reason": None,
    }
    store.register_checkpoints(run_id, [cp])
    assert len(store.list_checkpoints(run_id)) == 1
    # upsert same id with different strategy
    cp["strategy_in"] = "updated"
    store.register_checkpoints(run_id, [cp])
    assert store.list_checkpoints(run_id)[0]["strategy_in"] == "updated"
    store.close()
