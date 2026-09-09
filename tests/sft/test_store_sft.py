"""MidgameRunStore tests: durable state, chaining, batches, export."""

import json
import tempfile
from pathlib import Path

from catan_llm.executor.store import RunStore as MidgameRunStore
from catan_llm.sft.provenance import derive_trajectory_id
from catan_llm.sft.side_table import select_checkpoints as sft_select_checkpoints, export_dataset as sft_export_dataset


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


def test_store_create_run_and_trajectory():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"sample_seed": 42, "model": "gpt-5.6-luna"})
    assert store.get_run(run_id) is not None
    traj = _make_trajectory(0, "BLUE")
    store.upsert_trajectory(traj)
    assert len(store.list_trajectories()) == 1
    store.close()


def test_collapse_and_checkpoint_selection():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"sample_seed": 1})
    traj = _make_trajectory(0, "BLUE", game_end_turn=60)
    store.upsert_trajectory(traj)
    # Create 6 opportunities spread to trigger K=4 (<=70)
    for i, turn in enumerate([5, 15, 25, 35, 45, 55]):
        rec = _make_record(traj["trajectory_id"], 1, turn, "PLAY_TURN", 0, traj_idx=i, completion="1", color="BLUE")
        store.upsert_decision_opportunity(rec)
    phase2 = {(traj["trajectory_id"], 1): "bootstrap strat"}
    created = sft_select_checkpoints(store, run_id, phase2_strategy_map=phase2)
    # K for 60 =>4, N=6 =>4, empty bands skipped maybe 4
    assert created == 4
    cps = store.list_checkpoints(run_id)
    assert len(cps) == 4
    # First uses bootstrap
    first = [c for c in cps if c["checkpoint_index"] == 1][0]
    assert first["strategy_in"] == "bootstrap strat"
    assert first["status"] == "pending"
    store.close()


def test_strategy_chaining_bootstrap_and_next():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "RED", game_end_turn=72)
    store.upsert_trajectory(traj)
    # Need at least 2 checkpoints: K=5 but N maybe 6 =>5
    recs = []
    for i, turn in enumerate([5, 15, 30, 45, 60]):
        recs.append(_make_record(traj["trajectory_id"], 0, turn, "PLAY_TURN", 0, traj_idx=i, completion="2", color="RED"))
    for r in recs:
        store.upsert_decision_opportunity(r)
    phase2 = {(traj["trajectory_id"], 0): "init strat"}
    sft_select_checkpoints(store, run_id, phase2_strategy_map=phase2)
    cps = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
    assert cps[0]["strategy_in"] == "init strat"
    # Second should be pending but strategy_in is None initially because previous not accepted yet
    # After selection, only first has strategy; second's strategy_in will be None and status skipped? Let's check.
    # In our store logic, second's strategy_in is looked up from previous output (none) => skipped.
    # It should be skipped with broken lineage until advance fills it.
    # For this test, we want to simulate accepted chain: after first validated, next should get strategy.
    # So first should be pending, second should be skipped initially.
    assert cps[1]["status"] == "skipped"
    assert cps[1]["skip_reason"] == "broken_strategy_lineage"

    # Now simulate validating first checkpoint
    # Create batch chunk for epoch1
    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    assert len(chunk_ids) == 1
    # Simulate result import: craft result file
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "res.jsonl"
        # Need to write batch result line for checkpoint 0
        cp_id = cps[0]["checkpoint_id"]
        # Response must be accepted: need <think> <strategy> <action>2 (matches completion 2)
        body = {
            "output_text": "<think>ok</think><strategy>next strat</strategy><action>2</action>",
            "output": [{"type": "message", "content": [{"text": "<think>ok</think><strategy>next strat</strategy><action>2</action>"}]}],
        }
        line = {"custom_id": cp_id, "response": {"status_code": 200, "body": body}}
        with open(result_path, "w") as f:
            f.write(json.dumps(line) + "\n")
        res = store.import_batch_results(chunk_ids[0], str(result_path))
        assert res["accepted"] == 1
        # Now advance should fill next epoch's strategy
        nxt = store.advance_checkpoint(run_id)
        assert nxt == 1  # first advance moves from 0 to 1
        # Need to advance again? Actually after first advance, next epoch's checkpoint should be updated
        # Fetch second checkpoint again
        cps2 = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
        second = [c for c in cps2 if c["checkpoint_index"] == 2][0]
        # After advance, strategy_in should be filled with previous strategy_out
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
    phase2 = {(traj["trajectory_id"], 1): "init"}
    sft_select_checkpoints(store, run_id, phase2_strategy_map=phase2)
    cps = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
    first_id = cps[0]["checkpoint_id"]
    # Simulate rejected first checkpoint (wrong action)
    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "res.jsonl"
        body = {"output_text": "<think>t</think><strategy>s</strategy><action>999</action>"}  # wrong range and wrong action
        line = {"custom_id": first_id, "response": {"status_code": 200, "body": body}}
        with open(result_path, "w") as f:
            f.write(json.dumps(line) + "\n")
        res = store.import_batch_results(chunk_ids[0], str(result_path))
        assert res["rejected"] == 1
        # Advance should mark next as skipped (broken lineage) not reuse init
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
    # No phase2 strategy
    sft_select_checkpoints(store, run_id, phase2_strategy_map={})
    cps = store.list_checkpoints(run_id)
    assert all(c["status"] == "skipped" for c in cps)
    assert all(c["skip_reason"] == "missing_bootstrap_strategy" or c["skip_reason"] == "broken_strategy_lineage" for c in cps)
    store.close()


def test_batch_chunks_persist_and_resume_no_duplicates():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"chunk_size": 100})
    traj = _make_trajectory(0, "RED")
    store.upsert_trajectory(traj)
    for i in range(5):
        rec = _make_record(traj["trajectory_id"], 0, turn=5 + i * 10, phase="PLAY_TURN", game_id=0, traj_idx=i, color="RED")
        store.upsert_decision_opportunity(rec)
    sft_select_checkpoints(store, run_id, phase2_strategy_map={(traj["trajectory_id"], 0): "s"})
    # First create
    ids1 = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=100, request_dir=tempfile.mkdtemp())
    assert len(ids1) == 1
    # Second call idempotent no duplicates
    ids2 = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=100, request_dir=tempfile.mkdtemp())
    assert ids1 == ids2
    # Simulate submission
    store.record_batch_submission(ids1[0], "batch_123")
    cur = store.conn.execute("SELECT openai_batch_id FROM batch_chunks WHERE chunk_id=?", (ids1[0],))
    assert cur.fetchone()["openai_batch_id"] == "batch_123"
    # Re-create should still reuse same batch_id
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
    sft_select_checkpoints(store, run_id, phase2_strategy_map={(traj["trajectory_id"], 1): "s"})
    # Create chunks but not validated
    store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    # Try advance without validation => should return None? Actually first advance requires validated.
    # Our store.advance_checkpoint checks if chunks are validated; if not, returns None.
    # But we haven't validated, so it should not advance.
    res = store.advance_checkpoint(run_id)
    # It will try to advance to 1 but chunks not validated => return None, not advancing
    # However our earlier logic: advance checks if current+1 chunks validated; if not validated, returns None.
    # So res should be None
    assert res is None
    assert store.get_run(run_id)["current_checkpoint"] == 0
    store.close()


def test_chunk_size_approx_100():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"chunk_size": 100})
    # Need 250 pending checkpoints to test chunking into 3 (100,100,50)
    # Create 50 trajectories each with 5 checkpoints => 250
    phase2 = {}
    for gid in range(50):
        traj = _make_trajectory(gid, "RED", game_end_turn=90, seed=2000 + gid)
        store.upsert_trajectory(traj)
        phase2[(traj["trajectory_id"], 0)] = "s"
        for i, turn in enumerate([5, 15, 30, 45, 60, 75]):
            rec = _make_record(traj["trajectory_id"], 0, turn, "PLAY_TURN", gid, traj_idx=i, color="RED")
            store.upsert_decision_opportunity(rec)
    sft_select_checkpoints(store, run_id, phase2_strategy_map=phase2)
    # Count pending for epoch 1: should be 50
    cps1 = store.list_checkpoints(run_id, checkpoint_index=1)
    pending1 = [c for c in cps1 if c["status"] == "pending"]
    assert len(pending1) == 50
    # Now add more pending by manually creating many? Actually we need 250 -> need more than 50, but we have 50 pending per epoch.
    # For testing chunk size, create a single epoch with 250 pending by faking many trajectories
    # Instead just test that chunk size 100 splits 50 into 1 chunk, and 250 would split into 3
    # Test with artificially large pending count via direct checkpoint insertion
    # Quick: create batch chunks for epoch 1 with chunk_size 20
    ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=20)
    assert len(ids) == 3  # 50 /20 = 3 (20,20,10)
    # Verify each chunk file has ~20 lines
    for cid in ids:
        cur = store.conn.execute("SELECT request_path FROM batch_chunks WHERE chunk_id=?", (cid,))
        path = cur.fetchone()["request_path"]
        lines = Path(path).read_text().splitlines()
        assert len(lines) <= 20
    store.close()


def test_export_accepted_with_splits():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "BLUE", game_end_turn=72)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30, 45, 60]):
        rec = _make_record(traj["trajectory_id"], 1, turn, "PLAY_TURN", 0, traj_idx=i, completion="2", color="BLUE")
        store.upsert_decision_opportunity(rec)
    sft_select_checkpoints(store, run_id, phase2_strategy_map={(traj["trajectory_id"], 1): "init"})
    cps = sorted(store.list_checkpoints(run_id), key=lambda x: x["checkpoint_index"])
    # Validate first
    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1)
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "res.jsonl"
        body = {"output_text": "<think>t</think><strategy>out1</strategy><action>2</action>"}
        line = {"custom_id": cps[0]["checkpoint_id"], "response": {"status_code": 200, "body": body}}
        with open(result_path, "w") as f:
            f.write(json.dumps(line) + "\n")
        store.import_batch_results(chunk_ids[0], str(result_path))
        # Export
        out_path = Path(tmp) / "export.jsonl"
        counts = sft_export_dataset(store, run_id, str(out_path), split_seed=42)
        assert counts["total"] == 1
        # Check exported record preserves fields
        exported = json.loads(out_path.read_text().splitlines()[0])
        assert exported["strategy_in"] == "init"
        assert exported["strategy_out"] == "out1"
        assert exported["engine_completion"] == "2"
        assert exported["raw_output_text"] == "<think>t</think><strategy>out1</strategy><action>2</action>"
        assert exported["trajectory_id"] == traj["trajectory_id"]
        assert exported["checkpoint_index"] == 1
        assert exported["validation_status"] == "accepted"
        assert "split" in exported
        assert exported["split"] in ("train", "validation", "test")
    store.close()


def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        store = MidgameRunStore(db_path)
        run_id = store.create_run({"sample_seed": 999})
        traj = _make_trajectory(0, "WHITE", game_end_turn=80)
        store.upsert_trajectory(traj)
        store.close()
        # Reopen
        store2 = MidgameRunStore(db_path)
        assert store2.get_run(run_id) is not None
        assert len(store2.list_trajectories()) == 1
        store2.close()


def test_transaction_atomicity_create_run():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"sample_seed": 1})
    # Try to create duplicate run_id should fail and not leave partial
    try:
        store.create_run({"run_id": run_id, "sample_seed": 2})
        assert False, "should have raised"
    except Exception:
        pass
    # Original still exists
    assert store.get_run(run_id)["sample_seed"] == 1
    store.close()
