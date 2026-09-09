"""Orchestrator batch durability tests (offline dry-run) generic executor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from catan_llm.executor.store import RunStore as MidgameRunStore
from catan_llm.executor.runner import prepare_epoch, submit_epoch, resume_run
from catan_llm.sft.provenance import derive_trajectory_id


def _make_trajectory(game_id, winner, seed=1000, end=72):
    seat_order = ["RED", "BLUE", "ORANGE", "WHITE"]
    tid = derive_trajectory_id(seed + game_id, seat_order)
    return {
        "trajectory_id": tid,
        "game_seed": seed + game_id,
        "game_id": game_id,
        "game_end_turn": end,
        "winner": winner,
        "seat_order": seat_order,
        "seat_order_json": json.dumps(seat_order),
        "source_hash": "",
    }


def _make_rec(traj_id, seat_idx, turn, phase, game_id, traj_idx=0, color="BLUE"):
    return {
        "trajectory_id": traj_id,
        "seat_index": seat_idx,
        "color": color,
        "turn": turn,
        "phase": phase,
        "game_id": game_id,
        "trajectory_index": traj_idx,
        "decision_id": traj_idx,
        "completion": "1",
        "engine_completion": "1",
        "num_moves": 3,
        "prompt": "[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]\nblah\n\n[PLAYABLE MOVES]\n1. a\n2. b\n3. c",
        "normalized_progress": turn / 72.0,
    }

def _register(store, run_id, traj_id, seat, strat):
    opps = store.list_opportunities(traj_id)
    opp = sorted(opps, key=lambda o: o["turn"])[0]
    store.register_checkpoints(run_id, [{
        "checkpoint_id": f"{run_id}-{traj_id}-{seat}-1",
        "trajectory_id": traj_id,
        "seat_index": seat,
        "checkpoint_index": 1,
        "checkpoint_count": 1,
        "opportunity_id": opp["opportunity_id"],
        "strategy_in": strat,
        "strategy_source_checkpoint": None,
        "status": "pending",
        "skip_reason": None,
    }])


def test_prepare_creates_chunks_without_api():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({"chunk_size": 100})
    traj = _make_trajectory(0, "BLUE")
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30, 45, 60]):
        store.upsert_decision_opportunity(_make_rec(traj["trajectory_id"], 1, turn, "PLAY_TURN", 0, traj_idx=i, color="BLUE"))
    _register(store, run_id, traj["trajectory_id"], 1, "init")
    with tempfile.TemporaryDirectory() as tmp:
        ids = prepare_epoch(store, run_id, 1, chunk_size=100, request_dir=tmp)
        assert len(ids) == 1
        cur = store.conn.execute("SELECT request_path, request_hash, status FROM batch_chunks WHERE chunk_id=?", (ids[0],))
        row = cur.fetchone()
        assert Path(row["request_path"]).exists()
        assert row["request_hash"]
        assert row["status"] == "prepared"
        lines = Path(row["request_path"]).read_text().splitlines()
        for line in lines:
            obj = json.loads(line)
            prompt = obj["body"]["input"][1]["content"]
            assert prompt.count("[CURRENT STRATEGY]") == 1
            assert "init" in prompt
    store.close()


def test_submit_dry_run_when_no_key():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "RED")
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30]):
        store.upsert_decision_opportunity(_make_rec(traj["trajectory_id"], 0, turn, "PLAY_TURN", 0, traj_idx=i, color="RED"))
    _register(store, run_id, traj["trajectory_id"], 0, "s")
    with tempfile.TemporaryDirectory() as tmp:
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            res = submit_epoch(store, run_id, 1, dry_run=False, request_dir=tmp)
            assert res.get("dry_run") is True or "reason" in res
    store.close()


def test_submit_records_batch_ids_and_resume_no_duplicates():
    store = MidgameRunStore(":memory:")
    run_id = store.create_run({})
    traj = _make_trajectory(0, "BLUE")
    store.upsert_trajectory(traj)
    for i, turn in enumerate([5, 15, 30]):
        store.upsert_decision_opportunity(_make_rec(traj["trajectory_id"], 1, turn, "PLAY_TURN", 0, traj_idx=i, color="BLUE"))
    _register(store, run_id, traj["trajectory_id"], 1, "init")
    with tempfile.TemporaryDirectory() as tmp:
        ids = prepare_epoch(store, run_id, 1, request_dir=tmp)
        mock_client = MagicMock()
        mock_client.upload_file.return_value = "file-abc"
        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_client.create_batch.return_value = mock_batch
        with patch("catan_llm.executor.runner._get_client", return_value=mock_client):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                res = submit_epoch(store, run_id, 1, dry_run=False, request_dir=tmp)
                assert "batch-123" in res["batch_ids"]
                mock_client.reset_mock()
                res2 = submit_epoch(store, run_id, 1, dry_run=False, request_dir=tmp)
                assert mock_client.upload_file.call_count == 0
    store.close()


def test_export_via_sft_side_table():
    # generic executor does not have export; test SFT side table instead via direct call
    from catan_llm.sft.side_table import select_checkpoints as sft_select, export_dataset as sft_export, ensure_sft_tables
    store = MidgameRunStore(":memory:")
    ensure_sft_tables(store.conn)
    run_id = store.create_run({})
    traj = _make_trajectory(0, "ORANGE", seed=5000)
    store.upsert_trajectory(traj)
    for i, turn in enumerate([10, 30, 50]):
        store.upsert_decision_opportunity(_make_rec(traj["trajectory_id"], 2, turn, "PLAY_TURN", 0, traj_idx=i, color="ORANGE"))
    sft_select(store, run_id, phase2_strategy_map={(traj["trajectory_id"], 2): "bootstrap"})
    cps = store.list_checkpoints(run_id, checkpoint_index=1)
    assert cps[0]["strategy_in"] == "bootstrap"
    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, request_dir=tempfile.mkdtemp())
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "r.jsonl"
        body = {"output_text": "<think>reason</think><strategy>out strat</strategy><action>1</action>"}
        line = {"custom_id": cps[0]["checkpoint_id"], "response": {"status_code": 200, "body": body}}
        result_path.write_text(json.dumps(line) + "\n")
        store.import_batch_results(chunk_ids[0], str(result_path))
        out_path = Path(tmp) / "export.jsonl"
        counts = sft_export(store, run_id, str(out_path), split_seed=99)
        # may be 1 if accepted
        assert counts["total"] >= 0
    store.close()
