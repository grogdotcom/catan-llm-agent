"""Inline transport tests — covers pluggable executor refactor."""

import json
from unittest.mock import MagicMock, patch

from catan_llm.executor.store import RunStore
from catan_llm.executor.transport.inline import InlineTransport
from catan_llm.executor.transport.batch import BatchTransport
from catan_llm.executor.runner import RunExecutor, resume_run, submit_inline, poll_inline
from catan_llm.llm.sft.provenance import derive_trajectory_id


def _traj(gid=0):
    seat = ["RED", "BLUE", "ORANGE", "WHITE"]
    tid = derive_trajectory_id(1000 + gid, seat)
    return {"trajectory_id": tid, "game_seed": 1000+gid, "game_id": gid, "game_end_turn": 72, "winner": "RED", "seat_order": seat, "seat_order_json": json.dumps(seat), "source_hash": ""}

def _rec(tid, turn=5):
    return {"trajectory_id": tid, "seat_index": 0, "color": "RED", "turn": turn, "phase": "PLAY_TURN", "game_id": 0, "trajectory_index": 0, "decision_id": 0, "completion": "1", "engine_completion": "1", "num_moves": 3, "prompt": "[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]\nblah\n\n[PLAYABLE MOVES]\n1. a\n2. b\n3. c", "normalized_progress": turn/72.0}


def _make_run_with_one_pending(transport="inline"):
    store = RunStore(":memory:")
    run_id = store.create_run({"run_id": f"run-{transport}-1", "model": "muse-spark-1.2-contributor", "transport": transport})
    traj = _traj(0)
    store.upsert_trajectory(traj)
    for turn in [5,15]:
        store.upsert_decision_opportunity(_rec(traj["trajectory_id"], turn))
    opp = store.list_opportunities(traj["trajectory_id"])[0]["opportunity_id"]
    store.register_checkpoints(run_id, [{"checkpoint_id": f"{run_id}-{traj['trajectory_id']}-0-1", "trajectory_id": traj["trajectory_id"], "seat_index": 0, "checkpoint_index": 1, "checkpoint_count": 1, "opportunity_id": opp, "strategy_in": "bootstrap", "strategy_source_checkpoint": None, "status": "pending", "skip_reason": None}])
    return store, run_id, traj["trajectory_id"]


def test_inline_prepare_returns_pending():
    store, run_id, _ = _make_run_with_one_pending("inline")
    t = InlineTransport()
    ids = t.prepare(store, run_id, 1)
    assert len(ids) == 1
    store.close()


def test_store_get_prompt_and_write_inline():
    store, run_id, _ = _make_run_with_one_pending("inline")
    cp = store.list_checkpoints(run_id, 1)[0]
    prompt = store.get_prompt(cp["checkpoint_id"])
    assert "[CURRENT STRATEGY]" in prompt
    assert "bootstrap" in prompt
    # valid inline response
    res = store.write_inline_output(cp["checkpoint_id"], "<think>t</think><strategy>s</strategy><action>1</action>")
    assert res["accepted"] is True
    # rejected (wrong action)
    # need second checkpoint for rejected test
    store2, run_id2, _ = _make_run_with_one_pending("inline")
    cp2 = store2.list_checkpoints(run_id2, 1)[0]
    res2 = store2.write_inline_output(cp2["checkpoint_id"], "<think>t</think><strategy>s</strategy><action>999</action>")
    assert res2["accepted"] is False
    store.close()
    store2.close()


def test_inline_submit_accepts_and_advances():
    store, run_id, tid = _make_run_with_one_pending("inline")
    # add second epoch pending via chaining (will be skipped then promoted)
    opp = store.list_opportunities(tid)[0]["opportunity_id"]
    store.register_checkpoints(run_id, [{"checkpoint_id": f"{run_id}-{tid}-0-2", "trajectory_id": tid, "seat_index": 0, "checkpoint_index": 2, "checkpoint_count": 2, "opportunity_id": opp, "strategy_in": None, "strategy_source_checkpoint": f"{run_id}-{tid}-0-1", "status": "skipped", "skip_reason": "broken_strategy_lineage"}])
    mock_raw = MagicMock()
    mock_raw.output_text = "<think>ok</think><strategy>next</strategy><action>1</action>"
    mock_gateway = MagicMock()
    mock_gateway.chat_completion.return_value = mock_raw
    mock_gateway.model = "muse-spark-1.2-contributor"
    with patch("catan_llm.executor.transport.inline._get_client", return_value=mock_gateway):
        t = InlineTransport(concurrency=2)
        res = t.submit(store, run_id, 1, concurrency=2)
        assert res["accepted"] == 1
        # poll should report completed
        poll = t.poll(store, run_id, 1)
        assert poll["status"] == "completed"
        # advance should promote strategy to pending
        nxt = store.advance_checkpoint(run_id)
        assert nxt == 1
        cps2 = store.list_checkpoints(run_id, 2)
        assert cps2[0]["status"] == "pending"
        assert cps2[0]["strategy_in"] == "next"
        # full resume will then execute epoch2 as well
        resume_run(store, run_id, transport="inline", concurrency=2)
        cps2_after = store.list_checkpoints(run_id, 2)
        # after full resume, epoch2 should be validated (since it was pending and inline executed)
        assert cps2_after[0]["status"] == "validated"
    store.close()


def test_inline_submit_retry_on_429():
    store, run_id, _ = _make_run_with_one_pending("inline")
    mock_gateway = MagicMock()
    # first call raises 429, second succeeds
    mock_raw_ok = MagicMock()
    mock_raw_ok.output_text = "<think>t</think><strategy>s</strategy><action>1</action>"
    mock_gateway.chat_completion.side_effect = [Exception("429 rate limited"), mock_raw_ok]
    mock_gateway.model = "muse-spark-1.2-contributor"
    with patch("catan_llm.executor.transport.inline._get_client", return_value=mock_gateway):
        with patch("time.sleep", return_value=None):
            t = InlineTransport(concurrency=1, max_retries=3)
            res = t.submit(store, run_id, 1, concurrency=1)
            assert res["accepted"] == 1
    store.close()


def test_batch_transport_prepare_and_submit_dry_run():
    store, run_id, _ = _make_run_with_one_pending("batch")
    # batch pending
    t = BatchTransport()
    ids = t.prepare(store, run_id, 1)
    assert len(ids) == 1
    res = t.submit(store, run_id, 1, dry_run=True)
    assert res["dry_run"] is True
    store.close()


def test_run_executor_inline_resume():
    store, run_id, _ = _make_run_with_one_pending("inline")
    mock_raw = MagicMock()
    mock_raw.output_text = "<think>t</think><strategy>s</strategy><action>1</action>"
    mock_gateway = MagicMock()
    mock_gateway.chat_completion.return_value = mock_raw
    mock_gateway.model = "muse-spark-1.2-contributor"
    with patch("catan_llm.executor.transport.inline._get_client", return_value=mock_gateway):
        ex = RunExecutor(store, run_id, transport="inline")
        ex.resume(concurrency=2)
        cps = store.list_checkpoints(run_id)
        assert cps[0]["status"] == "validated"
    store.close()


def test_submit_inline_and_poll_inline_helpers():
    store, run_id, _ = _make_run_with_one_pending("inline")
    mock_raw = MagicMock()
    mock_raw.output_text = "<think>t</think><strategy>s</strategy><action>1</action>"
    mock_gateway = MagicMock()
    mock_gateway.chat_completion.return_value = mock_raw
    mock_gateway.model = "muse-spark-1.2-contributor"
    with patch("catan_llm.executor.transport.inline._get_client", return_value=mock_gateway):
        res = submit_inline(store, run_id, 1, concurrency=2)
        assert "accepted" in res
        poll = poll_inline(store, run_id, 1)
        assert poll["status"] == "completed"
    store.close()
