"""Midgame CLI integration: prepare -> submit(dry) -> status -> export."""

import argparse
import json
from pathlib import Path

from catan_llm.sft import cli
from catan_llm.sft.provenance import derive_trajectory_id


def _corpus(tmp_path, n_games=2, per_game=6):
    seat_order = ["RED", "BLUE", "ORANGE", "WHITE"]
    path = tmp_path / "corpus.jsonl"
    lines = []
    for gid in range(n_games):
        tid = derive_trajectory_id(1000 + gid, seat_order)
        for i, turn in enumerate([5, 15, 25, 35, 45, 55]):
            rec = {
                "trajectory_id": tid,
                "game_seed": 1000 + gid,
                "game_id": gid,
                "seat_index": 0,
                "color": "RED",
                "turn": turn,
                "phase": "PLAY_TURN",
                "trajectory_index": i,
                "decision_id": i,
                "completion": "2",
                "engine_completion": "2",
                "num_moves": 3,
                "winner": "RED",
                "game_end_turn": 72,
                "prompt": f"[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]\nturn {turn}\n\n[PLAYABLE MOVES]\n1. a\n2. b\n3. c",
                "chosen_label": "b",
                "normalized_progress": turn / 72.0,
            }
            lines.append(rec)
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    return path


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_cli_prepare_status_export(tmp_path):
    corpus = _corpus(tmp_path)
    db = str(tmp_path / "run.db")
    req_dir = str(tmp_path / "batches")
    out = str(tmp_path / "sft.jsonl")

    cli.cmd_prepare(
        _ns(
            db=db,
            corpus=str(corpus),
            sample_size=2,
            sample_seed=42,
            run_id="run1",
            model="gpt-5.6-luna",
            reasoning_effort="medium",
            chunk_size=100,
            request_dir=req_dir,
            phase2_strategies=None,
        )
    )

    # status
    cli.cmd_status(_ns(db=db, run_id="run1"))

    # export (no accepted outputs yet -> counts 0)
    cli.cmd_export(_ns(db=db, run_id="run1", output=out, split_seed=42))
    assert Path(out).exists()
    assert Path(out).read_text().strip() == ""

    # submit dry-run (no API key)
    cli.cmd_submit(
        _ns(db=db, run_id="run1", checkpoint_index=1, dry_run=True, chunk_size=100, request_dir=req_dir, base_url=None)
    )

    # resume (no chunks pending submission, dry path)
    cli.cmd_resume(_ns(db=db, run_id="run1", base_url=None))


def test_cli_main_prepare_with_phase2_map(tmp_path):
    corpus = _corpus(tmp_path, n_games=1)
    db = str(tmp_path / "run2.db")
    req_dir = str(tmp_path / "batches2")
    # phase2 as JSON mapping trajectory-seat -> strategy
    pmap = tmp_path / "phase2.json"
    tid = derive_trajectory_id(1000, ["RED", "BLUE", "ORANGE", "WHITE"])
    pmap.write_text(json.dumps({f"{tid}-0": "phase2 strategy"}))

    cli.cmd_prepare(
        _ns(
            db=db,
            corpus=str(corpus),
            sample_size=1,
            sample_seed=1,
            run_id="run2",
            model="gpt-5.6-luna",
            reasoning_effort="medium",
            chunk_size=100,
            request_dir=req_dir,
            phase2_strategies=str(pmap),
        )
    )


def test_ensure_provenance_synthesizes():
    recs = [{"game_id": 3, "color": "BLUE", "turn": 10, "decision_id": 2}]
    out = cli._ensure_provenance(recs)
    assert out[0]["trajectory_id"]
    assert out[0]["seat_index"] == 1
    assert out[0]["game_end_turn"] == 60
    assert out[0]["game_id"] == 3
