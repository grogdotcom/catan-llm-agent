"""CLI for SFT checkpoint pipeline (strategy-chained checkpoints, SFT export)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "src")

from catan_llm.executor.store import RunStore
from catan_llm.sft.provenance import derive_trajectory_id, sample_winning_trajectories
from catan_llm.executor.runner import prepare_epoch, submit_epoch, poll_epoch, resume_run
from catan_llm.sft.side_table import select_checkpoints as sft_select_checkpoints
from catan_llm.sft.side_table import export_dataset as sft_export_dataset


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _ensure_provenance(records: List[Dict[str, Any]], default_game_end_turn: int = 60) -> List[Dict[str, Any]]:
    """Ensure trajectory provenance fields exist; synthesize if missing for backward compat."""
    out = []
    for r in records:
        rr = dict(r)
        if "trajectory_id" not in rr:
            seed = rr.get("game_seed", 1000 + int(rr.get("game_id", 0)) * 97)
            rr["game_seed"] = seed
            rr["trajectory_id"] = derive_trajectory_id(seed, ["RED", "BLUE", "ORANGE", "WHITE"])
        if "seat_index" not in rr:
            color = rr.get("color", "RED")
            try:
                rr["seat_index"] = ["RED", "BLUE", "ORANGE", "WHITE"].index(color)
            except ValueError:
                rr["seat_index"] = 0
        if "game_end_turn" not in rr:
            rr["game_end_turn"] = default_game_end_turn
        if "trajectory_index" not in rr:
            rr["trajectory_index"] = rr.get("decision_id", 0)
        if "game_id" not in rr:
            rr["game_id"] = 0
        out.append(rr)
    return out


def cmd_prepare(args: argparse.Namespace) -> None:
    store = RunStore(args.db)
    # Ensure SFT side tables exist
    from catan_llm.sft.side_table import ensure_sft_tables

    ensure_sft_tables(store.conn)
    records = _load_jsonl(args.corpus)
    records = _ensure_provenance(records)
    from collections import defaultdict

    by_game: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_game[int(r.get("game_id", 0))].append(r)

    traj_metas: List[Dict[str, Any]] = []
    for gid, recs in by_game.items():
        winner = recs[0].get("winner", "RED")
        seed = recs[0].get("game_seed", 1000 + gid * 97)
        tid = recs[0].get("trajectory_id", derive_trajectory_id(seed, ["RED", "BLUE", "ORANGE", "WHITE"]))
        try:
            game_end_turn = max(int(r.get("turn", 0)) for r in recs) + 10
        except Exception:
            game_end_turn = 72
        for r in recs:
            if r.get("game_end_turn"):
                game_end_turn = int(r["game_end_turn"])
                break
        traj_metas.append(
            {
                "trajectory_id": tid,
                "game_seed": seed,
                "game_id": gid,
                "game_end_turn": game_end_turn,
                "winner": winner,
                "seat_order": ["RED", "BLUE", "ORANGE", "WHITE"],
                "seat_order_json": json.dumps(["RED", "BLUE", "ORANGE", "WHITE"]),
                "source_hash": "",
                "winner_seat_index": ["RED", "BLUE", "ORANGE", "WHITE"].index(winner) if winner in ["RED", "BLUE", "ORANGE", "WHITE"] else 0,
            }
        )

    sampled = sample_winning_trajectories(traj_metas, sample_size=args.sample_size, seed=args.sample_seed)
    sampled_ids = {t["trajectory_id"] for t in sampled}
    print(f"Sampled {len(sampled)} trajectories from {len(traj_metas)} (seed {args.sample_seed})")
    for s in sampled[:3]:
        print(f"  {s['trajectory_id']} winner {s['winner']} end {s['game_end_turn']}")

    phase2_map: Dict[tuple, str] = {}
    if args.phase2_strategies:
        p = Path(args.phase2_strategies)
        if p.suffix == ".json":
            with p.open() as f:
                raw = json.load(f)
            for k, v in raw.items():
                if "-" in k:
                    try:
                        tid, seat = k.rsplit("-", 1)
                        phase2_map[(tid, int(seat))] = v
                    except Exception:
                        phase2_map[(k, 0)] = v
                else:
                    phase2_map[(k, 0)] = v
        else:
            for r in _load_jsonl(args.phase2_strategies):
                if r.get("injected_strategy"):
                    tid = r.get("trajectory_id") or derive_trajectory_id(r.get("game_seed", 0), ["RED", "BLUE", "ORANGE", "WHITE"])
                    seat = r.get("seat_index")
                    if seat is None:
                        try:
                            seat = ["RED", "BLUE", "ORANGE", "WHITE"].index(r.get("color", "RED"))
                        except Exception:
                            seat = 0
                    phase2_map[(tid, int(seat))] = r["injected_strategy"]
        if not phase2_map and args.phase2_strategies:
            for r in _load_jsonl(args.phase2_strategies):
                if "injected_strategy" in r and r.get("game_id") is not None:
                    gid = int(r["game_id"])
                    color = r.get("color", "")
                    for t in sampled:
                        if int(t["game_id"]) == gid:
                            try:
                                seat = ["RED", "BLUE", "ORANGE", "WHITE"].index(color)
                            except ValueError:
                                seat = 0
                            phase2_map[(t["trajectory_id"], seat)] = r["injected_strategy"]
                            break

    if not phase2_map:
        for t in sampled:
            try:
                seat = ["RED", "BLUE", "ORANGE", "WHITE"].index(t["winner"])
            except Exception:
                seat = 0
            phase2_map[(t["trajectory_id"], seat)] = f"Bootstrap strategy for {t['trajectory_id']} winner {t['winner']} — balanced development with expansion toward high-pip nodes."

    config = {
        "run_id": args.run_id,
        "source_version": "sft-v1",
        "sample_seed": args.sample_seed,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "chunk_size": args.chunk_size,
        "status": "prepared",
    }
    run_id = store.create_run(config)
    print(f"Created run_id={run_id}")

    for t in sampled:
        store.upsert_trajectory(t)

    filtered_recs = [r for r in records if r["trajectory_id"] in sampled_ids]
    winner_records = []
    for r in filtered_recs:
        traj_for_r = next((t for t in sampled if t["trajectory_id"] == r["trajectory_id"]), None)
        if traj_for_r and r.get("color") == traj_for_r.get("winner"):
            winner_records.append(r)
        elif not traj_for_r:
            winner_records.append(r)
    print(f"Upserting {len(winner_records)} winner decision opportunities")
    for r in winner_records:
        if "normalized_progress" not in r:
            try:
                r["normalized_progress"] = float(r.get("turn", 0)) / float(next((t["game_end_turn"] for t in sampled if t["trajectory_id"] == r["trajectory_id"]), 72))
            except Exception:
                r["normalized_progress"] = 0.0
        store.upsert_decision_opportunity(r)

    created = sft_select_checkpoints(store, run_id, phase2_strategy_map=phase2_map)
    print(f"Created {created} checkpoints")

    cps_epoch1 = store.list_checkpoints(run_id, checkpoint_index=1)
    pending1 = [c for c in cps_epoch1 if c["status"] == "pending"]
    print(f"Epoch 1: {len(cps_epoch1)} total, {len(pending1)} pending")

    chunk_ids = store.create_batch_chunks(run_id=run_id, checkpoint_index=1, chunk_size=args.chunk_size, request_dir=args.request_dir)
    print(f"Created {len(chunk_ids)} chunks for epoch 1")
    for cid in chunk_ids:
        print(f"  {cid} -> {store.conn.execute('SELECT request_path FROM batch_chunks WHERE chunk_id=?', (cid,)).fetchone()['request_path']}")

    store.close()


def cmd_submit(args: argparse.Namespace) -> None:
    store = RunStore(args.db)
    res = submit_epoch(store, args.run_id, args.checkpoint_index, dry_run=args.dry_run, chunk_size=args.chunk_size, request_dir=args.request_dir, base_url=args.base_url)
    print(json.dumps(res, indent=2))
    store.close()


def cmd_poll(args: argparse.Namespace) -> None:
    store = RunStore(args.db)
    res = poll_epoch(store, args.run_id, args.checkpoint_index, interval=args.interval, timeout=args.timeout, base_url=args.base_url)
    print(json.dumps(res, indent=2))
    store.close()


def cmd_resume(args: argparse.Namespace) -> None:
    store = RunStore(args.db)
    resume_run(store, args.run_id, base_url=args.base_url)
    print("Resume complete")
    store.close()


def cmd_export(args: argparse.Namespace) -> None:
    store = RunStore(args.db)
    counts = sft_export_dataset(store, args.run_id, args.output, split_seed=args.split_seed)
    print(f"Exported to {args.output}: {counts}")
    store.close()


def cmd_status(args: argparse.Namespace) -> None:
    store = RunStore(args.db)
    run = store.get_run(args.run_id)
    print(json.dumps(run, indent=2, default=str))
    cps = store.list_checkpoints(args.run_id)
    print(f"Total checkpoints: {len(cps)}")
    from collections import Counter

    print(Counter(c["status"] for c in cps))
    cur = store.conn.execute("SELECT checkpoint_index, COUNT(*) as cnt FROM checkpoints WHERE run_id=? GROUP BY checkpoint_index ORDER BY checkpoint_index", (args.run_id,))
    for row in cur.fetchall():
        print(f"  epoch {row['checkpoint_index']}: {row['cnt']} checkpoints")
    chunks = store.conn.execute("SELECT * FROM batch_chunks WHERE run_id=? ORDER BY checkpoint_index, chunk_index", (args.run_id,)).fetchall()
    print(f"Chunks: {len(chunks)}")
    for ch in chunks:
        print(f"  {ch['chunk_id']} status={ch['status']} batch={ch['openai_batch_id']} accepted={ch['accepted_count']} rejected={ch['rejected_count']}")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT strategy-checkpoint pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Create run, sample trajectories, select checkpoints, prepare epoch-1 batches (offline)")
    p_prepare.add_argument("--db", default="data/sft/sft_runs.db", help="SQLite DB path (use :memory: for testing)")
    p_prepare.add_argument("--corpus", default="data/sft/high_decision_moves.jsonl", help="High-decision corpus JSONL")
    p_prepare.add_argument("--sample-size", type=int, default=500, help="Number of winning trajectories to sample")
    p_prepare.add_argument("--sample-seed", type=int, default=42)
    p_prepare.add_argument("--run-id", default=None, help="Explicit run_id or auto")
    p_prepare.add_argument("--model", default="gpt-5.6-luna")
    p_prepare.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "minimal"])
    p_prepare.add_argument("--chunk-size", type=int, default=100)
    p_prepare.add_argument("--request-dir", default="data/sft/batches")
    p_prepare.add_argument("--phase2-strategies", default=None, help="Path to phase2 strategies JSON/JSONL for bootstrap")
    p_prepare.set_defaults(func=cmd_prepare)

    p_submit = sub.add_parser("submit", help="Submit an epoch's chunks concurrently")
    p_submit.add_argument("--db", default="data/sft/sft_runs.db")
    p_submit.add_argument("--run-id", required=True)
    p_submit.add_argument("--checkpoint-index", type=int, default=1)
    p_submit.add_argument("--chunk-size", type=int, default=100)
    p_submit.add_argument("--request-dir", default="data/sft/batches")
    p_submit.add_argument("--base-url", default=None)
    p_submit.add_argument("--dry-run", action="store_true", help="Prepare without calling API")
    p_submit.set_defaults(func=cmd_submit)

    p_poll = sub.add_parser("poll", help="Poll an epoch until validated and download results")
    p_poll.add_argument("--db", default="data/sft/sft_runs.db")
    p_poll.add_argument("--run-id", required=True)
    p_poll.add_argument("--checkpoint-index", type=int, default=1)
    p_poll.add_argument("--interval", type=int, default=30)
    p_poll.add_argument("--timeout", type=int, default=3600)
    p_poll.add_argument("--base-url", default=None)
    p_poll.set_defaults(func=cmd_poll)

    p_resume = sub.add_parser("resume", help="Resume a run from SQLite, reusing batch IDs, no duplicates")
    p_resume.add_argument("--db", default="data/sft/sft_runs.db")
    p_resume.add_argument("--run-id", required=True)
    p_resume.add_argument("--base-url", default=None)
    p_resume.set_defaults(func=cmd_resume)

    p_export = sub.add_parser("export", help="Export accepted SFT JSONL with splits")
    p_export.add_argument("--db", default="data/sft/sft_runs.db")
    p_export.add_argument("--run-id", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--split-seed", type=int, default=42)
    p_export.set_defaults(func=cmd_export)

    p_status = sub.add_parser("status", help="Show run status")
    p_status.add_argument("--db", default="data/sft/sft_runs.db")
    p_status.add_argument("--run-id", required=True)
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
