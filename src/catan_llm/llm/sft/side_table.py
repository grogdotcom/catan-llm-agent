"""SFT side table + export helpers — owns band/progress + splits, JOINs to executor store."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SFT_SCHEMA = """
CREATE TABLE IF NOT EXISTS sft_checkpoint_details (
    checkpoint_id TEXT PRIMARY KEY REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE,
    band INTEGER,
    target_progress REAL,
    normalized_progress REAL,
    selection_reason TEXT
);

CREATE TABLE IF NOT EXISTS dataset_membership (
    checkpoint_id TEXT PRIMARY KEY REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE,
    split TEXT,
    split_seed INTEGER,
    export_status TEXT
);
"""


def ensure_sft_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SFT_SCHEMA)
    conn.commit()


def upsert_sft_detail(
    conn: sqlite3.Connection,
    checkpoint_id: str,
    band: Optional[int],
    target_progress: Optional[float],
    normalized_progress: Optional[float],
    selection_reason: Optional[str],
) -> None:
    conn.execute(
        """INSERT INTO sft_checkpoint_details
           (checkpoint_id, band, target_progress, normalized_progress, selection_reason)
           VALUES (?,?,?,?,?)
           ON CONFLICT(checkpoint_id) DO UPDATE SET
             band=excluded.band,
             target_progress=excluded.target_progress,
             normalized_progress=excluded.normalized_progress,
             selection_reason=excluded.selection_reason
        """,
        (checkpoint_id, band, target_progress, normalized_progress, selection_reason),
    )


def get_sft_detail(conn: sqlite3.Connection, checkpoint_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM sft_checkpoint_details WHERE checkpoint_id=?", (checkpoint_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def list_sft_details(conn: sqlite3.Connection, run_id: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """SELECT s.* FROM sft_checkpoint_details s
           JOIN checkpoints c ON c.checkpoint_id = s.checkpoint_id
           WHERE c.run_id=?""",
        (run_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def select_checkpoints(
    store: Any,
    run_id: str,
    *,
    phase2_strategy_map: Optional[Dict[Tuple[str, int], str]] = None,
) -> int:
    """SFT checkpoint selection — K-bands + lineage, writes to executor + side table.

    Transactionally inserts generic rows into ``checkpoints`` and SFT-specific
    columns into ``sft_checkpoint_details``. Returns number created.
    """
    from catan_llm.llm.sft.selection import collapse_opportunities, filter_midgame_candidates, select_checkpoints_for_trajectory

    ensure_sft_tables(store.conn)
    phase2_strategy_map = phase2_strategy_map or {}
    trajs = store.list_trajectories()
    all_opps = store.list_opportunities()

    from collections import defaultdict

    opps_by_player: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for o in all_opps:
        try:
            src = json.loads(o["source_record_json"]) if o["source_record_json"] else {}
        except Exception:
            src = {}
        rec = dict(src)
        rec["trajectory_id"] = o["trajectory_id"]
        rec["seat_index"] = o["seat_index"]
        rec["color"] = o["color"]
        rec["turn"] = o["turn"]
        rec["phase"] = o["phase"]
        rec["trajectory_index"] = o["trajectory_index"]
        rec["game_id"] = o["game_id"]
        rec["num_moves"] = o["num_moves"]
        if o["normalized_progress"] is not None:
            rec["normalized_progress"] = o["normalized_progress"]
        opps_by_player[(o["trajectory_id"], int(o["seat_index"]))].append(rec)

    created = 0
    with store.conn:
        for traj in trajs:
            tid = traj["trajectory_id"]
            winner = traj["winner"]
            try:
                seat_order = json.loads(traj["seat_order_json"] or "[]")
            except Exception:
                seat_order = []
            winner_seat = None
            if winner and seat_order:
                try:
                    winner_seat = seat_order.index(winner)
                except ValueError:
                    winner_seat = None
            players_to_process = []
            if winner_seat is not None:
                players_to_process = [(tid, winner_seat)]
            else:
                players_to_process = [k for k in opps_by_player.keys() if k[0] == tid]

            for (ptraj, pseat) in players_to_process:
                opps = opps_by_player.get((ptraj, pseat), [])
                if not opps:
                    continue
                candidates = filter_midgame_candidates(opps)
                game_end_turn = int(traj.get("game_end_turn") or 60)
                for c in candidates:
                    if "normalized_progress" not in c or c["normalized_progress"] is None:
                        try:
                            c["normalized_progress"] = float(c.get("turn", 0)) / float(game_end_turn) if game_end_turn else 0.0
                        except Exception:
                            c["normalized_progress"] = 0.0
                collapsed = collapse_opportunities(candidates)
                for c in collapsed:
                    if "normalized_progress" not in c or c["normalized_progress"] is None:
                        c["normalized_progress"] = float(c.get("turn", 0)) / float(game_end_turn) if game_end_turn else 0.0

                selected = select_checkpoints_for_trajectory(
                    collapsed=collapsed,
                    game_end_turn=game_end_turn,
                    trajectory_id=ptraj,
                    seat_index=pseat,
                )
                for idx, sel in enumerate(selected, start=1):
                    opp = sel["opportunity"]
                    opp_id = None
                    for stored in all_opps:
                        if stored["trajectory_id"] == ptraj and stored["seat_index"] == pseat and stored["turn"] == opp.get("turn") and stored["phase"] == opp.get("phase"):
                            opp_id = stored["opportunity_id"]
                            break
                    if opp_id is None:
                        key = f"{ptraj}-{pseat}-{opp.get('turn')}-{opp.get('phase')}-{opp.get('trajectory_index')}"
                        opp_id = _hash_bytes(key.encode())[:16]

                    from catan_llm.llm.strategy import resolve_lineage

                    prev_accepted = None
                    prev_ckpt_id = None
                    if idx > 1:
                        prev_ckpt_id = f"{run_id}-{ptraj}-{pseat}-{idx-1}"
                        cur2 = store.conn.execute("SELECT * FROM outputs WHERE checkpoint_id=?", (prev_ckpt_id,))
                        prev_out = cur2.fetchone()
                        if prev_out and prev_out["validation_status"] == "accepted" and prev_out["strategy_out"]:
                            prev_accepted = prev_out["strategy_out"]

                    lineage = resolve_lineage(
                        idx,
                        phase2_strategy=phase2_strategy_map.get((ptraj, pseat)),
                        prev_accepted_strategy=prev_accepted,
                        prev_checkpoint_id=prev_ckpt_id,
                    )
                    strategy_in = lineage.strategy_in
                    strategy_source = lineage.strategy_source
                    status = lineage.status
                    skip_reason = lineage.skip_reason

                    checkpoint_id = f"{run_id}-{ptraj}-{pseat}-{idx}"
                    # Generic checkpoint
                    store.conn.execute(
                        """INSERT INTO checkpoints
                           (checkpoint_id, run_id, trajectory_id, seat_index, checkpoint_index,
                            checkpoint_count, opportunity_id,
                            strategy_in, strategy_source_checkpoint, status, skip_reason)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(checkpoint_id) DO UPDATE SET
                             run_id=excluded.run_id,
                             trajectory_id=excluded.trajectory_id,
                             seat_index=excluded.seat_index,
                             checkpoint_index=excluded.checkpoint_index,
                             checkpoint_count=excluded.checkpoint_count,
                             opportunity_id=excluded.opportunity_id,
                             strategy_in=excluded.strategy_in,
                             strategy_source_checkpoint=excluded.strategy_source_checkpoint,
                             status=excluded.status,
                             skip_reason=excluded.skip_reason
                        """,
                        (
                            checkpoint_id,
                            run_id,
                            ptraj,
                            pseat,
                            idx,
                            sel["checkpoint_count"],
                            opp_id,
                            strategy_in,
                            strategy_source,
                            status if idx == 1 or status != "pending" else status,
                            skip_reason,
                        ),
                    )
                    # SFT details
                    upsert_sft_detail(
                        store.conn,
                        checkpoint_id=checkpoint_id,
                        band=sel["checkpoint_band"],
                        target_progress=sel["target_progress"],
                        normalized_progress=sel["normalized_progress"],
                        selection_reason=sel["selection_reason"],
                    )
                    created += 1
    return created


def _prompt_with_strategy(prompt: str, strategy_in: Optional[str]) -> str:
    from catan_llm.llm.prompt.strategy import ensure_strategy_block

    return ensure_strategy_block(prompt, strategy_in or "None")


def assign_splits(store: Any, run_id: str, split_seed: int = 42) -> None:
    """Assign 80/10/10 splits to accepted checkpoints (SFT export policy)."""
    ensure_sft_tables(store.conn)
    cps = store.list_checkpoints(run_id)
    accepted_ids = []
    for cp in cps:
        cur = store.conn.execute("SELECT validation_status FROM outputs WHERE checkpoint_id=?", (cp["checkpoint_id"],))
        row = cur.fetchone()
        if row and row["validation_status"] == "accepted":
            accepted_ids.append(cp["checkpoint_id"])
    rng = random.Random(split_seed)
    rng.shuffle(accepted_ids)
    n = len(accepted_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    with store.conn:
        for idx, cid in enumerate(accepted_ids):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "validation"
            else:
                split = "test"
            store.conn.execute(
                """INSERT INTO dataset_membership (checkpoint_id, split, split_seed, export_status)
                   VALUES (?,?,?,?)
                   ON CONFLICT(checkpoint_id) DO UPDATE SET split=excluded.split, split_seed=excluded.split_seed
                """,
                (cid, split, split_seed, "pending"),
            )


def export_dataset(store: Any, run_id: str, output_path: str, split_seed: int = 42) -> Dict[str, int]:
    """Export accepted full-reasoning SFT JSONL records (JOIN executor + side table)."""
    ensure_sft_tables(store.conn)
    assign_splits(store, run_id, split_seed=split_seed)
    cps = store.list_checkpoints(run_id)
    from catan_llm.llm.dataset import build_sft_record

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "test": 0, "total": 0}
    with out_p.open("w", encoding="utf-8") as f, store.conn:
        for cp in cps:
            cur = store.conn.execute("SELECT * FROM outputs WHERE checkpoint_id=?", (cp["checkpoint_id"],))
            out = cur.fetchone()
            if not out or out["validation_status"] != "accepted":
                continue
            cur2 = store.conn.execute("SELECT * FROM dataset_membership WHERE checkpoint_id=?", (cp["checkpoint_id"],))
            mem = cur2.fetchone()
            split = mem["split"] if mem else "train"
            cur3 = store.conn.execute("SELECT * FROM decision_opportunities WHERE opportunity_id=?", (cp["opportunity_id"],))
            opp = cur3.fetchone()
            try:
                src = json.loads(opp["source_record_json"] or "{}") if opp else {}
            except Exception:
                src = {}
            prompt_with_strategy = _prompt_with_strategy(src.get("prompt", ""), cp.get("strategy_in") or "None")
            traj = store.conn.execute("SELECT * FROM trajectories WHERE trajectory_id=?", (cp["trajectory_id"],)).fetchone()
            # Enrich checkpoint with SFT band info for export
            sft_detail = get_sft_detail(store.conn, cp["checkpoint_id"])
            cp_enriched = dict(cp)
            if sft_detail:
                # Keep original key 'band' for exporter compatibility (it reads checkpoint.get("band"))
                cp_enriched["band"] = sft_detail.get("band")
                cp_enriched["target_progress"] = sft_detail.get("target_progress")
                # normalized_progress already via side table but also in opportunity; prefer side table if needed
                if sft_detail.get("normalized_progress") is not None:
                    cp_enriched["normalized_progress"] = sft_detail.get("normalized_progress")
            else:
                cp_enriched["band"] = None
            record = build_sft_record(
                checkpoint=cp_enriched,
                output=dict(out),
                opportunity=dict(opp) if opp else None,
                trajectory=dict(traj) if traj else None,
                prompt=prompt_with_strategy,
                split=split,
                split_seed=split_seed,
                run_id=run_id,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] = counts.get(split, 0) + 1
            counts["total"] += 1
            store.conn.execute("UPDATE dataset_membership SET export_status=? WHERE checkpoint_id=?", ("exported", cp["checkpoint_id"]))
    return counts
