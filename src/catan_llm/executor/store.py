"""SQLite-backed RunStore — generic executor control + record plane.

Generic counterpart to the former MidgameRunStore. No SFT concepts here:
checkpoints are plain chained steps; band/progress columns live in the
SFT side table (not in this store). Batch transport and strategy chaining
are shared across pipelines (SFT batch, RL inline).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return _hash_bytes(p.read_bytes())


def _prompt_with_strategy(prompt: str, strategy_in: Optional[str]) -> str:
    """Render the [CURRENT STRATEGY] block deterministically (canonical helper)."""
    from catan_llm.prompt.strategy import ensure_strategy_block

    return ensure_strategy_block(prompt, strategy_in or "None")


def _build_request(checkpoint_id: str, prompt: str, model: str, reasoning_effort: str) -> Dict[str, Any]:
    """Build a Responses-API batch line for a checkpoint."""
    from catan_llm.teacher.models import DEFAULT_SYSTEM_PROMPT

    messages: List[Dict[str, str]] = []
    if DEFAULT_SYSTEM_PROMPT and DEFAULT_SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    return {
        "custom_id": checkpoint_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": messages,
            "max_output_tokens": 4096,
            "reasoning": {"effort": reasoning_effort},
        },
    }


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_version TEXT,
    sample_seed INTEGER,
    model TEXT,
    reasoning_effort TEXT,
    chunk_size INTEGER,
    status TEXT,
    current_checkpoint INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    game_seed INTEGER,
    game_id INTEGER,
    game_end_turn INTEGER,
    winner TEXT,
    seat_order_json TEXT,
    source_hash TEXT
);

CREATE TABLE IF NOT EXISTS players (
    trajectory_id TEXT,
    seat_index INTEGER,
    color TEXT,
    is_winner INTEGER,
    PRIMARY KEY (trajectory_id, seat_index),
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS decision_opportunities (
    opportunity_id TEXT PRIMARY KEY,
    trajectory_id TEXT,
    seat_index INTEGER,
    color TEXT,
    turn INTEGER,
    phase TEXT,
    trajectory_index INTEGER,
    normalized_progress REAL,
    prompt_json TEXT,
    engine_completion TEXT,
    source_record_json TEXT,
    game_id INTEGER,
    num_moves INTEGER,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT,
    trajectory_id TEXT,
    seat_index INTEGER,
    checkpoint_index INTEGER,
    checkpoint_count INTEGER,
    opportunity_id TEXT,
    strategy_in TEXT,
    strategy_source_checkpoint TEXT,
    status TEXT,
    skip_reason TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id) ON DELETE CASCADE,
    FOREIGN KEY (opportunity_id) REFERENCES decision_opportunities(opportunity_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS batch_chunks (
    chunk_id TEXT PRIMARY KEY,
    run_id TEXT,
    checkpoint_index INTEGER,
    chunk_index INTEGER,
    request_path TEXT,
    request_hash TEXT,
    openai_batch_id TEXT,
    result_path TEXT,
    result_hash TEXT,
    status TEXT,
    submitted_at TEXT,
    completed_at TEXT,
    accepted_count INTEGER DEFAULT 0,
    rejected_count INTEGER DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS outputs (
    checkpoint_id TEXT PRIMARY KEY,
    raw_output_path TEXT,
    raw_output_hash TEXT,
    think_text TEXT,
    strategy_out TEXT,
    predicted_action INTEGER,
    validation_status TEXT,
    rejection_reason TEXT,
    parsed_at TEXT,
    raw_output_text TEXT,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE
);
"""


class RunStore:
    """Durable control + record plane for the generic executor."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_db()

    def _init_db(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # -- runs --

    def create_run(self, config: Dict[str, Any]) -> str:
        """Atomic creation of run row. Returns run_id."""
        run_id = config.get("run_id") or f"run-{uuid.uuid4().hex[:8]}"
        now = _now()
        row = {
            "run_id": run_id,
            "source_version": config.get("source_version", "v1"),
            "sample_seed": config.get("sample_seed", 42),
            "model": config.get("model", "gpt-5.6-luna"),
            "reasoning_effort": config.get("reasoning_effort", "medium"),
            "chunk_size": config.get("chunk_size", 100),
            "status": config.get("status", "prepared"),
            "current_checkpoint": config.get("current_checkpoint", 0),
            "created_at": now,
            "updated_at": now,
            "config_json": json.dumps(config, ensure_ascii=False),
        }
        with self.conn:
            self.conn.execute(
                """INSERT INTO runs (run_id, source_version, sample_seed, model, reasoning_effort,
                   chunk_size, status, current_checkpoint, created_at, updated_at, config_json)
                   VALUES (:run_id, :source_version, :sample_seed, :model, :reasoning_effort,
                   :chunk_size, :status, :current_checkpoint, :created_at, :updated_at, :config_json)""",
                row,
            )
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        r = cur.fetchone()
        return dict(r) if r else None

    def update_run_status(self, run_id: str, status: str, current_checkpoint: Optional[int] = None) -> None:
        with self.conn:
            if current_checkpoint is not None:
                self.conn.execute(
                    "UPDATE runs SET status=?, current_checkpoint=?, updated_at=? WHERE run_id=?",
                    (status, current_checkpoint, _now(), run_id),
                )
            else:
                self.conn.execute("UPDATE runs SET status=?, updated_at=? WHERE run_id=?", (status, _now(), run_id))

    # -- trajectories --

    def upsert_trajectory(self, metadata: Dict[str, Any]) -> None:
        """Upsert trajectory + players."""
        tid = metadata["trajectory_id"]
        seat_order = metadata.get("seat_order") or metadata.get("seat_order_json")
        if isinstance(seat_order, list):
            seat_order_json = json.dumps(seat_order)
        elif isinstance(seat_order, str):
            seat_order_json = seat_order
            seat_order = json.loads(seat_order) if seat_order else []
        else:
            seat_order_json = json.dumps([])
            seat_order = []

        winner = metadata.get("winner")
        with self.conn:
            self.conn.execute(
                """INSERT INTO trajectories (trajectory_id, game_seed, game_id, game_end_turn, winner, seat_order_json, source_hash)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(trajectory_id) DO UPDATE SET
                     game_seed=excluded.game_seed,
                     game_id=excluded.game_id,
                     game_end_turn=excluded.game_end_turn,
                     winner=excluded.winner,
                     seat_order_json=excluded.seat_order_json,
                     source_hash=excluded.source_hash
                """,
                (
                    tid,
                    metadata.get("game_seed"),
                    metadata.get("game_id"),
                    metadata.get("game_end_turn"),
                    winner,
                    seat_order_json,
                    metadata.get("source_hash", ""),
                ),
            )
            # Players
            players = metadata.get("players")
            if players is None and seat_order:
                players = [
                    {"seat_index": idx, "color": c, "is_winner": 1 if c == winner else 0}
                    for idx, c in enumerate(seat_order)
                ]
            if players:
                for p in players:
                    self.conn.execute(
                        """INSERT INTO players (trajectory_id, seat_index, color, is_winner)
                           VALUES (?,?,?,?)
                           ON CONFLICT(trajectory_id, seat_index) DO UPDATE SET
                             color=excluded.color, is_winner=excluded.is_winner
                        """,
                        (tid, int(p["seat_index"]), p["color"], int(p.get("is_winner", 0))),
                    )

    def list_trajectories(self) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM trajectories")
        return [dict(r) for r in cur.fetchall()]

    # -- opportunities --

    def upsert_decision_opportunity(self, record: Dict[str, Any]) -> str:
        """Insert or replace a decision opportunity. Returns opportunity_id."""
        tid = record.get("trajectory_id")
        if not tid:
            raise ValueError("record missing trajectory_id")
        opp_id = record.get("opportunity_id")
        if not opp_id:
            key = f"{tid}-{record.get('seat_index')}-{record.get('turn')}-{record.get('phase')}-{record.get('trajectory_index')}"
            opp_id = _hash_bytes(key.encode())[:16]

        with self.conn:
            self.conn.execute(
                """INSERT INTO decision_opportunities
                   (opportunity_id, trajectory_id, seat_index, color, turn, phase, trajectory_index,
                    normalized_progress, prompt_json, engine_completion, source_record_json, game_id, num_moves)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(opportunity_id) DO UPDATE SET
                     trajectory_id=excluded.trajectory_id,
                     seat_index=excluded.seat_index,
                     color=excluded.color,
                     turn=excluded.turn,
                     phase=excluded.phase,
                     trajectory_index=excluded.trajectory_index,
                     normalized_progress=excluded.normalized_progress,
                     prompt_json=excluded.prompt_json,
                     engine_completion=excluded.engine_completion,
                     source_record_json=excluded.source_record_json,
                     game_id=excluded.game_id,
                     num_moves=excluded.num_moves
                """,
                (
                    opp_id,
                    tid,
                    record.get("seat_index"),
                    record.get("color"),
                    record.get("turn"),
                    record.get("phase"),
                    record.get("trajectory_index"),
                    record.get("normalized_progress"),
                    json.dumps(record.get("prompt", ""), ensure_ascii=False) if isinstance(record.get("prompt"), str) else json.dumps(record.get("prompt", "")),
                    str(record.get("completion") or record.get("engine_completion") or ""),
                    json.dumps(record, ensure_ascii=False),
                    record.get("game_id"),
                    record.get("num_moves"),
                ),
            )
        return opp_id

    def list_opportunities(self, trajectory_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if trajectory_id:
            cur = self.conn.execute("SELECT * FROM decision_opportunities WHERE trajectory_id=? ORDER BY trajectory_index", (trajectory_id,))
        else:
            cur = self.conn.execute("SELECT * FROM decision_opportunities ORDER BY trajectory_id, trajectory_index")
        return [dict(r) for r in cur.fetchall()]

    # -- checkpoints (generic) --

    def register_checkpoints(self, run_id: str, checkpoints: List[Dict[str, Any]]) -> int:
        """Generic chain registration — insert generic checkpoint rows.

        Each dict must contain:
          checkpoint_id, trajectory_id, seat_index, checkpoint_index,
          checkpoint_count, opportunity_id, strategy_in, strategy_source_checkpoint,
          status, skip_reason
        Missing keys default to None / pending.
        Returns number inserted.
        """
        created = 0
        with self.conn:
            for cp in checkpoints:
                self.conn.execute(
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
                        cp["checkpoint_id"],
                        run_id,
                        cp.get("trajectory_id"),
                        cp.get("seat_index"),
                        cp.get("checkpoint_index"),
                        cp.get("checkpoint_count"),
                        cp.get("opportunity_id"),
                        cp.get("strategy_in"),
                        cp.get("strategy_source_checkpoint"),
                        cp.get("status", "pending"),
                        cp.get("skip_reason"),
                    ),
                )
                created += 1
        return created

    def list_checkpoints(self, run_id: str, checkpoint_index: Optional[int] = None) -> List[Dict[str, Any]]:
        if checkpoint_index is not None:
            cur = self.conn.execute(
                "SELECT * FROM checkpoints WHERE run_id=? AND checkpoint_index=? ORDER BY trajectory_id, seat_index", (run_id, checkpoint_index)
            )
        else:
            cur = self.conn.execute("SELECT * FROM checkpoints WHERE run_id=? ORDER BY checkpoint_index, trajectory_id", (run_id,))
        return [dict(r) for r in cur.fetchall()]

    def pending_batch_chunks(self, run_id: str, checkpoint_index: int) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM batch_chunks WHERE run_id=? AND checkpoint_index=? ORDER BY chunk_index", (run_id, checkpoint_index)
        )
        return [dict(r) for r in cur.fetchall()]

    # -- batch chunks --

    def create_batch_chunks(
        self,
        *,
        run_id: str,
        checkpoint_index: int,
        chunk_size: int = 100,
        request_dir: str = "data/executor/batches",
        base_output_dir: Optional[str] = None,
    ) -> List[str]:
        """Partition pending checkpoints of an epoch into ~100-request JSONL chunks.

        Persists batch_chunks rows. Returns list of chunk_ids.
        """
        cps = self.list_checkpoints(run_id, checkpoint_index=checkpoint_index)
        # Only pending ones (not skipped)
        pending = [c for c in cps if c["status"] == "pending"]
        if not pending:
            return []
        # Chunk
        chunks: List[List[Dict[str, Any]]] = []
        for i in range(0, len(pending), chunk_size):
            chunks.append(pending[i : i + chunk_size])

        chunk_ids: List[str] = []
        with self.conn:
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{run_id}-ckpt{checkpoint_index}-chunk{idx}"
                # Check if already exists (idempotent)
                cur = self.conn.execute("SELECT * FROM batch_chunks WHERE chunk_id=?", (chunk_id,))
                existing = cur.fetchone()
                if existing:
                    chunk_ids.append(chunk_id)
                    continue
                # Build request file path
                dir_path = Path(request_dir) / run_id
                dir_path.mkdir(parents=True, exist_ok=True)
                request_path = str(dir_path / f"ckpt{checkpoint_index}_chunk{idx}.batch.jsonl")
                # Build file content
                lines = []
                for cp in chunk:
                    opp_id = cp["opportunity_id"]
                    cur2 = self.conn.execute("SELECT * FROM decision_opportunities WHERE opportunity_id=?", (opp_id,))
                    opp_row = cur2.fetchone()
                    if not opp_row:
                        continue
                    try:
                        src = json.loads(opp_row["source_record_json"] or "{}")
                    except Exception:
                        src = {}
                    prompt = src.get("prompt", "")
                    # Inject strategy_in deterministically via canonical helper
                    prompt = _prompt_with_strategy(prompt, cp.get("strategy_in") or "None")

                    # Build batch request line (Responses API, unified helper)
                    run = self.get_run(run_id) or {}
                    model = run.get("model", "gpt-5.6-luna")
                    reasoning_effort = run.get("reasoning_effort", "medium")
                    req = _build_request(cp["checkpoint_id"], prompt, model, reasoning_effort)
                    lines.append(req)

                # Write file
                with open(request_path, "w", encoding="utf-8") as f:
                    for ln in lines:
                        f.write(json.dumps(ln, ensure_ascii=False) + "\n")
                request_hash = _hash_file(request_path) if Path(request_path).exists() else ""
                self.conn.execute(
                    """INSERT INTO batch_chunks
                       (chunk_id, run_id, checkpoint_index, chunk_index, request_path, request_hash, status, submitted_at)
                       VALUES (?,?,?,?,?,?,?,?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         request_path=excluded.request_path,
                         request_hash=excluded.request_hash,
                         status=excluded.status
                    """,
                    (chunk_id, run_id, checkpoint_index, idx, request_path, request_hash, "prepared", _now()),
                )
                chunk_ids.append(chunk_id)
        return chunk_ids

    def record_batch_submission(self, chunk_id: str, batch_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE batch_chunks SET openai_batch_id=?, status=?, submitted_at=? WHERE chunk_id=?",
                (batch_id, "submitted", _now(), chunk_id),
            )

    def mark_chunk_polling(self, chunk_id: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE batch_chunks SET status=? WHERE chunk_id=?", ("polling", chunk_id))

    def import_batch_results(self, chunk_id: str, result_path: str) -> Dict[str, int]:
        """Parse result JSONL, validate each line, write outputs + checkpoint status.

        Returns {"accepted": int, "rejected": int}
        """
        from catan_llm.sft.validation import validate_response, validate_batch_response_obj

        p = Path(result_path)
        if not p.exists():
            raise FileNotFoundError(f"result file not found: {result_path}")
        result_hash = _hash_file(result_path)
        # Load lines
        lines: List[Dict[str, Any]] = []
        with p.open() as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))

        accepted = 0
        rejected = 0
        # Need run_id and checkpoint info to validate lineage
        cur = self.conn.execute("SELECT * FROM batch_chunks WHERE chunk_id=?", (chunk_id,))
        chunk_row = cur.fetchone()
        if not chunk_row:
            raise ValueError(f"chunk_id not found: {chunk_id}")
        run_id = chunk_row["run_id"]

        with self.conn:
            for obj in lines:
                custom_id = obj.get("custom_id", "")
                # Find checkpoint
                cur2 = self.conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id=?", (custom_id,))
                cp = cur2.fetchone()
                if not cp:
                    continue
                # Extract response text
                resp_ok, reason = validate_batch_response_obj(obj)
                if not resp_ok:
                    # api_error
                    self.conn.execute(
                        """INSERT INTO outputs
                           (checkpoint_id, raw_output_path, raw_output_hash, validation_status, rejection_reason, parsed_at, raw_output_text)
                           VALUES (?,?,?,?,?,?,?)
                           ON CONFLICT(checkpoint_id) DO UPDATE SET
                             raw_output_path=excluded.raw_output_path,
                             raw_output_hash=excluded.raw_output_hash,
                             validation_status=excluded.validation_status,
                             rejection_reason=excluded.rejection_reason,
                             parsed_at=excluded.parsed_at,
                             raw_output_text=excluded.raw_output_text
                        """,
                        (custom_id, result_path, result_hash, "rejected", reason, _now(), json.dumps(obj, ensure_ascii=False)),
                    )
                    rejected += 1
                    continue

                # Extract text from batch response body (unified helper)
                from catan_llm.teacher.parsing import extract_text_from_batch_body

                resp = obj.get("response", obj)
                body = resp.get("body", resp)
                text = extract_text_from_batch_body(body) if isinstance(body, dict) else ""
                if not text and isinstance(obj, dict):
                    text = extract_text_from_batch_body(obj)

                # Load prompt for validation (need engine completion etc)
                opp_id = cp["opportunity_id"]
                cur3 = self.conn.execute("SELECT * FROM decision_opportunities WHERE opportunity_id=?", (opp_id,))
                opp_row = cur3.fetchone()
                engine_completion = opp_row["engine_completion"] if opp_row else ""
                num_moves = opp_row["num_moves"] if opp_row and opp_row["num_moves"] else None
                strategy_in = cp["strategy_in"]
                try:
                    src = json.loads(opp_row["source_record_json"] or "{}") if opp_row else {}
                except Exception:
                    src = {}
                prompt = _prompt_with_strategy(src.get("prompt", ""), strategy_in or "None")

                v = validate_response(
                    prompt=prompt,
                    response_text=text,
                    engine_completion=engine_completion,
                    num_moves=num_moves,
                    strategy_in=strategy_in,
                    strategy_lineage_ok=True,
                )
                if v.accepted:
                    self.conn.execute(
                        """INSERT INTO outputs
                           (checkpoint_id, raw_output_path, raw_output_hash, think_text, strategy_out, predicted_action, validation_status, rejection_reason, parsed_at, raw_output_text)
                           VALUES (?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(checkpoint_id) DO UPDATE SET
                             raw_output_path=excluded.raw_output_path,
                             raw_output_hash=excluded.raw_output_hash,
                             think_text=excluded.think_text,
                             strategy_out=excluded.strategy_out,
                             predicted_action=excluded.predicted_action,
                             validation_status=excluded.validation_status,
                             rejection_reason=excluded.rejection_reason,
                             parsed_at=excluded.parsed_at,
                             raw_output_text=excluded.raw_output_text
                        """,
                        (
                            custom_id,
                            result_path,
                            result_hash,
                            v.think_text,
                            v.strategy_out,
                            v.predicted_action,
                            "accepted",
                            None,
                            _now(),
                            text,
                        ),
                    )
                    self.conn.execute("UPDATE checkpoints SET status=? WHERE checkpoint_id=?", ("validated", custom_id))
                    accepted += 1
                else:
                    self.conn.execute(
                        """INSERT INTO outputs
                           (checkpoint_id, raw_output_path, raw_output_hash, think_text, strategy_out, predicted_action, validation_status, rejection_reason, parsed_at, raw_output_text)
                           VALUES (?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(checkpoint_id) DO UPDATE SET
                             raw_output_path=excluded.raw_output_path,
                             raw_output_hash=excluded.raw_output_hash,
                             think_text=excluded.think_text,
                             strategy_out=excluded.strategy_out,
                             predicted_action=excluded.predicted_action,
                             validation_status=excluded.validation_status,
                             rejection_reason=excluded.rejection_reason,
                             parsed_at=excluded.parsed_at,
                             raw_output_text=excluded.raw_output_text
                        """,
                        (
                            custom_id,
                            result_path,
                            result_hash,
                            v.think_text,
                            v.strategy_out,
                            v.predicted_action,
                            "rejected",
                            v.rejection_reason,
                            _now(),
                            text,
                        ),
                    )
                    self.conn.execute("UPDATE checkpoints SET status=? WHERE checkpoint_id=?", ("rejected", custom_id))
                    rejected += 1

            # Update chunk status
            self.conn.execute(
                "UPDATE batch_chunks SET result_path=?, result_hash=?, status=?, completed_at=?, accepted_count=?, rejected_count=? WHERE chunk_id=?",
                (result_path, result_hash, "validated", _now(), accepted, rejected, chunk_id),
            )
        return {"accepted": accepted, "rejected": rejected}

    def advance_checkpoint(self, run_id: str) -> Optional[int]:
        """If current epoch is fully validated, propagate strategy to next epoch and advance.

        Returns next checkpoint_index if advanced, None if not ready.
        Spec §13.2: do not advance until current epoch is validated.
        """
        cur_run = self.get_run(run_id)
        if not cur_run:
            raise ValueError(f"run not found: {run_id}")
        current = int(cur_run.get("current_checkpoint") or 0)
        # Determine next epoch to check: if current==0, first epoch is 1
        next_epoch = current + 1 if current else 1
        # Check if there are chunks for next_epoch
        chunks = self.pending_batch_chunks(run_id, next_epoch)
        # Determine max checkpoint index for run
        cur = self.conn.execute("SELECT MAX(checkpoint_index) as mx FROM checkpoints WHERE run_id=?", (run_id,))
        mx_row = cur.fetchone()
        max_idx = int(mx_row["mx"]) if mx_row and mx_row["mx"] is not None else 0
        if next_epoch > max_idx:
            return None  # No more epochs
        # If chunks exist, check they are validated
        if chunks:
            for ch in chunks:
                if ch["status"] not in ("validated", "downloaded"):
                    return None  # Not ready
        else:
            # No chunks means this epoch had no pending checkpoints (all skipped) -> can advance
            pass

        # Propagate strategy to next epoch's pending checkpoints where needed
        with self.conn:
            nxt_cps = self.list_checkpoints(run_id, checkpoint_index=next_epoch)
            if next_epoch + 1 <= max_idx:
                future_cps = self.list_checkpoints(run_id, checkpoint_index=next_epoch + 1)
                for fc in future_cps:
                    if fc["strategy_in"] is not None and fc["status"] != "skipped":
                        continue
                    if fc["strategy_in"] is not None and fc["status"] == "pending":
                        continue
                    # Find source checkpoint (same trajectory/seat, index = future-1)
                    src_id = f"{run_id}-{fc['trajectory_id']}-{fc['seat_index']}-{next_epoch}"
                    cur2 = self.conn.execute("SELECT * FROM outputs WHERE checkpoint_id=?", (src_id,))
                    src_out = cur2.fetchone()
                    if src_out and src_out["validation_status"] == "accepted" and src_out["strategy_out"]:
                        self.conn.execute(
                            "UPDATE checkpoints SET strategy_in=?, strategy_source_checkpoint=?, status=? WHERE checkpoint_id=?",
                            (src_out["strategy_out"], src_id, "pending", fc["checkpoint_id"]),
                        )
                    else:
                        # Broken lineage -> mark skipped
                        self.conn.execute(
                            "UPDATE checkpoints SET status=?, skip_reason=? WHERE checkpoint_id=?",
                            ("skipped", "broken_strategy_lineage", fc["checkpoint_id"]),
                        )
            # Advance run's current_checkpoint
            self.conn.execute("UPDATE runs SET current_checkpoint=?, status=?, updated_at=? WHERE run_id=?", (next_epoch, "advanced" if next_epoch < max_idx else "validated", _now(), run_id))
        return next_epoch

    def is_epoch_validated(self, run_id: str, checkpoint_index: int) -> bool:
        chunks = self.pending_batch_chunks(run_id, checkpoint_index)
        if not chunks:
            # No chunks -> check checkpoints status directly
            cps = self.list_checkpoints(run_id, checkpoint_index=checkpoint_index)
            if not cps:
                return True
            return all(c["status"] in ("validated", "rejected", "skipped") for c in cps)
        return all(c["status"] == "validated" for c in chunks)
