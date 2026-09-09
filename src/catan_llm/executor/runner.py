"""Durable batch orchestration for the generic executor. Spec §13.

- ~100 requests per chunk
- Submit all chunks for an epoch concurrently
- Persist request paths, hashes, batch IDs, result paths, statuses in SQLite
- poll/resume without duplicate submissions
- Do not advance to next checkpoint until current epoch validated
- Offline prepare/dry-run without API credentials
- Resident ``RunExecutor.run()`` drives itself; step methods retained for tests.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from catan_llm.executor.spec import RetryPolicy, RunSpec
from catan_llm.executor.store import RunStore

# Backwards-compat alias for tests that patch executor.runner._get_client
MidgameRunStore = RunStore


def _get_client(model: str, base_url: Optional[str] = None):
    try:
        from catan_llm.teacher import TeacherGateway

        return TeacherGateway(model=model, base_url=base_url)
    except Exception as exc:
        raise RuntimeError(f"OpenAI client unavailable: {exc}") from exc


def prepare_epoch(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    *,
    chunk_size: int = 100,
    request_dir: str = "data/executor/batches",
) -> List[str]:
    """Create batch JSONL chunks for an epoch (offline, no API). Idempotent."""
    return store.create_batch_chunks(
        run_id=run_id, checkpoint_index=checkpoint_index, chunk_size=chunk_size, request_dir=request_dir
    )


def submit_epoch(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    *,
    dry_run: bool = False,
    chunk_size: int = 100,
    request_dir: str = "data/executor/batches",
    base_url: Optional[str] = None,
    max_workers: int = 5,
) -> Dict[str, Any]:
    """Submit all chunks for an epoch concurrently. Persists batch IDs.

    If ``dry_run`` is True or API credentials absent, only prepares files
    and marks chunks as prepared without calling OpenAI.
    """
    chunk_ids = prepare_epoch(store, run_id, checkpoint_index, chunk_size=chunk_size, request_dir=request_dir)
    if not chunk_ids:
        return {"chunk_ids": [], "batch_ids": [], "dry_run": dry_run, "message": "no pending checkpoints for epoch"}

    if dry_run:
        return {"chunk_ids": chunk_ids, "batch_ids": [], "dry_run": True}

    run = store.get_run(run_id)
    model = run.get("model", "gpt-5.6-luna") if run else "gpt-5.6-luna"

    # If API key missing, fallback to dry_run behavior
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        return {"chunk_ids": chunk_ids, "batch_ids": [], "dry_run": True, "reason": "OPENAI_API_KEY missing"}

    client = _get_client(model=model, base_url=base_url)

    # Only submit chunks that haven't been submitted yet (resume semantics)
    to_submit: List[str] = []
    for cid in chunk_ids:
        cur = store.conn.execute("SELECT * FROM batch_chunks WHERE chunk_id=?", (cid,))
        row = cur.fetchone()
        if row and row["openai_batch_id"]:
            continue
        if row and row["status"] in ("submitted", "polling", "validated"):
            # Already submitted/validated => skip duplicate
            if row["openai_batch_id"]:
                continue
        to_submit.append(cid)

    if not to_submit:
        return {"chunk_ids": chunk_ids, "batch_ids": [], "already_submitted": True}

    batch_ids: List[str] = []
    # Submit concurrently
    def _upload_and_create(cid: str) -> str:
        cur = store.conn.execute("SELECT request_path FROM batch_chunks WHERE chunk_id=?", (cid,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"chunk not found: {cid}")
        req_path = row["request_path"]
        # Use OpenAI Batch via files + batches
        file_id = client.upload_file(req_path)
        batch = client.create_batch(file_id, endpoint="/v1/responses", completion_window="24h", metadata={"run_id": run_id, "checkpoint_index": str(checkpoint_index)})
        batch_id = getattr(batch, "id", batch.get("id") if isinstance(batch, dict) else str(batch))
        store.record_batch_submission(cid, batch_id)
        return batch_id

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_upload_and_create, cid): cid for cid in to_submit}
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                bid = fut.result()
                batch_ids.append(bid)
            except Exception as exc:
                # Keep chunk in prepared state for retry
                print(f"Failed to submit {cid}: {exc}")

    return {"chunk_ids": chunk_ids, "batch_ids": batch_ids, "dry_run": False}


def poll_epoch(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    *,
    interval: int = 30,
    timeout: int = 3600,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Poll all chunks of an epoch until terminal. Download results.

    Respects resume: reuses existing batch IDs and detects already downloaded results.
    """
    run = store.get_run(run_id)
    model = run.get("model", "gpt-5.6-luna") if run else "gpt-5.6-luna"
    client = _get_client(model=model, base_url=base_url)
    chunks = store.pending_batch_chunks(run_id, checkpoint_index)
    if not chunks:
        return {"status": "no_chunks"}

    # Poll loop
    start = time.time()
    # Mark polling
    for ch in chunks:
        if ch["status"] == "submitted":
            store.mark_chunk_polling(ch["chunk_id"])

    while True:
        all_terminal = True
        for ch in chunks:
            # Reload
            cur = store.conn.execute("SELECT * FROM batch_chunks WHERE chunk_id=?", (ch["chunk_id"],))
            row = cur.fetchone()
            if not row:
                continue
            if row["status"] == "validated":
                continue
            batch_id = row["openai_batch_id"]
            if not batch_id:
                all_terminal = False
                continue
            try:
                batch = client.retrieve_batch(batch_id)
                status = getattr(batch, "status", None) or (batch.get("status") if isinstance(batch, dict) else None)
            except Exception as exc:
                print(f"Poll error for {batch_id}: {exc}")
                all_terminal = False
                continue
            if status in ("completed", "failed", "cancelled", "expired"):
                # Download if not already downloaded
                cur2 = store.conn.execute("SELECT result_path FROM batch_chunks WHERE chunk_id=?", (ch["chunk_id"],))
                r = cur2.fetchone()
                if r and r["result_path"] and Path(r["result_path"]).exists():
                    # Already downloaded; ensure import
                    if row["status"] != "validated":
                        try:
                            store.import_batch_results(ch["chunk_id"], r["result_path"])
                        except Exception as e:
                            print(f"Revalidate failed for {ch['chunk_id']}: {e}")
                else:
                    # Download
                    out_path = f"data/executor/results/{run_id}_ckpt{checkpoint_index}_{ch['chunk_id']}_results.jsonl"
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    # Retrieve file id
                    out_file_id = getattr(batch, "output_file_id", None) or (batch.get("output_file_id") if isinstance(batch, dict) else None)
                    if not out_file_id:
                        err_file = getattr(batch, "error_file_id", None) or (batch.get("error_file_id") if isinstance(batch, dict) else None)
                        if err_file:
                            print(f"Batch {batch_id} has error_file {err_file} status {status}")
                        all_terminal = False
                        continue
                    # Download via gateway
                    try:
                        out_path = client.download_batch_output(batch, out_path)
                        store.import_batch_results(ch["chunk_id"], out_path)
                    except Exception as e:
                        print(f"Download failed for {batch_id}: {e}")
                        all_terminal = False
                        continue
            else:
                all_terminal = False

        if all_terminal:
            break
        if time.time() - start > timeout:
            return {"status": "timeout", "elapsed": time.time() - start}
        time.sleep(interval)
        # Refresh chunks
        chunks = store.pending_batch_chunks(run_id, checkpoint_index)

    return {"status": "completed", "chunks": store.pending_batch_chunks(run_id, checkpoint_index)}


def resume_run(
    store: RunStore,
    run_id: str,
    *,
    base_url: Optional[str] = None,
) -> None:
    """Resume a run from SQLite without duplicate submissions."""
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"run not found: {run_id}")
    cur = store.conn.execute("SELECT MAX(checkpoint_index) as mx FROM checkpoints WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    max_idx = int(row["mx"]) if row and row["mx"] is not None else 0
    for idx in range(1, max_idx + 1):
        if store.is_epoch_validated(run_id, idx):
            # Ensure advance
            try:
                store.advance_checkpoint(run_id)
            except Exception:
                pass
            continue
        # Need to handle this epoch: if not prepared, prepare; if prepared but not submitted, submit; if submitted, poll
        chunks = store.pending_batch_chunks(run_id, idx)
        if not chunks:
            # Prepare if pending checkpoints exist but no chunks
            cps = store.list_checkpoints(run_id, checkpoint_index=idx)
            pending_cps = [c for c in cps if c["status"] == "pending"]
            if pending_cps:
                store.create_batch_chunks(run_id=run_id, checkpoint_index=idx)
                chunks = store.pending_batch_chunks(run_id, idx)
        # If chunks still none, no pending for this epoch => skip
        if not chunks:
            continue
        # Check if any chunk missing batch_id => submit
        missing = [c for c in chunks if not c["openai_batch_id"]]
        if missing:
            # Submit missing
            submit_epoch(store, run_id, idx, base_url=base_url)
        else:
            # All have batch ids but not validated => poll
            not_validated = [c for c in chunks if c["status"] != "validated"]
            if not_validated:
                poll_epoch(store, run_id, idx, base_url=base_url)
        # After handling, try advance
        if store.is_epoch_validated(run_id, idx):
            store.advance_checkpoint(run_id)
        else:
            # Stop; do not advance epoch until validated (spec)
            break


class RunExecutor:
    """Resident runner for an executor run.

    Wraps the functional epoch primitives with an object that drives
    itself. Suitable for pipeline-agnostic orchestration (SFT batch,
    future RL inline).
    """

    def __init__(self, store: RunStore, run_id: str, base_url: Optional[str] = None) -> None:
        self.store = store
        self.run_id = run_id
        self.base_url = base_url

    # step methods retained for tests / granular control
    def prepare(self, checkpoint_index: int, **kwargs: Any) -> List[str]:
        return prepare_epoch(self.store, self.run_id, checkpoint_index, **kwargs)

    def submit(self, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        return submit_epoch(self.store, self.run_id, checkpoint_index, base_url=self.base_url, **kwargs)

    def poll(self, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        return poll_epoch(self.store, self.run_id, checkpoint_index, base_url=self.base_url, **kwargs)

    def resume(self) -> None:
        return resume_run(self.store, self.run_id, base_url=self.base_url)

    def run(self) -> None:
        """Drive the run to completion (poll/submit/advance loop)."""
        return self.resume()

    def status(self) -> Dict[str, Any]:
        run = self.store.get_run(self.run_id)
        if not run:
            return {"run_id": self.run_id, "status": "not_found"}
        cps = self.store.list_checkpoints(self.run_id)
        from collections import Counter

        counter = Counter(c["status"] for c in cps)
        chunks = self.store.conn.execute(
            "SELECT * FROM batch_chunks WHERE run_id=? ORDER BY checkpoint_index, chunk_index", (self.run_id,)
        ).fetchall()
        return {
            "run": run,
            "checkpoint_counts": dict(counter),
            "total_checkpoints": len(cps),
            "chunks": [dict(ch) for ch in chunks],
        }

    def cancel(self) -> None:
        self.store.update_run_status(self.run_id, "cancelled")

    def requeue(self, *, only_rejected: bool = False) -> int:
        """Requeue rejected/skipped steps per retry policy.

        Default executor policy is record failures; this method is explicit.
        Returns number of steps requeued.
        """
        # Default: requeue rejected outputs as pending for retry
        status_filter = ("rejected",) if only_rejected else ("rejected", "skipped")
        placeholders = ",".join("?" for _ in status_filter)
        cur = self.store.conn.execute(
            f"SELECT checkpoint_id FROM checkpoints WHERE run_id=? AND status IN ({placeholders})",
            (self.run_id, *status_filter),
        )
        ids = [r["checkpoint_id"] for r in cur.fetchall()]
        if not ids:
            return 0
        with self.store.conn:
            for cid in ids:
                # Only requeue if no accepted output exists; otherwise keep skipped lineage
                if only_rejected:
                    self.store.conn.execute("UPDATE checkpoints SET status=?, skip_reason=NULL WHERE checkpoint_id=?", ("pending", cid))
                else:
                    # For generic requeue, reset skipped with broken lineage to pending only if caller explicitly wants it
                    self.store.conn.execute("UPDATE checkpoints SET status=?, skip_reason=NULL WHERE checkpoint_id=? AND skip_reason='broken_strategy_lineage'", ("pending", cid))
        # Count how many now pending that were touched
        return len(ids)
