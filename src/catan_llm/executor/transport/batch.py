"""Batch transport — OpenAI Files + Batches (24h, /v1/responses or /v1/chat/completions)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_client(model: str, base_url: Optional[str] = None):
    # Route through runner._get_client so tests patching runner._get_client work
    try:
        import catan_llm.executor.runner as runner_mod

        # If runner has a (possibly patched) _get_client, use it
        if hasattr(runner_mod, "_get_client"):
            return runner_mod._get_client(model=model, base_url=base_url)  # type: ignore
    except Exception:
        pass
    from catan_llm.llm.teacher import TeacherGateway

    return TeacherGateway(model=model, base_url=base_url)


class BatchTransport:
    """Batch execution via OpenAI Batch API."""

    def __init__(self, base_url: Optional[str] = None, chunk_size: int = 100, request_dir: str = "data/executor/batches", max_workers: int = 5):
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.request_dir = request_dir
        self.max_workers = max_workers

    def prepare(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> List[str]:
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        request_dir = kwargs.get("request_dir", self.request_dir)
        return store.create_batch_chunks(run_id=run_id, checkpoint_index=checkpoint_index, chunk_size=chunk_size, request_dir=request_dir)

    def submit(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        dry_run = kwargs.get("dry_run", False)
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        request_dir = kwargs.get("request_dir", self.request_dir)
        base_url = kwargs.get("base_url", self.base_url)
        max_workers = kwargs.get("max_workers", self.max_workers)

        chunk_ids = self.prepare(store, run_id, checkpoint_index, chunk_size=chunk_size, request_dir=request_dir)
        if not chunk_ids:
            return {"chunk_ids": [], "batch_ids": [], "dry_run": dry_run, "message": "no pending checkpoints for epoch"}
        if dry_run:
            return {"chunk_ids": chunk_ids, "batch_ids": [], "dry_run": True}
        run = store.get_run(run_id)
        model = run.get("model", "gpt-5.6-luna") if run else "gpt-5.6-luna"
        if not os.environ.get("OPENAI_API_KEY"):
            return {"chunk_ids": chunk_ids, "batch_ids": [], "dry_run": True, "reason": "OPENAI_API_KEY missing"}
        client = _get_client(model=model, base_url=base_url)
        to_submit: List[str] = []
        for cid in chunk_ids:
            cur = store.conn.execute("SELECT * FROM batch_chunks WHERE chunk_id=?", (cid,))
            row = cur.fetchone()
            if row and row["openai_batch_id"]:
                continue
            if row and row["status"] in ("submitted", "polling", "validated"):
                if row["openai_batch_id"]:
                    continue
            to_submit.append(cid)
        if not to_submit:
            return {"chunk_ids": chunk_ids, "batch_ids": [], "already_submitted": True}

        # Resolve endpoint via spec
        from catan_llm.llm.teacher.models import spec_for

        spec = spec_for(model, base_url)
        endpoint = "/v1/responses" if spec.api == "responses" else "/v1/chat/completions"

        batch_ids: List[str] = []

        def _upload_and_create(cid: str) -> str:
            cur = store.conn.execute("SELECT request_path FROM batch_chunks WHERE chunk_id=?", (cid,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"chunk not found: {cid}")
            req_path = row["request_path"]
            file_id = client.upload_file(req_path)
            batch = client.create_batch(file_id, endpoint=endpoint, completion_window="24h", metadata={"run_id": run_id, "checkpoint_index": str(checkpoint_index)})
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
                except Exception as exc:  # keep chunk in prepared state for retry
                    print(f"Failed to submit {cid}: {exc}")
        return {"chunk_ids": chunk_ids, "batch_ids": batch_ids, "dry_run": False}

    def poll(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        interval = kwargs.get("interval", 30)
        timeout = kwargs.get("timeout", 3600)
        base_url = kwargs.get("base_url", self.base_url)
        run = store.get_run(run_id)
        model = run.get("model", "gpt-5.6-luna") if run else "gpt-5.6-luna"
        client = _get_client(model=model, base_url=base_url)
        chunks = store.pending_batch_chunks(run_id, checkpoint_index)
        if not chunks:
            return {"status": "no_chunks"}
        start = time.time()
        for ch in chunks:
            if ch["status"] == "submitted":
                store.mark_chunk_polling(ch["chunk_id"])
        while True:
            all_terminal = True
            for ch in chunks:
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
                    cur2 = store.conn.execute("SELECT result_path FROM batch_chunks WHERE chunk_id=?", (ch["chunk_id"],))
                    r = cur2.fetchone()
                    if r and r["result_path"] and Path(r["result_path"]).exists():
                        if row["status"] != "validated":
                            try:
                                store.import_batch_results(ch["chunk_id"], r["result_path"])
                            except Exception as e:
                                print(f"Revalidate failed for {ch['chunk_id']}: {e}")
                    else:
                        out_path = f"data/executor/results/{run_id}_ckpt{checkpoint_index}_{ch['chunk_id']}_results.jsonl"
                        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                        out_file_id = getattr(batch, "output_file_id", None) or (batch.get("output_file_id") if isinstance(batch, dict) else None)
                        if not out_file_id:
                            err_file = getattr(batch, "error_file_id", None) or (batch.get("error_file_id") if isinstance(batch, dict) else None)
                            if err_file:
                                print(f"Batch {batch_id} has error_file {err_file} status {status}")
                            all_terminal = False
                            continue
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
            chunks = store.pending_batch_chunks(run_id, checkpoint_index)
        return {"status": "completed", "chunks": store.pending_batch_chunks(run_id, checkpoint_index)}
