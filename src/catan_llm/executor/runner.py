"""Durable orchestration for generic executor — batch + inline.

Thin facade over Transport implementations. Keeps legacy batch functions
for tests (prepare_epoch/submit_epoch/poll_epoch) while resume/run
now dispatch via transport="batch"|"inline" in runs.config_json.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from catan_llm.executor.metrics import observe_checkpoints
from catan_llm.executor.store import RunStore


def _get_client(model: str, base_url: Optional[str] = None):
    """Legacy shim for tests patching runner._get_client."""
    try:
        from catan_llm.llm.teacher import TeacherGateway

        return TeacherGateway(model=model, base_url=base_url)
    except Exception as exc:
        raise RuntimeError(f"OpenAI client unavailable: {exc}") from exc


def _resolve_transport(store: RunStore, run_id: str, base_url: Optional[str] = None, override: Optional[str] = None):
    """Pick Batch vs Inline transport from run config or override."""
    transport_name = override
    if not transport_name:
        run = store.get_run(run_id)
        if run:
            # Check explicit transport column via config_json or direct field
            try:
                cfg = json.loads(run.get("config_json") or "{}")
                transport_name = cfg.get("transport") or run.get("transport") or "batch"
            except Exception:
                transport_name = run.get("transport") or "batch"
        else:
            transport_name = "batch"
    # Normalize
    transport_name = (transport_name or "batch").lower()
    if transport_name == "inline":
        from catan_llm.executor.transport.inline import InlineTransport

        return InlineTransport(base_url=base_url)
    else:
        from catan_llm.executor.transport.batch import BatchTransport

        return BatchTransport(base_url=base_url)


# -- Legacy batch helpers (preserved for tests) --


def prepare_epoch(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    *,
    chunk_size: int = 100,
    request_dir: str = "data/executor/batches",
) -> List[str]:
    """Create batch JSONL chunks for an epoch (offline, no API). Idempotent. Legacy batch path."""
    from catan_llm.executor.transport.batch import BatchTransport

    t = BatchTransport(chunk_size=chunk_size, request_dir=request_dir)
    return t.prepare(store, run_id, checkpoint_index, chunk_size=chunk_size, request_dir=request_dir)


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
    """Submit all chunks for an epoch via Batch transport. Legacy entry."""
    from catan_llm.executor.transport.batch import BatchTransport

    t = BatchTransport(base_url=base_url, chunk_size=chunk_size, request_dir=request_dir, max_workers=max_workers)
    return t.submit(store, run_id, checkpoint_index, dry_run=dry_run, chunk_size=chunk_size, request_dir=request_dir, base_url=base_url, max_workers=max_workers)


def poll_epoch(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    *,
    interval: int = 30,
    timeout: int = 3600,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Poll batch epoch until terminal. Legacy batch path."""
    from catan_llm.executor.transport.batch import BatchTransport

    t = BatchTransport(base_url=base_url)
    return t.poll(store, run_id, checkpoint_index, interval=interval, timeout=timeout, base_url=base_url)


# -- New inline helpers --


def submit_inline(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    *,
    base_url: Optional[str] = None,
    concurrency: int = 8,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Submit inline (direct parallel) for an epoch."""
    from catan_llm.executor.transport.inline import InlineTransport

    t = InlineTransport(base_url=base_url, concurrency=concurrency, max_retries=max_retries, max_tokens=max_tokens)
    return t.submit(store, run_id, checkpoint_index, concurrency=concurrency, max_retries=max_retries, base_url=base_url, max_tokens=max_tokens)


def poll_inline(
    store: RunStore,
    run_id: str,
    checkpoint_index: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    from catan_llm.executor.transport.inline import InlineTransport

    t = InlineTransport(base_url=kwargs.get("base_url"))
    return t.poll(store, run_id, checkpoint_index, **kwargs)


def resume_run(
    store: RunStore,
    run_id: str,
    *,
    base_url: Optional[str] = None,
    transport: Optional[str] = None,
    concurrency: int = 8,
    max_retries: int = 3,
) -> None:
    """Resume a run from SQLite — dispatches to batch or inline transport."""
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"run not found: {run_id}")
    # Resolve transport
    resolved = _resolve_transport(store, run_id, base_url=base_url, override=transport)
    is_inline = resolved.__class__.__name__ == "InlineTransport"

    cur = store.conn.execute("SELECT MAX(checkpoint_index) as mx FROM checkpoints WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    max_idx = int(row["mx"]) if row and row["mx"] is not None else 0
    # initial checkpoint gauge
    try:
        observe_checkpoints(run_id, store.list_checkpoints(run_id))
    except Exception:
        pass
    for idx in range(1, max_idx + 1):
        if store.is_epoch_validated(run_id, idx):
            try:
                store.advance_checkpoint(run_id)
            except Exception:
                pass
            try:
                observe_checkpoints(run_id, store.list_checkpoints(run_id))
            except Exception:
                pass
            continue

        if is_inline:
            # Inline: direct submit (synchronous) then advance if validated
            # Use submit_inline which handles pending filtering + retries
            submit_inline(store, run_id, idx, base_url=base_url, concurrency=concurrency, max_retries=max_retries)
            # No separate poll needed; poll reports status
        else:
            # Batch: original flow
            chunks = store.pending_batch_chunks(run_id, idx)
            if not chunks:
                cps = store.list_checkpoints(run_id, checkpoint_index=idx)
                pending_cps = [c for c in cps if c["status"] == "pending"]
                if pending_cps:
                    store.create_batch_chunks(run_id=run_id, checkpoint_index=idx)
                    chunks = store.pending_batch_chunks(run_id, idx)
            if not chunks:
                continue
            missing = [c for c in chunks if not c["openai_batch_id"]]
            if missing:
                submit_epoch(store, run_id, idx, base_url=base_url)
            else:
                not_validated = [c for c in chunks if c["status"] != "validated"]
                if not_validated:
                    poll_epoch(store, run_id, idx, base_url=base_url)

        if store.is_epoch_validated(run_id, idx):
            store.advance_checkpoint(run_id)
        else:
            try:
                observe_checkpoints(run_id, store.list_checkpoints(run_id))
            except Exception:
                pass
            break
        try:
            observe_checkpoints(run_id, store.list_checkpoints(run_id))
        except Exception:
            pass


class RunExecutor:
    """Resident runner — transport-aware."""

    def __init__(self, store: RunStore, run_id: str, base_url: Optional[str] = None, transport: Optional[str] = None) -> None:
        self.store = store
        self.run_id = run_id
        self.base_url = base_url
        self.transport_name = transport  # explicit override or None → resolve from run config

    def _transport(self):
        return _resolve_transport(self.store, self.run_id, base_url=self.base_url, override=self.transport_name)

    def prepare(self, checkpoint_index: int, **kwargs: Any) -> List[str]:
        t = self._transport()
        return t.prepare(self.store, self.run_id, checkpoint_index, **kwargs)

    def submit(self, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        t = self._transport()
        # BatchTransport expects batch kwargs; InlineTransport expects concurrency etc.
        # Pass through; each transport ignores unknown kwargs
        if t.__class__.__name__ == "InlineTransport":
            return t.submit(self.store, self.run_id, checkpoint_index, base_url=self.base_url, **kwargs)
        return t.submit(self.store, self.run_id, checkpoint_index, base_url=self.base_url, **kwargs)

    def poll(self, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        t = self._transport()
        return t.poll(self.store, self.run_id, checkpoint_index, base_url=self.base_url, **kwargs)

    def resume(self, **kwargs: Any) -> None:
        # Allow override transport via kwargs
        transport = kwargs.get("transport", self.transport_name)
        concurrency = kwargs.get("concurrency", 8)
        max_retries = kwargs.get("max_retries", 3)
        return resume_run(self.store, self.run_id, base_url=self.base_url, transport=transport, concurrency=concurrency, max_retries=max_retries)

    def run(self) -> None:
        """Drive the run to completion."""
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
                if only_rejected:
                    self.store.conn.execute("UPDATE checkpoints SET status=?, skip_reason=NULL WHERE checkpoint_id=?", ("pending", cid))
                else:
                    self.store.conn.execute("UPDATE checkpoints SET status=?, skip_reason=NULL WHERE checkpoint_id=? AND skip_reason='broken_strategy_lineage'", ("pending", cid))
        return len(ids)
