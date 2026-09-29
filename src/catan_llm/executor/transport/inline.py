"""Inline transport — direct parallel TeacherGateway calls (opencode/muse-spark)."""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional


def _get_client(model: str, base_url: Optional[str] = None):
    from catan_llm.llm.teacher import TeacherGateway

    return TeacherGateway(model=model, base_url=base_url)


class InlineTransport:
    """Direct parallel execution without Batch API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        concurrency: int = 8,
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
    ):
        self.base_url = base_url
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def prepare(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> List[str]:
        # No files for inline; return pending checkpoint ids for introspection
        cps = store.list_checkpoints(run_id, checkpoint_index=checkpoint_index)
        pending = [c["checkpoint_id"] for c in cps if c["status"] == "pending"]
        return pending

    def submit(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        concurrency = kwargs.get("concurrency", self.concurrency)
        max_retries = kwargs.get("max_retries", self.max_retries)
        base_url = kwargs.get("base_url", self.base_url)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        cps = store.list_checkpoints(run_id, checkpoint_index=checkpoint_index)
        pending = [c for c in cps if c["status"] == "pending"]
        if not pending:
            return {"checkpoint_ids": [], "submitted": 0, "message": "no pending checkpoints for epoch"}

        run = store.get_run(run_id)
        model = run.get("model", "muse-spark-1.2-contributor") if run else "muse-spark-1.2-contributor"
        from catan_llm.llm.teacher.models import spec_for

        spec = spec_for(model, base_url)
        # Resolve max_tokens via spec if not overridden
        resolved_max_tokens = max_tokens or spec.default_max_tokens

        client = _get_client(model=model, base_url=base_url)

        # Reuse store helpers for prompt rendering; gateway handles system_prompt internally
        from catan_llm.llm.teacher.models import DEFAULT_SYSTEM_PROMPT, normalize_base_url

        # For resume: skip already validated outputs (write_inline_output handles upsert)
        # Build work items
        # Pre-render call records sequentially to avoid DB concurrency
        # Gateway requires {prompt, completion} — build from source_record_json + rendered prompt
        call_recs: Dict[str, Dict[str, Any]] = {}
        for cp in pending:
            try:
                prompt = store.get_prompt(cp["checkpoint_id"])
                # Fetch source record for completion etc.
                cur = store.conn.execute("SELECT source_record_json, engine_completion FROM decision_opportunities WHERE opportunity_id=?", (cp["opportunity_id"],))
                row = cur.fetchone()
                src = {}
                if row and row["source_record_json"]:
                    try:
                        src = json.loads(row["source_record_json"])
                    except Exception:
                        src = {}
                # Base record is source json (has completion, chosen_index, etc.)
                rec = dict(src)
                rec["prompt"] = prompt
                # Ensure completion field (gateway validates)
                if "completion" not in rec or not rec["completion"]:
                    rec["completion"] = row["engine_completion"] if row else src.get("engine_completion") or src.get("completion") or ""
                # Also ensure engine_completion for fallback
                if "engine_completion" not in rec:
                    rec["engine_completion"] = rec["completion"]
                call_recs[cp["checkpoint_id"]] = rec
            except Exception as e:
                call_recs[cp["checkpoint_id"]] = {"prompt": "", "completion": "1"}
        accepted = 0
        rejected = 0
        errors: List[str] = []
        lock = threading.Lock()

        def _one(cp: Dict[str, Any]) -> None:
            nonlocal accepted, rejected
            cid = cp["checkpoint_id"]
            call_rec = call_recs.get(cid, {"prompt": "", "completion": "1"})
            last_err = None
            response_text = ""
            t0 = time.time()
            for attempt in range(1, max_retries + 1):
                try:
                    raw = client.chat_completion(
                        call_rec,
                        model=client.model if hasattr(client, "model") else model,
                        system_prompt=DEFAULT_SYSTEM_PROMPT,
                        temperature=0.0 if not spec.supports_temperature else 0.0,
                        max_tokens=resolved_max_tokens,
                    )
                    from catan_llm.llm.teacher.models import _extract_response_text

                    response_text = _extract_response_text(raw) or ""
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    is_retriable = "429" in str(e) or "rate" in str(e).lower() or "timeout" in str(e).lower() or "500" in str(e) or "502" in str(e) or "503" in str(e)
                    if attempt < max_retries and is_retriable:
                        backoff = (2 ** (attempt - 1)) + random.random()
                        time.sleep(backoff)
                        continue
                    else:
                        break
            duration = time.time() - t0
            # Persist with lock (DB write)
            try:
                with lock:
                    res = store.write_inline_output(cid, response_text or "")
                    # metrics
                    try:
                        from catan_llm.executor.metrics import observe_request

                        status = "validated" if res["accepted"] else "rejected"
                        reason = None if res["accepted"] else res.get("rejection_reason")
                        observe_request(transport="inline", model=model, epoch=cp.get("checkpoint_index", 0), status=status, duration=duration, output_bytes=len(response_text or ""), rejection_reason=reason)
                    except Exception:
                        pass
                    if res["accepted"]:
                        accepted += 1
                    else:
                        rejected += 1
                    if last_err and not response_text:
                        errors.append(f"{cid}: {last_err}")
            except Exception as e:
                with lock:
                    rejected += 1
                    errors.append(f"{cid}: write failed: {e}")

        # Execute in pool — incremental writes
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_one, cp): cp["checkpoint_id"] for cp in pending}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    with lock:
                        errors.append(str(e))

        return {
            "checkpoint_ids": [c["checkpoint_id"] for c in pending],
            "accepted": accepted,
            "rejected": rejected,
            "errors": errors,
            "concurrency": concurrency,
            "model": model,
        }

    def poll(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        # Inline is synchronous; poll is a status check
        cps = store.list_checkpoints(run_id, checkpoint_index=checkpoint_index)
        if not cps:
            return {"status": "no_checkpoints"}
        pending = [c for c in cps if c["status"] == "pending"]
        if pending:
            return {"status": "pending", "pending": len(pending)}
        # All terminal (validated/rejected/skipped) → completed
        validated = sum(1 for c in cps if c["status"] == "validated")
        rejected = sum(1 for c in cps if c["status"] == "rejected")
        skipped = sum(1 for c in cps if c["status"] == "skipped")
        return {"status": "completed", "validated": validated, "rejected": rejected, "skipped": skipped}
