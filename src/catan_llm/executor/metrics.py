"""Executor runtime metrics — Prometheus with no-op fallback.

Leaf module: no imports from format/sft/executor. Transports and Store call
these helpers; tests can assert via the in-memory fallback registry.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

# Try prometheus_client, else fallback to in-memory no-op that still records for tests
try:
    from prometheus_client import Counter, Gauge, Histogram, REGISTRY  # type: ignore

    _HAS_PROM = True
except Exception:  # pragma: no cover - fallback
    _HAS_PROM = False
    REGISTRY = None  # type: ignore

    class _NoopMetric:
        def __init__(self, *a, **kw):
            self._value = 0
            self._labels: Dict[tuple, Any] = {}

        def labels(self, *a, **kw):
            return self

        def inc(self, amount=1):
            self._value += amount

        def set(self, v):
            self._value = v

        def observe(self, v):
            self._value = v

        def time(self):
            class _Timer:
                def __enter__(self):
                    self.t0 = time.time()
                    return self

                def __exit__(self, *a):
                    pass

            return _Timer()

    Counter = Gauge = Histogram = _NoopMetric  # type: ignore


# -- Metrics (transport-agnostic) --

# Gauges for checkpoint counts per status/epoch — updated via poll
_catan_checkpoints = None
_catan_batch_chunks = None
_catan_run_info = None

# Counters / Histograms for request lifecycle
_catan_requests_total = None
_catan_request_duration = None
_catan_output_bytes = None
_catan_rejections_total = None
_catan_lineage_skipped_total = None


def _get_or_create():
    global _catan_checkpoints, _catan_batch_chunks, _catan_run_info
    global _catan_requests_total, _catan_request_duration, _catan_output_bytes
    global _catan_rejections_total, _catan_lineage_skipped_total
    if _catan_checkpoints is not None:
        return
    # Use try to avoid duplicate registration in tests (REGISTRY already has it)
    def _safe(metric_cls, name, doc, labelnames=None, **kw):
        try:
            if labelnames is not None:
                return metric_cls(name, doc, labelnames, **kw)
            return metric_cls(name, doc, **kw)
        except ValueError:
            # Already registered — fetch from registry
            if REGISTRY is not None:
                for c in REGISTRY.collect():
                    for s in c.samples:
                        if s.name == name or s.name.startswith(name + "_"):
                            # Return a no-op wrapper that still exposes labels()
                            return metric_cls(name, doc, labelnames or [], **kw)
            # fallback
            return metric_cls(name, doc, labelnames or [], **kw)

    _catan_checkpoints = _safe(Gauge, "catan_checkpoints", "Checkpoints per status/epoch", labelnames=["run_id", "epoch", "status"])
    _catan_batch_chunks = _safe(Gauge, "catan_batch_chunks", "Batch chunks per status", labelnames=["run_id", "epoch", "status"])
    _catan_run_info = _safe(Gauge, "catan_run_info", "Run info", labelnames=["run_id", "model", "transport"])
    _catan_requests_total = _safe(Counter, "catan_requests_total", "Requests total", labelnames=["transport", "model", "epoch", "status"])
    _catan_request_duration = _safe(Histogram, "catan_request_duration_seconds", "Request latency", labelnames=["transport", "model"], buckets=(1, 5, 15, 30, 60, 120, 300))
    _catan_output_bytes = _safe(Histogram, "catan_output_bytes", "Output size bytes", labelnames=["transport"], buckets=(200, 500, 1000, 2000, 4000, 8000, 16000))
    _catan_rejections_total = _safe(Counter, "catan_rejections_total", "Rejections by reason", labelnames=["reason"])
    _catan_lineage_skipped_total = _safe(Counter, "catan_lineage_skipped_total", "Lineage skipped", labelnames=["reason"])


_get_or_create()

# In-memory fallback for tests when prometheus_client not installed or to allow assertions
_fallback: Dict[str, Any] = {
    "checkpoints": {},  # (run_id, epoch, status) -> count
    "requests": [],  # list of dicts
    "rejections": {},  # reason -> count
    "output_bytes": [],
    "durations": [],
}


def observe_checkpoints(run_id: str, checkpoints) -> None:
    """Call periodically (e.g. every poll) to update gauge per epoch/status."""
    from collections import Counter as CC

    # checkpoints is list[dict] with checkpoint_index and status
    by_epoch: Dict[int, CC] = {}
    for cp in checkpoints:
        epoch = int(cp.get("checkpoint_index", 0) or 0)
        status = cp.get("status", "unknown")
        by_epoch.setdefault(epoch, CC())[status] += 1
    for epoch, counter in by_epoch.items():
        for status, cnt in counter.items():
            try:
                _catan_checkpoints.labels(run_id=run_id, epoch=str(epoch), status=status).set(cnt)  # type: ignore
            except Exception:
                pass
            _fallback["checkpoints"][(run_id, str(epoch), status)] = cnt
    # Also expose run info
    try:
        _catan_run_info.labels(run_id=run_id, model="unknown", transport="unknown").set(1)  # placeholder
    except Exception:
        pass


def observe_request(transport: str, model: str, epoch: int, status: str, duration: float, output_bytes: int, rejection_reason: Optional[str] = None) -> None:
    try:
        _catan_requests_total.labels(transport=transport, model=model, epoch=str(epoch), status=status).inc()  # type: ignore
        _catan_request_duration.labels(transport=transport, model=model).observe(duration)  # type: ignore
        _catan_output_bytes.labels(transport=transport).observe(output_bytes)  # type: ignore
        if rejection_reason:
            _catan_rejections_total.labels(reason=rejection_reason).inc()  # type: ignore
    except Exception:
        pass
    _fallback["requests"].append({"transport": transport, "model": model, "epoch": epoch, "status": status, "duration": duration, "output_bytes": output_bytes, "rejection_reason": rejection_reason})
    if rejection_reason:
        _fallback["rejections"][rejection_reason] = _fallback["rejections"].get(rejection_reason, 0) + 1
    if output_bytes:
        _fallback["output_bytes"].append(output_bytes)
    if duration:
        _fallback["durations"].append(duration)


def observe_rejection(reason: str) -> None:
    try:
        _catan_rejections_total.labels(reason=reason).inc()  # type: ignore
    except Exception:
        pass
    _fallback["rejections"][reason] = _fallback["rejections"].get(reason, 0) + 1


def observe_lineage_skipped(reason: str = "broken_strategy_lineage") -> None:
    try:
        _catan_lineage_skipped_total.labels(reason=reason).inc()  # type: ignore
    except Exception:
        pass


def observe_batch_chunks(run_id: str, chunks) -> None:
    from collections import Counter as CC

    by: Dict[tuple, int] = {}
    for ch in chunks:
        epoch = str(ch.get("checkpoint_index", "0"))
        status = ch.get("status", "unknown")
        by[(epoch, status)] = by.get((epoch, status), 0) + 1
    for (epoch, status), cnt in by.items():
        try:
            _catan_batch_chunks.labels(run_id=run_id, epoch=epoch, status=status).set(cnt)  # type: ignore
        except Exception:
            pass


def start_metrics_server(port: int = 9090) -> Optional[Any]:
    """Start prometheus_client HTTP server if available. Returns server thread or None."""
    if not _HAS_PROM:
        return None
    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        return port
    except Exception:
        return None


def get_fallback_metrics() -> Dict[str, Any]:
    """For tests — returns in-memory fallback snapshot."""
    return dict(_fallback)


def reset_fallback() -> None:
    _fallback["checkpoints"].clear()
    _fallback["requests"].clear()
    _fallback["rejections"].clear()
    _fallback["output_bytes"].clear()
    _fallback["durations"].clear()
