"""Run reporting — acceptance and rejection analytics from validated outputs.

``outputs`` are plain dicts with ``validation_status`` and ``rejection_reason``
(as persisted by ``MidgameRunStore.import_batch_results``). All functions are
pure and database-free.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


def acceptance(outputs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Counts of accepted / rejected outputs."""
    counts = {"accepted": 0, "rejected": 0, "total": 0}
    for o in outputs:
        ok = o.get("validation_status") == "accepted"
        counts["accepted" if ok else "rejected"] += 1
        counts["total"] += 1
    return counts


def rejection_reasons(outputs: List[Dict[str, Any]]) -> Counter:
    """Distribution of rejection reasons (accepted rows contribute nothing)."""
    return Counter(
        o.get("rejection_reason", "unknown")
        for o in outputs
        if o.get("validation_status") != "accepted"
    )


def acceptance_rate(outputs: List[Dict[str, Any]]) -> float:
    counts = acceptance(outputs)
    return counts["accepted"] / counts["total"] if counts["total"] else 0.0


def coverage_by(outputs: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    """Count accepted outputs grouped by a metadata key (phase, seat_index...)."""
    counts: Dict[str, int] = defaultdict(int)
    for o in outputs:
        if o.get("validation_status") != "accepted":
            continue
        k = o.get(key)
        counts[str(k) if k is not None else "(none)"] += 1
    return dict(counts)


def run_report(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Consolidated report for a run's validated outputs."""
    return {
        "acceptance": acceptance(outputs),
        "acceptance_rate": acceptance_rate(outputs),
        "rejection_reasons": dict(rejection_reasons(outputs)),
        "by_phase": coverage_by(outputs, "phase"),
        "by_column": coverage_by(outputs, "color"),
        "by_seat": coverage_by(outputs, "seat_index"),
        "by_checkpoint": coverage_by(outputs, "checkpoint_index"),
    }
