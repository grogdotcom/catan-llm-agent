"""Offline evaluation metrics — pure aggregation over evaluated responses.

``results`` are plain dicts produced by ``teacher.requests.evaluate_response``
(keyed ``correct``, ``valid``, ``predicted_index``, ``expected_index``, plus
metadata like ``phase``/``color``). Aggregations are side-effect free so they
are unit-testable without a database or live API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


def total(results: List[Dict[str, Any]]) -> int:
    return len(results)


def accuracy(results: List[Dict[str, Any]]) -> float:
    """Fraction of results where ``correct`` is True (0.0 for empty)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct")) / len(results)


def valid_rate(results: List[Dict[str, Any]]) -> float:
    """Fraction of results where ``valid`` is True (0.0 for empty)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("valid")) / len(results)


def count_correct(results: List[Dict[str, Any]]) -> int:
    return sum(1 for r in results if r.get("correct"))


def count_valid(results: List[Dict[str, Any]]) -> int:
    return sum(1 for r in results if r.get("valid"))


def accuracy_by(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Group accuracy by a metadata key (e.g. ``phase``, ``color``)."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        k = r.get(key)
        groups[str(k) if k is not None else "(none)"].append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for k, rs in groups.items():
        out[k] = {
            "total": len(rs),
            "correct": count_correct(rs),
            "valid": count_valid(rs),
            "accuracy": accuracy(rs),
        }
    return out


def predicted_vs_expected(results: List[Dict[str, Any]]) -> Counter:
    """Distribution of (expected, predicted) action pairs."""
    out = Counter()
    for r in results:
        out[(r.get("expected_index"), r.get("predicted_index"))] += 1
    return out


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One-dict summary of a batch of evaluations."""
    return {
        "total": total(results),
        "correct": count_correct(results),
        "valid": count_valid(results),
        "accuracy": accuracy(results),
        "valid_rate": valid_rate(results),
        "by_phase": accuracy_by(results, "phase"),
        "by_color": accuracy_by(results, "color"),
    }
