"""Evaluation metrics and run reporting."""

from __future__ import annotations

from catan_llm.llm.evaluation.metrics import (
    accuracy,
    accuracy_by,
    count_correct,
    count_valid,
    predicted_vs_expected,
    summarize,
    total,
    valid_rate,
)
from catan_llm.llm.evaluation.report import (
    acceptance,
    acceptance_rate,
    coverage_by,
    rejection_reasons,
    run_report,
)

__all__ = [
    "accuracy",
    "accuracy_by",
    "count_correct",
    "count_valid",
    "predicted_vs_expected",
    "summarize",
    "total",
    "valid_rate",
    "acceptance",
    "acceptance_rate",
    "coverage_by",
    "rejection_reasons",
    "run_report",
]
