"""Shared domain records for the Catan LLM pipeline.

These are the canonical vocabulary that every module (game snapshotting,
prompt building, corpus collection, teacher transport, checkpoint policy,
dataset export) depends on. Keeping them in one place lets the pipeline grow
without drifting between ``dict``-shaped records in different modules.

``DecisionRecord`` is intentionally close to the on-disk JSONL schema so it can
round-trip with ``as_dict``/``from_dict`` without a lossy translation layer.
"""

from __future__ import annotations

from catan_llm.domain.records import (
    BatchRequest,
    Checkpoint,
    DecisionRecord,
    PromptArtifact,
    PromptContext,
    TeacherResponse,
    Trajectory,
    ValidationResult,
)

__all__ = [
    "BatchRequest",
    "Checkpoint",
    "DecisionRecord",
    "PromptArtifact",
    "PromptContext",
    "TeacherResponse",
    "Trajectory",
    "ValidationResult",
]
