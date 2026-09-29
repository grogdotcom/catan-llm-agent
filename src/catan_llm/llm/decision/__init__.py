"""Decision surface — the seam between engine actions and LLM-visible moves.

The entire pipeline (runtime agents, corpus collection, prompt building) should
cross this interface instead of reaching into ``format.moves`` internals, so
runtime behavior and corpus ground truth can never disagree on move semantics.
"""

from __future__ import annotations

from catan_llm.llm.decision.execution import MoveExecutor
from catan_llm.llm.decision.surface import DecisionSurface, MoveChoices, PlannedMove

__all__ = ["DecisionSurface", "MoveChoices", "MoveExecutor", "PlannedMove"]
