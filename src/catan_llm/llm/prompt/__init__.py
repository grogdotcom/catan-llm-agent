"""Canonical prompt construction.

``PromptBuilder.build`` is the single entry point for rendering a decision
prompt from a ``PromptContext``. Strategy is an input to construction, never a
post-render string edit. ``strategy`` holds the deterministic block utilities
used only when legacy rendered prompts (which lack a ``PromptContext``) must be
normalised.
"""

from __future__ import annotations

from catan_llm.llm.prompt.builder import PromptBuilder
from catan_llm.llm.prompt.strategy import (
    CURRENT_STRATEGY_MARKER,
    ensure_strategy_block,
    inject_strategy_into_prompt,
    normalize_strategy_block,
)

__all__ = [
    "PromptBuilder",
    "CURRENT_STRATEGY_MARKER",
    "ensure_strategy_block",
    "inject_strategy_into_prompt",
    "normalize_strategy_block",
]
