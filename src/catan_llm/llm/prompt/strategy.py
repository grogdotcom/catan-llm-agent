"""Deterministic [CURRENT STRATEGY] block handling.

The canonical pipeline passes ``current_strategy`` into ``PromptBuilder`` so the
block is rendered correctly at construction time. These helpers exist purely for
*legacy* corpus records that were rendered without a ``PromptContext`` (the
corpus carries only the final string): they normalise the well-defined marker
form deterministically, with no regexes over arbitrary prose.
"""

from __future__ import annotations

CURRENT_STRATEGY_MARKER = "[CURRENT STRATEGY]"
NONE_VALUE = "None"
_RECENT_TURNS = "[RECENT TURNS"


def _strategy_value(strategy: str) -> str:
    s = strategy.strip()
    return s if s else NONE_VALUE


def ensure_strategy_block(prompt: str, strategy: Optional[str] = None) -> str:
    """Return a prompt that has exactly one ``[CURRENT STRATEGY]`` block.

    When the marker is absent it is inserted immediately before ``[RECENT
    TURNS`` (the canonical position); when present the existing block's value is
    replaced in place. ``None`` renders the literal ``None`` value.
    """
    value = _strategy_value(strategy) if strategy is not None else NONE_VALUE
    if CURRENT_STRATEGY_MARKER not in prompt:
        if _RECENT_TURNS in prompt:
            return prompt.replace(_RECENT_TURNS, f"{CURRENT_STRATEGY_MARKER}\n{value}\n\n{_RECENT_TURNS}", 1)
        # Neither marker present (edge case): prepend the strategy block
        return f"{CURRENT_STRATEGY_MARKER}\n{value}\n\n{prompt}"
    return inject_strategy_into_prompt(prompt, value)


def inject_strategy_into_prompt(prompt: str, strategy: Optional[str]) -> str:
    """Replace the value line of the first ``[CURRENT STRATEGY]`` block.

    Only the first marker occurrence is touched. If the marker is absent the
    prompt is returned unchanged (callers should use ``ensure_strategy_block``
    when insertion is desired).
    """
    if not strategy:
        return prompt
    value = _strategy_value(strategy)
    idx = prompt.find(CURRENT_STRATEGY_MARKER)
    if idx < 0:
        return prompt
    block_start = idx + len(CURRENT_STRATEGY_MARKER)
    rest = prompt[block_start:]
    value_end = rest.find("\n", 0)
    if value_end < 0:
        value_end = len(rest)
    # Skip a leading blank line so we always replace the value line itself
    lead = 0
    while lead < len(rest) and rest[lead] == "\n":
        lead += 1
        if rest.find("\n", lead) == -1:
            value_end = len(rest)
            lead = len(rest)
            break
        nxt = rest.find("\n", lead)
        if nxt == -1:
            value_end = len(rest)
        else:
            value_end = nxt
        break
    return prompt[: block_start + lead] + value + prompt[block_start + value_end :]


def normalize_strategy_block(prompt: str, strategy: Optional[str]) -> str:
    """Idempotent normalisation: insert-or-replace, then validate single marker."""
    out = ensure_strategy_block(prompt, strategy)
    if out.count(CURRENT_STRATEGY_MARKER) != 1:  # pragma: no cover - defensive
        raise ValueError("prompt must contain exactly one [CURRENT STRATEGY] marker after normalisation")
    return out
