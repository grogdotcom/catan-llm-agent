"""Parsing of teacher responses (``<think><strategy><action>``).

Shared by generic evaluation, the two-phase placement pipeline, and SFT
checkpoint validation so all pipelines parse the teacher contract identically.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from catan_llm.domain import TeacherResponse

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_STRATEGY_RE = re.compile(r"<strategy>(.*?)</strategy>", re.S | re.I)
_ACTION_RE = re.compile(r"<action>\s*(\d+)\s*</action>", re.S | re.I)


def extract_block(text: str, pattern: re.Pattern) -> Optional[str]:
    """Return the inner content of the first match, or None if empty/missing."""
    if not text:
        return None
    m = pattern.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None


def extract_think(text: str) -> Optional[str]:
    return extract_block(text or "", _THINK_RE)


def extract_strategy(text: str) -> Optional[str]:
    return extract_block(text or "", _STRATEGY_RE)


def extract_action(text: str) -> Optional[int]:
    m = _ACTION_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_teacher_response(text: str) -> TeacherResponse:
    """Parse a raw teacher response into a structured ``TeacherResponse``."""
    text = text or ""
    return TeacherResponse(
        think_text=extract_think(text),
        strategy=extract_strategy(text),
        action=extract_action(text),
        raw_text=text,
    )


def extract_text_from_batch_body(body: Dict[str, Any]) -> str:
    """Extract assistant text from a Batch output ``response.body``.

    Handles chat (``choices[].message.content`` / ``reasoning_content``) and
    responses (``output_text`` / ``output[].type==message``) shapes.
    """
    if not isinstance(body, dict):
        return ""
    try:
        if body.get("output_text"):
            return body["output_text"]
        if body.get("output"):
            for item in body.get("output", []):
                if item.get("type") == "message":
                    c = item.get("content", [])
                    if c and c[0].get("text"):
                        return c[0].get("text")
        if body.get("choices"):
            msg = body["choices"][0].get("message", {})
            if msg.get("content"):
                return msg["content"]
            if msg.get("reasoning_content"):
                return msg["reasoning_content"]
    except Exception:
        pass
    return ""
