"""Validation gates for teacher responses. Spec §14."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

REJECTION_REASONS = [
    "missing_think",
    "missing_strategy",
    "missing_action",
    "invalid_action_range",
    "wrong_alpha_beta_action",
    "duplicate_strategy_block",
    "broken_strategy_lineage",
    "api_error",
]


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    rejection_reason: Optional[str] = None
    think_text: Optional[str] = None
    strategy_out: Optional[str] = None
    predicted_action: Optional[int] = None


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_STRATEGY_RE = re.compile(r"<strategy>(.*?)</strategy>", re.S | re.I)
_ACTION_RE = re.compile(r"<action>\s*(\d+)\s*</action>", re.S | re.I)
_CURRENT_STRAT_BLOCK = "[CURRENT STRATEGY]"


def _extract_block(text: str, pattern: re.Pattern) -> Optional[str]:
    if not text:
        return None
    m = pattern.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    return inner if inner else None


def count_strategy_blocks(prompt: str) -> int:
    """Count occurrences of [CURRENT STRATEGY] in prompt."""
    if not prompt:
        return 0
    return prompt.count(_CURRENT_STRAT_BLOCK)


def validate_response(
    *,
    prompt: str,
    response_text: str,
    engine_completion: Any,
    num_moves: Optional[int] = None,
    strategy_in: Optional[str] = None,
    strategy_lineage_ok: bool = True,
) -> ValidationResult:
    """Apply gates 1-6 plus lineage gate.

    Gates:
    1. non-empty <think>
    2. non-empty <strategy>
    3. parseable <action> integer
    4. action within 1..num_moves
    5. action equals engine_completion
    6. exactly one [CURRENT STRATEGY] block
    7-8. lineage ok (caller-provided)
    """
    # Gate 6: exactly one strategy block in prompt (do first as prompt error)
    blocks = count_strategy_blocks(prompt or "")
    if blocks != 1:
        return ValidationResult(accepted=False, rejection_reason="duplicate_strategy_block")

    if not strategy_lineage_ok:
        return ValidationResult(accepted=False, rejection_reason="broken_strategy_lineage")

    if response_text is None or "error" in (response_text or "").lower() and "<think>" not in (response_text or ""):
        # Let other gates catch api_error if text missing think etc.
        pass

    think = _extract_block(response_text or "", _THINK_RE)
    if not think:
        return ValidationResult(accepted=False, rejection_reason="missing_think", predicted_action=None)

    strategy = _extract_block(response_text or "", _STRATEGY_RE)
    if not strategy:
        return ValidationResult(accepted=False, rejection_reason="missing_strategy", think_text=think)

    m = _ACTION_RE.search(response_text or "")
    if not m:
        return ValidationResult(accepted=False, rejection_reason="missing_action", think_text=think, strategy_out=strategy)
    try:
        pred = int(m.group(1))
    except Exception:
        return ValidationResult(accepted=False, rejection_reason="missing_action", think_text=think, strategy_out=strategy)

    # Gate 4: range
    if num_moves is not None:
        try:
            nm = int(num_moves)
            if not (1 <= pred <= nm):
                return ValidationResult(
                    accepted=False,
                    rejection_reason="invalid_action_range",
                    think_text=think,
                    strategy_out=strategy,
                    predicted_action=pred,
                )
        except Exception:
            pass

    # Gate 5: equals engine completion
    try:
        expected = int(str(engine_completion).strip())
    except Exception:
        expected = None
    if expected is not None and pred != expected:
        return ValidationResult(
            accepted=False,
            rejection_reason="wrong_alpha_beta_action",
            think_text=think,
            strategy_out=strategy,
            predicted_action=pred,
        )

    return ValidationResult(
        accepted=True,
        rejection_reason=None,
        think_text=think,
        strategy_out=strategy,
        predicted_action=pred,
    )


def validate_batch_response_obj(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Detect API-level errors in batch result line.

    Batch line shape: {custom_id, response:{status_code, body:{...}}}
    """
    try:
        resp = obj.get("response") or {}
        status = resp.get("status_code")
        if status is not None and int(status) != 200:
            return False, "api_error"
        body = resp.get("body") or {}
        if body.get("error"):
            return False, "api_error"
    except Exception:
        pass
    return True, None
