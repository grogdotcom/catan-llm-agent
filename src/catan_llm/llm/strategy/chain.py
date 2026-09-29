"""Strategy-chain policy — pure lineage resolution for checkpoint selection.

A checkpoint's ``strategy_in`` comes from either the initial phase-2 bootstrap
(first checkpoint of a player's chain) or the accepted output of the previous
checkpoint in the same chain. These decisions are deterministic and side-effect
free, so they live in a pure module testable without a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LineageDecision:
    """What strategy feeds a given checkpoint and whether it should run."""

    strategy_in: Optional[str]
    strategy_source: Optional[str]
    status: str  # "pending" or "skipped"
    skip_reason: Optional[str]

    @property
    def can_run(self) -> bool:
        return self.status == "pending"


def resolve_lineage(
    checkpoint_index: int,
    *,
    phase2_strategy: Optional[str],
    prev_accepted_strategy: Optional[str] = None,
    prev_checkpoint_id: Optional[str] = None,
) -> LineageDecision:
    """Resolve the strategy feeding checkpoint ``checkpoint_index`` (1-based).

    - First checkpoint: seeded from ``phase2_strategy``. Skipped with
      ``missing_bootstrap_strategy`` when absent.
    - Later checkpoints: seeded from the previous checkpoint's accepted
      ``strategy_out``. Skipped with ``broken_strategy_lineage`` when the
      previous checkpoint was not accepted or produced no strategy.
    """
    if checkpoint_index == 1:
        if phase2_strategy:
            return LineageDecision(
                strategy_in=phase2_strategy,
                strategy_source=None,
                status="pending",
                skip_reason=None,
            )
        return LineageDecision(
            strategy_in=None,
            strategy_source=None,
            status="skipped",
            skip_reason="missing_bootstrap_strategy",
        )

    if prev_accepted_strategy:
        return LineageDecision(
            strategy_in=prev_accepted_strategy,
            strategy_source=prev_checkpoint_id,
            status="pending",
            skip_reason=None,
        )
    return LineageDecision(
        strategy_in=None,
        strategy_source=None,
        status="skipped",
        skip_reason="broken_strategy_lineage",
    )
