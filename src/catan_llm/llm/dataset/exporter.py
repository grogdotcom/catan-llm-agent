"""Accepted SFT dataset export — pure record construction.

``build_sft_record`` turns a validated checkpoint output into the canonical SFT
line (full teacher response + engine ground truth + strategy in/out + lineage +
split). The SQLite adapter (``MidgameRunStore.export_dataset``) uses this so the
record shape is defined once and unit-testable without a database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catan_llm.llm.prompt.strategy import ensure_strategy_block


def prompt_with_strategy(prompt: str, strategy_in: Optional[str]) -> str:
    """Render the prompt exactly as sent to the teacher (deterministic)."""
    return ensure_strategy_block(prompt, strategy_in or "None")


def build_sft_record(
    *,
    checkpoint: Dict[str, Any],
    output: Dict[str, Any],
    opportunity: Optional[Dict[str, Any]],
    trajectory: Optional[Dict[str, Any]],
    prompt: str,
    split: str,
    split_seed: int,
    run_id: str,
) -> Dict[str, Any]:
    """Build one exported SFT record from validated entities.

    ``checkpoint`` / ``output`` / ``opportunity`` / ``trajectory`` are plain
    dicts (convert ``sqlite3.Row`` with ``dict()`` first); ``prompt`` is the
    already-rendered prompt (strategy block included, as sent to the teacher).
    """
    src = opportunity or {}
    strategy_in = checkpoint.get("strategy_in") or "None"
    return {
        "prompt": prompt,
        "completion": output.get("raw_output_text") or "",
        "engine_completion": opportunity.get("engine_completion", "") if opportunity else "",
        "chosen_label": src.get("chosen_label"),
        "predicted_action": output.get("predicted_action"),
        "strategy_in": strategy_in,
        "strategy_out": output.get("strategy_out"),
        "strategy_source_checkpoint": checkpoint.get("strategy_source_checkpoint"),
        "trajectory_id": checkpoint.get("trajectory_id"),
        "seat_index": checkpoint.get("seat_index"),
        "color": src.get("color") or (opportunity.get("color") if opportunity else None),
        "turn": src.get("turn") or (opportunity.get("turn") if opportunity else None),
        "game_end_turn": trajectory.get("game_end_turn") if trajectory else None,
        "game_seed": trajectory.get("game_seed") if trajectory else None,
        "game_id": src.get("game_id") if src.get("game_id") is not None else (opportunity.get("game_id") if opportunity else None),
        "trajectory_index": src.get("trajectory_index") if src.get("trajectory_index") is not None else (opportunity.get("trajectory_index") if opportunity else None),
        "checkpoint_index": checkpoint.get("checkpoint_index"),
        "checkpoint_count": checkpoint.get("checkpoint_count"),
        "checkpoint_band": checkpoint.get("band"),
        "normalized_progress": checkpoint.get("normalized_progress"),
        "phase": src.get("phase") or (opportunity.get("phase") if opportunity else None),
        "validation_status": output.get("validation_status"),
        "split": split,
        "split_seed": split_seed,
        "raw_output_text": output.get("raw_output_text"),
        "think_text": output.get("think_text"),
        "winner": trajectory.get("winner") if trajectory else None,
        "opportunity_id": checkpoint.get("opportunity_id"),
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "run_id": run_id,
    }


def split_summary(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count accepted records per split."""
    counts = {"train": 0, "validation": 0, "test": 0, "total": 0}
    for rec in records:
        split = rec.get("split", "train")
        counts[split] = counts.get(split, 0) + 1
        counts["total"] += 1
    return counts
