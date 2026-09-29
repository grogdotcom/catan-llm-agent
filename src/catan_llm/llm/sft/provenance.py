"""Trajectory provenance helpers.

Every midgame decision record must be joinable to immutable trajectory metadata.
See docs/sft_pipeline.md §5.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

SEAT_COLORS = ["RED", "BLUE", "ORANGE", "WHITE"]


def derive_trajectory_id(game_seed: int, seat_order: List[str]) -> str:
    """Stable trajectory_id from seed + seat order.

    Example: ``seed-123456-seat-order-hash``. We use a short hash of the
    seat order so the id is stable and filesystem-safe.
    """
    seat_str = "-".join(seat_order)
    h = hashlib.sha256(f"{game_seed}-{seat_str}".encode()).hexdigest()[:12]
    return f"seed-{game_seed}-{h}"


def seat_index_for_color(color: str, seat_order: List[str]) -> int:
    """Return seat index for a color name within the seat order."""
    try:
        return seat_order.index(color)
    except ValueError as exc:
        raise ValueError(f"color {color!r} not in seat_order {seat_order!r}") from exc


def build_trajectory_manifest(
    *,
    trajectory_id: str,
    game_seed: int,
    game_id: int,
    game_end_turn: int,
    seat_order: List[str],
    winner: str,
    trajectory_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the manifest entry described in §5."""
    manifest: Dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "game_seed": game_seed,
        "game_end_turn": game_end_turn,
        "game_id": game_id,
        "seat_order": seat_order,
        "winner": winner,
        "players": [
            {"seat_index": idx, "color": color, "is_winner": color == winner}
            for idx, color in enumerate(seat_order)
        ],
    }
    if trajectory_index is not None:
        manifest["trajectory_index"] = trajectory_index
    return manifest


def annotate_record_with_provenance(
    record: Dict[str, Any],
    *,
    trajectory_id: str,
    game_seed: int,
    game_end_turn: int,
    seat_index: int,
    trajectory_index: int,
) -> Dict[str, Any]:
    """Return a copy of *record* annotated with provenance fields.

    Required scope §1:
      trajectory_id, game_seed, game_end_turn, seat_index, color,
      winner, trajectory_index
    ``color`` and ``winner`` are already present in existing records; this
    helper adds the missing immutable join keys without overwriting them.
    """
    out = dict(record)
    out["trajectory_id"] = trajectory_id
    out["game_seed"] = game_seed
    out["game_end_turn"] = game_end_turn
    out["seat_index"] = seat_index
    out["trajectory_index"] = trajectory_index
    # Ensure game_id exists (local display/index)
    if "game_id" not in out:
        out["game_id"] = -1
    if "color" not in out:
        raise ValueError("record missing 'color' required for provenance")
    if "winner" not in out:
        out["winner"] = None
    return out


def sample_winning_trajectories(
    trajectories: List[Dict[str, Any]],
    *,
    sample_size: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample winning trajectories with fixed-seed seat stratification.

    Targets ~ equal per seat_index (e.g. 125 per seat for 4 seats). Winners
    only; caller should pre-filter to winner trajectories.
    Uses ``random.Random(seed)`` for reproducibility.
    """
    if sample_size <= 0:
        return []
    # Group by seat_index of winner
    from collections import defaultdict

    by_seat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for t in trajectories:
        # winner seat_index: look up via seat_order + winner, or explicit seat_index / winner_seat_index
        winner = t.get("winner")
        seat_order = t.get("seat_order") or t.get("seat_order_json")
        if isinstance(seat_order, str):
            try:
                seat_order = json.loads(seat_order)
            except Exception:
                seat_order = []
        seat_idx = t.get("winner_seat_index")
        if seat_idx is None and winner and seat_order:
            try:
                seat_idx = seat_order.index(winner)
            except ValueError:
                seat_idx = 0
        if seat_idx is None:
            seat_idx = t.get("seat_index", 0)
        by_seat[int(seat_idx)].append(t)

    rng = random.Random(seed)
    # Determine per-seat target
    seats = sorted(by_seat.keys())
    if not seats:
        return []
    per_seat = sample_size // len(seats)
    remainder = sample_size % len(seats)
    sampled: List[Dict[str, Any]] = []
    for i, s in enumerate(seats):
        target = per_seat + (1 if i < remainder else 0)
        pool = list(by_seat[s])
        rng.shuffle(pool)
        take = min(target, len(pool))
        sampled.extend(pool[:take])

    # If still short (some seats had fewer than target), fill from remaining
    if len(sampled) < sample_size:
        remaining = [t for t in trajectories if t not in sampled]
        rng.shuffle(remaining)
        needed = sample_size - len(sampled)
        sampled.extend(remaining[:needed])

    rng.shuffle(sampled)
    return sampled[:sample_size]


def ensure_game_end_turn(record: Dict[str, Any], game_end_turn: Optional[int] = None) -> int:
    """Resolve game_end_turn for a record, falling back to record field."""
    if game_end_turn is not None:
        return int(game_end_turn)
    if "game_end_turn" in record and record["game_end_turn"] is not None:
        return int(record["game_end_turn"])
    # Fallback: infer from turn if possible (not ideal)
    raise ValueError("game_end_turn not available for record")
