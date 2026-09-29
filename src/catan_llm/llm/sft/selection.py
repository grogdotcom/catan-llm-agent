"""Checkpoint selection: collapse, K-bands, nearest to midpoint.

Spec §8-§10.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


K_BY_LENGTH_THRESHOLDS = [(70, 4), (84, 5)]  # up to inclusive


def checkpoint_count_for_length(game_end_turn: int, num_opportunities: int) -> int:
    """K = min(K_by_length, N). §9."""
    if num_opportunities == 0:
        return 0
    if game_end_turn <= 70:
        k_by_len = 4
    elif game_end_turn <= 84:
        k_by_len = 5
    else:
        k_by_len = 6
    return min(k_by_len, num_opportunities)


K_BY_LENGTH = checkpoint_count_for_length  # alias


def _value_gap(record: Dict[str, Any]) -> float:
    """Smallest top2 gap; larger means less ambiguous. Prefer small gap."""
    vals = record.get("top_action_values")
    if not vals or len(vals) < 2:
        # Use 0 if missing? Prefer present; missing => larger gap fallback
        return float("inf")
    try:
        v0 = float(vals[0].get("value", 0))
        v1 = float(vals[1].get("value", 0))
        return abs(v0 - v1)
    except Exception:
        return float("inf")


def _is_discard(record: Dict[str, Any]) -> bool:
    return record.get("phase") == "DISCARD"


def _rank_key(record: Dict[str, Any]) -> Tuple[int, float, int, int]:
    """Ranking for collapsed group retention §8.

    1. Prefer strategic over discard (0 = strategic, 1 = discard)
    2. Smallest value gap
    3. Earliest trajectory_index
    4. Earliest decision_id
    """
    discard_priority = 1 if _is_discard(record) else 0
    gap = _value_gap(record)
    traj_idx = int(record.get("trajectory_index", record.get("decision_id", 0)) or 0)
    dec_id = int(record.get("decision_id", 0) or 0)
    return (discard_priority, gap, traj_idx, dec_id)


def collapse_opportunities(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse same (trajectory_id, seat_index, turn, phase) records.

    Returns one retained record per collapsed group using deterministic ranking.
    Non-grouped records are returned as-is. Excludes initial-placement phase
    records? Caller should filter before calling; this function collapses whatever
    is passed but is documented to be called on midgame candidates only.
    """
    # Group by key
    groups: Dict[Tuple[Any, Any, Any, Any], List[Dict[str, Any]]] = {}
    for r in records:
        key = (
            r.get("trajectory_id"),
            r.get("seat_index"),
            r.get("turn"),
            r.get("phase"),
        )
        groups.setdefault(key, []).append(r)

    retained: List[Dict[str, Any]] = []
    for _key, group in groups.items():
        if len(group) == 1:
            retained.append(group[0])
        else:
            # Deterministic ranking
            chosen = min(group, key=_rank_key)
            retained.append(chosen)
    # Sort by progress for determinism: trajectory_index then turn
    retained.sort(key=lambda r: (r.get("trajectory_index", 0), r.get("turn", 0), r.get("decision_id", 0)))
    return retained


def filter_midgame_candidates(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exclude initial-placement phase records (§8)."""
    return [r for r in records if r.get("phase") != "BUILD_INITIAL_SETTLEMENT"]


def select_checkpoints_for_trajectory(
    *,
    collapsed: List[Dict[str, Any]],
    game_end_turn: int,
    trajectory_id: str,
    seat_index: int,
) -> List[Dict[str, Any]]:
    """Divide normalized progress into K equal bands and select nearest to midpoint.

    Returns list of selected checkpoint dicts with:
      checkpoint_index (1-based), checkpoint_count K, checkpoint_band, normalized_progress,
      selection_reason, opportunity record reference (the retained record).
    Skips empty bands (§10). Uses K dynamic per §9.
    """
    if not collapsed or game_end_turn <= 0:
        return []
    # Ensure normalized_progress (mutate in place so caller sees it)
    for r in collapsed:
        if "normalized_progress" not in r or r["normalized_progress"] is None:
            turn = r.get("turn", 0)
            try:
                r["normalized_progress"] = float(turn) / float(game_end_turn) if game_end_turn else 0.0
            except Exception:
                r["normalized_progress"] = 0.0
    # Compute K
    k = checkpoint_count_for_length(game_end_turn, len(collapsed))
    if k == 0:
        return []
    # Sort candidates by normalized_progress
    cands = sorted(collapsed, key=lambda r: r.get("normalized_progress", 0.0))
    # Precompute band ranges and midpoints
    selected: List[Dict[str, Any]] = []
    for band in range(k):
        band_low = band / k
        band_high = (band + 1) / k
        midpoint = (band + 0.5) / k
        # Candidates inside band: [low, high) except last inclusive
        in_band = []
        for c in cands:
            np = float(c.get("normalized_progress", 0.0))
            inside = (np >= band_low and np < band_high) if band < k - 1 else (np >= band_low and np <= 1.0)
            if inside:
                in_band.append(c)
        if not in_band:
            continue
        # Select closest to midpoint; tie via ranking
        def _dist(c: Dict[str, Any]) -> Tuple[float, Tuple[int, float, int, int]]:
            d = abs(float(c.get("normalized_progress", 0)) - midpoint)
            return (d, _rank_key(c))

        best = min(in_band, key=_dist)
        # Determine target progress (midpoint)
        sel = {
            "trajectory_id": trajectory_id,
            "seat_index": seat_index,
            "checkpoint_index": len(selected) + 1,  # 1-based, will be renumbered after skipping? spec expects sequential
            "checkpoint_count": k,
            "checkpoint_band": band,
            "normalized_progress": float(best.get("normalized_progress", 0)),
            "target_progress": midpoint,
            "selection_reason": "nearest_midpoint",
            "opportunity": best,
            "band_low": band_low,
            "band_high": band_high,
        }
        selected.append(sel)

    # After skipping, checkpoint_count stays as K (max intended), but checkpoint_index is sequential among selected
    # Already assigned sequentially; ensure count remains K
    for s in selected:
        s["checkpoint_count"] = k
    return selected
