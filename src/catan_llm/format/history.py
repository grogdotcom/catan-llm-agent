"""
History formatting — action records, turn grouping, and public history.

Covers describe_action_record, group_action_records_by_turn, describe_turn,
format_public_history and windowed variants.
"""

from collections import Counter, defaultdict
from typing import Any, List, Optional, Sequence, Tuple

from catanatron.models.enums import ActionRecord, ActionType, RESOURCES

from catan_llm.format.models import _SETUP_ACTION_TYPES
from catan_llm.format.utils import (
    _format_maritime_trade_value,
    _format_resource_counts,
    _format_trade_offer_value,
    _name_of,
    get_pip_count,
)


def _describe_roll_resources(public_state, dice_total: int) -> str:
    """Return concise resource-collection summary for a dice total."""

    if public_state is None or dice_total == 7:
        return ""
    # Skip robber tile
    robber_tile_id = getattr(public_state.board, "robber_tile_id", None)
    gains: dict[str, List[str]] = defaultdict(list)
    tiles = getattr(public_state.board.map, "tiles", {})
    adjacent_tiles = getattr(public_state.board.map, "adjacent_tiles", {})
    buildings = getattr(public_state.board, "buildings", {})
    for tile_id, (resource, roll) in tiles.items():
        if roll != dice_total:
            continue
        if tile_id == robber_tile_id:
            continue
        if resource is None:
            continue
        resource_name = resource.name if hasattr(resource, "name") else str(resource)
        # nodes adjacent to this tile
        for node_id, tids in adjacent_tiles.items():
            if tile_id not in tids:
                continue
            b = buildings.get(node_id)
            if b is None:
                continue
            owner, btype = b
            is_city = str(btype) == "CITY" or (hasattr(btype, "name") and btype.name == "CITY")
            owner_name = _name_of(owner)
            count = 2 if is_city else 1
            for _ in range(count):
                gains[owner_name].append(resource_name)
    if not gains:
        return " — no resources (no settlements on roll)"
    parts = []
    for owner in sorted(gains.keys()):
        cnt = Counter(gains[owner])
        # Order by RESOURCES order
        ordered = [r for r in RESOURCES if r in cnt]
        inner = ", ".join(f"{cnt[r]} {r}" for r in ordered)
        parts.append(f"{owner} +{inner}")
    return " — " + "; ".join(parts)


def describe_action_record(record: ActionRecord, public_state=None) -> str:
    """Describe a single ActionRecord as one structured human-readable line.

    Uses the sanitized public_history conventions: redacted fields (e.g. hidden
    stolen card, opponent dev-card identity) are phrased as unknown/hidden.

    When ``public_state`` is supplied, settlement/city/road/robber and roll
    lines are enriched with the same tile/port/pip detail used in the
    playable-move list (adjacent tile + port info with pips, road endpoints,
    robber tile info, resources collected on roll).

    Args:
        record: A (possibly sanitized) ActionRecord from Observation.public_history.
        public_state: Optional board snapshot for enriched detail.

    Returns:
        A single-line description such as ``RED rolled 4+3 = 7`` or
        ``RED built settlement at Node 5: (Tile 0: 11 SHEEP (2 pips)) ...``.
    """
    action = record.action
    color = _name_of(action.color)
    action_type = action.action_type
    value = action.value
    result = record.result

    if action_type == ActionType.ROLL:
        dice = result if result is not None else value
        if dice is not None and len(dice) == 2:
            total = dice[0] + dice[1]
            base = f"{color} rolled {dice[0]}+{dice[1]} = {total}"
            if public_state is not None:
                base += _describe_roll_resources(public_state, total)
            return base
        return f"{color} rolled"

    if action_type == ActionType.END_TURN:
        return f"{color} ended turn"

    if action_type == ActionType.BUILD_SETTLEMENT:
        if public_state is not None:
            try:
                from catan_llm.format.moves import _describe_node

                node_desc = _describe_node(public_state, value)
                return f"{color} built settlement at {node_desc}"
            except Exception:
                pass
        return f"{color} built settlement at node {value}"

    if action_type == ActionType.BUILD_CITY:
        if public_state is not None:
            try:
                from catan_llm.format.moves import _describe_node

                node_desc = _describe_node(public_state, value)
                return f"{color} built city at {node_desc}"
            except Exception:
                pass
        return f"{color} built city at node {value}"

    if action_type == ActionType.BUILD_ROAD:
        edge = tuple(sorted(value)) if value is not None else value
        if public_state is not None and edge is not None:
            try:
                from catan_llm.format.moves import _describe_node

                # History road: show both endpoint nodes with tile/port/pips
                # (mirrors playable-move node detail but without prospective
                # reachability that would depend on the *current* board's future
                # buildings — e.g., a setup road should not show as blocked by a
                # city that was built later).
                a_desc = _describe_node(public_state, edge[0])
                b_desc = _describe_node(public_state, edge[1])
                return f"{color} built road on edge {edge} | connects {a_desc} <-> {b_desc}"
            except Exception:
                pass
        return f"{color} built road on edge {edge}"

    if action_type == ActionType.BUY_DEVELOPMENT_CARD:
        card = result if result is not None else value
        if card is None:
            return f"{color} bought a development card"
        return f"{color} bought development card: {_name_of(card)}"

    if action_type == ActionType.MOVE_ROBBER:
        coordinate = None
        victim = None
        if value is not None:
            coordinate, victim = value[0], value[1]
        # Enriched tile detail mirrors playable-move robber label
        tile_detail = None
        if public_state is not None and coordinate is not None:
            try:
                from catan_llm.format.moves import _robber_tile_detail

                tile_detail = _robber_tile_detail(public_state, coordinate)
            except Exception:
                tile_detail = None
        coord_str = tile_detail if tile_detail is not None else (coordinate if coordinate is not None else "unknown")
        # Fall back to raw coordinate formatting if tile_detail unavailable
        if tile_detail is None:
            coord_str = coordinate if coordinate is not None else "unknown"
        if victim is None:
            return f"{color} moved robber to {coord_str} (no steal)"
        victim_name = _name_of(victim)
        if result is None:
            return (
                f"{color} moved robber to {coord_str} and stole from "
                f"{victim_name} (card hidden)"
            )
        return (
            f"{color} moved robber to {coord_str} and stole "
            f"{_name_of(result)} from {victim_name}"
        )

    if action_type == ActionType.DISCARD_RESOURCE:
        discarded = result if result is not None else value
        return f"{color} discarded {_name_of(discarded)}"

    if action_type == ActionType.PLAY_KNIGHT_CARD:
        return f"{color} played Knight"

    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        if value is None:
            return f"{color} played Year of Plenty"
        cards = ", ".join(_name_of(r) for r in value)
        return f"{color} played Year of Plenty: took {cards}"

    if action_type == ActionType.PLAY_MONOPOLY:
        return f"{color} played Monopoly on {_name_of(value)}"

    if action_type == ActionType.PLAY_ROAD_BUILDING:
        return f"{color} played Road Building"

    if action_type == ActionType.MARITIME_TRADE:
        if value is None:
            return f"{color} maritime traded"
        return f"{color} maritime trade: {_format_maritime_trade_value(value)}"

    if action_type == ActionType.OFFER_TRADE:
        if value is None:
            return f"{color} offered a trade"
        return f"{color} {_format_trade_offer_value(value)}"

    if action_type == ActionType.ACCEPT_TRADE:
        if value is None:
            return f"{color} accepted a trade"
        return f"{color} accepted trade: {_format_trade_offer_value(value)}"

    if action_type == ActionType.REJECT_TRADE:
        if value is None:
            return f"{color} rejected a trade"
        return f"{color} rejected trade: {_format_trade_offer_value(value)}"

    if action_type == ActionType.CONFIRM_TRADE:
        if value is None:
            return f"{color} confirmed a trade"
        trade_part = _format_trade_offer_value(value[:10])
        acceptor = _name_of(value[10]) if len(value) > 10 else "unknown"
        return f"{color} confirmed trade with {acceptor}: {trade_part}"

    if action_type == ActionType.CANCEL_TRADE:
        return f"{color} cancelled trade"

    return f"{color} {action_type.name}: value={value!r}, result={result!r}"


def group_action_records_by_turn(
    records: Sequence[ActionRecord],
) -> List[Tuple[ActionRecord, ...]]:
    """Split a sequence of ActionRecords into turn groups.

    Rules:
    - Initial placement (only BUILD_SETTLEMENT / BUILD_ROAD from game start
      until the first non-setup action) is its own leading group.
    - After setup, each group is a contiguous run of records ending with
      END_TURN (the END_TURN is included in that group).
    - A trailing open turn (no END_TURN yet) is returned as the final group.

    Discards and trade responses by other colors stay inside the active
    player's turn, matching engine turn boundaries.

    Args:
        records: Tuple/list of ActionRecords (e.g. Observation.public_history).

    Returns:
        List of turn groups; each group is a non-empty tuple of ActionRecords.
    """
    if not records:
        return []

    groups: List[Tuple[ActionRecord, ...]] = []
    current: List[ActionRecord] = []
    in_setup = True

    for record in records:
        action_type = record.action.action_type

        if in_setup:
            if action_type in _SETUP_ACTION_TYPES:
                current.append(record)
                continue
            # First non-setup action ends the setup group.
            if current:
                groups.append(tuple(current))
                current = []
            in_setup = False

        current.append(record)
        if action_type == ActionType.END_TURN:
            groups.append(tuple(current))
            current = []

    if current:
        groups.append(tuple(current))

    return groups


def describe_turn(
    records: Sequence[ActionRecord],
    turn_label: Optional[str] = None,
    public_state=None,
) -> str:
    """Describe one turn group as structured human-readable text.

    Args:
        records: ActionRecords belonging to a single turn (from
            ``group_action_records_by_turn``).
        turn_label: Optional header label (e.g. ``"SETUP"``, ``"TURN 3"``).
            When omitted, a label is inferred from the records.
        public_state: Optional board snapshot for enriched detail (settlement
            tile/port/pips, road endpoints, robber tile, roll resources).

    Returns:
        Multi-line string: a header line plus one bullet per event.
    """
    if not records:
        return f"[{turn_label or 'TURN'}]\n  (no events)"

    if turn_label is None:
        first_type = records[0].action.action_type
        if first_type in _SETUP_ACTION_TYPES and all(
            r.action.action_type in _SETUP_ACTION_TYPES for r in records
        ):
            turn_label = "SETUP"
        else:
            actor = _name_of(records[0].action.color)
            turn_label = f"TURN ({actor})"

    # For SETUP, annotate the *second* settlement per color with its
    # starting resources — mirrors _setup_settlement_moves for playable moves.
    is_setup = turn_label == "SETUP" or (
        turn_label is None
        and records
        and records[0].action.action_type in _SETUP_ACTION_TYPES
        and all(r.action.action_type in _SETUP_ACTION_TYPES for r in records)
    )
    settlement_counts: dict[Any, int] = {}
    if is_setup and public_state is not None:
        settlement_counts = defaultdict(int)

    lines = [f"[{turn_label}]"]
    for record in records:
        base = describe_action_record(record, public_state=public_state)
        # Second initial settlement per color → append starting resources
        if is_setup and public_state is not None and record.action.action_type == ActionType.BUILD_SETTLEMENT:
            key = record.action.color
            settlement_counts[key] = settlement_counts.get(key, 0) + 1
            if settlement_counts[key] == 2:
                try:
                    from catan_llm.format.board import format_starting_resources

                    sr = format_starting_resources(public_state, record.action.value)
                    base += f" → Starting resources: {sr}"
                except Exception:
                    pass
        lines.append(f"  - {base}")
    return "\n".join(lines)


def format_public_history(records: Sequence[ActionRecord], public_state=None) -> str:
    """Format a full public_history as turn-grouped human-readable text.

    Groups records via ``group_action_records_by_turn``, then describes each
    turn. Setup is labeled ``SETUP``; subsequent turns are ``TURN 1``,
    ``TURN 2``, ... matching completed END_TURN boundaries (and a final
    open turn if present).

    Args:
        records: Observation.public_history (or any ActionRecord sequence).
        public_state: Optional board snapshot for enriched detail.

    Returns:
        Multi-line string ready for LLM consumption.
    """
    groups = group_action_records_by_turn(records)
    if not groups:
        return "[PUBLIC HISTORY]\n  (empty)"

    sections = ["[PUBLIC HISTORY]"]
    turn_number = 0
    for idx, group in enumerate(groups):
        is_setup = all(r.action.action_type in _SETUP_ACTION_TYPES for r in group)
        if is_setup and turn_number == 0:
            label = "SETUP"
        else:
            turn_number += 1
            actor = _name_of(group[0].action.color)
            label = f"TURN {turn_number} ({actor})"
            # trailing open group (no END_TURN) is the in-progress current turn
            is_last = idx == len(groups) - 1
            is_open = group[-1].action.action_type != ActionType.END_TURN
            if is_last and is_open and not is_setup:
                label += " - CURRENT"
        # Skip the outer [PUBLIC HISTORY] duplication inside describe_turn body
        sections.append(describe_turn(group, turn_label=label, public_state=public_state))

    return "\n".join(sections)


def format_public_history_window(
    records: Sequence[ActionRecord],
    window_size: Optional[int] = None,
    public_state=None,
) -> str:
    """Format public_history with a sliding window of the last N turns.

    This function efficiently formats only the last N turns without calculating
    descriptions for all previous turns. The setup phase is always included if
    present, as it provides important context about initial placements.

    Args:
        records: Observation.public_history (or any ActionRecord sequence).
        window_size: Number of recent turns to include (excluding setup).
            If None, formats all turns (equivalent to format_public_history).
            If 0, only includes setup phase if present.
        public_state: Optional board snapshot for enriched detail (til/port/pips
            on settlements/cities/roads, robber tile, roll resources).

    Returns:
        Multi-line string ready for LLM consumption with turn window indicator.
    """
    groups = group_action_records_by_turn(records)
    if not groups:
        return "[PUBLIC HISTORY]\n  (empty)"

    # Identify setup group (if present)
    setup_group = None
    turn_groups = []
    
    for group in groups:
        is_setup = all(r.action.action_type in _SETUP_ACTION_TYPES for r in group)
        if is_setup:
            setup_group = group
        else:
            turn_groups.append(group)

    # Separate completed turns (END_TURN) from trailing open turn (no END_TURN yet)
    # so previous actions of the current turn are always visible.
    completed = []
    open_group = None
    if turn_groups:
        # trailing group without END_TURN is the in-progress current turn
        if turn_groups[-1][-1].action.action_type != ActionType.END_TURN:
            open_group = turn_groups[-1]
            completed = turn_groups[:-1]
        else:
            completed = turn_groups

    total_completed = len(completed)
    # total for indicator includes open if present (matches previous total_turns semantics)
    total_turns = len(turn_groups)

    # Apply sliding window to *completed* turns only — open is always appended
    windowed = completed
    if window_size is not None and window_size >= 0:
        if window_size == 0:
            windowed = []
        elif window_size < total_completed:
            windowed = completed[-window_size:]

    # Build sections
    sections = ["[PUBLIC HISTORY]"]

    # Add window indicator if we're using a window
    if window_size is not None:
        if window_size == 0:
            sections.append("[Showing setup phase only]")
        elif open_group is not None:
            if window_size < total_completed:
                sections.append(f"[Showing last {len(windowed)} of {total_completed} turns + current (TURN {total_completed + 1})]")
            elif window_size < total_turns:
                sections.append(f"[Showing last {len(windowed)} of {total_completed} turns + current]")
        elif window_size < total_completed:
            sections.append(f"[Showing last {len(windowed)} of {total_completed} turns]")

    # Add setup phase if present
    if setup_group:
        sections.append(describe_turn(setup_group, turn_label="SETUP", public_state=public_state))

    # Add windowed completed turns with absolute numbering
    # E.g. total_completed=44, window=8 -> labels 37..44
    offset = total_completed - len(windowed)
    for idx, group in enumerate(windowed):
        turn_number = offset + idx + 1
        actor = _name_of(group[0].action.color)
        label = f"TURN {turn_number} ({actor})"
        sections.append(describe_turn(group, turn_label=label, public_state=public_state))

    # Append the in-progress current turn (if any) so previous actions
    # of the current turn are visible even when window truncates.
    # window_size==0 is explicitly "setup only" — do not include open.
    if open_group is not None and window_size != 0:
        actor = _name_of(open_group[0].action.color)
        # open is the next turn after completed — mark as CURRENT
        label = f"TURN {total_completed + 1} ({actor}) - CURRENT"
        sections.append(describe_turn(open_group, turn_label=label, public_state=public_state))

    return "\n".join(sections)
