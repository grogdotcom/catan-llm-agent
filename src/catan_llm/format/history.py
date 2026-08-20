"""
History formatting — action records, turn grouping, and public history.

Covers describe_action_record, group_action_records_by_turn, describe_turn,
format_public_history and windowed variants.
"""

from typing import Any, List, Optional, Sequence, Tuple

from catanatron.models.enums import ActionRecord, ActionType, RESOURCES

from catan_llm.format.models import _SETUP_ACTION_TYPES
from catan_llm.format.utils import (
    _format_maritime_trade_value,
    _format_resource_counts,
    _format_trade_offer_value,
    _name_of,
)


def describe_action_record(record: ActionRecord) -> str:
    """Describe a single ActionRecord as one structured human-readable line.

    Uses the sanitized public_history conventions: redacted fields (e.g. hidden
    stolen card, opponent dev-card identity) are phrased as unknown/hidden.

    Args:
        record: A (possibly sanitized) ActionRecord from Observation.public_history.

    Returns:
        A single-line description such as ``RED rolled 4+3 = 7``.
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
            return f"{color} rolled {dice[0]}+{dice[1]} = {total}"
        return f"{color} rolled"

    if action_type == ActionType.END_TURN:
        return f"{color} ended turn"

    if action_type == ActionType.BUILD_SETTLEMENT:
        return f"{color} built settlement at node {value}"

    if action_type == ActionType.BUILD_CITY:
        return f"{color} built city at node {value}"

    if action_type == ActionType.BUILD_ROAD:
        edge = tuple(sorted(value)) if value is not None else value
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
) -> str:
    """Describe one turn group as structured human-readable text.

    Args:
        records: ActionRecords belonging to a single turn (from
            ``group_action_records_by_turn``).
        turn_label: Optional header label (e.g. ``"SETUP"``, ``"TURN 3"``).
            When omitted, a label is inferred from the records.

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

    lines = [f"[{turn_label}]"]
    for record in records:
        lines.append(f"  - {describe_action_record(record)}")
    return "\n".join(lines)


def format_public_history(records: Sequence[ActionRecord]) -> str:
    """Format a full public_history as turn-grouped human-readable text.

    Groups records via ``group_action_records_by_turn``, then describes each
    turn. Setup is labeled ``SETUP``; subsequent turns are ``TURN 1``,
    ``TURN 2``, ... matching completed END_TURN boundaries (and a final
    open turn if present).

    Args:
        records: Observation.public_history (or any ActionRecord sequence).

    Returns:
        Multi-line string ready for LLM consumption.
    """
    groups = group_action_records_by_turn(records)
    if not groups:
        return "[PUBLIC HISTORY]\n  (empty)"

    sections = ["[PUBLIC HISTORY]"]
    turn_number = 0
    for group in groups:
        is_setup = all(r.action.action_type in _SETUP_ACTION_TYPES for r in group)
        if is_setup and turn_number == 0:
            label = "SETUP"
        else:
            turn_number += 1
            actor = _name_of(group[0].action.color)
            label = f"TURN {turn_number} ({actor})"
        # Skip the outer [PUBLIC HISTORY] duplication inside describe_turn body
        sections.append(describe_turn(group, turn_label=label))

    return "\n".join(sections)


def format_public_history_window(
    records: Sequence[ActionRecord],
    window_size: Optional[int] = None,
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

    # Apply sliding window to turn groups
    if window_size is not None and window_size >= 0:
        turn_groups = turn_groups[-window_size:] if window_size > 0 else []

    # Build sections
    sections = ["[PUBLIC HISTORY]"]
    
    # Add window indicator if we're using a window
    if window_size is not None:
        total_turns = len([g for g in groups if not all(r.action.action_type in _SETUP_ACTION_TYPES for r in g)])
        if window_size == 0:
            sections.append("[Showing setup phase only]")
        elif window_size < total_turns:
            sections.append(f"[Showing last {len(turn_groups)} of {total_turns} turns]")

    # Add setup phase if present
    if setup_group:
        sections.append(describe_turn(setup_group, turn_label="SETUP"))

    # Add turn groups with proper numbering
    turn_number = 0
    for group in turn_groups:
        turn_number += 1
        actor = _name_of(group[0].action.color)
        label = f"TURN {turn_number} ({actor})"
        sections.append(describe_turn(group, turn_label=label))

    return "\n".join(sections)
