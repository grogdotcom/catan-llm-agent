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
    _abbr_resource,
    _format_maritime_trade_value,
    _format_resource_counts,
    _format_trade_offer_value,
    _name_of,
    get_pip_count,
)

# --- Patch engine to record monopoly steals for history ---
# The base engine's apply_play_monopoly returns ActionRecord with result=None,
# so history cannot show per-player losses. We capture stolen counts before
# the engine mutates state and store them as result=(resource, stolen_tuple, total)
# where stolen_tuple is tuple of (Color, count) pairs.
try:
    import catanatron.apply_action as _apply_mod  # type: ignore

    _orig_monopoly = _apply_mod.apply_play_monopoly

    def _patched_monopoly(state, action):  # type: ignore[no-redef]
        mono_resource = action.value
        stolen: dict = {}
        try:
            from catanatron.state_functions import player_key

            for col in state.colors:
                if col != action.color:
                    key = player_key(state, col)
                    cnt = state.player_state.get(f"{key}_{mono_resource}_IN_HAND", 0)
                    if cnt:
                        stolen[col] = cnt
        except Exception:
            stolen = {}
        rec = _orig_monopoly(state, action)
        total = sum(stolen.values()) if stolen else 0
        # Use sorted tuple for deterministic, pickle-friendly result
        try:
            stolen_tuple = tuple(
                sorted(stolen.items(), key=lambda kv: kv[0].name if hasattr(kv[0], "name") else str(kv[0]))
            )
        except Exception:
            stolen_tuple = tuple(stolen.items())
        # Keep original resource in result for formatting
        return ActionRecord(action=rec.action, result=(mono_resource, stolen_tuple, total))

    _apply_mod.apply_play_monopoly = _patched_monopoly
except Exception:
    pass


def _describe_roll_resources(public_state, dice_total: int) -> str:
    """Return concise resource-collection summary for a dice total.

    Includes blocked production when the roll matches the robber tile.
    """

    if public_state is None or dice_total == 7:
        return ""
    robber_tile_id = getattr(public_state.board, "robber_tile_id", None)
    # Fallback for older board shape using robber_coordinate -> tile lookup
    if robber_tile_id is None:
        # Try to derive from coordinate if needed (not critical for tests)
        robber_tile_id = getattr(public_state.board, "robber_tile", None)
    gains: dict[str, List[str]] = defaultdict(list)
    blocked_gains: dict[str, List[str]] = defaultdict(list)
    blocked_pips: dict[str, int] = defaultdict(int)
    tiles = getattr(public_state.board.map, "tiles", {})
    adjacent_tiles = getattr(public_state.board.map, "adjacent_tiles", {})
    buildings = getattr(public_state.board, "buildings", {})

    # Pre-fetch robber tile info for blocked check
    robber_resource = None
    robber_roll = None
    robber_pips = 0
    if robber_tile_id is not None and robber_tile_id in tiles:
        robber_resource, robber_roll = tiles[robber_tile_id]
        robber_pips = get_pip_count(robber_roll)

    for tile_id, (resource, roll) in tiles.items():
        if roll != dice_total:
            continue
        if resource is None:
            continue
        resource_name = resource.name if hasattr(resource, "name") else str(resource)
        is_robber = tile_id == robber_tile_id
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
            if is_robber:
                # Blocked production — record pips and resource counts
                for _ in range(count):
                    blocked_gains[owner_name].append(resource_name)
                blocked_pips[owner_name] += robber_pips * count
            else:
                for _ in range(count):
                    gains[owner_name].append(resource_name)

    # Format gains part — abbreviated resources with bracketed counts
    # Example: WHITE + [1 Br, 1 Wh], RED + [2 Br]
    parts = []
    for owner in sorted(gains.keys()):
        cnt = Counter(gains[owner])
        ordered = [r for r in RESOURCES if r in cnt]
        inner = ", ".join(f"{cnt[r]} {_abbr_resource(r)}" for r in ordered)
        parts.append(f"{owner} + [{inner}]")

    blocked_parts = []
    for owner in sorted(blocked_gains.keys()):
        cnt = Counter(blocked_gains[owner])
        ordered = [r for r in RESOURCES if r in cnt]
        inner = ", ".join(f"{cnt[r]} {_abbr_resource(r)}" for r in ordered)
        blocked_parts.append(f"{owner} [{inner}]")

    if gains and blocked_parts:
        return " | " + ", ".join(parts) + f" | blk " + ", ".join(blocked_parts)
    if gains:
        return " | " + ", ".join(parts)
    if blocked_parts:
        return f" | blk " + ", ".join(blocked_parts)
    return " | no resources"


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
                return f"{color} built S {node_desc}"
            except Exception:
                pass
        return f"{color} built S Node {value}"

    if action_type == ActionType.BUILD_CITY:
        if public_state is not None:
            try:
                from catan_llm.format.moves import _describe_node

                node_desc = _describe_node(public_state, value)
                return f"{color} built C {node_desc}"
            except Exception:
                pass
        return f"{color} built C Node {value}"

    if action_type == ActionType.BUILD_ROAD:
        edge = tuple(sorted(value)) if value is not None else value
        return f"{color} built road {edge}"

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
        # Tile display mirrors board layout: Tile N: ROLL-Abbr(Pips) / DESERT
        tile_detail = None
        if public_state is not None and coordinate is not None:
            try:
                tile_id = None
                for tid, coord in public_state.board.map.tile_coordinates.items():
                    if coord == coordinate:
                        tile_id = tid
                        break
                if tile_id is not None:
                    resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
                    if resource is None:
                        tile_detail = f"Tile {tile_id}: DESERT"
                    else:
                        res_name = resource.name if hasattr(resource, "name") else str(resource)
                        abbr = _abbr_resource(res_name)
                        pips = get_pip_count(roll)
                        tile_detail = f"Tile {tile_id}: {roll}-{abbr}({pips}p)"
                if tile_detail is None:
                    from catan_llm.format.utils import _format_coordinate as _fmt_coord

                    tile_detail = _fmt_coord(coordinate)
            except Exception:
                tile_detail = None
        if tile_detail is not None:
            coord_str = tile_detail
        else:
            if coordinate is not None:
                from catan_llm.format.utils import _format_coordinate as _fmt_coord

                try:
                    coord_str = _fmt_coord(coordinate)
                except Exception:
                    coord_str = str(coordinate)
            else:
                coord_str = "unknown"
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
        return f"{color} discarded {_abbr_resource(_name_of(discarded))}"

    if action_type == ActionType.PLAY_KNIGHT_CARD:
        return f"{color} played Knight"

    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        if value is None:
            return f"{color} played YOP"
        cards = ", ".join(_abbr_resource(_name_of(r)) for r in value)
        return f"{color} played YOP: took {cards}"

    if action_type == ActionType.PLAY_MONOPOLY:
        # Condensed: BLUE played MONOPOLY on OR | RED - 2 OR, WHITE - 1 OR (total 3)
        raw_res = _name_of(value) if value is not None else ""
        abbr = _abbr_resource(raw_res).upper() if raw_res else ""
        if not abbr and value is not None:
            abbr = str(value).upper()
        base = f"{color} played MONOPOLY on {abbr}" if abbr else f"{color} played MONOPOLY"
        if result is not None:
            try:
                # Patched result is (resource, stolen_tuple, total)
                if isinstance(result, tuple) and len(result) == 3:
                    res_resource, stolen_tuple, total = result
                    res_abbr = (
                        _abbr_resource(_name_of(res_resource)).upper()
                        if res_resource
                        else abbr
                    )
                    if not res_abbr:
                        res_abbr = abbr
                    if isinstance(stolen_tuple, (list, tuple)):
                        parts = []
                        for c, cnt in stolen_tuple:
                            if cnt and cnt > 0:
                                parts.append(f"{_name_of(c)} - {cnt} {res_abbr}")
                        if parts:
                            base += f" | {', '.join(parts)} (total {total})"
                        elif total == 0:
                            base += " | stole nothing (total 0)"
                        else:
                            base += f" (total {total})"
                        return base
                if isinstance(result, dict) and "stolen" in result:
                    stolen = result["stolen"]
                    total = result.get("total", sum(v for v in stolen.values() if isinstance(v, int)))
                    # try to get resource abbr from dict
                    res_abbr = abbr
                    if "resource" in result and result["resource"]:
                        res_abbr = _abbr_resource(_name_of(result["resource"])).upper()
                    parts = [
                        f"{_name_of(c)} - {cnt} {res_abbr}"
                        for c, cnt in sorted(stolen.items(), key=lambda kv: _name_of(kv[0]))
                        if cnt and cnt > 0
                    ]
                    if parts:
                        base += f" | {', '.join(parts)} (total {total})"
                    else:
                        base += f" | stole nothing (total {total})"
                    return base
                if isinstance(result, int):
                    base += f" (total {result})"
                    return base
            except Exception:
                pass
        return base

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
            return f"{color} traded"
        acceptor = _name_of(value[10]) if len(value) > 10 else "unknown"
        try:
            offered = _format_resource_counts(value[:5])
            asking = _format_resource_counts(value[5:10])
            return f"{color} gave {acceptor} [{offered}] for [{asking}]"
        except Exception:
            trade_part = _format_trade_offer_value(value[:10])
            return f"{color} gave {acceptor} {trade_part}"

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
    i = 0
    while i < len(records):
        record = records[i]
        # Filter trade noise — only the offer and final execution (traded)
        # are kept. Rejections, intermediate accepts, and cancels are implicit
        # (no traded line = no deal) and just add spam; this gives
        # "Blue offers [1 Wd] for [1 Br], Red offers [2 Or] for [1 Wd], Red traded ... with Blue"
        # without per-player reject/cancel spam.
        if record.action.action_type in (
            ActionType.REJECT_TRADE,
            ActionType.ACCEPT_TRADE,
            ActionType.CANCEL_TRADE,
        ):
            i += 1
            continue
        # Aggregate contiguous DISCARD_RESOURCE records (e.g. 7-roll) per player
        # to match bundled playable-move format: "ORANGE discarded WOOD, WHEAT, SHEEP, WHEAT (WOOD: 1, WHEAT: 2, SHEEP: 1)"
        if record.action.action_type == ActionType.DISCARD_RESOURCE:
            j = i
            batch = []
            while j < len(records) and records[j].action.action_type == ActionType.DISCARD_RESOURCE:
                batch.append(records[j])
                j += 1
            # Group by color preserving first-appearance order
            per_color: dict[str, list[str]] = {}
            order: list[str] = []
            for rec in batch:
                color = _name_of(rec.action.color)
                res_raw = _name_of(rec.result if rec.result is not None else rec.action.value)
                res = _abbr_resource(res_raw)
                if color not in per_color:
                    per_color[color] = []
                    order.append(color)
                per_color[color].append(res)
            for color in order:
                resources = per_color[color]
                if len(resources) == 1:
                    lines.append(f"  - {color} discarded {resources[0]}")
                else:
                    counts = Counter(resources)
                    seen: list[str] = []
                    for r in resources:
                        if r not in seen:
                            seen.append(r)
                    summary = ", ".join(f"{r}:{counts[r]}" for r in seen)
                    res_str = ", ".join(resources)
                    lines.append(f"  - {color} discarded {res_str} ({summary})")
            i = j
            continue

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
        i += 1
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
    current_turn_number: Optional[int] = None,
) -> str:
    """Format public_history with a sliding window of the last N turns.

    Setup is included only when the window covers the full history (early game)
    or when explicitly requested (``window_size==0``). Once the game has
    progressed beyond the window (midgame, truncated), setup is omitted to
    save prompt budget — recent turns carry the relevant signal.

    Args:
        records: Observation.public_history (or any ActionRecord sequence).
        window_size: Number of recent turns to include (excluding setup).
            If None, formats all turns (equivalent to format_public_history).
            If 0, only includes setup phase if present.
        public_state: Optional board snapshot for enriched detail (til/port/pips
            on settlements/cities/roads, robber tile, roll resources).
        current_turn_number: Optional header turn number (e.g. game.state.num_turns)
            to align history labels with the [TURN: N] header. When provided,
            completed window labels are computed as current_turn_number - len(windowed)
            .. current_turn_number-1 and the open current turn (if any) is labeled
            current_turn_number - CURRENT, matching the header.

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
                if current_turn_number is not None:
                    sections.append(f"[Showing last {len(windowed)} of {total_completed} turns + current (TURN {current_turn_number})]")
                else:
                    sections.append(f"[Showing last {len(windowed)} of {total_completed} turns + current (TURN {total_completed + 1})]")
            elif window_size < total_turns:
                sections.append(f"[Showing last {len(windowed)} of {total_completed} turns + current]")
        elif window_size < total_completed:
            sections.append(f"[Showing last {len(windowed)} of {total_completed} turns]")

    # Add setup phase if present — only when not truncated (early game) or explicit setup-only
    should_include_setup = False
    if setup_group is not None:
        if window_size is None:
            should_include_setup = True
        elif window_size == 0:
            should_include_setup = True
        elif total_completed <= window_size:
            # Window covers all turns (early game) — setup still relevant
            should_include_setup = True
        else:
            # Midgame truncated — omit setup to save budget
            should_include_setup = False
    if should_include_setup and setup_group is not None:
        sections.append(describe_turn(setup_group, turn_label="SETUP", public_state=public_state))

    # Add windowed completed turns with numbering.
    # When current_turn_number is provided and window is truncated, align
    # labels with header [TURN: N] so history current == header.
    # Otherwise use absolute sequential numbering (1..total).
    use_header_alignment = (
        current_turn_number is not None
        and window_size is not None
        and window_size > 0
        and window_size < total_completed
    )
    if use_header_alignment:
        # windowed are current_turn_number - len(windowed) .. current_turn_number -1
        # open (if any) will be current_turn_number
        for idx, group in enumerate(windowed):
            turn_number = current_turn_number - len(windowed) + idx
            # when open exists, last windowed should be current-1; without open, same
            # adjust for open presence: windowed last is current-1 regardless
            actor = _name_of(group[0].action.color)
            label = f"TURN {turn_number} ({actor})"
            sections.append(describe_turn(group, turn_label=label, public_state=public_state))
    else:
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
        if current_turn_number is not None:
            label = f"TURN {current_turn_number} ({actor}) - CURRENT"
        else:
            label = f"TURN {total_completed + 1} ({actor}) - CURRENT"
        sections.append(describe_turn(open_group, turn_label=label, public_state=public_state))

    return "\n".join(sections)
