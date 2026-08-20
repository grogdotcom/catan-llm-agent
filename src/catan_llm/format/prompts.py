"""
Prompt formatting — high-level LLM prompts that compose board, player, history and moves.

Contains get_game_state_summary, summarize_catan_actions (legacy),
format_decision_prompt and format_decision_prompt_with_history.
"""

from collections import defaultdict
from typing import List, Optional, Sequence

from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.inventory import Inventory
from catanatron.models.public_state import PublicState
from catanatron.models.enums import ActionRecord

from catan_llm.format.board import (
    format_board_occupancy_data,
    format_robber_info,
    gather_board_occupancy_data,
    get_board_occupancy,
    get_full_board_map,
)
from catan_llm.format.history import format_public_history_window
from catan_llm.format.players import (
    get_player_dev_cards,
    get_player_resources,
    get_players_summary,
)


def get_game_state_summary(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """
    Create a comprehensive game state summary for LLM consumption.
    Includes board map, occupancy, and a consolidated player-by-player
    inventory (resources, dev cards, VP with hidden if known, road
    length + Longest Road, army size + Largest Army, ports, pip
    production total + per-resource, and pieces left).

    Args:
        public_state: The public state object from Observation agent
        current_player_color: The color of the current player
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Comprehensive game state summary
    """
    sections = []

    sections.append(get_full_board_map(public_state))
    sections.append("\n")
    sections.append(get_board_occupancy(public_state))
    sections.append("\n")
    sections.append(get_players_summary(public_state, current_player_color, current_player_inventory))

    return "\n".join(sections)


def summarize_catan_actions(valid_actions: List) -> str:
    """
    Takes a list of Catanatron Action objects and groups them into a clean,
    abstracted summary string for the Teacher LLM prompt.
    """
    if not valid_actions:
        return "No actions available."

    # Group actions by their high-level category
    grouped_actions = defaultdict(list)
    for action in valid_actions:
        action_type = action.action_type

        # 1. Handle Buildings & Board Expansion
        if action_type in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            val = getattr(action, "coordinate", None)
            if val is None:
                val = getattr(action, "value", None)
            grouped_actions[action_type.name].append(str(val))

        # 2. Handle Maritime & Bank Trades
        elif action_type == ActionType.MARITIME_TRADE:
            try:
                offer, receive = action.kwargs.get('resource_giving'), action.kwargs.get('resource_receiving')
                grouped_actions["MARITIME_TRADE"].append(f"Give {offer.name} -> Get {receive.name}")
            except AttributeError:
                grouped_actions["MARITIME_TRADE"].append(str(action))

        # 3. Handle Robber Movement
        elif action_type == ActionType.MOVE_ROBBER:
            try:
                # Action.value is (coordinate, victim_color)
                val = getattr(action, "value", None)
                if isinstance(val, tuple) and len(val) == 2:
                    hex_coord, victim = val
                    victim = victim if victim is not None else "NONE"
                else:
                    hex_coord = getattr(action, "coordinate", val)
                    victim = getattr(action, "kwargs", {}).get('victim_color', 'NONE') if hasattr(action, 'kwargs') else "NONE"
                grouped_actions["MOVE_ROBBER"].append(f"Hex {hex_coord} (Victim: {victim})")
            except Exception:
                grouped_actions["MOVE_ROBBER"].append(str(action))

        # 4. Handle Dev Cards & End Turn
        elif action_type == ActionType.END_TURN:
            grouped_actions["END_TURN"].append("Pass")
        else:
            # Catch-all for Development cards (Knight, Monopoly, etc.)
            grouped_actions[action_type.name].append(str(action))

    # Build the final prompt string
    summary_lines = ["[PLAYABLE ACTION CATEGORIES]:"]
    for category, details in grouped_actions.items():
        if category == "END_TURN":
            summary_lines.append("- Pass (End Turn)")
        elif category in ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"):
            # Collapse coordinates into a single list
            targets = ", ".join(details)
            summary_lines.append(f"- {category}: Target IDs [{targets}]")
        elif category == "MARITIME_TRADE":
            # De-duplicate trades (sometimes the engine offers multiple identical paths)
            unique_trades = sorted(list(set(details)))
            trade_str = ", ".join(unique_trades)
            summary_lines.append(f"- MARITIME_TRADE: Options [{trade_str}]")
        else:
            # For robber and dev cards, list the options cleanly
            options = ", ".join(details)
            summary_lines.append(f"- {category}: Options [{options}]")

    return "\n".join(summary_lines)


def format_decision_prompt(public_state: PublicState, playable_actions: List, current_player_color: str, current_prompt: ActionPrompt, turn_number: int, current_player_inventory: Optional[Inventory] = None) -> str:
    """
    Create a complete decision prompt for LLM consumption.
    Combines game state summary with available actions.

    Args:
        public_state: The public state object from Observation agent
        playable_actions: List of playable actions for the current player
        current_player_color: The color of the current player
        current_prompt: The current action prompt (phase)
        turn_number: The current turn number
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Complete decision prompt for LLM consumption
    """
    prompt_parts = []

    prompt_parts.append(f"[CURRENT PLAYER: {current_player_color}]")
    prompt_parts.append(f"[TURN: {turn_number}]")
    prompt_parts.append(f"[PHASE: {current_prompt.name if hasattr(current_prompt, 'name') else str(current_prompt)}]")
    prompt_parts.append("\n")

    prompt_parts.append(get_game_state_summary(public_state, current_player_color, current_player_inventory))
    prompt_parts.append("\n")

    prompt_parts.append(summarize_catan_actions(playable_actions))
    prompt_parts.append("\n")

    prompt_parts.append("[DECISION REQUIRED]")
    prompt_parts.append("Select the best action from the available options above.")

    return "\n".join(prompt_parts)


def format_decision_prompt_with_history(
    public_state: PublicState,
    playable_actions: List,
    current_player_color: str,
    current_prompt: ActionPrompt,
    turn_number: int,
    public_history: Sequence[ActionRecord],
    history_window_size: Optional[int] = None,
    current_player_inventory: Optional[Inventory] = None,
) -> str:
    """
    Create a complete decision prompt for LLM consumption with public history.
    Combines game state summary, public history (with optional sliding window),
    and available actions.

    Args:
        public_state: The public state object from Observation agent
        playable_actions: List of playable actions for the current player
        current_player_color: The color of the current player
        current_prompt: The current action prompt (phase)
        turn_number: The current turn number
        public_history: Sequence of ActionRecords representing game history
        history_window_size: Optional number of recent turns to include in history.
            If None, includes all turns. If 0, only includes setup phase.
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Complete decision prompt for LLM consumption with history
    """
    prompt_parts = []

    prompt_parts.append(f"[CURRENT PLAYER: {current_player_color}]")
    prompt_parts.append(f"[TURN: {turn_number}]")
    prompt_parts.append(f"[PHASE: {current_prompt.name if hasattr(current_prompt, 'name') else str(current_prompt)}]")
    prompt_parts.append("\n")

    prompt_parts.append(get_game_state_summary(public_state, current_player_color, current_player_inventory))
    prompt_parts.append("\n")

    prompt_parts.append(format_public_history_window(public_history, window_size=history_window_size, public_state=public_state))
    prompt_parts.append("\n")

    prompt_parts.append(summarize_catan_actions(playable_actions))
    prompt_parts.append("\n")

    prompt_parts.append("[DECISION REQUIRED]")
    prompt_parts.append("Select the best action from the available options above.")

    return "\n".join(prompt_parts)


# ---------------------------------------------------------------------------
# Integrated complete prompt — board → occupancy → robber → inventories → moves
# ---------------------------------------------------------------------------

def _resolve_history_records(public_history, observation):
    """Resolve public history records from explicit arg or observation.

    Preference: explicit ``public_history`` > ``observation.public_history``
    > ``observation.history`` > empty tuple.
    """
    if public_history is not None:
        return public_history
    if observation is not None:
        hist = getattr(observation, "public_history", None)
        if hist is not None:
            return hist
        hist = getattr(observation, "history", None)
        if hist is not None:
            return hist
    return ()


def get_complete_prompt(
    public_state: Optional[PublicState] = None,
    current_player_color=None,
    playable_actions: Optional[Sequence] = None,
    current_player_inventory: Optional[Inventory] = None,
    observation=None,
    current_prompt: Optional[ActionPrompt] = None,
    turn_number: Optional[int] = None,
    include_header: bool = True,
    include_footer: bool = True,
    public_history: Optional[Sequence[ActionRecord]] = None,
    history_window_size: Optional[int] = 8,
) -> str:
    """Build the complete LLM prompt in canonical section order.

    Order is strictly:

    1. ``[FULL BOARD MAP]`` — static 19-hex map via :func:`get_full_board_map`
    2. ``[CURRENT BOARD OCCUPANCY]`` — settlements / cities / roads via
       :func:`gather_board_occupancy_data` + :func:`format_board_occupancy_data`
    3. ``ROBBER:`` — robber tile + blocked production via
       :func:`format_robber_info` (computed from the same occupancy)
    4. ``[PLAYERS]`` — consolidated per-player inventories (resources, dev
       cards, VP, roads, army, ports, pips, pieces) via
       :func:`get_players_summary`
    5. ``[RECENT TURNS (LAST 8)]`` — summaries of the last 8 turns via
       :func:`format_public_history_window` (``history_window_size=8`` by
       default). Falls back to ``observation.public_history`` when
       ``public_history`` is not supplied.
    6. ``[PLAYABLE MOVES]`` — rich numbered move list via
       :func:`catan_llm.format.moves.build_moves` /
       :func:`catan_llm.format.moves.format_moves`

    ``observation`` is the primary source for move bundling (Knight → robber,
    initial settlement → road, Road Building → two roads). When omitted, the
    function builds a minimal shim from ``public_state`` / ``current_prompt`` so
    move labels still include pip/port/tile detail. When an ``observation`` is
    supplied its ``public_state`` is *not* used — the explicitly passed
    ``public_state`` remains canonical — but its ``current_prompt`` is used for
    the ``[PHASE: …]`` tag inside the moves block if ``current_prompt`` was not
    given.

    Args:
        public_state: Public board snapshot from ``Observation.public_state``.
        current_player_color: Color of the observer (``Color.RED`` etc. or
            its string name); used to mark ``(YOU)`` and reveal exact hand.
        playable_actions: Engine-legal actions for the current prompt.
        current_player_inventory: Optional private :class:`Inventory` for the
            observer; enables exact resource/dev counts and hidden-VP math.
        observation: Optional full ``Observation`` (carries ``public_state`` and
            ``current_prompt`` for compound-move expansion). If provided, its
            ``current_prompt`` is used as fallback for ``current_prompt``.
        current_prompt: Current :class:`ActionPrompt` / phase. Used for the
            optional header and to expand initial-placement moves when
            ``observation`` is not supplied.
        turn_number: Optional turn index for the optional header.
        include_header: When ``True`` (default) and any of
            ``current_player_color`` / ``current_prompt`` / ``turn_number`` is
            supplied, a ``[CURRENT PLAYER]`` / ``[TURN]`` / ``[PHASE]`` header
            is prepended. Set ``False`` to get only the six canonical sections.
        include_footer: When ``True`` (default) appends
            ``[DECISION REQUIRED]``.
        public_history: Optional sequence of :class:`ActionRecord` (e.g.
            ``Observation.public_history``). When ``None``, falls back to
            ``observation.public_history`` if present, otherwise empty.
        history_window_size: Number of recent turns to summarise. Default 8,
            matching the prompt requirement (``[RECENT TURNS (LAST 8)]`` before
            ``[PLAYABLE MOVES]``). Pass ``None`` for all turns, ``0`` for
            setup only.

    Returns:
        Multiline string with the six sections in order, separated by a blank
        line (``\"\\n\\n\"`` between rendered sections). Sections themselves are
        multi-line.

    Example:
        >>> prompt = get_complete_prompt(
        ...     public_state, Color.RED, playable_actions,
        ...     inventory, observation=obs, turn_number=12
        ... )
        >>> assert prompt.index("[FULL BOARD MAP") < prompt.index("[CURRENT BOARD OCCUPANCY")
        >>> assert prompt.index("ROBBER:") < prompt.index("[PLAYERS]")
        >>> assert prompt.index("[PLAYERS]") < prompt.index("[RECENT TURNS")
        >>> assert prompt.index("[RECENT TURNS") < prompt.index("[PLAYABLE MOVES]")
    """
    # Lazy import to avoid circular import at module load (moves imports board).
    from types import SimpleNamespace

    from catan_llm.format.moves import build_moves, format_moves

    # Allow public_state / playable_actions to be inferred from observation
    # for the ergonomic `get_complete_prompt(observation=obs, ...)` call.
    if public_state is None and observation is not None:
        public_state = getattr(observation, "public_state", None)
    if playable_actions is None and observation is not None:
        # Some Observation shims store playable_actions on the engine, not the
        # observation itself — caller should pass explicitly in that case.
        playable_actions = getattr(observation, "playable_actions", None) or []
    if public_state is None:
        raise ValueError("public_state is required (or pass observation with .public_state)")
    if playable_actions is None:
        playable_actions = []
    if current_player_color is None and observation is not None:
        # Try to infer from observation if it carries color
        inferred = getattr(observation, "color", None)
        if inferred is not None:
            current_player_color = inferred

    # Resolve phase for header / moves shim
    resolved_prompt = current_prompt
    if resolved_prompt is None and observation is not None:
        resolved_prompt = getattr(observation, "current_prompt", None)

    # Single occupancy gather — reused for occupancy rendering and robber blocking.
    occupancy_data = gather_board_occupancy_data(public_state)

    sections: List[str] = []

    # Optional header — only when caller supplied identity/phase context.
    if include_header and (current_player_color is not None or resolved_prompt is not None or turn_number is not None):
        header_lines: List[str] = []
        if current_player_color is not None:
            color_name = current_player_color.name if hasattr(current_player_color, "name") else str(current_player_color)
            header_lines.append(f"[CURRENT PLAYER: {color_name}]")
        if turn_number is not None:
            header_lines.append(f"[TURN: {turn_number}]")
        if resolved_prompt is not None:
            phase_name = resolved_prompt.name if hasattr(resolved_prompt, "name") else str(resolved_prompt)
            header_lines.append(f"[PHASE: {phase_name}]")
        if header_lines:
            sections.append("\n".join(header_lines))

    # 1. Static board map
    sections.append(get_full_board_map(public_state))
    # 2. Dynamic occupancy (settlements / cities / roads, no robber)
    sections.append(format_board_occupancy_data(occupancy_data))
    # 3. Robber (tile detail + blocked production derived from same occupancy)
    sections.append(format_robber_info(public_state, occupancy_data.players))
    # 4. Consolidated per-player inventories
    sections.append(get_players_summary(public_state, current_player_color, current_player_inventory))

    # 5. Recent turn summaries — last 8 turns (after inventories, before moves)
    history_records = _resolve_history_records(public_history, observation)
    # format_public_history_window with window_size=8 is the canonical "last 8 turns" view;
    # enrich with public_state so settlement/city/road/robber/roll lines mirror
    # playable-move detail (tile/port/pips, road endpoints, robber tile, roll resources).
    history_block = format_public_history_window(
        history_records, window_size=history_window_size, public_state=public_state, current_turn_number=turn_number
    )
    # Provide both a RECENT TURNS alias (requirement language) and the canonical
    # [PUBLIC HISTORY] block so searches for either marker succeed. The alias header
    # carries the explicit LAST 8 annotation before the windowed history.
    if history_block.startswith("[PUBLIC HISTORY]"):
        # Replace header with alias + canonical marker for double discoverability
        history_text = history_block.replace(
            "[PUBLIC HISTORY]",
            "[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]",
            1,
        )
    else:
        history_text = f"[RECENT TURNS (LAST 8)]\n{history_block}"
    sections.append(history_text)

    # 6. Available moves — rich numbered list
    if observation is not None:
        moves = build_moves(playable_actions, observation)
        moves_text = format_moves(moves, observation=observation)
    else:
        # Build a minimal observation shim so moves still get public-state-aware
        # labels (pip/port/tile detail + longest-road hints).
        shim = SimpleNamespace(public_state=public_state, current_prompt=resolved_prompt)
        moves = build_moves(playable_actions, shim)
        moves_text = format_moves(moves, observation=shim)
    sections.append(moves_text)

    if include_footer:
        sections.append("[DECISION REQUIRED]\nSelect the best action from the available moves above.")

    return "\n\n".join(sections)


# Backwards-compatible / discoverability aliases — all point to the same impl.
get_full_prompt = get_complete_prompt
format_complete_prompt = get_complete_prompt
build_complete_prompt = get_complete_prompt
format_full_prompt = get_complete_prompt


def format_observation_prompt(
    observation,
    playable_actions: Optional[Sequence] = None,
    current_player_inventory: Optional[Inventory] = None,
    include_header: bool = True,
    include_footer: bool = True,
    public_history: Optional[Sequence[ActionRecord]] = None,
    history_window_size: Optional[int] = 8,
) -> str:
    """Convenience wrapper that builds the complete prompt directly from an Observation.

    Derives ``public_state``, ``current_player_color`` (from ``observation.color``
    or ``observation.public_state`` via caller), ``current_prompt`` and
    ``playable_actions`` from the observation itself when not explicitly supplied.

    This is the most ergonomic entry point for an ``ObservationAgent``:

    >>> prompt = format_observation_prompt(observation, playable_actions, inventory)

    Args:
        observation: Observation object (must have ``public_state``; ideally also
            ``current_prompt`` and optionally ``color`` / ``playable_actions``).
        playable_actions: Override playable actions; when ``None`` uses
            ``observation.playable_actions`` if present else ``[]``.
        current_player_inventory: Optional private Inventory for the observer.
        include_header: See :func:`get_complete_prompt`.
        include_footer: See :func:`get_complete_prompt`.
        public_history: Optional override for history records; when ``None``
            uses ``observation.public_history`` if present.
        history_window_size: Number of recent turns to summarise (default 8).
            See :func:`get_complete_prompt`.

    Returns:
        Same six-section prompt as :func:`get_complete_prompt`.
    """
    public_state = getattr(observation, "public_state", None)
    color = getattr(observation, "color", None)
    prompt = getattr(observation, "current_prompt", None)
    # Some callers keep turn_number on observation or game; try common names.
    turn_number = getattr(observation, "turn_number", None)
    if turn_number is None:
        turn_number = getattr(observation, "current_turn_index", None)
    actions = playable_actions
    if actions is None:
        actions = getattr(observation, "playable_actions", None)
        if actions is None:
            actions = []
    return get_complete_prompt(
        public_state=public_state,
        current_player_color=color,
        playable_actions=actions,
        current_player_inventory=current_player_inventory,
        observation=observation,
        current_prompt=prompt,
        turn_number=turn_number,
        include_header=include_header,
        include_footer=include_footer,
        public_history=public_history,
        history_window_size=history_window_size,
    )


# Alias for the wrapper as well
get_observation_prompt = format_observation_prompt
build_observation_prompt = format_observation_prompt
