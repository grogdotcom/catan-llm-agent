"""
Player formatting — resources, development cards and consolidated per-player overview.

The consolidated view goes player-by-player and shows in one place:

* resources (exact for the observer, hidden count otherwise)
* development cards (exact held + played for the observer, hidden count + played otherwise)
* visible points (+ hidden VP if known from the observer's inventory)
* road length + Longest Road badge
* army size (played knights) + Largest Army badge
* ports controlled (e.g. ``3:1`` or ``WOOD``) + pip production (total and per-resource)
* available pieces (settlements / cities / roads remaining)
"""

from typing import Dict, List, Optional

from catanatron.models.inventory import Inventory
from catanatron.models.public_state import PublicState

from catan_llm.format.utils import _name_of


def get_player_resources(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """Format player resource information for LLM consumption.

    Args:
        public_state: The public state object from Observation agent
        current_player_color: The color of the current player
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Formatted string representation of player resources
    """
    lines = ["[PLAYER RESOURCES]"]

    # Use public_state for all players' public information
    for color, player_data in public_state.players.items():
        color_name = _name_of(color)

        # For current player, use detailed inventory if provided
        if color == current_player_color and current_player_inventory is not None:
            # This is the current player - use detailed inventory
            resource_list = []
            if current_player_inventory.wood > 0:
                resource_list.append(f"WOOD: {current_player_inventory.wood}")
            if current_player_inventory.brick > 0:
                resource_list.append(f"BRICK: {current_player_inventory.brick}")
            if current_player_inventory.sheep > 0:
                resource_list.append(f"SHEEP: {current_player_inventory.sheep}")
            if current_player_inventory.wheat > 0:
                resource_list.append(f"WHEAT: {current_player_inventory.wheat}")
            if current_player_inventory.ore > 0:
                resource_list.append(f"ORE: {current_player_inventory.ore}")
            lines.append(f"- {color_name}: {', '.join(resource_list) if resource_list else 'No resources'}")
        else:
            # For other players, only show public information (hand count)
            hand_count = player_data.hand_resource_count
            lines.append(f"- {color_name}: {hand_count} resource cards (hidden)")

    return "\n".join(lines)


def get_player_dev_cards(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """Format player development card information for LLM consumption.

    Args:
        public_state: The public state object from Observation agent
        current_player_color: The color of the current player
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Formatted string representation of player development cards
    """
    lines = ["[PLAYER DEVELOPMENT CARDS]"]

    # Use public_state for all players' public information
    for color, player_data in public_state.players.items():
        color_name = _name_of(color)

        # For current player, use detailed inventory if provided
        if color == current_player_color and current_player_inventory is not None:
            # This is the current player - use detailed inventory
            card_list = []
            if current_player_inventory.knight > 0:
                card_list.append(f"KNIGHT: {current_player_inventory.knight}")
            if current_player_inventory.year_of_plenty > 0:
                card_list.append(f"YEAR_OF_PLENTY: {current_player_inventory.year_of_plenty}")
            if current_player_inventory.monopoly > 0:
                card_list.append(f"MONOPOLY: {current_player_inventory.monopoly}")
            if current_player_inventory.road_building > 0:
                card_list.append(f"ROAD_BUILDING: {current_player_inventory.road_building}")
            if current_player_inventory.victory_point > 0:
                card_list.append(f"VICTORY_POINT: {current_player_inventory.victory_point}")
            
            # Add played cards (public information)
            played_list = []
            if player_data.played_knight > 0:
                played_list.append(f"KNIGHT: {player_data.played_knight}")
            if player_data.played_year_of_plenty > 0:
                played_list.append(f"YEAR_OF_PLENTY: {player_data.played_year_of_plenty}")
            if player_data.played_monopoly > 0:
                played_list.append(f"MONOPOLY: {player_data.played_monopoly}")
            if player_data.played_road_building > 0:
                played_list.append(f"ROAD_BUILDING: {player_data.played_road_building}")
            if player_data.played_victory_point > 0:
                played_list.append(f"VICTORY_POINT: {player_data.played_victory_point}")
            
            # Combine held and played cards
            held_str = ', '.join(card_list) if card_list else 'No dev cards'
            played_str = ', '.join(played_list) if played_list else None
            
            if played_str:
                lines.append(f"- {color_name}: {held_str} (Played: {played_str})")
            else:
                lines.append(f"- {color_name}: {held_str}")
        else:
            # For other players, show played cards (public) and hidden count for held cards
            hand_count = player_data.hand_dev_count
            
            # Add played cards (public information)
            played_list = []
            if player_data.played_knight > 0:
                played_list.append(f"KNIGHT: {player_data.played_knight}")
            if player_data.played_year_of_plenty > 0:
                played_list.append(f"YEAR_OF_PLENTY: {player_data.played_year_of_plenty}")
            if player_data.played_monopoly > 0:
                played_list.append(f"MONOPOLY: {player_data.played_monopoly}")
            if player_data.played_road_building > 0:
                played_list.append(f"ROAD_BUILDING: {player_data.played_road_building}")
            if player_data.played_victory_point > 0:
                played_list.append(f"VICTORY_POINT: {player_data.played_victory_point}")
            
            played_str = ', '.join(played_list) if played_list else None
            
            if played_str:
                lines.append(f"- {color_name}: {hand_count} dev cards (hidden) (Played: {played_str})")
            else:
                lines.append(f"- {color_name}: {hand_count} dev cards (hidden)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consolidated per-player overview
# ---------------------------------------------------------------------------

def _format_resources_for_overview(player_data, inventory: Optional[Inventory], is_current: bool) -> str:
    """Return the resource fragment for one player in the consolidated view."""
    if is_current and inventory is not None:
        parts = []
        if inventory.wood > 0:
            parts.append(f"WOOD: {inventory.wood}")
        if inventory.brick > 0:
            parts.append(f"BRICK: {inventory.brick}")
        if inventory.sheep > 0:
            parts.append(f"SHEEP: {inventory.sheep}")
        if inventory.wheat > 0:
            parts.append(f"WHEAT: {inventory.wheat}")
        if inventory.ore > 0:
            parts.append(f"ORE: {inventory.ore}")
        return ", ".join(parts) if parts else "No resources"
    # Hidden for opponents / when inventory unavailable
    return f"{player_data.hand_resource_count} resource cards (hidden)"


def _format_dev_for_overview(player_data, inventory: Optional[Inventory], is_current: bool) -> str:
    """Return the dev-card fragment for one player in the consolidated view."""
    # Collect played (always public)
    played = []
    if player_data.played_knight > 0:
        played.append(f"KNIGHT: {player_data.played_knight}")
    if player_data.played_year_of_plenty > 0:
        played.append(f"YEAR_OF_PLENTY: {player_data.played_year_of_plenty}")
    if player_data.played_monopoly > 0:
        played.append(f"MONOPOLY: {player_data.played_monopoly}")
    if player_data.played_road_building > 0:
        played.append(f"ROAD_BUILDING: {player_data.played_road_building}")
    if player_data.played_victory_point > 0:
        played.append(f"VICTORY_POINT: {player_data.played_victory_point}")
    played_str = ", ".join(played) if played else None

    if is_current and inventory is not None:
        held = []
        if inventory.knight > 0:
            held.append(f"KNIGHT: {inventory.knight}")
        if inventory.year_of_plenty > 0:
            held.append(f"YEAR_OF_PLENTY: {inventory.year_of_plenty}")
        if inventory.monopoly > 0:
            held.append(f"MONOPOLY: {inventory.monopoly}")
        if inventory.road_building > 0:
            held.append(f"ROAD_BUILDING: {inventory.road_building}")
        if inventory.victory_point > 0:
            held.append(f"VICTORY_POINT: {inventory.victory_point}")
        held_str = ", ".join(held) if held else "No dev cards"
        if played_str:
            return f"{held_str} (Played: {played_str})"
        return held_str
    # Opponents / no inventory: hidden count + played
    hidden = f"{player_data.hand_dev_count} dev cards (hidden)"
    if played_str:
        return f"{hidden} (Played: {played_str})"
    return hidden


def _format_vp_for_overview(player_data, inventory: Optional[Inventory], is_current: bool) -> str:
    """Visible points (+ hidden VP if known from inventory). Returns the value part after ``VP: ``."""
    public = player_data.public_vps
    if is_current and inventory is not None:
        actual = inventory.actual_vps
        hidden = actual - public
        if hidden > 0:
            return f"{actual} ({public} visible + {hidden} hidden)"
        if hidden < 0:
            # Defensive: should not happen, but show actual
            return f"{actual} ({public} visible)"
        return f"{public}"
    return f"{public}"


def _format_road_for_overview(player_data, public_state: PublicState, color) -> str:
    """Road length + Longest Road badge."""
    length = player_data.longest_road_length
    has_longest = bool(player_data.has_road)
    if has_longest:
        # Longest Road is only meaningful at 5+, but show badge anyway
        if length >= 5:
            return f"{length} [Longest Road, +2 VP]"
        return f"{length} [Longest Road]"
    return f"{length}"


def _format_army_for_overview(player_data) -> str:
    """Army size (played knights) + Largest Army badge."""
    knights = player_data.played_knight
    has_army = bool(player_data.has_army)
    # Singular vs plural for display inside army fragment is handled by caller label;
    # we include the noun here for compactness.
    noun = "knight" if knights == 1 else "knights"
    base = f"{knights} {noun}"
    if has_army:
        return f"{base} [Largest Army, +2 VP]"
    return base


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _format_pieces_for_overview(player_data) -> str:
    """Available pieces remaining (e.g. '1 settlement, 4 cities, 12 roads left')."""
    s = player_data.settlements_left
    c = player_data.cities_left
    r = player_data.roads_left
    # e.g. "1 settlement, 4 cities, 12 roads remaining" with correct plural
    return (
        f"{s} {_plural(s, 'settlement', 'settlements')}, "
        f"{c} {_plural(c, 'city', 'cities')}, "
        f"{r} {_plural(r, 'road', 'roads')} left"
    )


def _format_ports_for_overview(board_player) -> str:
    """Ports controlled by a player (from occupancy)."""
    if board_player is None:
        return "None"
    ports: List[str] = []
    for b in board_player.settlements + board_player.cities:
        if b.port:
            ports.append(b.port)
    uniq = sorted(set(ports))
    return ", ".join(uniq) if uniq else "None"


def _format_pips_for_overview(board_player) -> str:
    """Pip production: total and per-resource (cities count double)."""
    if board_player is None or (not board_player.settlements and not board_player.cities):
        return "0"
    # Order matters for deterministic output: WOOD, BRICK, SHEEP, WHEAT, ORE
    resource_order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    resource_pips: Dict[str, int] = {r: 0 for r in resource_order}
    total = 0
    for b in board_player.settlements:
        total += b.total_pips
        for hx in b.adjacent_hexes:
            if hx.resource in resource_pips:
                resource_pips[hx.resource] += hx.pips
    for b in board_player.cities:
        total += b.total_pips * 2
        for hx in b.adjacent_hexes:
            if hx.resource in resource_pips:
                resource_pips[hx.resource] += hx.pips * 2
    if total == 0:
        return "0"
    parts = [f"{r}: {resource_pips[r]}" for r in resource_order if resource_pips[r] > 0]
    if parts:
        return f"{total} ({', '.join(parts)})"
    return f"{total}"


def get_players_summary(
    public_state: PublicState,
    current_player_color,
    current_player_inventory: Optional[Inventory] = None,
) -> str:
    """Consolidated per-player overview for LLM consumption.

    One block, one line (plus header) per player, showing in order:
    resources, dev cards, VP (visible + hidden if known), road length
    (+ Longest Road badge), army size (+ Largest Army badge), ports,
    pip production (total and per-resource, cities ×2), and available
    pieces. The current player is marked ``(YOU)`` and shows exact
    resource/dev counts when an inventory is supplied; opponents show
    hidden counts plus any public ``Played:`` detail. Ports and pips are
    derived from ``gather_board_occupancy_data`` so they match the board
    occupancy view.

    Args:
        public_state: The public snapshot from Observation.
        current_player_color: Color of the observer (to mark YOU and
            reveal exact hand when inventory is present).
        current_player_inventory: Optional private Inventory for the
            observer; enables exact resources/dev, and hidden-VP math
            (actual_vps - public_vps). When None, even the observer is
            shown as hidden (useful for tests).

    Returns:
        Multiline string beginning with ``[PLAYERS]`` and one ``- COLOR``
        line per player. Each line is ``Resources: … | Dev: … | VP: … |
        Roads: … | Army: … | Ports: … | Pips: … | Pieces: …`` so the LLM
        can scan player-by-player without joining two separate sections.

    Example:
        ``- RED (YOU): Resources: WOOD: 2 | Dev: KNIGHT: 1 (Played: KNIGHT: 1) | VP: 5 (3 visible + 2 hidden) | Roads: 3 | Army: 1 knight | Ports: 3:1 | Pips: 9 (SHEEP: 9) | Pieces: 3 settlements, 4 cities, 12 roads left``
    """
    # Lazily import to avoid circular dependency (board imports models only)
    from catan_llm.format.board import gather_board_occupancy_data

    occupancy = gather_board_occupancy_data(public_state)
    # PlayerBoardData is keyed by color name string
    occ_by_color: Dict[str, object] = {p.color: p for p in occupancy.players}

    lines = ["[PLAYERS]"]
    for color, player_data in public_state.players.items():
        color_name = _name_of(color)
        is_current = (color == current_player_color)
        tag = " (YOU)" if is_current else ""

        inv = current_player_inventory if is_current else None

        resources = _format_resources_for_overview(player_data, inv, is_current)
        dev = _format_dev_for_overview(player_data, inv, is_current)
        vp = _format_vp_for_overview(player_data, inv, is_current)
        road = _format_road_for_overview(player_data, public_state, color)
        army = _format_army_for_overview(player_data)
        board_player = occ_by_color.get(color_name)
        ports = _format_ports_for_overview(board_player)
        pips = _format_pips_for_overview(board_player)
        pieces = _format_pieces_for_overview(player_data)

        lines.append(
            f"- {color_name}{tag}: "
            f"Resources: {resources} | "
            f"Dev: {dev} | "
            f"VP: {vp} | "
            f"Roads: {road} | "
            f"Army: {army} | "
            f"Ports: {ports} | "
            f"Pips: {pips} | "
            f"Pieces: {pieces}"
        )
    return "\n".join(lines)


# Backwards-compatible alias — some callers may expect the singular form.
get_player_summary = get_players_summary
