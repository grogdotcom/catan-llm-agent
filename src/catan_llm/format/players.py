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

from catan_llm.format.utils import _abbr_resource, _name_of


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
            parts.append(f"Wd:{inventory.wood}")
        if inventory.brick > 0:
            parts.append(f"Br:{inventory.brick}")
        if inventory.sheep > 0:
            parts.append(f"Sh:{inventory.sheep}")
        if inventory.wheat > 0:
            parts.append(f"Wh:{inventory.wheat}")
        if inventory.ore > 0:
            parts.append(f"Or:{inventory.ore}")
        return ", ".join(parts) if parts else "No resources"
    # Hidden for opponents / when inventory unavailable
    return f"{player_data.hand_resource_count}c hidden"


_DEV_ABBR = {
    "KNIGHT": "Kn",
    "YEAR_OF_PLENTY": "YOP",
    "MONOPOLY": "Mo",
    "ROAD_BUILDING": "RB",
    "VICTORY_POINT": "VP",
}


def _abbr_dev(name: str) -> str:
    return _DEV_ABBR.get(name, name)


def _format_dev_for_overview(player_data, inventory: Optional[Inventory], is_current: bool) -> str:
    """Return the dev-card fragment for one player in the consolidated view."""
    # Collect played (always public) — abbreviated
    played = []
    if player_data.played_knight > 0:
        played.append(f"Kn:{player_data.played_knight}")
    if player_data.played_year_of_plenty > 0:
        played.append(f"YOP:{player_data.played_year_of_plenty}")
    if player_data.played_monopoly > 0:
        played.append(f"Mo:{player_data.played_monopoly}")
    if player_data.played_road_building > 0:
        played.append(f"RB:{player_data.played_road_building}")
    if player_data.played_victory_point > 0:
        played.append(f"VP:{player_data.played_victory_point}")
    played_str = ", ".join(played) if played else None

    if is_current and inventory is not None:
        held = []
        if inventory.knight > 0:
            held.append(f"Kn:{inventory.knight}")
        if inventory.year_of_plenty > 0:
            held.append(f"YOP:{inventory.year_of_plenty}")
        if inventory.monopoly > 0:
            held.append(f"Mo:{inventory.monopoly}")
        if inventory.road_building > 0:
            held.append(f"RB:{inventory.road_building}")
        if inventory.victory_point > 0:
            held.append(f"VP:{inventory.victory_point}")
        held_str = ", ".join(held) if held else "No dev"
        if played_str:
            return f"{held_str} (Pd:{played_str})"
        return held_str
    # Opponents / no inventory: hidden count + played
    hidden = f"{player_data.hand_dev_count}d hidden"
    if played_str:
        return f"{hidden} (Pd:{played_str})"
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


def _format_pieces_compact(player_data) -> str:
    """Compact pieces: '5/4/15' (S/C/R)."""
    return f"{player_data.settlements_left}/{player_data.cities_left}/{player_data.roads_left}"


def _format_ports_for_overview(board_player) -> str:
    """Ports controlled by a player (from occupancy). Abbreviated."""
    if board_player is None:
        return "None"
    ports: List[str] = []
    for b in board_player.settlements + board_player.cities:
        if b.port:
            ports.append(_abbr_resource(b.port))
    # Normalize 3:1 stays "3:1"
    uniq = sorted(set(ports))
    return ", ".join(uniq) if uniq else "None"


def _format_pips_for_overview(board_player) -> str:
    """Pip production: total and per-resource (cities count double). Abbreviated."""
    if board_player is None or (not board_player.settlements and not board_player.cities):
        return "0"
    resource_order = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    abbr_order = ["Wd", "Br", "Sh", "Wh", "Or"]
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
    parts = [f"{abbr}:{resource_pips[res]}" for res, abbr in zip(resource_order, abbr_order) if resource_pips[res] > 0]
    if parts:
        return f"{total} ({', '.join(parts)})"
    return f"{total}"


def get_players_summary(
    public_state: PublicState,
    current_player_color,
    current_player_inventory: Optional[Inventory] = None,
    current_prompt=None,
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
        current_prompt: Optional current ActionPrompt. When it is an
            initial-setup prompt (BUILD_INITIAL_SETTLEMENT/ROAD) a
            compressed variant is used that omits uniform zero fields
            (Resources/Dev/VP/Roads/Army when all zero) and collapses
            identical players to one line, but still shows per-player
            Pips/Ports/Pieces — and Resources/Dev if any player has
            inventory (e.g. after 2nd settlement).

    Returns:
        Multiline string beginning with ``[PLAYERS]`` (or
        ``[PLAYERS - INITIAL SETUP]`` for the compressed variant) and
        one ``- COLOR`` line per player. Each line is ``Resources: … |
        Dev: … | VP: … | Roads: … | Army: … | Ports: … | Pips: … |
        Pieces: …`` so the LLM can scan player-by-player without
        joining two separate sections.

    Example:
        ``- RED (YOU): Resources: WOOD: 2 | Dev: KNIGHT: 1 (Played: KNIGHT: 1) | VP: 5 (3 visible + 2 hidden) | Roads: 3 | Army: 1 knight | Ports: 3:1 | Pips: 9 (SHEEP: 9) | Pieces: 3/4/12``
    """
    # Lazily import to avoid circular dependency (board imports models only)
    from catan_llm.format.board import gather_board_occupancy_data

    # Detect initial-setup prompts for compressed variant
    is_initial = False
    if current_prompt is not None:
        pname = getattr(current_prompt, "name", str(current_prompt))
        is_initial = pname in ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD")

    occupancy = gather_board_occupancy_data(public_state)
    # PlayerBoardData is keyed by color name string
    occ_by_color: Dict[str, object] = {p.color: p for p in occupancy.players}

    # --- Initial-setup compressed variant ---
    if is_initial:
        # Use compact pieces (5/4/15) and omit uniform zero fields
        # Check if any player has non-zero Resources/Dev/VP/Roads/Army
        has_any_resources = False
        has_any_dev = False
        has_any_vp = False
        has_any_roads = False
        has_any_army = False
        per_player_vals = []
        for color, player_data in public_state.players.items():
            is_current = (color == current_player_color)
            inv = current_player_inventory if is_current else None
            # Resources non-zero?
            if is_current and inv is not None:
                if any(getattr(inv, r, 0) > 0 for r in ("wood", "brick", "sheep", "wheat", "ore")):
                    has_any_resources = True
            else:
                if player_data.hand_resource_count > 0:
                    has_any_resources = True
            # Dev non-zero?
            if is_current and inv is not None:
                if any(getattr(inv, r, 0) > 0 for r in ("knight", "year_of_plenty", "monopoly", "road_building", "victory_point")):
                    has_any_dev = True
            if player_data.hand_dev_count > 0 or player_data.played_knight > 0 or player_data.played_year_of_plenty > 0 or player_data.played_monopoly > 0 or player_data.played_road_building > 0 or player_data.played_victory_point > 0:
                has_any_dev = True
            if player_data.public_vps != 0:
                has_any_vp = True
            if player_data.longest_road_length != 0 or player_data.has_road:
                has_any_roads = True
            if player_data.played_knight != 0 or player_data.has_army:
                has_any_army = True
            board_player = occ_by_color.get(_name_of(color))
            ports = _format_ports_for_overview(board_player)
            pips = _format_pips_for_overview(board_player)
            pieces = _format_pieces_compact(player_data)
            per_player_vals.append((color, _name_of(color), pips, ports, pieces))

        # If all players identical (start of game: 5/4/15, 0 pips, no ports) collapse to one line
        all_same = len(set((p, ports, pcs) for _, _, p, ports, pcs in per_player_vals)) == 1
        if all_same and not has_any_resources and not has_any_dev and not has_any_vp and not has_any_roads and not has_any_army:
            you_name = _name_of(current_player_color) if current_player_color is not None else "RED"
            # Use first player's pieces/pips/ports as representative
            _, _, pips0, ports0, pieces0 = per_player_vals[0] if per_player_vals else (None, None, "0", "None", "5/4/15")
            # Keep header distinct so tests can detect variant, but still starts with [PLAYERS
            return f"[PLAYERS] - INITIAL SETUP\nAll players start {pieces0} (S/C/R), 0 VP, 0 pips, no ports/resources — {you_name} (YOU) to place"

        # Otherwise per-player compact (only Pips/Ports/Pieces, plus any non-zero Resources/Dev/VP/Roads/Army if needed)
        lines = ["[PLAYERS] - INITIAL SETUP"]
        for color, player_data in public_state.players.items():
            color_name = _name_of(color)
            is_current = (color == current_player_color)
            tag = " (YOU)" if is_current else ""
            board_player = occ_by_color.get(color_name)
            ports = _format_ports_for_overview(board_player)
            pips = _format_pips_for_overview(board_player)
            pieces = _format_pieces_compact(player_data)
            parts = [f"Pips: {pips}", f"Ports: {ports}", f"Pieces: {pieces}"]
            # Include Resources/Dev/VP/Roads/Army only if any player has non-zero in that category
            if has_any_resources:
                inv = current_player_inventory if is_current else None
                resources = _format_resources_for_overview(player_data, inv, is_current)
                parts.insert(0, f"Resources: {resources}")
            if has_any_dev:
                inv = current_player_inventory if is_current else None
                dev = _format_dev_for_overview(player_data, inv, is_current)
                parts.insert(1 if has_any_resources else 0, f"Dev: {dev}")
            if has_any_vp:
                vp = _format_vp_for_overview(player_data, current_player_inventory if is_current else None, is_current)
                parts.append(f"VP: {vp}")
            if has_any_roads:
                road = _format_road_for_overview(player_data, public_state, color)
                parts.append(f"Roads: {road}")
            if has_any_army:
                army = _format_army_for_overview(player_data)
                parts.append(f"Army: {army}")
            lines.append(f"- {color_name}{tag}: " + " | ".join(parts))
        return "\n".join(lines)

    # --- Midgame (default) — keep full detail but compress Pieces to compact form ---
    # Hide zero Dev/Army when no player holds any (pure noise in early/midgame)
    has_any_dev = False
    has_any_army = False
    for color, player_data in public_state.players.items():
        is_current = (color == current_player_color)
        inv = current_player_inventory if is_current else None
        if is_current and inv is not None:
            if any(getattr(inv, r, 0) > 0 for r in ("knight", "year_of_plenty", "monopoly", "road_building", "victory_point")):
                has_any_dev = True
        if (
            player_data.hand_dev_count > 0
            or player_data.played_knight > 0
            or player_data.played_year_of_plenty > 0
            or player_data.played_monopoly > 0
            or player_data.played_road_building > 0
            or player_data.played_victory_point > 0
        ):
            has_any_dev = True
        if player_data.played_knight != 0 or bool(player_data.has_army):
            has_any_army = True

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
        pieces = _format_pieces_compact(player_data)

        # Assemble only non-zero Dev/Army when relevant
        parts = [f"Resources: {resources}"]
        if has_any_dev:
            parts.append(f"Dev: {dev}")
        parts.append(f"VP: {vp}")
        parts.append(f"Roads: {road}")
        if has_any_army:
            parts.append(f"Army: {army}")
        parts.extend([f"Ports: {ports}", f"Pips: {pips}", f"Pieces: {pieces}"])

        lines.append(f"- {color_name}{tag}: " + " | ".join(parts))
    return "\n".join(lines)


# Backwards-compatible alias — some callers may expect the singular form.
get_player_summary = get_players_summary
