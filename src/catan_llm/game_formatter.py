"""
Game Formatter Module

Converts Catanatron game state and actions into LLM-friendly text formats.
Provides clean, structured representations of board state, player information,
and available actions for LLM consumption.

Now works with Observation agent's public_state and player inventory instead of
direct Game/State access or features for better information hiding.
"""

import re
from typing import Dict, Any, List, Optional, Sequence, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
from catanatron.models.enums import Action, ActionType, ActionPrompt, ActionRecord, RESOURCES
from catanatron.models.public_state import PublicState, PublicBoard, PublicPlayer
from catanatron.models.inventory import Inventory
from catanatron.models.board import STATIC_GRAPH

# Initial placement is only BUILD_SETTLEMENT / BUILD_ROAD until the first
# non-setup action (typically ROLL). After that, those build types belong
# to normal turns.
_SETUP_ACTION_TYPES = frozenset({ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD})



@dataclass
class AdjacentHexInfo:
    """Data class for adjacent hex information."""
    resource: str
    roll: Optional[int]
    pips: int
    tile_id: Optional[int] = None


@dataclass
class BuildingInfo:
    """Data class for building (settlement or city) information."""
    node_id: int
    adjacent_hexes: List[AdjacentHexInfo]
    port: Optional[str] = None

    total_pips: int = field(init=False)

    def __post_init__(self):
        self.total_pips =  sum([hex.pips for hex in self.adjacent_hexes])


@dataclass
class PlayerBoardData:
    """Data class for player's board occupancy information."""
    color: str
    settlements: List[BuildingInfo]
    cities: List[BuildingInfo]
    roads: List[tuple]  # List of (node1, node2) tuples


@dataclass
class BoardOccupancyData:
    """Data class for board occupancy information."""
    players: List[PlayerBoardData]


def get_pip_count(roll_num):
    """Calculate pip count from roll number."""
    if roll_num is None:
        return 0
    pip_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    return pip_map.get(roll_num, 0)


def get_adjacent_hex_info(public_state: PublicState, node_id: int) -> tuple[List[AdjacentHexInfo], Optional[str]]:
    """Get adjacent hex information for a node including resources, rolls, pips, and port.

    Args:
        public_state: The public state object from Observation agent containing map information
        node_id: The node ID to get adjacent hex information for

    Returns:
        tuple: (adjacent_hexes, port_resource)
    """
    adjacent_hexes = []

    # Get adjacent tile IDs from public_state.board.map.adjacent_tiles
    # adjacent_tiles: Dict[NodeId, Tuple[int, ...]] - node_id -> tile ids touching it
    adjacent_tile_ids = public_state.board.map.adjacent_tiles.get(node_id, ())

    # Get tile information from public_state.board.map.tiles
    # tiles: Dict[int, Tuple[Optional[FastResource], Optional[int]]] - tile_id -> (resource, roll)
    tiles = public_state.board.map.tiles

    for tile_id in adjacent_tile_ids:
        resource, roll = tiles.get(tile_id, (None, None))

        if resource is not None:
            # This is a resource tile (not desert)
            resource_name = resource.name if hasattr(resource, 'name') else str(resource)
            pips = get_pip_count(roll)

            adjacent_hexes.append(AdjacentHexInfo(
                resource=resource_name,
                roll=roll,
                pips=pips,
                tile_id=tile_id
            ))

    # Check if this node is on a port
    # ports: Dict[int, Tuple[Optional[FastResource], Tuple[NodeId, NodeId]]] - port_id -> (resource, (node_a, node_b))
    port_resource = None
    for port_id, (resource, (node_a, node_b)) in public_state.board.map.ports.items():
        if node_id == node_a or node_id == node_b:
            # This node is on a port
            if resource is None:
                port_resource = "3:1"  # Generic 3:1 port
            else:
                port_resource = resource.name if hasattr(resource, 'name') else str(resource)
            break

    return adjacent_hexes, port_resource


def get_full_board_map(public_state: PublicState) -> str:
    """
    Returns a formatted text representation of all 19 hexes from the public_state.
    This is static throughout the game - robber position is not included.
    Now includes adjacent node IDs for each tile.

    Args:
        public_state: The public state object from Observation agent containing map information

    Returns:
        str: Formatted string representation of the board map
    """
    lines = ["[FULL BOARD MAP - 19 HEXES]"]

    # Extract tile information from public_state.board.map.tiles
    # tiles: Dict[int, Tuple[Optional[FastResource], Optional[int]]] - tile_id -> (resource, roll)
    tiles = public_state.board.map.tiles

    # Build tile_id -> node_ids mapping by inverting adjacent_tiles
    # adjacent_tiles: Dict[NodeId, Tuple[int, ...]] - node_id -> tile ids touching it
    tile_to_nodes = {}
    for node_id, tile_ids in public_state.board.map.adjacent_tiles.items():
        for tile_id in tile_ids:
            if tile_id not in tile_to_nodes:
                tile_to_nodes[tile_id] = []
            tile_to_nodes[tile_id].append(node_id)

    # Sort node IDs for each tile for deterministic output
    for tile_id in tile_to_nodes:
        tile_to_nodes[tile_id].sort()

    # Sort tile IDs for deterministic output
    for tile_id in sorted(tiles.keys()):
        resource, roll = tiles[tile_id]

        if resource is None:
            # Desert tile
            resource_pips_str = "DESERT"
        else:
            # Regular resource tile with roll number
            resource_name = resource.name if hasattr(resource, 'name') else str(resource)
            pips = get_pip_count(roll)
            resource_pips_str = f"{roll} {resource_name} ({pips} pips)"

        # Get adjacent node IDs for this tile
        adjacent_node_ids = tile_to_nodes.get(tile_id, [])
        nodes_str = f"Nodes: {adjacent_node_ids}" if adjacent_node_ids else "Nodes: []"

        lines.append(f"Tile {tile_id:>2}: {resource_pips_str}, {nodes_str}")

    return "\n".join(lines)


def gather_board_occupancy_data(public_state: PublicState) -> BoardOccupancyData:
    """Gather board occupancy data including buildings, roads, and basic hex info.

    Args:
        public_state: The public state object from Observation agent (dynamic state)

    Returns:
        BoardOccupancyData: Structured board occupancy information
    """
    players_data = []

    # Iterate over players in public_state
    for color in public_state.players.keys():
        color_name = color.name if hasattr(color, 'name') else str(color)
        settlements = []
        cities = []
        roads = []

        # Find player's buildings with basic hex info
        # public_state.board.buildings is Dict[NodeId, Tuple[Color, FastBuildingType]]
        for node_id, (building_color, building_type) in public_state.board.buildings.items():
            if building_color == color:
                adjacent_hexes, port_resource = get_adjacent_hex_info(public_state, node_id)

                building_info = BuildingInfo(
                    node_id=node_id,
                    adjacent_hexes=adjacent_hexes,
                    port=port_resource
                )

                # building_type might be a string or enum
                if str(building_type) == "CITY" or (hasattr(building_type, 'name') and building_type.name == "CITY"):
                    cities.append(building_info)
                else:
                    settlements.append(building_info)

        # Sort by node_id for consistency
        settlements.sort(key=lambda x: x.node_id)
        cities.sort(key=lambda x: x.node_id)

        # Find player's roads edge pairs
        # public_state.board.roads is Dict[EdgeId, Color]
        seen_roads = set()
        for edge_id, road_color in public_state.board.roads.items():
            if road_color == color:
                # edge_id is a tuple (node1, node2)
                # Normalize to always have smaller node first
                normalized_edge = tuple(sorted(edge_id))
                if normalized_edge not in seen_roads:
                    seen_roads.add(normalized_edge)
                    roads.append(normalized_edge)
        roads.sort()

        players_data.append(PlayerBoardData(
            color=color_name,
            settlements=settlements,
            cities=cities,
            roads=roads
        ))

    return BoardOccupancyData(
        players=players_data
    )


def calculate_blocked_production(robber_tile_id: int, players: List[PlayerBoardData]) -> Dict[str, str]:
    """Calculate blocked production for all players based on robber position.

    Args:
        robber_tile_id: The tile ID where the robber is located
        players: List of player board data to calculate blocked production

    Returns:
        Dict[str, str]: Dictionary mapping player colors to blocked production strings
    """
    blocked_production = {}

    for player_data in players:
        blocked_pips = 0
        blocked_resource_pips = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}

        # Check each building to see if it's adjacent to the robber tile
        for building in player_data.settlements + player_data.cities:
            for hex_info in building.adjacent_hexes:
                if hex_info.tile_id == robber_tile_id:
                    # This building is adjacent to the robber
                    multiplier = 2 if building in player_data.cities else 1
                    blocked_pips += hex_info.pips * multiplier
                    if hex_info.resource in blocked_resource_pips:
                        blocked_resource_pips[hex_info.resource] += hex_info.pips * multiplier

        if blocked_pips > 0:
            blocked_str = f"{blocked_pips} pips"
            blocked_production[player_data.color] = blocked_str

    return blocked_production


def format_robber_info(public_state: PublicState, players: List[PlayerBoardData]) -> str:
    """Format robber information including tile details and blocked production.

    Args:
        public_state: The public state object from Observation agent containing robber information
        players: List of player board data to calculate blocked production

    Returns:
        str: Formatted string representation of robber information
    """
    lines = []

    # Get robber tile ID directly from public_state
    robber_tile_id = public_state.board.robber_tile_id

    # Get tile information from public_state.board.map.tiles
    # tiles: Dict[int, Tuple[Optional[FastResource], Optional[int]]] - tile_id -> (resource, roll)
    robber_resource = None
    robber_roll = None
    robber_pips = 0

    if robber_tile_id is not None and robber_tile_id in public_state.board.map.tiles:
        resource, roll = public_state.board.map.tiles[robber_tile_id]
        robber_resource = resource
        robber_roll = roll
        robber_pips = get_pip_count(roll)

    # Calculate blocked production if we have a valid robber tile
    blocked_production = {}
    if robber_tile_id is not None:
        blocked_production = calculate_blocked_production(robber_tile_id, players)

    # Format robber information
    if robber_tile_id is not None:
        if robber_resource is None:
            tile_info = f"Tile {robber_tile_id}: DESERT"
        else:
            resource_name = robber_resource.name if hasattr(robber_resource, 'name') else str(robber_resource)
            tile_info = f"Tile {robber_tile_id}: {robber_roll} {resource_name} ({robber_pips} pips)"

        lines.append(f"ROBBER: Tile {robber_tile_id} - {tile_info}")

        # Add blocked production information
        if blocked_production:
            for color, blocked_str in sorted(blocked_production.items()):
                lines.append(f"  * Blocking {color}: {blocked_str}")
        else:
            lines.append(f"  * Blocking: None")
    else:
        # Fallback if we couldn't find tile information
        lines.append(f"ROBBER: Unknown position")
        lines.append(f"  * Tile info: Could not determine robber tile")
        lines.append(f"  * Blocking: None")

    return "\n".join(lines)


def _calculate_production(buildings: List[BuildingInfo], multiplier: int = 1) -> tuple[int, Dict[str, int]]:
    """Calculate production statistics from a list of buildings.

    Args:
        buildings: List of BuildingInfo objects
        multiplier: Production multiplier (1 for settlements, 2 for cities)

    Returns:
        tuple: (total_pips, resource_pips_dict)
    """
    total_pips = 0
    resource_pips = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}

    for building in buildings:
        total_pips += building.total_pips * multiplier
        for hex_info in building.adjacent_hexes:
            if hex_info.resource in resource_pips:
                resource_pips[hex_info.resource] += hex_info.pips * multiplier

    return total_pips, resource_pips


def _format_building_string(building: BuildingInfo) -> str:
    """Format a BuildingInfo object into a display string.

    Args:
        building: The BuildingInfo object to format

    Returns:
        str: Formatted string representation of the building
    """
    hex_info_list = []
    for hex in building.adjacent_hexes:
        if hex.resource == "DESERT":
            hex_info_list.append(f"(Tile {hex.tile_id}: DESERT)")
        else:
            hex_info_list.append(f"(Tile {hex.tile_id}: {hex.roll if hex.roll else 'None'} {hex.resource} ({hex.pips} pips))")
    hex_info = ", ".join(hex_info_list)
    return f"Node {building.node_id}: {hex_info}, Total: {building.total_pips} pips"


def format_board_occupancy_data(occupancy_data: BoardOccupancyData) -> str:
    """Format board occupancy data into a readable string.

    Args:
        occupancy_data: The board occupancy data to format

    Returns:
        str: Formatted string representation of board occupancy
    """
    lines = ["[CURRENT BOARD OCCUPANCY]"]

    # Sort players by color for consistent output
    sorted_players = sorted(occupancy_data.players, key=lambda p: p.color)

    for player_data in sorted_players:
        color_name = player_data.color
        settlements = player_data.settlements
        cities = player_data.cities
        roads = player_data.roads

        # Calculate production statistics
        settlement_pips, settlement_resource_pips = _calculate_production(settlements, multiplier=1)
        city_pips, city_resource_pips = _calculate_production(cities, multiplier=2)

        # Combine production from settlements and cities
        total_pips = settlement_pips + city_pips
        resource_pips = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}
        for resource in resource_pips:
            resource_pips[resource] = settlement_resource_pips[resource] + city_resource_pips[resource]

        # Format production string
        resource_strings = [f"{res}: {pips}" for res, pips in resource_pips.items() if pips > 0]
        production_str = f"Total: {total_pips} pips ({', '.join(resource_strings)})" if resource_strings else f"Total: {total_pips} pips"

        # Collect port information
        ports = []
        for building in settlements + cities:
            if building.port:
                ports.append(building.port)
        # Remove duplicates and sort
        ports = sorted(list(set(ports)))
        port_str = f", ".join(ports) if ports else "None"

        # Convert BuildingInfo objects to strings for display
        settlement_strings = [_format_building_string(building) for building in settlements]
        city_strings = [_format_building_string(building) for building in cities]

        # Convert road tuples to strings
        road_strings = [f"({n1}, {n2})" for n1, n2 in roads]

        lines.append(f"- {color_name}: {production_str}")
        lines.append(f"  * Ports: {port_str}")
        lines.append(f"  * Settlements: [{', '.join(settlement_strings) if settlement_strings else 'None'}]")
        lines.append(f"  * Cities (x2 production): [{', '.join(city_strings) if city_strings else 'None'}]")
        lines.append(f"  * Roads: Edges [{', '.join(road_strings) if road_strings else 'None'}]")

    return "\n".join(lines)


def get_board_occupancy(public_state: PublicState) -> str:
    """
    Extracts only the dynamic buildings, roads, and player claims on the board.
    For each settlement and city, includes basic hex information (resource, roll, port).

    Args:
        public_state: The public state object from Observation agent (dynamic state)

    Returns:
        str: Formatted string representation of board occupancy
    """
    occupancy_data = gather_board_occupancy_data(public_state)
    return format_board_occupancy_data(occupancy_data)


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
        color_name = color.name if hasattr(color, 'name') else str(color)

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
        color_name = color.name if hasattr(color, 'name') else str(color)

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


def get_game_state_summary(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """
    Create a comprehensive game state summary for LLM consumption.
    Includes board map, occupancy, resources, and development cards.

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
    sections.append(get_player_resources(public_state, current_player_color, current_player_inventory))
    sections.append("\n")
    sections.append(get_player_dev_cards(public_state, current_player_color, current_player_inventory))

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
            grouped_actions[action_type.name].append(str(action.coordinate))

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
                hex_coord = action.coordinate
                victim = action.kwargs.get('victim_color', 'NONE')
                grouped_actions["MOVE_ROBBER"].append(f"Hex {hex_coord} (Victim: {victim})")
            except AttributeError:
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


def _name_of(value: Any) -> str:
    """Return a stable display name for enums/colors/resources."""
    if value is None:
        return "None"
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _format_resource_counts(counts: Sequence[Any], resources: Sequence[str] = RESOURCES) -> str:
    """Format parallel resource counts as '2 WOOD, 1 BRICK' (skip zeros)."""
    parts = []
    for resource, count in zip(resources, counts):
        if count:
            parts.append(f"{count} {_name_of(resource)}")
    return ", ".join(parts) if parts else "nothing"


def _format_trade_offer_value(value: Sequence[Any]) -> str:
    """Format an OFFER/ACCEPT/REJECT 10-tuple as offered -> asking."""
    offered = _format_resource_counts(value[:5])
    asking = _format_resource_counts(value[5:10])
    return f"offers [{offered}] for [{asking}]"


def _format_maritime_trade_value(value: Sequence[Any]) -> str:
    """Format a MARITIME_TRADE 5-tuple (given..., received)."""
    giving = [r for r in value[:4] if r is not None]
    receiving = value[4]
    give_str = ", ".join(_name_of(r) for r in giving) if giving else "nothing"
    return f"gives [{give_str}] to bank for {_name_of(receiving)}"


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

    prompt_parts.append(format_public_history_window(public_history, window_size=history_window_size))
    prompt_parts.append("\n")

    prompt_parts.append(summarize_catan_actions(playable_actions))
    prompt_parts.append("\n")

    prompt_parts.append("[DECISION REQUIRED]")
    prompt_parts.append("Select the best action from the available options above.")

    return "\n".join(prompt_parts)


# ============================================================================
# Playable-action formatting and compound-move planning
# ============================================================================
#
# The LLM-facing view of a decision is a numbered list of "moves". A move is a
# sequence of engine actions that together form one coherent decision: the LLM
# picks a single move (by its stable index number) and the agent then drives
# every engine prompt of that move without re-consulting the LLM.
#
# Roads are represented as sorted node pairs ``(n1, n2)`` (matching the board
# occupancy data). Legal road edges are derived from the player's current roads
# and settlements against the map's static graph, so compound moves that include
# road placements (initial settlement + road, Road Building + two roads) bundle
# the concrete edges the LLM wants instead of auto-completing them.

AUTO_ROAD = "AUTO_ROAD"
"""Fallback sentinel token in a Move's action list.

Means: "resolve a legal BUILD_ROAD from the current prompt's playable actions".
Only used when no public state is available to enumerate the concrete road
edges upfront. With a public state, initial-placement roads and the Road
Building card's two roads are bundled concretely instead (each as a sorted
``(n1, n2)`` node pair), so the LLM picks the exact roads it wants.
"""


@dataclass
class Move:
    """A single LLM-choosable move: one or more engine actions to execute.

    ``actions`` lists the exact engine Actions to return in order; a string
    entry is a sentinel (see ``AUTO_ROAD``) resolved from the live prompt's
    playable actions at execution time. The first entry is returned immediately
    and the rest are queued, so compound moves (Knight + robber move, Road
    Building + two roads, initial settlement + road) are decided once.
    """

    label: str
    actions: List[Union[Action, str]]


def _node_pip_total(public_state: Optional[PublicState], node_id: int) -> int:
    """Total pips of the resource tiles touching a node (0 for desert/sea)."""
    if public_state is None:
        return 0
    total = 0
    for tile_id in public_state.board.map.adjacent_tiles.get(node_id, ()):
        resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
        if resource is not None:
            total += get_pip_count(roll)
    return total


def _format_coordinate(coordinate) -> str:
    """Render a cube coordinate as a compact string."""
    if coordinate is None:
        return "(unknown)"
    return f"({coordinate[0]}, {coordinate[1]}, {coordinate[2]})"


def _coordinate_tile_label(public_state: Optional[PublicState], coordinate) -> str:
    """Render a coordinate as its board-map tile ID (falls back to the raw
    coordinate when the public map is unavailable)."""
    if public_state is None or coordinate is None:
        return _format_coordinate(coordinate)
    for tile_id, coord in public_state.board.map.tile_coordinates.items():
        if coord == coordinate:
            return f"Tile {tile_id}"
    return _format_coordinate(coordinate)


def _label_action(action: Action, public_state: Optional[PublicState] = None) -> str:
    """A concise human description of a single action worth choosing."""
    kind = action.action_type
    value = action.value
    name = getattr(kind, "name", str(kind))

    if kind == ActionType.ROLL:
        return "Roll the dice"
    if kind == ActionType.END_TURN:
        return "End turn"
    if kind == ActionType.BUILD_ROAD:
        return f"Build road on edge {tuple(sorted(value))}"
    if kind == ActionType.BUILD_SETTLEMENT:
        return f"Build settlement at node {value}"
    if kind == ActionType.BUILD_CITY:
        return f"Build city at node {value}"
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return "Buy a development card"
    if kind == ActionType.PLAY_KNIGHT_CARD:
        return "Play Knight (then move the robber)"
    if kind == ActionType.PLAY_YEAR_OF_PLENTY:
        cards = ", ".join(_name_of(r) for r in value)
        return f"Play Year of Plenty: take {cards}"
    if kind == ActionType.PLAY_MONOPOLY:
        return f"Play Monopoly: steal all {_name_of(value)}"
    if kind == ActionType.PLAY_ROAD_BUILDING:
        return "Play Road Building (then build two roads)"
    if kind == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        coord_str = _coordinate_tile_label(public_state, coordinate)
        if victim is None:
            return f"Move robber to {coord_str} (no steal)"
        return f"Move robber to {coord_str} and steal from {_name_of(victim)}"
    if kind == ActionType.DISCARD_RESOURCE:
        return f"Discard one {_name_of(value)}"
    if kind == ActionType.MARITIME_TRADE:
        return f"Maritime trade: {_format_maritime_trade_value(value)}"
    if kind == ActionType.OFFER_TRADE:
        return f"Offer trade: {_format_trade_offer_value(value)}"
    if kind == ActionType.ACCEPT_TRADE:
        return f"Accept trade: {_format_trade_offer_value(value)}"
    if kind == ActionType.REJECT_TRADE:
        return f"Reject trade: {_format_trade_offer_value(value)}"
    if kind == ActionType.CONFIRM_TRADE:
        trade_part = _format_trade_offer_value(value[:10])
        acceptor = _name_of(value[10]) if len(value) > 10 else "unknown"
        return f"Confirm trade with {acceptor}: {trade_part}"
    if kind == ActionType.CANCEL_TRADE:
        return "Cancel trade"
    return f"{name}: value={value!r}"


def _knight_robber_followups(public_state: PublicState, color) -> List[Tuple]:
    """Legal (coordinate, victim_or_None) MOVE_ROBBER targets after a Knight.

    Mirrors the engine's ``robber_possibilities`` for a non-friendly-robber game,
    derived entirely from public data: every land tile except the current robber
    tile, stealing from any enemy holding at least one card. Kept in exact parity
    so a bundled Knight move is accepted by the engine on the follow-up prompt.
    """
    map_data = public_state.board.map
    robber_coordinate = map_data.tile_coordinates.get(public_state.board.robber_tile_id)

    tiles_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_id, tile_ids in map_data.adjacent_tiles.items():
        for tile_id in tile_ids:
            tiles_to_nodes[tile_id].append(node_id)

    targets = []
    for tile_id in sorted(map_data.tile_coordinates):
        coordinate = map_data.tile_coordinates[tile_id]
        if coordinate == robber_coordinate:
            continue

        victims = set()
        for node_id in tiles_to_nodes.get(tile_id, ()):
            building = public_state.board.buildings.get(node_id)
            if building is None:
                continue
            owner, _ = building
            if owner != color and public_state.players[owner].hand_resource_count >= 1:
                victims.add(owner)

        if victims:
            for victim in sorted(victims, key=lambda c: getattr(c, "name", str(c))):
                targets.append((coordinate, victim))
        else:
            targets.append((coordinate, None))
    return targets


def _knight_moves(knight_action: Action, public_state: PublicState) -> List[Move]:
    """Expand one PLAY_KNIGHT_CARD into bundled Knight + MOVE_ROBBER moves.

    Each bundle is a single LLM choice that includes where to move the robber
    and who to steal from, so the agent never decides the knight and the robber
    in disjoint prompts.
    """
    color = knight_action.color
    moves = []
    for coordinate, victim in _knight_robber_followups(public_state, color):
        followup = Action(color, ActionType.MOVE_ROBBER, (coordinate, victim))
        tile_str = _coordinate_tile_label(public_state, coordinate)
        if victim is None:
            label = f"Play Knight -> move robber to {tile_str} (no steal)"
        else:
            label = f"Play Knight -> move robber to {tile_str} and steal from {_name_of(victim)}"
        moves.append(Move(label=label, actions=[knight_action, followup]))
    return moves


def _own_network_nodes(public_state: PublicState, color) -> Set[int]:
    """Nodes of the player's road/settlement network.

    Mirrors the engine's connected-component nodes: own buildings plus the
    endpoints of own roads, minus nodes occupied by an enemy settlement/city
    (the network cannot pass through or build out of an enemy node).
    """
    board = public_state.board
    own_buildings = {n for n, (owner, _) in board.buildings.items() if owner == color}
    enemy_buildings = {n for n, (owner, _) in board.buildings.items() if owner != color}
    endpoints = set()
    for edge, owner in board.roads.items():
        if owner == color:
            endpoints.update(edge)
    return (own_buildings | endpoints) - enemy_buildings


def _land_edges_from(public_state: PublicState, color, nodes) -> List[Tuple[int, int]]:
    """Sorted list of unowned land edges touching any of ``nodes``.

    Mirrors the engine's ``Board.buildable_edges`` for a player whose network is
    ``nodes``: every static-graph edge incident to a non-enemy network node
    whose endpoints are land nodes and which no one owns yet. Enemy nodes are
    excluded from the originating set because a network can never extend out of
    an enemy settlement/city (roads may only run up to it).
    """
    land = public_state.board.map.land_nodes
    owned = set(public_state.board.roads.keys())  # keys are already sorted pairs
    nodes = set(nodes) - {
        n for n, (owner, _) in public_state.board.buildings.items() if owner != color
    }
    edges = set()
    for node in nodes:
        for neighbor in STATIC_GRAPH.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            if edge in owned or node not in land or neighbor not in land:
                continue
            edges.add(edge)
    return sorted(edges)


def _road_building_moves(play_card: Action, public_state: PublicState) -> List[Move]:
    """Expand PLAY_ROAD_BUILDING into concrete bundles of both road edges.

    Emits one move per unordered pair of roads, so building two disconnected
    edges "first" in either order is a single move (the resulting board is
    identical). Roads are shown as sorted ``(n1, n2)`` node pairs. When only a
    single road is possible the bundle has just one road.
    """
    color = play_card.color
    base_network = _own_network_nodes(public_state, color)
    first_roads = _land_edges_from(public_state, color, base_network)

    moves = []
    seen_pairs = set()
    for first in first_roads:
        second_network = base_network | set(first)
        seconds = [
            e
            for e in _land_edges_from(public_state, color, second_network)
            if e != first
        ]
        if not seconds:
            label = f"Play Road Building -> build road {first}"
            moves.append(
                Move(
                    label=label,
                    actions=[play_card, Action(color, ActionType.BUILD_ROAD, first)],
                )
            )
        else:
            for second in seconds:
                pair = frozenset((first, second))
                if pair in seen_pairs:
                    continue  # the reverse order is the same move
                seen_pairs.add(pair)
                road_a, road_b = sorted((first, second))
                label = f"Play Road Building -> build roads {road_a} and {road_b}"
                moves.append(
                    Move(
                        label=label,
                        actions=[
                            play_card,
                            Action(color, ActionType.BUILD_ROAD, first),
                            Action(color, ActionType.BUILD_ROAD, second),
                        ],
                    )
                )
    return moves


def _setup_settlement_moves(settle: Action, public_state: PublicState) -> List[Move]:
    """Expand an initial-placement settlement into concrete settlement + road moves.

    Each settlement node is bundled with every legal road edge incident to it,
    so the LLM chooses the road as part of the same initial-placement move.
    """
    color = settle.color
    node = settle.value
    road_options = _land_edges_from(public_state, color, {node})
    if not road_options:
        # Degenerate safety net: no road is legal from this node.
        return [Move(label=_label_action(settle, public_state), actions=[settle])]
    return [
        Move(
            label=f"Build settlement at node {node} -> build road {edge}",
            actions=[settle, Action(color, ActionType.BUILD_ROAD, edge)],
        )
        for edge in road_options
    ]


def build_moves(playable_actions: Sequence[Action], observation=None) -> List[Move]:
    """Build the LLM-choosable moves for a set of playable engine actions.

    Args:
        playable_actions: The engine's legal actions for the current prompt.
        observation: Optional Observation. When provided, its public_state and
            current_prompt drive compound-move expansion:
            - PLAY_KNIGHT_CARD is expanded into one move per (robber tile,
              steal victim) option, so playing the card and moving the robber
              are decided together.
            - An initial settlement (BUILD_INITIAL_SETTLEMENT prompt) is
              bundled with every legal road edge incident to that node, so the
              placement is decided as one move.
            - PLAY_ROAD_BUILDING is bundled with each legal (first road, second
              road) pair, so both free roads are chosen as part of the move.
            Year of Plenty and Monopoly already carry their resource parameters
            in a single action and are formatted as one move.
        Without an observation the road-carrying moves degrade to an annotated
        opener that the agent completes from the live prompt (see AUTO_ROAD).

    Returns:
        List of Moves; the LLM selects exactly one by index.
    """
    public_state = getattr(observation, "public_state", None)
    current_prompt = getattr(observation, "current_prompt", None)

    if not playable_actions:
        return []

    moves: List[Move] = []
    for action in playable_actions:
        kind = action.action_type

        if kind == ActionType.PLAY_KNIGHT_CARD and public_state is not None:
            moves.extend(_knight_moves(action, public_state))
        elif kind == ActionType.PLAY_ROAD_BUILDING and public_state is not None:
            moves.extend(_road_building_moves(action, public_state))
        elif kind == ActionType.PLAY_ROAD_BUILDING:
            label = "Play Road Building -> then build two roads"
            moves.append(Move(label=label, actions=[action, AUTO_ROAD, AUTO_ROAD]))
        elif (
            kind == ActionType.BUILD_SETTLEMENT
            and current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
            and public_state is not None
        ):
            moves.extend(_setup_settlement_moves(action, public_state))
        elif (
            kind == ActionType.BUILD_SETTLEMENT
            and current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
        ):
            label = f"{_label_action(action, public_state)} -> then place your initial road"
            moves.append(Move(label=label, actions=[action, AUTO_ROAD]))
        else:
            moves.append(Move(label=_label_action(action, public_state), actions=[action]))
    return moves


def format_moves(moves: Sequence[Move], observation=None) -> str:
    """Render moves as a numbered, LLM-readable list.

    The list index is the stable handle the LLM returns; see ``parse_move``.
    """
    current_prompt = getattr(observation, "current_prompt", None)
    lines = ["[PLAYABLE MOVES]"]
    if current_prompt is not None:
        phase = getattr(current_prompt, "name", str(current_prompt))
        lines.append(f"[PHASE: {phase}]")
    if not moves:
        lines.append("  (no moves available)")
        return "\n".join(lines)
    for i, move in enumerate(moves, start=1):
        lines.append(f"{i}. {move.label}")
    return "\n".join(lines)


def format_playable_actions(playable_actions: Sequence[Action], observation=None) -> str:
    """Convenience wrapper: build moves for ``playable_actions`` and format them.

    For agents that need the moves back (to map the LLM's chosen index to an
    Action), use ``build_moves`` + ``parse_move`` instead.
    """
    return format_moves(build_moves(playable_actions, observation), observation=observation)


def parse_move(response, moves: Sequence[Move]) -> Move:
    """Convert an LLM's response (a stable move index) into the chosen Move.

    Accepts an int, a bare number, a bracketed number like ``[3]``, or a
    numbered line like ``3. Build city at node 10``.
    """
    if isinstance(response, int):
        index = response
    else:
        match = re.match(r"\s*\[?(\d+)\]?", str(response))
        if match is None:
            raise ValueError(f"Cannot parse move index from response: {response!r}")
        index = int(match.group(1))

    if not 1 <= index <= len(moves):
        raise ValueError(f"Move index {index} out of range (1..{len(moves)})")
    return moves[index - 1]


def pick_auto_road(playable_actions: Sequence[Action], public_state=None) -> Optional[Action]:
    """Pick a legal BUILD_ROAD from the current prompt's playable actions.

    Used to complete moves whose road placement cannot be bundled upfront
    (initial-placement road, Road Building card roads). Scores each legal edge
    by the pip value of both endpoints so the road extends toward productive
    terrain; ties break deterministically by edge order.
    """
    roads = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
    if not roads:
        return None

    def score(action: Action) -> Tuple[int, Tuple]:
        edge = tuple(sorted(action.value))
        return _node_pip_total(public_state, edge[0]) + _node_pip_total(public_state, edge[1]), edge

    return max(roads, key=score)


# ============================================================================


