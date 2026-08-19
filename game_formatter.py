"""
Game Formatter Module

Converts Catanatron game state and actions into LLM-friendly text formats.
Provides clean, structured representations of board state, player information,
and available actions for LLM consumption.

Now works with Observation agent's public_state and player inventory instead of
direct Game/State access or features for better information hiding.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from catanatron.models.enums import ActionType, ActionPrompt
from catanatron.models.public_state import PublicState, PublicBoard, PublicPlayer
from catanatron.models.inventory import Inventory



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
        total_pips = 0
        resource_pips = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}

        # Add settlement production (1x)
        for building in settlements:
            total_pips += building.total_pips
            for hex_info in building.adjacent_hexes:
                if hex_info.resource in resource_pips:
                    resource_pips[hex_info.resource] += hex_info.pips

        # Add city production (2x)
        for building in cities:
            total_pips += building.total_pips * 2  # Cities produce 2x
            for hex_info in building.adjacent_hexes:
                if hex_info.resource in resource_pips:
                    resource_pips[hex_info.resource] += hex_info.pips * 2  # Cities produce 2x

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

        # Convert BuildingInfo objects to strings for display (without port info)
        settlement_strings = []
        for building in settlements:
            hex_info_list = []
            for hex in building.adjacent_hexes:
                if hex.resource == "DESERT":
                    hex_info_list.append(f"(Tile {hex.tile_id}: DESERT)")
                else:
                    hex_info_list.append(f"(Tile {hex.tile_id}: {hex.roll if hex.roll else 'None'} {hex.resource} ({hex.pips} pips))")
            hex_info = ", ".join(hex_info_list)
            settlement_strings.append(f"Node {building.node_id}: {hex_info}, Total: {building.total_pips} pips")

        city_strings = []
        for building in cities:
            hex_info_list = []
            for hex in building.adjacent_hexes:
                if hex.resource == "DESERT":
                    hex_info_list.append(f"(Tile {hex.tile_id}: DESERT)")
                else:
                    hex_info_list.append(f"(Tile {hex.tile_id}: {hex.roll if hex.roll else 'None'} {hex.resource} ({hex.pips} pips))")
            hex_info = ", ".join(hex_info_list)
            city_strings.append(f"Node {building.node_id}: {hex_info}, Total: {building.total_pips} pips")

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
            lines.append(f"- {color_name}: {', '.join(card_list) if card_list else 'No dev cards'}")
        else:
            # For other players, only show public information (hand count)
            hand_count = player_data.hand_dev_count
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


# ============================================================================


