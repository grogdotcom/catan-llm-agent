"""
Board formatting — map, occupancy, and robber detail.

All functions here take a PublicState (from Observation) and produce
LLM-friendly text or structured data about the static board and dynamic
occupancy.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from catanatron.models.public_state import PublicState

from catan_llm.format.models import (
    AdjacentHexInfo,
    BoardOccupancyData,
    BuildingInfo,
    PlayerBoardData,
)
from catan_llm.format.utils import get_pip_count


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
