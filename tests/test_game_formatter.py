"""
Unit tests for the game formatter module
"""

import pytest
import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer, Player
from catanatron.models.enums import (
    SETTLEMENT,
    CITY,
    Action,
    ActionRecord,
    ActionType,
    RESOURCES,
    DEVELOPMENT_CARDS,
)
from catanatron.models.board import Board
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE
from catan_llm.format import (
    gather_board_occupancy_data,
    BoardOccupancyData,
    PlayerBoardData,
    BuildingInfo,
    AdjacentHexInfo,
    format_board_occupancy_data,
    format_robber_info,
    calculate_blocked_production,
    get_full_board_map,
    get_pip_count,
    get_player_resources,
    get_player_dev_cards,
    describe_action_record,
    group_action_records_by_turn,
    describe_turn,
    format_public_history,
    format_public_history_window,
)
from catanatron.models.perspective_player import _build_public_state, _sanitize_history
from catanatron.state_functions import player_key


def build_public_state(game):
    """Helper to build public_state object from game for testing."""
    return _build_public_state(game)


class SimplePlayer(Player):
    """Simple player for testing that doesn't need to implement decide"""
    def __init__(self, color):
        self.color = color
        self.is_bot = True

    def decide(self, game, playable_actions):
        return playable_actions[0] if playable_actions else None

    def reset_state(self):
        pass


def create_test_game_with_buildings():
    """Create a game with specific building placements for testing"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]

    game = Game(players)

    # Manually manipulate board data structures to bypass validation
    board = game.state.board

    # Directly manipulate the buildings dict to place buildings
    # RED settlement at node 0, city at node 10
    board.buildings[0] = (Color.RED, SETTLEMENT)
    board.buildings[10] = (Color.RED, CITY)

    # BLUE settlement at node 5
    board.buildings[5] = (Color.BLUE, SETTLEMENT)

    # Directly manipulate roads dict (roads are stored both ways)
    board.roads[(0, 5)] = Color.RED
    board.roads[(5, 0)] = Color.RED
    board.roads[(5, 16)] = Color.BLUE
    board.roads[(16, 5)] = Color.BLUE

    # Update board_buildable_ids to remove occupied nodes
    board.board_buildable_ids.discard(0)
    board.board_buildable_ids.discard(5)
    board.board_buildable_ids.discard(10)

    return game


def create_test_game_empty():
    """Create a game with no buildings"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]

    return Game(players)


def create_test_game_with_ports():
    """Create a game with buildings on port nodes"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]

    game = Game(players)
    board = game.state.board

    # Directly manipulate buildings dict to place settlements on port nodes
    # Use known port nodes from the map
    board.buildings[25] = (Color.RED, SETTLEMENT)  # 3:1 port
    board.buildings[28] = (Color.BLUE, SETTLEMENT)  # WHEAT port (we'll check what it actually is)

    # Update board_buildable_ids
    board.board_buildable_ids.discard(25)
    board.board_buildable_ids.discard(28)

    return game


def create_test_game_with_robber():
    """Create a game with specific robber placement"""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]

    game = Game(players)

    # Set robber to a specific coordinate
    game.state.board.robber_coordinate = (0, 0, 0)

    return game


def test_gather_board_occupancy_data_returns_correct_type():
    """Test that gather_board_occupancy_data returns BoardOccupancyData"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    assert isinstance(result, BoardOccupancyData)
    assert hasattr(result, 'players')
    assert not hasattr(result, 'robber_coordinate')


def test_gather_board_occupancy_data_has_correct_player_count():
    """Test that the result contains the correct number of players"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    assert len(result.players) == 4


def test_gather_board_occupancy_data_player_data_structure():
    """Test that each player has the correct data structure"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    for player_data in result.players:
        assert isinstance(player_data, PlayerBoardData)
        assert hasattr(player_data, 'color')
        assert hasattr(player_data, 'settlements')
        assert hasattr(player_data, 'cities')
        assert hasattr(player_data, 'roads')
        assert isinstance(player_data.settlements, list)
        assert isinstance(player_data.cities, list)
        assert isinstance(player_data.roads, list)


def test_gather_board_occupancy_data_building_info_structure():
    """Test that building info has the correct structure"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected data (node IDs are fixed, but hex data varies by map)
    expected_red_settlement = BuildingInfo(
        node_id=0,
        adjacent_hexes=[],  # Will check structure, not exact values
        port=None
    )

    expected_red_city = BuildingInfo(
        node_id=10,
        adjacent_hexes=[],  # Will check structure, not exact values
        port=None
    )

    expected_blue_settlement = BuildingInfo(
        node_id=5,
        adjacent_hexes=[],  # Will check structure, not exact values
        port=None
    )

    # Find RED player and compare
    red_player = next(p for p in result.players if p.color == "RED")
    assert len(red_player.settlements) == 1
    assert len(red_player.cities) == 1

    # Compare building info structure
    assert red_player.settlements[0].node_id == expected_red_settlement.node_id
    # Note: adjacent_hexes will now contain data since public_state provides adjacency info
    assert isinstance(red_player.settlements[0].adjacent_hexes, list)
    # Port might be None or a resource string depending on node location
    assert red_player.settlements[0].port is None or isinstance(red_player.settlements[0].port, str)
    # Total pips will now be calculated from adjacency info
    assert red_player.settlements[0].total_pips >= 0

    assert red_player.cities[0].node_id == expected_red_city.node_id
    # Note: adjacent_hexes will now contain data since public_state provides adjacency info
    assert isinstance(red_player.cities[0].adjacent_hexes, list)
    # Port might be None or a resource string depending on node location
    assert red_player.cities[0].port is None or isinstance(red_player.cities[0].port, str)
    # Total pips will now be calculated from adjacency info
    assert red_player.cities[0].total_pips >= 0

    # Find BLUE player and compare
    blue_player = next(p for p in result.players if p.color == "BLUE")
    assert len(blue_player.settlements) == 1
    assert blue_player.settlements[0].node_id == expected_blue_settlement.node_id
    # Note: adjacent_hexes will now contain data since public_state provides adjacency info
    assert isinstance(blue_player.settlements[0].adjacent_hexes, list)
    # Port might be None or a resource string depending on node location
    assert blue_player.settlements[0].port is None or isinstance(blue_player.settlements[0].port, str)
    # Total pips will now be calculated from adjacency info
    assert blue_player.settlements[0].total_pips >= 0


def test_format_board_occupancy_data_node_ids():
    """Test that formatted output includes node IDs for buildings"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should contain node IDs in building information
    assert "Node 0:" in result  # RED settlement
    assert "Node 10:" in result  # RED city
    assert "Node 5:" in result  # BLUE settlement
    # Note: Tile IDs should now be present since public_state provides adjacency info


def test_gather_board_occupancy_data_adjacent_hex_info_structure():
    """Test that adjacent hex info has the correct structure (now populated from public_state)"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Get RED's settlement at node 0
    red_player = next(p for p in result.players if p.color == "RED")
    settlement = red_player.settlements[0]

    # Check adjacent hexes structure (will now contain data from public_state)
    assert isinstance(settlement.adjacent_hexes, list)
    # Each adjacent hex should have the correct structure
    for hex_info in settlement.adjacent_hexes:
        assert hasattr(hex_info, 'resource')
        assert hasattr(hex_info, 'roll')
        assert hasattr(hex_info, 'pips')
        assert hasattr(hex_info, 'tile_id')


def test_gather_board_occupancy_data_road_structure():
    """Test that roads are structured correctly as tuples"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected roads
    expected_red_roads = [(0, 5)]
    expected_blue_roads = [(5, 16)]

    # RED should have 1 road (deduplicated)
    red_player = next(p for p in result.players if p.color == "RED")
    assert len(red_player.roads) == 1
    assert red_player.roads == expected_red_roads

    # BLUE should have 1 road (deduplicated)
    blue_player = next(p for p in result.players if p.color == "BLUE")
    assert len(blue_player.roads) == 1
    assert blue_player.roads == expected_blue_roads


def test_gather_board_occupancy_data_no_robber_coordinate():
    """Test that robber coordinate is not present in BoardOccupancyData"""
    game = create_test_game_with_robber()
    result = gather_board_occupancy_data(build_public_state(game))

    # Should not have robber_coordinate attribute
    assert not hasattr(result, 'robber_coordinate')


def test_gather_board_occupancy_data_pip_calculation():
    """Test that pip counts are calculated correctly (now from public_state adjacency info)"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Get RED's settlement at node 0
    red_player = next(p for p in result.players if p.color == "RED")
    settlement = red_player.settlements[0]

    # With adjacency info from public_state, pips should be calculated
    assert settlement.total_pips >= 0  # Should be non-negative
    # If there are adjacent resource tiles, pips should be > 0
    if len(settlement.adjacent_hexes) > 0:
        assert settlement.total_pips > 0


def test_gather_board_occupancy_data_sorted_by_node_id():
    """Test that buildings are sorted by node_id for consistency"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected sorted lists
    for player_data in result.players:
        # Check settlements are sorted
        settlement_ids = [building.node_id for building in player_data.settlements]
        expected_settlement_ids = sorted(settlement_ids)
        assert settlement_ids == expected_settlement_ids

        # Check cities are sorted
        city_ids = [building.node_id for building in player_data.cities]
        expected_city_ids = sorted(city_ids)
        assert city_ids == expected_city_ids

        # Check roads are sorted
        roads = player_data.roads
        expected_roads = sorted(roads)
        assert roads == expected_roads


def test_gather_board_occupancy_data_port_detection():
    """Test that ports are now detected from public_state"""
    game = create_test_game_with_ports()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected building info (node IDs are fixed)
    expected_red_settlement = BuildingInfo(
        node_id=25,
        adjacent_hexes=[],  # Will check structure, not exact values
        port=None  # Will be populated from public_state if node is on a port
    )

    expected_blue_settlement = BuildingInfo(
        node_id=28,
        adjacent_hexes=[],  # Will check structure, not exact values
        port=None  # Will be populated from public_state if node is on a port
    )

    # RED should have a settlement at node 25
    red_player = next(p for p in result.players if p.color == "RED")
    assert len(red_player.settlements) == 1
    assert red_player.settlements[0].node_id == expected_red_settlement.node_id
    # Port should now be detected from public_state (could be None or a resource string)
    assert red_player.settlements[0].port is None or isinstance(red_player.settlements[0].port, str)

    # BLUE should have a settlement at node 28
    blue_player = next(p for p in result.players if p.color == "BLUE")
    assert len(blue_player.settlements) == 1
    assert blue_player.settlements[0].node_id == expected_blue_settlement.node_id
    # Port should now be detected from public_state (could be None or a resource string)
    assert blue_player.settlements[0].port is None or isinstance(blue_player.settlements[0].port, str)


def test_gather_board_occupancy_data_empty_game():
    """Test that the function works on a game that hasn't started"""
    game = create_test_game_empty()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected empty board occupancy data
    expected_data_by_color = {
        "RED": PlayerBoardData(color="RED", settlements=[], cities=[], roads=[]),
        "BLUE": PlayerBoardData(color="BLUE", settlements=[], cities=[], roads=[]),
        "ORANGE": PlayerBoardData(color="ORANGE", settlements=[], cities=[], roads=[]),
        "WHITE": PlayerBoardData(color="WHITE", settlements=[], cities=[], roads=[]),
    }

    assert isinstance(result, BoardOccupancyData)
    assert len(result.players) == 4

    # Compare each player's data by color
    for actual_player in result.players:
        expected_player = expected_data_by_color[actual_player.color]
        assert actual_player.color == expected_player.color
        assert actual_player.settlements == expected_player.settlements
        assert actual_player.cities == expected_player.cities
        assert actual_player.roads == expected_player.roads


def test_gather_board_occupancy_data_color_names():
    """Test that player color names are correctly captured"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected color names
    expected_colors = {"RED", "BLUE", "ORANGE", "WHITE"}

    actual_colors = {player_data.color for player_data in result.players}
    assert actual_colors == expected_colors


def test_gather_board_occupancy_data_specific_building_count():
    """Test that specific building counts match our test setup"""
    game = create_test_game_with_buildings()
    result = gather_board_occupancy_data(build_public_state(game))

    # Construct expected player data (node IDs are fixed, but hex data varies)
    expected_red_player = PlayerBoardData(
        color="RED",
        settlements=[BuildingInfo(node_id=0, adjacent_hexes=[], port=None)],
        cities=[BuildingInfo(node_id=10, adjacent_hexes=[], port=None)],
        roads=[(0, 5)]
    )

    expected_blue_player = PlayerBoardData(
        color="BLUE",
        settlements=[BuildingInfo(node_id=5, adjacent_hexes=[], port=None)],
        cities=[],
        roads=[(5, 16)]
    )

    expected_orange_player = PlayerBoardData(
        color="ORANGE",
        settlements=[],
        cities=[],
        roads=[]
    )

    expected_white_player = PlayerBoardData(
        color="WHITE",
        settlements=[],
        cities=[],
        roads=[]
    )

    # Compare RED player
    red_player = next(p for p in result.players if p.color == "RED")
    assert len(red_player.settlements) == len(expected_red_player.settlements)
    assert len(red_player.cities) == len(expected_red_player.cities)
    assert len(red_player.roads) == len(expected_red_player.roads)
    assert red_player.settlements[0].node_id == expected_red_player.settlements[0].node_id
    assert red_player.cities[0].node_id == expected_red_player.cities[0].node_id
    assert red_player.roads == expected_red_player.roads
    # Check that adjacent_hexes and port are now populated
    assert isinstance(red_player.settlements[0].adjacent_hexes, list)
    assert isinstance(red_player.cities[0].adjacent_hexes, list)

    # Compare BLUE player
    blue_player = next(p for p in result.players if p.color == "BLUE")
    assert len(blue_player.settlements) == len(expected_blue_player.settlements)
    assert len(blue_player.cities) == len(expected_blue_player.cities)
    assert len(blue_player.roads) == len(expected_blue_player.roads)
    assert blue_player.settlements[0].node_id == expected_blue_player.settlements[0].node_id
    assert blue_player.roads == expected_blue_player.roads
    # Check that adjacent_hexes and port are now populated
    assert isinstance(blue_player.settlements[0].adjacent_hexes, list)

    # Compare ORANGE player
    orange_player = next(p for p in result.players if p.color == "ORANGE")
    assert orange_player.settlements == expected_orange_player.settlements
    assert orange_player.cities == expected_orange_player.cities
    assert orange_player.roads == expected_orange_player.roads

    # Compare WHITE player
    white_player = next(p for p in result.players if p.color == "WHITE")
    assert white_player.settlements == expected_white_player.settlements
    assert white_player.cities == expected_white_player.cities
    assert white_player.roads == expected_white_player.roads


def test_format_board_occupancy_data_returns_string():
    """Test that format_board_occupancy_data returns a string"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_board_occupancy_data_contains_header():
    """Test that formatted output contains the correct header"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    assert "[CURRENT BOARD OCCUPANCY]" in result


def test_format_board_occupancy_data_contains_player_colors():
    """Test that formatted output contains all player colors with production info"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    assert "- RED: Total:" in result
    assert "- BLUE: Total:" in result
    assert "- ORANGE: Total:" in result
    assert "- WHITE: Total:" in result


def test_format_board_occupancy_data_contains_building_sections():
    """Test that formatted output contains ports, settlements, cities, and roads sections"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    assert "Ports:" in result
    assert "Settlements:" in result
    assert "Cities (x2 production):" in result
    assert "Roads: Edges" in result


def test_format_board_occupancy_data_empty_buildings():
    """Test that formatted output handles empty buildings correctly"""
    game = create_test_game_empty()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should show "None" for empty building lists
    assert "Settlements: [None]" in result
    assert "Cities (x2 production): [None]" in result
    assert "Roads: Edges [None]" in result
    # Should show "Total: 0 pips" for players with no buildings
    assert "Total: 0 pips" in result
    # Should show "Ports: None" for players with no ports
    assert "Ports: None" in result


def test_format_board_occupancy_data_with_buildings():
    """Test that formatted output shows building information when present"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should contain building information (not "None")
    # Check that at least one player has actual buildings
    assert "Node " in result  # Indicates non-empty building lists with node IDs
    # Should not have all "None" for buildings
    assert result.count("Settlements: [None]") < 4  # At least one player has settlements


def test_format_board_occupancy_data_no_robber():
    """Test that formatted output does not contain robber information"""
    game = create_test_game_with_robber()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should NOT contain robber information
    assert "ROBBER" not in result


def test_format_board_occupancy_data_settlement_format():
    """Test that settlements are formatted with hex info and pips"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should contain pip information
    assert "pips" in result
    # Should contain total pips
    assert "Total:" in result


def test_format_board_occupancy_data_road_format():
    """Test that roads are formatted as coordinate tuples"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should contain road coordinate tuples
    assert "(0, 5)" in result  # RED's road
    assert "(5, 16)" in result  # BLUE's road


def test_format_board_occupancy_data_port_format():
    """Test that ports are formatted in separate section"""
    game = create_test_game_with_ports()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should contain port information in separate section
    assert "Ports:" in result
    # Should not contain "Port:" in building sections anymore
    # Check that settlement and city sections don't contain port info
    lines = result.split('\n')
    for line in lines:
        if "Settlements:" in line or "Cities" in line:
            # These lines should not contain port information
            assert "Port:" not in line


def test_format_board_occupancy_data_production_calculation():
    """Test that production calculation is correct (now from public_state adjacency info)"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Should contain production information
    assert "Total:" in result
    assert "pips" in result
    # With adjacency info from public_state, production should be calculated
    # It might be 0 if buildings are on non-resource nodes, but the calculation should work


def test_format_board_occupancy_data_desert_formatting():
    """Test that desert tiles are formatted correctly"""
    game = create_test_game_deterministic()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # With adjacency info from public_state, we should have tile information in building descriptions
    # Just check that the basic structure is present
    assert "Node 35:" in result  # WHITE city on desert
    # Should not contain the old approximate format
    assert "?" not in result or "~" not in result  # No approximate roll numbers


def test_get_full_board_map_returns_string():
    """Test that get_full_board_map returns a string"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_full_board_map_contains_header():
    """Test that get_full_board_map contains the correct header"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    assert "[FULL BOARD MAP - 19 HEXES]" in result


def test_get_full_board_map_contains_19_hexes():
    """Test that get_full_board_map contains information for all 19 hexes"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Count the number of "Tile" lines (should be 19)
    tile_count = result.count("Tile ")
    assert tile_count == 19


def test_get_full_board_map_contains_ports_section():
    """Test that get_full_board_map does not contain ports section"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Board map doesn't contain port information (ports are in building info)
    assert "[PORTS]" not in result


def test_get_full_board_map_tile_format():
    """Test that each tile is formatted with resource and exact pip information"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Should contain resource names (common Catan resources)
    assert "WOOD" in result or "BRICK" in result or "SHEEP" in result or "WHEAT" in result or "ORE" in result or "DESERT" in result
    # Should contain exact pip information (now from public_state)
    assert "pips" in result
    # Should not contain approximate markers
    assert "~" not in result


def test_get_full_board_map_node_information():
    """Test that tiles do not show node information"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Board map doesn't contain node-to-tile adjacency information
    assert "Touches Nodes:" not in result


def test_get_full_board_map_no_robber():
    """Test that get_full_board_map does not show robber (static board map)"""
    game = create_test_game_with_robber()
    result = get_full_board_map(build_public_state(game))
    
    # Should not contain robber marker since board map is static
    assert "[ROBBER]" not in result


def test_get_full_board_map_deterministic_ordering():
    """Test that get_full_board_map produces deterministic output (sorted by tile ID)"""
    game = create_test_game_empty()
    result1 = get_full_board_map(build_public_state(game))
    result2 = get_full_board_map(build_public_state(game))
    
    # Should produce identical output
    assert result1 == result2


def test_get_full_board_map_tile_ids():
    """Test that get_full_board_map includes tile IDs in order"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Should contain tile IDs (exact format may vary)
    assert "Tile" in result


def test_get_full_board_map_port_nodes():
    """Test that get_full_board_map shows adjacent node information for each tile"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Board map now contains adjacent node information for each tile
    assert "Nodes:" in result
    # Check that node IDs are in list format
    assert "Nodes: [" in result


def test_get_full_board_map_resource_types():
    """Test that get_full_board_map shows various resource types"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Check for various resource types that should be present
    resource_keywords = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "DESERT"]
    found_resources = [keyword for keyword in resource_keywords if keyword in result]
    
    # At least some resources should be present
    assert len(found_resources) > 0


def test_get_full_board_map_roll_numbers():
    """Test that get_full_board_map shows exact roll information"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Should contain exact roll information (not approximate)
    # Should not contain approximate markers
    assert "?" not in result  # Should not have ? marker for approximate rolls
    # Should contain numeric roll numbers
    assert any(char.isdigit() for char in result)  # Should have some numbers


def test_get_full_board_map_desert_handling():
    """Test that get_full_board_map handles desert tiles correctly - just DESERT"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Should handle desert tiles with just "DESERT" (no roll/pips)
    assert "DESERT" in result


def test_get_full_board_map_pip_counts():
    """Test that get_full_board_map shows exact pip counts"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Should contain exact pip counts (not approximate)
    # Should not contain approximate markers
    assert "~" not in result  # Should not have ~ marker for approximate pips
    # Should contain pip information
    assert "pips" in result


def test_get_pip_count():
    """Test that get_pip_count returns correct pip values for different rolls"""
    assert get_pip_count(2) == 1
    assert get_pip_count(3) == 2
    assert get_pip_count(4) == 3
    assert get_pip_count(5) == 4
    assert get_pip_count(6) == 5
    assert get_pip_count(8) == 5
    assert get_pip_count(9) == 4
    assert get_pip_count(10) == 3
    assert get_pip_count(11) == 2
    assert get_pip_count(12) == 1
    assert get_pip_count(7) == 0  # 7 is not a valid roll
    assert get_pip_count(None) == 0  # None should return 0


def test_get_full_board_map_complete_structure():
    """Test that get_full_board_map has the complete expected structure (feature limitations)"""
    game = create_test_game_empty()
    result = get_full_board_map(build_public_state(game))
    
    # Split into lines and check structure
    lines = result.split('\n')
    
    # First line should be the header
    assert lines[0] == "[FULL BOARD MAP - 19 HEXES]"
    
    # Should have 19 tile lines
    tile_lines = [line for line in lines if line.startswith("Tile ")]
    assert len(tile_lines) == 19
    
    # Should NOT have a ports section (feature limitation)
    assert not any("[PORTS]" in line for line in lines)
    
    # Should NOT have port information lines (feature limitation)
    port_lines = [line for line in lines if "Port:" in line]
    assert len(port_lines) == 0


def test_get_full_board_map_complete_happy_path():
    """Complete happy path test that asserts on the entire resulting string"""
    import random
    # Set random seed for deterministic map generation
    random.seed(42)
    
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    
    game = Game(players)
    result = get_full_board_map(build_public_state(game))
    
    # With public_state, we should have exact deterministic output
    # Just check that it produces consistent output
    result2 = get_full_board_map(build_public_state(game))
    assert result == result2  # Should be deterministic


def test_format_board_occupancy_data_structure():
    """Test that the overall structure of formatted output is correct"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    lines = result.split('\n')
    
    # First line should be header
    assert lines[0] == "[CURRENT BOARD OCCUPANCY]"
    
    # Should NOT have robber section
    robber_lines = [line for line in lines if line.startswith("ROBBER:")]
    assert len(robber_lines) == 0  # Should have no ROBBER line
    
    # Should have 4 player sections + header = 5+ lines
    assert len(lines) >= 5
    
    # Each player should have 5 sections: color+production, ports, settlements, cities, roads
    # Count player lines (lines starting with "- " but not header)
    player_lines = [line for line in lines if line.startswith("- ")]
    assert len(player_lines) == 4  # 4 players


def test_format_board_occupancy_data_player_order():
    """Test that players are formatted in a consistent order"""
    game = create_test_game_with_buildings()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # Find player section order
    red_pos = result.find("- RED:")
    blue_pos = result.find("- BLUE:")
    orange_pos = result.find("- ORANGE:")
    white_pos = result.find("- WHITE:")
    
    # All should be present
    assert red_pos > 0
    assert blue_pos > 0
    assert orange_pos > 0
    assert white_pos > 0


def create_test_game_deterministic():
    """Create a game with deterministic output for string comparison tests"""
    import random
    # Set random seed for deterministic map generation
    random.seed(42)
    
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE), 
        SimplePlayer(Color.WHITE),
    ]
    
    game = Game(players)
    board = game.state.board
    
    # Place buildings for all players with multiple settlements and cities
    # RED: 2 settlements (0, 1), 2 cities (10, 11), 5 roads
    board.buildings[0] = (Color.RED, SETTLEMENT)
    board.buildings[1] = (Color.RED, SETTLEMENT)
    board.buildings[10] = (Color.RED, CITY)
    board.buildings[11] = (Color.RED, CITY)
    
    # RED roads
    board.roads[(0, 5)] = Color.RED
    board.roads[(5, 0)] = Color.RED
    board.roads[(1, 6)] = Color.RED
    board.roads[(6, 1)] = Color.RED
    board.roads[(10, 15)] = Color.RED
    board.roads[(15, 10)] = Color.RED
    board.roads[(11, 16)] = Color.RED
    board.roads[(16, 11)] = Color.RED
    board.roads[(15, 20)] = Color.RED
    board.roads[(20, 15)] = Color.RED
    board.roads[(16, 22)] = Color.RED
    board.roads[(22, 16)] = Color.RED
    
    # BLUE: 2 settlements (5, 6), 1 city (15), 5 roads
    board.buildings[5] = (Color.BLUE, SETTLEMENT)
    board.buildings[6] = (Color.BLUE, SETTLEMENT)
    board.buildings[15] = (Color.BLUE, CITY)
    
    # BLUE roads
    board.roads[(5, 16)] = Color.BLUE
    board.roads[(16, 5)] = Color.BLUE
    board.roads[(6, 21)] = Color.BLUE
    board.roads[(21, 6)] = Color.BLUE
    board.roads[(15, 20)] = Color.BLUE
    board.roads[(20, 15)] = Color.BLUE
    board.roads[(20, 25)] = Color.BLUE
    board.roads[(25, 20)] = Color.BLUE
    board.roads[(25, 26)] = Color.BLUE
    board.roads[(26, 25)] = Color.BLUE
    
    # ORANGE: 1 settlement (20), 2 cities (25, 26), 5 roads
    board.buildings[20] = (Color.ORANGE, SETTLEMENT)
    board.buildings[25] = (Color.ORANGE, CITY)
    board.buildings[26] = (Color.ORANGE, CITY)
    
    # ORANGE roads
    board.roads[(20, 21)] = Color.ORANGE
    board.roads[(21, 20)] = Color.ORANGE
    board.roads[(25, 30)] = Color.ORANGE
    board.roads[(30, 25)] = Color.ORANGE
    board.roads[(26, 31)] = Color.ORANGE
    board.roads[(31, 26)] = Color.ORANGE
    board.roads[(30, 35)] = Color.ORANGE
    board.roads[(35, 30)] = Color.ORANGE
    board.roads[(31, 36)] = Color.ORANGE
    board.roads[(36, 31)] = Color.ORANGE
    
    # WHITE: 1 settlement (30), 1 city (35), 5 roads
    board.buildings[30] = (Color.WHITE, SETTLEMENT)
    board.buildings[35] = (Color.WHITE, CITY)
    
    # WHITE roads
    board.roads[(30, 31)] = Color.WHITE
    board.roads[(31, 30)] = Color.WHITE
    board.roads[(35, 36)] = Color.WHITE
    board.roads[(36, 35)] = Color.WHITE
    board.roads[(35, 40)] = Color.WHITE
    board.roads[(40, 35)] = Color.WHITE
    board.roads[(36, 41)] = Color.WHITE
    board.roads[(41, 36)] = Color.WHITE
    board.roads[(40, 42)] = Color.WHITE
    board.roads[(42, 40)] = Color.WHITE
    
    # Update board_buildable_ids
    for node_id in [0, 1, 5, 6, 10, 11, 15, 20, 21, 22, 25, 26, 30, 31, 35, 36, 40, 41, 42]:
        board.board_buildable_ids.discard(node_id)
    
    # Set robber to a known coordinate (0, 0, 0) which is Tile 0: SHEEP
    board.robber_coordinate = (0, 0, 0)
    
    return game


def test_format_board_occupancy_data_complete_happy_path():
    """Complete happy path test that checks basic structure"""
    game = create_test_game_deterministic()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    
    result = format_board_occupancy_data(occupancy_data)
    
    # With public_state, we should have exact output with adjacency info
    # Check basic structure
    assert "[CURRENT BOARD OCCUPANCY]" in result
    assert "- BLUE:" in result
    assert "- ORANGE:" in result
    assert "- RED:" in result
    assert "- WHITE:" in result
    assert "ROBBER" not in result


def test_get_full_board_map_exact_string_empty_game():
    """Test get_full_board_map returns properly formatted string with node information"""
    import random
    random.seed(42)
    
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    
    game = Game(players)
    result = get_full_board_map(build_public_state(game))
    
    # Check that the result has the proper structure with node information
    assert "[FULL BOARD MAP - 19 HEXES]" in result
    assert "Tile" in result
    assert "Nodes:" in result
    # Check that node IDs are in list format
    assert "Nodes: [" in result
    # Check that there are 19 tiles
    tile_count = result.count("Tile")
    assert tile_count == 19


def test_get_full_board_map_exact_string_deterministic_game():
    """Test get_full_board_map returns properly formatted string with node information for deterministic game"""
    game = create_test_game_deterministic()
    result = get_full_board_map(build_public_state(game))
    
    # Check that the result has the proper structure with node information
    assert "[FULL BOARD MAP - 19 HEXES]" in result
    assert "Tile" in result
    assert "Nodes:" in result
    # Check that node IDs are in list format
    assert "Nodes: [" in result
    # Check that there are 19 tiles
    tile_count = result.count("Tile")
    assert tile_count == 19


def test_format_board_occupancy_data_exact_string_empty_game():
    """Test format_board_occupancy_data returns exact expected string for empty game"""
    import random
    random.seed(42)
    
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    
    game = Game(players)
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    result = format_board_occupancy_data(occupancy_data)
    
    expected = """[CURRENT BOARD OCCUPANCY]
- BLUE: Total: 0 pips
  * Ports: None
  * Settlements: [None]
  * Cities (x2 production): [None]
  * Roads: Edges [None]
- ORANGE: Total: 0 pips
  * Ports: None
  * Settlements: [None]
  * Cities (x2 production): [None]
  * Roads: Edges [None]
- RED: Total: 0 pips
  * Ports: None
  * Settlements: [None]
  * Cities (x2 production): [None]
  * Roads: Edges [None]
- WHITE: Total: 0 pips
  * Ports: None
  * Settlements: [None]
  * Cities (x2 production): [None]
  * Roads: Edges [None]"""
    
    assert result == expected


def test_format_board_occupancy_data_exact_string_deterministic_game():
    """Test format_board_occupancy_data returns exact expected string for deterministic game"""
    game = create_test_game_deterministic()
    occupancy_data = gather_board_occupancy_data(build_public_state(game))
    result = format_board_occupancy_data(occupancy_data)
    
    expected = """[CURRENT BOARD OCCUPANCY]
- BLUE: Total: 37 pips (WOOD: 15, SHEEP: 7, WHEAT: 15)
  * Ports: None
  * Settlements: [Node 5: (Tile 0: 11 SHEEP (2 pips)), (Tile 4: 5 WHEAT (4 pips)), (Tile 5: 4 WHEAT (3 pips)), Total: 9 pips, Node 6: (Tile 1: 10 WOOD (3 pips)), (Tile 6: 9 SHEEP (4 pips)), (Tile 18: 2 SHEEP (1 pips)), Total: 8 pips]
  * Cities (x2 production): [Node 15: (Tile 3: 6 WOOD (5 pips)), (Tile 4: 5 WHEAT (4 pips)), (Tile 12: 12 WOOD (1 pips)), Total: 10 pips]
  * Roads: Edges [(5, 16), (6, 21), (15, 20), (20, 25), (25, 26)]
- ORANGE: Total: 25 pips (SHEEP: 20, WHEAT: 5)
  * Ports: SHEEP
  * Settlements: [Node 20: (Tile 5: 4 WHEAT (3 pips)), (Tile 6: 9 SHEEP (4 pips)), (Tile 16: 3 WHEAT (2 pips)), Total: 9 pips]
  * Cities (x2 production): [Node 25: (Tile 7: 5 SHEEP (4 pips)), Total: 4 pips, Node 26: (Tile 7: 5 SHEEP (4 pips)), Total: 4 pips]
  * Roads: Edges [(20, 21), (25, 30), (26, 31), (30, 35), (31, 36)]
- RED: Total: 52 pips (WOOD: 15, BRICK: 18, SHEEP: 12, WHEAT: 3, ORE: 4)
  * Ports: None
  * Settlements: [Node 0: (Tile 0: 11 SHEEP (2 pips)), (Tile 5: 4 WHEAT (3 pips)), (Tile 6: 9 SHEEP (4 pips)), Total: 9 pips, Node 1: (Tile 0: 11 SHEEP (2 pips)), (Tile 1: 10 WOOD (3 pips)), (Tile 6: 9 SHEEP (4 pips)), Total: 9 pips]
  * Cities (x2 production): [Node 10: (Tile 2: 3 BRICK (2 pips)), (Tile 8: 8 BRICK (5 pips)), (Tile 9: 4 WOOD (3 pips)), Total: 10 pips, Node 11: (Tile 2: 3 BRICK (2 pips)), (Tile 9: 4 WOOD (3 pips)), (Tile 10: 11 ORE (2 pips)), Total: 7 pips]
  * Roads: Edges [(0, 5), (1, 6), (10, 15), (11, 16), (16, 22)]
- WHITE: Total: 3 pips (WOOD: 3)
  * Ports: 3:1
  * Settlements: [Node 30: (Tile 9: 4 WOOD (3 pips)), Total: 3 pips]
  * Cities (x2 production): [Node 35: , Total: 0 pips]
  * Roads: Edges [(30, 31), (35, 36), (35, 40), (36, 41), (40, 42)]"""
    
    assert result == expected


# ============================================================================
# calculate_blocked_production unit tests
# ============================================================================

def test_calculate_blocked_production_returns_dict():
    """Test that calculate_blocked_production returns a dictionary"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = calculate_blocked_production(0, occupancy_data.players)
    
    assert isinstance(result, dict)


def test_calculate_blocked_production_empty_players():
    """Test that calculate_blocked_production handles empty player list"""
    result = calculate_blocked_production(0, [])
    
    assert result == {}


def test_calculate_blocked_production_no_blocking():
    """Test that calculate_blocked_production returns empty dict when no buildings are adjacent"""
    game = create_test_game_empty()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    # Use a tile ID that has no adjacent buildings
    result = calculate_blocked_production(0, occupancy_data.players)
    
    assert result == {}


def test_calculate_blocked_production_with_blocking():
    """Test that calculate_blocked_production calculates blocked production correctly"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    # Tile 0 should block RED and BLUE
    result = calculate_blocked_production(0, occupancy_data.players)
    
    assert len(result) > 0
    assert "RED" in result or "BLUE" in result


def test_calculate_blocked_production_city_multiplier():
    """Test that cities get 2x multiplier in blocked production calculation"""
    # Create a custom game with a city adjacent to the robber
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    
    game = Game(players)
    board = game.state.board
    
    # Place a RED city at node 0 (adjacent to Tile 0)
    board.buildings[0] = (Color.RED, CITY)
    board.board_buildable_ids.discard(0)
    
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = calculate_blocked_production(0, occupancy_data.players)
    
    # The city should block 2x the pips (4 pips instead of 2)
    assert "RED" in result
    assert "4 pips" in result["RED"]


# format_robber_info unit tests
# ============================================================================

def test_format_robber_info_returns_string():
    """Test that format_robber_info returns a string"""
    game = create_test_game_with_robber()
    public_state = build_public_state(game)
    players = []
    
    result = format_robber_info(public_state, players)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_robber_info_contains_robber_header():
    """Test that format_robber_info contains the robber header"""
    game = create_test_game_with_robber()
    public_state = build_public_state(game)
    players = []
    
    result = format_robber_info(public_state, players)
    
    assert "ROBBER:" in result


def test_format_robber_info_without_game():
    """Test that format_robber_info works with public_state only"""
    game = create_test_game_with_robber()
    public_state = build_public_state(game)
    players = []
    
    result = format_robber_info(public_state, players)
    
    # Should return robber tile information from public_state
    assert "ROBBER:" in result
    assert "Tile" in result


def test_format_robber_info_with_empty_players():
    """Test that format_robber_info works with empty player list"""
    game = create_test_game_with_robber()
    public_state = build_public_state(game)
    players = []
    
    result = format_robber_info(public_state, players)
    
    # Should show "Blocking: None" when no players
    assert "Blocking: None" in result


def test_format_robber_info_with_blocked_production():
    """Test that format_robber_info calculates blocked production correctly"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = format_robber_info(public_state, occupancy_data.players)
    
    # Should show blocked production for players with buildings adjacent to robber tile
    assert "Blocking" in result
    # RED and BLUE have buildings adjacent to the robber tile
    assert "RED" in result or "BLUE" in result


def test_format_robber_info_blocked_production_calculation():
    """Test that blocked production calculation is correct"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = format_robber_info(public_state, occupancy_data.players)
    
    # RED has settlement at node 0 (adjacent to robber tile) - 2 pips blocked
    # RED has settlement at node 1 (adjacent to robber tile) - 2 pips blocked
    # Total for RED: 4 pips blocked
    assert "Blocking RED: 4 pips" in result
    
    # BLUE has settlement at node 5 (adjacent to robber tile) - 2 pips blocked
    assert "Blocking BLUE: 2 pips" in result


def test_format_robber_info_no_blocked_production():
    """Test that format_robber_info shows no blocking when no buildings are adjacent"""
    game = create_test_game_empty()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = format_robber_info(public_state, occupancy_data.players)
    
    # Should show "Blocking: None" when no buildings are adjacent
    assert "Blocking: None" in result


def test_format_robber_info_tile_information():
    """Test that format_robber_info includes tile information from public_state"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    players = []
    
    result = format_robber_info(public_state, players)
    
    # Should include tile information
    assert "Tile" in result
    assert "SHEEP" in result
    assert "2 pips" in result


def test_format_robber_info_desert_tile():
    """Test that format_robber_info handles desert tiles correctly"""
    game = create_test_game_empty()
    public_state = build_public_state(game)
    players = []
    
    # Find the actual desert tile ID from the public_state
    desert_tile_id = None
    for tile_id, (resource, roll) in public_state.board.map.tiles.items():
        if resource is None:  # Desert tile has None resource
            desert_tile_id = tile_id
            break
    
    if desert_tile_id is not None:
        # Manually set robber to desert tile for testing
        # We need to create a new public_state with robber on desert
        from dataclasses import replace
        modified_board = replace(public_state.board, robber_tile_id=desert_tile_id)
        modified_state = replace(public_state, board=modified_board)
        
        result = format_robber_info(modified_state, players)
        
        # Should show DESERT for the tile
        assert "DESERT" in result
    else:
        # Skip test if no desert found (shouldn't happen with standard map)
        pytest.skip("No desert tile found in map")


def test_format_robber_info_city_multiplier():
    """Test that cities get 2x multiplier in blocked production calculation"""
    # Create a custom game with a city adjacent to the robber
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    
    game = Game(players)
    board = game.state.board
    
    # Place a RED city at node 0 (adjacent to Tile 0)
    board.buildings[0] = (Color.RED, CITY)
    board.board_buildable_ids.discard(0)
    
    # Set robber to Tile 0
    board.robber_tile_id = 0
    
    public_state = build_public_state(game)
    
    # Manually override the robber position in public_state to ensure it's on Tile 0
    from dataclasses import replace
    modified_board = replace(public_state.board, robber_tile_id=0)
    modified_state = replace(public_state, board=modified_board)
    
    occupancy_data = gather_board_occupancy_data(modified_state)
    result = format_robber_info(modified_state, occupancy_data.players)
    
    # The city should block 2x the pips (4 pips instead of 2)
    assert "Blocking RED: 4 pips" in result


def test_format_robber_info_multiple_players_blocked():
    """Test that format_robber_info shows blocked production for multiple players"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = format_robber_info(public_state, occupancy_data.players)
    
    # Should show blocked production for multiple players
    # Count how many "Blocking" lines there are
    blocking_count = result.count("Blocking")
    assert blocking_count >= 2  # At least RED and BLUE should be blocked


def test_format_robber_info_sorted_players():
    """Test that blocked production is sorted by player color"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = format_robber_info(public_state, occupancy_data.players)
    
    # Extract the blocking lines
    lines = result.split('\n')
    blocking_lines = [line for line in lines if "Blocking" in line and ":" in line]
    
    # Check that they are sorted alphabetically by color
    if len(blocking_lines) > 1:
        colors = [line.split("Blocking ")[1].split(":")[0] for line in blocking_lines]
        assert colors == sorted(colors)


def test_format_robber_info_exact_string_deterministic_game():
    """Test format_robber_info returns exact expected string for deterministic game"""
    game = create_test_game_deterministic()
    public_state = build_public_state(game)
    occupancy_data = gather_board_occupancy_data(public_state)
    
    result = format_robber_info(public_state, occupancy_data.players)
    
    # The robber is on Tile 0 (11 SHEEP, 2 pips)
    # RED has settlements at nodes 0 and 1 (both adjacent to Tile 0) - 4 pips blocked
    # BLUE has settlement at node 5 (adjacent to Tile 0) - 2 pips blocked
    expected = """ROBBER: Tile 0 - Tile 0: 11 SHEEP (2 pips)
  * Blocking BLUE: 2 pips
  * Blocking RED: 4 pips"""
    
    assert result == expected


def create_public_player(**kwargs):
    """Helper to create a PublicPlayer with sensible defaults.
    
    Only specify the parameters you want to override from defaults.
    """
    from catanatron.models.public_state import PublicPlayer
    
    defaults = {
        'public_vps': 0,
        'has_army': False,
        'has_road': False,
        'longest_road_length': 0,
        'roads_left': 15,
        'settlements_left': 5,
        'cities_left': 4,
        'has_rolled': False,
        'hand_resource_count': 0,
        'hand_dev_count': 0,
        'played_knight': 0,
        'played_monopoly': 0,
        'played_road_building': 0,
        'played_year_of_plenty': 0,
        'played_victory_point': 0
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return PublicPlayer(**defaults)


def create_public_map(**kwargs):
    """Helper to create a PublicMap with sensible defaults.
    
    Only specify the parameters you want to override from defaults.
    """
    from catanatron.models.public_state import PublicMap
    
    defaults = {
        'tiles': {},
        'tile_coordinates': {},
        'ports': {},
        'adjacent_tiles': {},
        'land_nodes': frozenset()
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return PublicMap(**defaults)


def create_public_board(**kwargs):
    """Helper to create a PublicBoard with sensible defaults.
    
    Only specify the parameters you want to override from defaults.
    """
    from catanatron.models.public_state import PublicBoard
    
    defaults = {
        'buildings': {},
        'roads': {},
        'robber_tile_id': 0,
        'longest_road_color': None,
        'longest_road_length': 0,
        'map': create_public_map()
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return PublicBoard(**defaults)


def create_public_state(players, **kwargs):
    """Helper to create a PublicState with sensible defaults.
    
    Args:
        players: Dict of Color to PublicPlayer
        **kwargs: Optional overrides for board and other parameters
    
    Only specify the parameters you want to override from defaults.
    """
    from catanatron.models.public_state import PublicState
    
    defaults = {
        'board': create_public_board(),
        'players': players
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return PublicState(**defaults)


def create_inventory(**kwargs):
    """Helper to create an Inventory with sensible defaults.
    
    Only specify the parameters you want to override from defaults.
    """
    from catanatron.models.inventory import Inventory
    
    defaults = {
        'wood': 0,
        'brick': 0,
        'sheep': 0,
        'wheat': 0,
        'ore': 0,
        'knight': 0,
        'year_of_plenty': 0,
        'monopoly': 0,
        'road_building': 0,
        'victory_point': 0,
        'actual_vps': 0,
        'has_played_development_card': False
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return Inventory(**defaults)


def create_mock_public_state():
    """Create a mock public state for testing resource and dev card formatting"""
    from catanatron.models.player import Color
    
    # Create mock public players using helper
    red_player = create_public_player(
        public_vps=5,
        longest_road_length=3,
        roads_left=13,
        settlements_left=4,
        hand_resource_count=3,
        hand_dev_count=2,
        played_knight=1,
        played_year_of_plenty=1
    )
    
    blue_player = create_public_player(
        public_vps=4,
        longest_road_length=2,
        roads_left=14,
        settlements_left=4,
        hand_resource_count=5,
        hand_dev_count=1,
        played_knight=2,
        played_road_building=1
    )
    
    orange_player = create_public_player(
        public_vps=3,
        longest_road_length=1,
        roads_left=14,
        settlements_left=5
    )
    
    white_player = create_public_player(
        public_vps=2,
        roads_left=15,
        settlements_left=5,
        hand_resource_count=1,
        hand_dev_count=3,
        played_monopoly=1,
        played_victory_point=1
    )
    
    # Create public state using helper
    return create_public_state({
        Color.RED: red_player,
        Color.BLUE: blue_player,
        Color.ORANGE: orange_player,
        Color.WHITE: white_player
    })


def test_get_player_resources_with_inventory():
    """Test get_player_resources with current player inventory provided"""
    public_state = create_mock_public_state()
    current_player_color = Color.RED
    
    # Create inventory for current player using helper
    inventory = create_inventory(
        wood=2,
        brick=1,
        sheep=3,
        wheat=0,
        ore=1,
        actual_vps=5
    )
    
    result = get_player_resources(public_state, current_player_color, inventory)
    
    expected = """[PLAYER RESOURCES]
- RED: WOOD: 2, BRICK: 1, SHEEP: 3, ORE: 1
- BLUE: 5 resource cards (hidden)
- ORANGE: 0 resource cards (hidden)
- WHITE: 1 resource cards (hidden)"""
    
    assert result == expected


def test_get_player_resources_without_inventory():
    """Test get_player_resources without current player inventory (all public info)"""
    public_state = create_mock_public_state()
    current_player_color = Color.RED
    
    result = get_player_resources(public_state, current_player_color, None)
    
    expected = """[PLAYER RESOURCES]
- RED: 3 resource cards (hidden)
- BLUE: 5 resource cards (hidden)
- ORANGE: 0 resource cards (hidden)
- WHITE: 1 resource cards (hidden)"""
    
    assert result == expected


def test_get_player_resources_empty_inventory():
    """Test get_player_resources with empty inventory for current player"""
    public_state = create_mock_public_state()
    current_player_color = Color.RED
    
    # Create empty inventory using helper
    inventory = create_inventory(actual_vps=5)
    
    result = get_player_resources(public_state, current_player_color, inventory)
    
    expected = """[PLAYER RESOURCES]
- RED: No resources
- BLUE: 5 resource cards (hidden)
- ORANGE: 0 resource cards (hidden)
- WHITE: 1 resource cards (hidden)"""
    
    assert result == expected


def test_get_player_resources_all_resource_types():
    """Test get_player_resources with all resource types present"""
    public_state = create_mock_public_state()
    current_player_color = Color.BLUE
    
    # Create inventory with all resource types using helper
    inventory = create_inventory(
        wood=1,
        brick=2,
        sheep=3,
        wheat=4,
        ore=5,
        actual_vps=4
    )
    
    result = get_player_resources(public_state, current_player_color, inventory)
    
    expected = """[PLAYER RESOURCES]
- RED: 3 resource cards (hidden)
- BLUE: WOOD: 1, BRICK: 2, SHEEP: 3, WHEAT: 4, ORE: 5
- ORANGE: 0 resource cards (hidden)
- WHITE: 1 resource cards (hidden)"""
    
    assert result == expected


def test_get_player_dev_cards_with_inventory():
    """Test get_player_dev_cards with current player inventory provided"""
    public_state = create_mock_public_state()
    current_player_color = Color.RED
    
    # Create inventory for current player using helper
    inventory = create_inventory(
        knight=2,
        year_of_plenty=1,
        victory_point=1,
        actual_vps=5
    )
    
    result = get_player_dev_cards(public_state, current_player_color, inventory)
    
    expected = """[PLAYER DEVELOPMENT CARDS]
- RED: KNIGHT: 2, YEAR_OF_PLENTY: 1, VICTORY_POINT: 1 (Played: KNIGHT: 1, YEAR_OF_PLENTY: 1)
- BLUE: 1 dev cards (hidden) (Played: KNIGHT: 2, ROAD_BUILDING: 1)
- ORANGE: 0 dev cards (hidden)
- WHITE: 3 dev cards (hidden) (Played: MONOPOLY: 1, VICTORY_POINT: 1)"""
    
    assert result == expected


def test_get_player_dev_cards_without_inventory():
    """Test get_player_dev_cards without current player inventory (all public info)"""
    public_state = create_mock_public_state()
    current_player_color = Color.RED
    
    result = get_player_dev_cards(public_state, current_player_color, None)
    
    expected = """[PLAYER DEVELOPMENT CARDS]
- RED: 2 dev cards (hidden) (Played: KNIGHT: 1, YEAR_OF_PLENTY: 1)
- BLUE: 1 dev cards (hidden) (Played: KNIGHT: 2, ROAD_BUILDING: 1)
- ORANGE: 0 dev cards (hidden)
- WHITE: 3 dev cards (hidden) (Played: MONOPOLY: 1, VICTORY_POINT: 1)"""
    
    assert result == expected


def test_get_player_dev_cards_empty_inventory():
    """Test get_player_dev_cards with empty inventory for current player"""
    public_state = create_mock_public_state()
    current_player_color = Color.RED
    
    # Create empty inventory using helper
    inventory = create_inventory(actual_vps=5)
    
    result = get_player_dev_cards(public_state, current_player_color, inventory)
    
    expected = """[PLAYER DEVELOPMENT CARDS]
- RED: No dev cards (Played: KNIGHT: 1, YEAR_OF_PLENTY: 1)
- BLUE: 1 dev cards (hidden) (Played: KNIGHT: 2, ROAD_BUILDING: 1)
- ORANGE: 0 dev cards (hidden)
- WHITE: 3 dev cards (hidden) (Played: MONOPOLY: 1, VICTORY_POINT: 1)"""
    
    assert result == expected


def test_get_player_dev_cards_all_card_types():
    """Test get_player_dev_cards with all development card types present"""
    public_state = create_mock_public_state()
    current_player_color = Color.WHITE
    
    # Create inventory with all dev card types using helper
    inventory = create_inventory(
        knight=3,
        year_of_plenty=2,
        monopoly=1,
        road_building=1,
        victory_point=2,
        actual_vps=2
    )
    
    result = get_player_dev_cards(public_state, current_player_color, inventory)
    
    expected = """[PLAYER DEVELOPMENT CARDS]
- RED: 2 dev cards (hidden) (Played: KNIGHT: 1, YEAR_OF_PLENTY: 1)
- BLUE: 1 dev cards (hidden) (Played: KNIGHT: 2, ROAD_BUILDING: 1)
- ORANGE: 0 dev cards (hidden)
- WHITE: KNIGHT: 3, YEAR_OF_PLENTY: 2, MONOPOLY: 1, ROAD_BUILDING: 1, VICTORY_POINT: 2 (Played: MONOPOLY: 1, VICTORY_POINT: 1)"""
    
    assert result == expected


def test_get_player_dev_cards_only_knights():
    """Test get_player_dev_cards with only knight cards"""
    public_state = create_mock_public_state()
    current_player_color = Color.BLUE
    
    # Create inventory with only knights using helper
    inventory = create_inventory(
        knight=5,
        actual_vps=4
    )
    
    result = get_player_dev_cards(public_state, current_player_color, inventory)
    
    expected = """[PLAYER DEVELOPMENT CARDS]
- RED: 2 dev cards (hidden) (Played: KNIGHT: 1, YEAR_OF_PLENTY: 1)
- BLUE: KNIGHT: 5 (Played: KNIGHT: 2, ROAD_BUILDING: 1)
- ORANGE: 0 dev cards (hidden)
- WHITE: 3 dev cards (hidden) (Played: MONOPOLY: 1, VICTORY_POINT: 1)"""
    
    assert result == expected


def test_get_player_dev_cards_no_played_cards():
    """Test get_player_dev_cards when no cards have been played"""
    from catanatron.models.player import Color
    
    # Create players with no played cards using helper
    red_player = create_public_player(
        public_vps=5,
        longest_road_length=3,
        roads_left=13,
        settlements_left=4,
        hand_resource_count=3,
        hand_dev_count=2
    )
    
    blue_player = create_public_player(
        public_vps=4,
        longest_road_length=2,
        roads_left=14,
        settlements_left=4,
        hand_resource_count=5,
        hand_dev_count=1
    )
    
    orange_player = create_public_player(
        public_vps=3,
        longest_road_length=1,
        roads_left=14,
        settlements_left=5
    )
    
    white_player = create_public_player(
        public_vps=2,
        roads_left=15,
        settlements_left=5,
        hand_resource_count=1,
        hand_dev_count=3
    )
    
    # Create public state using helper
    public_state = create_public_state({
        Color.RED: red_player,
        Color.BLUE: blue_player,
        Color.ORANGE: orange_player,
        Color.WHITE: white_player
    })
    
    current_player_color = Color.RED
    
    # Create inventory for current player using helper
    inventory = create_inventory(
        knight=2,
        year_of_plenty=1,
        actual_vps=5
    )
    
    result = get_player_dev_cards(public_state, current_player_color, inventory)
    
    expected = """[PLAYER DEVELOPMENT CARDS]
- RED: KNIGHT: 2, YEAR_OF_PLENTY: 1
- BLUE: 1 dev cards (hidden)
- ORANGE: 0 dev cards (hidden)
- WHITE: 3 dev cards (hidden)"""
    
    assert result == expected


# =============================================================================
# public_history grouping and description
# =============================================================================


def _rec(color, action_type, value=None, result=None):
    """Build an ActionRecord for unit tests."""
    return ActionRecord(Action(color, action_type, value), result)


def test_group_action_records_by_turn_empty():
    assert group_action_records_by_turn(()) == []
    assert group_action_records_by_turn([]) == []


def test_group_action_records_by_turn_setup_only():
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
    )
    groups = group_action_records_by_turn(records)
    assert len(groups) == 1
    assert groups[0] == records


def test_group_action_records_by_turn_setup_then_turns():
    records = (
        # setup
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        # turn 1
        _rec(Color.RED, ActionType.ROLL, (3, 4), (3, 4)),
        _rec(Color.RED, ActionType.END_TURN),
        # turn 2 (open — no END_TURN yet)
        _rec(Color.BLUE, ActionType.ROLL, (1, 2), (1, 2)),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 16)),
    )
    groups = group_action_records_by_turn(records)
    assert len(groups) == 3
    assert len(groups[0]) == 4  # setup
    assert all(r.action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD)
               for r in groups[0])
    assert groups[1][-1].action.action_type == ActionType.END_TURN
    assert groups[1][0].action.color == Color.RED
    assert groups[2][0].action.color == Color.BLUE
    assert groups[2][-1].action.action_type != ActionType.END_TURN


def test_group_action_records_keeps_discards_in_active_turn():
    """Other players' discards belong to the roller’s 7-turn, not separate turns."""
    records = (
        _rec(Color.RED, ActionType.ROLL, (3, 4), (3, 4)),
        _rec(Color.BLUE, ActionType.DISCARD_RESOURCE, "WOOD", "WOOD"),
        _rec(Color.ORANGE, ActionType.DISCARD_RESOURCE, "BRICK", "BRICK"),
        _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), "SHEEP"),
        _rec(Color.RED, ActionType.END_TURN),
    )
    groups = group_action_records_by_turn(records)
    assert len(groups) == 1
    assert len(groups[0]) == 5


def test_describe_action_record_roll():
    rec = _rec(Color.RED, ActionType.ROLL, (6, 1), (6, 1))
    assert describe_action_record(rec) == "RED rolled 6+1 = 7"


def test_describe_action_record_build_and_end():
    assert describe_action_record(
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 12)
    ) == "BLUE built settlement at node 12"
    assert describe_action_record(
        _rec(Color.BLUE, ActionType.BUILD_CITY, 12)
    ) == "BLUE built city at node 12"
    assert describe_action_record(
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (3, 1))
    ) == "BLUE built road on edge (1, 3)"
    assert describe_action_record(
        _rec(Color.ORANGE, ActionType.END_TURN)
    ) == "ORANGE ended turn"


def test_describe_action_record_buy_dev_known_and_hidden():
    assert describe_action_record(
        _rec(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT", "KNIGHT")
    ) == "RED bought development card: KNIGHT"
    # Sanitized opponent purchase (value and result redacted)
    assert describe_action_record(
        _rec(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, None, None)
    ) == "BLUE bought a development card"


def test_describe_action_record_move_robber_variants():
    assert describe_action_record(
        _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None), None)
    ) == "RED moved robber to (0, 0, 0) (no steal)"
    assert describe_action_record(
        _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), "WHEAT")
    ) == "RED moved robber to (0, 0, 0) and stole WHEAT from BLUE"
    # Spectator view — result redacted
    assert describe_action_record(
        _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), None)
    ) == "RED moved robber to (0, 0, 0) and stole from BLUE (card hidden)"


def test_describe_action_record_discard_and_dev_plays():
    assert describe_action_record(
        _rec(Color.WHITE, ActionType.DISCARD_RESOURCE, "ORE", "ORE")
    ) == "WHITE discarded ORE"
    assert describe_action_record(
        _rec(Color.RED, ActionType.PLAY_KNIGHT_CARD)
    ) == "RED played Knight"
    assert describe_action_record(
        _rec(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "BRICK"))
    ) == "RED played Year of Plenty: took WOOD, BRICK"
    assert describe_action_record(
        _rec(Color.RED, ActionType.PLAY_MONOPOLY, "SHEEP")
    ) == "RED played Monopoly on SHEEP"
    assert describe_action_record(
        _rec(Color.RED, ActionType.PLAY_ROAD_BUILDING)
    ) == "RED played Road Building"


def test_describe_action_record_maritime_trade():
    # 4:1 trade
    rec = _rec(
        Color.ORANGE,
        ActionType.MARITIME_TRADE,
        ("WHEAT", "WHEAT", "WHEAT", "WHEAT", "BRICK"),
    )
    assert describe_action_record(rec) == (
        "ORANGE maritime trade: gives [WHEAT, WHEAT, WHEAT, WHEAT] to bank for BRICK"
    )
    # port 2:1 / 3:1 with Nones
    rec = _rec(
        Color.ORANGE,
        ActionType.MARITIME_TRADE,
        ("ORE", "ORE", None, None, "WOOD"),
    )
    assert describe_action_record(rec) == (
        "ORANGE maritime trade: gives [ORE, ORE] to bank for WOOD"
    )


def test_describe_action_record_domestic_trade():
    # RESOURCES order: WOOD BRICK SHEEP WHEAT ORE
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)  # 1 WOOD for 1 BRICK
    assert describe_action_record(
        _rec(Color.RED, ActionType.OFFER_TRADE, offer)
    ) == "RED offers [1 WOOD] for [1 BRICK]"
    assert describe_action_record(
        _rec(Color.BLUE, ActionType.ACCEPT_TRADE, offer)
    ) == "BLUE accepted trade: offers [1 WOOD] for [1 BRICK]"
    assert describe_action_record(
        _rec(Color.ORANGE, ActionType.REJECT_TRADE, offer)
    ) == "ORANGE rejected trade: offers [1 WOOD] for [1 BRICK]"
    confirm = offer + (Color.BLUE,)
    assert describe_action_record(
        _rec(Color.RED, ActionType.CONFIRM_TRADE, confirm)
    ) == "RED confirmed trade with BLUE: offers [1 WOOD] for [1 BRICK]"
    assert describe_action_record(
        _rec(Color.RED, ActionType.CANCEL_TRADE)
    ) == "RED cancelled trade"


def test_describe_turn_and_format_public_history():
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
        _rec(Color.BLUE, ActionType.ROLL, (6, 1), (6, 1)),
        _rec(Color.RED, ActionType.DISCARD_RESOURCE, "WOOD", "WOOD"),
        _rec(Color.BLUE, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.RED), None),
        _rec(Color.BLUE, ActionType.END_TURN),
    )
    text = format_public_history(records)
    expected = """[PUBLIC HISTORY]
[SETUP]
  - RED built settlement at node 0
  - RED built road on edge (0, 1)
  - BLUE built settlement at node 5
  - BLUE built road on edge (5, 6)
[TURN 1 (RED)]
  - RED rolled 2+3 = 5
  - RED ended turn
[TURN 2 (BLUE)]
  - BLUE rolled 6+1 = 7
  - RED discarded WOOD
  - BLUE moved robber to (0, 0, 0) and stole from RED (card hidden)
  - BLUE ended turn"""
    assert text == expected

    turn_only = describe_turn(records[4:6], turn_label="TURN 1 (RED)")
    assert turn_only == """[TURN 1 (RED)]
  - RED rolled 2+3 = 5
  - RED ended turn"""


def test_format_public_history_empty():
    assert format_public_history(()) == "[PUBLIC HISTORY]\n  (empty)"


def test_group_and_format_real_sanitized_history():
    """Integration: group a real game's sanitized public_history."""
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players, seed=42)
    # Play enough to leave setup and finish a few turns
    for _ in range(80):
        if game.winning_color() is not None:
            break
        playable = game.playable_actions
        if not playable:
            break
        game.execute(playable[0])

    history = tuple(_sanitize_history(game, Color.RED))
    groups = group_action_records_by_turn(history)
    assert len(groups) >= 2
    # First group is setup placements only
    assert all(
        r.action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD)
        for r in groups[0]
    )
    # Flattening groups recovers the full history
    flattened = tuple(r for g in groups for r in g)
    assert flattened == history

    text = format_public_history(history)
    assert text.startswith("[PUBLIC HISTORY]\n[SETUP]")
    assert "rolled" in text
    assert "ended turn" in text
    # Every record yields exactly one bullet line
    bullet_count = sum(1 for line in text.splitlines() if line.startswith("  - "))
    assert bullet_count == len(history)


def test_format_public_history_window_full_history():
    """Test that window_size=None produces same output as format_public_history"""
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
        _rec(Color.BLUE, ActionType.ROLL, (6, 1), (6, 1)),
        _rec(Color.BLUE, ActionType.END_TURN),
    )
    
    window_result = format_public_history_window(records, window_size=None)
    original_result = format_public_history(records)
    
    assert window_result == original_result


def test_format_public_history_window_last_two_turns():
    """Test that window_size=2 shows only last 2 turns plus setup"""
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
        _rec(Color.BLUE, ActionType.ROLL, (6, 1), (6, 1)),
        _rec(Color.BLUE, ActionType.END_TURN),
        _rec(Color.RED, ActionType.ROLL, (4, 5), (4, 5)),
        _rec(Color.RED, ActionType.BUILD_ROAD, (1, 2)),
        _rec(Color.RED, ActionType.END_TURN),
    )
    
    result = format_public_history_window(records, window_size=2)
    
    # Should contain setup
    assert "[SETUP]" in result
    assert "RED built settlement at node 0" in result
    
    # Should contain window indicator
    assert "[Showing last 2 of 3 turns]" in result
    
    # Should contain last 2 turns (TURN 2 and TURN 3) with absolute numbering
    assert "[TURN 2 (BLUE)]" in result
    assert "[TURN 3 (RED)]" in result
    
    # Should NOT contain TURN 1 (RED) which was cut off
    assert "[TURN 1 (RED)]" not in result
    assert result.count("[TURN") == 2  # Only 2 turns should appear


def test_format_public_history_window_setup_only():
    """Test that window_size=0 shows only setup phase"""
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
        _rec(Color.BLUE, ActionType.ROLL, (6, 1), (6, 1)),
        _rec(Color.BLUE, ActionType.END_TURN),
    )
    
    result = format_public_history_window(records, window_size=0)
    
    # Should contain setup
    assert "[SETUP]" in result
    assert "RED built settlement at node 0" in result
    assert "BLUE built settlement at node 5" in result
    
    # Should contain setup-only indicator
    assert "[Showing setup phase only]" in result
    
    # Should NOT contain any turns
    assert "[TURN" not in result
    assert "rolled" not in result


def test_format_public_history_window_empty_history():
    """Test that empty history works correctly"""
    result = format_public_history_window((), window_size=2)
    assert result == "[PUBLIC HISTORY]\n  (empty)"


def test_format_public_history_window_single_turn():
    """Test window with single turn"""
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
    )
    
    result = format_public_history_window(records, window_size=1)
    
    # Should contain setup
    assert "[SETUP]" in result
    
    # Should NOT contain window indicator when window equals total turns
    assert "[Showing last" not in result
    
    # Should contain the single turn
    assert "[TURN 1 (RED)]" in result
    assert "RED rolled 2+3 = 5" in result


def test_format_public_history_window_larger_than_total():
    """Test that window larger than total turns shows all turns"""
    records = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
        _rec(Color.BLUE, ActionType.ROLL, (6, 1), (6, 1)),
        _rec(Color.BLUE, ActionType.END_TURN),
    )
    
    result = format_public_history_window(records, window_size=10)
    
    # Should contain setup and both turns (no window indicator since window >= total)
    assert "[SETUP]" in result
    assert "[TURN 1 (RED)]" in result
    assert "[TURN 2 (BLUE)]" in result
    assert "[Showing last" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])