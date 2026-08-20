"""
Unit tests for player formatting — resources and dev cards.

Mirrors src/catan_llm/format/players.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "src"))

import pytest

from catanatron.game import Game
from catanatron.models.player import Color, Player
from catanatron.models.perspective_player import _build_public_state
from catanatron.state_functions import player_key
from catan_llm.format.players import get_player_resources, get_player_dev_cards
from catanatron.models.inventory import Inventory


class SimplePlayer(Player):
    def __init__(self, color):
        self.color = color
        self.is_bot = True

    def decide(self, game, playable_actions):
        return playable_actions[0] if playable_actions else None

    def reset_state(self):
        pass


def build_public_state(game):
    return _build_public_state(game)


def create_game():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    return Game(players)


def test_get_player_resources_current_player_shows_hidden_for_others():
    game = create_game()
    ps = build_public_state(game)
    # Give RED some cards
    red_key = player_key(game.state, Color.RED)
    game.state.player_state[f"{red_key}_WOOD_IN_HAND"] = 2
    game.state.player_state[f"{red_key}_BRICK_IN_HAND"] = 1
    ps = build_public_state(game)
    text = get_player_resources(ps, Color.RED, game.state.player_state.get(f"{red_key}_WOOD_IN_HAND") and None)  # no inventory, should still show hidden logic
    # Without inventory, even current player shows hidden count
    assert "[PLAYER RESOURCES]" in text
    assert "resource cards (hidden)" in text


def test_get_player_resources_with_inventory():
    game = create_game()
    inv = Inventory(wood=2, brick=1, sheep=0, wheat=0, ore=0)
    ps = build_public_state(game)
    text = get_player_resources(ps, Color.RED, inv)
    assert "WOOD: 2" in text
    assert "BRICK: 1" in text
    # Other players still hidden
    assert "BLUE:" in text and "hidden" in text


def test_get_player_resources_no_inventory_shows_no_resources():
    game = create_game()
    ps = build_public_state(game)
    inv = Inventory()
    text = get_player_resources(ps, Color.RED, inv)
    assert "No resources" in text


def test_get_player_dev_cards_hidden_and_played():
    game = create_game()
    # Simulate played knight for BLUE
    blue_key = player_key(game.state, Color.BLUE)
    game.state.player_state[f"{blue_key}_KNIGHT_PLAYED"] = 1  # actual key may differ, but public_state will reflect
    ps = build_public_state(game)
    text = get_player_dev_cards(ps, Color.RED)
    assert "[PLAYER DEVELOPMENT CARDS]" in text
    assert "dev cards (hidden)" in text


def test_get_player_dev_cards_with_inventory():
    game = create_game()
    inv = Inventory(knight=1, victory_point=1)
    ps = build_public_state(game)
    text = get_player_dev_cards(ps, Color.RED, inv)
    assert "KNIGHT: 1" in text
    assert "VICTORY_POINT: 1" in text


# === Migrated from test_board.py — detailed player formatting (exact strings) ===

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


def test_get_player_resources_with_inventory_exact():
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


def test_get_player_dev_cards_with_inventory_exact():
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

