"""
Simple test to demonstrate the game formatter functionality
"""

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from game_formatter import (
    get_full_board_map,
    get_board_occupancy, 
    get_player_resources,
    get_player_dev_cards,
    get_game_state_summary,
    summarize_catan_actions,
    format_decision_prompt,
    gather_board_occupancy_data,
    PlayerBoardData,
    BoardOccupancyData,
    AdjacentHexInfo,
    BuildingInfo,
)

def test_formatter():
    # Create a simple game with random players
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE), 
        RandomPlayer(Color.WHITE),
    ]
    
    game = Game(players)
    
    # Play a few turns to get some interesting state
    # Just play one turn to get some state
    game.play(accumulators=[])
    
    print("=" * 80)
    print("GAME FORMATTER TEST")
    print("=" * 80)
    
    # Test individual components
    print("\n[1] FULL BOARD MAP:")
    print(get_full_board_map(game))
    
    print("\n[2] BOARD OCCUPANCY:")
    print(get_board_occupancy(game))
    
    # Test the new dataclass interface
    print("\n[2.5] BOARD OCCUPANCY DATA OBJECT:")
    occupancy_data = gather_board_occupancy_data(game.state)
    print(f"Type: {type(occupancy_data)}")
    print(f"Robber Coordinate: {occupancy_data.robber_coordinate}")
    print(f"Number of Players: {len(occupancy_data.players)}")
    for player_data in occupancy_data.players:
        print(f"  {player_data.color}: {len(player_data.settlements)} settlements, {len(player_data.cities)} cities, {len(player_data.roads)} roads")
        
        # Show detailed building info for first player with buildings
        if player_data.settlements or player_data.cities:
            print(f"    Detailed Building Info:")
            for building in player_data.settlements[:2]:  # Show first 2 settlements
                print(f"      Settlement at Node {building.node_id}:")
                print(f"        Total Pips: {building.total_pips}")
                print(f"        Port: {building.port if building.port else 'None'}")
                print(f"        Adjacent Hexes:")
                for adjacent_hex in building.adjacent_hexes:
                    print(f"          {adjacent_hex.resource} (Roll: {adjacent_hex.roll}, Pips: {adjacent_hex.pips})")
    
    print("\n[3] PLAYER RESOURCES:")
    print(get_player_resources(game))
    
    print("\n[4] PLAYER DEV CARDS:")
    print(get_player_dev_cards(game))
    
    print("\n[5] COMPLETE GAME STATE SUMMARY:")
    print(get_game_state_summary(game))
    
    # Test action formatting
    current_player = game.state.players[game.state.current_player_index]
    playable_actions = game.playable_actions
    
    print("\n[6] PLAYABLE ACTIONS:")
    print(summarize_catan_actions(playable_actions))
    
    print("\n[7] COMPLETE DECISION PROMPT:")
    print(format_decision_prompt(game, playable_actions, current_player.color.name))
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    test_formatter()