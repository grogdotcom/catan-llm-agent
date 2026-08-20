"""
Simple test to demonstrate the game formatter functionality
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'src'))

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.perspective_player import _build_public_state
from catan_llm.format.board import get_board_occupancy, get_full_board_map, gather_board_occupancy_data
from catan_llm.format.players import get_player_dev_cards, get_player_resources
from catan_llm.format.prompts import (
    format_decision_prompt,
    get_game_state_summary,
    summarize_catan_actions,
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

    public_state = _build_public_state(game)
    current_player = game.state.players[game.state.current_player_index]
    current_player_color = current_player.color
    current_prompt = game.state.current_prompt
    turn_number = game.state.current_turn_index
    playable_actions = game.playable_actions

    print("=" * 80)
    print("GAME FORMATTER TEST")
    print("=" * 80)

    # Test individual components
    print("\n[1] FULL BOARD MAP:")
    print(get_full_board_map(public_state))

    print("\n[2] BOARD OCCUPANCY:")
    print(get_board_occupancy(public_state))

    # Test the dataclass interface
    print("\n[2.5] BOARD OCCUPANCY DATA OBJECT:")
    occupancy_data = gather_board_occupancy_data(public_state)
    print(f"Type: {type(occupancy_data)}")
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
    print(get_player_resources(public_state, current_player_color))

    print("\n[4] PLAYER DEV CARDS:")
    print(get_player_dev_cards(public_state, current_player_color))

    print("\n[5] COMPLETE GAME STATE SUMMARY:")
    print(get_game_state_summary(public_state, current_player_color))

    print("\n[6] PLAYABLE ACTIONS:")
    print(summarize_catan_actions(playable_actions))

    print("\n[7] COMPLETE DECISION PROMPT:")
    print(format_decision_prompt(
        public_state,
        playable_actions,
        current_player_color.name,
        current_prompt,
        turn_number,
    ))

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    test_formatter()
