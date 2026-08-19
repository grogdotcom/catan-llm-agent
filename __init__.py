"""
Catanatron LLM - LLM-friendly formatting for Catanatron game states

Now works with Observation agent's public_state, features, and inventory instead of
direct Game/State access for better information hiding.
"""

from game_formatter import (
    get_full_board_map,
    get_board_occupancy,
    get_player_resources,
    get_player_dev_cards,
    get_game_state_summary,
    summarize_catan_actions,
    format_decision_prompt,
    get_pip_count,
    get_adjacent_hex_info,
    gather_board_occupancy_data,
    format_board_occupancy_data,
    PlayerBoardData,
    BoardOccupancyData,
    AdjacentHexInfo,
    BuildingInfo,
)

__all__ = [
    "get_full_board_map",
    "get_board_occupancy",
    "get_player_resources",
    "get_player_dev_cards",
    "get_game_state_summary",
    "summarize_catan_actions",
    "format_decision_prompt",
    "get_pip_count",
    "get_adjacent_hex_info",
    "gather_board_occupancy_data",
    "format_board_occupancy_data",
    "PlayerBoardData",
    "BoardOccupancyData",
    "AdjacentHexInfo",
    "BuildingInfo",
]