"""
Catanatron LLM - LLM-friendly formatting for Catanatron game states

Now works with Observation agent's public_state, features, and inventory instead of
direct Game/State access for better information hiding.

Deep module layout (see AGENTS.md):
  - catan_llm.domain      — shared records (DecisionRecord, Trajectory, ...)
  - catan_llm.decision    — decision surface (present/resolve/execute moves)
  - catan_llm.prompt      — canonical prompt builder + strategy block handling
  - catan_llm.teacher     — unified batch/inference transport + parsing
  - catan_llm.strategy    — strategy-chain lineage policy
  - catan_llm.dataset     — accepted SFT export record construction
  - catan_llm.evaluation  — offline metrics + run reporting
  - catan_llm.executor    — generic durable executor (store/runner/spec)
  - catan_llm.sft         — SFT checkpoint pipeline (selection/validation/provenance/cli + side table)
  - catan_llm.openai_batch — facade + CLI over catan_llm.teacher
"""

from catan_llm.format import (
    get_full_board_map,
    get_board_occupancy,
    get_player_resources,
    get_player_dev_cards,
    get_players_summary,
    get_player_summary,
    get_game_state_summary,
    summarize_catan_actions,
    format_decision_prompt,
    get_pip_count,
    get_adjacent_hex_info,
    gather_board_occupancy_data,
    format_board_occupancy_data,
    format_robber_info,
    calculate_blocked_production,
    describe_action_record,
    group_action_records_by_turn,
    describe_turn,
    format_public_history,
    PlayerBoardData,
    BoardOccupancyData,
    AdjacentHexInfo,
    BuildingInfo,
    Move,
    AUTO_ROAD,
    build_moves,
    format_moves,
    format_playable_actions,
    parse_move,
    pick_auto_road,
)

__all__ = [
    "get_full_board_map",
    "get_board_occupancy",
    "get_player_resources",
    "get_player_dev_cards",
    "get_players_summary",
    "get_player_summary",
    "get_game_state_summary",
    "summarize_catan_actions",
    "format_decision_prompt",
    "get_pip_count",
    "get_adjacent_hex_info",
    "gather_board_occupancy_data",
    "format_board_occupancy_data",
    "format_robber_info",
    "calculate_blocked_production",
    "describe_action_record",
    "group_action_records_by_turn",
    "describe_turn",
    "format_public_history",
    "PlayerBoardData",
    "BoardOccupancyData",
    "AdjacentHexInfo",
    "BuildingInfo",
    "Move",
    "AUTO_ROAD",
    "build_moves",
    "format_moves",
    "format_playable_actions",
    "parse_move",
    "pick_auto_road",
]