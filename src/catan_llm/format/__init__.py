"""
catan_llm.format — LLM-friendly formatting for Catanatron game states.

Re-exports the full public surface so callers can do
``from catan_llm.format import get_full_board_map, build_moves`` or import
from the focused submodules (``catan_llm.format.board``, ``.moves``, etc.).

Submodules:
  - models   — dataclasses and constants
  - utils    — shared helpers (pip counts, name formatting)
  - board    — board map & occupancy
  - players  — player resources / dev cards
  - history  — action records & turn grouping
  - moves    — playable-action → Move expansion & formatting
  - prompts  — high-level decision prompts
"""

from catan_llm.format.models import (
    AdjacentHexInfo,
    BoardOccupancyData,
    BuildingInfo,
    PlayerBoardData,
    _SETUP_ACTION_TYPES,
)
from catan_llm.format.utils import (
    get_pip_count,
    _format_maritime_trade_value,
    _format_resource_counts,
    _format_trade_offer_value,
    _name_of,
    _format_coordinate,
)
from catan_llm.format.board import (
    calculate_blocked_production,
    format_board_occupancy_data,
    format_robber_info,
    gather_board_occupancy_data,
    get_adjacent_hex_info,
    get_board_occupancy,
    get_full_board_map,
)
from catan_llm.format.players import (
    get_player_dev_cards,
    get_player_resources,
    get_player_summary,
    get_players_summary,
)
from catan_llm.format.history import (
    describe_action_record,
    describe_turn,
    format_public_history,
    format_public_history_window,
    group_action_records_by_turn,
)
from catan_llm.format.prompts import (
    format_decision_prompt,
    format_decision_prompt_with_history,
    get_game_state_summary,
    summarize_catan_actions,
)
from catan_llm.format.moves import (
    AUTO_ROAD,
    Move,
    _describe_node,
    _is_buildable_node,
    _knight_moves,
    _knight_robber_followups,
    _label_action,
    _land_edges_from,
    _longest_road_suffix,
    _node_buildability_detail,
    _node_pip_total,
    _own_network_nodes,
    _player_longest_road_length,
    _road_building_moves,
    _road_node_detail,
    _robber_tile_detail,
    _setup_settlement_moves,
    _tile_id_for_coordinate,
    build_moves,
    format_moves,
    format_playable_actions,
    parse_move,
    pick_auto_road,
)

__all__ = [
    # models
    "AdjacentHexInfo",
    "BuildingInfo",
    "PlayerBoardData",
    "BoardOccupancyData",
    "_SETUP_ACTION_TYPES",
    # utils
    "get_pip_count",
    "_name_of",
    "_format_resource_counts",
    "_format_trade_offer_value",
    "_format_maritime_trade_value",
    "_format_coordinate",
    # board
    "get_adjacent_hex_info",
    "get_full_board_map",
    "gather_board_occupancy_data",
    "calculate_blocked_production",
    "format_robber_info",
    "format_board_occupancy_data",
    "get_board_occupancy",
    # players
    "get_player_resources",
    "get_player_dev_cards",
    "get_players_summary",
    "get_player_summary",
    # history
    "describe_action_record",
    "group_action_records_by_turn",
    "describe_turn",
    "format_public_history",
    "format_public_history_window",
    # prompts
    "get_game_state_summary",
    "summarize_catan_actions",
    "format_decision_prompt",
    "format_decision_prompt_with_history",
    # moves
    "AUTO_ROAD",
    "Move",
    "build_moves",
    "format_moves",
    "format_playable_actions",
    "parse_move",
    "pick_auto_road",
    # moves internals (kept public for exact-string tests)
    "_describe_node",
    "_is_buildable_node",
    "_node_buildability_detail",
    "_node_pip_total",
    "_player_longest_road_length",
    "_longest_road_suffix",
    "_tile_id_for_coordinate",
    "_robber_tile_detail",
    "_road_node_detail",
    "_label_action",
    "_knight_robber_followups",
    "_knight_moves",
    "_own_network_nodes",
    "_land_edges_from",
    "_road_building_moves",
    "_setup_settlement_moves",
]
