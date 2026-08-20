"""
Deprecated shim — use ``catan_llm.format`` instead.

This module re-exports the full public surface from ``catan_llm.format``
so existing imports (``from catan_llm.game_formatter import ...``) keep
working. New code should import from ``catan_llm.format`` or its submodules
(e.g. ``catan_llm.format.board``, ``catan_llm.format.moves``).

Will be removed in a future version.
"""

import warnings

warnings.warn(
    "catan_llm.game_formatter is deprecated — use catan_llm.format instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new package so old imports keep working.
from catan_llm.format import (  # noqa: F401,F403
    AUTO_ROAD,
    AdjacentHexInfo,
    BoardOccupancyData,
    BuildingInfo,
    Move,
    PlayerBoardData,
    _SETUP_ACTION_TYPES,
    _describe_node,
    _format_coordinate,
    _format_maritime_trade_value,
    _format_resource_counts,
    _format_trade_offer_value,
    _is_buildable_node,
    _knight_moves,
    _knight_robber_followups,
    _label_action,
    _land_edges_from,
    _longest_road_suffix,
    _name_of,
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
    calculate_blocked_production,
    describe_action_record,
    describe_turn,
    format_board_occupancy_data,
    format_decision_prompt,
    format_decision_prompt_with_history,
    format_moves,
    format_playable_actions,
    format_public_history,
    format_public_history_window,
    format_robber_info,
    gather_board_occupancy_data,
    get_adjacent_hex_info,
    get_board_occupancy,
    get_full_board_map,
    get_game_state_summary,
    get_pip_count,
    get_player_dev_cards,
    get_player_resources,
    group_action_records_by_turn,
    parse_move,
    pick_auto_road,
    summarize_catan_actions,
)

# Also re-export submodules for attribute access if anyone does
# ``import catan_llm.game_formatter as gf`` and expects helpers.
from catan_llm.format import board as board  # noqa: F401
from catan_llm.format import history as history  # noqa: F401
from catan_llm.format import moves as moves  # noqa: F401
from catan_llm.format import players as players  # noqa: F401
from catan_llm.format import prompts as prompts  # noqa: F401
from catan_llm.format import utils as utils  # noqa: F401
from catan_llm.format import models as models  # noqa: F401
