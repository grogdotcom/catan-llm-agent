"""
Exact-string unit tests for every playable-action case.

Each test asserts the *exact* string returned by ``_label_action`` /
``build_moves`` / ``format_moves`` for a single ActionType. This locks the
LLM-facing ``[PLAYABLE MOVES]`` list so refactors cannot silently change
prompts.

Covers all 18 ActionTypes handled in game_formatter._label_action, plus the
compound-move expansions (Knight→robber, initial settlement+road, Road Building)
and the enriched details added for settlements/cities/roads/robber/longest-road.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "venv/lib/python3.14/site-packages"))

import pytest
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color
from catanatron.models.board import STATIC_GRAPH
from catan_llm.game_formatter import (
    _label_action,
    build_moves,
    format_moves,
    Move,
    _describe_node,
    _player_longest_road_length,
    _longest_road_suffix,
    _road_node_detail,
)
from catanatron.models.public_state import PublicState, PublicBoard, PublicMap, PublicPlayer
from catanatron.models.enums import SETTLEMENT, CITY
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers to build tiny deterministic public_states for detailed cases
# ---------------------------------------------------------------------------

def _mock_public_state_for_node5():
    """Minimal public_state where Node 5 = Tile0: 8 WOOD (5 pips), no port,
    land_nodes={0,5,...}, one tile, no buildings/roads. Used to hard-code
    expected settlement/city/road strings.
    """
    tiles = {0: (getattr(__import__("catanatron.models.enums", fromlist=["WOOD"]), "WOOD"), 8)}
    # Need actual FastResource enum, use string via mock? Use real enum
    from catanatron.models.enums import WOOD
    tiles = {0: (WOOD, 8), 1: (WOOD, 6)}
    tile_coordinates = {0: (0, 0, 0), 1: (1, -1, 0)}
    # Node adjacency: 0 touches tile0, 5 touches tile0+tile1, etc.
    adjacent_tiles = {
        0: (0,),
        5: (0, 1),
        1: (0,),
        6: (1,),
    }
    land_nodes = frozenset([0, 1, 5, 6, 20])
    ports = {}  # no ports to keep expectation simple
    public_map = PublicMap(
        tiles={0: (WOOD, 8), 1: (WOOD, 6)},
        tile_coordinates=tile_coordinates,
        ports=ports,
        adjacent_tiles=adjacent_tiles,
        land_nodes=land_nodes,
    )
    buildings = {}
    roads = {}
    board = PublicBoard(
        buildings=buildings,
        roads=roads,
        robber_tile_id=1,
        longest_road_color=None,
        longest_road_length=0,
        map=public_map,
    )
    players = {
        Color.RED: PublicPlayer(public_vps=0, has_army=False, has_road=False, longest_road_length=0, roads_left=15, settlements_left=5, cities_left=4, has_rolled=False, hand_resource_count=0, hand_dev_count=0, played_knight=0, played_monopoly=0, played_road_building=0, played_year_of_plenty=0, played_victory_point=0),
        Color.BLUE: PublicPlayer(public_vps=0, has_army=False, has_road=False, longest_road_length=0, roads_left=15, settlements_left=5, cities_left=4, has_rolled=False, hand_resource_count=0, hand_dev_count=0, played_knight=0, played_monopoly=0, played_road_building=0, played_year_of_plenty=0, played_victory_point=0),
    }
    return PublicState(board=board, players=players)


def _mock_public_state_with_settlements():
    """State where RED has settlement at 0, BLUE at 1, used for robber tests."""
    from catanatron.models.enums import WOOD, BRICK
    public_map = PublicMap(
        tiles={0: (WOOD, 8), 1: (BRICK, 6)},
        tile_coordinates={0: (0, 0, 0), 1: (1, -1, 0)},
        ports={},
        adjacent_tiles={0: (0,), 1: (0, 1), 5: (1,)},
        land_nodes=frozenset([0, 1, 5]),
    )
    buildings = {
        0: (Color.RED, SETTLEMENT),
        1: (Color.BLUE, SETTLEMENT),
    }
    board = PublicBoard(
        buildings=buildings,
        roads={},
        robber_tile_id=1,
        longest_road_color=None,
        longest_road_length=0,
        map=public_map,
    )
    players = {
        Color.RED: PublicPlayer(public_vps=0, has_army=False, has_road=False, longest_road_length=0, roads_left=15, settlements_left=4, cities_left=4, has_rolled=False, hand_resource_count=2, hand_dev_count=0, played_knight=0, played_monopoly=0, played_road_building=0, played_year_of_plenty=0, played_victory_point=0),
        Color.BLUE: PublicPlayer(public_vps=0, has_army=False, has_road=False, longest_road_length=0, roads_left=15, settlements_left=4, cities_left=4, has_rolled=False, hand_resource_count=3, hand_dev_count=0, played_knight=0, played_monopoly=0, played_road_building=0, played_year_of_plenty=0, played_victory_point=0),
    }
    return PublicState(board=board, players=players)


# ---------------------------------------------------------------------------
# 1. Simple _label_action exact strings with public_state=None
# ---------------------------------------------------------------------------

def test_label_roll_exact():
    a = Action(Color.RED, ActionType.ROLL, None)
    assert _label_action(a, None) == "Roll the dice"


def test_label_end_turn_exact():
    a = Action(Color.RED, ActionType.END_TURN, None)
    assert _label_action(a, None) == "End turn"


def test_label_build_road_simple_exact():
    a = Action(Color.RED, ActionType.BUILD_ROAD, (5, 0))
    # edge is sorted
    assert _label_action(a, None) == "Build road on edge (0, 5)"


def test_label_build_settlement_simple_exact():
    a = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 12)
    assert _label_action(a, None) == "Build settlement at node 12"


def test_label_build_city_simple_exact():
    a = Action(Color.RED, ActionType.BUILD_CITY, 12)
    assert _label_action(a, None) == "Build city at node 12"


def test_label_buy_dev_card_exact():
    a = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    assert _label_action(a, None) == "Buy a development card"


def test_label_play_knight_exact():
    a = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)
    assert _label_action(a, None) == "Play Knight (then move the robber)"


def test_label_play_year_of_plenty_exact():
    a = Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "SHEEP"))
    assert _label_action(a, None) == "Play Year of Plenty: take WOOD, SHEEP"


def test_label_play_monopoly_exact():
    a = Action(Color.RED, ActionType.PLAY_MONOPOLY, "ORE")
    assert _label_action(a, None) == "Play Monopoly: steal all ORE"


def test_label_play_road_building_exact():
    a = Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
    assert _label_action(a, None) == "Play Road Building (then build two roads)"


def test_label_move_robber_no_steal_exact():
    # With no public_state, falls back to coordinate label
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None))
    assert _label_action(a, None) == "Move robber to (0, 0, 0) (no steal)"


def test_label_move_robber_steal_exact():
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((1, -1, 0), Color.BLUE))
    assert _label_action(a, None) == "Move robber to (1, -1, 0) and steal from BLUE"


def test_label_discard_exact():
    a = Action(Color.RED, ActionType.DISCARD_RESOURCE, "WOOD")
    assert _label_action(a, None) == "Discard one WOOD"


def test_label_maritime_trade_exact():
    # 4 WOOD -> BRICK
    a = Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "BRICK"))
    assert _label_action(a, None) == "Maritime trade: gives [WOOD, WOOD, WOOD, WOOD] to bank for BRICK"


def test_label_maritime_trade_port_exact():
    a = Action(Color.RED, ActionType.MARITIME_TRADE, ("ORE", "ORE", None, None, "WOOD"))
    assert _label_action(a, None) == "Maritime trade: gives [ORE, ORE] to bank for WOOD"


def test_label_offer_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)  # 1 WOOD for 1 BRICK
    a = Action(Color.RED, ActionType.OFFER_TRADE, offer)
    assert _label_action(a, None) == "Offer trade: offers [1 WOOD] for [1 BRICK]"


def test_label_accept_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    a = Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer)
    assert _label_action(a, None) == "Accept trade: offers [1 WOOD] for [1 BRICK]"


def test_label_reject_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    a = Action(Color.BLUE, ActionType.REJECT_TRADE, offer)
    assert _label_action(a, None) == "Reject trade: offers [1 WOOD] for [1 BRICK]"


def test_label_confirm_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    confirm = offer + (Color.BLUE,)
    a = Action(Color.RED, ActionType.CONFIRM_TRADE, confirm)
    assert _label_action(a, None) == "Confirm trade with BLUE: offers [1 WOOD] for [1 BRICK]"


def test_label_cancel_trade_exact():
    a = Action(Color.RED, ActionType.CANCEL_TRADE, None)
    assert _label_action(a, None) == "Cancel trade"


def test_label_fallback_unknown_exact():
    class Dummy:
        name = "CUSTOM"
    a = Action(Color.RED, Dummy(), {"x": 1})
    # when action_type is not a known ActionType, falls back to f"{name}: value=..."
    assert _label_action(a, None) == "CUSTOM: value={'x': 1}"


# ---------------------------------------------------------------------------
# 2. Enriched settlement / city / road / robber with public_state
# ---------------------------------------------------------------------------

def test_label_build_settlement_enriched_exact():
    ps = _mock_public_state_for_node5()
    a = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 5)
    # Node 5 touches Tile0: 8 WOOD (5 pips), Tile1: 6 WOOD (5 pips) => total 10
    assert _label_action(a, ps) == "Build settlement at Node 5: (Tile 0: 8 WOOD (5 pips)), (Tile 1: 6 WOOD (5 pips)) Total: 10 pips"


def test_label_build_city_enriched_exact():
    ps = _mock_public_state_for_node5()
    a = Action(Color.RED, ActionType.BUILD_CITY, 0)
    # Node 0: only Tile0
    assert _label_action(a, ps) == "Build city at Node 0: (Tile 0: 8 WOOD (5 pips)) Total: 5 pips"


def test_label_build_road_enriched_exact_with_longest():
    ps = _mock_public_state_for_node5()
    # Edge (0,5) with network empty -> new tip is 0 (sorted first non-excluded). But with no roads, network is empty set? Let's use explicit color.
    # For a fresh board with no roads, player network is empty, so _own_network_nodes returns empty.
    # _road_node_detail will pick first sorted endpoint as tip (0), then forward neighbors of 0 excluding 5.
    # STATIC_GRAPH neighbors: 0->[1,5,20] ; forward from 0 excluding 5 => [1,20] but 20 not in land_nodes for this mock (land_nodes={0,1,5,6,20} includes 20)
    # So reaches Node0, extends toward Node1 and Node20
    a = Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))
    label = _label_action(a, ps)
    # Hard-coded expected based on mock above and longest road 0->1
    # Node0: Tile0 only => (Tile 0: 8 WOOD (5 pips)) Total:5
    # Node1: Tile0 only => same
    # Node20: no adjacent tiles in mock => "(no resource tiles)" Total:0 but actually 20 not in adjacent_tiles, so no resource tiles
    expected = (
        "Build road on edge (0, 5) | reaches Node 0: (Tile 0: 8 WOOD (5 pips)) Total: 5 pips [available] "
        "| extends toward Node 1: (Tile 0: 8 WOOD (5 pips)) Total: 5 pips [available], "
        "Node 20: (no resource tiles) Total: 0 pips [available] | Longest road: 0 -> 1 (+1)"
    )
    assert label == expected


def test_label_move_robber_enriched_exact():
    ps = _mock_public_state_with_settlements()
    # Tile 0 (0,0,0) WOOD 8 touches Node0 (RED) and Node1 (BLUE)
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None))
    label = _label_action(a, ps)
    assert label == "Move robber to Tile 0: 8 WOOD (5 pips) | Occupants: BLUE: settlement at Node 1 (5 pips) | 5 pips blocked, 3 cards; RED: settlement at Node 0 (5 pips) | 5 pips blocked, 2 cards (no steal)"


def test_label_move_robber_enriched_steal_exact():
    ps = _mock_public_state_with_settlements()
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE))
    label = _label_action(a, ps)
    assert label == "Move robber to Tile 0: 8 WOOD (5 pips) | Occupants: BLUE: settlement at Node 1 (5 pips) | 5 pips blocked, 3 cards; RED: settlement at Node 0 (5 pips) | 5 pips blocked, 2 cards and steal from BLUE"


# ---------------------------------------------------------------------------
# 3. build_moves compound expansions — exact strings
# ---------------------------------------------------------------------------

def test_build_moves_single_road_exact():
    a = Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))
    moves = build_moves([a], observation=None)
    assert len(moves) == 1
    assert moves[0].label == "Build road on edge (0, 5)"
    assert moves[0].actions == [a]


def test_build_moves_year_of_plenty_exact():
    a = Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "SHEEP"))
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Play Year of Plenty: take WOOD, SHEEP"


def test_build_moves_monopoly_exact():
    a = Action(Color.RED, ActionType.PLAY_MONOPOLY, "ORE")
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Play Monopoly: steal all ORE"


def test_build_moves_roll_exact():
    a = Action(Color.RED, ActionType.ROLL, None)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Roll the dice"


def test_build_moves_end_turn_exact():
    a = Action(Color.RED, ActionType.END_TURN, None)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "End turn"


# ---------------------------------------------------------------------------
# 4. format_moves — exact numbered list
# ---------------------------------------------------------------------------

def test_format_moves_numbered_exact():
    # Without observation, simple list
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None),
    ]
    moves = build_moves(actions, observation=None)
    text = format_moves(moves, observation=None)
    assert text == "[PLAYABLE MOVES]\n1. Roll the dice\n2. End turn"


def test_format_moves_empty_exact():
    text = format_moves([], observation=None)
    assert text == "[PLAYABLE MOVES]\n  (no moves available)"


def test_format_moves_with_phase_exact():
    from catanatron.models.enums import ActionPrompt
    from catanatron.models.observation import Observation
    ps = _mock_public_state_for_node5()
    obs = Observation(color=Color.RED, current_prompt=ActionPrompt.PLAY_TURN, public_state=ps, features={})
    moves = build_moves([Action(Color.RED, ActionType.ROLL, None)], observation=obs)
    text = format_moves(moves, observation=obs)
    # Phase line is second line
    assert text.startswith("[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n1. Roll the dice")


# ---------------------------------------------------------------------------
# 5. Longest-road suffix — exact strings
# ---------------------------------------------------------------------------

def test_longest_road_suffix_no_change_exact():
    ps = _mock_public_state_for_node5()
    # No roads: current 0, adding edge (0,5) => 1
    suffix = _longest_road_suffix(ps, Color.RED, [(0, 5)])
    assert suffix == " | Longest road: 0 -> 1 (+1)"


def test_longest_road_suffix_already_has_road_exact():
    # Create state where RED already has road (0,5)
    ps = _mock_public_state_for_node5()
    ps.board.roads[(0, 5)] = Color.RED
    ps.board.roads[(5, 0)] = Color.RED
    # Now building (0,5) again would be no change (but we test suffix for same edge)
    suffix = _longest_road_suffix(ps, Color.RED, [(0, 5)])
    assert suffix == " | Longest road: 1 -> 1 (no change)"


def test_longest_road_claim_exact():
    # Chain 5-0-1-2 length 3, adding (2,3) ->4 not yet claim, adding two more ->5 claim
    from catanatron.game import Game
    from catanatron.models.player import Player
    import random
    random.seed(0)
    # Use Game to get a real board with STATIC_GRAPH
    class Dummy(Player):
        def __init__(self, c): self.color=c; self.is_bot=True
        def decide(self, g, a): return a[0]
        def reset_state(self): pass
    game = Game([Dummy(Color.RED), Dummy(Color.BLUE), Dummy(Color.ORANGE), Dummy(Color.WHITE)], seed=11)
    game.state.board.build_settlement(Color.RED, 5, True)
    game.state.board.build_road(Color.RED, (5, 0))
    game.state.board.build_road(Color.RED, (0, 1))
    game.state.board.build_road(Color.RED, (1, 2))
    from catanatron.models.perspective_player import _build_public_state as bps
    ps = bps(game)
    # After 3 roads, adding (2,3) and (3,4) should yield 5 and claim
    suffix = _longest_road_suffix(ps, Color.RED, [(2, 3), (3, 4)])
    assert suffix == " | Longest road: 3 -> 5 (+2) [would claim Longest Road, +2 VP]"


# ---------------------------------------------------------------------------
# 6. Road detail — exact tip + forward formatting
# ---------------------------------------------------------------------------

def test_road_detail_tip_blocked_occupied_exact():
    ps = _mock_public_state_for_node5()
    ps.board.buildings[0] = (Color.BLUE, SETTLEMENT)
    label = _road_node_detail(ps, (0, 5), network_nodes=set(), extra_occupied=None)
    # Tip 0 is occupied, so no extends
    assert label == " | reaches Node 0: (Tile 0: 8 WOOD (5 pips)) Total: 5 pips [blocked (occupied by BLUE settlement at Node 0)]"


def test_road_detail_tip_blocked_too_close_shows_extends_exact():
    ps = _mock_public_state_for_node5()
    # Simulate future settlement at 0, tip 5 is too close
    detail = _road_node_detail(ps, (0, 5), network_nodes={0}, extra_occupied={0}, extra_occupied_color=Color.RED)
    # Tip is 5, blocked too close to RED at 0, but forward nodes still shown
    assert "reaches Node 5:" in detail
    assert "[blocked (too close to RED settlement at Node 0)]" in detail
    assert "extends toward" in detail


# ---------------------------------------------------------------------------
# 7. Additional exact playable-move cases via build_moves
# ---------------------------------------------------------------------------

def test_build_moves_discard_exact():
    a = Action(Color.RED, ActionType.DISCARD_RESOURCE, "WOOD")
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Discard one WOOD"


def test_build_moves_maritime_trade_exact():
    a = Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "BRICK"))
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Maritime trade: gives [WOOD, WOOD, WOOD, WOOD] to bank for BRICK"


def test_build_moves_offer_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    a = Action(Color.RED, ActionType.OFFER_TRADE, offer)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Offer trade: offers [1 WOOD] for [1 BRICK]"


def test_build_moves_accept_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    a = Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Accept trade: offers [1 WOOD] for [1 BRICK]"


def test_build_moves_reject_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    a = Action(Color.BLUE, ActionType.REJECT_TRADE, offer)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Reject trade: offers [1 WOOD] for [1 BRICK]"


def test_build_moves_confirm_trade_exact():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    confirm = offer + (Color.BLUE,)
    a = Action(Color.RED, ActionType.CONFIRM_TRADE, confirm)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Confirm trade with BLUE: offers [1 WOOD] for [1 BRICK]"


def test_build_moves_cancel_trade_exact():
    a = Action(Color.RED, ActionType.CANCEL_TRADE, None)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Cancel trade"


def test_build_moves_buy_dev_card_exact():
    a = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Buy a development card"


def test_build_moves_initial_settlement_exact():
    # Uses tiny mock to hard-code expected settlement+road + longest suffix
    ps = _mock_public_state_for_node5()
    a = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 0)
    from catan_llm.game_formatter import _setup_settlement_moves
    moves = _setup_settlement_moves(a, ps)
    # First road is (0,1) — exact string from earlier manual run
    expected = (
        "Build settlement at Node 0: (Tile 0: 8 WOOD (5 pips)) Total: 5 pips -> build road (0, 1) "
        "| reaches Node 1: (Tile 0: 8 WOOD (5 pips)) Total: 5 pips [blocked (too close to RED settlement at Node 0)] "
        "| extends toward Node 2: (no resource tiles) Total: 0 pips [blocked (water/non-land)], "
        "Node 6: (Tile 1: 6 WOOD (5 pips)) Total: 5 pips [available] | Longest road: 0 -> 1 (+1)"
    )
    assert moves[0].label == expected


def test_build_moves_knight_bundling_exact():
    ps = _mock_public_state_with_settlements()
    knight = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)
    from catan_llm.game_formatter import _knight_moves
    moves = _knight_moves(knight, ps)
    # First follow-up is smallest tile_id (0) with victim BLUE (since BLUE at Node1)
    # Tile 0 detail includes both occupants, sorted BLUE then RED
    first = moves[0]
    assert first.label == "Play Knight -> move robber to Tile 0: 8 WOOD (5 pips) | Occupants: BLUE: settlement at Node 1 (5 pips) | 5 pips blocked, 3 cards; RED: settlement at Node 0 (5 pips) | 5 pips blocked, 2 cards and steal from BLUE"
    assert len(first.actions) == 2
    assert first.actions[0] == knight
    assert first.actions[1].action_type == ActionType.MOVE_ROBBER


def test_build_moves_build_city_exact():
    a = Action(Color.RED, ActionType.BUILD_CITY, 10)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Build city at node 10"


def test_build_moves_build_settlement_simple_exact():
    a = Action(Color.RED, ActionType.BUILD_SETTLEMENT, 5)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Build settlement at node 5"


def test_build_moves_move_robber_exact():
    a = Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None))
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Move robber to (0, 0, 0) (no steal)"


def test_build_moves_play_knight_simple_exact():
    a = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)
    moves = build_moves([a], observation=None)
    # Without public_state, falls back to simple label (no robber detail)
    assert moves[0].label == "Play Knight (then move the robber)"


def test_build_moves_play_road_building_simple_exact():
    a = Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Play Road Building -> then build two roads"
    assert moves[0].actions == [a, "AUTO_ROAD", "AUTO_ROAD"]


def test_build_moves_discard_exact_via_build_moves():
    a = Action(Color.RED, ActionType.DISCARD_RESOURCE, "SHEEP")
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Discard one SHEEP"


def test_build_moves_maritime_exact_via_build_moves():
    a = Action(Color.RED, ActionType.MARITIME_TRADE, ("SHEEP", "SHEEP", "SHEEP", None, "WOOD"))
    moves = build_moves([a], observation=None)
    assert moves[0].label == "Maritime trade: gives [SHEEP, SHEEP, SHEEP] to bank for WOOD"
