"""
Exact-string golden tests for board.py uncovered branches.

Covers:
  - 250->256, 257->260, 277  (ROBBER Unknown / robber_tile_id is None)
  - robber on DESERT vs resource tile with blocking
  - 322 hx.roll is None branch in _format_building_string
  - 316 DESERT skip branch
  - 424-425 non-canonical resource in format_starting_resources
  - _calculate_production desert vs resource
  - seating order preserved (shuffled Game seed / synthetic insertion order)
  - 182 duplicate roads dedup, 222/296 resource not in dict branches

Every test asserts an exact multiline literal.
Run: venv/bin/python -m pytest tests/format/test_board_full_coverage.py -vv
     venv/bin/python -m pytest --cov=catan_llm.format.board --cov-branch --cov-report=term-missing
"""

import sys
import os
import random
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from catanatron.game import Game
from catanatron.models.player import Color, Player
from catanatron.models.enums import CITY, SETTLEMENT
from catanatron.models.public_state import PublicState, PublicBoard, PublicMap, PublicPlayer

from catan_llm.format.board import (
    calculate_blocked_production,
    format_board_occupancy_data,
    format_robber_info,
    format_starting_resources,
    gather_board_occupancy_data,
    get_adjacent_hex_info,
    get_starting_resources,
    _calculate_production,
    _format_building_string,
)
from catan_llm.format.models import AdjacentHexInfo, BoardOccupancyData, BuildingInfo, PlayerBoardData
from catanatron.models.perspective_player import _build_public_state


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class S(Player):
    def __init__(self, c):
        super().__init__(c)
    def decide(self, g, a):
        return a[0] if a else None
    def reset_state(self):
        pass

def _empty_public_player():
    return PublicPlayer(
        public_vps=0, has_army=False, has_road=False, longest_road_length=0,
        roads_left=15, settlements_left=4, cities_left=4, has_rolled=False,
        hand_resource_count=0, hand_dev_count=0,
        played_knight=0, played_monopoly=0, played_road_building=0,
        played_year_of_plenty=0, played_victory_point=0,
    )

def _make_public_state(*, tiles, robber_tile_id, buildings=None, roads=None, players_order=None, adjacent_tiles=None, ports=None):
    """Build a minimal synthetic PublicState."""
    if buildings is None:
        buildings = {}
    if roads is None:
        roads = {}
    if players_order is None:
        players_order = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    if adjacent_tiles is None:
        adjacent_tiles = {}
    if ports is None:
        ports = {}
    tile_coordinates = {tid: (tid, 0, 0) for tid in tiles}
    pmap = PublicMap(
        tiles=tiles,
        tile_coordinates=tile_coordinates,
        ports=ports,
        adjacent_tiles=adjacent_tiles,
        land_nodes=frozenset(adjacent_tiles.keys()) if adjacent_tiles else frozenset(),
    )
    board = PublicBoard(
        buildings=buildings,
        roads=roads,
        robber_tile_id=robber_tile_id,
        longest_road_color=None,
        longest_road_length=0,
        map=pmap,
    )
    players = {c: _empty_public_player() for c in players_order}
    return PublicState(board=board, players=players)


# ---------------------------------------------------------------------------
# 1. ROBBER Unknown position  (lines 250->256, 257->260, 277)
# ---------------------------------------------------------------------------

def test_format_robber_info_unknown_position_exact():
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8), 1: (None, None)},
        robber_tile_id=None,
        players_order=[Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE],
    )
    result = format_robber_info(public_state, [])
    expected = "ROBBER: Unknown position | Blocking: None"
    assert result == expected


def test_format_robber_info_unknown_with_players_exact():
    # even with players, Unknown still Blocking: None (257->260 not entered)
    occ_players = [
        PlayerBoardData(color="RED", settlements=[BuildingInfo(node_id=0, adjacent_hexes=[AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=0)], port=None)], cities=[], roads=[]),
    ]
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8)},
        robber_tile_id=None,
    )
    result = format_robber_info(public_state, occ_players)
    expected = "ROBBER: Unknown position | Blocking: None"
    assert result == expected


def test_format_robber_info_robber_not_in_tiles_exact():
    # 250 false branch: robber_tile_id not in tiles -> treated like desert? but still tile_info from robber_resource None
    # tiles has 0 only, robber 999 not in dict => robber_resource stays None => DESERT line
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8)},
        robber_tile_id=999,
    )
    result = format_robber_info(public_state, [])
    expected = "ROBBER: Tile 999: DESERT | Blocking: None"
    assert result == expected


# ---------------------------------------------------------------------------
# 2. ROBBER on DESERT vs resource tile with blocking
# ---------------------------------------------------------------------------

def test_format_robber_info_desert_no_blocking_exact():
    # tile 11 desert, no adjacent buildings -> Blocking: None
    public_state = _make_public_state(
        tiles={11: (None, None), 0: ("WOOD", 8)},
        robber_tile_id=11,
    )
    result = format_robber_info(public_state, [])
    expected = "ROBBER: Tile 11: DESERT | Blocking: None"
    assert result == expected


def test_format_robber_info_desert_with_adjacent_buildings_still_none_exact():
    # Even if a building exists adjacent to desert, get_adjacent_hex_info would exclude desert,
    # so blocked remains None. Use synthetic occupancy where building claims adjacency to desert
    # but resource is DESERT (DESERT hex not counted as blocking? Actually calculate_blocked checks tile_id match
    # regardless of resource, but our synthetic building has DESERT resource adjacent to desert tile.
    # The blocked_pips would be 0 so still None. Use exact expectation.
    building = BuildingInfo(
        node_id=35,
        adjacent_hexes=[AdjacentHexInfo(resource="DESERT", roll=None, pips=0, tile_id=11)],
        port=None,
    )
    players = [PlayerBoardData(color="WHITE", settlements=[building], cities=[], roads=[])]
    public_state = _make_public_state(
        tiles={11: (None, None)},
        robber_tile_id=11,
    )
    result = format_robber_info(public_state, players)
    expected = "ROBBER: Tile 11: DESERT | Blocking: None"
    assert result == expected


def test_format_robber_info_resource_with_blocking_exact():
    # Synthetic: tile 0 is 8-Wd (5p), RED settlement adjacent to tile 0 (5p), BLUE settlement adjacent (2p via 11-Sh)
    # Use exact pips: RED 5p, BLUE 2p => sorted BLUE first
    red_settlement = BuildingInfo(
        node_id=0,
        adjacent_hexes=[AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=0)],
        port=None,
    )
    blue_settlement = BuildingInfo(
        node_id=5,
        adjacent_hexes=[AdjacentHexInfo(resource="SHEEP", roll=11, pips=2, tile_id=0)],
        port=None,
    )
    players = [
        PlayerBoardData(color="RED", settlements=[red_settlement], cities=[], roads=[]),
        PlayerBoardData(color="BLUE", settlements=[blue_settlement], cities=[], roads=[]),
    ]
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8), 1: ("WOOD", 11)},
        robber_tile_id=0,
    )
    result = format_robber_info(public_state, players)
    expected = "ROBBER: Tile 0: 8-Wd (5p) | Blocking BLUE: 2p, RED: 5p"
    assert result == expected


def test_format_robber_info_resource_city_multiplier_exact():
    # RED city adjacent to tile 0 doubles pips: 5*2=10
    red_city = BuildingInfo(
        node_id=0,
        adjacent_hexes=[AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=0)],
        port=None,
    )
    players = [
        PlayerBoardData(color="RED", settlements=[], cities=[red_city], roads=[]),
    ]
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8)},
        robber_tile_id=0,
    )
    result = format_robber_info(public_state, players)
    expected = "ROBBER: Tile 0: 8-Wd (5p) | Blocking RED: 10p"
    assert result == expected


# ---------------------------------------------------------------------------
# 3. _format_building_string with roll None (line 322) and DESERT skip (316)
# ---------------------------------------------------------------------------

def test_format_building_string_roll_none_exact():
    building = BuildingInfo(
        node_id=42,
        adjacent_hexes=[AdjacentHexInfo(resource="WOOD", roll=None, pips=3, tile_id=99)],
        port=None,
    )
    result = _format_building_string(building)
    expected = "Node 42 [Wd | 3p]"
    assert result == expected


def test_format_building_string_desert_skipped_exact():
    building = BuildingInfo(
        node_id=7,
        adjacent_hexes=[
            AdjacentHexInfo(resource="DESERT", roll=None, pips=0, tile_id=1),
            AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=0),
            AdjacentHexInfo(resource="DESERT", roll=None, pips=0, tile_id=2),
        ],
        port=None,
    )
    result = _format_building_string(building)
    # DESERT entries skipped, only Wd remains
    expected = "Node 7 [8 Wd | 5p]"
    assert result == expected


def test_format_building_string_only_desert_exact():
    building = BuildingInfo(
        node_id=35,
        adjacent_hexes=[AdjacentHexInfo(resource="DESERT", roll=None, pips=0, tile_id=11)],
        port=None,
    )
    result = _format_building_string(building)
    expected = "Node 35 [no tiles | 0p]"
    assert result == expected


def test_format_building_string_mixed_roll_none_and_present_exact():
    building = BuildingInfo(
        node_id=10,
        adjacent_hexes=[
            AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=0),
            AdjacentHexInfo(resource="SHEEP", roll=None, pips=2, tile_id=1),
            AdjacentHexInfo(resource="BRICK", roll=6, pips=5, tile_id=2),
        ],
        port=None,
    )
    result = _format_building_string(building)
    expected = "Node 10 [8 Wd, Sh, 6 Br | 12p]"
    assert result == expected


# ---------------------------------------------------------------------------
# 4. _calculate_production desert vs resource (line 296->295 etc)
# ---------------------------------------------------------------------------

def test_calculate_production_resource_and_desert_exact_via_occupancy():
    # Two buildings: one with WOOD 5p + DESERT 0p, one with DESERT only
    b1 = BuildingInfo(
        node_id=0,
        adjacent_hexes=[
            AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=0),
            AdjacentHexInfo(resource="DESERT", roll=None, pips=0, tile_id=11),
        ],
        port=None,
    )
    b2 = BuildingInfo(
        node_id=1,
        adjacent_hexes=[AdjacentHexInfo(resource="DESERT", roll=None, pips=0, tile_id=11)],
        port=None,
    )
    # direct _calculate_production
    total, res = _calculate_production([b1, b2], multiplier=1)
    assert total == 5
    expected_res = {"WOOD": 5, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}
    assert res == expected_res

    # also via format_board_occupancy_data exact string
    occ = BoardOccupancyData(players=[
        PlayerBoardData(color="RED", settlements=[b1, b2], cities=[], roads=[]),
        PlayerBoardData(color="BLUE", settlements=[], cities=[], roads=[]),
        PlayerBoardData(color="WHITE", settlements=[], cities=[], roads=[]),
        PlayerBoardData(color="ORANGE", settlements=[], cities=[], roads=[]),
    ])
    result = format_board_occupancy_data(occ)
    expected = """[CURRENT BOARD OCCUPANCY]
- RED: Total: 5p (Wd:5)
  * Settlements: Node 0 [8 Wd | 5p], Node 1 [no tiles | 0p]
- BLUE: Total: 0p
  * (no buildings/roads)
- WHITE: Total: 0p
  * (no buildings/roads)
- ORANGE: Total: 0p
  * (no buildings/roads)"""
    assert result == expected


def test_calculate_production_non_canonical_resource_branch_exact():
    # Resource GOLD not in RESOURCE dict -> should not increment resource_pips (296 false branch)
    b = BuildingInfo(
        node_id=0,
        adjacent_hexes=[AdjacentHexInfo(resource="GOLD", roll=8, pips=5, tile_id=0)],
        port=None,
    )
    total, res = _calculate_production([b], multiplier=1)
    # total includes pips via total_pips sum regardless of resource
    assert total == 5
    # GOLD not counted in resource_pips dict
    expected_res = {"WOOD": 0, "BRICK": 0, "SHEEP": 0, "WHEAT": 0, "ORE": 0}
    assert res == expected_res


def test_calculate_blocked_production_non_canonical_resource_exact():
    # 222 false branch: resource GOLD not in blocked_resource_pips dict
    building = BuildingInfo(
        node_id=0,
        adjacent_hexes=[AdjacentHexInfo(resource="GOLD", roll=8, pips=5, tile_id=0)],
        port=None,
    )
    players = [PlayerBoardData(color="RED", settlements=[building], cities=[], roads=[])]
    result = calculate_blocked_production(0, players)
    expected = {"RED": "5 pips"}
    assert result == expected


# ---------------------------------------------------------------------------
# 5. Non-canonical resource in format_starting_resources (424-425)
# ---------------------------------------------------------------------------

def test_format_starting_resources_non_canonical_single_exact():
    # Synthetic map: node 0 touches tiles 0 and 1, both GOLD
    # Tiles: 0-> GOLD 8, 1-> SHEEP 6, node 0 adjacent to both? Actually need variation.
    # Simpler: use GOLD as resource string not in RESOURCES
    tiles = {0: ("GOLD", 8)}
    adjacent_tiles = {0: (0,)}
    public_state = _make_public_state(
        tiles=tiles,
        robber_tile_id=0,
        adjacent_tiles=adjacent_tiles,
        ports={},
    )
    result = format_starting_resources(public_state, 0)
    # get_starting_resources returns ["GOLD"], format => "GOLD"
    expected = "GOLD"
    assert result == expected


def test_format_starting_resources_non_canonical_multiple_counts_exact():
    tiles = {0: ("GOLD", 8), 1: ("GOLD", 6), 2: ("WOOD", 5)}
    adjacent_tiles = {5: (0, 1, 2)}
    public_state = _make_public_state(
        tiles=tiles,
        robber_tile_id=0,
        adjacent_tiles=adjacent_tiles,
    )
    result = format_starting_resources(public_state, 5)
    # Counter: GOLD 2, WOOD 1
    # First loop for RESOURCES: WOOD 1 => "Wd"
    # Second loop for non-canonical: GOLD 2 => "2 GOLD"
    # joined: "Wd, 2 GOLD" ??? Actually order: first RESOURCES order then non-RESOURCES insertion order
    # RESOURCES order is WOOD first, so Wd appears before GOLD
    expected = "Wd, 2 GOLD"
    assert result == expected


def test_format_starting_resources_canonical_and_non_canonical_mixed_exact():
    tiles = {0: ("WOOD", 8), 1: ("GOLD", 6)}
    adjacent_tiles = {5: (0, 1)}
    public_state = _make_public_state(
        tiles=tiles,
        robber_tile_id=0,
        adjacent_tiles=adjacent_tiles,
    )
    result = format_starting_resources(public_state, 5)
    expected = "Wd, GOLD"
    assert result == expected


def test_get_starting_resources_filters_desert_exact():
    tiles = {0: (None, None), 1: ("WOOD", 8)}
    adjacent_tiles = {0: (0, 1)}
    public_state = _make_public_state(
        tiles=tiles,
        robber_tile_id=0,
        adjacent_tiles=adjacent_tiles,
    )
    result = get_starting_resources(public_state, 0)
    expected = ["WOOD"]
    assert result == expected
    # formatted exact
    formatted = format_starting_resources(public_state, 0)
    assert formatted == "Wd"


def test_format_starting_resources_none_exact():
    tiles = {0: (None, None)}
    adjacent_tiles = {0: (0,)}
    public_state = _make_public_state(
        tiles=tiles,
        robber_tile_id=0,
        adjacent_tiles=adjacent_tiles,
    )
    result = format_starting_resources(public_state, 0)
    expected = "none"
    assert result == expected


# ---------------------------------------------------------------------------
# 6. Seating order preserved (shuffled Game seed / insertion order)
# ---------------------------------------------------------------------------

def test_gather_board_occupancy_seating_order_preserved_exact():
    # Synthetic insertion order BLUE, WHITE, RED, ORANGE
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8)},
        robber_tile_id=0,
        players_order=[Color.BLUE, Color.WHITE, Color.RED, Color.ORANGE],
    )
    occ = gather_board_occupancy_data(public_state)
    order = [p.color for p in occ.players]
    expected = ["BLUE", "WHITE", "RED", "ORANGE"]
    assert order == expected

    # exact formatted occupancy respects that order
    result = format_board_occupancy_data(occ)
    expected_str = """[CURRENT BOARD OCCUPANCY]
- BLUE: Total: 0p
  * (no buildings/roads)
- WHITE: Total: 0p
  * (no buildings/roads)
- RED: Total: 0p
  * (no buildings/roads)
- ORANGE: Total: 0p
  * (no buildings/roads)"""
    assert result == expected_str


def test_gather_board_occupancy_seating_order_shuffled_game_seed_exact():
    # Use actual Game with shuffled player list and random seed 123
    # State shuffles seating via random.sample, which respects the seed.
    # The test confirms gather_board_occupancy_data preserves that shuffled
    # seating order (not alphabetical) and locks the exact formatted string.
    random.seed(123)
    players = [S(Color.WHITE), S(Color.ORANGE), S(Color.BLUE), S(Color.RED)]
    g = Game(players)
    ps = _build_public_state(g)
    occ = gather_board_occupancy_data(ps)
    order = [p.color for p in occ.players]
    # seating order is whatever Game shuffled to — verify gather preserves it
    expected_order = [c.name for c in g.state.colors]
    assert order == expected_order
    # verify format order matches shuffled seating
    formatted = format_board_occupancy_data(occ)
    lines = formatted.splitlines()
    for idx, cname in enumerate(expected_order):
        assert lines[1 + idx * 2] == f"- {cname}: Total: 0p"
    # exact string built from the shuffled order — no alphabetical assumption
    expected_full = "[CURRENT BOARD OCCUPANCY]\n" + "\n".join(
        f"- {c}: Total: 0p\n  * (no buildings/roads)" for c in expected_order
    )
    assert formatted == expected_full


def test_gather_board_occupancy_duplicate_roads_dedup_exact():
    # 182->177 branch: roads with both orientations should dedup
    public_state = _make_public_state(
        tiles={0: ("WOOD", 8)},
        robber_tile_id=0,
        buildings={0: (Color.RED, SETTLEMENT)},
        roads={(0, 5): Color.RED, (5, 0): Color.RED, (5, 16): Color.BLUE, (16, 5): Color.BLUE},
        players_order=[Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE],
        adjacent_tiles={0: (0,)},
    )
    occ = gather_board_occupancy_data(public_state)
    red = next(p for p in occ.players if p.color == "RED")
    blue = next(p for p in occ.players if p.color == "BLUE")
    assert red.roads == [(0, 5)]
    assert blue.roads == [(5, 16)]
    # exact occupancy with deduped roads
    result = format_board_occupancy_data(occ)
    expected = """[CURRENT BOARD OCCUPANCY]
- RED: Total: 5p (Wd:5)
  * Settlements: Node 0 [8 Wd | 5p]
  * Roads: (0, 5)
- BLUE: Total: 0p
  * Roads: (5, 16)
- ORANGE: Total: 0p
  * (no buildings/roads)
- WHITE: Total: 0p
  * (no buildings/roads)"""
    assert result == expected

