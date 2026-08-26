"""
Exact-string unit tests for every ActionRecord case.

Each test constructs a single ActionRecord and asserts the *exact* string
returned by ``describe_action_record``. This locks the LLM-facing history
format so refactors cannot silently change the prompt.

Covers all 18 ActionTypes in catanatron.models.enums.ActionType plus the
sanitized / fallback branches inside describe_action_record.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "src"))

from catanatron.models.enums import Action, ActionRecord, ActionType
from catanatron.models.player import Color
from catan_llm.format.history import describe_action_record, format_public_history_window
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, SETTLEMENT, CITY
from catanatron.models.public_state import PublicBoard, PublicMap, PublicPlayer, PublicState


def _rec(color, action_type, value=None, result=None):
    return ActionRecord(Action(color, action_type, value), result)


def _mock_roll_state() -> PublicState:
    """PublicState for roll-resource test: every non-7 roll distributes
    resources to demonstrate pipe+bracket format.

    Tiles:
      0: BRICK 5  -> RED, ORANGE
      1: WOOD 4   -> BLUE
      2: SHEEP 8  -> ORANGE
      3: WOOD 8   -> WHITE
      4: BRICK 6  -> WHITE+RED (WHITE touches 4+5, RED city on 4)
      5: WHEAT 6  -> WHITE
      6: ORE 9    -> RED
      7: WOOD 12  -> ORANGE city
      8: BRICK 10 -> ORANGE
      9: DESERT 7 -> robber (no production)
    This yields:
      5 -> RED+ORANGE [1 Br]
      4 -> BLUE [1 Wd]
      8 -> ORANGE [1 Sh], WHITE [1 Wd]
      6 -> RED [2 Br], WHITE [1 Br, 1 Wh]
      9 -> RED [1 Or]
      12 -> ORANGE [2 Wd]
      10 -> ORANGE [1 Br]
    """
    tiles = {
        0: (BRICK, 5),
        1: (WOOD, 4),
        2: (SHEEP, 8),
        3: (WOOD, 8),
        4: (BRICK, 6),
        5: (WHEAT, 6),
        6: (ORE, 9),
        7: (WOOD, 12),
        8: (BRICK, 10),
        9: (None, 7),
    }
    tile_coordinates = {i: (i, 0, -i) for i in tiles}
    adjacent_tiles = {
        10: (0,),
        11: (0,),
        12: (1,),
        13: (2,),
        14: (3,),
        15: (4, 5),
        16: (4,),
        17: (6,),
        18: (7,),
        19: (8,),
    }
    land_nodes = frozenset(adjacent_tiles.keys())
    public_map = PublicMap(
        tiles=tiles,
        tile_coordinates=tile_coordinates,
        ports={},
        adjacent_tiles=adjacent_tiles,
        land_nodes=land_nodes,
    )
    buildings = {
        10: (Color.RED, SETTLEMENT),
        11: (Color.ORANGE, SETTLEMENT),
        12: (Color.BLUE, SETTLEMENT),
        13: (Color.ORANGE, SETTLEMENT),
        14: (Color.WHITE, SETTLEMENT),
        15: (Color.WHITE, SETTLEMENT),
        16: (Color.RED, CITY),
        17: (Color.RED, SETTLEMENT),
        18: (Color.ORANGE, CITY),
        19: (Color.ORANGE, SETTLEMENT),
    }
    board = PublicBoard(
        buildings=buildings,
        roads={},
        robber_tile_id=9,
        longest_road_color=None,
        longest_road_length=0,
        map=public_map,
    )
    players = {
        c: PublicPlayer(
            public_vps=0,
            has_army=False,
            has_road=False,
            longest_road_length=0,
            roads_left=15,
            settlements_left=4,
            cities_left=4,
            has_rolled=False,
            hand_resource_count=0,
            hand_dev_count=0,
            played_knight=0,
            played_monopoly=0,
            played_road_building=0,
            played_year_of_plenty=0,
            played_victory_point=0,
        )
        for c in Color
    }
    return PublicState(board=board, players=players)


# ---------------------------------------------------------------------------
# 1. ROLL
# ---------------------------------------------------------------------------

def test_describe_roll_with_result():
    rec = _rec(Color.RED, ActionType.ROLL, (6, 1), (6, 1))
    assert describe_action_record(rec) == "RED rolled 6+1 = 7"


def test_describe_roll_with_value_fallback():
    rec = _rec(Color.RED, ActionType.ROLL, (3, 4), None)
    assert describe_action_record(rec) == "RED rolled 3+4 = 7"


def test_describe_roll_no_dice():
    rec = _rec(Color.RED, ActionType.ROLL, None, None)
    assert describe_action_record(rec) == "RED rolled"


# ---------------------------------------------------------------------------
# 2. END_TURN
# ---------------------------------------------------------------------------

def test_describe_end_turn():
    rec = _rec(Color.ORANGE, ActionType.END_TURN, None, None)
    assert describe_action_record(rec) == "ORANGE ended turn"


# ---------------------------------------------------------------------------
# 3. BUILD_SETTLEMENT
# ---------------------------------------------------------------------------

def test_describe_build_settlement():
    rec = _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 12, None)
    assert describe_action_record(rec) == "BLUE built S Node 12"


# ---------------------------------------------------------------------------
# 4. BUILD_CITY
# ---------------------------------------------------------------------------

def test_describe_build_city():
    rec = _rec(Color.BLUE, ActionType.BUILD_CITY, 12, None)
    assert describe_action_record(rec) == "BLUE built C Node 12"


# ---------------------------------------------------------------------------
# 5. BUILD_ROAD (edge is sorted)
# ---------------------------------------------------------------------------

def test_describe_build_road_sorted():
    rec = _rec(Color.BLUE, ActionType.BUILD_ROAD, (3, 1), None)
    assert describe_action_record(rec) == "BLUE built road (1, 3)"


def test_describe_build_road_already_sorted():
    rec = _rec(Color.RED, ActionType.BUILD_ROAD, (0, 5), None)
    assert describe_action_record(rec) == "RED built road (0, 5)"


# ---------------------------------------------------------------------------
# 6. BUY_DEVELOPMENT_CARD — known and sanitized hidden
# ---------------------------------------------------------------------------

def test_describe_buy_dev_card_known():
    rec = _rec(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT", "KNIGHT")
    assert describe_action_record(rec) == "RED bought development card: KNIGHT"


def test_describe_buy_dev_card_hidden_sanitized():
    rec = _rec(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, None, None)
    assert describe_action_record(rec) == "BLUE bought a development card"


def test_describe_buy_dev_card_result_fallback():
    rec = _rec(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None, "VICTORY_POINT")
    assert describe_action_record(rec) == "RED bought development card: VICTORY_POINT"


# ---------------------------------------------------------------------------
# 7. MOVE_ROBBER — three branches
# ---------------------------------------------------------------------------

def test_describe_move_robber_no_steal():
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None), None)
    assert describe_action_record(rec) == "RED moved robber to (0, 0, 0) (no steal)"


def test_describe_move_robber_steal_hidden():
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), None)
    assert describe_action_record(rec) == "RED moved robber to (0, 0, 0) and stole from BLUE (card hidden)"


def test_describe_move_robber_steal_revealed():
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), "WHEAT")
    assert describe_action_record(rec) == "RED moved robber to (0, 0, 0) and stole WHEAT from BLUE"


def test_describe_move_robber_unknown_coordinate():
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, (None, None), None)
    assert describe_action_record(rec) == "RED moved robber to unknown (no steal)"


# Enriched TILE display — same tile string as board layout (Tile N: ROLL-Abbr(pips) / DESERT)
def _enriched_ps(seed=42):
    from catanatron.game import Game
    from catanatron.models.player import Player
    from catanatron.models.perspective_player import _build_public_state

    class _P(Player):
        def __init__(self, color):
            self.color = color
            self.is_bot = True
        def decide(self, game, playable_actions):
            return playable_actions[0] if playable_actions else None
        def reset_state(self):
            pass
    players = [_P(Color.RED), _P(Color.BLUE), _P(Color.ORANGE), _P(Color.WHITE)]
    game = Game(players, seed=seed)
    return _build_public_state(game)


def test_describe_move_robber_enriched_shows_tile_no_steal():
    from catan_llm.format.utils import _abbr_resource, get_pip_count
    ps = _enriched_ps()
    # pick a non-desert tile deterministically
    tile_id, (resource, roll) = next((tid, v) for tid, v in ps.board.map.tiles.items() if v[0] is not None)
    coord = ps.board.map.tile_coordinates[tile_id]
    abbr = _abbr_resource(resource.name if hasattr(resource, "name") else str(resource))
    pips = get_pip_count(roll)
    expected_tile = f"Tile {tile_id}: {roll}-{abbr}({pips}p)"
    rec = _rec(Color.BLUE, ActionType.MOVE_ROBBER, (coord, None), None)
    assert describe_action_record(rec, public_state=ps) == f"BLUE moved robber to {expected_tile} (no steal)"


def test_describe_move_robber_enriched_shows_tile_steal_hidden():
    from catan_llm.format.utils import _abbr_resource, get_pip_count
    ps = _enriched_ps()
    tile_id, (resource, roll) = next((tid, v) for tid, v in ps.board.map.tiles.items() if v[0] is not None)
    coord = ps.board.map.tile_coordinates[tile_id]
    abbr = _abbr_resource(resource.name if hasattr(resource, "name") else str(resource))
    pips = get_pip_count(roll)
    expected_tile = f"Tile {tile_id}: {roll}-{abbr}({pips}p)"
    rec = _rec(Color.BLUE, ActionType.MOVE_ROBBER, (coord, Color.RED), None)
    assert describe_action_record(rec, public_state=ps) == f"BLUE moved robber to {expected_tile} and stole from RED (card hidden)"


def test_describe_move_robber_enriched_shows_tile_steal_revealed():
    from catan_llm.format.utils import _abbr_resource, get_pip_count
    ps = _enriched_ps()
    tile_id, (resource, roll) = next((tid, v) for tid, v in ps.board.map.tiles.items() if v[0] is not None)
    coord = ps.board.map.tile_coordinates[tile_id]
    abbr = _abbr_resource(resource.name if hasattr(resource, "name") else str(resource))
    pips = get_pip_count(roll)
    expected_tile = f"Tile {tile_id}: {roll}-{abbr}({pips}p)"
    rec = _rec(Color.BLUE, ActionType.MOVE_ROBBER, (coord, Color.RED), "WHEAT")
    assert describe_action_record(rec, public_state=ps) == f"BLUE moved robber to {expected_tile} and stole WHEAT from RED"


def test_describe_move_robber_enriched_shows_desert():
    ps = _enriched_ps()
    tile_id, (resource, roll) = next((tid, v) for tid, v in ps.board.map.tiles.items() if v[0] is None)
    coord = ps.board.map.tile_coordinates[tile_id]
    expected_tile = f"Tile {tile_id}: DESERT"
    rec = _rec(Color.BLUE, ActionType.MOVE_ROBBER, (coord, Color.RED), None)
    assert describe_action_record(rec, public_state=ps) == f"BLUE moved robber to {expected_tile} and stole from RED (card hidden)"


# ---------------------------------------------------------------------------
# 8. DISCARD_RESOURCE
# ---------------------------------------------------------------------------

def test_describe_discard_resource():
    rec = _rec(Color.WHITE, ActionType.DISCARD_RESOURCE, "ORE", "ORE")
    assert describe_action_record(rec) == "WHITE discarded Or"


def test_describe_discard_resource_result_fallback():
    rec = _rec(Color.WHITE, ActionType.DISCARD_RESOURCE, "WOOD", "BRICK")
    assert describe_action_record(rec) == "WHITE discarded Br"


# ---------------------------------------------------------------------------
# 9. PLAY_KNIGHT_CARD
# ---------------------------------------------------------------------------

def test_describe_play_knight():
    rec = _rec(Color.RED, ActionType.PLAY_KNIGHT_CARD, None, None)
    assert describe_action_record(rec) == "RED played Knight"


# ---------------------------------------------------------------------------
# 10. PLAY_YEAR_OF_PLENTY
# ---------------------------------------------------------------------------

def test_describe_year_of_plenty_two_cards():
    rec = _rec(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "BRICK"), None)
    assert describe_action_record(rec) == "RED played YOP: took Wd, Br"


def test_describe_year_of_plenty_single_card():
    rec = _rec(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("ORE",), None)
    assert describe_action_record(rec) == "RED played YOP: took Or"


def test_describe_year_of_plenty_no_value():
    rec = _rec(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, None, None)
    assert describe_action_record(rec) == "RED played YOP"


# ---------------------------------------------------------------------------
# 11. PLAY_MONOPOLY
# ---------------------------------------------------------------------------

def test_describe_play_monopoly():
    rec = _rec(Color.RED, ActionType.PLAY_MONOPOLY, "SHEEP", None)
    assert describe_action_record(rec) == "RED played MONOPOLY on SH"


def test_describe_play_monopoly_with_stolen():
    # Patched engine stores (resource, stolen_tuple, total); history shows per-player losses condensed
    rec = _rec(Color.BLUE, ActionType.PLAY_MONOPOLY, "ORE", ("ORE", ((Color.RED, 2), (Color.WHITE, 1)), 3))
    assert describe_action_record(rec) == "BLUE played MONOPOLY on OR | RED - 2 OR, WHITE - 1 OR (total 3)"


def test_describe_play_monopoly_stole_nothing():
    rec = _rec(Color.BLUE, ActionType.PLAY_MONOPOLY, "SHEEP", ("SHEEP", (), 0))
    assert describe_action_record(rec) == "BLUE played MONOPOLY on SH | stole nothing (total 0)"


# ---------------------------------------------------------------------------
# 12. PLAY_ROAD_BUILDING
# ---------------------------------------------------------------------------

def test_describe_play_road_building():
    rec = _rec(Color.RED, ActionType.PLAY_ROAD_BUILDING, None, None)
    assert describe_action_record(rec) == "RED played Road Building"


# ---------------------------------------------------------------------------
# 13. MARITIME_TRADE
# ---------------------------------------------------------------------------

def test_describe_maritime_trade_4to1():
    rec = _rec(Color.ORANGE, ActionType.MARITIME_TRADE, ("WHEAT", "WHEAT", "WHEAT", "WHEAT", "BRICK"), None)
    assert describe_action_record(rec) == "ORANGE maritime trade: gives [4 Wh] to bank for Br"


def test_describe_maritime_trade_2to1_port():
    rec = _rec(Color.ORANGE, ActionType.MARITIME_TRADE, ("ORE", "ORE", None, None, "WOOD"), None)
    assert describe_action_record(rec) == "ORANGE maritime trade: gives [2 Or] to bank for Wd"


def test_describe_maritime_trade_none_value():
    rec = _rec(Color.ORANGE, ActionType.MARITIME_TRADE, None, None)
    assert describe_action_record(rec) == "ORANGE maritime traded"


# ---------------------------------------------------------------------------
# 14. OFFER_TRADE
# ---------------------------------------------------------------------------

def test_describe_offer_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    rec = _rec(Color.RED, ActionType.OFFER_TRADE, offer, None)
    assert describe_action_record(rec) == "RED offers [1 Wd] for [1 Br]"


def test_describe_offer_trade_none_value():
    rec = _rec(Color.RED, ActionType.OFFER_TRADE, None, None)
    assert describe_action_record(rec) == "RED offered a trade"


# ---------------------------------------------------------------------------
# 15. ACCEPT_TRADE
# ---------------------------------------------------------------------------

def test_describe_accept_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    rec = _rec(Color.BLUE, ActionType.ACCEPT_TRADE, offer, None)
    assert describe_action_record(rec) == "BLUE accepted trade: offers [1 Wd] for [1 Br]"


def test_describe_accept_trade_none_value():
    rec = _rec(Color.BLUE, ActionType.ACCEPT_TRADE, None, None)
    assert describe_action_record(rec) == "BLUE accepted a trade"


# ---------------------------------------------------------------------------
# 16. REJECT_TRADE
# ---------------------------------------------------------------------------

def test_describe_reject_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    rec = _rec(Color.ORANGE, ActionType.REJECT_TRADE, offer, None)
    assert describe_action_record(rec) == "ORANGE rejected trade: offers [1 Wd] for [1 Br]"


def test_describe_reject_trade_none_value():
    rec = _rec(Color.ORANGE, ActionType.REJECT_TRADE, None, None)
    assert describe_action_record(rec) == "ORANGE rejected a trade"


# ---------------------------------------------------------------------------
# 17. CONFIRM_TRADE
# ---------------------------------------------------------------------------

def test_describe_confirm_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    confirm = offer + (Color.BLUE,)
    rec = _rec(Color.RED, ActionType.CONFIRM_TRADE, confirm, None)
    assert describe_action_record(rec) == "RED gave BLUE [1 Wd] for [1 Br]"


def test_describe_confirm_trade_none_value():
    rec = _rec(Color.RED, ActionType.CONFIRM_TRADE, None, None)
    assert describe_action_record(rec) == "RED traded"


# ---------------------------------------------------------------------------
# 18. CANCEL_TRADE
# ---------------------------------------------------------------------------

def test_describe_cancel_trade():
    rec = _rec(Color.RED, ActionType.CANCEL_TRADE, None, None)
    assert describe_action_record(rec) == "RED cancelled trade"


# ---------------------------------------------------------------------------
# 19. Fallback / unknown ActionType (defensive branch)
# ---------------------------------------------------------------------------

def test_describe_fallback_unknown_action():
    # Simulate a future ActionType not explicitly handled; should hit the last
    # return f"{color} {action_type.name}: value={value!r}, result={result!r}"
    # We craft a dummy enum-like object.
    class DummyType:
        name = "CUSTOM_ACTION"

    dummy_action = Action(Color.RED, DummyType(), {"foo": 1})
    rec = ActionRecord(dummy_action, None)
    assert describe_action_record(rec) == "RED CUSTOM_ACTION: value={'foo': 1}, result=None"

# === Migrated from test_board.py — grouping & windowed history (proper home = history) ===

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.models.perspective_player import _build_public_state, _sanitize_history

class SimplePlayer(Player):
    def __init__(self, color):
        self.color = color
        self.is_bot = True
    def decide(self, game, playable_actions):
        return playable_actions[0] if playable_actions else None
    def reset_state(self):
        pass

from catan_llm.format.history import (
    describe_turn,
    format_public_history,
    format_public_history_window,
    group_action_records_by_turn,
)
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
  - RED built S Node 0
  - RED built road (0, 1)
  - BLUE built S Node 5
  - BLUE built road (5, 6)
[TURN 1 (RED)]
  - RED rolled 2+3 = 5
  - RED ended turn
[TURN 2 (BLUE)]
  - BLUE rolled 6+1 = 7
  - RED discarded Wd
  - BLUE moved robber to (0, 0, 0) and stole from RED (card hidden)
  - BLUE ended turn"""
    assert text == expected

    turn_only = describe_turn(records[4:6], turn_label="TURN 1 (RED)")
    assert turn_only == "[TURN 1 (RED)]\n  - RED rolled 2+3 = 5\n  - RED ended turn"



def test_format_public_history_empty():
    assert format_public_history(()) == "[PUBLIC HISTORY]\n  (empty)"
    assert format_public_history_window((), window_size=2) == "[PUBLIC HISTORY]\n  (empty)"



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
    # Discards are aggregated per player per contiguous block (e.g. 4 discards -> 1 bullet)
    # and REJECT/ACCEPT/CANCEL trades are filtered (only offers + traded remain)
    bullet_count = sum(1 for line in text.splitlines() if line.startswith("  - "))
    filtered_types = (
        ActionType.DISCARD_RESOURCE,
        ActionType.REJECT_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.CANCEL_TRADE,
    )
    non_filtered = sum(1 for r in history if r.action.action_type not in filtered_types)
    # aggregated discards reduce bullets, filtered trades also reduce, so bullet_count <= len(history)
    assert bullet_count <= len(history)
    assert bullet_count >= non_filtered or bullet_count >= 1
    if any(r.action.action_type == ActionType.DISCARD_RESOURCE for r in history):
        assert "discarded" in text
    # Rejects/accepts/cancels should not appear in formatted history
    assert "rejected trade" not in text
    assert "accepted trade" not in text
    assert "cancelled trade" not in text



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
    
    # Setup omitted in midgame truncated window (user intent: drop initial placement after early game)
    assert "[SETUP]" not in result
    
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
    assert "RED built S Node 0" in result
    assert "BLUE built S Node 5" in result
    
    # Should contain setup-only indicator
    assert "[Showing setup phase only]" in result
    
    # Should NOT contain any turns
    assert "[TURN" not in result
    assert "rolled" not in result



def test_format_public_history_window_empty_history():
    """Test that empty history works correctly"""
    result = format_public_history_window((), window_size=2)
    assert result == "[PUBLIC HISTORY]\n  (empty)"
    assert format_public_history_window((), window_size=None) == "[PUBLIC HISTORY]\n  (empty)"



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


def test_format_public_history_window_last_twelve_turns():
    """Exact-string: window_size=12 shows last 12 turns with realistic variety.

    Builds a 15-turn game with diverse actions (maritime trade, city, domestic
    trade offer+gave, robber steals, Road Building, YOP, Monopoly, Knight,
    settlements/roads) and asserts the *exact* LLM-facing window string.
    This demonstrates the previous-action history the agent sees and guards
    mid-game compression (setup dropped, indicator, absolute numbering).
    """
    setup = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.ORANGE, ActionType.BUILD_SETTLEMENT, 10),
        _rec(Color.ORANGE, ActionType.BUILD_ROAD, (10, 11)),
        _rec(Color.WHITE, ActionType.BUILD_SETTLEMENT, 15),
        _rec(Color.WHITE, ActionType.BUILD_ROAD, (15, 16)),
    )
    turns_data = [
        (Color.RED, [(ActionType.ROLL, (3, 2), (3, 2)), (ActionType.BUILD_ROAD, (0, 5)), (ActionType.END_TURN, None)]),
        (Color.BLUE, [(ActionType.ROLL, (2, 2), (2, 2)), (ActionType.BUY_DEVELOPMENT_CARD, None, "KNIGHT"), (ActionType.END_TURN, None)]),
        (Color.ORANGE, [(ActionType.ROLL, (4, 4), (4, 4)), (ActionType.BUILD_SETTLEMENT, 20), (ActionType.END_TURN, None)]),
        (Color.WHITE, [(ActionType.ROLL, (3, 3), (3, 3)), (ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "BRICK")), (ActionType.END_TURN, None)]),
        (Color.RED, [(ActionType.ROLL, (5, 4), (5, 4)), (ActionType.BUILD_CITY, 0), (ActionType.END_TURN, None)]),
        (Color.BLUE, [(ActionType.ROLL, (2, 5), (2, 5)), (ActionType.OFFER_TRADE, (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)), (ActionType.CONFIRM_TRADE, (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, Color.WHITE)), (ActionType.END_TURN, None)]),
        (Color.ORANGE, [(ActionType.ROLL, (3, 4), (3, 4)), (ActionType.MOVE_ROBBER, ((0, 0, 0), Color.RED), "WOOD"), (ActionType.END_TURN, None)]),
        (Color.WHITE, [(ActionType.ROLL, (4, 3), (4, 3)), (ActionType.PLAY_ROAD_BUILDING, None), (ActionType.BUILD_ROAD, (15, 20)), (ActionType.BUILD_ROAD, (20, 21)), (ActionType.END_TURN, None)]),
        (Color.RED, [(ActionType.ROLL, (6, 2), (6, 2)), (ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "BRICK")), (ActionType.END_TURN, None)]),
        (Color.BLUE, [(ActionType.ROLL, (5, 3), (5, 3)), (ActionType.PLAY_MONOPOLY, "ORE", ("ORE", ((Color.RED, 2), (Color.WHITE, 1)), 3)), (ActionType.END_TURN, None)]),
        (Color.ORANGE, [(ActionType.ROLL, (6, 6), (6, 6)), (ActionType.PLAY_KNIGHT_CARD, None), (ActionType.MOVE_ROBBER, ((1, -1, 0), Color.BLUE), None), (ActionType.END_TURN, None)]),
        (Color.WHITE, [(ActionType.ROLL, (3, 5), (3, 5)), (ActionType.BUILD_SETTLEMENT, 25), (ActionType.BUILD_ROAD, (25, 26)), (ActionType.END_TURN, None)]),
        (Color.RED, [(ActionType.ROLL, (4, 2), (4, 2)), (ActionType.MARITIME_TRADE, ("ORE", "ORE", None, None, "WOOD")), (ActionType.END_TURN, None)]),
        (Color.BLUE, [(ActionType.ROLL, (2, 4), (2, 4)), (ActionType.OFFER_TRADE, (0, 0, 1, 0, 0, 0, 0, 0, 1, 0)), (ActionType.END_TURN, None)]),
        (Color.ORANGE, [(ActionType.ROLL, (5, 5), (5, 5)), (ActionType.BUILD_CITY, 10), (ActionType.BUILD_ROAD, (10, 15)), (ActionType.END_TURN, None)]),
    ]
    records = list(setup)
    for color, actions in turns_data:
        for at, val, *rest in actions:
            res = rest[0] if rest else None
            records.append(_rec(color, at, val, res))
    records = tuple(records)

    ps = _mock_roll_state()
    result = format_public_history_window(records, window_size=12, public_state=ps)

    expected = """[PUBLIC HISTORY]
[Showing last 12 of 15 turns]
[TURN 4 (WHITE)]
  - WHITE rolled 3+3 = 6 | RED + [2 Br], WHITE + [1 Br, 1 Wh]
  - WHITE maritime trade: gives [4 Wd] to bank for Br
  - WHITE ended turn
[TURN 5 (RED)]
  - RED rolled 5+4 = 9 | RED + [1 Or]
  - RED built C Node 0 [no tiles | 0p]
  - RED ended turn
[TURN 6 (BLUE)]
  - BLUE rolled 2+5 = 7
  - BLUE offers [1 Wd] for [1 Br]
  - BLUE gave WHITE [1 Wd] for [1 Br]
  - BLUE ended turn
[TURN 7 (ORANGE)]
  - ORANGE rolled 3+4 = 7
  - ORANGE moved robber to Tile 0: 5-Br(4p) and stole WOOD from RED
  - ORANGE ended turn
[TURN 8 (WHITE)]
  - WHITE rolled 4+3 = 7
  - WHITE played Road Building
  - WHITE built road (15, 20)
  - WHITE built road (20, 21)
  - WHITE ended turn
[TURN 9 (RED)]
  - RED rolled 6+2 = 8 | ORANGE + [1 Sh], WHITE + [1 Wd]
  - RED played YOP: took Wd, Br
  - RED ended turn
[TURN 10 (BLUE)]
  - BLUE rolled 5+3 = 8 | ORANGE + [1 Sh], WHITE + [1 Wd]
  - BLUE played MONOPOLY on OR | RED - 2 OR, WHITE - 1 OR (total 3)
  - BLUE ended turn
[TURN 11 (ORANGE)]
  - ORANGE rolled 6+6 = 12 | ORANGE + [2 Wd]
  - ORANGE played Knight
  - ORANGE moved robber to (1, -1, 0) and stole from BLUE (card hidden)
  - ORANGE ended turn
[TURN 12 (WHITE)]
  - WHITE rolled 3+5 = 8 | ORANGE + [1 Sh], WHITE + [1 Wd]
  - WHITE built S Node 25 [no tiles | 0p]
  - WHITE built road (25, 26)
  - WHITE ended turn
[TURN 13 (RED)]
  - RED rolled 4+2 = 6 | RED + [2 Br], WHITE + [1 Br, 1 Wh]
  - RED maritime trade: gives [2 Or] to bank for Wd
  - RED ended turn
[TURN 14 (BLUE)]
  - BLUE rolled 2+4 = 6 | RED + [2 Br], WHITE + [1 Br, 1 Wh]
  - BLUE offers [1 Sh] for [1 Wh]
  - BLUE ended turn
[TURN 15 (ORANGE)]
  - ORANGE rolled 5+5 = 10 | ORANGE + [1 Br]
  - ORANGE built C Node 10 [5-Br | 4p]
  - ORANGE built road (10, 15)
  - ORANGE ended turn"""
    assert result == expected


def test_format_public_history_window_twelve_equals_total_includes_setup():
    """Window == total should include setup and emit no truncated indicator."""
    setup = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
    )
    turns = []
    for i in range(12):
        turns.append(_rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)))
        turns.append(_rec(Color.RED, ActionType.END_TURN))
    records = setup + tuple(turns)

    result = format_public_history_window(records, window_size=12)

    assert "[SETUP]" in result
    assert "[Showing last" not in result
    assert result.count("[TURN") == 12
    assert "[TURN 1 (RED)]" in result
    assert "[TURN 12 (RED)]" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
