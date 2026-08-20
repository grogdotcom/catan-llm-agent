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
from catan_llm.format.history import describe_action_record


def _rec(color, action_type, value=None, result=None):
    return ActionRecord(Action(color, action_type, value), result)


# ---------------------------------------------------------------------------
# 1. ROLL
# ---------------------------------------------------------------------------

def test_describe_roll_with_result():
    rec = _rec(Color.RED, ActionType.ROLL, (6, 1), (6, 1))
    assert describe_action_record(rec) == "RED rolled 6+1 = 7"


def test_describe_roll_with_value_fallback():
    # ROLL without a result (should fall back to value, or just "rolled")
    rec = _rec(Color.RED, ActionType.ROLL, (3, 4), None)
    # result is None, so the implementation falls back to value (3,4) -> still formats total
    # Actually describe_action_record does: dice = result if result is not None else value
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
    assert describe_action_record(rec) == "BLUE built settlement at node 12"


# ---------------------------------------------------------------------------
# 4. BUILD_CITY
# ---------------------------------------------------------------------------

def test_describe_build_city():
    rec = _rec(Color.BLUE, ActionType.BUILD_CITY, 12, None)
    assert describe_action_record(rec) == "BLUE built city at node 12"


# ---------------------------------------------------------------------------
# 5. BUILD_ROAD (edge is sorted)
# ---------------------------------------------------------------------------

def test_describe_build_road_sorted():
    rec = _rec(Color.BLUE, ActionType.BUILD_ROAD, (3, 1), None)
    assert describe_action_record(rec) == "BLUE built road on edge (1, 3)"


def test_describe_build_road_already_sorted():
    rec = _rec(Color.RED, ActionType.BUILD_ROAD, (0, 5), None)
    assert describe_action_record(rec) == "RED built road on edge (0, 5)"


# ---------------------------------------------------------------------------
# 6. BUY_DEVELOPMENT_CARD — known and sanitized hidden
# ---------------------------------------------------------------------------

def test_describe_buy_dev_card_known():
    rec = _rec(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, "KNIGHT", "KNIGHT")
    assert describe_action_record(rec) == "RED bought development card: KNIGHT"


def test_describe_buy_dev_card_hidden_sanitized():
    # Opponent purchase is redacted: value/result both None
    rec = _rec(Color.BLUE, ActionType.BUY_DEVELOPMENT_CARD, None, None)
    assert describe_action_record(rec) == "BLUE bought a development card"


def test_describe_buy_dev_card_result_fallback():
    # Value None but result carries the card (e.g., after sanitization the result is preserved for self)
    rec = _rec(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None, "VICTORY_POINT")
    assert describe_action_record(rec) == "RED bought development card: VICTORY_POINT"


# ---------------------------------------------------------------------------
# 7. MOVE_ROBBER — three branches
# ---------------------------------------------------------------------------

def test_describe_move_robber_no_steal():
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None), None)
    assert describe_action_record(rec) == "RED moved robber to (0, 0, 0) (no steal)"


def test_describe_move_robber_steal_hidden():
    # Spectator view: victim known, stolen resource redacted (result None)
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), None)
    assert describe_action_record(rec) == "RED moved robber to (0, 0, 0) and stole from BLUE (card hidden)"


def test_describe_move_robber_steal_revealed():
    # Self view: stolen resource is revealed in result
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE), "WHEAT")
    assert describe_action_record(rec) == "RED moved robber to (0, 0, 0) and stole WHEAT from BLUE"


def test_describe_move_robber_unknown_coordinate():
    rec = _rec(Color.RED, ActionType.MOVE_ROBBER, (None, None), None)
    assert describe_action_record(rec) == "RED moved robber to unknown (no steal)"


# ---------------------------------------------------------------------------
# 8. DISCARD_RESOURCE
# ---------------------------------------------------------------------------

def test_describe_discard_resource():
    rec = _rec(Color.WHITE, ActionType.DISCARD_RESOURCE, "ORE", "ORE")
    assert describe_action_record(rec) == "WHITE discarded ORE"


def test_describe_discard_resource_result_fallback():
    # If result is set, it takes precedence over value
    rec = _rec(Color.WHITE, ActionType.DISCARD_RESOURCE, "WOOD", "BRICK")
    assert describe_action_record(rec) == "WHITE discarded BRICK"


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
    assert describe_action_record(rec) == "RED played Year of Plenty: took WOOD, BRICK"


def test_describe_year_of_plenty_single_card():
    rec = _rec(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("ORE",), None)
    assert describe_action_record(rec) == "RED played Year of Plenty: took ORE"


def test_describe_year_of_plenty_no_value():
    rec = _rec(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, None, None)
    assert describe_action_record(rec) == "RED played Year of Plenty"


# ---------------------------------------------------------------------------
# 11. PLAY_MONOPOLY
# ---------------------------------------------------------------------------

def test_describe_play_monopoly():
    rec = _rec(Color.RED, ActionType.PLAY_MONOPOLY, "SHEEP", None)
    assert describe_action_record(rec) == "RED played Monopoly on SHEEP"


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
    assert describe_action_record(rec) == "ORANGE maritime trade: gives [WHEAT, WHEAT, WHEAT, WHEAT] to bank for BRICK"


def test_describe_maritime_trade_2to1_port():
    rec = _rec(Color.ORANGE, ActionType.MARITIME_TRADE, ("ORE", "ORE", None, None, "WOOD"), None)
    assert describe_action_record(rec) == "ORANGE maritime trade: gives [ORE, ORE] to bank for WOOD"


def test_describe_maritime_trade_none_value():
    rec = _rec(Color.ORANGE, ActionType.MARITIME_TRADE, None, None)
    assert describe_action_record(rec) == "ORANGE maritime traded"


# ---------------------------------------------------------------------------
# 14. OFFER_TRADE
# ---------------------------------------------------------------------------

def test_describe_offer_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)  # 1 WOOD for 1 BRICK (RESOURCES order: WOOD BRICK SHEEP WHEAT ORE)
    rec = _rec(Color.RED, ActionType.OFFER_TRADE, offer, None)
    assert describe_action_record(rec) == "RED offers [1 WOOD] for [1 BRICK]"


def test_describe_offer_trade_none_value():
    rec = _rec(Color.RED, ActionType.OFFER_TRADE, None, None)
    assert describe_action_record(rec) == "RED offered a trade"


# ---------------------------------------------------------------------------
# 15. ACCEPT_TRADE
# ---------------------------------------------------------------------------

def test_describe_accept_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    rec = _rec(Color.BLUE, ActionType.ACCEPT_TRADE, offer, None)
    assert describe_action_record(rec) == "BLUE accepted trade: offers [1 WOOD] for [1 BRICK]"


def test_describe_accept_trade_none_value():
    rec = _rec(Color.BLUE, ActionType.ACCEPT_TRADE, None, None)
    assert describe_action_record(rec) == "BLUE accepted a trade"


# ---------------------------------------------------------------------------
# 16. REJECT_TRADE
# ---------------------------------------------------------------------------

def test_describe_reject_trade():
    offer = (1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    rec = _rec(Color.ORANGE, ActionType.REJECT_TRADE, offer, None)
    assert describe_action_record(rec) == "ORANGE rejected trade: offers [1 WOOD] for [1 BRICK]"


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
    assert describe_action_record(rec) == "RED confirmed trade with BLUE: offers [1 WOOD] for [1 BRICK]"


def test_describe_confirm_trade_none_value():
    rec = _rec(Color.RED, ActionType.CONFIRM_TRADE, None, None)
    assert describe_action_record(rec) == "RED confirmed a trade"


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
    # Just assert it contains the color, name, and value repr — exact string
    assert describe_action_record(rec) == "RED CUSTOM_ACTION: value={'foo': 1}, result=None"
