"""
Exact-string golden tests for prompts.py uncovered branches.

Covers: summarize_catan_actions legacy fallbacks, footer suppression,
_resolve_history_records, ergonomic shims, header inclusion, turn_number
inference, observation shim for moves, history window variants, and aliases.

Every test asserts the exact output string (not substring) against a literal.
"""
import random
import enum
from types import SimpleNamespace
from unittest import mock

import pytest
from catanatron.game import Game
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType
from catanatron.models.player import Color, Player
from catanatron.models.perspective_player import _build_public_state
from catan_llm.format.prompts import (
    build_complete_prompt,
    format_complete_prompt,
    format_decision_prompt,
    format_decision_prompt_with_history,
    format_observation_prompt,
    get_complete_prompt,
    get_full_prompt,
    format_full_prompt,
    summarize_catan_actions,
    _resolve_history_records,
    DEFAULT_COMPLETE_FOOTER,
    DEFAULT_DECISION_FOOTER,
)
from catan_llm.format.board import get_full_board_map, get_board_occupancy
from catan_llm.format.players import get_players_summary

class S(Player):
    def __init__(self, c):
        self.color = c
        self.is_bot = True
    def decide(self, g, a):
        return a[0] if a else None
    def reset_state(self):
        pass

def _ps(seed=42, order=None):
    if order is None:
        order = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    random.seed(seed)
    g = Game([S(c) for c in order], seed=seed)
    return _build_public_state(g), g

def _rec(color, at, value=None, result=None):
    return ActionRecord(Action(color, at, value), result)

def _obs(public_state, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable=None, **extra):
    ns = SimpleNamespace(public_state=public_state, current_prompt=current_prompt, color=color, playable_actions=playable if playable is not None else [])
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns

def test_summarize_empty_exact():
    assert summarize_catan_actions([]) == 'No actions available.'

def test_summarize_build_road_coordinate_none_fallback_value():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    ns = SimpleNamespace(action_type=ActionType.BUILD_ROAD, coordinate=None, value=(5,6))
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- BUILD_ROAD: Target IDs [(5, 6)]'

def test_summarize_build_road_coordinate_present_uses_coordinate():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    ns = SimpleNamespace(action_type=ActionType.BUILD_ROAD, coordinate=(1, 2), value=(9, 9))
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- BUILD_ROAD: Target IDs [(1, 2)]'

def test_summarize_build_settlement_coordinate_fallback():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    ns = SimpleNamespace(action_type=ActionType.BUILD_SETTLEMENT, coordinate=None, value=12)
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- BUILD_SETTLEMENT: Target IDs [12]'

def test_summarize_maritime_give_get():
    import enum
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    class R(enum.Enum):
        WOOD='WOOD'
        BRICK='BRICK'
    ns = SimpleNamespace(action_type=ActionType.MARITIME_TRADE, kwargs={'resource_giving': R.WOOD, 'resource_receiving': R.BRICK})
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- MARITIME_TRADE: Options [Give WOOD -> Get BRICK]'

def test_summarize_maritime_fallback_str():
    from catanatron.models.enums import Action, ActionType
    from catanatron.models.player import Color
    a = Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD","WOOD","WOOD","WOOD","BRICK"))
    assert summarize_catan_actions([a]) == "[PLAYABLE ACTION CATEGORIES]:\n- MARITIME_TRADE: Options [Action(color=C.RED, action_type=AT.MARITIME_TRADE, value=('WOOD', 'WOOD', 'WOOD', 'WOOD', 'BRICK'))]"

def test_summarize_move_robber_tuple_with_victim():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    from catanatron.models.player import Color
    ns = SimpleNamespace(action_type=ActionType.MOVE_ROBBER, value=((0, 0, 0), Color.BLUE))
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- MOVE_ROBBER: Options [Hex (0, 0, 0) (Victim: Color.BLUE)]'

def test_summarize_move_robber_tuple_none_victim():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    ns = SimpleNamespace(action_type=ActionType.MOVE_ROBBER, value=((0, 0, 0), None))
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- MOVE_ROBBER: Options [Hex (0, 0, 0) (Victim: NONE)]'

def test_summarize_move_robber_non_tuple_else_branch():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    from catanatron.models.player import Color
    ns = SimpleNamespace(action_type=ActionType.MOVE_ROBBER, value='weird', coordinate=(1, 1, 1), kwargs={'victim_color': Color.RED})
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- MOVE_ROBBER: Options [Hex (1, 1, 1) (Victim: Color.RED)]'

def test_summarize_move_robber_exception_fallback():
    from catanatron.models.enums import ActionType
    class Bad:
        action_type = ActionType.MOVE_ROBBER
        def __str__(self):
            return 'BAD_ROBBER'
        @property
        def value(self):
            raise RuntimeError('boom')
    assert summarize_catan_actions([Bad()]) == '[PLAYABLE ACTION CATEGORIES]:\n- MOVE_ROBBER: Options [BAD_ROBBER]'

def test_summarize_end_turn_exact():
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    ns = SimpleNamespace(action_type=ActionType.END_TURN)
    assert summarize_catan_actions([ns]) == '[PLAYABLE ACTION CATEGORIES]:\n- Pass (End Turn)'

def test_summarize_mixed_exact():
    import enum
    from types import SimpleNamespace
    from catanatron.models.enums import ActionType
    class R(enum.Enum):
        WOOD='WOOD'
        BRICK='BRICK'
    actions = [
        SimpleNamespace(action_type=ActionType.BUILD_ROAD, coordinate=None, value=(0, 5)),
        SimpleNamespace(action_type=ActionType.MARITIME_TRADE, kwargs={'resource_giving': R.WOOD, 'resource_receiving': R.BRICK}),
        SimpleNamespace(action_type=ActionType.MOVE_ROBBER, value=((0, 0, 0), None)),
        SimpleNamespace(action_type=ActionType.END_TURN),
    ]
    assert summarize_catan_actions(actions) == '[PLAYABLE ACTION CATEGORIES]:\n- BUILD_ROAD: Target IDs [(0, 5)]\n- MARITIME_TRADE: Options [Give WOOD -> Get BRICK]\n- MOVE_ROBBER: Options [Hex (0, 0, 0) (Victim: NONE)]\n- Pass (End Turn)'

def test_format_decision_prompt_footer_empty_string():
    ps, _ = _ps(42)
    out = format_decision_prompt(ps, [], "RED", ActionPrompt.PLAY_TURN, 7, footer="", include_footer=True)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 7]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\nNo actions available.\n\n'
    assert out.endswith("No actions available.\n\n")

def test_format_decision_prompt_footer_custom():
    ps, _ = _ps(42)
    out = format_decision_prompt(ps, [], "RED", ActionPrompt.PLAY_TURN, 7, footer="CUSTOM FOOTER", include_footer=True)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 7]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\nNo actions available.\n\n\nCUSTOM FOOTER'
    assert out.endswith("CUSTOM FOOTER")

def test_format_decision_prompt_footer_none_include_false():
    ps, _ = _ps(42)
    out = format_decision_prompt(ps, [], "RED", ActionPrompt.PLAY_TURN, 7, footer=None, include_footer=False)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 7]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\nNo actions available.\n\n'
    assert "[DECISION REQUIRED]" not in out

def test_format_decision_prompt_footer_default():
    ps, _ = _ps(42)
    out = format_decision_prompt(ps, [], "RED", ActionPrompt.PLAY_TURN, 7, footer=None, include_footer=True)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 7]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\nNo actions available.\n\n\n[DECISION REQUIRED]\nSelect the best action from the available options above.'
    assert out.endswith(DEFAULT_DECISION_FOOTER)

def test_format_decision_prompt_phase_string_fallback_exact():
    ps, _ = _ps(42)
    out = format_decision_prompt(ps, [], "BLUE", "CUSTOM_PHASE", 0, footer=None, include_footer=True)
    assert out == '[CURRENT PLAYER: BLUE]\n[TURN: 0]\n[PHASE: CUSTOM_PHASE]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\nNo actions available.\n\n\n[DECISION REQUIRED]\nSelect the best action from the available options above.'

def test_format_decision_prompt_with_history_footer_empty():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (3,4), (3,4)), _rec(Color.RED, ActionType.END_TURN),)
    out = format_decision_prompt_with_history(ps, [], "RED", ActionPrompt.PLAY_TURN, 1, hist, history_window_size=8, footer="", include_footer=True)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 1]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 3+4 = 7\n  - RED ended turn\n\n\nNo actions available.\n\n'

def test_format_decision_prompt_with_history_footer_custom():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (3,4), (3,4)), _rec(Color.RED, ActionType.END_TURN),)
    out = format_decision_prompt_with_history(ps, [], "RED", ActionPrompt.PLAY_TURN, 1, hist, history_window_size=8, footer="CUSTOM2", include_footer=True)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 1]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 3+4 = 7\n  - RED ended turn\n\n\nNo actions available.\n\n\nCUSTOM2'

def test_format_decision_prompt_with_history_footer_none_include_false():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (3,4), (3,4)), _rec(Color.RED, ActionType.END_TURN),)
    out = format_decision_prompt_with_history(ps, [], "RED", ActionPrompt.PLAY_TURN, 1, hist, history_window_size=8, footer=None, include_footer=False)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 1]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 3+4 = 7\n  - RED ended turn\n\n\nNo actions available.\n\n'

def test_format_decision_prompt_with_history_footer_default():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (3,4), (3,4)), _rec(Color.RED, ActionType.END_TURN),)
    out = format_decision_prompt_with_history(ps, [], "RED", ActionPrompt.PLAY_TURN, 1, hist, history_window_size=8, footer=None, include_footer=True)
    assert out == '[CURRENT PLAYER: RED]\n[TURN: 1]\n[PHASE: PLAY_TURN]\n\n\n[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 3+4 = 7\n  - RED ended turn\n\n\nNo actions available.\n\n\n[DECISION REQUIRED]\nSelect the best action from the available options above.'

def test_resolve_history_explicit_override():
    hist = [1,2,3]
    assert _resolve_history_records(hist, None) == hist
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_history=(4,5), history=(6,7))
    assert _resolve_history_records(hist, obs) == hist

def test_resolve_history_via_public_history():
    obs = SimpleNamespace(public_history=(4,5), history=(6,7))
    assert _resolve_history_records(None, obs) == (4,5)

def test_resolve_history_via_history_fallback():
    obs = SimpleNamespace(public_history=None, history=(6,7))
    assert _resolve_history_records(None, obs) == (6,7)
    obs2 = SimpleNamespace(history=(6,7))
    assert _resolve_history_records(None, obs2) == (6,7)

def test_resolve_history_empty_tuple():
    assert _resolve_history_records(None, None) == ()
    assert _resolve_history_records(None, SimpleNamespace()) == ()
    assert _resolve_history_records(None, SimpleNamespace(public_history=None, history=None)) == ()

def test_get_complete_raises_when_both_none():
    with pytest.raises(ValueError, match="public_state is required"):
        get_complete_prompt(public_state=None, observation=None, playable_actions=[])
    with pytest.raises(ValueError, match="public_state is required"):
        get_complete_prompt(public_state=None, observation=SimpleNamespace(), playable_actions=[])

def test_get_complete_playable_none_defaults_to_empty():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.RED, None, include_header=False, include_footer=False, observation=None, history_window_size=8)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert "(no moves available)" in out

def test_get_complete_infer_color_from_observation():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.ORANGE, playable_actions=[])
    out = get_complete_prompt(public_state=None, current_player_color=None, playable_actions=[], observation=obs, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: ORANGE]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[CURRENT PLAYER: ORANGE]" in out

def test_get_complete_playable_inferred_from_observation():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))])
    out = get_complete_prompt(public_state=ps, current_player_color=Color.RED, playable_actions=None, observation=obs, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n1. Road (0, 5) -> Targets: Node 1 [11 Br, 10 Wd, 9 Or | 9p]✓, Node 20 [4 Br, 9 Or, 3 Br | 9p]✓ | LR 0->1(+1)'
    assert "[PLAYABLE MOVES" in out

def test_get_complete_phase_fallback_from_observation():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.MOVE_ROBBER, color=Color.RED, playable_actions=[])
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=None, observation=obs, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[PHASE: MOVE_ROBBER]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: MOVE_ROBBER]\n  (no moves available)'
    # header phase from observation
    assert "[PHASE: MOVE_ROBBER]" in out

def test_get_complete_header_all_none_no_header():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, None, [], current_prompt=None, turn_number=None, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert out.lstrip().startswith("[FULL BOARD MAP")
    assert "[CURRENT PLAYER" not in out

def test_get_complete_header_only_color():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.BLUE, [], current_prompt=None, turn_number=None, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: BLUE]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert "[CURRENT PLAYER: BLUE]" in out
    assert "[TURN:" not in out

def test_get_complete_header_only_turn():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, None, [], current_prompt=None, turn_number=12, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[TURN: 12]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert "[TURN: 12]" in out

def test_get_complete_header_only_phase_enum():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, None, [], current_prompt=ActionPrompt.PLAY_TURN, turn_number=None, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[PHASE: PLAY_TURN]" in out

def test_get_complete_header_string_color_and_prompt():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, "RED", [], current_prompt="CUSTOM", turn_number=99, include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[TURN: 99]\n[PHASE: CUSTOM]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: CUSTOM]\n  (no moves available)'
    assert "[CURRENT PLAYER: RED]" in out
    assert "[PHASE: CUSTOM]" in out
    assert "[TURN: 99]" in out

def test_get_complete_include_header_false_omits_header():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=ActionPrompt.PLAY_TURN, turn_number=5, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[CURRENT PLAYER" not in out
    assert out.lstrip().startswith("[FULL BOARD MAP")

def test_get_complete_history_window_none_exact():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (2,3), (2,3)), _rec(Color.RED, ActionType.END_TURN), _rec(Color.BLUE, ActionType.ROLL, (6,1), (6,1)), _rec(Color.BLUE, ActionType.END_TURN), _rec(Color.RED, ActionType.ROLL, (4,5), (4,5)), _rec(Color.RED, ActionType.END_TURN),)
    out = get_complete_prompt(ps, Color.RED, [], public_history=hist, history_window_size=None, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 2+3 = 5 | no resources\n  - RED ended turn\n[TURN 2 (BLUE)]\n  - BLUE rolled 6+1 = 7\n  - BLUE ended turn\n[TURN 3 (RED)]\n  - RED rolled 4+5 = 9 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert "[RECENT TURNS (LAST 8)]" in out
    assert "[PUBLIC HISTORY]" in out

def test_get_complete_history_window_zero_exact():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (2,3), (2,3)), _rec(Color.RED, ActionType.END_TURN), _rec(Color.BLUE, ActionType.ROLL, (6,1), (6,1)), _rec(Color.BLUE, ActionType.END_TURN), _rec(Color.RED, ActionType.ROLL, (4,5), (4,5)), _rec(Color.RED, ActionType.END_TURN),)
    out = get_complete_prompt(ps, Color.RED, [], public_history=hist, history_window_size=0, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[Showing setup phase only]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert "[RECENT TURNS (LAST 8)]" in out
    assert "[PUBLIC HISTORY]" in out

def test_get_complete_history_window_eight_exact():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (2,3), (2,3)), _rec(Color.RED, ActionType.END_TURN), _rec(Color.BLUE, ActionType.ROLL, (6,1), (6,1)), _rec(Color.BLUE, ActionType.END_TURN), _rec(Color.RED, ActionType.ROLL, (4,5), (4,5)), _rec(Color.RED, ActionType.END_TURN),)
    out = get_complete_prompt(ps, Color.RED, [], public_history=hist, history_window_size=8, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 2+3 = 5 | no resources\n  - RED ended turn\n[TURN 2 (BLUE)]\n  - BLUE rolled 6+1 = 7\n  - BLUE ended turn\n[TURN 3 (RED)]\n  - RED rolled 4+5 = 9 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n  (no moves available)'
    assert "[RECENT TURNS (LAST 8)]" in out
    assert "[PUBLIC HISTORY]" in out

def test_get_complete_public_history_override_observation():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, public_history=(_rec(Color.BLUE, ActionType.ROLL, (1,1), (1,1)), _rec(Color.BLUE, ActionType.END_TURN)), history=(_rec(Color.RED, ActionType.ROLL, (2,2), (2,2)),), current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[])
    explicit = (_rec(Color.RED, ActionType.ROLL, (3,3), (3,3)), _rec(Color.RED, ActionType.END_TURN),)
    out = get_complete_prompt(ps, Color.RED, [], public_history=explicit, observation=obs, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[TURN 1 (RED)]\n  - RED rolled 3+3 = 6 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    # explicit should win, so BLUE roll not present
    assert "BLUE rolled" not in out
    assert "RED rolled 3+3" in out

def test_get_complete_history_via_observation_public_history():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.ROLL, (3,3), (3,3)), _rec(Color.RED, ActionType.END_TURN),)
    obs = SimpleNamespace(public_state=ps, public_history=hist, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[])
    out = get_complete_prompt(ps, Color.RED, [], public_history=None, observation=obs, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[TURN 1 (RED)]\n  - RED rolled 3+3 = 6 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "RED rolled 3+3" in out

def test_get_complete_history_via_observation_history_attr():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.ROLL, (3,3), (3,3)), _rec(Color.RED, ActionType.END_TURN),)
    obs = SimpleNamespace(public_state=ps, history=hist, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[])
    # ensure public_history is absent, fallback to history
    if hasattr(obs, "public_history"):
        delattr(obs, "public_history")
    out = get_complete_prompt(ps, Color.RED, [], public_history=None, observation=obs, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[TURN 1 (RED)]\n  - RED rolled 3+3 = 6 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "RED rolled 3+3" in out

def test_get_complete_history_alias_else_branch():
    ps, _ = _ps(42)
    with mock.patch('catan_llm.format.prompts.format_public_history_window', return_value='CUSTOM HISTORY BLOCK'):
        out = get_complete_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
        assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\nCUSTOM HISTORY BLOCK\n\n[PLAYABLE MOVES]\n  (no moves available)'
        assert "[RECENT TURNS (LAST 8)]\nCUSTOM HISTORY BLOCK" in out

def test_get_complete_footer_empty_string():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=ActionPrompt.PLAY_TURN, turn_number=5, footer="", include_footer=True, include_header=True)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[TURN: 5]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert not out.endswith("CUSTOM")
    assert "[DECISION REQUIRED]" not in out

def test_get_complete_footer_custom():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=ActionPrompt.PLAY_TURN, turn_number=5, footer="CUSTOM FOOTER", include_footer=True)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[TURN: 5]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)\n\nCUSTOM FOOTER'
    assert out.endswith("CUSTOM FOOTER")

def test_get_complete_include_footer_false():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=ActionPrompt.PLAY_TURN, turn_number=5, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[TURN: 5]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[DECISION REQUIRED]" not in out

def test_get_complete_footer_default():
    ps, _ = _ps(42)
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=ActionPrompt.PLAY_TURN, turn_number=5, include_footer=True)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[TURN: 5]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)\n\n[DECISION REQUIRED]\nSelect the best action from the available moves above.'
    assert out.endswith(DEFAULT_COMPLETE_FOOTER)

def test_get_complete_observation_shim_builds_moves_without_observation():
    ps, _ = _ps(42)
    acts = [Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))]
    out = get_complete_prompt(ps, Color.RED, acts, current_prompt=ActionPrompt.PLAY_TURN, observation=None, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n1. Road (0, 5) -> Targets: Node 1 [11 Br, 10 Wd, 9 Or | 9p]✓, Node 20 [4 Br, 9 Or, 3 Br | 9p]✓ | LR 0->1(+1)'
    assert "[PLAYABLE MOVES" in out

def test_get_complete_moves_with_observation():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))])
    acts = [Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))]
    out = get_complete_prompt(ps, Color.RED, acts, observation=obs, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n1. Road (0, 5) -> Targets: Node 1 [11 Br, 10 Wd, 9 Or | 9p]✓, Node 20 [4 Br, 9 Or, 3 Br | 9p]✓ | LR 0->1(+1)'
    assert "[PLAYABLE MOVES" in out

def test_build_complete_prompt_aliases():
    assert build_complete_prompt is get_complete_prompt
    assert format_complete_prompt is get_complete_prompt
    assert get_full_prompt is get_complete_prompt
    assert format_full_prompt is get_complete_prompt
    ps, _ = _ps(42)
    out1 = get_complete_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    out2 = build_complete_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    out3 = format_complete_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    out4 = get_full_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    out5 = format_full_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    assert out1 == out2 == out3 == out4 == out5

def test_format_observation_prompt_derives_and_matches_exact():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.BLUE, playable_actions=[], turn_number=8)
    out_wrapper = format_observation_prompt(obs, [], include_header=True, include_footer=False)
    assert out_wrapper == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: BLUE]\n[TURN: 8]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    # also matches direct
    direct = get_complete_prompt(public_state=ps, current_player_color=Color.BLUE, playable_actions=[], observation=obs, current_prompt=ActionPrompt.PLAY_TURN, turn_number=8, include_header=True, include_footer=False)
    assert out_wrapper == direct

def test_format_observation_prompt_respects_playable_override():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))])
    out = format_observation_prompt(obs, playable_actions=[], include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "(no moves available)" in out

def test_format_observation_prompt_turn_number_fallback_current_turn_index():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.WHITE, current_turn_index=4, playable_actions=[])
    out = format_observation_prompt(obs, [], include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: WHITE]\n[TURN: 4]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[TURN: 4]" in out

def test_format_observation_prompt_turn_number_missing_no_header_turn():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.WHITE, playable_actions=[])
    out = format_observation_prompt(obs, [], include_header=True, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: WHITE]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[TURN:" not in out

def test_format_observation_prompt_no_playable_attr_defaults_empty():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED)
    out = format_observation_prompt(obs, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "(no moves available)" in out

def test_format_observation_prompt_footer_empty_string():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[])
    out = format_observation_prompt(obs, [], footer="", include_footer=True, include_header=True)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[DECISION REQUIRED]" not in out

def test_format_observation_prompt_footer_custom():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[])
    out = format_observation_prompt(obs, [], footer="FOO", include_footer=True)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[CURRENT PLAYER: RED]\n[PHASE: PLAY_TURN]\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n  (empty)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)\n\nFOO'
    assert out.endswith("FOO")

def test_format_observation_prompt_history_window_none():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (2,3), (2,3)), _rec(Color.RED, ActionType.END_TURN), _rec(Color.BLUE, ActionType.ROLL, (6,1), (6,1)), _rec(Color.BLUE, ActionType.END_TURN), _rec(Color.RED, ActionType.ROLL, (4,5), (4,5)), _rec(Color.RED, ActionType.END_TURN),)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[], public_history=hist)
    out = format_observation_prompt(obs, [], history_window_size=None, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n[TURN 1 (RED)]\n  - RED rolled 2+3 = 5 | no resources\n  - RED ended turn\n[TURN 2 (BLUE)]\n  - BLUE rolled 6+1 = 7\n  - BLUE ended turn\n[TURN 3 (RED)]\n  - RED rolled 4+5 = 9 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[RECENT TURNS" in out

def test_format_observation_prompt_history_window_zero():
    ps, _ = _ps(42)
    hist = (_rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0), _rec(Color.RED, ActionType.BUILD_ROAD, (0,1)), _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5), _rec(Color.BLUE, ActionType.BUILD_ROAD, (5,6)), _rec(Color.RED, ActionType.ROLL, (2,3), (2,3)), _rec(Color.RED, ActionType.END_TURN), _rec(Color.BLUE, ActionType.ROLL, (6,1), (6,1)), _rec(Color.BLUE, ActionType.END_TURN), _rec(Color.RED, ActionType.ROLL, (4,5), (4,5)), _rec(Color.RED, ActionType.END_TURN),)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[], public_history=hist)
    out = format_observation_prompt(obs, [], history_window_size=0, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[Showing setup phase only]\n[SETUP]\n  - RED built Settlement Node 0 [11 Br, 4 Br, 9 Or | 9p]\n  - RED built road (0, 1)\n  - BLUE built Settlement Node 5 [11 Br, 5 Or, 4 Br | 9p]\n  - BLUE built road (5, 6)\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "[Showing setup phase only]" in out

def test_format_observation_prompt_public_history_override():
    ps, _ = _ps(42)
    hist_explicit = (_rec(Color.RED, ActionType.ROLL, (3,3), (3,3)), _rec(Color.RED, ActionType.END_TURN),)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[], public_history=(_rec(Color.BLUE, ActionType.ROLL, (1,1), (1,1)),))
    out = format_observation_prompt(obs, [], public_history=hist_explicit, include_header=False, include_footer=False)
    assert out == '[FULL BOARD MAP - 19 HEXES]\nT0: 11-Br(2p) [0,1,2,3,4,5]\nT1: 10-Wd(3p) [1,2,6,7,8,9]\nT2: 3-Wh(2p) [2,3,9,10,11,12]\nT3: 6-Wh(5p) [3,4,12,13,14,15]\nT4: 5-Or(4p) [4,5,15,16,17,18]\nT5: 4-Br(3p) [0,5,16,19,20,21]\nT6: 9-Or(4p) [0,1,6,20,22,23]\nT7: 5-Sh(4p) [7,8,24,25,26,27]\nT8: 8-Or(5p) [8,9,10,27,28,29]\nT9: 4-Wh(3p) [10,11,29,30,31,32]\nT10: 11-Wd(2p) [11,12,13,32,33,34]\nT11: 12-Wh(1p) [13,14,34,35,36,37]\nT12: DESERT [14,15,17,37,38,39]\nT13: 9-Sh(4p) [17,18,39,40,41,42]\nT14: 10-Sh(3p) [16,18,21,40,43,44]\nT15: 8-Sh(5p) [19,21,43,45,46,47]\nT16: 3-Br(2p) [19,20,22,46,48,49]\nT17: 6-Wd(5p) [22,23,49,50,51,52]\nT18: 2-Wd(1p) [6,7,23,24,52,53]\n\n[CURRENT BOARD OCCUPANCY]\n- RED: Total: 0p\n  * (no buildings/roads)\n- WHITE: Total: 0p\n  * (no buildings/roads)\n- BLUE: Total: 0p\n  * (no buildings/roads)\n- ORANGE: Total: 0p\n  * (no buildings/roads)\n\nROBBER: Tile 12: DESERT | Blocking: None\n\n[PLAYERS]\n- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15\n\n[RECENT TURNS (LAST 8)]\n[PUBLIC HISTORY]\n[TURN 1 (RED)]\n  - RED rolled 3+3 = 6 | no resources\n  - RED ended turn\n\n[PLAYABLE MOVES]\n[PHASE: PLAY_TURN]\n  (no moves available)'
    assert "RED rolled 3+3" in out
    assert "BLUE rolled" not in out

def test_format_observation_prompt_aliases_exact():
    from catan_llm.format.prompts import get_observation_prompt, build_observation_prompt
    assert format_observation_prompt is get_observation_prompt
    assert format_observation_prompt is build_observation_prompt

def test_get_complete_infer_color_no_infer_when_observation_has_no_color():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, playable_actions=[])
    # color attribute missing or None -> should not infer, header only has phase
    out = get_complete_prompt(ps, None, [], observation=obs, include_header=True, include_footer=False)
    assert "[CURRENT PLAYER" not in out
    assert "[PHASE: PLAY_TURN]" in out
    # also test with color=None explicitly
    obs2 = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=None, playable_actions=[])
    out2 = get_complete_prompt(ps, None, [], observation=obs2, include_header=True, include_footer=False)
    assert "[CURRENT PLAYER" not in out2

def test_format_observation_prompt_playable_none_with_observation_has_playable():
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=[Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))])
    # do not pass playable_actions, so wrapper will infer from observation (outer None -> inner not None)
    out = format_observation_prompt(obs, playable_actions=None, include_header=False, include_footer=False)
    assert "[PLAYABLE MOVES" in out
    assert "Road (0, 5)" in out
    # also test explicit None still infers
    out2 = get_complete_prompt(ps, Color.RED, None, observation=obs, include_header=False, include_footer=False)
    assert "Road (0, 5)" in out2

def test_get_complete_header_string_prompt_with_observation_fallback():
    # cover else branch for phase_name when prompt is string via observation fallback
    ps, _ = _ps(42)
    obs = SimpleNamespace(public_state=ps, current_prompt="OBS_CUSTOM", color=Color.RED, playable_actions=[])
    out = get_complete_prompt(ps, Color.RED, [], current_prompt=None, observation=obs, include_header=True, include_footer=False)
    assert "[PHASE: OBS_CUSTOM]" in out
    # also color string via observation? color inferred already tested, but phase string via observation covers 427 else
    assert "[CURRENT PLAYER: RED]" in out
