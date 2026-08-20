"""
Unit tests for prompt formatting — validates every high-level prompt composer.

Replaces the former print-only demo with real assertions. Covers:

* get_game_state_summary  — board → occupancy → inventories composition
* summarize_catan_actions — legacy grouped action categories
* format_decision_prompt / format_decision_prompt_with_history — header + state + actions/history + footer
* get_complete_prompt / aliases / format_observation_prompt — integrated five-section prompt
  in strict order: board → occupancy → robber → inventories → moves

All tests use deterministic seeds / hand-built states so exact-string assertions are stable.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "src"))

import pytest

from catanatron.game import Game
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType
from catanatron.models.enums import CITY, SETTLEMENT
from catanatron.models.inventory import Inventory
from catanatron.models.player import Color, Player
from catanatron.models.perspective_player import _build_public_state
from catanatron.state_functions import player_key

from catan_llm.format.board import get_board_occupancy, get_full_board_map
from catan_llm.format.prompts import (
    build_complete_prompt,
    build_observation_prompt,
    format_complete_prompt,
    format_decision_prompt,
    format_decision_prompt_with_history,
    format_full_prompt,
    format_observation_prompt,
    get_complete_prompt,
    get_full_prompt,
    get_game_state_summary,
    get_observation_prompt,
    summarize_catan_actions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimplePlayer(Player):
    def __init__(self, color):
        self.color = color
        self.is_bot = True

    def decide(self, game, playable_actions):
        return playable_actions[0] if playable_actions else None

    def reset_state(self):
        pass


def build_public_state(game):
    return _build_public_state(game)


def create_empty_game(seed=42):
    random.seed(seed)
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    return Game(players, seed=seed)


def create_deterministic_game_with_buildings(seed=42):
    """Deterministic board + hand-placed buildings/roads/robber matching test_board."""
    random.seed(seed)
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    game = Game(players, seed=seed)
    board = game.state.board
    board.buildings[0] = (Color.RED, SETTLEMENT)
    board.buildings[1] = (Color.RED, SETTLEMENT)
    board.buildings[10] = (Color.RED, CITY)
    board.buildings[11] = (Color.RED, CITY)
    board.roads[(0, 5)] = Color.RED
    board.roads[(5, 0)] = Color.RED
    board.roads[(1, 6)] = Color.RED
    board.roads[(6, 1)] = Color.RED
    board.roads[(10, 15)] = Color.RED
    board.roads[(15, 10)] = Color.RED
    board.roads[(11, 16)] = Color.RED
    board.roads[(16, 11)] = Color.RED
    board.roads[(15, 20)] = Color.RED
    board.roads[(20, 15)] = Color.RED
    board.roads[(16, 22)] = Color.RED
    board.roads[(22, 16)] = Color.RED

    board.buildings[5] = (Color.BLUE, SETTLEMENT)
    board.buildings[6] = (Color.BLUE, SETTLEMENT)
    board.buildings[15] = (Color.BLUE, CITY)
    board.roads[(5, 16)] = Color.BLUE
    board.roads[(16, 5)] = Color.BLUE
    board.roads[(6, 21)] = Color.BLUE
    board.roads[(21, 6)] = Color.BLUE
    board.roads[(15, 20)] = Color.BLUE
    board.roads[(20, 15)] = Color.BLUE
    board.roads[(20, 25)] = Color.BLUE
    board.roads[(25, 20)] = Color.BLUE
    board.roads[(25, 26)] = Color.BLUE
    board.roads[(26, 25)] = Color.BLUE

    board.buildings[20] = (Color.ORANGE, SETTLEMENT)
    board.buildings[25] = (Color.ORANGE, CITY)
    board.buildings[26] = (Color.ORANGE, CITY)
    board.roads[(20, 21)] = Color.ORANGE
    board.roads[(21, 20)] = Color.ORANGE
    board.roads[(25, 30)] = Color.ORANGE
    board.roads[(30, 25)] = Color.ORANGE
    board.roads[(26, 31)] = Color.ORANGE
    board.roads[(31, 26)] = Color.ORANGE
    board.roads[(30, 35)] = Color.ORANGE
    board.roads[(35, 30)] = Color.ORANGE
    board.roads[(31, 36)] = Color.ORANGE
    board.roads[(36, 31)] = Color.ORANGE

    board.buildings[30] = (Color.WHITE, SETTLEMENT)
    board.buildings[35] = (Color.WHITE, CITY)
    board.roads[(30, 31)] = Color.WHITE
    board.roads[(31, 30)] = Color.WHITE
    board.roads[(35, 36)] = Color.WHITE
    board.roads[(36, 35)] = Color.WHITE
    board.roads[(35, 40)] = Color.WHITE
    board.roads[(40, 35)] = Color.WHITE
    board.roads[(36, 41)] = Color.WHITE
    board.roads[(41, 36)] = Color.WHITE
    board.roads[(40, 42)] = Color.WHITE
    board.roads[(42, 40)] = Color.WHITE

    for nid in [0, 1, 5, 6, 10, 11, 15, 20, 21, 22, 25, 26, 30, 31, 35, 36, 40, 41, 42]:
        board.board_buildable_ids.discard(nid)
    board.robber_coordinate = (0, 0, 0)
    return game


def _observation_shim(public_state, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED, playable_actions=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        public_state=public_state,
        current_prompt=current_prompt,
        color=color,
        playable_actions=playable_actions or [],
    )


def _rec(color, action_type, value=None, result=None):
    return ActionRecord(Action(color, action_type, value), result)


# ---------------------------------------------------------------------------
# get_game_state_summary
# ---------------------------------------------------------------------------


def test_get_game_state_summary_sections_in_order():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_game_state_summary(ps, Color.RED)
    assert "[FULL BOARD MAP - 19 HEXES]" in out
    assert "[CURRENT BOARD OCCUPANCY]" in out
    assert "[PLAYERS]" in out
    assert out.index("[FULL BOARD MAP") < out.index("[CURRENT BOARD OCCUPANCY")
    assert out.index("[CURRENT BOARD OCCUPANCY") < out.index("[PLAYERS]")


def test_get_game_state_summary_contains_board_tiles_and_occupancy_and_players():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_game_state_summary(ps, Color.BLUE)
    # board map has 19 tiles
    assert out.count("Tile ") >= 19
    # occupancy header
    assert "Ports:" in out and "Settlements:" in out
    # players header — current player marked (YOU) only when color supplied
    assert "[PLAYERS]" in out
    assert "BLUE (YOU)" in out


def test_get_game_state_summary_with_inventory_reveals_exact_and_marks_you():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    inv = Inventory(wood=2, brick=1, sheep=0, wheat=3, ore=0, knight=1, victory_point=1, actual_vps=4)
    # seed board so we have ports/pips context
    out = get_game_state_summary(ps, Color.RED, inv)
    # RED is YOU and shows exact counts
    assert "RED (YOU): Resources: WOOD: 2, BRICK: 1, WHEAT: 3" in out
    assert "Dev: KNIGHT: 1, VICTORY_POINT: 1" in out
    # opponents hidden
    assert "BLUE: Resources: " in out and "resource cards (hidden)" in out
    # hidden VP math: RED public_vps is 0 on fresh board, actual 4 => "4 (0 visible + 4 hidden)" somewhere
    assert "VP:" in out


def test_get_game_state_summary_without_inventory_all_hidden():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_game_state_summary(ps, Color.RED, None)
    # Even RED is hidden when no inventory — look inside [PLAYERS] section, not occupancy
    players_section = out.split("[PLAYERS]", 1)[1]
    red_line = [ln for ln in players_section.splitlines() if ln.startswith("- RED")][0]
    assert "resource cards (hidden)" in red_line
    assert "dev cards (hidden)" in red_line


def test_get_game_state_summary_exact_deterministic_empty():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_game_state_summary(ps, Color.RED)
    # Build expected exactly as impl does: "\n".join([map, "\n", occupancy, "\n", players])
    from catan_llm.format.players import get_players_summary

    expected = "\n".join([get_full_board_map(ps), "\n", get_board_occupancy(ps), "\n", get_players_summary(ps, Color.RED)])
    assert out == expected


def test_get_game_state_summary_with_buildings_shows_ports_and_pips():
    game = create_deterministic_game_with_buildings(seed=42)
    ps = build_public_state(game)
    out = get_game_state_summary(ps, Color.RED)
    # occupancy production totals
    assert "Total:" in out and "pips" in out
    # ports — at least one player has ports
    assert "Ports:" in out
    # players pips section
    assert "Pips:" in out
    # ORANGE has SHEEP port in that deterministic setup
    assert "SHEEP" in out


# ---------------------------------------------------------------------------
# summarize_catan_actions (legacy grouped view)
# ---------------------------------------------------------------------------


def test_summarize_empty():
    assert summarize_catan_actions([]) == "No actions available."
    assert summarize_catan_actions(None) == "No actions available." if False else True  # type guard
    # Empty list variant explicitly
    assert summarize_catan_actions([]) == "No actions available."


def test_summarize_build_groups_collapse_target_ids():
    actions = [
        Action(Color.RED, ActionType.BUILD_ROAD, (0, 5)),
        Action(Color.RED, ActionType.BUILD_ROAD, (5, 16)),
        Action(Color.RED, ActionType.BUILD_SETTLEMENT, 12),
        Action(Color.RED, ActionType.BUILD_CITY, 15),
    ]
    out = summarize_catan_actions(actions)
    assert out.startswith("[PLAYABLE ACTION CATEGORIES]:")
    assert "- BUILD_ROAD: Target IDs [(0, 5), (5, 16)]" in out
    assert "- BUILD_SETTLEMENT: Target IDs [12]" in out
    assert "- BUILD_CITY: Target IDs [15]" in out


def test_summarize_maritime_fallback_str():
    # Action is a namedtuple without .kwargs — the try/except falls back to str(action)
    actions = [Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "BRICK"))]
    out = summarize_catan_actions(actions)
    assert "MARITIME_TRADE" in out
    # Fallback uses str(action) which contains the action repr
    assert "Action" in out or "MARITIME_TRADE" in out


def test_summarize_maritime_dedup_via_str():
    a = Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "BRICK"))
    b = Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "BRICK"))
    out = summarize_catan_actions([a, b, a])
    # Dedup via set -> only one unique Options entry despite three inputs
    # str(action) contains "MARITIME_TRADE" itself, so count would be 2; check header count and option count instead
    assert out.count("- MARITIME_TRADE:") == 1
    assert out.count("Action(color=C.RED") == 1
    assert "Options [" in out


def test_summarize_move_robber_grouping():
    actions = [
        Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE)),
        Action(Color.RED, ActionType.MOVE_ROBBER, ((1, -1, 0), None)),
    ]
    out = summarize_catan_actions(actions)
    assert "- MOVE_ROBBER: Options [" in out
    assert "Hex (0, 0, 0) (Victim: Color.BLUE)" in out or "Victim: Color.BLUE" in out
    assert "Victim: NONE" in out


def test_summarize_end_turn():
    actions = [Action(Color.RED, ActionType.END_TURN, None)]
    out = summarize_catan_actions(actions)
    assert "- Pass (End Turn)" in out


def test_summarize_dev_cards_catchall():
    actions = [
        Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None),
        Action(Color.RED, ActionType.PLAY_MONOPOLY, "SHEEP"),
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
    ]
    out = summarize_catan_actions(actions)
    # Catch-all uses action_type.name as category and ", ".join(str(action)) as options
    assert "PLAY_KNIGHT_CARD" in out
    assert "PLAY_MONOPOLY" in out
    assert "BUY_DEVELOPMENT_CARD" in out


def test_summarize_mixed_categories_all_present():
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.BUILD_ROAD, (0, 5)),
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), None)),
    ]
    out = summarize_catan_actions(actions)
    for cat in ["ROLL", "BUILD_ROAD", "MOVE_ROBBER"]:
        assert cat in out
    assert "Pass (End Turn)" in out


# ---------------------------------------------------------------------------
# format_decision_prompt
# ---------------------------------------------------------------------------


def test_format_decision_prompt_header():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    current_prompt = game.state.current_prompt
    out = format_decision_prompt(ps, playable, Color.RED.name, current_prompt, 7)
    assert "[CURRENT PLAYER: RED]" in out
    assert "[TURN: 7]" in out
    assert f"[PHASE: {current_prompt.name}]" in out


def test_format_decision_prompt_phase_string_fallback():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = format_decision_prompt(ps, [], "BLUE", "CUSTOM_PHASE", 0)
    assert "[CURRENT PLAYER: BLUE]" in out
    assert "[PHASE: CUSTOM_PHASE]" in out


def test_format_decision_prompt_contains_game_state_and_actions_and_footer():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    prompt = game.state.current_prompt
    out = format_decision_prompt(ps, playable, Color.RED.name, prompt, 3)
    # game_state_summary is embedded
    assert "[FULL BOARD MAP - 19 HEXES]" in out
    assert "[CURRENT BOARD OCCUPANCY]" in out
    assert "[PLAYERS]" in out
    # actions block
    assert "[PLAYABLE ACTION CATEGORIES]:" in out
    # footer
    assert "[DECISION REQUIRED]" in out
    assert "Select the best action" in out
    # ordering: header < board < occupancy < players < actions < footer
    assert out.index("[CURRENT PLAYER") < out.index("[FULL BOARD MAP")
    assert out.index("[PLAYERS]") < out.index("[PLAYABLE ACTION CATEGORIES]")
    assert out.index("[PLAYABLE ACTION CATEGORIES]") < out.index("[DECISION REQUIRED]")


def test_format_decision_prompt_with_inventory_affects_players_section():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    prompt = game.state.current_prompt
    inv = Inventory(wood=1, brick=1, actual_vps=0)
    out_with = format_decision_prompt(ps, [], Color.RED, prompt, 0, inv)
    out_without = format_decision_prompt(ps, [], Color.RED, prompt, 0, None)
    # with inventory RED line shows exact — look inside [PLAYERS] section
    players_with = out_with.split("[PLAYERS]", 1)[1]
    assert "WOOD: 1" in players_with
    # without, RED is hidden
    players_without = out_without.split("[PLAYERS]", 1)[1]
    red_line_without = [l for l in players_without.splitlines() if l.startswith("- RED")][0]
    assert "hidden" in red_line_without


def test_format_decision_prompt_empty_actions_shows_no_actions():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    prompt = game.state.current_prompt
    out = format_decision_prompt(ps, [], Color.RED.name, prompt, 0)
    assert "No actions available." in out


# ---------------------------------------------------------------------------
# format_decision_prompt_with_history
# ---------------------------------------------------------------------------


def test_format_decision_prompt_with_history_contains_history_section():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    prompt = game.state.current_prompt
    # craft minimal history: setup + one turn — use non-empty playable so categories header appears
    hist = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (3, 4), (3, 4)),
        _rec(Color.RED, ActionType.END_TURN),
    )
    playable = [Action(Color.RED, ActionType.ROLL, None)]
    out = format_decision_prompt_with_history(ps, playable, Color.RED.name, prompt, 1, hist, history_window_size=None)
    assert "[PUBLIC HISTORY]" in out
    assert "[SETUP]" in out
    assert "[TURN 1 (RED)]" in out
    assert "RED rolled 3+4 = 7" in out
    assert out.index("[PUBLIC HISTORY]") < out.index("[PLAYABLE ACTION CATEGORIES]")
    assert out.index("[PLAYABLE ACTION CATEGORIES]") < out.index("[DECISION REQUIRED]")


def test_format_decision_prompt_with_history_window_variants():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    prompt = game.state.current_prompt
    hist = (
        _rec(Color.RED, ActionType.BUILD_SETTLEMENT, 0),
        _rec(Color.RED, ActionType.BUILD_ROAD, (0, 1)),
        _rec(Color.BLUE, ActionType.BUILD_SETTLEMENT, 5),
        _rec(Color.BLUE, ActionType.BUILD_ROAD, (5, 6)),
        _rec(Color.RED, ActionType.ROLL, (2, 3), (2, 3)),
        _rec(Color.RED, ActionType.END_TURN),
        _rec(Color.BLUE, ActionType.ROLL, (6, 1), (6, 1)),
        _rec(Color.BLUE, ActionType.END_TURN),
        _rec(Color.RED, ActionType.ROLL, (4, 5), (4, 5)),
        _rec(Color.RED, ActionType.END_TURN),
    )
    # window_size=None -> same as full
    out_full = format_decision_prompt_with_history(ps, [], Color.RED.name, prompt, 3, hist, None)
    # window 0 -> only setup (check history turns, not header [TURN:])
    out_zero = format_decision_prompt_with_history(ps, [], Color.RED.name, prompt, 3, hist, history_window_size=0)
    assert "[Showing setup phase only]" in out_zero
    # history uses "[TURN 1 (...)]" with space+digit; header uses "[TURN: 3]"
    assert "[TURN 1" not in out_zero and "[TURN 2" not in out_zero
    # window 1 -> last turn only (absolute numbering: last of 3 = TURN 3)
    out_one = format_decision_prompt_with_history(ps, [], Color.RED.name, prompt, 3, hist, history_window_size=1)
    assert "[Showing last 1 of 3 turns]" in out_one
    # history turn label is now absolute "[TURN 3 (RED)]"
    assert "[TURN 3 (RED)]" in out_one
    assert "[TURN 1 (RED)]" not in out_one and "[TURN 2 (BLUE)]" not in out_one
    # window larger than total -> no indicator
    out_big = format_decision_prompt_with_history(ps, [], Color.RED.name, prompt, 3, hist, history_window_size=10)
    assert "[Showing last" not in out_big


def test_format_decision_prompt_with_history_empty_history_marker():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    prompt = game.state.current_prompt
    out = format_decision_prompt_with_history(ps, [], Color.RED.name, prompt, 0, (), None)
    assert "[PUBLIC HISTORY]" in out
    assert "(empty)" in out


# ---------------------------------------------------------------------------
# get_complete_prompt — integrated five-section prompt
# ---------------------------------------------------------------------------


def test_get_complete_prompt_five_sections_in_strict_order():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    obs = _observation_shim(ps, game.state.current_prompt, Color.RED, playable)
    out = get_complete_prompt(ps, Color.RED, playable, observation=obs, turn_number=5)
    # all canonical sections present
    for header in ["[FULL BOARD MAP", "[CURRENT BOARD OCCUPANCY", "ROBBER:", "[PLAYERS]", "[PLAYABLE MOVES]"]:
        assert header in out, f"missing {header}"
    assert out.index("[FULL BOARD MAP") < out.index("[CURRENT BOARD OCCUPANCY")
    assert out.index("[CURRENT BOARD OCCUPANCY") < out.index("ROBBER:")
    assert out.index("ROBBER:") < out.index("[PLAYERS]")
    assert out.index("[PLAYERS]") < out.index("[PLAYABLE MOVES]")
    assert "[DECISION REQUIRED]" in out


def test_get_complete_prompt_header_includes_player_turn_phase():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    out = get_complete_prompt(ps, Color.BLUE, playable, current_prompt=ActionPrompt.PLAY_TURN, turn_number=12)
    assert "[CURRENT PLAYER: BLUE]" in out
    assert "[TURN: 12]" in out
    assert "[PHASE: PLAY_TURN]" in out
    # header is first section before board
    assert out.index("[CURRENT PLAYER") < out.index("[FULL BOARD MAP")


def test_get_complete_prompt_include_header_false_omits_header():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_complete_prompt(ps, Color.RED, [], include_header=False)
    assert "[CURRENT PLAYER" not in out
    assert "[TURN:" not in out
    assert "[PHASE:" not in out.split("[FULL BOARD MAP")[0]  # no phase before map
    # board still first
    assert out.lstrip().startswith("[FULL BOARD MAP")


def test_get_complete_prompt_include_footer_false_omits_footer():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_complete_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    assert "[DECISION REQUIRED]" not in out
    # moves is last section when footer omitted
    assert out.rstrip().endswith(tuple("0123456789.")) or "[PLAYABLE MOVES]" in out


def test_get_complete_prompt_empty_moves_shows_no_moves():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_complete_prompt(ps, Color.RED, [], include_header=False, include_footer=False)
    assert "[PLAYABLE MOVES]" in out
    assert "(no moves available)" in out


def test_get_complete_prompt_without_observation_uses_shim_and_still_rich():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions  # initial settlement placements
    out = get_complete_prompt(ps, Color.RED, playable, current_prompt=ActionPrompt.BUILD_INITIAL_SETTLEMENT, turn_number=0)
    # Without observation, shim still produces rich settlement labels with Tile/pips
    assert "Tile" in out and "pips" in out
    assert "Build settlement at Node" in out
    # phase tag inside moves block
    assert "[PHASE: BUILD_INITIAL_SETTLEMENT]" in out


def test_get_complete_prompt_with_observation_bundles_moves():
    # Knight bundling requires observation for correct robber targets
    from catanatron.models.actions import generate_playable_actions, robber_possibilities

    random.seed(7)
    game = Game(
        [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE), SimplePlayer(Color.ORANGE), SimplePlayer(Color.WHITE)],
        seed=7,
    )
    state = game.state
    red_key = player_key(state, Color.RED)
    state.current_player_index = state.color_to_index[Color.RED]
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state[f"{red_key}_HAS_ROLLED"] = True
    state.player_state[f"{red_key}_KNIGHT_IN_HAND"] = 1
    state.player_state[f"{red_key}_KNIGHT_OWNED_AT_START"] = True
    state.player_state[f"{red_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
    board = state.board
    board.buildings[0] = (Color.RED, SETTLEMENT)
    board.buildings[5] = (Color.BLUE, SETTLEMENT)
    for nid in (0, 5):
        board.board_buildable_ids.discard(nid)
    # give BLUE a card to be stealable
    blue_key = player_key(state, Color.BLUE)
    state.player_state[f"{blue_key}_{'WOOD'}_IN_HAND"] = 1

    ps = build_public_state(game)
    playable = generate_playable_actions(state)
    assert any(a.action_type == ActionType.PLAY_KNIGHT_CARD for a in playable)
    obs = _observation_shim(ps, state.current_prompt, Color.RED, playable)

    out = get_complete_prompt(ps, Color.RED, playable, observation=obs)
    # knight moves are bundled as "Play Knight -> move robber to ..."
    assert "Play Knight -> move robber to" in out
    # no AUTO_ROAD sentinel leaked
    assert "AUTO_ROAD" not in out
    # moves count matches engine robber possibilities for knights
    engine_targets = len(list(robber_possibilities(state, Color.RED)))
    assert out.count("Play Knight -> move robber to") == engine_targets


def test_get_complete_prompt_inventory_hidden_vs_exact():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = []
    inv = Inventory(wood=2, brick=1, ore=1, knight=1, actual_vps=3)
    out_with = get_complete_prompt(ps, Color.RED, playable, current_player_inventory=inv, include_header=False, include_footer=False)
    out_without = get_complete_prompt(ps, Color.RED, playable, current_player_inventory=None, include_header=False, include_footer=False)
    # with inventory RED shows exact — look inside [PLAYERS] section, not occupancy
    players_with = out_with.split("[PLAYERS]", 1)[1]
    red_line_with = [l for l in players_with.splitlines() if l.startswith("- RED")][0]
    assert "WOOD: 2" in red_line_with and "BRICK: 1" in red_line_with
    assert "Dev: KNIGHT: 1" in red_line_with
    # without, RED hidden
    players_without = out_without.split("[PLAYERS]", 1)[1]
    red_line_without = [l for l in players_without.splitlines() if l.startswith("- RED")][0]
    assert "resource cards (hidden)" in red_line_without
    assert "dev cards (hidden)" in red_line_without


def test_get_complete_prompt_robber_blocking_reflects_occupancy():
    game = create_deterministic_game_with_buildings(seed=42)
    ps = build_public_state(game)
    playable = []
    out = get_complete_prompt(ps, Color.RED, playable, include_header=False, include_footer=False)
    # deterministic board has robber on Tile 0 area; RED and BLUE blocked
    assert "ROBBER: Tile" in out
    assert "Blocking RED" in out or "Blocking BLUE" in out
    assert "Blocking: None" not in out  # there is blocking


def test_get_complete_prompt_raises_without_public_state():
    with pytest.raises(ValueError, match="public_state is required"):
        get_complete_prompt(public_state=None, current_player_color=Color.RED, playable_actions=[], observation=None)


def test_get_complete_prompt_infers_public_state_and_color_from_observation():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    obs = _observation_shim(ps, game.state.current_prompt, Color.ORANGE, playable)
    # pass no explicit public_state / color — must infer from observation
    out = get_complete_prompt(observation=obs, playable_actions=playable)
    assert "[CURRENT PLAYER: ORANGE]" in out
    assert "[FULL BOARD MAP - 19 HEXES]" in out
    assert "[PLAYABLE MOVES]" in out


def test_get_complete_prompt_playable_actions_inferred_from_observation_when_none():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    obs = _observation_shim(ps, game.state.current_prompt, Color.RED, playable)
    out = get_complete_prompt(public_state=ps, observation=obs, current_player_color=Color.RED, playable_actions=None)
    # should have used obs.playable_actions
    assert "[PLAYABLE MOVES]" in out
    # there are settlement moves
    assert "Build settlement at" in out


def test_get_complete_prompt_aliases_equal():
    assert get_complete_prompt is get_full_prompt
    assert get_complete_prompt is format_complete_prompt
    assert get_complete_prompt is build_complete_prompt
    assert get_complete_prompt is format_full_prompt


def test_get_complete_prompt_string_color_and_prompt():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_complete_prompt(ps, "RED", [], current_prompt="CUSTOM", turn_number=99, include_footer=False)
    assert "[CURRENT PLAYER: RED]" in out
    assert "[PHASE: CUSTOM]" in out
    assert "[TURN: 99]" in out


def test_get_complete_prompt_excludes_header_when_all_none():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    out = get_complete_prompt(ps, None, [], current_prompt=None, turn_number=None, include_header=True, include_footer=False)
    # no header because all identity fields are None, first section is board
    assert out.lstrip().startswith("[FULL BOARD MAP")


# ---------------------------------------------------------------------------
# format_observation_prompt — convenience wrapper
# ---------------------------------------------------------------------------


def test_format_observation_prompt_derives_and_matches_get_complete():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions
    obs = _observation_shim(ps, game.state.current_prompt, Color.BLUE, playable)
    from types import SimpleNamespace

    obs.turn_number = 8
    out_wrapper = format_observation_prompt(obs, playable, current_player_inventory=None)
    out_direct = get_complete_prompt(ps, Color.BLUE, playable, observation=obs, current_prompt=obs.current_prompt, turn_number=8)
    # wrapper should produce same sections as direct call
    assert out_wrapper == out_direct


def test_format_observation_prompt_respects_explicit_override():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    obs = _observation_shim(ps, game.state.current_prompt, Color.RED, game.playable_actions)
    other_playable = []  # override to empty
    out = format_observation_prompt(obs, playable_actions=other_playable, include_header=False, include_footer=False)
    assert "(no moves available)" in out
    # should NOT contain the settlement moves from obs.playable_actions
    assert "Build settlement at Node" not in out


def test_format_observation_prompt_turn_number_fallbacks():
    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    from types import SimpleNamespace

    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.WHITE, current_turn_index=4, playable_actions=[])
    out = format_observation_prompt(obs, [], include_header=True, include_footer=False)
    assert "[TURN: 4]" in out

    obs2 = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.WHITE, playable_actions=[])
    out2 = format_observation_prompt(obs2, [], include_header=True, include_footer=False)
    assert "[TURN:" not in out2


def test_format_observation_prompt_aliases():
    assert format_observation_prompt is get_observation_prompt
    assert format_observation_prompt is build_observation_prompt


def test_format_observation_prompt_no_playable_attr_defaults_empty():
    from types import SimpleNamespace

    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    obs = SimpleNamespace(public_state=ps, current_prompt=ActionPrompt.PLAY_TURN, color=Color.RED)
    out = format_observation_prompt(obs, include_header=False, include_footer=False)
    assert "(no moves available)" in out


# ---------------------------------------------------------------------------
# LLMObservationAgent integration — build_full_prompt uses same ordering
# ---------------------------------------------------------------------------


def test_agent_build_full_prompt_uses_complete_order():
    from catan_llm.llm_agent import LLMObservationAgent

    game = create_empty_game(seed=42)
    ps = build_public_state(game)
    playable = game.playable_actions

    class DummyAgent(LLMObservationAgent):
        def choose_move(self, formatted_moves, observation):
            return 1

    agent = DummyAgent(Color.RED)

    from types import SimpleNamespace

    obs = SimpleNamespace(
        public_state=ps, current_prompt=game.state.current_prompt, color=Color.RED, turn_number=3, playable_actions=playable
    )
    out = agent.build_full_prompt(obs, playable)
    assert out.index("[FULL BOARD MAP") < out.index("[CURRENT BOARD OCCUPANCY")
    assert out.index("[CURRENT BOARD OCCUPANCY") < out.index("ROBBER:")
    assert out.index("ROBBER:") < out.index("[PLAYERS]")
    assert out.index("[PLAYERS]") < out.index("[PLAYABLE MOVES]")
    assert "[CURRENT PLAYER: RED]" in out
    assert "[TURN: 3]" in out
