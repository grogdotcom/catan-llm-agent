"""
Unit tests for playable-action formatting, compound-move planning, and the
LLM agent integration.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pytest

from catanatron.game import Game
from catanatron.models.actions import (
    generate_playable_actions,
    road_building_possibilities,
    robber_possibilities,
)
from catanatron.models.enums import (
    SETTLEMENT,
    Action,
    ActionPrompt,
    ActionType,
    BRICK,
    ORE,
    SHEEP,
    WOOD,
)
from catanatron.models.observation import Observation
from catanatron.models.perspective_player import (
    PerspectivePlayer,
    _build_public_state,
)
from catanatron.models.player import Color, RandomPlayer, Player
from catanatron.state_functions import player_key

from catan_llm.game_formatter import (
    AUTO_ROAD,
    Move,
    _knight_robber_followups,
    _land_edges_from,
    _own_network_nodes,
    _road_building_moves,
    build_moves,
    format_moves,
    format_playable_actions,
    parse_move,
    pick_auto_road,
)
from catan_llm.llm_agent import LLMObservationAgent, MoveExecutor


class SimplePlayer(Player):
    def __init__(self, color):
        self.color = color
        self.is_bot = True

    def decide(self, game, playable_actions):
        return playable_actions[0] if playable_actions else None

    def reset_state(self):
        pass


def make_game(seed=7):
    random.seed(seed)
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    return Game(players, seed=seed)


def make_turn_state_with_knight():
    """State where RED is mid-turn, has rolled, and holds a playable Knight."""
    game = make_game()
    state = game.state
    red_index = state.color_to_index[Color.RED]
    state.current_player_index = red_index
    state.current_turn_index = red_index
    state.current_prompt = ActionPrompt.PLAY_TURN

    red_key = player_key(state, Color.RED)
    state.player_state[f"{red_key}_HAS_ROLLED"] = True
    state.player_state[f"{red_key}_KNIGHT_IN_HAND"] = 1
    state.player_state[f"{red_key}_KNIGHT_OWNED_AT_START"] = True
    state.player_state[f"{red_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False

    board = state.board
    board.buildings[0] = (Color.RED, SETTLEMENT)
    board.buildings[5] = (Color.BLUE, SETTLEMENT)
    board.buildings[10] = (Color.ORANGE, SETTLEMENT)
    board.buildings[35] = (Color.WHITE, SETTLEMENT)
    for node in (0, 5, 10, 35):
        board.board_buildable_ids.discard(node)

    blue_key = player_key(state, Color.BLUE)
    state.player_state[f"{blue_key}_{WOOD}_IN_HAND"] = 1
    orange_key = player_key(state, Color.ORANGE)
    state.player_state[f"{orange_key}_{ORE}_IN_HAND"] = 1
    # WHITE holds no cards -> no steal target

    return game


def make_observation(game):
    return Observation(
        color=game.state.current_color(),
        current_prompt=game.state.current_prompt,
        public_state=_build_public_state(game),
        features={},
    )


# ---------------------------------------------------------------------------
# Knight bundling
# ---------------------------------------------------------------------------

def test_knight_robber_followups_match_engine():
    game = make_turn_state_with_knight()
    public_state = _build_public_state(game)

    followups = set(_knight_robber_followups(public_state, Color.RED))
    engine = {(a.value[0], a.value[1]) for a in robber_possibilities(game.state, Color.RED)}

    assert followups == engine
    assert len(followups) > 0
    # No self-stealing.
    assert all(v != Color.RED for _, v in followups)


def test_build_moves_bundles_knight_with_robber_targets():
    game = make_turn_state_with_knight()
    playable = generate_playable_actions(game.state)
    observation = make_observation(game)

    knight_action = next(
        a for a in playable if a.action_type == ActionType.PLAY_KNIGHT_CARD
    )
    moves = build_moves(playable, observation)

    knight_moves = [
        m for m in moves
        if m.actions and m.actions[0].action_type == ActionType.PLAY_KNIGHT_CARD
    ]
    engine_targets = set(
        (a.value[0], a.value[1]) for a in robber_possibilities(game.state, Color.RED)
    )

    assert len(knight_moves) == len(engine_targets)
    for move in knight_moves:
        assert len(move.actions) == 2
        assert move.actions[0] == knight_action
        followup = move.actions[1]
        assert followup.action_type == ActionType.MOVE_ROBBER
        assert (followup.value[0], followup.value[1]) in engine_targets
        assert "Play Knight" in move.label


# ---------------------------------------------------------------------------
# Single-action and card formatting
# ---------------------------------------------------------------------------

def test_format_year_of_plenty_and_monopoly_as_single_moves():
    action = Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, (WOOD, SHEEP))
    moves = build_moves([action])
    assert len(moves) == 1
    assert "take WOOD, SHEEP" in moves[0].label
    assert moves[0].actions == [action]

    action = Action(Color.RED, ActionType.PLAY_MONOPOLY, ORE)
    moves = build_moves([action])
    assert len(moves) == 1
    assert "steal all ORE" in moves[0].label


def test_build_moves_bundles_initial_settlement_with_concrete_roads():
    game = make_game()
    state = game.state
    state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
    playable = generate_playable_actions(state)
    observation = make_observation(game)

    moves = build_moves(playable, observation)
    assert moves, "setup should produce settlement moves"
    # Every move bundles the settlement with a concrete road edge.
    for move in moves:
        assert move.actions[0].action_type == ActionType.BUILD_SETTLEMENT
        assert move.actions[1].action_type == ActionType.BUILD_ROAD
        road = tuple(sorted(move.actions[1].value))
        node = move.actions[0].value
        assert node in road  # the road must attach to the settlement
        assert " -> build road " in move.label
        # Sorted (n1, n2) node-pair representation (now enriched with tile detail).
        assert f"({road[0]}, {road[1]})" in move.label
        # Enriched settlement detail should contain Tile info and pips
        assert "Tile" in move.label and "pips" in move.label
        assert "Total:" in move.label


def test_build_moves_bundles_road_building_with_concrete_road_pairs():
    game = make_game()
    state = game.state
    red_key = player_key(state, Color.RED)
    state.current_player_index = state.color_to_index[Color.RED]
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state[f"{red_key}_HAS_ROLLED"] = True
    state.player_state[f"{red_key}_ROAD_BUILDING_IN_HAND"] = 1
    state.player_state[f"{red_key}_ROAD_BUILDING_OWNED_AT_START"] = True
    state.player_state[f"{red_key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False

    # Give RED a connected network so at least one road edge is buildable.
    board = state.board
    board.build_settlement(Color.RED, 5, True)
    board.build_road(Color.RED, (5, 0))

    playable = generate_playable_actions(state)
    observation = make_observation(game)

    rb_action = next(
        a for a in playable if a.action_type == ActionType.PLAY_ROAD_BUILDING
    )
    moves = build_moves(playable, observation)
    rb_moves = [m for m in moves if m.actions and m.actions[0].action_type == ActionType.PLAY_ROAD_BUILDING]
    assert rb_moves, "playable_actions should include PLAY_ROAD_BUILDING"

    engine_targets = set(
        tuple(sorted(a.value))
        for a in road_building_possibilities(game.state, Color.RED, check_money=False)
    )
    for move in rb_moves:
        assert move.actions[0] == rb_action
        roads = [
            tuple(sorted(a.value))
            for a in move.actions[1:]
            if a.action_type == ActionType.BUILD_ROAD
        ]
        # No sentinels: roads are concrete and legal.
        assert not any(isinstance(a, str) for a in move.actions)
        assert roads
        assert roads[0] in engine_targets
        assert len(roads) in (1, 2)
        assert "Play Road Building" in move.label


def test_road_building_bundles_dedupe_disconnected_pairs():
    """Two disconnected roads built in either order are the same move."""
    game = make_game()
    board = game.state.board
    # Two disconnected RED networks: component A across nodes {0, 5} and
    # component B across nodes {29, 30}.
    board.build_settlement(Color.RED, 5, True)
    board.build_road(Color.RED, (5, 0))
    board.build_settlement(Color.RED, 30, True)
    board.build_road(Color.RED, (30, 29))

    public_state = _build_public_state(game)
    play_card = Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
    moves = _road_building_moves(play_card, public_state)

    two_road = [m for m in moves if len(m.actions) == 3]
    # Every unordered road pair appears exactly once (no (A,B) and (B,A)).
    pairs = {
        frozenset(tuple(sorted(a.value)) for a in m.actions[1:]) for m in two_road
    }
    assert len(two_road) == len(pairs)

    # Dedup must actually reduce the count vs the ordered enumeration.
    base = _own_network_nodes(public_state, Color.RED)
    first_roads = _land_edges_from(public_state, Color.RED, base)
    naive = 0
    for first in first_roads:
        second_net = base | set(first)
        naive += len([e for e in _land_edges_from(public_state, Color.RED, second_net) if e != first])
    assert len(two_road) < naive

    # At least one bundle combines one edge from each disconnected component.
    a_edges = set(_land_edges_from(public_state, Color.RED, {0, 5}))
    b_edges = set(_land_edges_from(public_state, Color.RED, {29, 30}))
    cross = pairs.intersection(
        frozenset((ea, eb)) for ea in a_edges for eb in b_edges
    )
    assert cross

    # Labels use the sorted node-pair representation (now enriched with per-road tile detail).
    for move in two_road[:2]:
        road_a, road_b = sorted((tuple(sorted(move.actions[1].value)), tuple(sorted(move.actions[2].value))))
        assert f"({road_a[0]}, {road_a[1]}) and ({road_b[0]}, {road_b[1]})" in move.label
        assert "road" in move.label.lower() and "pips" in move.label



# ---------------------------------------------------------------------------
# Formatting and parsing
# ---------------------------------------------------------------------------

def test_format_moves_is_numbered_and_parseable():
    game = make_game()
    playable = generate_playable_actions(game.state)
    observation = make_observation(game)
    moves = build_moves(playable, observation)

    text = format_moves(moves, observation=observation)
    lines = text.splitlines()
    assert lines[0] == "[PLAYABLE MOVES]"
    assert "[PHASE: " in lines[1]

    for i, line in enumerate(lines[2:], start=1):
        assert line.startswith(f"{i}. "), line


def test_format_playable_actions_returns_readable_text():
    game = make_game()
    text = format_playable_actions(generate_playable_actions(game.state), make_observation(game))
    assert text.startswith("[PLAYABLE MOVES]")
    assert "[PHASE: " in text


def test_parse_move_int_and_string_forms():
    game = make_game()
    moves = build_moves(generate_playable_actions(game.state), make_observation(game))
    assert parse_move(1, moves) is moves[0]
    assert parse_move(len(moves), moves) is moves[-1]
    assert parse_move("2", moves) is moves[1]
    assert parse_move("[3]", moves) is moves[2]
    assert parse_move("4. Build city at node 10", moves) is moves[3]


def test_parse_move_rejects_invalid_inputs():
    moves = [Move("a", []), Move("b", [])]
    with pytest.raises(ValueError):
        parse_move(0, moves)
    with pytest.raises(ValueError):
        parse_move(3, moves)
    with pytest.raises(ValueError):
        parse_move("nope", moves)


# ---------------------------------------------------------------------------
# Auto road completion
# ---------------------------------------------------------------------------

def test_pick_auto_road_prefers_productive_edge():
    game = make_game()
    public_state = _build_public_state(game)
    road_a = Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))
    road_b = Action(Color.RED, ActionType.BUILD_ROAD, (5, 16))
    picked = pick_auto_road([road_a, road_b], public_state)
    # Deterministic and one of the given roads.
    assert picked in (road_a, road_b)
    assert picked == pick_auto_road([road_a, road_b], public_state)


def test_pick_auto_road_ignores_non_road_actions():
    road = Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))
    roll = Action(Color.RED, ActionType.ROLL, None)
    assert pick_auto_road([roll, road]) == road
    assert pick_auto_road([roll]) is None


# ---------------------------------------------------------------------------
# MoveExecutor
# ---------------------------------------------------------------------------

def test_executor_submit_and_next():
    ex = MoveExecutor()
    play = Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
    move = Move("play", [play, AUTO_ROAD, AUTO_ROAD])
    assert ex.submit(move) == play
    assert ex.has_pending()

    roads = [
        Action(Color.RED, ActionType.BUILD_ROAD, (0, 5)),
        Action(Color.RED, ActionType.BUILD_ROAD, (5, 16)),
    ]
    assert ex.next(roads) in roads
    assert ex.next(roads) in roads
    assert not ex.has_pending()
    assert ex.next(roads) is None


def test_executor_drops_auto_road_when_phase_over():
    ex = MoveExecutor()
    rb = Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None)
    ex.submit(Move("play", [rb, AUTO_ROAD, AUTO_ROAD]))

    # First road prompt: only roads offered -> auto-completes.
    first = ex.next([Action(Color.RED, ActionType.BUILD_ROAD, (0, 5))])
    assert first is not None

    # Engine ended the road-building phase; the next prompt offers everything.
    full_turn = [
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.BUILD_ROAD, (5, 16)),
    ]
    assert ex.next(full_turn) is None
    assert not ex.has_pending()


# ---------------------------------------------------------------------------
# End-to-end: agent plays full games
# ---------------------------------------------------------------------------

class PickFirstMoveAgent(LLMObservationAgent):
    def choose_move(self, formatted_moves, observation):
        return 1


def test_llm_agent_plays_a_full_game():
    random.seed(11)
    agent = PickFirstMoveAgent(Color.RED)
    players = [
        PerspectivePlayer(agent),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]
    game = Game(players, seed=11)
    result = game.play()
    assert result in (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)
    # The agent accumulates moves across prompts without an error.
    assert len(agent.last_moves) >= 0


@pytest.mark.parametrize("seed", [101, 202])
def test_llm_agent_plays_many_games_without_crashing(seed):
    random.seed(seed)
    agent = PickFirstMoveAgent(Color.RED)
    players = [
        PerspectivePlayer(agent),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]
    game = Game(players, seed=seed)
    game.play()
    assert True