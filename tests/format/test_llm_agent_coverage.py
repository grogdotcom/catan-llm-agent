"""Exact-string tests for llm_agent — executor and prompt building."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color
from catan_llm.llm.agent import MoveExecutor
from catan_llm.format import Move, AUTO_ROAD, pick_auto_road

def test_executor_empty():
    ex = MoveExecutor()
    assert ex.has_pending() is False
    assert ex.submit(Move(actions=[], label="empty")) is None
    assert ex.next([]) is None

def test_executor_submit_and_next():
    ex = MoveExecutor()
    a1 = Action(Color.RED, ActionType.BUILD_ROAD, (0,1))
    a2 = Action(Color.RED, ActionType.BUILD_ROAD, (1,2))
    m = Move(actions=[a1, a2], label="two")
    first = ex.submit(m)
    assert first == a1
    assert ex.has_pending() is True
    nxt = ex.next([a2], public_state=None)
    assert nxt == a2
    assert ex.next([]) is None

def test_executor_auto_road_resolves():
    ex = MoveExecutor()
    m = Move(actions=[Action(Color.RED, ActionType.BUILD_ROAD, (0,1)), AUTO_ROAD], label="auto")
    ex.submit(m)
    # AUTO_ROAD should resolve when playable are all roads
    road = Action(Color.RED, ActionType.BUILD_ROAD, (5,6))
    nxt = ex.next([road], public_state=None)
    assert nxt == road

def test_executor_auto_road_drops_when_not_roads_only():
    ex = MoveExecutor()
    m = Move(actions=[AUTO_ROAD], label="auto")
    ex._pending = [AUTO_ROAD]
    # playable includes non-road -> should drop and return None
    mixed = [Action(Color.RED, ActionType.BUILD_SETTLEMENT, 1), Action(Color.RED, ActionType.BUILD_ROAD, (0,1))]
    assert ex.next(mixed) is None
    assert ex.has_pending() is False

def test_executor_auto_road_none_when_no_pick():
    ex = MoveExecutor()
    ex._pending = [AUTO_ROAD]
    # empty playable -> pick_auto_road returns None -> drops
    assert ex.next([], public_state=None) is None

def test_llm_agent_build_full_prompt_inventory_fallback():
    from catan_llm.llm.agent import LLMObservationAgent
    from catanatron.game import Game
    from catanatron.models.player import Player
    import random
    class S(Player):
        def __init__(self,c): self.color=c; self.is_bot=True
        def decide(self,g,a): return a[0]
        def reset_state(self): pass
    random.seed(42)
    g = Game([S(Color.RED), S(Color.BLUE), S(Color.ORANGE), S(Color.WHITE)])
    from catanatron.models.perspective_player import _build_public_state
    from catanatron.models.observation import Observation
    from catanatron.features import create_sample
    from catanatron.models.perspective_player import _sanitize_history, _build_pending_trades
    ps = _build_public_state(g)
    obs = Observation(color=Color.RED, features=create_sample(g, Color.RED), public_history=(), current_prompt=g.state.current_prompt, pending_trades={}, public_state=ps, inventory=None)
    # No inventory, should still build prompt with hidden
    agent = LLMObservationAgent(Color.RED)
    prompt = agent.build_full_prompt(obs, [], current_player_inventory=None)
    assert "[PLAYERS]" in prompt
    assert "RED (YOU)" in prompt

def test_llm_agent_choose_move_not_implemented():
    from catan_llm.llm.agent import LLMObservationAgent
    agent = LLMObservationAgent(Color.RED)
    try:
        agent.choose_move("moves", None)
        assert False
    except NotImplementedError:
        assert True
