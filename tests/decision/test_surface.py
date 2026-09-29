"""DecisionSurface — present/resolve/plan across the shared seam."""

from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color

from catan_llm.llm.decision import DecisionSurface, MoveExecutor, MoveChoices, PlannedMove
from catan_llm.format import AUTO_ROAD, Move


class _Obs:
    def __init__(self, current_prompt=None, public_state=None):
        self.current_prompt = current_prompt
        self.public_state = public_state


def test_present_counts_and_text():
    acts = [Action(Color.RED, ActionType.END_TURN, None)]
    s = DecisionSurface(_Obs())
    choices = s.present(acts)
    assert isinstance(choices, MoveChoices)
    assert len(choices) == 1
    assert "1. " in choices.text
    assert choices.count == 1


def test_present_empty():
    s = DecisionSurface(_Obs())
    choices = s.present([])
    assert len(choices) == 0
    assert "no moves available" in choices.text


def test_parse_and_plan():
    acts = [
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.END_TURN, None),
    ]
    s = DecisionSurface(_Obs())
    choices = s.present(acts)
    assert len(choices) == 2
    mv = s.parse("2", choices)
    assert mv.actions[0].action_type == ActionType.END_TURN
    plan = s.plan("1", choices)
    assert isinstance(plan, PlannedMove)
    assert plan.first_action() is not None
    assert plan.pending == []


def test_plan_preserves_followups():
    a1 = Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))
    a2 = Action(Color.RED, ActionType.BUILD_ROAD, (1, 2))
    m = Move(actions=[a1, a2], label="two")
    # build a surface whose format embeds both actions as one move
    class FakeFormatter:
        def build(self):
            return [m]

    observed = _Obs()
    choices = MoveChoices(moves=[m], text="1. two")
    s = DecisionSurface(observed)
    s.present = lambda acts: choices
    plan = s.plan("1", choices)
    assert plan.first_action() == a1
    assert plan.pending == [a2]


def test_plan_invalid_raises():
    s = DecisionSurface(_Obs())
    choices = MoveChoices(moves=[Move(actions=[], label="x")], text="1. x")
    try:
        s.plan("9", choices)
        assert False
    except ValueError:
        assert True


def test_executor_consumes_pending():
    ex = MoveExecutor()
    a1 = Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))
    a2 = Action(Color.RED, ActionType.BUILD_ROAD, (1, 2))
    ex._pending = [a2]
    assert ex.next([a2], None) == a2


def test_executor_queues_auto_road():
    ex = MoveExecutor()
    road = Action(Color.RED, ActionType.BUILD_ROAD, (5, 6))
    m = Move(actions=[Action(Color.RED, ActionType.BUILD_ROAD, (0, 1)), AUTO_ROAD], label="auto")
    ex.submit(m)
    assert ex.has_pending() is True
    assert ex.next([road], None) == road
