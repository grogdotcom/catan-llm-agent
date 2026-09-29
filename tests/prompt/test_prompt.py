"""PromptBuilder + deterministic strategy-block handling."""

import random

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color, Player
from catanatron.models.perspective_player import _build_public_state

from catan_llm.domain import PromptArtifact, PromptContext
from catan_llm.llm.prompt import PromptBuilder, ensure_strategy_block, inject_strategy_into_prompt, normalize_strategy_block


def _pub_state():
    class S(Player):
        def __init__(self, c):
            super().__init__(c)

        def decide(self, g, a):
            return a[0]

        def reset_state(self):
            pass

    random.seed(42)
    g = Game([S(Color.RED), S(Color.BLUE), S(Color.ORANGE), S(Color.WHITE)])
    return _build_public_state(g)


class _Obs:
    def __init__(self, color=None, public_state=None, current_prompt=None):
        self.color = color
        self.public_state = public_state
        self.current_prompt = current_prompt


def test_build_returns_artifact():
    obs = _Obs(color=Color.RED, public_state=_pub_state())
    b = PromptBuilder()
    art = b.build(PromptContext(observation=obs, playable_actions=[], current_strategy=None))
    assert isinstance(art, PromptArtifact)
    assert "[CURRENT STRATEGY]" in art.text
    assert art.moves == []
    assert art.version == "1.0"


def test_build_renders_strategy():
    obs = _Obs(color=Color.RED, public_state=_pub_state())
    b = PromptBuilder()
    art = b.build(PromptContext(observation=obs, playable_actions=[], current_strategy="Go ore"))
    assert "[CURRENT STRATEGY]\nGo ore" in art.text
    assert "Go ore\n\n" in art.text


def test_render_convenience():
    obs = _Obs(color=Color.RED, public_state=_pub_state())
    b = PromptBuilder()
    assert "[CURRENT STRATEGY]\nNone" in b.render(PromptContext(observation=obs, playable_actions=[]))


def test_build_embeds_moves():
    acts = [
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.END_TURN, None),
    ]
    obs = _Obs(color=Color.RED, public_state=_pub_state(), current_prompt=None)
    b = PromptBuilder()
    art = b.build(PromptContext(observation=obs, playable_actions=acts, current_strategy="None"))
    assert [m.actions[0].action_type for m in art.moves] == [ActionType.END_TURN, ActionType.END_TURN]
    assert "1. " in art.text


def test_inject_replaces_none_block():
    p = "[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]"
    out = inject_strategy_into_prompt(p, "Plan A")
    assert out == "[CURRENT STRATEGY]\nPlan A\n\n[RECENT TURNS]"


def test_inject_replaces_existing_value():
    p = "[CURRENT STRATEGY]\nOld plan\n\n[RECENT TURNS]"
    out = inject_strategy_into_prompt(p, "New plan")
    assert "[CURRENT STRATEGY]\nNew plan\n\n[RECENT TURNS]" in out


def test_inject_noop_when_empty():
    p = "[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]"
    assert inject_strategy_into_prompt(p, "") == p


def test_inject_absent_marker_returns_unchanged():
    p = "no marker here"
    assert inject_strategy_into_prompt(p, "x") == p


def test_ensure_inserts_before_recent_turns():
    p = "board stuff\n\n[RECENT TURNS]\nhist"
    out = ensure_strategy_block(p, "Plan")
    assert "[CURRENT STRATEGY]\nPlan\n\n[RECENT TURNS]" in out
    assert out.count("[CURRENT STRATEGY]") == 1


def test_ensure_none_value():
    p = "x\n\n[RECENT TURNS]"
    assert "[CURRENT STRATEGY]\nNone\n\n[RECENT TURNS]" in ensure_strategy_block(p, None)


def test_normalize_idempotent():
    p = "x\n\n[RECENT TURNS]\nhist"
    once = normalize_strategy_block(p, "S")
    assert normalize_strategy_block(once, "S") == once


def test_ensure_prepends_when_no_markers():
    p = " lone prompt"
    out = ensure_strategy_block(p, "Plan")
    assert out.startswith("[CURRENT STRATEGY]\nPlan\n\n")
    assert out.count("[CURRENT STRATEGY]") == 1
