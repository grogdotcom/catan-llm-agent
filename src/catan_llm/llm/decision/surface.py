"""The decision surface: present engine actions as LLM choices and resolve a choice.

``present`` renders the numbered ``[PLAYABLE MOVES]`` block plus the ordered
``Move`` list; ``resolve`` turns a model response into a concrete plan (the
chosen Move plus the queued engine follow-up actions, resolving ``AUTO_ROAD``
tokens against the live prompt).

This is the single seam shared by runtime agents, corpus collection, prompt
building, and any evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from catanatron.models.enums import Action

from catan_llm.format import Move, build_moves, format_moves, parse_move


@dataclass
class MoveChoices:
    """The numbered, labeled set of moves a model can pick from.

    ``moves[i]`` is the ``(i+1)``-indexed move described by ``text``.
    """

    moves: List[Move]
    text: str

    def __len__(self) -> int:
        return len(self.moves)

    @property
    def count(self) -> int:
        return len(self.moves)


@dataclass
class PlannedMove:
    """A resolved decision: the chosen Move plus the engine actions to execute.

    ``pending`` holds the follow-up actions after the first one; the first
    action is what the agent returns immediately to the engine. ``AUTO_ROAD``
    sentinels are preserved for runtime resolution against the live prompt.
    """

    move: Optional[Move]
    pending: List[Union[Action, str]] = field(default_factory=list)

    def first_action(self) -> Optional[Action]:
        """The action to hand the engine now (the head of the Move queue)."""
        if not self.move or not self.move.actions:
            return None
        return self.move.actions[0]

    @property
    def pending_count(self) -> int:
        return len(self.pending)


class DecisionSurface:
    """Deep interface to move semantics for one observation."""

    def __init__(self, observation: Any):
        self.observation = observation

    def present(self, playable_actions: Optional[List[Action]] = None) -> MoveChoices:
        """Build numbered moves and their formatted text for the observation."""
        actions = list(playable_actions or [])
        moves = build_moves(actions, self.observation)
        text = format_moves(moves, self.observation)
        return MoveChoices(moves=moves, text=text)

    def parse(self, response: Any, choices: MoveChoices) -> Move:
        """Convert a model response (a stable 1-indexed move id) into a Move."""
        return parse_move(response, choices.moves)

    def plan(self, response: Any, choices: MoveChoices) -> PlannedMove:
        """Resolve a model response into a concrete execution plan.

        The head action is returned to the engine immediately; follow-up
        actions (including ``AUTO_ROAD`` tokens) are queued in ``pending``.
        """
        move = self.parse(response, choices)
        if move is None:
            return PlannedMove(move=None, pending=[])
        queue = list(move.actions)
        return PlannedMove(move=move, pending=queue[1:])
