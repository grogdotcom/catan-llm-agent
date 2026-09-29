"""Execution of a planned move across engine prompts.

``AUTO_ROAD`` tokens (from Road Building cards / initial-placement roads) are
resolved against the live prompt: they are only satisfied while the engine is
still exclusively offering roads; otherwise the remaining tokens are dropped so
the agent resumes normal decision-making.
"""

from __future__ import annotations

from typing import List, Optional, Union

from catanatron.models.enums import Action, ActionType

from catan_llm.format import AUTO_ROAD, Move, pick_auto_road


class MoveExecutor:
    """Queues the follow-up engine Actions of a chosen Move and resolves them.

    The executor is fed the plan (via ``submit``) and hands back one action per
    engine prompt (via ``next``). When the queue is empty the agent must consult
    the model again.
    """

    def __init__(self):
        self._pending: List[Union[Action, str]] = []

    def reset(self):
        self._pending = []

    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, move: Move) -> Optional[Action]:
        """Queue a chosen Move and return its first Action (or None)."""
        if move is None:
            return None
        queue = list(move.actions)
        if not queue:
            return None
        first = queue.pop(0)
        self._pending = queue
        return first

    def next(
        self,
        playable_actions: List[Action],
        public_state=None,
    ) -> Optional[Action]:
        """Return the next queued Action, or None when the queue is empty.

        ``AUTO_ROAD`` is resolved from the live prompt only while the prompt is
        still exclusively offering roads (i.e. the road-building phase is still
        active). If the phase ended early, remaining tokens are dropped.
        """
        while self._pending:
            item = self._pending.pop(0)
            if item == AUTO_ROAD:
                roads_only = playable_actions and all(
                    a.action_type == ActionType.BUILD_ROAD for a in playable_actions
                )
                if not roads_only:
                    self._pending = []
                    return None
                action = pick_auto_road(playable_actions, public_state)
                if action is None:
                    self._pending = []
                    return None
                return action
            return item
        return None
