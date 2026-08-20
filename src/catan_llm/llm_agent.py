"""
LLM-driven ObservationAgent built on the Move formatter.

Compound moves (Knight + robber move, Road Building + two roads, initial
settlement + road) span multiple engine prompts. ``MoveExecutor`` queues the
follow-up actions of a chosen Move so ``decide_observation`` completes the move
without asking the LLM again. Road placements that cannot be bundled upfront
(initial-placement road, Road Building roads) are resolved automatically from
the live prompt's playable actions (see ``pick_auto_road``).
"""

from typing import List, Optional, Union

from catanatron.models.enums import Action, ActionType
from catanatron.models.observation_agent import ObservationAgent
from catanatron.models.public_state import PublicState

from catan_llm.format import (
    AUTO_ROAD,
    Move,
    build_moves,
    format_moves,
    parse_move,
    pick_auto_road,
)


class MoveExecutor:
    """Holds the remaining engine Actions of the currently decided Move.

    The executor is fed the chosen Move (via ``submit``) and hands back one
    action per engine prompt (via ``next``). When the queue is empty the agent
    must consult the LLM again.
    """

    def __init__(self):
        self._pending: List[Union[Action, str]] = []

    def reset(self):
        self._pending = []

    def has_pending(self) -> bool:
        return bool(self._pending)

    def submit(self, move: Move) -> Optional[Action]:
        """Queue a chosen Move and return its first Action (or None)."""
        queue = list(move.actions)
        if not queue:
            return None
        first = queue.pop(0)
        self._pending = queue
        return first

    def next(
        self,
        playable_actions: List[Action],
        public_state: Optional[PublicState] = None,
    ) -> Optional[Action]:
        """Return the next queued Action, or None when the queue is empty.

        Sentinels are resolved from the live prompt: ``AUTO_ROAD`` picks a legal
        road from the current playable actions, but only while the prompt is
        still exclusively offering roads (i.e. the road-building phase is still
        active). If the phase ended early (e.g. only one road was available),
        the remaining tokens are dropped and None is returned so the LLM decides
        the rest of the turn normally.
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


class LLMObservationAgent(ObservationAgent):
    """ObservationAgent that asks an LLM to pick from the formatted moves.

    Subclass responsibility: implement ``choose_move`` to send ``formatted_moves``
    to your model and return the chosen move's index number (as an int or str).
    """

    def __init__(self, color):
        super().__init__(color)
        self.executor = MoveExecutor()
        self.last_moves: List[Move] = []

    def reset_state(self):
        super().reset_state()
        self.executor.reset()
        self.last_moves = []

    def decide_observation(self, observation, playable_actions):
        queued = self.executor.next(playable_actions, observation.public_state)
        if queued is not None:
            return queued

        moves = build_moves(playable_actions, observation)
        self.last_moves = moves
        text = format_moves(moves, observation=observation)
        response = self.choose_move(text, observation)
        move = parse_move(response, moves)
        return self.executor.submit(move)

    def choose_move(self, formatted_moves: str, observation) -> str:
        """Send the formatted move list to the model; return the chosen index.

        Args:
            formatted_moves: The numbered ``[PLAYABLE MOVES]`` block.
            observation: The current Observation (for building a richer prompt).

        Returns:
            The index number of the chosen move, as an int or numeric string.
        """
        raise NotImplementedError("Implement choose_move in a subclass.")