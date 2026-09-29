"""
LLM-driven ObservationAgent built on the decision surface.

Compound moves (Knight + robber move, Road Building + two roads, initial
settlement + road) span multiple engine prompts. ``MoveExecutor`` queues the
follow-up actions of a chosen Move so ``decide_observation`` completes the move
without asking the LLM again. Road placements that cannot be bundled upfront
(initial-placement road, Road Building roads) are resolved automatically from
the live prompt's playable actions.
"""

from typing import List, Optional

from catanatron.models.enums import Action
from catanatron.models.observation_agent import ObservationAgent

from catan_llm.llm.decision import MoveExecutor
from catan_llm.format import get_complete_prompt

# Public agent surface used by runtime adapters and corpus collection.
__all__ = ["MoveExecutor", "LLMObservationAgent"]


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

        from catan_llm.llm.decision import DecisionSurface

        surface = DecisionSurface(observation)
        choices = surface.present(playable_actions)
        self.last_moves = choices.moves
        response = self.choose_move(choices.text, observation)
        plan = surface.plan(response, choices)
        return self.executor.submit(plan.move)

    def build_full_prompt(
        self,
        observation,
        playable_actions,
        current_player_inventory=None,
        include_footer: bool = True,
        footer: Optional[str] = None,
        current_strategy: Optional[str] = None,
    ) -> str:
        """Build the integrated prompt for this observation.

        Order is board → occupancy → robber → header → players → current strategy
        → recent turns (last 8) → moves. This is the recommended prompt to send
        to the LLM; ``choose_move`` subclasses that want the full context can
        call this instead of using ``formatted_moves`` directly, then parse the
        chosen index via ``parse_move``.

        Args:
            observation: Current Observation (carries ``public_state`` and phase).
            playable_actions: Legal actions for the current prompt (if ``None``,
                the agent's ``color`` and inventory are still used to render
                the ``[PLAYERS]`` section with ``(YOU)`` marked).
            current_player_inventory: Optional private Inventory for the
                observer. When ``None``, attempts to use ``observation.inventory``
                / ``observation.player_state`` if available, otherwise renders
                the observer as hidden.
            include_footer: See :func:`catan_llm.format.get_complete_prompt`.
                Set ``False`` to omit the default ``[DECISION REQUIRED]`` suffix.
            footer: See :func:`catan_llm.format.get_complete_prompt`. Custom
                text appended after ``[PLAYABLE MOVES]``. When provided it
                overrides ``include_footer``. Example for chain-of-thought:
                ``footer="[THINK]\\nExplain your reasoning, then output the move number."``.
            current_strategy: Optional strategy text carried from the previous
                turn. When ``None`` (e.g. first initial placement) the
                ``[CURRENT STRATEGY]`` section shows ``None``. Falls back to
                ``observation.current_strategy`` / ``observation.strategy`` when
                not supplied.

        Returns:
            Complete prompt string. See :func:`catan_llm.format.get_complete_prompt`.
        """
        inventory = current_player_inventory
        if inventory is None:
            inventory = getattr(observation, "inventory", None)
            if inventory is None:
                inventory = getattr(observation, "player_state", None)
        # Prefer observation's own public_state / color
        public_state = getattr(observation, "public_state", None)
        color = getattr(observation, "color", self.color)
        # Some Observations expose turn_number; fall back to None
        turn_number = getattr(observation, "turn_number", None)
        if turn_number is None:
            turn_number = getattr(observation, "current_turn_index", None)
        # Strategy: explicit arg > observation.current_strategy > observation.strategy
        strategy = current_strategy
        if strategy is None and observation is not None:
            strategy = getattr(observation, "current_strategy", None)
            if strategy is None:
                strategy = getattr(observation, "strategy", None)
        return get_complete_prompt(
            public_state=public_state,
            current_player_color=color,
            playable_actions=playable_actions,
            current_player_inventory=inventory,
            observation=observation,
            current_prompt=getattr(observation, "current_prompt", None),
            turn_number=turn_number,
            include_footer=include_footer,
            footer=footer,
            current_strategy=strategy,
        )

    def choose_move(self, formatted_moves: str, observation) -> str:
        """Send the formatted move list to the model; return the chosen index.

        Args:
            formatted_moves: The numbered ``[PLAYABLE MOVES]`` block.
            observation: The current Observation (for building a richer prompt).

        Returns:
            The index number of the chosen move, as an int or numeric string.
        """
        raise NotImplementedError("Implement choose_move in a subclass.")
