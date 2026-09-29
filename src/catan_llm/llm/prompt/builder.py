"""Canonical prompt builder.

Renders the six-section decision prompt from a structured ``PromptContext`` and
returns a ``PromptArtifact`` that carries the rendered text plus the numbered
``Move`` list and a template version. Strategy is a first-class input, so the
two-phase pipeline never has to patch already-rendered strings.

Individual renderers (board, players, history, moves) stay behind the format
package; this module is the composition seam and version owner.
"""

from __future__ import annotations

from typing import Any, List, Optional

from catan_llm.llm.decision import DecisionSurface
from catan_llm.domain import PromptArtifact, PromptContext
from catan_llm.format import get_complete_prompt

PROMPT_TEMPLATE_VERSION = "1.0"


class PromptBuilder:
    """Compose a decision prompt from a structured context.

    The builder is stateless between calls: pass a ``PromptContext`` per prompt
    and receive a ``PromptArtifact``. Configuration that varies per pipeline
    (e.g. default history window) is fixed at construction time.
    """

    def __init__(self, *, history_window_size: int = 8, template_version: str = PROMPT_TEMPLATE_VERSION):
        self.history_window_size = history_window_size
        self.template_version = template_version

    def build(self, context: PromptContext) -> PromptArtifact:
        observation = context.observation
        actions = context.playable_actions or []

        # Build moves once so artifact and text necessarily agree.
        surface = DecisionSurface(observation)
        choices = surface.present(actions)

        text = get_complete_prompt(
            public_state=getattr(observation, "public_state", None),
            current_player_color=context.current_player_color
            if context.current_player_color is not None
            else getattr(observation, "color", None),
            playable_actions=actions,
            current_player_inventory=context.current_player_inventory,
            observation=observation,
            history_window_size=context.history_window_size
            if context.history_window_size is not None
            else self.history_window_size,
            current_strategy=context.current_strategy,
            footer=context.footer,
            include_footer=context.include_footer,
        )
        return PromptArtifact(
            text=text,
            moves=choices.moves,
            version=self.template_version,
        )

    def render(self, context: PromptContext) -> str:
        """Convenience: render just the text."""
        return self.build(context).text


def build_artifact(
    observation: Any,
    playable_actions: Optional[List[Any]] = None,
    current_strategy: Optional[str] = None,
    footer: Optional[str] = None,
    include_footer: bool = True,
    current_player_color: Any = None,
    current_player_inventory: Any = None,
    history_window_size: Optional[int] = 8,
    builder: Optional[PromptBuilder] = None,
) -> PromptArtifact:
    """Module-level convenience, mirroring the old ergonomic wrappers."""
    b = builder or PromptBuilder()
    return b.build(
        PromptContext(
            observation=observation,
            playable_actions=playable_actions,
            current_player_color=current_player_color,
            current_player_inventory=current_player_inventory,
            current_strategy=current_strategy,
            footer=footer,
            include_footer=include_footer,
            history_window_size=history_window_size,
        )
    )
