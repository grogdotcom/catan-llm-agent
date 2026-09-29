"""Canonical domain records.

These dataclasses define the shared vocabulary across the pipeline:

- ``DecisionRecord`` — one supervised decision (prompt + ground-truth action).
- ``Trajectory`` — provenance for one simulated game.
- ``PromptArtifact`` — the output of the prompt builder.
- ``PromptContext`` — the input to the prompt builder.
- ``BatchRequest`` — one OpenAI Batch line.
- ``TeacherResponse`` — a parsed teacher answer (<think>/strategy/action).
- ``ValidationResult`` — outcome of running the validation gates.
- ``Checkpoint`` — one selected decision opportunity for teacher querying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptContext:
    """Input to the prompt builder.

    ``playable_actions`` carries the engine-legal actions; ``current_strategy``
    is the strategy text to render in ``[CURRENT STRATEGY]`` (``None`` renders
    as ``None``). ``footer_policy`` selects which footer to emit.
    """

    observation: Any
    playable_actions: Optional[List[Any]] = None
    current_player_color: Any = None
    current_player_inventory: Any = None
    current_strategy: Optional[str] = None
    footer: Optional[str] = None
    include_footer: bool = True
    history_window_size: Optional[int] = 8


@dataclass
class PromptArtifact:
    """Result of the prompt builder.

    ``text`` is the rendered prompt string; ``moves`` is the ordered, numbered
    set of LLM-choosable moves the prompt describes (1-indexed by position);
    ``version`` identifies the prompt template revision so corpus consumers can
    detect semantic drift.
    """

    text: str
    moves: List[Any] = field(default_factory=list)
    version: str = "1.0"


@dataclass
class DecisionRecord:
    """One supervised decision, mirroring the JSONL corpus schema.

    ``prompt`` is the rendered prompt, ``completion`` is the ground-truth
    (1-indexed) move index as a string, and the remaining fields carry the
    metadata needed to join back to provenance and batch results.
    """

    prompt: str
    completion: str
    chosen_index: Optional[int] = None
    chosen_label: Optional[str] = None
    phase: Optional[str] = None
    color: Optional[str] = None
    turn: Optional[int] = None
    num_moves: Optional[int] = None
    move_labels: Optional[List[str]] = None
    selected_action: Optional[Dict[str, Any]] = None
    winner: Optional[str] = None
    game_id: Optional[int] = None
    decision_id: Optional[int] = None
    # provenance + strategy annotations (may be absent in legacy corpora)
    trajectory_id: Optional[str] = None
    game_seed: Optional[int] = None
    seat_index: Optional[int] = None
    game_end_turn: Optional[int] = None
    placement_round: Optional[int] = None
    is_first_placement: Optional[bool] = None
    injected_strategy: Optional[str] = None
    original_prompt: Optional[str] = None
    top_action_values: Optional[List[Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "prompt": self.prompt,
            "completion": self.completion,
        }
        for key in (
            "chosen_index",
            "chosen_label",
            "phase",
            "color",
            "turn",
            "num_moves",
            "move_labels",
            "selected_action",
            "winner",
            "game_id",
            "decision_id",
            "trajectory_id",
            "game_seed",
            "seat_index",
            "game_end_turn",
            "placement_round",
            "is_first_placement",
            "injected_strategy",
            "original_prompt",
            "top_action_values",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        known = {
            "prompt",
            "completion",
            "chosen_index",
            "chosen_label",
            "phase",
            "color",
            "turn",
            "num_moves",
            "move_labels",
            "selected_action",
            "winner",
            "game_id",
            "decision_id",
            "trajectory_id",
            "game_seed",
            "seat_index",
            "game_end_turn",
            "placement_round",
            "is_first_placement",
            "injected_strategy",
            "original_prompt",
            "top_action_values",
        }
        kwargs = {k: data[k] for k in known if k in data}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extra=extra)


@dataclass
class Trajectory:
    """Provenance metadata for one simulated game."""

    trajectory_id: str
    game_seed: Optional[int] = None
    game_id: Optional[int] = None
    game_end_turn: Optional[int] = None
    winner: Optional[str] = None
    seat_order: Optional[List[str]] = None
    players: Optional[List[Dict[str, Any]]] = None
    source_hash: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"trajectory_id": self.trajectory_id}
        for key in (
            "game_seed",
            "game_id",
            "game_end_turn",
            "winner",
            "seat_order",
            "players",
            "source_hash",
        ):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        return out


@dataclass
class BatchRequest:
    """One OpenAI Batch line (``custom_id``/``method``/``url``/``body``)."""

    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def as_dict(self, *, strict: bool = True) -> Dict[str, Any]:
        out = {
            "custom_id": self.custom_id,
            "method": self.method,
            "url": self.url,
            "body": self.body,
        }
        if not strict and self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class TeacherResponse:
    """Parsed teacher answer (``<think><strategy><action>``)."""

    think_text: Optional[str] = None
    strategy: Optional[str] = None
    action: Optional[int] = None
    raw_text: str = ""
    api_error: Optional[str] = None


@dataclass
class ValidationResult:
    """Outcome of running validation gates on a teacher response."""

    accepted: bool
    rejection_reason: Optional[str] = None
    think_text: Optional[str] = None
    strategy_out: Optional[str] = None
    predicted_action: Optional[int] = None


@dataclass
class Checkpoint:
    """One selected decision opportunity to query the teacher at."""

    trajectory_id: str
    seat_index: int
    checkpoint_index: int
    checkpoint_count: int
    band: Optional[int] = None
    target_progress: Optional[float] = None
    normalized_progress: Optional[float] = None
    selection_reason: Optional[str] = None
    opportunity: Optional[Dict[str, Any]] = None
    strategy_in: Optional[str] = None
    strategy_source_checkpoint: Optional[str] = None
    status: str = "pending"
    skip_reason: Optional[str] = None
