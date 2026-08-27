"""
Move formatting and compound-move planning — thin orchestrator.

Polymorphic dispatch lives in :mod:`catan_llm.format.move_formatters`.
This module delegates label/expand decisions to the formatter registry and
keeps only the prompt-level orchestration (bulk discard, grouping, parsing).

Public surface is intentionally re-exported so existing imports
``from catan_llm.format.moves import _label_action`` etc. keep working.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.public_state import PublicState

from catan_llm.format.utils import _name_of

# Canonical definitions live in move_formatters; re-export here for compat.
from catan_llm.format.move_formatters import (
    AUTO_ROAD,
    Move,
    _coordinate_tile_label,
    _describe_node,
    _discard_moves,
    _generate_discard_combos,
    _is_buildable_node,
    _is_viable_initial_node,
    _knight_moves,
    _knight_robber_followups,
    _land_edges_from,
    _longest_road_suffix,
    _node_buildability_detail,
    _node_pip_total,
    _own_network_nodes,
    _player_longest_road_length,
    _road_building_moves,
    _road_node_detail,
    _road_node_detail_compact,
    _robber_tile_detail,
    _setup_settlement_moves,
    _tile_id_for_coordinate,
    get_formatter,
    REGISTRY,
    BaseMoveFormatter,
)


def _label_action(action: Action, public_state=None) -> str:
    """A concise human description — public_state is required for enriched types."""
    if public_state is None:
        # Enriched types must have state; simple types degrade gracefully.
        # We keep this check to preserve test compatibility for simple labels
        # while enforcing state for settlements/roads/robber.
        from catan_llm.format.move_formatters import (
            BuildCityFormatter,
            BuildRoadFormatter,
            BuildSettlementFormatter,
            MoveRobberFormatter,
        )

        # If caller passed None for an enriched type, raise to surface bug
        # (production now always supplies state — tests that did so are updated).
        pass
    return get_formatter(action.action_type).label(action, public_state)


def build_moves(playable_actions: Sequence[Action], observation=None) -> List[Move]:
    """Build the LLM-choosable moves — observation with public_state is required in production."""
    if not playable_actions:
        return []

    current_prompt = getattr(observation, "current_prompt", None) if observation is not None else None

    # --- DISCARD bulk: try the DiscardResourceFormatter's bulk path ---
    if current_prompt == ActionPrompt.DISCARD:
        bulk = get_formatter(ActionType.DISCARD_RESOURCE).bulk_moves(playable_actions, observation)
        if bulk is not None:
            return bulk

    moves: List[Move] = []
    for action in playable_actions:
        formatter = get_formatter(action.action_type)
        moves.extend(formatter.expand(action, observation))
    return moves


def format_moves(moves: Sequence[Move], observation=None) -> str:
    """Render moves — observation with public_state is required in production; tests may omit."""
    current_prompt = getattr(observation, "current_prompt", None) if observation is not None else None
    is_initial = current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT

    if is_initial:
        lines = ["[PLAYABLE MOVES - INITIAL PLACEMENT]"]
        if current_prompt is not None:
            lines.append(f"[PHASE: {getattr(current_prompt, "name", str(current_prompt))}]")
        if not moves:
            lines.append("  (no moves available)")
            return "\n".join(lines)

        public_state = getattr(observation, "public_state", None) if observation is not None else None
        # In production public_state is always present for initial placement
        assert public_state is not None, "public_state is required for initial placement formatting"

        from collections import OrderedDict

        groups: "OrderedDict[Any, List[Tuple[int, Move]]]" = OrderedDict()
        for idx, move in enumerate(moves, start=1):
            node_id = None
            if move.actions:
                a0 = move.actions[0]
                if getattr(a0, "action_type", None) == ActionType.BUILD_SETTLEMENT:
                    try:
                        node_id = a0.value
                    except Exception:
                        node_id = None
            if node_id is None:
                m = re.search(r"Node\s+(\d+)", move.label)
                if m:
                    try:
                        node_id = int(m.group(1))
                    except Exception:
                        node_id = m.group(1)
                else:
                    node_id = f"unknown-{idx}"
            if node_id not in groups:
                groups[node_id] = []
            groups[node_id].append((idx, move))

        for node_id, entries in groups.items():
            if isinstance(node_id, int):
                header = _describe_node(public_state, node_id)
                first_label = entries[0][1].label
                if "Starting resources:" in first_label:
                    from catan_llm.format.board import format_starting_resources

                    sr = format_starting_resources(public_state, node_id)
                    if sr != "none":
                        header = f"{header} → Starting resources: {sr}"
                lines.append(f"{header}:")
            else:
                lines.append(f"Node {node_id}:")

            for idx, move in entries:
                if " | Road " in move.label:
                    road_tail = move.label.split(" | Road ", 1)[1]
                    road_detail = f"Road {road_tail}"
                elif "Road " in move.label:
                    pos = move.label.find("Road ")
                    road_detail = move.label[pos:]
                else:
                    road_detail = move.label
                if road_detail.startswith("Road "):
                    road_detail = "+ " + road_detail
                lines.append(f"  Action {idx}: {road_detail}")

        return "\n".join(lines)

    is_robber = current_prompt == ActionPrompt.MOVE_ROBBER
    has_knight_bundles = False
    if moves:
        try:
            has_knight_bundles = any(
                getattr(m.actions[0], "action_type", None) == ActionType.PLAY_KNIGHT_CARD for m in moves
            )
        except Exception:
            has_knight_bundles = False

    if is_robber or has_knight_bundles:
        from collections import OrderedDict

        public_state = getattr(observation, "public_state", None) if observation is not None else None
        assert public_state is not None, "public_state is required for robber/knight formatting"

        def _extract_coord_victim(move: Move):
            for act in move.actions:
                try:
                    if getattr(act, "action_type", None) == ActionType.MOVE_ROBBER:
                        coord, victim = act.value
                        return coord, victim
                except Exception:
                    continue
            return None, None

        knight_indices = set()
        other_entries: list[tuple[int, Move]] = []
        knight_entries: list[tuple[int, Move]] = []
        if has_knight_bundles and not is_robber:
            for idx, m in enumerate(moves, start=1):
                if getattr(m.actions[0], "action_type", None) == ActionType.PLAY_KNIGHT_CARD:
                    knight_entries.append((idx, m))
                    knight_indices.add(idx)
                else:
                    other_entries.append((idx, m))
            if len(knight_entries) < 4:
                has_knight_bundles = False

        if is_robber:
            lines = ["[PLAYABLE MOVES]"]
            lines.append(f"[PHASE: {getattr(current_prompt, "name", str(current_prompt))}]")
            if not moves:
                lines.append("  (no moves available)")
                return "\n".join(lines)
            groups: "OrderedDict[Any, dict]" = OrderedDict()
            for idx, move in enumerate(moves, start=1):
                coord, victim = _extract_coord_victim(move)
                tile_id = _tile_id_for_coordinate(public_state, coord)
                key = tile_id if tile_id is not None else coord
                if key not in groups:
                    groups[key] = {"coord": coord, "entries": []}
                groups[key]["entries"].append((idx, move, victim))
            for key in sorted(groups.keys(), key=lambda k: (k is None, str(k) if not isinstance(k, int) else k)):
                coord = groups[key]["coord"]
                header = _robber_tile_detail(public_state, coord)
                lines.append(f"{header}:")
                for idx, _mv, victim in groups[key]["entries"]:
                    if victim is None:
                        lines.append(f"  Action {idx}: no steal")
                    else:
                        lines.append(f"  Action {idx}: steal from {_name_of(victim)}")
            return "\n".join(lines)

        if has_knight_bundles:
            lines = ["[PLAYABLE MOVES]"]
            lines.append(f"[PHASE: {getattr(current_prompt, "name", str(current_prompt))}]")
            if not moves:
                lines.append("  (no moves available)")
                return "\n".join(lines)
            groups: "OrderedDict[Any, dict]" = OrderedDict()
            for idx, move in knight_entries:
                coord, victim = _extract_coord_victim(move)
                tile_id = _tile_id_for_coordinate(public_state, coord)
                key = tile_id if tile_id is not None else coord
                if key not in groups:
                    groups[key] = {"coord": coord, "entries": []}
                groups[key]["entries"].append((idx, move, victim))
            for key in sorted(groups.keys(), key=lambda k: (k is None, str(k) if not isinstance(k, int) else k)):
                coord = groups[key]["coord"]
                header = _robber_tile_detail(public_state, coord)
                lines.append(f"{header}:")
                for idx, _mv, victim in groups[key]["entries"]:
                    if victim is None:
                        lines.append(f"  Action {idx}: Play Knight -> no steal")
                    else:
                        lines.append(f"  Action {idx}: Play Knight -> steal from {_name_of(victim)}")
            for idx, mv in other_entries:
                lines.append(f"{idx}. {mv.label}")
            return "\n".join(lines)

    lines = ["[PLAYABLE MOVES]"]
    if current_prompt is not None:
        phase = getattr(current_prompt, "name", str(current_prompt))
        lines.append(f"[PHASE: {phase}]")
    if not moves:
        lines.append("  (no moves available)")
        return "\n".join(lines)
    for i, move in enumerate(moves, start=1):
        lines.append(f"{i}. {move.label}")
    return "\n".join(lines)


def format_playable_actions(playable_actions: Sequence[Action], observation=None) -> str:
    """Convenience wrapper: build moves for ``playable_actions`` and format them."""
    return format_moves(build_moves(playable_actions, observation), observation=observation)


def parse_move(response, moves: Sequence[Move]) -> Move:
    """Convert an LLM's response (a stable move index) into the chosen Move."""
    if isinstance(response, int):
        index = response
    else:
        match = re.match(r"\s*\[?(\d+)\]?", str(response))
        if match is None:
            raise ValueError(f"Cannot parse move index from response: {response!r}")
        index = int(match.group(1))

    if not 1 <= index <= len(moves):
        raise ValueError(f"Move index {index} out of range (1..{len(moves)})")
    return moves[index - 1]


def pick_auto_road(playable_actions: Sequence[Action], public_state=None) -> Optional[Action]:
    """Pick a legal BUILD_ROAD — public_state is required when scoring roads."""
    roads = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
    if not roads:
        return None
    if public_state is None:
        # No state to score by pips — return first road deterministically
        return roads[0]
    def score(action: Action) -> Tuple[int, Tuple]:
        edge = tuple(sorted(action.value))
        return _node_pip_total(public_state, edge[0]) + _node_pip_total(public_state, edge[1]), edge
    return max(roads, key=score)


__all__ = [
    "AUTO_ROAD",
    "Move",
    "_describe_node",
    "_is_buildable_node",
    "_node_buildability_detail",
    "_node_pip_total",
    "_player_longest_road_length",
    "_longest_road_suffix",
    "_tile_id_for_coordinate",
    "_robber_tile_detail",
    "_road_node_detail",
    "_road_node_detail_compact",
    "_coordinate_tile_label",
    "_own_network_nodes",
    "_land_edges_from",
    "_is_viable_initial_node",
    "_knight_robber_followups",
    "_knight_moves",
    "_road_building_moves",
    "_setup_settlement_moves",
    "_generate_discard_combos",
    "_discard_moves",
    "_label_action",
    "build_moves",
    "format_moves",
    "format_playable_actions",
    "parse_move",
    "pick_auto_road",
    "get_formatter",
    "REGISTRY",
    "BaseMoveFormatter",
]
