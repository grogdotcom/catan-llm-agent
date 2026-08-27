"""
Polymorphic move formatters — one concrete type per ActionType.

This module owns the polymorphic dispatch for move labeling and expansion.
``moves.py`` remains the thin orchestrator (build_moves / format_moves) and
re-exports this module's public surface so existing imports keep working.

Design:
    BaseMoveFormatter (ABC) — interface every move type implements.
    18 concrete formatters — one per ActionType (plus Fallback).
    REGISTRY — ActionType -> singleton formatter instance.
    get_formatter(action_type) — dispatch entry point.

Simple types inherit the default ``expand`` (single Move).
Compound types (Knight, RoadBuilding, Initial Settlement, BuildRoad) override
``expand`` to emit bundled Moves. Discard bulk is handled via a dedicated
``bulk_moves`` path called by the orchestrator before the per-action loop.
"""

from __future__ import annotations

import abc
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType, RESOURCES
from catanatron.models.public_state import PublicState

from catan_llm.format.board import format_starting_resources, get_adjacent_hex_info
from catan_llm.format.utils import (
    _abbr_resource,
    _format_coordinate,
    _format_maritime_trade_value,
    _format_resource_counts,
    _format_trade_offer_value,
    _name_of,
    get_pip_count,
)

# ---------------------------------------------------------------------------
# Core data types — canonical here, re-exported by moves.py
# ---------------------------------------------------------------------------

AUTO_ROAD = "AUTO_ROAD"
"""Fallback sentinel token in a Move's action list."""

@dataclass
class Move:
    """A single LLM-choosable move: one or more engine actions to execute."""

    label: str
    actions: List[Union[Action, str]]


# ---------------------------------------------------------------------------
# Detail helpers (verbatim from original moves.py — owned here now)
# ---------------------------------------------------------------------------

def _node_pip_total(public_state: PublicState, node_id: int) -> int:
    total = 0
    for tile_id in public_state.board.map.adjacent_tiles.get(node_id, ()):
        resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
        if resource is not None:
            total += get_pip_count(roll)
    return total


def _describe_node(public_state: PublicState, node_id: int) -> str:
    adjacent_hexes, port = get_adjacent_hex_info(public_state, node_id)
    parts: list[str] = []
    for hx in adjacent_hexes:
        abbr = _abbr_resource(hx.resource)
        if hx.roll is not None:
            parts.append(f"{hx.roll}-{abbr}")
        else:
            parts.append(abbr)
    hex_str = ", ".join(parts) if parts else "no tiles"
    total = sum(h.pips for h in adjacent_hexes)
    return f"Node {node_id} [{hex_str} | {total}p]"


def _is_buildable_node(public_state: PublicState, node_id: int) -> bool:
    if node_id in public_state.board.buildings:
        return False
    if node_id not in public_state.board.map.land_nodes:
        return False
    for neighbor in STATIC_GRAPH.neighbors(node_id):
        if neighbor in public_state.board.buildings:
            return False
    return True


def _node_buildability_detail(
    public_state: PublicState,
    node_id: int,
    extra_occupied: Optional[Set[int]] = None,
    extra_occupied_color: Any = None,
) -> tuple[bool, str]:
    if node_id not in public_state.board.map.land_nodes:
        return False, "blocked (water/non-land)"
    building = public_state.board.buildings.get(node_id)
    if building is not None:
        owner, btype = building
        btype_name = getattr(btype, "name", str(btype)).lower()
        return False, f"blocked (occupied by {_name_of(owner)} {btype_name} at Node {node_id})"
    if extra_occupied is not None and node_id in extra_occupied:
        owner_name = _name_of(extra_occupied_color) if extra_occupied_color is not None else "you"
        return False, f"blocked (will be occupied by {owner_name} settlement at Node {node_id})"
    for neighbor in STATIC_GRAPH.neighbors(node_id):
        nb_building = public_state.board.buildings.get(neighbor)
        if nb_building is not None:
            owner, btype = nb_building
            btype_name = getattr(btype, "name", str(btype)).lower()
            return False, f"blocked (too close to {_name_of(owner)} {btype_name} at Node {neighbor})"
        if extra_occupied is not None and neighbor in extra_occupied:
            owner_name = _name_of(extra_occupied_color) if extra_occupied_color is not None else "you"
            return False, f"blocked (too close to {owner_name} settlement at Node {neighbor})"
    return True, "available"


def _player_longest_road_length(
    public_state: PublicState,
    color: Any,
    extra_edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> int:
    if color is None:
        return 0
    friendly: Set[Tuple[int, int]] = set()
    for edge, owner in public_state.board.roads.items():
        if owner == color:
            friendly.add(tuple(sorted(edge)))
    if extra_edges:
        for e in extra_edges:
            friendly.add(tuple(sorted(e)))
    if not friendly:
        return 0
    enemy_nodes: Set[int] = {
        n for n, (owner, _) in public_state.board.buildings.items() if owner != color
    }
    nodes: Set[int] = set()
    for a, b in friendly:
        nodes.add(a)
        nodes.add(b)
    best = 0
    for start in nodes:
        if start in enemy_nodes:
            continue
        stack: List[Tuple[int, List[Tuple[int, int]]]] = [(start, [])]
        while stack:
            node, path = stack.pop()
            if node in enemy_nodes:
                best = max(best, len(path))
                continue
            expanded = False
            for nb in STATIC_GRAPH.neighbors(node):
                edge = tuple(sorted((node, nb)))
                if edge not in friendly:
                    continue
                if edge in path:
                    continue
                if nb in enemy_nodes:
                    continue
                stack.append((nb, path + [edge]))
                expanded = True
            if not expanded:
                best = max(best, len(path))
    return best


def _longest_road_suffix(
    public_state: PublicState,
    color: Any,
    extra_edges: Sequence[Tuple[int, int]],
) -> str:
    if color is None:
        return ""
    current = _player_longest_road_length(public_state, color, None)
    projected = _player_longest_road_length(public_state, color, extra_edges)
    delta = projected - current
    if delta == 0:
        base = f" | LR {current}->{projected}"
    elif delta > 0:
        base = f" | LR {current}->{projected}(+{delta})"
    else:
        base = f" | LR {current}->{projected}({delta})"
    try:
        global_len = getattr(public_state.board, "longest_road_length", 0) or 0
        holder = getattr(public_state.board, "longest_road_color", None)
    except Exception:
        global_len, holder = 0, None
    would_claim = False
    if projected >= 5 and projected > global_len:
        if holder is None or holder != color or projected > global_len:
            would_claim = True
    if would_claim:
        if holder == color:
            base += " [LR ext]"
        else:
            base += " [LR+2VP]"
    elif projected >= 5 and holder == color and projected == global_len:
        base += " [holds LR]"
    return base


def _tile_id_for_coordinate(public_state: PublicState, coordinate) -> Optional[int]:
    if coordinate is None:
        return None
    for tile_id, coord in public_state.board.map.tile_coordinates.items():
        if coord == coordinate:
            return tile_id
    return None


def _robber_tile_detail(public_state: PublicState, coordinate) -> str:
    if coordinate is None:
        return _coordinate_tile_label(public_state, coordinate)
    tile_id = _tile_id_for_coordinate(public_state, coordinate)
    if tile_id is None:
        return _format_coordinate(coordinate)
    resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
    if resource is None:
        tile_str = f"Tile {tile_id}: DESERT"
        pips = 0
    else:
        abbr = _abbr_resource(resource.name if hasattr(resource, 'name') else str(resource))
        pips = get_pip_count(roll)
        tile_str = f"Tile {tile_id}: {roll}-{abbr}({pips}p)"
    tiles_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for nid, tids in public_state.board.map.adjacent_tiles.items():
        for tid in tids:
            tiles_to_nodes[tid].append(nid)
    occupants: Dict[Any, List[Tuple[int, str, int]]] = defaultdict(list)
    for node_id in tiles_to_nodes.get(tile_id, []):
        building = public_state.board.buildings.get(node_id)
        if building is None:
            continue
        owner, btype = building
        btype_name = btype.name if hasattr(btype, 'name') else str(btype)
        multiplier = 2 if btype_name == "CITY" else 1
        blocked = pips * multiplier if resource is not None else 0
        occupants[owner].append((node_id, btype_name, blocked))
    if not occupants:
        return f"{tile_str} | no occupants"
    parts = []
    for owner in sorted(occupants.keys(), key=lambda c: getattr(c, "name", str(c))):
        color_name = _name_of(owner)
        hand_cards = public_state.players.get(owner)
        card_count = getattr(hand_cards, "hand_resource_count", "?") if hand_cards is not None else "?"
        nodes_desc = ", ".join(
            f"{btype.lower()}@N{nid}({bp}p)" for nid, btype, bp in sorted(occupants[owner])
        )
        parts.append(f"{color_name} {nodes_desc} {card_count}c")
    return f"{tile_str} | {'; '.join(parts)}"


def _road_node_detail(
    public_state: PublicState,
    edge: Tuple[int, int],
    exclude_nodes: Optional[Set[int]] = None,
    network_nodes: Optional[Set[int]] = None,
    extra_occupied: Optional[Set[int]] = None,
    extra_occupied_color: Any = None,
) -> str:
    a, b = tuple(sorted(edge))
    exclude = set(exclude_nodes or [])
    if network_nodes is not None:
        new_tips = [n for n in (a, b) if n not in network_nodes and n not in exclude]
        if not new_tips:
            new_tips = [n for n in (a, b) if n not in exclude]
            if not new_tips:
                return ""
        new_tips = sorted(new_tips)
        if len(new_tips) > 1:
            new_tips = new_tips[:1]
    else:
        candidates = [n for n in (a, b) if n not in exclude]
        if not candidates:
            return ""
        new_tips = sorted(candidates)[:1]
    tip = new_tips[0]
    tip_occupied = (public_state.board.buildings.get(tip) is not None) or (
        extra_occupied is not None and tip in extra_occupied
    )
    forward: List[int] = []
    if not tip_occupied:
        for nb in STATIC_GRAPH.neighbors(tip):
            if nb in (a, b):
                continue
            if nb in exclude:
                continue
            forward.append(nb)
        forward = sorted(forward)
    targets: List[int] = forward if forward else [tip]
    parts: List[str] = []
    for nid in targets:
        desc = _describe_node(public_state, nid)
        ok, _rsn = _node_buildability_detail(
            public_state, nid, extra_occupied=extra_occupied, extra_occupied_color=extra_occupied_color
        )
        mark = "✓" if ok else "✗"
        parts.append(f"{desc}{mark}")
    return " -> Targets: " + ", ".join(parts)


def _road_node_detail_compact(
    public_state: PublicState,
    edge: Tuple[int, int],
    exclude_nodes: Optional[Set[int]] = None,
    network_nodes: Optional[Set[int]] = None,
    extra_occupied: Optional[Set[int]] = None,
) -> str:
    a, b = tuple(sorted(edge))
    exclude = set(exclude_nodes or [])
    if network_nodes is not None:
        new_tips = [n for n in (a, b) if n not in network_nodes and n not in exclude]
        if not new_tips:
            new_tips = [n for n in (a, b) if n not in exclude]
            if not new_tips:
                return ""
        new_tips = sorted(new_tips)
        if len(new_tips) > 1:
            new_tips = new_tips[:1]
    else:
        candidates = [n for n in (a, b) if n not in exclude]
        if not candidates:
            return ""
        new_tips = sorted(candidates)[:1]
    tip = new_tips[0]
    tip_occupied = (public_state.board.buildings.get(tip) is not None) or (
        extra_occupied is not None and tip in extra_occupied
    )
    forward: List[int] = []
    if not tip_occupied:
        for nb in STATIC_GRAPH.neighbors(tip):
            if nb in (a, b):
                continue
            if nb in exclude:
                continue
            forward.append(nb)
        forward = sorted(forward)
    targets: List[int] = forward if forward else [tip]
    parts: List[str] = []
    for nid in targets:
        pips = _node_pip_total(public_state, nid)
        parts.append(f"N{nid} ({pips}p)")
    return " -> Target " + ", ".join(parts)


def _coordinate_tile_label(public_state: PublicState, coordinate) -> str:
    if coordinate is None:
        return _format_coordinate(coordinate)
    for tile_id, coord in public_state.board.map.tile_coordinates.items():
        if coord == coordinate:
            return f"Tile {tile_id}"
    return _format_coordinate(coordinate)


def _own_network_nodes(public_state: PublicState, color) -> Set[int]:
    board = public_state.board
    own_buildings = {n for n, (owner, _) in board.buildings.items() if owner == color}
    enemy_buildings = {n for n, (owner, _) in board.buildings.items() if owner != color}
    endpoints = set()
    for edge, owner in board.roads.items():
        if owner == color:
            endpoints.update(edge)
    return (own_buildings | endpoints) - enemy_buildings


def _land_edges_from(public_state: PublicState, color, nodes) -> List[Tuple[int, int]]:
    land = public_state.board.map.land_nodes
    owned = set(public_state.board.roads.keys())
    nodes = set(nodes) - {
        n for n, (owner, _) in public_state.board.buildings.items() if owner != color
    }
    edges = set()
    for node in nodes:
        for neighbor in STATIC_GRAPH.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            if edge in owned or node not in land or neighbor not in land:
                continue
            edges.add(edge)
    return sorted(edges)


def _is_viable_initial_node(public_state: PublicState, node_id: int) -> bool:
    return True


# ---------------------------------------------------------------------------
# Shared helpers for compound moves (owned here, used by formatters)
# ---------------------------------------------------------------------------

def _knight_robber_followups(public_state: PublicState, color) -> List[Tuple]:
    map_data = public_state.board.map
    robber_coordinate = map_data.tile_coordinates.get(public_state.board.robber_tile_id)
    tiles_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_id, tile_ids in map_data.adjacent_tiles.items():
        for tile_id in tile_ids:
            tiles_to_nodes[tile_id].append(node_id)
    targets = []
    for tile_id in sorted(map_data.tile_coordinates):
        coordinate = map_data.tile_coordinates[tile_id]
        if coordinate == robber_coordinate:
            continue
        victims = set()
        for node_id in tiles_to_nodes.get(tile_id, ()):
            building = public_state.board.buildings.get(node_id)
            if building is None:
                continue
            owner, _ = building
            if owner != color and public_state.players[owner].hand_resource_count >= 1:
                victims.add(owner)
        if victims:
            for victim in sorted(victims, key=lambda c: getattr(c, "name", str(c))):
                targets.append((coordinate, victim))
        else:
            targets.append((coordinate, None))
    return targets


def _knight_moves(knight_action: Action, public_state: PublicState) -> List[Move]:
    color = knight_action.color
    moves = []
    for coordinate, victim in _knight_robber_followups(public_state, color):
        followup = Action(color, ActionType.MOVE_ROBBER, (coordinate, victim))
        tile_detail = _robber_tile_detail(public_state, coordinate)
        if victim is None:
            label = f"Play Knight -> move robber to {tile_detail} (no steal)"
        else:
            label = f"Play Knight -> move robber to {tile_detail} and steal from {_name_of(victim)}"
        moves.append(Move(label=label, actions=[knight_action, followup]))
    return moves


def _road_building_moves(play_card: Action, public_state: PublicState) -> List[Move]:
    color = play_card.color
    base_network = _own_network_nodes(public_state, color)
    first_roads = _land_edges_from(public_state, color, base_network)
    moves = []
    seen_pairs = set()
    for first in first_roads:
        second_network = base_network | set(first)
        seconds = [
            e
            for e in _land_edges_from(public_state, color, second_network)
            if e != first
        ]
        if not seconds:
            detail = _road_node_detail(public_state, first, network_nodes=base_network)
            longest = _longest_road_suffix(public_state, color, [first])
            label = f"Play Road Building -> build road {first}{detail}{longest}"
            moves.append(
                Move(
                    label=label,
                    actions=[play_card, Action(color, ActionType.BUILD_ROAD, first)],
                )
            )
        else:
            for second in seconds:
                pair = frozenset((first, second))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                road_a, road_b = sorted((first, second))
                detail_a = _road_node_detail(public_state, first, network_nodes=base_network)
                second_network_for_detail = base_network | set(first)
                detail_b = _road_node_detail(public_state, second, network_nodes=second_network_for_detail)
                longest = _longest_road_suffix(public_state, color, [first, second])
                label = f"Play RB roads {road_a} and {road_b} | Road {first}{detail_a} | Road {second}{detail_b}{longest}"
                moves.append(
                    Move(
                        label=label,
                        actions=[
                            play_card,
                            Action(color, ActionType.BUILD_ROAD, first),
                            Action(color, ActionType.BUILD_ROAD, second),
                        ],
                    )
                )
    return moves


def _setup_settlement_moves(settle: Action, public_state: PublicState) -> List[Move]:
    color = settle.color
    node = settle.value
    road_options = _land_edges_from(public_state, color, {node})
    existing = sum(1 for _, (c, _) in public_state.board.buildings.items() if c == color)
    is_second = existing == 1
    starting_suffix = ""
    if is_second:
        try:
            sr = format_starting_resources(public_state, node)
            starting_suffix = f" → Starting resources: {sr}"
        except Exception:
            starting_suffix = ""
    if not road_options:
        from catan_llm.format.move_formatters import get_formatter  # lazy to avoid issues at import time

        base_label = get_formatter(settle.action_type).label(settle, public_state)
        if is_second and "Starting resources" not in base_label:
            base_label += starting_suffix
        return [Move(label=base_label, actions=[settle])]
    settle_desc = _describe_node(public_state, node)
    settle_desc_with_resources = f"{settle_desc}{starting_suffix}"
    return [
        Move(
            label=f"Settlement {settle_desc_with_resources} | Road {edge}{_road_node_detail_compact(public_state, edge, exclude_nodes={node}, network_nodes={node}, extra_occupied={node})}",
            actions=[settle, Action(color, ActionType.BUILD_ROAD, edge)],
        )
        for edge in road_options
    ]


def _generate_discard_combos(hand: Dict[str, int], k: int) -> List[Tuple[str, ...]]:
    order = [r for r in RESOURCES if hand.get(r, 0) > 0]
    combos: List[Tuple[str, ...]] = []

    def rec(idx: int, remaining: int, chosen: Dict[str, int]):
        if remaining == 0:
            combo: List[str] = []
            for r in order:
                combo.extend([r] * chosen.get(r, 0))
            combos.append(tuple(combo))
            return
        if idx >= len(order):
            return
        max_possible = sum(hand[r] - chosen.get(r, 0) for r in order[idx:])
        if max_possible < remaining:
            return
        r = order[idx]
        max_take = min(hand[r] - chosen.get(r, 0), remaining)
        for take in range(max_take + 1):
            nxt = dict(chosen)
            if take:
                nxt[r] = take
            rec(idx + 1, remaining - take, nxt)

    rec(0, k, {})
    combos.sort()
    return combos


def _discard_moves(color, inventory, k: int, public_state: PublicState) -> List[Move]:
    if inventory is None or k <= 0:
        return []
    hand = {
        "WOOD": getattr(inventory, "wood", 0),
        "BRICK": getattr(inventory, "brick", 0),
        "SHEEP": getattr(inventory, "sheep", 0),
        "WHEAT": getattr(inventory, "wheat", 0),
        "ORE": getattr(inventory, "ore", 0),
    }
    hand = {r: c for r, c in hand.items() if c > 0}
    if not hand or k <= 0:
        return []
    combos = _generate_discard_combos(hand, k)
    if not combos:
        return []
    moves: List[Move] = []
    for combo in combos:
        actions = [Action(color, ActionType.DISCARD_RESOURCE, r) for r in combo]
        counts = Counter(combo)
        summary = ", ".join(f"{_abbr_resource(r)}:{counts[r]}" for r in RESOURCES if r in counts)
        resources_str = ", ".join(_abbr_resource(r) for r in combo)
        label = f"Discard {resources_str} ({summary} → {k}c)"
        moves.append(Move(label=label, actions=actions))
    return moves


def _describe_roll_resources(public_state: PublicState, dice_total: int) -> str:
    """Concise resource-collection summary for a dice total (history view)."""
    if dice_total == 7:
        return ""
    robber_tile_id = getattr(public_state.board, "robber_tile_id", None)
    if robber_tile_id is None:
        robber_tile_id = getattr(public_state.board, "robber_tile", None)
    gains: dict[str, List[str]] = defaultdict(list)
    blocked_gains: dict[str, List[str]] = defaultdict(list)
    blocked_pips: dict[str, int] = defaultdict(int)
    tiles = getattr(public_state.board.map, "tiles", {})
    adjacent_tiles = getattr(public_state.board.map, "adjacent_tiles", {})
    buildings = getattr(public_state.board, "buildings", {})
    robber_resource = None
    robber_roll = None
    robber_pips = 0
    if robber_tile_id is not None and robber_tile_id in tiles:
        robber_resource, robber_roll = tiles[robber_tile_id]
        robber_pips = get_pip_count(robber_roll)
    for tile_id, (resource, roll) in tiles.items():
        if roll != dice_total:
            continue
        if resource is None:
            continue
        resource_name = resource.name if hasattr(resource, "name") else str(resource)
        is_robber = tile_id == robber_tile_id
        for node_id, tids in adjacent_tiles.items():
            if tile_id not in tids:
                continue
            b = buildings.get(node_id)
            if b is None:
                continue
            owner, btype = b
            is_city = str(btype) == "CITY" or (hasattr(btype, "name") and btype.name == "CITY")
            owner_name = _name_of(owner)
            count = 2 if is_city else 1
            if is_robber:
                for _ in range(count):
                    blocked_gains[owner_name].append(resource_name)
                blocked_pips[owner_name] += robber_pips * count
            else:
                for _ in range(count):
                    gains[owner_name].append(resource_name)
    parts = []
    for owner in sorted(gains.keys()):
        cnt = Counter(gains[owner])
        ordered = [r for r in RESOURCES if r in cnt]
        inner = ", ".join(f"{cnt[r]} {_abbr_resource(r)}" for r in ordered)
        parts.append(f"{owner} + [{inner}]")
    blocked_parts = []
    for owner in sorted(blocked_gains.keys()):
        cnt = Counter(blocked_gains[owner])
        ordered = [r for r in RESOURCES if r in cnt]
        inner = ", ".join(f"{cnt[r]} {_abbr_resource(r)}" for r in ordered)
        blocked_parts.append(f"{owner} [{inner}]")
    if gains and blocked_parts:
        return " | " + ", ".join(parts) + " | blk " + ", ".join(blocked_parts)
    if gains:
        return " | " + ", ".join(parts)
    if blocked_parts:
        return " | blk " + ", ".join(blocked_parts)
    return " | no resources"


# ---------------------------------------------------------------------------
# Polymorphic hierarchy — one type per ActionType for BOTH move labels
# and public-history descriptions (reused). The same REGISTRY is used by
# moves.py (label/expand) and history.py (describe).
# ---------------------------------------------------------------------------

class BaseMoveFormatter(abc.ABC):
    """Abstract formatter for a single ActionType — owns BOTH move and history.

    Each concrete type implements:
      * label(action, public_state)  — LLM move list
      * describe(record, public_state) — public history line
    ``expand`` / ``bulk_moves`` are move-only; history uses only ``describe``.
    """

    @property
    @abc.abstractmethod
    def action_type(self) -> ActionType:
        """The ActionType this formatter owns."""

    @abc.abstractmethod
    def label(self, action: Action, public_state=None) -> str:
        """Human-readable label — simple types ignore state, enriched types require it."""

    def describe(self, record: ActionRecord, public_state=None) -> str:
        """Human-readable history line — enriched types require public_state."""
        action = record.action
        color = _name_of(action.color)
        return f"{color} {action.action_type.name}: value={action.value!r}, result={record.result!r}"

    def expand(self, action: Action, observation=None) -> List[Move]:
        """Expand — observation with public_state required for enriched types."""
        public_state = getattr(observation, "public_state", None) if observation is not None else None
        return [Move(label=self.label(action, public_state), actions=[action])]

    def bulk_moves(
        self, playable_actions: Sequence[Action], observation=None
    ) -> Optional[List[Move]]:
        """If this formatter can expand a *whole* playable list, return Moves."""
        return None


# -- Simple formatters (label only) ----------------------------------------

class RollFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.ROLL

    def label(self, action: Action, public_state: PublicState) -> str:
        return "Roll the dice"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        action = record.action
        color = _name_of(action.color)
        value = action.value
        result = record.result
        dice = result if result is not None else value
        if dice is not None and len(dice) == 2:
            total = dice[0] + dice[1]
            base = f"{color} rolled {dice[0]}+{dice[1]} = {total}"
            base += _describe_roll_resources(public_state, total)
            return base
        return f"{color} rolled"


class EndTurnFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.END_TURN

    def label(self, action: Action, public_state: PublicState) -> str:
        return "End turn"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        return f"{_name_of(record.action.color)} ended turn"


class BuyDevelopmentCardFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.BUY_DEVELOPMENT_CARD

    def label(self, action: Action, public_state: PublicState) -> str:
        return "Buy a development card"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        card = record.result if record.result is not None else record.action.value
        if card is None:
            return f"{color} bought a development card"
        return f"{color} bought development card: {_name_of(card)}"


class PlayYearOfPlentyFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.PLAY_YEAR_OF_PLENTY

    def label(self, action: Action, public_state: PublicState) -> str:
        cards = ", ".join(_name_of(r) for r in action.value)
        return f"Play Year of Plenty: take {cards}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        if value is None:
            return f"{color} played YOP"
        cards = ", ".join(_abbr_resource(_name_of(r)) for r in value)
        return f"{color} played YOP: took {cards}"


class PlayMonopolyFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.PLAY_MONOPOLY

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Play Monopoly: steal all {_name_of(action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        result = record.result
        raw_res = _name_of(value) if value is not None else ""
        abbr = _abbr_resource(raw_res).upper() if raw_res else ""
        if not abbr and value is not None:
            abbr = str(value).upper()
        base = f"{color} played MONOPOLY on {abbr}" if abbr else f"{color} played MONOPOLY"
        if result is not None:
            try:
                if isinstance(result, tuple) and len(result) == 3:
                    res_resource, stolen_tuple, total = result
                    res_abbr = (
                        _abbr_resource(_name_of(res_resource)).upper()
                        if res_resource
                        else abbr
                    )
                    if not res_abbr:
                        res_abbr = abbr
                    if isinstance(stolen_tuple, (list, tuple)):
                        parts = []
                        for c, cnt in stolen_tuple:
                            if cnt and cnt > 0:
                                parts.append(f"{_name_of(c)} - {cnt} {res_abbr}")
                        if parts:
                            base += f" | {', '.join(parts)} (total {total})"
                        elif total == 0:
                            base += " | stole nothing (total 0)"
                        else:
                            base += f" (total {total})"
                        return base
                if isinstance(result, dict) and "stolen" in result:
                    stolen = result["stolen"]
                    total = result.get("total", sum(v for v in stolen.values() if isinstance(v, int)))
                    res_abbr = abbr
                    if "resource" in result and result["resource"]:
                        res_abbr = _abbr_resource(_name_of(result["resource"])).upper()
                    parts = [
                        f"{_name_of(c)} - {cnt} {res_abbr}"
                        for c, cnt in sorted(stolen.items(), key=lambda kv: _name_of(kv[0]))
                        if cnt and cnt > 0
                    ]
                    if parts:
                        base += f" | {', '.join(parts)} (total {total})"
                    else:
                        base += f" | stole nothing (total {total})"
                    return base
                if isinstance(result, int):
                    base += f" (total {result})"
                    return base
            except Exception:
                pass
        return base


class MaritimeTradeFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.MARITIME_TRADE

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Maritime trade: {_format_maritime_trade_value(action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        if value is None:
            return f"{color} maritime traded"
        return f"{color} maritime trade: {_format_maritime_trade_value(value)}"


class OfferTradeFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.OFFER_TRADE

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Offer trade: {_format_trade_offer_value(action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        if value is None:
            return f"{color} offered a trade"
        return f"{color} {_format_trade_offer_value(value)}"


class AcceptTradeFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.ACCEPT_TRADE

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Accept trade: {_format_trade_offer_value(action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        if value is None:
            return f"{color} accepted a trade"
        return f"{color} accepted trade: {_format_trade_offer_value(value)}"


class RejectTradeFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.REJECT_TRADE

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Reject trade: {_format_trade_offer_value(action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        if value is None:
            return f"{color} rejected a trade"
        return f"{color} rejected trade: {_format_trade_offer_value(value)}"


class ConfirmTradeFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.CONFIRM_TRADE

    def label(self, action: Action, public_state: PublicState) -> str:
        trade_part = _format_trade_offer_value(action.value[:10])
        acceptor = _name_of(action.value[10]) if len(action.value) > 10 else "unknown"
        return f"Confirm trade with {acceptor}: {trade_part}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        if value is None:
            return f"{color} traded"
        acceptor = _name_of(value[10]) if len(value) > 10 else "unknown"
        try:
            offered = _format_resource_counts(value[:5])
            asking = _format_resource_counts(value[5:10])
            return f"{color} gave {acceptor} [{offered}] for [{asking}]"
        except Exception:
            trade_part = _format_trade_offer_value(value[:10])
            return f"{color} gave {acceptor} {trade_part}"


class CancelTradeFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.CANCEL_TRADE

    def label(self, action: Action, public_state: PublicState) -> str:
        return "Cancel trade"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        return f"{_name_of(record.action.color)} cancelled trade"


class BuildCityFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.BUILD_CITY

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"City {_describe_node(public_state, action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        node_desc = _describe_node(public_state, value)
        return f"{color} built C {node_desc}"


class DiscardResourceFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.DISCARD_RESOURCE

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Discard one {_name_of(action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        discarded = record.result if record.result is not None else record.action.value
        return f"{color} discarded {_abbr_resource(_name_of(discarded))}"

    def bulk_moves(
        self, playable_actions: Sequence[Action], observation
    ) -> Optional[List[Move]]:
        public_state: PublicState = observation.public_state
        current_prompt = observation.current_prompt
        if current_prompt != ActionPrompt.DISCARD:
            return None
        discard_actions = [a for a in playable_actions if a.action_type == ActionType.DISCARD_RESOURCE]
        if not discard_actions:
            return None
        inventory = observation.inventory  # type: ignore[attr-defined]
        hand_total = sum(getattr(inventory, r.lower(), 0) for r in RESOURCES)
        k = hand_total // 2 if hand_total > 7 else 0
        if k > 1:
            color = discard_actions[0].color
            bundled = _discard_moves(color, inventory, k, public_state)
            if bundled:
                moves: List[Move] = list(bundled)
                for a in playable_actions:
                    if a.action_type != ActionType.DISCARD_RESOURCE:
                        moves.append(Move(label=get_formatter(a.action_type).label(a, public_state), actions=[a]))
                return moves
        return None


# -- Compound / context-sensitive formatters --------------------------------

class BuildRoadFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.BUILD_ROAD

    def label(self, action: Action, public_state: PublicState) -> str:
        edge = tuple(sorted(action.value))
        color = getattr(action, "color", None)
        network = _own_network_nodes(public_state, color)
        detail = _road_node_detail(public_state, edge, network_nodes=network)
        longest = _longest_road_suffix(public_state, color, [edge])
        return f"Road {edge}{detail}{longest}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        edge = tuple(sorted(value)) if value is not None else value
        return f"{color} built road {edge}"


class BuildSettlementFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.BUILD_SETTLEMENT

    def label(self, action: Action, public_state: PublicState) -> str:
        return f"Settlement {_describe_node(public_state, action.value)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        node_desc = _describe_node(public_state, value)
        return f"{color} built S {node_desc}"

    def expand(self, action: Action, observation) -> List[Move]:
        public_state: PublicState = observation.public_state
        current_prompt = observation.current_prompt
        if current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            return _setup_settlement_moves(action, public_state)
        return super().expand(action, observation)


class MoveRobberFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.MOVE_ROBBER

    def label(self, action: Action, public_state: PublicState) -> str:
        coordinate, victim = action.value
        tile_detail = _robber_tile_detail(public_state, coordinate)
        if victim is None:
            return f"Move robber to {tile_detail} (no steal)"
        return f"Move robber to {tile_detail} and steal from {_name_of(victim)}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        value = record.action.value
        result = record.result
        coordinate = None
        victim = None
        if value is not None:
            coordinate, victim = value[0], value[1]
        tile_detail = None
        if coordinate is not None:
            try:
                tile_id = None
                for tid, coord in public_state.board.map.tile_coordinates.items():
                    if coord == coordinate:
                        tile_id = tid
                        break
                if tile_id is not None:
                    resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
                    if resource is None:
                        tile_detail = f"Tile {tile_id}: DESERT"
                    else:
                        res_name = resource.name if hasattr(resource, "name") else str(resource)
                        abbr = _abbr_resource(res_name)
                        pips = get_pip_count(roll)
                        tile_detail = f"Tile {tile_id}: {roll}-{abbr}({pips}p)"
                if tile_detail is None:
                    tile_detail = _format_coordinate(coordinate)
            except Exception:
                tile_detail = None
        if tile_detail is not None:
            coord_str = tile_detail
        else:
            if coordinate is not None:
                try:
                    coord_str = _format_coordinate(coordinate)
                except Exception:
                    coord_str = str(coordinate)
            else:
                coord_str = "unknown"
        if victim is None:
            return f"{color} moved robber to {coord_str} (no steal)"
        victim_name = _name_of(victim)
        if result is None:
            return f"{color} moved robber to {coord_str} and stole from {victim_name} (card hidden)"
        return f"{color} moved robber to {coord_str} and stole {_name_of(result)} from {victim_name}"


class PlayKnightFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.PLAY_KNIGHT_CARD

    def label(self, action: Action, public_state: PublicState) -> str:
        return "Play Knight (then move the robber)"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        return f"{_name_of(record.action.color)} played Knight"

    def expand(self, action: Action, observation) -> List[Move]:
        public_state: PublicState = observation.public_state
        return _knight_moves(action, public_state)


class PlayRoadBuildingFormatter(BaseMoveFormatter):
    @property
    def action_type(self) -> ActionType:
        return ActionType.PLAY_ROAD_BUILDING

    def label(self, action: Action, public_state: PublicState) -> str:
        return "Play Road Building (then build two roads)"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        return f"{_name_of(record.action.color)} played Road Building"

    def expand(self, action: Action, observation) -> List[Move]:
        public_state: PublicState = observation.public_state
        moves = _road_building_moves(action, public_state)
        # In practice at least one road is always legal when the card is playable.
        assert moves, "Road Building should have at least one legal road when played"
        return moves


class FallbackFormatter(BaseMoveFormatter):
    """Handles unknown ActionTypes gracefully."""

    @property
    def action_type(self):  # type: ignore[override]
        return None  # type: ignore[return-value]

    def label(self, action: Action, public_state: PublicState) -> str:
        name = getattr(action.action_type, "name", str(action.action_type))
        return f"{name}: value={action.value!r}"

    def describe(self, record: ActionRecord, public_state: PublicState) -> str:
        color = _name_of(record.action.color)
        return f"{color} {record.action.action_type.name}: value={record.action.value!r}, result={record.result!r}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FALLBACK = FallbackFormatter()

REGISTRY: Dict[ActionType, BaseMoveFormatter] = {
    ActionType.ROLL: RollFormatter(),
    ActionType.MOVE_ROBBER: MoveRobberFormatter(),
    ActionType.DISCARD_RESOURCE: DiscardResourceFormatter(),
    ActionType.BUILD_ROAD: BuildRoadFormatter(),
    ActionType.BUILD_SETTLEMENT: BuildSettlementFormatter(),
    ActionType.BUILD_CITY: BuildCityFormatter(),
    ActionType.BUY_DEVELOPMENT_CARD: BuyDevelopmentCardFormatter(),
    ActionType.PLAY_KNIGHT_CARD: PlayKnightFormatter(),
    ActionType.PLAY_YEAR_OF_PLENTY: PlayYearOfPlentyFormatter(),
    ActionType.PLAY_MONOPOLY: PlayMonopolyFormatter(),
    ActionType.PLAY_ROAD_BUILDING: PlayRoadBuildingFormatter(),
    ActionType.MARITIME_TRADE: MaritimeTradeFormatter(),
    ActionType.OFFER_TRADE: OfferTradeFormatter(),
    ActionType.ACCEPT_TRADE: AcceptTradeFormatter(),
    ActionType.REJECT_TRADE: RejectTradeFormatter(),
    ActionType.CONFIRM_TRADE: ConfirmTradeFormatter(),
    ActionType.CANCEL_TRADE: CancelTradeFormatter(),
    ActionType.END_TURN: EndTurnFormatter(),
}


def get_formatter(action_type) -> BaseMoveFormatter:
    """Return the formatter for *action_type*, or the fallback."""
    return REGISTRY.get(action_type, _FALLBACK)


__all__ = [
    "AUTO_ROAD",
    "Move",
    "_node_pip_total",
    "_describe_node",
    "_is_buildable_node",
    "_node_buildability_detail",
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
    "BaseMoveFormatter",
    "RollFormatter",
    "EndTurnFormatter",
    "BuyDevelopmentCardFormatter",
    "PlayYearOfPlentyFormatter",
    "PlayMonopolyFormatter",
    "MaritimeTradeFormatter",
    "OfferTradeFormatter",
    "AcceptTradeFormatter",
    "RejectTradeFormatter",
    "ConfirmTradeFormatter",
    "CancelTradeFormatter",
    "BuildCityFormatter",
    "DiscardResourceFormatter",
    "BuildRoadFormatter",
    "BuildSettlementFormatter",
    "MoveRobberFormatter",
    "PlayKnightFormatter",
    "PlayRoadBuildingFormatter",
    "FallbackFormatter",
    "REGISTRY",
    "get_formatter",
]
