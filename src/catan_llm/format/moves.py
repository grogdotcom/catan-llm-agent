"""
Move formatting and compound-move planning.

The LLM-facing view of a decision is a numbered list of "moves". A move is a
sequence of engine actions that together form one coherent decision: the LLM
picks a single move (by its stable index number) and the agent then drives
every engine prompt of that move without re-consulting the LLM.

Roads are represented as sorted node pairs ``(n1, n2)`` (matching the board
occupancy data). Legal road edges are derived from the player's current roads
and settlements against the map's static graph, so compound moves that include
road placements (initial settlement + road, Road Building + two roads) bundle
the concrete edges the LLM wants instead of auto-completing them.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import Action, ActionPrompt, ActionType
from catanatron.models.public_state import PublicState

from catan_llm.format.board import get_adjacent_hex_info
from catan_llm.format.utils import _format_maritime_trade_value, _format_trade_offer_value, _name_of, get_pip_count, _format_coordinate

AUTO_ROAD = "AUTO_ROAD"
"""Fallback sentinel token in a Move's action list.

Means: "resolve a legal BUILD_ROAD from the current prompt's playable actions".
Only used when no public state is available to enumerate the concrete road
edges upfront. With a public state, initial-placement roads and the Road
Building card's two roads are bundled concretely instead (each as a sorted
``(n1, n2)`` node pair), so the LLM picks the exact roads it wants.
"""


@dataclass
class Move:
    """A single LLM-choosable move: one or more engine actions to execute.

    ``actions`` lists the exact engine Actions to return in order; a string
    entry is a sentinel (see ``AUTO_ROAD``) resolved from the live prompt's
    playable actions at execution time. The first entry is returned immediately
    and the rest are queued, so compound moves (Knight + robber move, Road
    Building + two roads, initial settlement + road) are decided once.
    """

    label: str
    actions: List[Union[Action, str]]


def _node_pip_total(public_state: Optional[PublicState], node_id: int) -> int:
    """Total pips of the resource tiles touching a node (0 for desert/sea)."""
    if public_state is None:
        return 0
    total = 0
    for tile_id in public_state.board.map.adjacent_tiles.get(node_id, ()):
        resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
        if resource is not None:
            total += get_pip_count(roll)
    return total


def _describe_node(public_state: Optional[PublicState], node_id: int) -> str:
    """Concise node description with adjacent tiles, port and total pips.

    Mirrors ``_format_building_string`` / ``get_adjacent_hex_info`` so the
    tile strings are identical to board-occupancy formatting. Returns e.g.
    ``Node 5: (Tile 0: 11 SHEEP (2 pips), Tile 4: 5 WHEAT (4 pips)) Port: 3:1 Total: 6 pips``.
    """
    if public_state is None:
        return f"Node {node_id}"
    adjacent_hexes, port = get_adjacent_hex_info(public_state, node_id)
    hex_info_list = []
    for hx in adjacent_hexes:
        hex_info_list.append(f"(Tile {hx.tile_id}: {hx.roll} {hx.resource} ({hx.pips} pips))")
    hex_str = ", ".join(hex_info_list) if hex_info_list else "(no resource tiles)"
    port_str = f" Port: {port}" if port else ""
    total = sum(h.pips for h in adjacent_hexes)
    return f"Node {node_id}: {hex_str}{port_str} Total: {total} pips"


def _is_buildable_node(public_state: Optional[PublicState], node_id: int) -> bool:
    """Whether a settlement could be placed at ``node_id`` (empty + distance rule).

    Mirrors the engine's ``Board.board_buildable_ids`` distance rule without
    requiring the full State: node must be unoccupied, be a land node, and
    have no neighboring building.
    """
    if public_state is None:
        return False
    if node_id in public_state.board.buildings:
        return False
    if node_id not in public_state.board.map.land_nodes:
        return False
    # Distance rule: no adjacent node may have a building
    for neighbor in STATIC_GRAPH.neighbors(node_id):
        if neighbor in public_state.board.buildings:
            return False
    return True


def _node_buildability_detail(
    public_state: Optional[PublicState],
    node_id: int,
    extra_occupied: Optional[Set[int]] = None,
    extra_occupied_color: Any = None,
) -> tuple[bool, str]:
    """Return (is_buildable, reason) for settlement placement at node.

    Mirrors _is_buildable_node but provides human reason for blocked status.
    ``extra_occupied`` models buildings that will exist after the current move
    (e.g., the settlement being placed at node 0), so immediate neighbours are
    correctly reported as ``too close``.
    """
    if public_state is None:
        return False, "unknown"
    if node_id not in public_state.board.map.land_nodes:
        return False, "blocked (water/non-land)"
    building = public_state.board.buildings.get(node_id)
    if building is not None:
        owner, btype = building
        btype_name = getattr(btype, "name", str(btype)).lower()
        return False, f"blocked (occupied by {_name_of(owner)} {btype_name} at Node {node_id})"
    if extra_occupied is not None and node_id in extra_occupied:
        # The node itself will be occupied by the pending settlement
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
    public_state: Optional[PublicState],
    color: Any,
    extra_edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> int:
    """Projected longest road length for ``color`` after optionally adding ``extra_edges``.

    Mirrors ``Board.longest_acyclic_path`` over the friendly road graph, blocking
    traversal through enemy settlements/cities (same rule the engine uses). The
    result is the number of road segments in the longest simple path, matching
    ``Board.road_lengths[color]`` and the 5-road threshold for Longest Road.

    Args:
        public_state: Public board snapshot.
        color: Player to evaluate.
        extra_edges: Optional road edges to add for projection (each as
            ``(n1, n2)`` unsorted). Both directions are treated as owned.

    Returns:
        Length as number of edges (0 if no roads).
    """
    if public_state is None or color is None:
        return 0
    # Friendly edge set (normalized)
    friendly: Set[Tuple[int, int]] = set()
    for edge, owner in public_state.board.roads.items():
        if owner == color:
            friendly.add(tuple(sorted(edge)))
    if extra_edges:
        for e in extra_edges:
            friendly.add(tuple(sorted(e)))
    if not friendly:
        return 0
    # Enemy-occupied nodes block traversal
    enemy_nodes: Set[int] = {
        n for n, (owner, _) in public_state.board.buildings.items() if owner != color
    }
    # Nodes incident to friendly roads
    nodes: Set[int] = set()
    for a, b in friendly:
        nodes.add(a)
        nodes.add(b)

    best = 0
    # Iterative DFS per start node, tracking edge path only (nodes may be revisited
    # via different edges, matching the engine's edge-based visited set).
    for start in nodes:
        if start in enemy_nodes:
            continue
        stack: List[Tuple[int, List[Tuple[int, int]]]] = [(start, [])]
        # To avoid exponential blowup on dense boards, cap visited path length?
        # Longest road in Catan rarely exceeds 15, so exhaustive DFS is fine.
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
                    # Engine skips edge into enemy settlement entirely
                    continue
                stack.append((nb, path + [edge]))
                expanded = True
            if not expanded:
                best = max(best, len(path))
    return best


def _longest_road_suffix(
    public_state: Optional[PublicState],
    color: Any,
    extra_edges: Sequence[Tuple[int, int]],
) -> str:
    """Human suffix describing longest-road change for a road build.

    Shows ``current -> projected (+delta)`` and whether the move would
    claim/extend Longest Road (≥5 and beats the global holder).

    Returns:
        Suffix like ``" | Longest road: 2 -> 4 (+2)"`` or
        ``" | Longest road: 4 -> 6 (+2, would claim Longest Road, +2 VP)"``.
        Empty string if state/color unavailable.
    """
    if public_state is None or color is None:
        return ""
    current = _player_longest_road_length(public_state, color, None)
    projected = _player_longest_road_length(public_state, color, extra_edges)
    delta = projected - current
    # Base fragment
    if delta == 0:
        # Still inform current length; useful to see no growth (e.g., closing a loop)
        base = f" | Longest road: {current} -> {projected} (no change)"
    elif delta > 0:
        base = f" | Longest road: {current} -> {projected} (+{delta})"
    else:
        base = f" | Longest road: {current} -> {projected} ({delta})"

    # Longest Road bonus hint (5+ roads and strictly longer than current holder)
    try:
        global_len = getattr(public_state.board, "longest_road_length", 0) or 0
        holder = getattr(public_state.board, "longest_road_color", None)
    except Exception:
        global_len, holder = 0, None

    would_claim = False
    if projected >= 5 and projected > global_len:
        # If holder is the same color, extending still counts as holding, but
        # the interesting case is taking/retaining with a new longer length.
        # Show the hint whenever the player would be the holder after the move.
        if holder is None or holder != color or projected > global_len:
            would_claim = True

    if would_claim:
        if holder == color:
            base += " [would extend Longest Road]"
        else:
            base += " [would claim Longest Road, +2 VP]"
        if projected < 5:
            # Not actually claimable yet, but keep hint subtle
            base += " (needs 5)"
    elif projected >= 5 and holder == color and projected == global_len:
        base += " [holds Longest Road]"

    return base


def _tile_id_for_coordinate(public_state: Optional[PublicState], coordinate) -> Optional[int]:
    """Reverse lookup coordinate -> tile_id via public map."""
    if public_state is None or coordinate is None:
        return None
    for tile_id, coord in public_state.board.map.tile_coordinates.items():
        if coord == coordinate:
            return tile_id
    return None


def _robber_tile_detail(public_state: Optional[PublicState], coordinate) -> str:
    """Rich robber tile detail: tile resource/roll/pips + occupants + card counts.

    Returns e.g. ``Tile 7: 8 ORE (5 pips) | Occupants: RED city at Node 10 (10 pips, 4 cards), BLUE settlement at Node 11 (5 pips, 2 cards)``.
    """
    if public_state is None or coordinate is None:
        return _coordinate_tile_label(public_state, coordinate)
    tile_id = _tile_id_for_coordinate(public_state, coordinate)
    if tile_id is None:
        return _format_coordinate(coordinate)
    resource, roll = public_state.board.map.tiles.get(tile_id, (None, None))
    if resource is None:
        tile_str = f"Tile {tile_id}: DESERT"
        pips = 0
    else:
        resource_name = resource.name if hasattr(resource, 'name') else str(resource)
        pips = get_pip_count(roll)
        tile_str = f"Tile {tile_id}: {roll} {resource_name} ({pips} pips)"

    # Occupants on this tile
    tiles_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for nid, tids in public_state.board.map.adjacent_tiles.items():
        for tid in tids:
            tiles_to_nodes[tid].append(nid)
    occupants: Dict[Any, List[Tuple[int, str, int]]] = defaultdict(list)  # color -> list of (node, type, pips_blocked)
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
        return f"{tile_str} | No occupants"

    parts = []
    for owner in sorted(occupants.keys(), key=lambda c: getattr(c, "name", str(c))):
        color_name = _name_of(owner)
        hand_cards = public_state.players.get(owner)
        card_count = getattr(hand_cards, "hand_resource_count", "?") if hand_cards is not None else "?"
        # sum pips for this owner on this tile
        total_blocked = sum(bp for _, _, bp in occupants[owner])
        nodes_desc = ", ".join(
            f"{btype.lower()} at Node {nid} ({bp} pips)" for nid, btype, bp in sorted(occupants[owner])
        )
        parts.append(f"{color_name}: {nodes_desc} | {total_blocked} pips blocked, {card_count} cards")
    occupants_str = "; ".join(parts)
    return f"{tile_str} | Occupants: {occupants_str}"


def _road_node_detail(
    public_state: Optional[PublicState],
    edge: Tuple[int, int],
    exclude_nodes: Optional[Set[int]] = None,
    network_nodes: Optional[Set[int]] = None,
    extra_occupied: Optional[Set[int]] = None,
    extra_occupied_color: Any = None,
) -> str:
    """Describe settlement opportunities reachable via ``edge``.

    A road touches exactly one *new* node (the tip) and from that tip two
    further nodes are one road-length away. This mirrors the board geometry:
    interior nodes have degree 3, so from the tip there are two forward
    extensions. Both the direct tip and the two forward nodes are shown with
    full tile/port/pip detail and an explicit availability tag
    (``available`` vs ``blocked (occupied ...)`` / ``blocked (too close ...)``).

    Args:
        public_state: Public game state for tile/port lookups.
        edge: Sorted ``(n1, n2)`` road edge.
        exclude_nodes: Nodes to exclude from display (e.g., the settlement node
            of an initial-placement ``settlement -> road`` bundle, which will be
            occupied).
        network_nodes: Player's current road/settlement network. If provided,
            the *new* tip is the endpoint not in this network; otherwise the
            tip is inferred as the endpoint(s) not in ``exclude_nodes``. This
            keeps labels deterministic for roads that close a loop.
        extra_occupied: Nodes that will be occupied after the current move
            (e.g., the settlement at node 0 for initial placement). These are
            treated as occupied for distance-rule checks, so immediate neighbours
            of the new settlement are correctly reported as blocked.

    Returns:
        Suffix like ``" | reaches Node 5: ... [available] | extends toward
        Node 10: ... [blocked (...)] , Node 11: ... [available]"``.
    """
    if public_state is None:
        return ""
    a, b = tuple(sorted(edge))
    exclude = set(exclude_nodes or [])

    # Determine the "new" tip(s) of the road.
    if network_nodes is not None:
        new_tips = [n for n in (a, b) if n not in network_nodes and n not in exclude]
        # If both endpoints are already in the network (closing a loop) or both
        # are new (should not happen for legal moves), fall back to any
        # non-excluded endpoint so we still describe something.
        if not new_tips:
            new_tips = [n for n in (a, b) if n not in exclude]
            # If still empty (edge == excluded), nothing to describe.
            if not new_tips:
                return ""
        # Legal placements extend by one tip at a time; show the first new tip
        # deterministically (sorted) to avoid 4-node blowup and match user's
        # "1 new node + 2 forward" expectation.
        new_tips = sorted(new_tips)
        # If two new tips (rare), we describe both tips but cap extensions to
        # two per tip; total then is 2 direct + up to 4 extended. Prefer to
        # describe only the first tip to keep labels compact.
        if len(new_tips) > 1:
            new_tips = new_tips[:1]
    else:
        # No network context: treat the first non-excluded endpoint as the tip
        # (setup settlement case where settlement node is excluded).
        candidates = [n for n in (a, b) if n not in exclude]
        if not candidates:
            return ""
        new_tips = sorted(candidates)[:1]

    segments: List[str] = []
    # Collect forward nodes for the tip(s)
    for tip in new_tips:
        is_ok, reason = _node_buildability_detail(
            public_state, tip, extra_occupied=extra_occupied, extra_occupied_color=extra_occupied_color
        )
        segments.append(f"reaches {_describe_node(public_state, tip)} [{reason}]")
        # If the tip itself is already developed (has a settlement/city, or
        # will have one from the current move), no further extension is useful
        # — the network cannot grow past an occupied node.
        tip_occupied = (public_state.board.buildings.get(tip) is not None) or (
            extra_occupied is not None and tip in extra_occupied
        )
        if tip_occupied:
            continue
        # Forward extensions from the tip, excluding the edge itself and any
        # excluded nodes. Degree-3 interior => exactly 2 forward nodes.
        forward = []
        for nb in STATIC_GRAPH.neighbors(tip):
            if nb in (a, b):
                continue
            if nb in exclude:
                continue
            forward.append(nb)
        forward = sorted(forward)
        if forward:
            fwd_parts = []
            for fwd in forward:
                ok, rsn = _node_buildability_detail(
                    public_state, fwd, extra_occupied=extra_occupied, extra_occupied_color=extra_occupied_color
                )
                fwd_parts.append(f"{_describe_node(public_state, fwd)} [{rsn}]")
            segments.append(f"extends toward {', '.join(fwd_parts)}")

    if not segments:
        return " | no buildable settlement spots nearby"
    return " | " + " | ".join(segments)


def _coordinate_tile_label(public_state: Optional[PublicState], coordinate) -> str:
    """Render a coordinate as its board-map tile ID (falls back to the raw
    coordinate when the public map is unavailable)."""
    if public_state is None or coordinate is None:
        return _format_coordinate(coordinate)
    for tile_id, coord in public_state.board.map.tile_coordinates.items():
        if coord == coordinate:
            return f"Tile {tile_id}"
    return _format_coordinate(coordinate)


def _label_action(action: Action, public_state: Optional[PublicState] = None) -> str:
    """A concise human description of a single action worth choosing."""
    kind = action.action_type
    value = action.value
    name = getattr(kind, "name", str(kind))

    if kind == ActionType.ROLL:
        return "Roll the dice"
    if kind == ActionType.END_TURN:
        return "End turn"
    if kind == ActionType.BUILD_ROAD:
        edge = tuple(sorted(value))
        if public_state is not None:
            color = getattr(action, "color", None)
            network = None
            try:
                if color is not None:
                    network = _own_network_nodes(public_state, color)
            except Exception:
                network = None
            detail = _road_node_detail(public_state, edge, network_nodes=network)
            longest = _longest_road_suffix(public_state, color, [edge])
            return f"Build road on edge {edge}{detail}{longest}"
        return f"Build road on edge {edge}"
    if kind == ActionType.BUILD_SETTLEMENT:
        if public_state is not None:
            return f"Build settlement at {_describe_node(public_state, value)}"
        return f"Build settlement at node {value}"
    if kind == ActionType.BUILD_CITY:
        if public_state is not None:
            return f"Build city at {_describe_node(public_state, value)}"
        return f"Build city at node {value}"
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return "Buy a development card"
    if kind == ActionType.PLAY_KNIGHT_CARD:
        return "Play Knight (then move the robber)"
    if kind == ActionType.PLAY_YEAR_OF_PLENTY:
        cards = ", ".join(_name_of(r) for r in value)
        return f"Play Year of Plenty: take {cards}"
    if kind == ActionType.PLAY_MONOPOLY:
        return f"Play Monopoly: steal all {_name_of(value)}"
    if kind == ActionType.PLAY_ROAD_BUILDING:
        return "Play Road Building (then build two roads)"
    if kind == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        if public_state is not None:
            tile_detail = _robber_tile_detail(public_state, coordinate)
            if victim is None:
                return f"Move robber to {tile_detail} (no steal)"
            return f"Move robber to {tile_detail} and steal from {_name_of(victim)}"
        coord_str = _coordinate_tile_label(public_state, coordinate)
        if victim is None:
            return f"Move robber to {coord_str} (no steal)"
        return f"Move robber to {coord_str} and steal from {_name_of(victim)}"
    if kind == ActionType.DISCARD_RESOURCE:
        return f"Discard one {_name_of(value)}"
    if kind == ActionType.MARITIME_TRADE:
        return f"Maritime trade: {_format_maritime_trade_value(value)}"
    if kind == ActionType.OFFER_TRADE:
        return f"Offer trade: {_format_trade_offer_value(value)}"
    if kind == ActionType.ACCEPT_TRADE:
        return f"Accept trade: {_format_trade_offer_value(value)}"
    if kind == ActionType.REJECT_TRADE:
        return f"Reject trade: {_format_trade_offer_value(value)}"
    if kind == ActionType.CONFIRM_TRADE:
        trade_part = _format_trade_offer_value(value[:10])
        acceptor = _name_of(value[10]) if len(value) > 10 else "unknown"
        return f"Confirm trade with {acceptor}: {trade_part}"
    if kind == ActionType.CANCEL_TRADE:
        return "Cancel trade"
    return f"{name}: value={value!r}"


def _knight_robber_followups(public_state: PublicState, color) -> List[Tuple]:
    """Legal (coordinate, victim_or_None) MOVE_ROBBER targets after a Knight.

    Mirrors the engine's ``robber_possibilities`` for a non-friendly-robber game,
    derived entirely from public data: every land tile except the current robber
    tile, stealing from any enemy holding at least one card. Kept in exact parity
    so a bundled Knight move is accepted by the engine on the follow-up prompt.
    """
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
    """Expand one PLAY_KNIGHT_CARD into bundled Knight + MOVE_ROBBER moves.

    Each bundle is a single LLM choice that includes where to move the robber
    and who to steal from, so the agent never decides the knight and the robber
    in disjoint prompts.
    """
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


def _own_network_nodes(public_state: PublicState, color) -> Set[int]:
    """Nodes of the player's road/settlement network.

    Mirrors the engine's connected-component nodes: own buildings plus the
    endpoints of own roads, minus nodes occupied by an enemy settlement/city
    (the network cannot pass through or build out of an enemy node).
    """
    board = public_state.board
    own_buildings = {n for n, (owner, _) in board.buildings.items() if owner == color}
    enemy_buildings = {n for n, (owner, _) in board.buildings.items() if owner != color}
    endpoints = set()
    for edge, owner in board.roads.items():
        if owner == color:
            endpoints.update(edge)
    return (own_buildings | endpoints) - enemy_buildings


def _land_edges_from(public_state: PublicState, color, nodes) -> List[Tuple[int, int]]:
    """Sorted list of unowned land edges touching any of ``nodes``.

    Mirrors the engine's ``Board.buildable_edges`` for a player whose network is
    ``nodes``: every static-graph edge incident to a non-enemy network node
    whose endpoints are land nodes and which no one owns yet. Enemy nodes are
    excluded from the originating set because a network can never extend out of
    an enemy settlement/city (roads may only run up to it).
    """
    land = public_state.board.map.land_nodes
    owned = set(public_state.board.roads.keys())  # keys are already sorted pairs
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


def _road_building_moves(play_card: Action, public_state: PublicState) -> List[Move]:
    """Expand PLAY_ROAD_BUILDING into concrete bundles of both road edges.

    Emits one move per unordered pair of roads, so building two disconnected
    edges "first" in either order is a single move (the resulting board is
    identical). Roads are shown as sorted ``(n1, n2)`` node pairs. When only a
    single road is possible the bundle has just one road. Each road is
    annotated with reachable settlement spots (direct + one-away) so the LLM
    can judge expansion value.
    """
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
                    continue  # the reverse order is the same move
                seen_pairs.add(pair)
                road_a, road_b = sorted((first, second))
                detail_a = _road_node_detail(public_state, first, network_nodes=base_network)
                # Second road is placed after first, so its network includes first
                second_network_for_detail = base_network | set(first)
                detail_b = _road_node_detail(public_state, second, network_nodes=second_network_for_detail)
                longest = _longest_road_suffix(public_state, color, [first, second])
                # Keep the sorted-road pair prefix stable, then annotate each road
                label = f"Play Road Building -> build roads {road_a} and {road_b} | road {first}{detail_a} | road {second}{detail_b}{longest}"
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
    """Expand an initial-placement settlement into concrete settlement + road moves.

    Each settlement node is bundled with every legal road edge incident to it,
    so the LLM chooses the road as part of the same initial-placement move.
    Labels include rich node/edge context: settlement tile/port/pips and road
    expansion potential.
    """
    color = settle.color
    node = settle.value
    road_options = _land_edges_from(public_state, color, {node})
    if not road_options:
        # Degenerate safety net: no road is legal from this node.
        return [Move(label=_label_action(settle, public_state), actions=[settle])]
    settle_desc = _describe_node(public_state, node)
    return [
        Move(
            label=f"Build settlement at {settle_desc} -> build road {edge}{_road_node_detail(public_state, edge, exclude_nodes={node}, network_nodes={node}, extra_occupied={node}, extra_occupied_color=color)}{_longest_road_suffix(public_state, color, [edge])}",
            actions=[settle, Action(color, ActionType.BUILD_ROAD, edge)],
        )
        for edge in road_options
    ]


def build_moves(playable_actions: Sequence[Action], observation=None) -> List[Move]:
    """Build the LLM-choosable moves for a set of playable engine actions.

    Args:
        playable_actions: The engine's legal actions for the current prompt.
        observation: Optional Observation. When provided, its public_state and
            current_prompt drive compound-move expansion:
            - PLAY_KNIGHT_CARD is expanded into one move per (robber tile,
              steal victim) option, so playing the card and moving the robber
              are decided together.
            - An initial settlement (BUILD_INITIAL_SETTLEMENT prompt) is
              bundled with every legal road edge incident to that node, so the
              placement is decided as one move.
            - PLAY_ROAD_BUILDING is bundled with each legal (first road, second
              road) pair, so both free roads are chosen as part of the move.
            Year of Plenty and Monopoly already carry their resource parameters
            in a single action and are formatted as one move.
        Without an observation the road-carrying moves degrade to an annotated
        opener that the agent completes from the live prompt (see AUTO_ROAD).

    Returns:
        List of Moves; the LLM selects exactly one by index.
    """
    public_state = getattr(observation, "public_state", None)
    current_prompt = getattr(observation, "current_prompt", None)

    if not playable_actions:
        return []

    moves: List[Move] = []
    for action in playable_actions:
        kind = action.action_type

        if kind == ActionType.PLAY_KNIGHT_CARD and public_state is not None:
            moves.extend(_knight_moves(action, public_state))
        elif kind == ActionType.PLAY_ROAD_BUILDING and public_state is not None:
            moves.extend(_road_building_moves(action, public_state))
        elif kind == ActionType.PLAY_ROAD_BUILDING:
            label = "Play Road Building -> then build two roads"
            moves.append(Move(label=label, actions=[action, AUTO_ROAD, AUTO_ROAD]))
        elif (
            kind == ActionType.BUILD_SETTLEMENT
            and current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
            and public_state is not None
        ):
            moves.extend(_setup_settlement_moves(action, public_state))
        elif (
            kind == ActionType.BUILD_SETTLEMENT
            and current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
        ):
            label = f"{_label_action(action, public_state)} -> then place your initial road"
            moves.append(Move(label=label, actions=[action, AUTO_ROAD]))
        else:
            moves.append(Move(label=_label_action(action, public_state), actions=[action]))
    return moves


def format_moves(moves: Sequence[Move], observation=None) -> str:
    """Render moves as a numbered, LLM-readable list.

    The list index is the stable handle the LLM returns; see ``parse_move``.
    """
    current_prompt = getattr(observation, "current_prompt", None)
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
    """Convenience wrapper: build moves for ``playable_actions`` and format them.

    For agents that need the moves back (to map the LLM's chosen index to an
    Action), use ``build_moves`` + ``parse_move`` instead.
    """
    return format_moves(build_moves(playable_actions, observation), observation=observation)


def parse_move(response, moves: Sequence[Move]) -> Move:
    """Convert an LLM's response (a stable move index) into the chosen Move.

    Accepts an int, a bare number, a bracketed number like ``[3]``, or a
    numbered line like ``3. Build city at node 10``.
    """
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
    """Pick a legal BUILD_ROAD from the current prompt's playable actions.

    Used to complete moves whose road placement cannot be bundled upfront
    (initial-placement road, Road Building card roads). Scores each legal edge
    by the pip value of both endpoints so the road extends toward productive
    terrain; ties break deterministically by edge order.
    """
    roads = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
    if not roads:
        return None

    def score(action: Action) -> Tuple[int, Tuple]:
        edge = tuple(sorted(action.value))
        return _node_pip_total(public_state, edge[0]) + _node_pip_total(public_state, edge[1]), edge

    return max(roads, key=score)
