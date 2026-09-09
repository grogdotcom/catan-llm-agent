"""
Collect high-decision move prompts via LLMObservationAgent.build_full_prompt.

Each high decision (initial placement, robber, builds, or ambiguous value
difference <0.05) is turned into a supervised record:

  {
    "prompt": <six-section LLM prompt via build_full_prompt>,
    "completion": "3",
    "chosen_index": 3,
    "chosen_label": "...",
    "phase": "BUILD_ROAD",
    ...
  }

Grouped initial placements (settlement + road) are emitted as a single
record whose chosen move is the bundled settlement→road Move — so the LLM
chooses both as one decision, mirroring llm_agent.MoveExecutor.

Output is JSONL (one JSON object per line).
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from catanatron.game import Game, GameAccumulator
from catanatron.players.minimax import AlphaBetaPlayer, DebugStateNode
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType, ActionPrompt
from catanatron.models.observation import Observation
from catanatron.models.perspective_player import (
    _build_public_state,
    _build_inventory,
    _sanitize_history,
    _build_pending_trades,
)
from catanatron.features import create_sample

from catan_llm.llm_agent import LLMObservationAgent
from catan_llm.format import build_moves
from catan_llm.format.utils import _name_of


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_action(action: Action) -> Dict[str, Any]:
    """Serialize an Action to JSON-safe dict."""
    color = getattr(action.color, "name", str(action.color)) if action.color is not None else None
    typ = getattr(action.action_type, "name", str(action.action_type))
    val = action.value
    # Normalize common shapes
    if typ == "MOVE_ROBBER" and isinstance(val, tuple) and len(val) == 2:
        coord, victim = val
        victim_name = getattr(victim, "name", str(victim)) if victim is not None else None
        return {
            "color": color,
            "type": typ,
            "coordinate": list(coord) if coord is not None else None,
            "victim": victim_name,
            "raw": str(val),
        }
    if typ == "BUILD_ROAD" and isinstance(val, (tuple, list)):
        return {"color": color, "type": typ, "edge": list(val), "raw": str(val)}
    if typ in ("BUILD_SETTLEMENT", "BUILD_CITY") and val is not None:
        return {"color": color, "type": typ, "node": val, "raw": str(val)}
    if isinstance(val, tuple):
        return {"color": color, "type": typ, "value": list(val), "raw": str(val)}
    return {"color": color, "type": typ, "value": val if val is not None else None, "raw": str(val) if val is not None else None}


def _build_observation(game: Game, color: Color) -> Observation:
    """Build a sanitized Observation for color at this game snapshot."""
    public_state = _build_public_state(game)
    inventory = _build_inventory(game, color)
    history = tuple(_sanitize_history(game, color))
    pending = _build_pending_trades(game)
    features = create_sample(game, color)
    return Observation(
        color=color,
        features=features,
        public_history=history,
        current_prompt=game.state.current_prompt,
        pending_trades=pending,
        public_state=public_state,
        inventory=inventory,
    )


def _find_chosen_move(
    moves,
    selected: Action,
    next_road_selected: Optional[Action] = None,
) -> Tuple[Optional[int], Optional[Any]]:
    """Return (1-indexed chosen_index, Move) matching selected (+next road if grouped)."""
    if not moves:
        return None, None

    # Grouped initial placement: settlement + road
    if next_road_selected is not None:
        sel_edge = tuple(sorted(next_road_selected.value)) if isinstance(next_road_selected.value, (tuple, list)) else next_road_selected.value
        for idx, m in enumerate(moves, start=1):
            if not m.actions:
                continue
            if m.actions[0] != selected:
                continue
            # bundled move should have at least 2 actions
            if len(m.actions) >= 2:
                cand = m.actions[1]
                if cand.action_type == ActionType.BUILD_ROAD:
                    cand_edge = tuple(sorted(cand.value)) if isinstance(cand.value, (tuple, list)) else cand.value
                    if cand_edge == sel_edge:
                        return idx, m
        # Fallback: any move with settlement at same node
        for idx, m in enumerate(moves, start=1):
            if m.actions and m.actions[0] == selected:
                return idx, m
        return 1, moves[0]

    # Normal: match first action, or any bundled action containing selected
    for idx, m in enumerate(moves, start=1):
        if m.actions and m.actions[0] == selected:
            return idx, m
    for idx, m in enumerate(moves, start=1):
        if selected in m.actions:
            return idx, m
    # Edge-normalized fallback for road edges that may be unsorted vs sorted
    if selected.action_type == ActionType.BUILD_ROAD and isinstance(selected.value, (tuple, list)):
        sel_edge = tuple(sorted(selected.value))
        for idx, m in enumerate(moves, start=1):
            for a in m.actions:
                if a.action_type == ActionType.BUILD_ROAD and isinstance(a.value, (tuple, list)):
                    if tuple(sorted(a.value)) == sel_edge:
                        return idx, m
    return None, None


def decision_to_record(
    game_snapshot: Game,
    playable_actions: List[Action],
    selected_action: Action,
    next_road_selected: Optional[Action],
    game_id: int,
    decision_id: int,
    winner_color: Optional[Color],
    action_values: Optional[Dict[Action, float]] = None,
) -> Dict[str, Any]:
    """Convert a single high decision into a JSONL record with prompt.

    Args:
        game_snapshot: Game copy at decision time (before action).
        playable_actions: List of engine-legal actions at that point.
        selected_action: The AlphaBeta chosen action (first of bundled move).
        next_road_selected: For initial settlement groups, the chosen road action.
        game_id: Index of the game in the corpus run.
        decision_id: Sequential id within this game's high decisions.
        winner_color: Game winner (for filtering callers, included as metadata).
        action_values: Optional map action->expected value from search.

    Returns:
        Dict ready to be json.dumps'd as one JSONL line.
    """
    color = selected_action.color
    # Build observation via helper; moves + chosen index must be resolved
    # before the prompt so the initial-placement footer can embed the exact
    # 1-indexed Move ID selected by the Grandmaster engine.
    observation = _build_observation(game_snapshot, color)
    inventory = observation.inventory

    moves = build_moves(playable_actions, observation)
    chosen_index, chosen_move = _find_chosen_move(moves, selected_action, next_road_selected)

    # Fallback if chosen not found (should not happen) — pick first
    if chosen_index is None:
        chosen_index = 1
        chosen_move = moves[0] if moves else None

    # Prepare serialization helpers needed for footer decision
    phase = getattr(game_snapshot.state.current_prompt, "name", str(game_snapshot.state.current_prompt))
    turn_number = getattr(game_snapshot.state, "num_turns", None)
    if turn_number is None:
        turn_number = getattr(game_snapshot.state, "current_turn_index", None)

    # Initial placements get the Grandmaster disclosure footer with the exact
    # chosen Move ID; all other phases keep the default footer.
    is_initial = game_snapshot.state.current_prompt in [
        ActionPrompt.BUILD_INITIAL_SETTLEMENT,
        ActionPrompt.BUILD_INITIAL_ROAD,
    ]
    agent = LLMObservationAgent(color)
    if is_initial:
        footer = (
            "[DECISION REQUIRED]\n"
            f"The Grandmaster engine has selected Move ID {chosen_index} as the optimal action.\n"
            "Explain the strategic and tactical reasoning behind this exact choice, then output the action ID. "
        )
        prompt = agent.build_full_prompt(observation, playable_actions, inventory, footer=footer)
    else:
        prompt = agent.build_full_prompt(observation, playable_actions, inventory)

    chosen_label = chosen_move.label if chosen_move is not None else _serialize_action(selected_action).get("raw", "")

    move_labels = [f"{i}. {m.label}" for i, m in enumerate(moves, start=1)]

    record: Dict[str, Any] = {
        "prompt": prompt,
        "completion": str(chosen_index),
        "chosen_index": chosen_index,
        "chosen_label": chosen_label,
        "phase": phase,
        "color": getattr(color, "name", str(color)),
        "turn": turn_number,
        "num_moves": len(moves),
        "move_labels": move_labels,
        "selected_action": _serialize_action(selected_action),
        "winner": getattr(winner_color, "name", str(winner_color)) if winner_color is not None else None,
        "game_id": game_id,
        "decision_id": decision_id,
    }
    if next_road_selected is not None:
        record["next_road"] = _serialize_action(next_road_selected)
        record["grouped"] = True
        record["grouped_actions"] = [
            _serialize_action(selected_action),
            _serialize_action(next_road_selected),
        ]
    # Optional: include values for ambiguity analysis
    if action_values:
        # Serialize top few values for debuggability (sorted desc)
        try:
            sorted_vals = sorted(action_values.items(), key=lambda kv: kv[1], reverse=True)[:5]
            record["top_action_values"] = [
                {"action": _serialize_action(a), "value": float(v)} for a, v in sorted_vals
            ]
        except Exception:
            pass
    return record


# ---------------------------------------------------------------------------
# Player + accumulator
# ---------------------------------------------------------------------------

class CorpusCollectionPlayer(AlphaBetaPlayer):
    def __init__(self, color, depth=2, prunning=False):
        super().__init__(color, depth=depth, prunning=prunning)
        self.decisions: List[Dict[str, Any]] = []

    def decide(self, game: Game, playable_actions):
        # Keep full list for high-decision analysis; `playable_actions` is
        # the engine truth (unpruned, respects friendly_robber).
        # `actions` is the (possibly pruned) search frontier. With
        # prunning=False they are identical; with prunning=True the stored
        # `playable_actions` must remain the full set so corpus records show
        # all robber tiles instead of the single pruned target.
        actions = self.get_actions(game)

        if len(actions) == 1:
            # Even in the trivial branch, preserve the full engine options
            # for corpus analysis — don't collapse to the pruned singleton.
            decision = {
                "state": game.copy(),
                "actions": {actions[0]: 0.0},
                "selected": actions[0],
                "playable_actions": list(playable_actions),
            }
            self.decisions.append(decision)
            return actions[0]

        start = time.time()
        state_id = str(len(game.state.action_records))
        node = DebugStateNode(state_id, self.color)
        deadline = start + 20  # 20 seconds max
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline, node
        )

        action_values: Dict[Action, float] = {}
        for action_node in node.children:
            action_values[action_node.action] = action_node.expected_value

        selected_action = result[0] if result[0] is not None else playable_actions[0]

        self.decisions.append(
            {
                "state": game.copy(),
                "actions": action_values,
                "selected": selected_action,
                "playable_actions": list(playable_actions),
            }
        )
        return selected_action

    def reset_state(self):
        super().reset_state()
        self.decisions = []


class CorpusAccumulator(GameAccumulator):
    """Collects winner's high decisions and turns them into prompt records.

    Each entry in ``self.corpus`` is already a JSONL-ready dict with
    ``prompt``/``completion`` etc., not a raw Game snapshot.
    """

    def __init__(self, players, game_id: int = 0):
        self.players = players
        self.game_id = game_id
        self.corpus: List[Dict[str, Any]] = []

    def after(self, game):
        winner_color = game.winning_color()
        if winner_color is None:
            return

        winner_player = next((p for p in self.players if p.color == winner_color), None)
        if not isinstance(winner_player, CorpusCollectionPlayer):
            return

        decisions = winner_player.decisions

        i = 0
        decision_id = 0
        while i < len(decisions):
            decision = decisions[i]
            selected = decision["selected"]
            action_values = decision["actions"]
            state_before = decision["state"]
            playable = decision.get("playable_actions", list(action_values.keys()))

            # Fallback if playable was not captured (older logic)
            if not playable:
                playable = list(action_values.keys())

            is_high_decision = False

            # 1. Initial placements
            is_initial = state_before.state.current_prompt in [
                ActionPrompt.BUILD_INITIAL_SETTLEMENT,
                ActionPrompt.BUILD_INITIAL_ROAD,
            ]

            # 2. Moving robber
            is_robber = selected.action_type == ActionType.MOVE_ROBBER

            # 3. Building
            is_build = selected.action_type in [
                ActionType.BUILD_ROAD,
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
            ]

            # 4. Ambiguous (difference < 0.05)
            if len(action_values) > 1:
                try:
                    sorted_values = sorted(action_values.values(), reverse=True)
                    if sorted_values[0] - sorted_values[1] < 0.05:
                        is_high_decision = True
                except Exception:
                    pass

            if is_initial or is_robber or is_build or is_high_decision:
                # Group initial placements: Settlement + next Road as ONE prompt
                if state_before.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
                    next_initial_road_selected = None
                    # playable for the settlement is current; next decision holds road choice
                    if i + 1 < len(decisions):
                        potential_road_decision = decisions[i + 1]
                        if potential_road_decision["state"].state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
                            next_initial_road_selected = potential_road_decision["selected"]
                            i += 1  # skip next in loop

                    record = decision_to_record(
                        game_snapshot=state_before,
                        playable_actions=playable,
                        selected_action=selected,
                        next_road_selected=next_initial_road_selected,
                        game_id=self.game_id,
                        decision_id=decision_id,
                        winner_color=winner_color,
                        action_values=action_values,
                    )
                    self.corpus.append(record)
                    decision_id += 1
                else:
                    record = decision_to_record(
                        game_snapshot=state_before,
                        playable_actions=playable,
                        selected_action=selected,
                        next_road_selected=None,
                        game_id=self.game_id,
                        decision_id=decision_id,
                        winner_color=winner_color,
                        action_values=action_values,
                    )
                    self.corpus.append(record)
                    decision_id += 1
            i += 1


# ---------------------------------------------------------------------------
# Placements-only accumulator — all players, initial settlement+road only
# ---------------------------------------------------------------------------


class PlacementsAccumulator(GameAccumulator):
    """Collects *all* initial-placement decisions (not just winner).

    Each bundled settlement→road placement is one record, so a 4-player game
    yields 8 records.  Unlike :class:`CorpusAccumulator` this ignores
    ``winner_color`` for filtering — every player's placements are emitted.
    ``winner`` is still stored as metadata but does not gate inclusion.
    """

    def __init__(self, players, game_id: int = 0):
        self.players = players
        self.game_id = game_id
        self.corpus: List[Dict[str, Any]] = []

    def after(self, game):
        winner_color = game.winning_color()

        # Merge all players' decisions into a single chronological stream.
        # Sorting by the snapshot's action-record length recovers global order
        # (settlement and its road are consecutive globally).
        merged: List[Dict[str, Any]] = []
        for p in self.players:
            if not isinstance(p, CorpusCollectionPlayer):
                continue
            for d in p.decisions:
                merged.append(d)
        # Stable sort by time (action_records length at decision snapshot)
        try:
            merged.sort(key=lambda d: len(d["state"].state.action_records))
        except Exception:
            pass

        decision_id = 0
        i = 0
        while i < len(merged):
            decision = merged[i]
            selected = decision["selected"]
            action_values = decision["actions"]
            state_before = decision["state"]
            playable = decision.get("playable_actions", list(action_values.keys()))
            if not playable:
                playable = list(action_values.keys())

            is_initial = state_before.state.current_prompt in [
                ActionPrompt.BUILD_INITIAL_SETTLEMENT,
                ActionPrompt.BUILD_INITIAL_ROAD,
            ]
            if not is_initial:
                i += 1
                continue

            # Bundle settlement + next road as one prompt (same as CorpusAccumulator)
            if state_before.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
                next_road_selected = None
                if i + 1 < len(merged):
                    nxt = merged[i + 1]
                    if nxt["state"].state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
                        # Only bundle if same player (should always hold for initial)
                        if nxt["selected"].color == selected.color:
                            next_road_selected = nxt["selected"]
                            i += 1  # skip the road in the outer loop

                record = decision_to_record(
                    game_snapshot=state_before,
                    playable_actions=playable,
                    selected_action=selected,
                    next_road_selected=next_road_selected,
                    game_id=self.game_id,
                    decision_id=decision_id,
                    winner_color=winner_color,
                    action_values=action_values,
                )
                self.corpus.append(record)
                decision_id += 1
            else:
                # Standalone road that wasn't bundled (defensive — shouldn't
                # happen for legal initial placements, but emit it anyway).
                record = decision_to_record(
                    game_snapshot=state_before,
                    playable_actions=playable,
                    selected_action=selected,
                    next_road_selected=None,
                    game_id=self.game_id,
                    decision_id=decision_id,
                    winner_color=winner_color,
                    action_values=action_values,
                )
                self.corpus.append(record)
                decision_id += 1
            i += 1


# ---------------------------------------------------------------------------
# Simulation runner — writes JSONL
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _derive_trajectory_provenance(
    *,
    game_id: int,
    game_seed: int,
    seat_order: List[str],
    winner_color: Optional[Color],
    num_turns: int,
) -> Dict[str, Any]:
    """Build trajectory provenance dict for a finished game."""
    import hashlib

    seat_str = "-".join(seat_order)
    h = hashlib.sha256(f"{game_seed}-{seat_str}".encode()).hexdigest()[:12]
    trajectory_id = f"seed-{game_seed}-{h}"
    winner_name = getattr(winner_color, "name", str(winner_color)) if winner_color is not None else None
    # Map color -> seat_index
    color_to_seat = {c: i for i, c in enumerate(seat_order)}
    return {
        "trajectory_id": trajectory_id,
        "game_seed": game_seed,
        "game_end_turn": num_turns,
        "seat_order": seat_order,
        "winner": winner_name,
        "color_to_seat": color_to_seat,
    }


def run_simulation(
    num_games=1000,
    output_file="data/sft/high_decision_moves.jsonl",
    base_seed: int = 1000,
    manifest_path: Optional[str] = None,
):
    """Run AlphaBeta self-play and collect high-decision prompts to JSONL.

    Uses ``prunning=False`` and ``friendly_robber=False`` so robber moves
    expose the full unfiltered option set (every land tile except the robber
    tile, with one victim entry per steal target) instead of the single
    pruned target. This fixes the prior corpus where every ``MOVE_ROBBER``
    record had ``num_moves == 1`` and ``completion == "1"`` (no learning
    signal).

    Each record is annotated with trajectory provenance (``trajectory_id``,
    ``game_seed``, ``game_end_turn``, ``seat_index``, ``trajectory_index``)
    and the run also emits a trajectory manifest JSONL.

    Args:
        num_games: Number of games to simulate.
        output_file: Path to JSONL output. Each line is a full prompt record
            with ``prompt`` (six-section via build_full_prompt), ``completion``
            (1-indexed chosen move), and metadata. Overwritten incrementally
            every 10 games and at the end.
        base_seed: Base seed; game i uses ``base_seed + i`` as game_seed.
        manifest_path: Optional path for trajectory manifest JSONL (defaults to
            ``<output_dir>/trajectory_manifest.jsonl``).
    """
    all_high_decisions: List[Dict[str, Any]] = []
    seat_order = ["RED", "BLUE", "ORANGE", "WHITE"]
    manifests: List[Dict[str, Any]] = []
    # Determine manifest path
    if manifest_path is None:
        try:
            from pathlib import Path as _P

            manifest_path = str(_P(output_file).parent / "trajectory_manifest.jsonl")
        except Exception:
            manifest_path = None

    player_instances = [
        CorpusCollectionPlayer(Color.RED, prunning=False),
        CorpusCollectionPlayer(Color.BLUE, prunning=False),
        CorpusCollectionPlayer(Color.ORANGE, prunning=False),
        CorpusCollectionPlayer(Color.WHITE, prunning=False),
    ]

    for i in range(num_games):
        for p in player_instances:
            p.reset_state()
        game_seed = base_seed + i
        accumulator = CorpusAccumulator(player_instances, game_id=i)
        game = Game(player_instances, seed=game_seed, friendly_robber=False)
        game.play(accumulators=[accumulator])

        # Derive provenance from finished game
        num_turns = getattr(game.state, "num_turns", None)
        if num_turns is None:
            num_turns = getattr(game.state, "current_turn_index", 0) or 0
        winner_color = game.winning_color()
        prov = _derive_trajectory_provenance(
            game_id=i,
            game_seed=game_seed,
            seat_order=seat_order,
            winner_color=winner_color,
            num_turns=int(num_turns),
        )
        # Build manifest entry
        manifest = {
            "trajectory_id": prov["trajectory_id"],
            "game_seed": prov["game_seed"],
            "game_end_turn": prov["game_end_turn"],
            "game_id": i,
            "seat_order": seat_order,
            "winner": prov["winner"],
            "players": [
                {"seat_index": idx, "color": c, "is_winner": 1 if c == prov["winner"] else 0}
                for idx, c in enumerate(seat_order)
            ],
        }
        manifests.append(manifest)

        # Annotate each corpus record with provenance
        for idx, rec in enumerate(accumulator.corpus):
            # rec["game_id"] already equals i; keep but ensure
            color = rec.get("color", "RED")
            seat_index = prov["color_to_seat"].get(color, 0)
            # trajectory_index is global order within trajectory (decision_id mirrors it but use idx)
            rec["trajectory_id"] = prov["trajectory_id"]
            rec["game_seed"] = prov["game_seed"]
            rec["game_end_turn"] = prov["game_end_turn"]
            rec["seat_index"] = seat_index
            rec["trajectory_index"] = idx
            # Ensure winner consistent
            rec["winner"] = prov["winner"]
        all_high_decisions.extend(accumulator.corpus)
        print(f"Game {i+1}/{num_games} finished. Total high decisions collected: {len(all_high_decisions)}")

        # Intermediate saves (JSONL)
        if (i + 1) % 10 == 0 or i == 0:
            _write_jsonl(output_file, all_high_decisions)
            if manifest_path:
                _write_jsonl(manifest_path, manifests)

    _write_jsonl(output_file, all_high_decisions)
    if manifest_path:
        _write_jsonl(manifest_path, manifests)
    print(f"Finished. Saved {len(all_high_decisions)} moves to {output_file} (JSONL, one record per line)")
    if manifest_path:
        print(f"Wrote trajectory manifest to {manifest_path} ({len(manifests)} trajectories)")


def run_placements_simulation(
    num_games=1000,
    output_file="data/initial_placements/raw/initial_placements.jsonl",
    base_seed: int = 2000,
    manifest_path: Optional[str] = None,
):
    """Run AlphaBeta self-play and collect *only* initial placements.

    Unlike :func:`run_simulation`, this emits every initial-placement decision
    from *all* players (not just the winner) and ignores the high-decision
    filters.  Each game yields 8 records (4 players × 2 placements), with
    settlement+road bundled as a single choice (same prompting as the main
    corpus).

    Args:
        num_games: Number of games to simulate.
        output_file: Path to JSONL output (one record per placement).
    """
    all_placements: List[Dict[str, Any]] = []

    player_instances = [
        CorpusCollectionPlayer(Color.RED, prunning=False),
        CorpusCollectionPlayer(Color.BLUE, prunning=False),
        CorpusCollectionPlayer(Color.ORANGE, prunning=False),
        CorpusCollectionPlayer(Color.WHITE, prunning=False),
    ]

    seat_order = ["RED", "BLUE", "ORANGE", "WHITE"]
    manifests: List[Dict[str, Any]] = []
    if manifest_path is None:
        try:
            from pathlib import Path as _P

            manifest_path = str(_P(output_file).parent / "trajectory_manifest.jsonl")
        except Exception:
            manifest_path = None

    for i in range(num_games):
        for p in player_instances:
            p.reset_state()
        game_seed = base_seed + i
        accumulator = PlacementsAccumulator(player_instances, game_id=i)
        game = Game(player_instances, seed=game_seed, friendly_robber=False)
        game.play(accumulators=[accumulator])

        num_turns = getattr(game.state, "num_turns", None)
        if num_turns is None:
            num_turns = getattr(game.state, "current_turn_index", 0) or 0
        winner_color = game.winning_color()
        prov = _derive_trajectory_provenance(
            game_id=i,
            game_seed=game_seed,
            seat_order=seat_order,
            winner_color=winner_color,
            num_turns=int(num_turns),
        )
        manifest = {
            "trajectory_id": prov["trajectory_id"],
            "game_seed": prov["game_seed"],
            "game_end_turn": prov["game_end_turn"],
            "game_id": i,
            "seat_order": seat_order,
            "winner": prov["winner"],
            "players": [
                {"seat_index": idx, "color": c, "is_winner": 1 if c == prov["winner"] else 0}
                for idx, c in enumerate(seat_order)
            ],
        }
        manifests.append(manifest)
        for idx, rec in enumerate(accumulator.corpus):
            color = rec.get("color", "RED")
            seat_index = prov["color_to_seat"].get(color, 0)
            rec["trajectory_id"] = prov["trajectory_id"]
            rec["game_seed"] = prov["game_seed"]
            rec["game_end_turn"] = prov["game_end_turn"]
            rec["seat_index"] = seat_index
            rec["trajectory_index"] = idx
            rec["winner"] = prov["winner"]
        all_placements.extend(accumulator.corpus)
        print(
            f"Game {i+1}/{num_games} finished. "
            f"Total placements collected: {len(all_placements)} "
            f"(this game: {len(accumulator.corpus)})"
        )

        if (i + 1) % 10 == 0 or i == 0:
            _write_jsonl(output_file, all_placements)
            if manifest_path:
                _write_jsonl(manifest_path, manifests)

    _write_jsonl(output_file, all_placements)
    if manifest_path:
        _write_jsonl(manifest_path, manifests)
    print(f"Finished. Saved {len(all_placements)} placements to {output_file} (JSONL, one record per line)")
    if manifest_path:
        print(f"Wrote trajectory manifest to {manifest_path} ({len(manifests)} trajectories)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect high-decision prompts via LLMObservationAgent.build_full_prompt to JSONL"
    )
    parser.add_argument("num_games", nargs="?", type=int, default=1000, help="Number of games to simulate")
    parser.add_argument("output_file", nargs="?", default="data/sft/high_decision_moves.jsonl", help="Output JSONL path")
    parser.add_argument("-n", "--num-games", type=int, dest="num_games_flag", help="Number of games (flag form)")
    parser.add_argument("-o", "--output", "--out", type=str, dest="output_flag", help="Output JSONL path (flag form)")
    parser.add_argument(
        "--placements",
        "--initial-only",
        action="store_true",
        dest="placements",
        help="Collect only initial placements from all players (8 per game, settlement+road bundled)",
    )
    args = parser.parse_args()

    # Flag forms override positionals if provided
    n = args.num_games_flag if args.num_games_flag is not None else args.num_games
    out = args.output_flag if args.output_flag is not None else args.output_file
    placements_mode = bool(getattr(args, "placements", False))
    # Default placements output is data/initial_placements/raw/initial_placements.jsonl when --placements is set
    # and the user didn't explicitly choose an output path.
    if placements_mode and args.output_flag is None and args.output_file == "data/sft/high_decision_moves.jsonl":
        out = "data/initial_placements/raw/initial_placements.jsonl"

    # Handle swapped args: `script out.jsonl 100` case
    if isinstance(n, str) and n.endswith(".jsonl"):
        # user passed file as first positional where int expected -> argparse would error,
        # but handle the two-positional swapped case manually if parsed as string
        n, out = out, n  # type: ignore

    # If someone passed: `collect_corpus.py 100 --out file.jsonl`, n is 100, out is flag
    # Ensure extension is .jsonl
    if not out.endswith(".jsonl"):
        print(f"Note: output {out} does not end with .jsonl — using as-is, but JSONL is recommended.")
    if placements_mode:
        run_placements_simulation(n, output_file=out)
    else:
        run_simulation(n, output_file=out)
