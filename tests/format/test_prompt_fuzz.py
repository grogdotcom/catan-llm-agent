"""
Fuzz invariants for the assembled LLM prompt.

Runs 15 seeded games (SimplePlayer picks first legal move) and at every
decision asserts the 7-section structure, seating order, and phase-appropriate
move shape that must hold for *every* prompt regardless of board/seed.

This is the integration counterpart to the exact-string goldens — it doesn't
assert a literal, it asserts that cross-module wiring (board vs players vs
history vs moves) never drifts.
"""

import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from catanatron.game import Game
from catanatron.models.player import Color, Player
from catanatron.models.perspective_player import _build_public_state, _build_inventory, _sanitize_history, _build_pending_trades
from catanatron.features import create_sample
from catanatron.models.observation import Observation
from catanatron.models.enums import ActionPrompt

from catan_llm.format import build_moves
from catan_llm.format.board import gather_board_occupancy_data
from catan_llm.format.players import get_players_summary
from catan_llm.format.prompts import get_complete_prompt, format_observation_prompt
from catan_llm.format.utils import _name_of


class S(Player):
    def __init__(self, c):
        self.color = c
        self.is_bot = True
    def decide(self, g, a):
        return a[0]
    def reset_state(self):
        pass


def _assert_seven_sections(prompt: str):
    """7-section ordering, each header appears exactly once."""
    # Order: board < occupancy < robber < header < players < history < moves
    # Header is [CURRENT PLAYER] when present; players is [PLAYERS] (or [PLAYERS] - INITIAL SETUP)
    assert "[FULL BOARD MAP" in prompt
    assert "[CURRENT BOARD OCCUPANCY" in prompt
    assert "ROBBER:" in prompt
    assert "[PLAYERS" in prompt
    assert "[RECENT TURNS" in prompt
    assert "[PLAYABLE MOVES" in prompt
    # Exactly once per section
    assert prompt.count("[FULL BOARD MAP") == 1
    assert prompt.count("[CURRENT BOARD OCCUPANCY") == 1
    assert prompt.count("ROBBER:") == 1
    assert prompt.count("[PLAYERS") == 1
    assert prompt.count("[RECENT TURNS") == 1
    assert prompt.count("[PLAYABLE MOVES") == 1
    # Ordering
    assert prompt.index("[FULL BOARD MAP") < prompt.index("[CURRENT BOARD OCCUPANCY")
    assert prompt.index("[CURRENT BOARD OCCUPANCY") < prompt.index("ROBBER:")
    # Header [CURRENT PLAYER] may be absent when include_header=False, but our caller includes it
    if "[CURRENT PLAYER" in prompt:
        assert prompt.index("ROBBER:") < prompt.index("[CURRENT PLAYER")
        assert prompt.index("[CURRENT PLAYER") < prompt.index("[PLAYERS")
    else:
        assert prompt.index("ROBBER:") < prompt.index("[PLAYERS")
    assert prompt.index("[PLAYERS") < prompt.index("[RECENT TURNS")
    assert prompt.index("[RECENT TURNS") < prompt.index("[PLAYABLE MOVES")


def test_prompt_invariant_fuzz_50_games():
    # 50 seeds gives ~500-800 prompts, ~10-15s with SimplePlayer (no search)
    for seed in range(50):
        random.seed(seed)
        players = [S(Color.RED), S(Color.BLUE), S(Color.ORANGE), S(Color.WHITE)]
        game = Game(players, seed=seed)
        # Step decision by decision; capture prompt at each decision
        steps = 0
        while game.winning_color() is None and steps < 400:
            playable = game.playable_actions
            if not playable:
                break
            color = game.state.current_color()
            public_state = _build_public_state(game)
            inventory = _build_inventory(game, color)
            obs = Observation(
                color=color,
                features=create_sample(game, color),
                public_history=tuple(_sanitize_history(game, color)),
                current_prompt=game.state.current_prompt,
                pending_trades=_build_pending_trades(game),
                public_state=public_state,
                inventory=inventory,
            )
            # Build prompt via the two public entry points — both must satisfy invariants
            prompt_via_complete = get_complete_prompt(
                public_state=public_state,
                current_player_color=color,
                playable_actions=playable,
                current_player_inventory=inventory,
                observation=obs,
                current_prompt=game.state.current_prompt,
                turn_number=game.state.num_turns,
                include_header=True,
                include_footer=True,
            )
            prompt_via_obs = format_observation_prompt(obs, playable, inventory)

            for prompt in (prompt_via_complete, prompt_via_obs):
                # 1. No crash is implicit — we got here
                assert isinstance(prompt, str) and len(prompt) > 200
                assert len(prompt) < 80000  # budget sanity — discard with huge hand can hit ~30k
                # 2. 7-section ordering
                _assert_seven_sections(prompt)
                # 3. Seating order: occupancy vs players vs state.colors must match 1st→2nd→3rd→4th
                occ = gather_board_occupancy_data(public_state)
                occ_order = [p.color for p in occ.players]
                # get_players_summary seating order
                summary = get_players_summary(public_state, color, inventory, current_prompt=game.state.current_prompt)
                expected_order = [c.name for c in game.state.colors]
                # Occupancy must always match seating
                assert occ_order == expected_order, (
                    f"seed {seed} step {steps} color {color} occ {occ_order} expected {expected_order}"
                )
                # Players summary: when initial collapsed "All players start..." there are no per-player lines
                if "All players start" in summary:
                    assert _name_of(color) in summary  # YOU marked in collapsed line
                else:
                    summary_order = []
                    for line in summary.splitlines():
                        if line.startswith("- "):
                            name = line.split()[1].rstrip(":")
                            summary_order.append(name)
                    assert summary_order == expected_order, (
                        f"seed {seed} step {steps} color {color} players {summary_order} expected {expected_order}"
                    )
                # 5. No leakage: when Resources line is present, opponents must be hidden
                # Collapsed "[PLAYERS] - INITIAL SETUP / All players start..." and
                # BUILD_INITIAL_ROAD with no starting resources omit Resources entirely
                if "Resources:" in prompt:
                    assert "hidden" in prompt or "No resources" in prompt

                # 4. Phase-appropriate move shape (not exact count, just shape)
                moves = build_moves(playable, obs)
                assert len(moves) >= 1
                assert "[PLAYABLE MOVES" in prompt
                phase_name = game.state.current_prompt.name if hasattr(game.state.current_prompt, "name") else str(game.state.current_prompt)
                if phase_name == "BUILD_INITIAL_SETTLEMENT":
                    # Bundled placement header and starting-resources hint
                    assert "[PLAYABLE MOVES - INITIAL PLACEMENT]" in prompt
                elif phase_name == "DISCARD":
                    # Bundled discard shows "Discard" and distinct combos, not single-card
                    assert "Discard" in prompt
                elif phase_name == "MOVE_ROBBER":
                    assert "Tile" in prompt and ("steal" in prompt or "no steal" in prompt)

            # Advance game with deterministic first-move policy
            cur = game.state.current_color()
            cur_player = next(p for p in players if p.color == cur)
            action = cur_player.decide(game, playable)
            game.execute(action)
            steps += 1
        # Ensure we made progress and didn't infinite loop
        assert steps > 5
