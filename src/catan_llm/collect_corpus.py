import json
import random
import time
import os
from typing import List, Dict, Any, Tuple
from catanatron.game import Game, GameAccumulator
from catanatron.players.minimax import AlphaBetaPlayer, DebugStateNode
from catanatron.models.player import Color
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, DEVELOPMENT_CARDS
from catanatron.json import GameEncoder
from catanatron.models.perspective_player import _build_public_state
from catanatron.state_functions import player_key

from collections import defaultdict
from catan_llm.game_formatter import (
    get_full_board_map,
    get_board_occupancy,
    get_player_resources,
    get_player_dev_cards,
    get_game_state_summary,
    summarize_catan_actions,
    format_decision_prompt,
)

class CorpusCollectionPlayer(AlphaBetaPlayer):
    def __init__(self, color, depth=2, prunning=True):
        super().__init__(color, depth=depth, prunning=prunning)
        self.decisions = []

    def decide(self, game: Game, playable_actions):
        # We need the full list of actions even if only one is available for high-decision tracking
        actions = self.get_actions(game)

        if len(actions) == 1:
            decision = {
                "state": game.copy(),
                "actions": {actions[0]: 0.0},
                "selected": actions[0]
            }
            self.decisions.append(decision)
            return actions[0]

        # Use AlphaBeta search to get values
        start = time.time()
        state_id = str(len(game.state.action_records))
        node = DebugStateNode(state_id, self.color)
        deadline = start + 20 # 20 seconds max
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline, node
        )

        # Capture all children actions and their expected values
        action_values = {}
        for action_node in node.children:
            action_values[action_node.action] = action_node.expected_value

        selected_action = result[0] if result[0] is not None else playable_actions[0]

        self.decisions.append({
            "state": game.copy(),
            "actions": action_values,
            "selected": selected_action
        })

        return selected_action

    def reset_state(self):
        super().reset_state()
        self.decisions = []

class CorpusAccumulator(GameAccumulator):
    def __init__(self, players):
        self.players = players
        self.corpus = []

    def after(self, game):
        winner_color = game.winning_color()
        if winner_color is None:
            return

        winner_player = next(p for p in self.players if p.color == winner_color)
        if not isinstance(winner_player, CorpusCollectionPlayer):
            return

        decisions = winner_player.decisions

        i = 0
        while i < len(decisions):
            decision = decisions[i]
            selected = decision["selected"]
            action_values = decision["actions"]
            state_before = decision["state"]

            is_high_decision = False

            # 1. Initial placements
            is_initial = state_before.state.current_prompt in [
                ActionPrompt.BUILD_INITIAL_SETTLEMENT,
                ActionPrompt.BUILD_INITIAL_ROAD
            ]

            # 2. Moving robber
            is_robber = selected.action_type == ActionType.MOVE_ROBBER

            # 3. Building
            is_build = selected.action_type in [
                ActionType.BUILD_ROAD,
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY
            ]

            # 4. Ambiguous (difference < 0.05)
            if len(action_values) > 1:
                sorted_values = sorted(action_values.values(), reverse=True)
                if sorted_values[0] - sorted_values[1] < 0.05:
                    is_high_decision = True

            if is_initial or is_robber or is_build or is_high_decision:
                # Group initial placements: Settlement + next Road
                if state_before.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
                    next_initial_road = None
                    if i + 1 < len(decisions):
                        potential_road_decision = decisions[i+1]
                        if potential_road_decision["state"].state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
                            next_initial_road = {
                                "playable_actions": list(potential_road_decision["actions"].keys()),
                                "selected_action": potential_road_decision["selected"]
                            }
                            i += 1 # Skip next in loop

                    self.corpus.append({
                        "state": state_before,
                        "playable_actions": list(action_values.keys()),
                        "selected_action": selected,
                        "next_initial_road": next_initial_road,
                    })
                else:
                    self.corpus.append({
                        "state": state_before,
                        "playable_actions": list(action_values.keys()),
                        "selected_action": selected,
                    })
            i += 1


def run_simulation(num_games=1000, output_file="high_decision_moves.json"):
    all_high_decisions = []

    # Pre-create players to reuse them (they reset state)
    player_instances = [
        CorpusCollectionPlayer(Color.RED),
        CorpusCollectionPlayer(Color.BLUE),
        CorpusCollectionPlayer(Color.ORANGE),
        CorpusCollectionPlayer(Color.WHITE),
    ]

    for i in range(num_games):
        for p in player_instances:
            p.reset_state()

        accumulator = CorpusAccumulator(player_instances)
        game = Game(player_instances)
        game.play(accumulators=[accumulator])

        all_high_decisions.extend(accumulator.corpus)
        print(f"Game {i+1}/{num_games} finished. Total high decisions collected: {len(all_high_decisions)}")

        # Intermediate saves
        if (i + 1) % 10 == 0 or i == 0:
            with open(output_file, "w") as f:
                json.dump(all_high_decisions, f, cls=GameEncoder)

    with open(output_file, "w") as f:
        json.dump(all_high_decisions, f, cls=GameEncoder)
    print(f"Finished. Saved {len(all_high_decisions)} moves to {output_file}")

if __name__ == "__main__":
    import sys
    n = 1000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_simulation(n)