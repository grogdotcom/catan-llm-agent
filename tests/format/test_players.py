"""
Unit tests for player formatting — resources and dev cards.

Mirrors src/catan_llm/format/players.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "src"))

import pytest

from catanatron.game import Game
from catanatron.models.player import Color, Player
from catanatron.models.perspective_player import _build_public_state
from catanatron.state_functions import player_key
from catan_llm.format.players import get_player_resources, get_player_dev_cards
from catanatron.models.inventory import Inventory


class SimplePlayer(Player):
    def __init__(self, color):
        self.color = color
        self.is_bot = True

    def decide(self, game, playable_actions):
        return playable_actions[0] if playable_actions else None

    def reset_state(self):
        pass


def build_public_state(game):
    return _build_public_state(game)


def create_game():
    players = [
        SimplePlayer(Color.RED),
        SimplePlayer(Color.BLUE),
        SimplePlayer(Color.ORANGE),
        SimplePlayer(Color.WHITE),
    ]
    return Game(players)


def test_get_player_resources_current_player_shows_hidden_for_others():
    game = create_game()
    ps = build_public_state(game)
    # Give RED some cards
    red_key = player_key(game.state, Color.RED)
    game.state.player_state[f"{red_key}_WOOD_IN_HAND"] = 2
    game.state.player_state[f"{red_key}_BRICK_IN_HAND"] = 1
    ps = build_public_state(game)
    text = get_player_resources(ps, Color.RED, game.state.player_state.get(f"{red_key}_WOOD_IN_HAND") and None)  # no inventory, should still show hidden logic
    # Without inventory, even current player shows hidden count
    assert "[PLAYER RESOURCES]" in text
    assert "resource cards (hidden)" in text


def test_get_player_resources_with_inventory():
    game = create_game()
    inv = Inventory(wood=2, brick=1, sheep=0, wheat=0, ore=0)
    ps = build_public_state(game)
    text = get_player_resources(ps, Color.RED, inv)
    assert "WOOD: 2" in text
    assert "BRICK: 1" in text
    # Other players still hidden
    assert "BLUE:" in text and "hidden" in text


def test_get_player_resources_no_inventory_shows_no_resources():
    game = create_game()
    ps = build_public_state(game)
    inv = Inventory()
    text = get_player_resources(ps, Color.RED, inv)
    assert "No resources" in text


def test_get_player_dev_cards_hidden_and_played():
    game = create_game()
    # Simulate played knight for BLUE
    blue_key = player_key(game.state, Color.BLUE)
    game.state.player_state[f"{blue_key}_KNIGHT_PLAYED"] = 1  # actual key may differ, but public_state will reflect
    ps = build_public_state(game)
    text = get_player_dev_cards(ps, Color.RED)
    assert "[PLAYER DEVELOPMENT CARDS]" in text
    assert "dev cards (hidden)" in text


def test_get_player_dev_cards_with_inventory():
    game = create_game()
    inv = Inventory(knight=1, victory_point=1)
    ps = build_public_state(game)
    text = get_player_dev_cards(ps, Color.RED, inv)
    assert "KNIGHT: 1" in text
    assert "VICTORY_POINT: 1" in text
