"""
Prompt formatting — high-level LLM prompts that compose board, player, history and moves.

Contains get_game_state_summary, summarize_catan_actions (legacy),
format_decision_prompt and format_decision_prompt_with_history.
"""

from collections import defaultdict
from typing import List, Optional, Sequence

from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.models.inventory import Inventory
from catanatron.models.public_state import PublicState
from catanatron.models.enums import ActionRecord

from catan_llm.format.board import get_board_occupancy, get_full_board_map
from catan_llm.format.history import format_public_history_window
from catan_llm.format.players import get_player_dev_cards, get_player_resources


def get_game_state_summary(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """
    Create a comprehensive game state summary for LLM consumption.
    Includes board map, occupancy, resources, and development cards.

    Args:
        public_state: The public state object from Observation agent
        current_player_color: The color of the current player
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Comprehensive game state summary
    """
    sections = []

    sections.append(get_full_board_map(public_state))
    sections.append("\n")
    sections.append(get_board_occupancy(public_state))
    sections.append("\n")
    sections.append(get_player_resources(public_state, current_player_color, current_player_inventory))
    sections.append("\n")
    sections.append(get_player_dev_cards(public_state, current_player_color, current_player_inventory))

    return "\n".join(sections)


def summarize_catan_actions(valid_actions: List) -> str:
    """
    Takes a list of Catanatron Action objects and groups them into a clean,
    abstracted summary string for the Teacher LLM prompt.
    """
    if not valid_actions:
        return "No actions available."

    # Group actions by their high-level category
    grouped_actions = defaultdict(list)
    for action in valid_actions:
        action_type = action.action_type

        # 1. Handle Buildings & Board Expansion
        if action_type in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            grouped_actions[action_type.name].append(str(action.coordinate))

        # 2. Handle Maritime & Bank Trades
        elif action_type == ActionType.MARITIME_TRADE:
            try:
                offer, receive = action.kwargs.get('resource_giving'), action.kwargs.get('resource_receiving')
                grouped_actions["MARITIME_TRADE"].append(f"Give {offer.name} -> Get {receive.name}")
            except AttributeError:
                grouped_actions["MARITIME_TRADE"].append(str(action))

        # 3. Handle Robber Movement
        elif action_type == ActionType.MOVE_ROBBER:
            try:
                hex_coord = action.coordinate
                victim = action.kwargs.get('victim_color', 'NONE')
                grouped_actions["MOVE_ROBBER"].append(f"Hex {hex_coord} (Victim: {victim})")
            except AttributeError:
                grouped_actions["MOVE_ROBBER"].append(str(action))

        # 4. Handle Dev Cards & End Turn
        elif action_type == ActionType.END_TURN:
            grouped_actions["END_TURN"].append("Pass")
        else:
            # Catch-all for Development cards (Knight, Monopoly, etc.)
            grouped_actions[action_type.name].append(str(action))

    # Build the final prompt string
    summary_lines = ["[PLAYABLE ACTION CATEGORIES]:"]
    for category, details in grouped_actions.items():
        if category == "END_TURN":
            summary_lines.append("- Pass (End Turn)")
        elif category in ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY"):
            # Collapse coordinates into a single list
            targets = ", ".join(details)
            summary_lines.append(f"- {category}: Target IDs [{targets}]")
        elif category == "MARITIME_TRADE":
            # De-duplicate trades (sometimes the engine offers multiple identical paths)
            unique_trades = sorted(list(set(details)))
            trade_str = ", ".join(unique_trades)
            summary_lines.append(f"- MARITIME_TRADE: Options [{trade_str}]")
        else:
            # For robber and dev cards, list the options cleanly
            options = ", ".join(details)
            summary_lines.append(f"- {category}: Options [{options}]")

    return "\n".join(summary_lines)


def format_decision_prompt(public_state: PublicState, playable_actions: List, current_player_color: str, current_prompt: ActionPrompt, turn_number: int, current_player_inventory: Optional[Inventory] = None) -> str:
    """
    Create a complete decision prompt for LLM consumption.
    Combines game state summary with available actions.

    Args:
        public_state: The public state object from Observation agent
        playable_actions: List of playable actions for the current player
        current_player_color: The color of the current player
        current_prompt: The current action prompt (phase)
        turn_number: The current turn number
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Complete decision prompt for LLM consumption
    """
    prompt_parts = []

    prompt_parts.append(f"[CURRENT PLAYER: {current_player_color}]")
    prompt_parts.append(f"[TURN: {turn_number}]")
    prompt_parts.append(f"[PHASE: {current_prompt.name if hasattr(current_prompt, 'name') else str(current_prompt)}]")
    prompt_parts.append("\n")

    prompt_parts.append(get_game_state_summary(public_state, current_player_color, current_player_inventory))
    prompt_parts.append("\n")

    prompt_parts.append(summarize_catan_actions(playable_actions))
    prompt_parts.append("\n")

    prompt_parts.append("[DECISION REQUIRED]")
    prompt_parts.append("Select the best action from the available options above.")

    return "\n".join(prompt_parts)


def format_decision_prompt_with_history(
    public_state: PublicState,
    playable_actions: List,
    current_player_color: str,
    current_prompt: ActionPrompt,
    turn_number: int,
    public_history: Sequence[ActionRecord],
    history_window_size: Optional[int] = None,
    current_player_inventory: Optional[Inventory] = None,
) -> str:
    """
    Create a complete decision prompt for LLM consumption with public history.
    Combines game state summary, public history (with optional sliding window),
    and available actions.

    Args:
        public_state: The public state object from Observation agent
        playable_actions: List of playable actions for the current player
        current_player_color: The color of the current player
        current_prompt: The current action prompt (phase)
        turn_number: The current turn number
        public_history: Sequence of ActionRecords representing game history
        history_window_size: Optional number of recent turns to include in history.
            If None, includes all turns. If 0, only includes setup phase.
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Complete decision prompt for LLM consumption with history
    """
    prompt_parts = []

    prompt_parts.append(f"[CURRENT PLAYER: {current_player_color}]")
    prompt_parts.append(f"[TURN: {turn_number}]")
    prompt_parts.append(f"[PHASE: {current_prompt.name if hasattr(current_prompt, 'name') else str(current_prompt)}]")
    prompt_parts.append("\n")

    prompt_parts.append(get_game_state_summary(public_state, current_player_color, current_player_inventory))
    prompt_parts.append("\n")

    prompt_parts.append(format_public_history_window(public_history, window_size=history_window_size))
    prompt_parts.append("\n")

    prompt_parts.append(summarize_catan_actions(playable_actions))
    prompt_parts.append("\n")

    prompt_parts.append("[DECISION REQUIRED]")
    prompt_parts.append("Select the best action from the available options above.")

    return "\n".join(prompt_parts)
