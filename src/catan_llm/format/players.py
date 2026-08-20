"""
Player formatting — resources and development cards.
"""

from typing import Optional

from catanatron.models.inventory import Inventory
from catanatron.models.public_state import PublicState


def get_player_resources(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """Format player resource information for LLM consumption.

    Args:
        public_state: The public state object from Observation agent
        current_player_color: The color of the current player
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Formatted string representation of player resources
    """
    lines = ["[PLAYER RESOURCES]"]

    # Use public_state for all players' public information
    for color, player_data in public_state.players.items():
        color_name = color.name if hasattr(color, 'name') else str(color)

        # For current player, use detailed inventory if provided
        if color == current_player_color and current_player_inventory is not None:
            # This is the current player - use detailed inventory
            resource_list = []
            if current_player_inventory.wood > 0:
                resource_list.append(f"WOOD: {current_player_inventory.wood}")
            if current_player_inventory.brick > 0:
                resource_list.append(f"BRICK: {current_player_inventory.brick}")
            if current_player_inventory.sheep > 0:
                resource_list.append(f"SHEEP: {current_player_inventory.sheep}")
            if current_player_inventory.wheat > 0:
                resource_list.append(f"WHEAT: {current_player_inventory.wheat}")
            if current_player_inventory.ore > 0:
                resource_list.append(f"ORE: {current_player_inventory.ore}")
            lines.append(f"- {color_name}: {', '.join(resource_list) if resource_list else 'No resources'}")
        else:
            # For other players, only show public information (hand count)
            hand_count = player_data.hand_resource_count
            lines.append(f"- {color_name}: {hand_count} resource cards (hidden)")

    return "\n".join(lines)


def get_player_dev_cards(public_state: PublicState, current_player_color, current_player_inventory: Optional[Inventory] = None) -> str:
    """Format player development card information for LLM consumption.

    Args:
        public_state: The public state object from Observation agent
        current_player_color: The color of the current player
        current_player_inventory: Optional Inventory object for current player

    Returns:
        str: Formatted string representation of player development cards
    """
    lines = ["[PLAYER DEVELOPMENT CARDS]"]

    # Use public_state for all players' public information
    for color, player_data in public_state.players.items():
        color_name = color.name if hasattr(color, 'name') else str(color)

        # For current player, use detailed inventory if provided
        if color == current_player_color and current_player_inventory is not None:
            # This is the current player - use detailed inventory
            card_list = []
            if current_player_inventory.knight > 0:
                card_list.append(f"KNIGHT: {current_player_inventory.knight}")
            if current_player_inventory.year_of_plenty > 0:
                card_list.append(f"YEAR_OF_PLENTY: {current_player_inventory.year_of_plenty}")
            if current_player_inventory.monopoly > 0:
                card_list.append(f"MONOPOLY: {current_player_inventory.monopoly}")
            if current_player_inventory.road_building > 0:
                card_list.append(f"ROAD_BUILDING: {current_player_inventory.road_building}")
            if current_player_inventory.victory_point > 0:
                card_list.append(f"VICTORY_POINT: {current_player_inventory.victory_point}")
            
            # Add played cards (public information)
            played_list = []
            if player_data.played_knight > 0:
                played_list.append(f"KNIGHT: {player_data.played_knight}")
            if player_data.played_year_of_plenty > 0:
                played_list.append(f"YEAR_OF_PLENTY: {player_data.played_year_of_plenty}")
            if player_data.played_monopoly > 0:
                played_list.append(f"MONOPOLY: {player_data.played_monopoly}")
            if player_data.played_road_building > 0:
                played_list.append(f"ROAD_BUILDING: {player_data.played_road_building}")
            if player_data.played_victory_point > 0:
                played_list.append(f"VICTORY_POINT: {player_data.played_victory_point}")
            
            # Combine held and played cards
            held_str = ', '.join(card_list) if card_list else 'No dev cards'
            played_str = ', '.join(played_list) if played_list else None
            
            if played_str:
                lines.append(f"- {color_name}: {held_str} (Played: {played_str})")
            else:
                lines.append(f"- {color_name}: {held_str}")
        else:
            # For other players, show played cards (public) and hidden count for held cards
            hand_count = player_data.hand_dev_count
            
            # Add played cards (public information)
            played_list = []
            if player_data.played_knight > 0:
                played_list.append(f"KNIGHT: {player_data.played_knight}")
            if player_data.played_year_of_plenty > 0:
                played_list.append(f"YEAR_OF_PLENTY: {player_data.played_year_of_plenty}")
            if player_data.played_monopoly > 0:
                played_list.append(f"MONOPOLY: {player_data.played_monopoly}")
            if player_data.played_road_building > 0:
                played_list.append(f"ROAD_BUILDING: {player_data.played_road_building}")
            if player_data.played_victory_point > 0:
                played_list.append(f"VICTORY_POINT: {player_data.played_victory_point}")
            
            played_str = ', '.join(played_list) if played_list else None
            
            if played_str:
                lines.append(f"- {color_name}: {hand_count} dev cards (hidden) (Played: {played_str})")
            else:
                lines.append(f"- {color_name}: {hand_count} dev cards (hidden)")

    return "\n".join(lines)
