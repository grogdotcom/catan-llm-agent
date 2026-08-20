"""
Format models — shared data structures for the formatting package.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from catanatron.models.enums import ActionType


# Initial placement is only BUILD_SETTLEMENT / BUILD_ROAD until the first
# non-setup action (typically ROLL). After that, those build types belong
# to normal turns.
_SETUP_ACTION_TYPES = frozenset({ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD})


@dataclass
class AdjacentHexInfo:
    """Data class for adjacent hex information."""
    resource: str
    roll: Optional[int]
    pips: int
    tile_id: Optional[int] = None


@dataclass
class BuildingInfo:
    """Data class for building (settlement or city) information."""
    node_id: int
    adjacent_hexes: List[AdjacentHexInfo]
    port: Optional[str] = None

    total_pips: int = field(init=False)

    def __post_init__(self):
        self.total_pips = sum([hex.pips for hex in self.adjacent_hexes])


@dataclass
class PlayerBoardData:
    """Data class for player's board occupancy information."""
    color: str
    settlements: List[BuildingInfo]
    cities: List[BuildingInfo]
    roads: List[tuple]  # List of (node1, node2) tuples


@dataclass
class BoardOccupancyData:
    """Data class for board occupancy information."""
    players: List[PlayerBoardData]
