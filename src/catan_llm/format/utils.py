"""
Shared formatting utilities.

Pure helpers used across board / players / history / moves.
"""

from typing import Any, Sequence

from catanatron.models.enums import RESOURCES


def get_pip_count(roll_num) -> int:
    """Calculate pip count from roll number."""
    if roll_num is None:
        return 0
    pip_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    return pip_map.get(roll_num, 0)


def _name_of(value: Any) -> str:
    """Return a stable display name for enums/colors/resources."""
    if value is None:
        return "None"
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _format_resource_counts(counts: Sequence[Any], resources: Sequence[str] = RESOURCES) -> str:
    """Format parallel resource counts as '2 WOOD, 1 BRICK' (skip zeros)."""
    parts = []
    for resource, count in zip(resources, counts):
        if count:
            parts.append(f"{count} {_name_of(resource)}")
    return ", ".join(parts) if parts else "nothing"


def _format_trade_offer_value(value: Sequence[Any]) -> str:
    """Format an OFFER/ACCEPT/REJECT 10-tuple as offered -> asking."""
    offered = _format_resource_counts(value[:5])
    asking = _format_resource_counts(value[5:10])
    return f"offers [{offered}] for [{asking}]"


def _format_maritime_trade_value(value: Sequence[Any]) -> str:
    """Format a MARITIME_TRADE 5-tuple (given..., received)."""
    giving = [r for r in value[:4] if r is not None]
    receiving = value[4]
    give_str = ", ".join(_name_of(r) for r in giving) if giving else "nothing"
    return f"gives [{give_str}] to bank for {_name_of(receiving)}"


def _format_coordinate(coordinate) -> str:
    """Render a cube coordinate as a compact string."""
    if coordinate is None:
        return "(unknown)"
    return f"({coordinate[0]}, {coordinate[1]}, {coordinate[2]})"
