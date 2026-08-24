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


RESOURCE_ABBR = {
    "WOOD": "Wd",
    "BRICK": "Br",
    "SHEEP": "Sh",
    "WHEAT": "Wh",
    "ORE": "Or",
}

RESOURCE_ABBR_INV = {v: k for k, v in RESOURCE_ABBR.items()}


def _abbr_resource(name: str) -> str:
    """Abbreviate a resource name: WOOD->Wd etc. Pass-through for unknown."""
    if name is None:
        return "None"
    upper = str(name).upper()
    # Handle already-abbreviated or 3:1 port
    if upper == "3:1":
        return "3:1"
    if upper in RESOURCE_ABBR:
        return RESOURCE_ABBR[upper]
    # Try name attr
    if hasattr(name, "name"):
        inner = str(name.name).upper()
        if inner in RESOURCE_ABBR:
            return RESOURCE_ABBR[inner]
    return str(name)


def _name_of(value: Any) -> str:
    """Return a stable display name for enums/colors/resources."""
    if value is None:
        return "None"
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _format_resource_counts(counts: Sequence[Any], resources: Sequence[str] = RESOURCES) -> str:
    """Format parallel resource counts as '2 Wd, 1 Br' (skip zeros) abbreviated."""
    parts = []
    for resource, count in zip(resources, counts):
        if count:
            parts.append(f"{count} {_abbr_resource(_name_of(resource))}")
    return ", ".join(parts) if parts else "nothing"


def _format_trade_offer_value(value: Sequence[Any]) -> str:
    """Format an OFFER/ACCEPT/REJECT 10-tuple as offered -> asking."""
    offered = _format_resource_counts(value[:5])
    asking = _format_resource_counts(value[5:10])
    return f"offers [{offered}] for [{asking}]"


def _format_maritime_trade_value(value: Sequence[Any]) -> str:
    """Format a MARITIME_TRADE 5-tuple (given..., received). Abbreviated."""
    giving = [r for r in value[:4] if r is not None]
    receiving = value[4]
    give_str = ", ".join(_abbr_resource(_name_of(r)) for r in giving) if giving else "nothing"
    return f"gives [{give_str}] to bank for {_abbr_resource(_name_of(receiving))}"


def _format_coordinate(coordinate) -> str:
    """Render a cube coordinate as a compact string."""
    if coordinate is None:
        return "(unknown)"
    return f"({coordinate[0]}, {coordinate[1]}, {coordinate[2]})"
