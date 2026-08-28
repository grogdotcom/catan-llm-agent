"""Exact-string tests for utils — covers _abbr_resource enum, _format_maritime_trade_value empty."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from catanatron.models.enums import WOOD
from catan_llm.format.utils import _abbr_resource, _format_maritime_trade_value, _name_of, _format_resource_counts, _format_trade_offer_value, _format_coordinate, get_pip_count

def test_abbr_resource_enum_with_name():
    # Pass an enum-like object with .name attribute
    class E:
        name = "wood"
    assert _abbr_resource(E()) == "Wd"
    assert _abbr_resource("3:1") == "3:1"
    assert _abbr_resource(None) == "None"
    assert _abbr_resource("WOOD") == "Wd"
    assert _abbr_resource("unknown") == "unknown"

def test_abbr_resource_color_enum():
    from catanatron.models.player import Color
    # Color enum not in RESOURCE_ABBR -> falls through to str(name) -> "Color.RED"
    assert _abbr_resource(Color.RED) == "Color.RED"
    # But after upper -> "RED" check fails, it tries hasattr .name -> "RED" -> not in RESOURCE_ABBR, returns str
    assert _name_of(Color.BLUE) == "BLUE"
    assert _name_of(None) == "None"

def test_format_maritime_empty_giving():
    # No resources given -> "gives [nothing] to bank for Wd"
    assert _format_maritime_trade_value([None, None, None, None, WOOD]) == "gives [nothing] to bank for Wd"

def test_format_maritime_with_giving():
    assert _format_maritime_trade_value([WOOD, WOOD, WOOD, WOOD, WOOD]) == "gives [4 Wd] to bank for Wd"

def test_get_pip_count():
    assert get_pip_count(None) == 0
    assert get_pip_count(7) == 0
    assert get_pip_count(6) == 5

def test_format_resource_counts():
    from catanatron.models.enums import RESOURCES
    assert _format_resource_counts([0,0,0,0,0]) == "nothing"
    assert _format_resource_counts([1,0,2,0,0]) == "1 Wd, 2 Sh"

def test_format_trade_offer_value():
    assert "offers" in _format_trade_offer_value([WOOD, None, None, None, None, WOOD, None, None, None, None])

def test_format_coordinate():
    assert _format_coordinate(None) == "(unknown)"
    assert _format_coordinate((1,2,3)) == "(1, 2, 3)"
