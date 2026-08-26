"""
Unit tests for shared formatting utilities.

Mirrors src/catan_llm/format/utils.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from catan_llm.format.utils import get_pip_count, _abbr_resource, _name_of


def test_get_pip_count():
    """Exact pip values per roll — mirrors engine's dot count."""
    assert get_pip_count(2) == 1
    assert get_pip_count(3) == 2
    assert get_pip_count(4) == 3
    assert get_pip_count(5) == 4
    assert get_pip_count(6) == 5
    assert get_pip_count(8) == 5
    assert get_pip_count(9) == 4
    assert get_pip_count(10) == 3
    assert get_pip_count(11) == 2
    assert get_pip_count(12) == 1
    assert get_pip_count(7) == 0
    assert get_pip_count(None) == 0


def test_abbr_resource():
    assert _abbr_resource("WOOD") == "Wd"
    assert _abbr_resource("BRICK") == "Br"
    assert _abbr_resource("SHEEP") == "Sh"
    assert _abbr_resource("WHEAT") == "Wh"
    assert _abbr_resource("ORE") == "Or"
    assert _abbr_resource("3:1") == "3:1"
    assert _abbr_resource(None) == "None"


def test_name_of():
    class Fake:
        name = "RED"
    assert _name_of(Fake()) == "RED"
    assert _name_of(None) == "None"
    assert _name_of("WOOD") == "WOOD"
