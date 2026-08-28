"""
Exact-string golden tests for players.py uncovered branches.

Covers all previously missing branch lines:
151->153,153->155,156,176,190,192,200,202,204,209,228,241,259,
264-268,283,304->303,309->308,312,316,399-400,403,406-407,409,411,413,415,432-461

Each test asserts exact multiline output (== literal) showing the precise
Resources/Dev/VP/Roads/Army/Ports/Pips/Pieces line for each player in seating order.
"""

from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE
from catanatron.models.inventory import Inventory
from catanatron.models.player import Color
from catanatron.models.public_state import PublicBoard, PublicMap, PublicPlayer, PublicState

from catan_llm.format.players import (
    _abbr_dev,
    _format_army_for_overview,
    _format_pieces_for_overview,
    _format_pips_for_overview,
    _format_ports_for_overview,
    _plural,
    _format_resources_for_overview,
    _format_dev_for_overview,
    _format_vp_for_overview,
    _format_road_for_overview,
    get_players_summary,
)
from catan_llm.format.models import AdjacentHexInfo, BuildingInfo, PlayerBoardData


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/format/test_players.py)
# ---------------------------------------------------------------------------

def create_public_player(**kwargs):
    defaults = {
        "public_vps": 0,
        "has_army": False,
        "has_road": False,
        "longest_road_length": 0,
        "roads_left": 15,
        "settlements_left": 5,
        "cities_left": 4,
        "has_rolled": False,
        "hand_resource_count": 0,
        "hand_dev_count": 0,
        "played_knight": 0,
        "played_monopoly": 0,
        "played_road_building": 0,
        "played_year_of_plenty": 0,
        "played_victory_point": 0,
    }
    defaults.update(kwargs)
    return PublicPlayer(**defaults)


def create_public_map(**kwargs):
    defaults = {
        "tiles": {},
        "tile_coordinates": {},
        "ports": {},
        "adjacent_tiles": {},
        "land_nodes": frozenset(),
    }
    defaults.update(kwargs)
    return PublicMap(**defaults)


def create_public_board(**kwargs):
    defaults = {
        "buildings": {},
        "roads": {},
        "robber_tile_id": 0,
        "longest_road_color": None,
        "longest_road_length": 0,
        "map": create_public_map(),
    }
    defaults.update(kwargs)
    return PublicBoard(**defaults)


def create_public_state(players, **kwargs):
    defaults = {"board": create_public_board(), "players": players}
    defaults.update(kwargs)
    return PublicState(**defaults)


def create_inventory(**kwargs):
    defaults = {
        "wood": 0,
        "brick": 0,
        "sheep": 0,
        "wheat": 0,
        "ore": 0,
        "knight": 0,
        "year_of_plenty": 0,
        "monopoly": 0,
        "road_building": 0,
        "victory_point": 0,
        "actual_vps": 0,
        "has_played_development_card": False,
    }
    defaults.update(kwargs)
    return Inventory(**defaults)


class FakePrompt:
    def __init__(self, name):
        self.name = name


# ---------------------------------------------------------------------------
# 151-160: _format_resources_for_overview per-resource 0 vs >0 + hidden
# ---------------------------------------------------------------------------

def test_resources_wood_and_brick_only():
    """RED is YOU with Wd+Br; others hidden; tests 151 true, 153 true, 156 false."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(hand_resource_count=7),
            Color.ORANGE: create_public_player(hand_resource_count=0),
            Color.WHITE: create_public_player(hand_resource_count=3),
        }
    )
    inv = create_inventory(wood=2, brick=1)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: Wd:2, Br:1 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 7c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 3c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_resources_sheep_only():
    """Covers 156 true (sheep) with others zero."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(sheep=3)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: Sh:3 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_resources_wheat_only():
    """Covers wheat true branch."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(wheat=2)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: Wh:2 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_resources_ore_only():
    """Covers ore true branch."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(ore=4)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: Or:4 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_resources_hidden_and_no_resources():
    """Observer has empty inventory -> No resources; others hidden varying counts."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(hand_resource_count=2),
            Color.ORANGE: create_public_player(hand_resource_count=0),
            Color.WHITE: create_public_player(hand_resource_count=1),
        }
    )
    inv = create_inventory()  # empty -> No resources
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 2c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 1c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_resources_opponent_hidden_when_no_inventory():
    """Current player inventory is None -> even YOU shows hidden (covers hidden branch)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(hand_resource_count=3),
            Color.BLUE: create_public_player(hand_resource_count=5),
            Color.ORANGE: create_public_player(hand_resource_count=0),
            Color.WHITE: create_public_player(hand_resource_count=1),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 3c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 5c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 1c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


# ---------------------------------------------------------------------------
# 179-214: _format_dev_for_overview per-dev type (knight, YOP, monopoly, RB, VP)
#          both held and played + hidden vs No dev
# ---------------------------------------------------------------------------

def test_dev_held_knight_only_no_played():
    """Held knight only, no played cards -> covers knight held branch (197-198) and No-played path (209)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(knight=2)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: Kn:2 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_held_yop_only():
    """Held YOP only -> covers 200 branch."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(year_of_plenty=1)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: YOP:1 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_held_monopoly_only():
    """Held monopoly only -> covers 202 branch."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(monopoly=3)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: Mo:3 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_held_road_building_only():
    """Held road building only -> covers 204 branch."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(road_building=1)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: RB:1 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_held_vp_only():
    """Held VP only -> covers victory_point held branch (205-206)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(victory_point=2)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: VP:2 | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_held_no_dev_no_played():
    """Held No dev and no played cards -> covers held_str = No dev + played_str None (207-210)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory()  # No dev held
    result = get_players_summary(ps, Color.BLUE, inv)
    # BLUE is YOU but no dev held, no played -> No dev; opponents also 0d hidden (no played)
    # Since has_any_dev is False (no one has any dev at all), Dev column is hidden entirely.
    # To force has_any_dev True we need at least one player to have dev. So this test checks
    # that when no dev anywhere, Dev column is omitted entirely (midgame optimization).
    expected = """[PLAYERS]
- RED: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE (YOU): Resources: No resources | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_held_no_dev_but_played_knight():
    """Held No dev but played knight present -> No dev (Pd:Kn:1). Covers 209 played_str true."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(played_knight=1),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory()  # No held
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: No dev (Pd:Kn:1) | VP: 0 | Roads: 0 | Army: 1 knight | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_all_held_types_plus_all_played_types():
    """All 5 held types + all 5 played types -> hits 190,192,200,202,204 together with 209 combo."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(
                played_knight=1,
                played_year_of_plenty=1,
                played_monopoly=1,
                played_road_building=1,
                played_victory_point=1,
            ),
            Color.BLUE: create_public_player(played_road_building=2),
            Color.ORANGE: create_public_player(hand_dev_count=3),
            Color.WHITE: create_public_player(played_victory_point=2),
        }
    )
    inv = create_inventory(knight=1, year_of_plenty=1, monopoly=1, road_building=1, victory_point=1)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | Dev: Kn:1, YOP:1, Mo:1, RB:1, VP:1 (Pd:Kn:1, YOP:1, Mo:1, RB:1, VP:1) | VP: 0 | Roads: 0 | Army: 1 knight | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden (Pd:RB:2) | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 3d hidden | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden (Pd:VP:2) | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_hidden_with_played_and_without():
    """Hidden count branch for opponents: 3d hidden vs 2d hidden (Pd:Kn:1) etc. Covers 212-215."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(played_knight=1, hand_dev_count=2),
            Color.BLUE: create_public_player(hand_dev_count=1, played_road_building=1),
            Color.ORANGE: create_public_player(hand_dev_count=0),
            Color.WHITE: create_public_player(hand_dev_count=3, played_monopoly=1, played_victory_point=1),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | Dev: 2d hidden (Pd:Kn:1) | VP: 0 | Roads: 0 | Army: 1 knight | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 1d hidden (Pd:RB:1) | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 3d hidden (Pd:Mo:1, VP:1) | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_dev_played_yop_monopoly_rb_vp_individually():
    """Each played dev type individually visible on opponents to cover 185-192 per-type branches."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(played_year_of_plenty=2),
            Color.ORANGE: create_public_player(played_monopoly=3),
            Color.WHITE: create_public_player(played_road_building=1, played_victory_point=2),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    # has_any_dev true via played cards, army remains 0 so Army column hidden
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden (Pd:YOP:2) | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden (Pd:Mo:3) | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden (Pd:RB:1, VP:2) | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


# ---------------------------------------------------------------------------
# 218-231: hidden VP math (228 guard) etc.
# ---------------------------------------------------------------------------

def test_vp_hidden_positive():
    """actual 7 vs public 5 -> hidden +2."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(public_vps=5),
            Color.BLUE: create_public_player(public_vps=3),
            Color.ORANGE: create_public_player(public_vps=0),
            Color.WHITE: create_public_player(public_vps=2),
        }
    )
    inv = create_inventory(actual_vps=7)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | VP: 7 (5 visible + 2 hidden) | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 3 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 2 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_vp_hidden_zero():
    """actual == public -> shows public only (hidden ==0 branch)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(public_vps=5),
            Color.BLUE: create_public_player(public_vps=4),
            Color.ORANGE: create_public_player(public_vps=0),
            Color.WHITE: create_public_player(public_vps=0),
        }
    )
    inv = create_inventory(actual_vps=5)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | VP: 5 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 4 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_vp_hidden_negative_guard():
    """actual < public -> defensive hidden<0 guard (228) shows actual (public visible)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(public_vps=5),
            Color.BLUE: create_public_player(public_vps=0),
            Color.ORANGE: create_public_player(public_vps=0),
            Color.WHITE: create_public_player(public_vps=0),
        }
    )
    inv = create_inventory(actual_vps=3)
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | VP: 3 (5 visible) | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_vp_opponent_shows_public_only_even_with_inventory():
    """Opponent VP hidden math not revealed; observer inventory only applies to YOU."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(public_vps=3),
            Color.BLUE: create_public_player(public_vps=5),
            Color.ORANGE: create_public_player(public_vps=2),
            Color.WHITE: create_public_player(public_vps=0),
        }
    )
    inv = create_inventory(actual_vps=10)  # should only affect RED line
    result = get_players_summary(ps, Color.RED, inv)
    expected = """[PLAYERS]
- RED (YOU): Resources: No resources | VP: 10 (3 visible + 7 hidden) | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 5 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 2 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


# ---------------------------------------------------------------------------
# 233-255: Longest Road with +2VP vs without (>=5), Army badge, _plural
# ---------------------------------------------------------------------------

def test_roads_longest_badges_and_army_badge():
    """RED has road 4 no +2VP, BLUE has road 6 +2VP, ORANGE road 3 no badge, WHITE 0."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(has_road=True, longest_road_length=4),
            Color.BLUE: create_public_player(has_road=True, longest_road_length=6),
            Color.ORANGE: create_public_player(has_road=False, longest_road_length=3),
            Color.WHITE: create_public_player(has_road=False, longest_road_length=0),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 4 [Longest Road] | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 6 [Longest Road, +2 VP] | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 3 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_army_badge_and_plural():
    """Covers has_army True badge, 1 knight singular vs 2 knights plural (259)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(played_knight=1, has_army=True),
            Color.BLUE: create_public_player(played_knight=2, has_army=False),
            Color.ORANGE: create_public_player(played_knight=0, has_army=False),
            Color.WHITE: create_public_player(played_knight=3, has_army=True),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | Dev: 0d hidden (Pd:Kn:1) | VP: 0 | Roads: 0 | Army: 1 knight [Largest Army, +2 VP] | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Dev: 0d hidden (Pd:Kn:2) | VP: 0 | Roads: 0 | Army: 2 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | VP: 0 | Roads: 0 | Army: 0 knights | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Dev: 0d hidden (Pd:Kn:3) | VP: 0 | Roads: 0 | Army: 3 knights [Largest Army, +2 VP] | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_pieces_plural_branches():
    """Covers _plural and _format_pieces_for_overview branches 259,264-268."""
    # Direct helper checks for coverage of 259 + 264-268
    assert _plural(1, "settlement", "settlements") == "settlement"
    assert _plural(2, "settlement", "settlements") == "settlements"
    assert _plural(1, "city", "cities") == "city"
    assert _plural(0, "city", "cities") == "cities"
    assert _plural(1, "road", "roads") == "road"
    assert _plural(0, "road", "roads") == "roads"

    assert _format_pieces_for_overview(create_public_player(settlements_left=1, cities_left=1, roads_left=1)) == "1 settlement, 1 city, 1 road left"
    assert _format_pieces_for_overview(create_public_player(settlements_left=0, cities_left=0, roads_left=0)) == "0 settlements, 0 cities, 0 roads left"
    assert _format_pieces_for_overview(create_public_player(settlements_left=2, cities_left=4, roads_left=15)) == "2 settlements, 4 cities, 15 roads left"

    # Also via get_players_summary pieces compact form + per-player lines
    ps = create_public_state(
        {
            Color.RED: create_public_player(settlements_left=1, cities_left=1, roads_left=1),
            Color.BLUE: create_public_player(settlements_left=2, cities_left=4, roads_left=15),
            Color.ORANGE: create_public_player(settlements_left=0, cities_left=0, roads_left=0),
            Color.WHITE: create_public_player(settlements_left=5, cities_left=4, roads_left=15),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 1/1/1
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 2/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 0/0/0
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


# ---------------------------------------------------------------------------
# 280,293-316: board_player is None, Ports/ Pips per-resource loops,
#              total==0, per-resource loops 304->303, 309->308, 312, 316
# ---------------------------------------------------------------------------

def test_ports_none_via_summary_and_direct_helpers():
    """Ports None vs helpers with direct _format_ports_for_overview coverage (283)."""
    # Summary with empty board -> all ports None
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected

    # Direct helpers for board_player None and per-port branches
    assert _format_ports_for_overview(None) == "None"
    assert _format_pips_for_overview(None) == "0"
    # Empty PlayerBoardData (no settlements/cities) -> 0
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[], cities=[], roads=[])) == "0"
    # Settlement with no adjacent hexes but present -> total==0 -> 0 (covers 312)
    empty_hex = BuildingInfo(node_id=10, adjacent_hexes=[])
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[empty_hex], cities=[], roads=[])) == "0"
    # Ports deduplication + sorting
    b_wd = BuildingInfo(node_id=10, adjacent_hexes=[], port="WOOD")
    b_wd2 = BuildingInfo(node_id=11, adjacent_hexes=[], port="WOOD")
    b_3to1 = BuildingInfo(node_id=12, adjacent_hexes=[], port="3:1")
    assert _format_ports_for_overview(PlayerBoardData(color="RED", settlements=[b_wd, b_wd2], cities=[], roads=[])) == "Wd"
    assert _format_ports_for_overview(PlayerBoardData(color="RED", settlements=[b_wd, b_3to1], cities=[], roads=[])) == "3:1, Wd"
    assert _format_ports_for_overview(PlayerBoardData(color="RED", settlements=[b_wd], cities=[], roads=[])) == "Wd"
    # Empty port set -> None
    assert _format_ports_for_overview(PlayerBoardData(color="RED", settlements=[empty_hex], cities=[], roads=[])) == "None"
    # _abbr_dev fallback (176)
    assert _abbr_dev("UNKNOWN") == "UNKNOWN"
    assert _abbr_dev("KNIGHT") == "Kn"


def test_ports_wd_and_three_to_one_via_summary():
    """Ports Wd vs 3:1 via board occupancy settlements on port nodes."""
    board = create_public_board(
        buildings={10: (Color.RED, "SETTLEMENT")},
        map=create_public_map(
            tiles={1: (WOOD, 8)},
            adjacent_tiles={10: (1,)},
            ports={0: (WOOD, (10, 11))},
            land_nodes=frozenset([10, 11]),
        ),
    )
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        },
        board=board,
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: Wd | Pips: 5 (Wd:5) | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected

    # 3:1 variant
    board2 = create_public_board(
        buildings={10: (Color.RED, "SETTLEMENT")},
        map=create_public_map(
            tiles={1: (WOOD, 8)},
            adjacent_tiles={10: (1,)},
            ports={0: (None, (10, 11))},
            land_nodes=frozenset([10, 11]),
        ),
    )
    ps2 = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        },
        board=board2,
    )
    result2 = get_players_summary(ps2, Color.RED, None)
    expected2 = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: 3:1 | Pips: 5 (Wd:5) | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result2 == expected2


def test_pips_breakdown_and_total_only_and_zero():
    """Pips 0 vs breakdown (Wd/Br etc) vs total-only when resource not in WOOD..ORE.

    The total-only path (316) is reached when total>0 but no per-resource breakdown,
    e.g. a GOLD tile (resource not in WOOD/BRICK/SHEEP/WHEAT/ORE).
    Direct helper covers that branch; summary covers 0 and breakdown with per-resource loops
    including both settlements (304->303) and cities (309->308) x2 loops.
    """
    # Direct total-only (316) — both settlements and cities paths (covers 304->303 and 309->308 branches)
    b_gold_settlement = BuildingInfo(node_id=10, adjacent_hexes=[AdjacentHexInfo(resource="GOLD", roll=6, pips=3, tile_id=99)])
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[b_gold_settlement], cities=[], roads=[])) == "3"
    b_gold_city = BuildingInfo(node_id=11, adjacent_hexes=[AdjacentHexInfo(resource="GOLD", roll=6, pips=3, tile_id=99)])
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[], cities=[b_gold_city], roads=[])) == "6"
    b_gold_mixed = BuildingInfo(node_id=12, adjacent_hexes=[AdjacentHexInfo(resource="GOLD", roll=6, pips=2, tile_id=99)])
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[b_gold_settlement], cities=[b_gold_mixed], roads=[])) == "7"
    # Direct zero total (312)
    b_zero = BuildingInfo(node_id=10, adjacent_hexes=[])
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[b_zero], cities=[], roads=[])) == "0"
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[], cities=[], roads=[])) == "0"
    assert _format_pips_for_overview(None) == "0"
    # Normal breakdown via direct helper (wood)
    b_wood = BuildingInfo(node_id=10, adjacent_hexes=[AdjacentHexInfo(resource="WOOD", roll=8, pips=5, tile_id=1)])
    assert _format_pips_for_overview(PlayerBoardData(color="RED", settlements=[b_wood], cities=[], roads=[])) == "5 (Wd:5)"

    # Via summary: settlements + cities x2 loops (304,309) produce multi-resource breakdown
    board = create_public_board(
        buildings={10: (Color.RED, "SETTLEMENT"), 11: (Color.RED, "CITY"), 12: (Color.BLUE, "SETTLEMENT")},
        map=create_public_map(
            tiles={1: (WOOD, 8), 2: (BRICK, 6), 3: (SHEEP, 5)},
            adjacent_tiles={10: (1,), 11: (1, 2), 12: (3,)},
            land_nodes=frozenset([10, 11, 12]),
        ),
    )
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        },
        board=board,
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 25 (Wd:15, Br:10) | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 4 (Sh:4) | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected


def test_pips_full_resource_breakdown_all_five():
    """Hit per-resource loop for all 5 resource types (WOOD..ORE) via settlements+cities.

    Ensures the zip(resource_order, abbr_order) loop handles Wd,Br,Sh,Wh,Or all >0 paths.
    """
    board = create_public_board(
        buildings={
            10: (Color.RED, "SETTLEMENT"),
            11: (Color.RED, "CITY"),
            12: (Color.BLUE, "SETTLEMENT"),
            13: (Color.ORANGE, "SETTLEMENT"),
            14: (Color.WHITE, "SETTLEMENT"),
        },
        map=create_public_map(
            tiles={1: (WOOD, 8), 2: (BRICK, 6), 3: (SHEEP, 5), 4: (WHEAT, 9), 5: (ORE, 4)},
            adjacent_tiles={10: (1,), 11: (2,), 12: (3,), 13: (4,), 14: (5,)},
            land_nodes=frozenset([10, 11, 12, 13, 14]),
        ),
    )
    # Give each color distinct pieces so has_any_* stays false but pips vary
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        },
        board=board,
    )
    result = get_players_summary(ps, Color.RED, None)
    expected = """[PLAYERS]
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 15 (Wd:5, Br:10) | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 4 (Sh:4) | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 4 (Wh:4) | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 3 (Or:3) | Pieces: 5/4/15"""
    assert result == expected


# ---------------------------------------------------------------------------
# 399-415,432-461: initial-setup compressed variant has_any_* detection
#                  + all_same collapsed vs per-player compact with Resources/Dev
# ---------------------------------------------------------------------------

def test_initial_all_same_collapsed():
    """All players identical, no has_any_* -> collapsed single line (432-429)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    result = get_players_summary(ps, Color.RED, None, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = "[PLAYERS] - INITIAL SETUP\nAll players start 5/4/15 (S/C/R), 0 VP, 0 pips, no ports/resources — RED (YOU) to place"
    assert result == expected


def test_initial_with_string_prompt_fallback_all_same():
    """String prompt fallback via getattr str path also collapses (BUILD_INITIAL_ROAD)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )

    class StrPrompt:
        def __str__(self):
            return "BUILD_INITIAL_ROAD"

    result = get_players_summary(ps, Color.WHITE, None, current_prompt=StrPrompt())
    expected = "[PLAYERS] - INITIAL SETUP\nAll players start 5/4/15 (S/C/R), 0 VP, 0 pips, no ports/resources — WHITE (YOU) to place"
    assert result == expected


def test_initial_per_player_compact_with_resources():
    """has_any_resources True via observer inventory wood -> per-player compact with Resources (399-400)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(hand_resource_count=0),
            Color.BLUE: create_public_player(hand_resource_count=0),
            Color.ORANGE: create_public_player(hand_resource_count=0),
            Color.WHITE: create_public_player(hand_resource_count=0),
        }
    )
    inv = create_inventory(wood=1)
    result = get_players_summary(ps, Color.RED, inv, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Resources: Wd:1 | Pips: 0 | Ports: None | Pieces: 5/4/15
- BLUE: Resources: 0c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15"""
    assert result == expected


def test_initial_per_player_compact_opponent_resources():
    """has_any_resources True via opponent hand_resource_count>0 (403)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(hand_resource_count=0),
            Color.BLUE: create_public_player(hand_resource_count=3),
            Color.ORANGE: create_public_player(hand_resource_count=0),
            Color.WHITE: create_public_player(hand_resource_count=0),
        }
    )
    result = get_players_summary(ps, Color.RED, None, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Resources: 0c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- BLUE: Resources: 3c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | Pips: 0 | Ports: None | Pieces: 5/4/15"""
    assert result == expected


def test_initial_per_player_compact_with_dev_from_inventory():
    """has_any_dev True via observer inventory (406-407)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    inv = create_inventory(road_building=1)
    result = get_players_summary(ps, Color.RED, inv, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Dev: RB:1 | Pips: 0 | Ports: None | Pieces: 5/4/15
- BLUE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- ORANGE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15
- WHITE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15"""
    assert result == expected


def test_initial_per_player_compact_with_dev_from_played():
    """has_any_dev True via played cards (409) also triggers has_any_army via played_knight."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(played_knight=1),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    result = get_players_summary(ps, Color.RED, None, current_prompt=FakePrompt("BUILD_INITIAL_ROAD"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Dev: 0d hidden (Pd:Kn:1) | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 1 knight
- BLUE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 0 knights
- ORANGE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 0 knights
- WHITE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 0 knights"""
    assert result == expected


def test_initial_has_any_vp_via_public_vps():
    """has_any_vp True when any player public_vps !=0 (411)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(public_vps=2),
            Color.BLUE: create_public_player(public_vps=0),
            Color.ORANGE: create_public_player(public_vps=0),
            Color.WHITE: create_public_player(public_vps=0),
        }
    )
    result = get_players_summary(ps, Color.RED, None, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Pips: 0 | Ports: None | Pieces: 5/4/15 | VP: 2
- BLUE: Pips: 0 | Ports: None | Pieces: 5/4/15 | VP: 0
- ORANGE: Pips: 0 | Ports: None | Pieces: 5/4/15 | VP: 0
- WHITE: Pips: 0 | Ports: None | Pieces: 5/4/15 | VP: 0"""
    assert result == expected


def test_initial_has_any_roads_and_army():
    """has_any_roads via length+has_road (413) and has_any_army via played_knight+has_army (415)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(has_road=True, longest_road_length=4, has_army=True, played_knight=1),
            Color.BLUE: create_public_player(longest_road_length=1),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        }
    )
    result = get_players_summary(ps, Color.RED, None, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Dev: 0d hidden (Pd:Kn:1) | Pips: 0 | Ports: None | Pieces: 5/4/15 | Roads: 4 [Longest Road] | Army: 1 knight [Largest Army, +2 VP]
- BLUE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Roads: 1 | Army: 0 knights
- ORANGE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Roads: 0 | Army: 0 knights
- WHITE: Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Roads: 0 | Army: 0 knights"""
    assert result == expected


def test_initial_per_player_compact_with_pips_diff_no_has_any():
    """Pips differ but no has_any_* -> per-player compact which breaks all_same (432-461 per-player path).

    This exercises the per-player loop when has_any_* are all False but pips differ,
    so the header is still INITIAL SETUP but lines are per-player with Pips/Ports/Pieces only.
    """
    board = create_public_board(
        buildings={10: (Color.RED, "SETTLEMENT")},
        map=create_public_map(
            tiles={1: (WOOD, 8)},
            adjacent_tiles={10: (1,)},
            land_nodes=frozenset([10]),
        ),
    )
    ps = create_public_state(
        {
            Color.RED: create_public_player(),
            Color.BLUE: create_public_player(),
            Color.ORANGE: create_public_player(),
            Color.WHITE: create_public_player(),
        },
        board=board,
    )
    result = get_players_summary(ps, Color.RED, None, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Pips: 5 (Wd:5) | Ports: None | Pieces: 5/4/15
- BLUE: Pips: 0 | Ports: None | Pieces: 5/4/15
- ORANGE: Pips: 0 | Ports: None | Pieces: 5/4/15
- WHITE: Pips: 0 | Ports: None | Pieces: 5/4/15"""
    assert result == expected


def test_initial_per_player_compact_both_resources_and_dev():
    """Both has_any_resources and has_any_dev True -> insertion order of Resources then Dev (432-461)."""
    ps = create_public_state(
        {
            Color.RED: create_public_player(hand_resource_count=0, played_knight=1),
            Color.BLUE: create_public_player(hand_resource_count=0),
            Color.ORANGE: create_public_player(hand_resource_count=0),
            Color.WHITE: create_public_player(hand_resource_count=0),
        }
    )
    inv = create_inventory(wood=1, knight=1)
    result = get_players_summary(ps, Color.RED, inv, current_prompt=FakePrompt("BUILD_INITIAL_SETTLEMENT"))
    expected = """[PLAYERS] - INITIAL SETUP
- RED (YOU): Resources: Wd:1 | Dev: Kn:1 (Pd:Kn:1) | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 1 knight
- BLUE: Resources: 0c hidden | Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 0 knights
- ORANGE: Resources: 0c hidden | Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 0 knights
- WHITE: Resources: 0c hidden | Dev: 0d hidden | Pips: 0 | Ports: None | Pieces: 5/4/15 | Army: 0 knights"""
    assert result == expected


def test_direct_formatters_cover_hidden_and_branch_fallthroughs():
    """Direct calls to guarantee every low-level branch is hit at least once.

    Complements the summary-level goldens above and ensures line coverage for
    helpers when summary would hide branches behind has_any_* optimizations.
    """
    # _format_resources_for_overview hidden paths
    pp = create_public_player(hand_resource_count=5)
    assert _format_resources_for_overview(pp, None, False) == "5c hidden"
    assert _format_resources_for_overview(pp, None, True) == "5c hidden"
    assert _format_resources_for_overview(pp, create_inventory(wood=0), True) == "No resources"
    # Mixed resources (Wd+Sh+Or vs missing Br,Wh)
    mixed = create_inventory(wood=1, sheep=1, ore=1)
    assert _format_resources_for_overview(create_public_player(), mixed, True) == "Wd:1, Sh:1, Or:1"

    # _format_dev_for_overview held + hidden fallthroughs
    assert _format_dev_for_overview(create_public_player(), create_inventory(knight=1), True) == "Kn:1"
    assert _format_dev_for_overview(create_public_player(played_year_of_plenty=1), create_inventory(), True) == "No dev (Pd:YOP:1)"
    assert _format_dev_for_overview(create_public_player(hand_dev_count=2, played_monopoly=1), None, False) == "2d hidden (Pd:Mo:1)"
    assert _format_dev_for_overview(create_public_player(hand_dev_count=0), None, False) == "0d hidden"

    # _format_vp_for_overview all hidden math paths (triggered via direct as well)
    pp_vp = create_public_player(public_vps=5)
    assert _format_vp_for_overview(pp_vp, create_inventory(actual_vps=7), True) == "7 (5 visible + 2 hidden)"
    assert _format_vp_for_overview(pp_vp, create_inventory(actual_vps=5), True) == "5"
    assert _format_vp_for_overview(pp_vp, create_inventory(actual_vps=3), True) == "3 (5 visible)"
    assert _format_vp_for_overview(pp_vp, create_inventory(actual_vps=10), False) == "5"

    # _format_road_for_overview longest with and without +2VP vs no badge
    ps_road = create_public_state({Color.RED: create_public_player(has_road=True, longest_road_length=4)})
    assert _format_road_for_overview(ps_road.players[Color.RED], ps_road, Color.RED) == "4 [Longest Road]"
    ps_road2 = create_public_state({Color.RED: create_public_player(has_road=True, longest_road_length=6)})
    assert _format_road_for_overview(ps_road2.players[Color.RED], ps_road2, Color.RED) == "6 [Longest Road, +2 VP]"
    ps_road3 = create_public_state({Color.RED: create_public_player(has_road=False, longest_road_length=10)})
    assert _format_road_for_overview(ps_road3.players[Color.RED], ps_road3, Color.RED) == "10"

    # _format_army_for_overview plural + badge
    assert _format_army_for_overview(create_public_player(played_knight=1, has_army=True)) == "1 knight [Largest Army, +2 VP]"
    assert _format_army_for_overview(create_public_player(played_knight=2, has_army=False)) == "2 knights"
    assert _format_army_for_overview(create_public_player(played_knight=0, has_army=False)) == "0 knights"

    # _abbr_dev fallback
    assert _abbr_dev("KNIGHT") == "Kn"
    assert _abbr_dev("UNKNOWN_CARD") == "UNKNOWN_CARD"

    # board_player None and pips total==0 / total-only / per-resource loops already asserted above
    # but keep here for unified gate
    assert _format_ports_for_overview(None) == "None"
    assert _format_pips_for_overview(None) == "0"


def test_seating_order_preserved_in_summary():
    """Players appear in seating order (insertion order), not alphabetical; BLUE first if inserted first."""
    # Insert BLUE first, then RED, then WHITE, then ORANGE to prove order is insertion not alphabetical
    ps = create_public_state(
        {
            Color.BLUE: create_public_player(),
            Color.RED: create_public_player(),
            Color.WHITE: create_public_player(),
            Color.ORANGE: create_public_player(),
        }
    )
    result = get_players_summary(ps, Color.RED, None)
    # Order should be BLUE, RED, WHITE, ORANGE matching insertion
    lines = result.splitlines()
    assert lines[1].startswith("- BLUE:")
    assert lines[2].startswith("- RED (YOU):")
    assert lines[3].startswith("- WHITE:")
    assert lines[4].startswith("- ORANGE:")
    expected = """[PLAYERS]
- BLUE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- RED (YOU): Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- WHITE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15
- ORANGE: Resources: 0c hidden | VP: 0 | Roads: 0 | Ports: None | Pips: 0 | Pieces: 5/4/15"""
    assert result == expected
