"""Phase 3's search bookkeeping (scripts/rr_search.py): the halving schedule, the D8 score, a six's
identity, the shortlist's utility classes. No battles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gauntlet  # noqa: E402
import rr_search as rs  # noqa: E402  (switches gauntlet.GAME to "rr" on import)

gauntlet.GAME = "bdsp"  # the other tests run the BDSP game


def test_schedule_halves_the_field_and_doubles_the_battles():
    sizes, ns = rs.schedule(8, 120)
    assert sizes == [8, 4, 2] and ns == [5, 10, 20]
    assert sum(s * n for s, n in zip(sizes, ns)) <= 120
    sizes, ns = rs.schedule(48, 4000)
    assert sizes == [48, 24, 12, 6, 3, 2] and all(b == 2 * a for a, b in zip(ns, ns[1:]))
    assert 3500 <= sum(s * n for s, n in zip(sizes, ns)) <= 4000
    assert rs.schedule(2, 10) == ([2], [5])


def test_score_is_progress_while_losing_and_the_objective_after():
    # one_battle's tuple: won, deathless, fainted fraction, progress, ...
    lost = (False, False, 1.0, 0.5, None, None, False)
    won_two_dead = (True, False, 2 / 6, 1 + 4 / 6, None, None, False)
    assert rs.score(lost, True, 6) == 0.5 - 0.5
    assert rs.score(lost, False, 6) == 0.0
    assert abs(rs.score(won_two_dead, False, 6) - (1 - 0.5 * 2 / 6)) < 1e-9
    assert abs(rs.score((True, False, 0.5, 1.5, None, None, False), False, 2) - (1 - 0.5 * 1 / 6)) < 1e-9  # deaths over 6, not the party


def test_d14_score_prefers_sacrificing_expendable_over_carry():
    d14 = {"bronzong": 0.25, "bibarel": 0.0}
    win_lose_carry = (True, False, 1 / 6, 1 + 5 / 6, None, None, False, ("bronzong",))
    win_lose_filler = (True, False, 1 / 6, 1 + 5 / 6, None, None, False, ("bibarel",))
    assert rs.score(win_lose_filler, False, 6, d14) == 1.0
    assert abs(rs.score(win_lose_carry, False, 6, d14) - 0.75) < 1e-9
    assert rs.score(win_lose_filler, False, 6, d14) > rs.score(win_lose_carry, False, 6, d14)


def test_key_keeps_the_lead_but_not_the_order_of_the_rest_or_the_moves():
    s = lambda sp, moves: {"from": sp, "moves": moves, "item": None, "nature": "Modest", "ability": "x"}
    a = (s("a", ["m1", "m2"]), s("b", ["m3"]), s("c", ["m4"]))
    assert rs.key(a) == rs.key((s("a", ["m2", "m1"]), s("c", ["m4"]), s("b", ["m3"])))
    assert rs.key(a) != rs.key((s("b", ["m3"]), s("a", ["m1", "m2"]), s("c", ["m4"])))


def test_utility_classes():
    want = {"uturn": "pivot", "flipturn": "pivot", "quickattack": "priority", "aquajet": "priority", "spore": "slp",
            "thunderwave": "par", "nuzzle": "par", "willowisp": "brn", "toxic": "tox", "icywind": "speed",
            "tailwind": "speed", "roost": "recovery", "knockoff": "knockoff", "encore": "encore",
            "simplebeam": "change", "sunnyday": "change", "thunderbolt": None, "swordsdance": None}
    assert {m: rs.util_class(m) for m in want} == want
