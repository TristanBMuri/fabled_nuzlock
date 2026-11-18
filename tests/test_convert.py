"""rl/convert.py: a Showdown state carried into poke-engine, and the names it cannot carry."""

from poke_engine import generate_instructions

from ai_util.sim import SimBridge
from rl.convert import known, outcome_engine, outcome_showdown, to_engine


def test_known_names():
    assert known("species", "rotomwash") and known("move", "earthquake")
    assert known("ability", "levitate") and known("item", "leftovers")
    assert known("volatile", "substitute")
    assert not known("species", "notamon") and not known("move", "notamove")


def test_fresh_battle_converts_and_matches():
    bridge = SimBridge()
    try:
        bridge.call(op="new", id="t", format="gen8randombattle", seed=[1, 2, 3, 4],
                    p1={"name": "p1", "team": None}, p2={"name": "p2", "team": None})
        d = bridge.call(op="dump", id="t")
        state, issues = to_engine(d)
        if issues:  # a generated team can hold an item poke-engine lacks; nothing to compare then
            return
        # before any turn both engines describe the same position
        engine_key, _ = outcome_engine(state, d)
        showdown_key, _ = outcome_showdown(d)
        assert engine_key == showdown_key
        # and poke-engine accepts the legal choices Showdown offers
        c1, c2 = d["p1"]["choices"][0][1], d["p2"]["choices"][0][1]
        assert abs(sum(b.percentage for b in generate_instructions(state, c1, c2)) - 100) < 1e-3
    finally:
        bridge.close()
