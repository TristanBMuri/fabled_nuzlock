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


def test_rr_dump_converts():
    """A gen9rr battle on the RR fork converts for the RR build: typed Hidden Power -> the 60 BP
    id, a Z crystal -> no item, a mega stone kept (the engine megas on the first move), a mega
    forme marked as the side's spent mega."""
    from pathlib import Path

    from rl import convert
    fork = Path(__file__).resolve().parents[1] / "pokemon-showdown-rr"
    p1 = "Sandshrew|||sandveil|hiddenpowerice,rapidspin|Adamant|||||16|]Geodude|||sturdy|rockthrow|Adamant|||||16|"
    p2 = "Charmander||firiumz|blaze|ember|Modest|||||16|]Kingler||kinglerite|hypercutter|crabhammer|Adamant|||||16|"
    convert.use_rr()
    bridge = SimBridge(showdown=str(fork))
    try:
        bridge.call(op="new", id="t", format="gen9rrcustomgamenogimmicks", seed=[1, 2, 3, 4],
                    p1={"name": "p1", "team": p1}, p2={"name": "p2", "team": p2})
        d = bridge.call(op="dump", id="t")
        state, issues = convert.to_engine(d)
        assert issues == []
        assert "HIDDENPOWERICE60" in state.to_string() and "FIRIUMZ" not in state.to_string()
        bridge.call(op="choose", id="t", p1="move 2", p2="switch 2")
        d = bridge.call(op="dump", id="t")
        assert d["p2"]["canMega"]
        state, issues = convert.to_engine(d)
        assert issues == [] and "KINGLERITE" in state.to_string()
        assert any(o.endswith("-mega") for o in convert.pe.get_all_options(state)[1])
        bridge.call(op="choose", id="t", p1="move 2", p2="move 1 mega")
        d = bridge.call(op="dump", id="t")
        state, issues = convert.to_engine(d)
        assert issues == [] and state.side_two.pokemon[0].id == "kinglermega"
        assert state.side_two.pokemon[0].mega_evolved
        assert not any(o.endswith("-mega") for o in convert.pe.get_all_options(state)[1])
    finally:
        bridge.close()
        convert.use_rr(False)


def test_rr_locked_and_charging_volatiles_convert():
    """Mid-turn state Showdown keeps in volatiles: Rollout's hit count (rollout1-4, on a build that
    has the lock-in; an older build reports it), a charging Solar Beam (its own volatile; the
    twoturnmove timer is Showdown bookkeeping), Protosynthesis' boosted stat."""
    from pathlib import Path

    from rl import convert
    fork = Path(__file__).resolve().parents[1] / "pokemon-showdown-rr"
    convert.use_rr()
    bridge = SimBridge(showdown=str(fork))
    foe = "Snorlax|||thickfat|splash|Adamant|||||60|"
    try:
        def after_one_turn(p1, foe=foe):
            bridge.call(op="new", id="t", format="gen9rrcustomgamenogimmicks", seed=[1, 2, 3, 4],
                        p1={"name": "p1", "team": p1}, p2={"name": "p2", "team": foe})
            bridge.call(op="choose", id="t", p1="move 1", p2="move 1")
            d = bridge.call(op="dump", id="t")
            bridge.call(op="close", id="t")
            return (*convert.to_engine(d), d)
        state, issues, d = after_one_turn("Geodude|||sturdy|rollout|Adamant|||||30|")
        assert d["p1"]["pokemon"][0]["volatileData"]["rollout"]["hitCount"] == 1
        if convert.known("volatile", "rollout1"):
            assert issues == [] and "ROLLOUT1" in state.to_string()
        else:
            assert "build:rollout" in issues
        state, issues, _ = after_one_turn("Venusaur|||overgrow|solarbeam,tackle|Modest|||||50|")
        assert issues == [] and "SOLARBEAM" in state.to_string()
        assert convert.pe.get_all_options(state)[0] == ["solarbeam"]
        state, issues, _ = after_one_turn("Greattusk|||protosynthesis|tackle|Adamant|||||50|",
                                          "Torkoal|||drought|splash|Adamant|||||60|")
        assert issues == [] and "PROTOSYNTHESISATK" in state.to_string()
    finally:
        bridge.close()
        convert.use_rr(False)
