"""Unit tests for Phase 6 encounter routing, Celadon starter eggs, and D14 per-mon death costs."""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "scripts")]

import rr_route as rt


def _mock_tables():
    # Two bosses: B1 (easy) and B2 (hard, weather arena)
    t1 = {
        "boss": "B1", "progress": 10, "cap": 20, "arena": {},
        "enemies": ["e1", "e2"],
        "rows": {
            "starter_a": {"e1": 3.0, "e2": 1.0},
            "filler":    {"e1": 0.5, "e2": 0.5},
            "ace_b2":    {"e1": 1.0, "e2": 1.0},
            "negator":   {"e1": 0.5, "e2": 0.5},
        },
        "negate": set(), "mega": set(), "evolved": {},
    }
    t2 = {
        "boss": "B2", "progress": 80, "cap": 70, "arena": {"weather": "desolateland"},
        "enemies": ["boss1", "boss2"],
        "rows": {
            "starter_a": {"boss1": 0.2, "boss2": 0.2},
            "filler":    {"boss1": 0.1, "boss2": 0.1},
            "ace_b2":    {"boss1": 3.0, "boss2": 3.0},
            "negator":   {"boss1": 1.0, "boss2": 0.5},
        },
        "negate": {"negator"}, "mega": {"ace_b2"}, "evolved": {},
    }
    return [t1, t2]


def test_box_value_and_dupes_renormalization():
    tables = _mock_tables()
    box = {"starter_a", "filler"}
    w = rt.adaptive_boss_weights(box, tables)
    # B2 is much less covered than B1, so adaptive weights must prioritize B2
    assert w["B2"] > w["B1"]

    # Table with already-owned filler + unowned ace_b2: dupes clause filters filler out -> P(ace_b2)=1.0
    g_full, hits = rt.expected_table_gain([("filler", 90.0), ("ace_b2", 10.0)], box, tables, w)
    g_direct, _ = rt.marginal_gain("ace_b2", box, tables, w)
    assert abs(g_full - g_direct) < 1e-6
    assert hits[0][0] == "ace_b2" and abs(hits[0][1] - 1.0) < 1e-6


def test_d14_death_costs_rank_irreplaceable_above_filler():
    tables = _mock_tables()
    # At B1, future boss is B2; losing ace_b2 or sole negator must cost far more than losing filler
    costs = rt.death_costs(["starter_a", "filler", "ace_b2", "negator"], [tables[1]])
    by_sp = {r["species"]: r for r in costs}
    assert by_sp["ace_b2"]["raw_cost"] > by_sp["filler"]["raw_cost"]
    assert by_sp["negator"]["raw_cost"] > by_sp["filler"]["raw_cost"]
    assert by_sp["filler"]["raw_cost"] == 0.0
    assert "arena-negate" in by_sp["negator"]["hurts"][0][2]


def test_choice_encounters_includes_celadon_eggs():
    choices = dict((name, (t, opts)) for name, t, opts in rt.choice_encounters())
    assert "CELADON EGG VENDOR" in choices
    t, opts = choices["CELADON EGG VENDOR"]
    assert t == 33
    assert {"GREEN SHARD", "BLUE SHARD", "RED SHARD", "YELLOW SHARD"} <= set(opts)
    # Saffron Dojo uses real Showdown ids
    assert choices["SAFFRON DOJO"][1]["Tauros-Paldea-Blaze"] == ["taurospaldeablaze"]
    assert choices["SAFFRON DOJO"][1]["Tauros-Paldea-Aqua"] == ["taurospaldeaaqua"]


def test_branched_evolutions_and_slot_limits():
    import gauntlet
    from ai_util.matchup import cover

    old_game = gauntlet.GAME
    try:
        gauntlet.GAME = "rr"
        # Kubfu evolves into both Urshifu forms after Celadon (stone gate)
        assert set(gauntlet.evolve_options("kubfu", 68, caught_level=5, trainer_idx=80)) == {"urshifu", "urshifurapidstrike"}
        # Burmy cloaks stay on their own cloak (+ Mothim)
        assert set(gauntlet.evolve_options("burmytrash", 30, caught_level=5, trainer_idx=40)) == {"wormadamtrash", "mothim"}
        assert set(gauntlet.evolve_options("burmysandy", 30, caught_level=5, trainer_idx=40)) == {"wormadamsandy", "mothim"}
        # Eevee: Espeon/Umbreon pre-Celadon, all 8 post-Celadon
        assert set(gauntlet.evolve_options("eevee", 27, caught_level=5, trainer_idx=20)) == {"espeon", "umbreon"}
        assert len(gauntlet.evolve_options("eevee", 45, caught_level=5, trainer_idx=40)) == 8
        # Hisuian starters branch at level 36+
        assert set(gauntlet.evolve_options("oshawott", 45, caught_level=5, trainer_idx=40)) == {"samurott", "samurotthisui"}
        assert set(gauntlet.evolve_options("rowlet", 45, caught_level=5, trainer_idx=40)) == {"decidueye", "decidueyehisui"}
        assert set(gauntlet.evolve_options("cyndaquil", 45, caught_level=5, trainer_idx=40)) == {"typhlosion", "typhlosionhisui"}
        # Branch-pinned key resolves directly
        assert gauntlet.evolve("eevee>jolteon", 45, None, caught_level=5, trainer_idx=40) == "jolteon"

        # Slot limits in cover(): 1 Eevee -> at most 1 Eeveelution; 2 Eevees -> both allowed
        safe_pair = {"ko_them": 1, "ko_us": 4, "faster": True, "heals": False, "hit": 1.0, "take": 0.25}
        bad_pair = {"ko_them": 5, "ko_us": 1, "faster": False, "heals": False, "hit": 0.2, "take": 1.0}
        rows = {
            "eevee>jolteon": {"e1": safe_pair, "e2": bad_pair},
            "eevee>vaporeon": {"e1": bad_pair, "e2": safe_pair},
            "filler": {"e1": bad_pair, "e2": bad_pair},
        }
        team1 = cover(rows, ["e1", "e2"], k=2, limits={"eevee": 1, "filler": 1}, group_fn=gauntlet.base_species)
        assert sum(1 for x in team1 if gauntlet.base_species(x) == "eevee") == 1
        team2 = cover(rows, ["e1", "e2"], k=2, limits={"eevee": 2, "filler": 1}, group_fn=gauntlet.base_species)
        assert set(team2) == {"eevee>jolteon", "eevee>vaporeon"}
    finally:
        gauntlet.GAME = old_game


def test_gifts_accepted_and_trades_require_owned_mon():
    import random
    import gauntlet
    import rr_probe as rp

    old_game = gauntlet.GAME
    try:
        gauntlet.GAME = "rr"
        known = gauntlet.load_game("rr")["species"]
        # At Koga (t=80), trades like Floette-Eternal must NOT appear unless Flabebe/Floette/Florges was caught
        box = rp.sample_box(random.Random(7), 80, known, include_trades=False)
        species = [sp for sp, _ in box]
        assert "floetteeternal" not in species
        assert "mimikyu" not in species
        # Story gifts ARE accepted: Magikarp, Lapras, BOTH Eevees (Route 5 + Celadon 4F),
        # plus pre-Koga gym rematch gifts Finizen (Misty Rematch #76) and Ogerpon (Erika Rematch #78)
        assert "magikarp" in species
        assert "lapras" in species
        assert "finizen" in species
        assert "ogerpon" in species
        assert species.count("eevee") == 2
        # eligible_trades only triggers when the required line is owned
        assert rp.eligible_trades(["bulbasaur", "magikarp"], 85, known) == []
        elig = rp.eligible_trades(["bulbasaur", "flabebe"], 30, known)
        assert any(tr["species"] == "floetteeternal" and tr["owned_match"] == "flabebe" for tr in elig)
    finally:
        gauntlet.GAME = old_game
