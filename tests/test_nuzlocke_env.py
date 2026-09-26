"""Run-level env rules: encounter queue, dupes clause, box cap. Needs node +
built Showdown (the env owns a sim runner)."""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_DIST = ROOT / "pokemon-showdown" / "dist" / "sim" / "index.js"
pytestmark = pytest.mark.skipif(not (shutil.which("node") and _DIST.exists()),
                                reason="node + built Showdown required")

from gauntlet import Mon, encounter_moves, evolved_ability, roll_ability  # noqa: E402
from nuzlocke_env import ORDER, NuzlockeEnv, Rules  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = NuzlockeEnv(seed=0, risk_features=False, slice_len=59)
    yield e
    e.runner.bridge.close()


def two_areas_at_once(env):
    """Two areas gated on trainer 0 (plus the real Route 201)."""
    env.encounters = [
        {"name": "A", "before_trainer": 0, "encounters": [{"species": "bidoof", "level": [3, 3], "weight": 1}]},
        {"name": "B", "before_trainer": 0, "encounters": [{"species": "zubat", "level": [3, 3], "weight": 1}]},
    ]


def test_every_area_is_offered(env):
    real = env.encounters
    try:
        two_areas_at_once(env)
        env.reset(seed=1)
        assert [m.species for a in env.pending for m in a] == ["bidoof"]  # B waits for A's answer
        assert [r["name"] for r in env._queue] == ["B"] and env.phase == "ENCOUNTER"
        env.step(1)  # catch bidoof -> now answering zubat
        assert env.phase == "ENCOUNTER" and [m.species for a in env.pending for m in a] == ["zubat"]
        env.step(0)  # skip zubat
        assert env.phase == "PRE_BATTLE" and not env.pending
        assert [m.species for m in env.roster[1:]] == ["bidoof"]
    finally:
        env.encounters = real


def test_dupes_clause_rerolls_owned_families(env):
    real = env.encounters
    try:
        env.encounters = [{"name": "A", "before_trainer": 0, "encounters": [
            {"species": "starly", "level": [3, 3], "weight": 0.99},
            {"species": "shinx", "level": [3, 3], "weight": 0.01}]}]
        env.reset(seed=2)
        env.roster.append(Mon("staravia", 14, ["tackle"]))  # an evolved Starly is owned
        env.graveyard.append(Mon("bidoof", 5, ["tackle"], alive=False))
        assert {"starly", "staravia", "staraptor", "bidoof", "bibarel"} <= env._owned_families()
        env.pending, env._route_cursor = [], 0
        env._maybe_sample_encounter()
        assert [m.species for a in env.pending for m in a] == ["shinx"]  # starly rerolled away
        env.rules = Rules(dupes_clause=False)
        env.pending, env._route_cursor = [], 0
        for _ in range(20):
            env._maybe_sample_encounter()
            env._route_cursor = 0
        assert "starly" in {m.species for a in env.pending for m in a}
    finally:
        env.encounters, env.rules = real, Rules()


def test_dupes_clause_spans_the_whole_line(env):
    env.reset(seed=2)
    env.roster.append(Mon("crobat", 30, ["bite"]))
    assert {"zubat", "golbat", "crobat"} <= env._owned_families()  # not one step down only


def test_offers_roll_after_the_previous_catch(env):
    """Two areas gated at one trainer: the second rolls after the first is
    answered, so it cannot offer the family just caught."""
    real = env.encounters
    try:
        table = [{"species": "magikarp", "level": [3, 3], "weight": 0.999},
                 {"species": "zubat", "level": [3, 3], "weight": 0.001}]
        env.encounters = [{"name": "A", "before_trainer": 0, "encounters": table},
                          {"name": "B", "before_trainer": 0, "encounters": table}]
        env.reset(seed=5)
        assert [m.species for m in env.pending[0]] == ["magikarp"]
        env.step(1)
        assert [m.species for m in env.pending[0]] == ["zubat"]
    finally:
        env.encounters = real


def test_locked_party_is_rebuilt_by_identity(env):
    """Two value-equal mons (twins at the cap) must not both land in the
    locked League party (a 7-mon party crashed the runner)."""
    from nuzlocke_env import ORDER
    env.reset(seed=6)
    env.slice_len = 405
    env.trainer_idx = next(i for i in range(len(ORDER) - 1) if ORDER[i + 1].get("no_heal"))  # Aaron
    env.roster = [Mon("crobat", 66, ["bite"]), Mon("crobat", 66, ["bite"])]
    env.phase, env.party = "PRE_BATTLE", [0]
    run = env.runner.run
    env.runner.run = lambda *a, **k: (True, [True], [{"hp": 100, "status": "", "pp": [10]}])
    try:
        env.step(0)
    finally:
        env.runner.run = run
    assert env.party_locked and env.party == [0]


def test_box_cap_counts_pending_offers(env):
    real = env.encounters
    try:
        two_areas_at_once(env)
        env.max_roster = 2
        env.reset(seed=3)  # starter + one slot: only the first offer fits
        assert [m.species for a in env.pending for m in a] == ["bidoof"]
        assert env.dropped == ["B"]
    finally:
        env.encounters, env.max_roster = real, 12


def test_ability_slot_rolls_at_capture_and_survives_evolution():
    import random
    rolled = {roll_ability("shinx", random.Random(i)) for i in range(40)}
    assert rolled == {"Rivalry", "Intimidate"}  # slot 0 or 1, never the hidden Guts
    assert roll_ability("starly", random.Random(0)) == "Keen Eye"  # one slot only
    assert evolved_ability("shinx", "Intimidate", "luxio") == "Intimidate"  # slot 1 kept
    assert evolved_ability("starly", "Keen Eye", "staravia") == "Intimidate"  # slot 0 kept
    assert evolved_ability("hoothoot", "Keen Eye", "noctowl") == "Keen Eye"


def test_evolution_keeps_the_slot_in_the_env(env):
    env.reset(seed=4)
    env.roster = [Mon("shinx", 5, ["tackle"], ability="Intimidate"), Mon("starly", 5, ["tackle"], ability="Keen Eye")]
    env.trainer_idx = 50  # before Gardenia (#58, cap 22): Shinx evolves at 15, Starly at 14
    env._apply_level_rule({"max_level": 19})
    assert [(m.species, m.ability) for m in env.roster] == [("luxio", "Intimidate"), ("staravia", "Intimidate")]


def test_best4_moveset_rule():
    assert encounter_moves("meditite", 22, "last4") == ["detect", "hiddenpower", "mindreader", "feint"]
    best = encounter_moves("meditite", 22, "best4")
    assert len(best) == 4 and best[0] == "confusion" and "hiddenpower" in best  # attacks at last
    machop = encounter_moves("machop", 22, "best4")
    assert "seismictoss" in machop or "lowkick" in machop  # variable power counts as 50
    assert "selfdestruct" not in encounter_moves("geodude", 22, "best4")  # self-KO never
    lux = encounter_moves("luxio", 22, "best4")
    assert lux[0] == "spark"  # strongest STAB first
    assert len(set(lux)) == len(lux)


def test_tm_movesets_follow_the_route():
    """E7: a TM joins the best4 pool once its hand-out trainer is passed."""
    from gauntlet import load_game
    assert load_game()["tms"]["earthquake"]["before_trainer"] == 78  # Wayward Cave
    assert "earthquake" not in encounter_moves("gyarados", 66, "best4")  # no TMs
    assert "earthquake" not in encounter_moves("gyarados", 66, "best4", progress=77)
    assert "earthquake" in encounter_moves("gyarados", 66, "best4", progress=78)
    assert "icebeam" in encounter_moves("abomasnow", 66, "best4", progress=160)  # Veilstone Department Store
    assert encounter_moves("lapras", 66, "best4", progress=404)[:3] == ["surf", "icebeam", "thunderbolt"]
    assert "icebeam" not in encounter_moves("staraptor", 66, "best4", progress=404)  # cannot learn it
    assert encounter_moves("gyarados", 66, "last4", progress=404) == encounter_moves("gyarados", 66, "last4")
    tm_env = NuzlockeEnv(seed=0, risk_features=False, slice_len=405)
    try:
        tm_env.reset(seed=7)
        tm_env.roster = [Mon("gyarados", 5, ["splash"], caught_level=5)]
        tm_env.trainer_idx = 160
        tm_env._apply_level_rule({"max_level": 40})
        assert "earthquake" in tm_env.roster[0].moves  # Wayward Cave's TM26 is in hand at #160
        assert tm_env.roster[0].evs["atk"] == 85 and "85,85,85,85,85,85" in tm_env.roster[0].pack()
        tm_env.rules = Rules(evs="trained")
        tm_env._apply_level_rule({"max_level": 40})
        assert tm_env.roster[0].evs == {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252}  # Gyarados: physical
        tm_env.roster.append(Mon("alakazam", 40, ["psychic"]))
        tm_env._apply_level_rule({"max_level": 40})
        assert tm_env.roster[1].evs["spa"] == 252 and "4,0,0,252,0,252" in tm_env.roster[1].pack()
    finally:
        tm_env.runner.bridge.close()


def test_rules_reach_the_env(env):
    assert str(env.rules) == "level_rule=cap margin=0 box=50 dupes=True moveset=best4 tms=True evs=flat"
    last4 = NuzlockeEnv(seed=0, risk_features=False, slice_len=59, rules=Rules(moveset="last4", box=12))
    try:
        last4.reset(seed=4)
        last4.roster = [Mon("meditite", 5, ["bide"])]
        last4.trainer_idx = 50
        last4._apply_level_rule({"max_level": 19})
        assert last4.roster[0].moves == ["detect", "hiddenpower", "mindreader", "feint"]
        assert last4.max_roster == 12
    finally:
        last4.runner.bridge.close()


def test_only_the_countering_rival_is_fought(env):
    from nuzlocke_env import ORDER, TRAINERS
    counter = {"turtwig": "chimchar", "chimchar": "piplup", "piplup": "turtwig"}
    for seed in range(6):
        env.reset(seed=seed)
        starter = env.roster[0].species
        env.slice_len = 405
        for idx in (6, 99, 205):  # the first record of each rival battle
            env.trainer_idx = idx
            env._skip_other_rivals()
            e = ORDER[env.trainer_idx]
            party = {p["species"].lower() for p in TRAINERS[e["trainer_id"]]["party"]}
            assert e["trainer_name"] == "Pokémon Trainer Cedric" and any(
                sp in party for sp in env._rival_family), (starter, idx, party)


def test_method_alternatives_respect_the_dupes_clause(env):
    real = env.encounters
    try:
        env.encounters = [{"name": "A", "before_trainer": 0, "encounters": [
            {"species": "magikarp", "level": [3, 3], "weight": 1, "method": "old"},
            {"species": "zubat", "level": [3, 3], "weight": 1, "method": "walk"}]}]
        env.reset(seed=1)
        assert sorted(m.species for m in env.pending[0]) == ["magikarp", "zubat"]
        env.step(1 + [m.species for m in env.pending[0]].index("magikarp"))
        env.pending, env._route_cursor = [], 0
        env._maybe_sample_encounter()
        assert [m.species for m in env.pending[0]] == ["zubat"]  # the rod's only family is owned
    finally:
        env.encounters = real


def test_sim_chosen_moves_persist_while_the_pool_holds_them(env):
    """E7(d): a Mon.chosen set survives the level rule; the risk cache keys on moves."""
    env.reset(seed=0)
    env.trainer_idx = 100  # deep enough for a wide pool (level-up moves + TMs)
    m, entry = env.roster[0], ORDER[env.trainer_idx]
    env._apply_level_rule(entry)
    pool = encounter_moves(m.species, m.level, "best4", env.trainer_idx, k=None)
    assert pool[:4] == m.moves and len(pool) > 4
    m.chosen = [pool[4]] + pool[:3]
    env._apply_level_rule(entry)
    assert m.moves == m.chosen
    m.chosen = ["fly"]  # not in the pool: back to the rule
    env._apply_level_rule(entry)
    assert m.chosen is None and m.moves == pool[:4]
    env.risk((0,), n=2)
    m.chosen = pool[1:5]  # a different set is a new cache entry
    env.risk((0,), n=2)
    assert len([k for k in env._risk_cache if k[0] == (0,)]) == 2
