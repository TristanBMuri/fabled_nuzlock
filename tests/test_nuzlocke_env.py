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
from nuzlocke_env import NuzlockeEnv  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = NuzlockeEnv(seed=0, risk_features=False, slice_len=186)
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
        assert [m.species for m in env.pending] == ["bidoof", "zubat"]
        assert env.phase == "ENCOUNTER"
        env.step(1)  # catch bidoof -> still answering zubat
        assert env.phase == "ENCOUNTER" and [m.species for m in env.pending] == ["zubat"]
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
        assert [m.species for m in env.pending] == ["shinx"]  # starly rerolled away
        env.dupes_clause = False
        env.pending, env._route_cursor = [], 0
        for _ in range(20):
            env._maybe_sample_encounter()
            env._route_cursor = 0
        assert "starly" in {m.species for m in env.pending}
    finally:
        env.encounters, env.dupes_clause = real, True


def test_box_cap_counts_pending_offers(env):
    real = env.encounters
    try:
        two_areas_at_once(env)
        env.max_roster = 2
        env.reset(seed=3)  # starter + one slot: only the first offer fits
        assert [m.species for m in env.pending] == ["bidoof"]
        assert env.dropped == ["zubat"]
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
    env.trainer_idx = 139  # Gardenia's cap (22): Shinx evolves at 15, Starly at 14
    env._apply_level_rule({"max_level": 19})
    assert [(m.species, m.ability) for m in env.roster] == [("luxio", "Intimidate"), ("staravia", "Intimidate")]


def test_best4_moveset_rule():
    assert encounter_moves("meditite", 22, "last4") == ["detect", "hiddenpower", "mindreader", "feint"]
    best = encounter_moves("meditite", 22, "best4")
    assert len(best) == 4 and best[0] == "confusion" and "hiddenpower" in best  # attacks at last
    assert encounter_moves("bonsly", 22, "best4")[:2] == ["rockthrow", "lowkick"]  # variable power counts
    assert "selfdestruct" not in encounter_moves("geodude", 22, "best4")  # self-KO never
    lux = encounter_moves("luxio", 22, "best4")
    assert lux[0] == "spark"  # strongest STAB first
    assert len(set(lux)) == len(lux)
