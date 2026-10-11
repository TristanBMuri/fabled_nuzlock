"""The planner's nature and ability choices for a Radical Red boss (data/rr only; no engine)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gauntlet  # noqa: E402
import rr_probe as rp  # noqa: E402  (switches gauntlet.GAME to "rr" on import)

gauntlet.GAME = "bdsp"  # the other tests run the BDSP game; load_game caches per name
KNOWN = gauntlet.load_game("rr")["species"]


def test_nature_follows_the_set():
    assert rp.choose_nature("arbok", ["poisonjab", "crunch", "earthquake", "rest"], KNOWN) == "Adamant"
    assert rp.choose_nature("jynx", ["psychic", "icebeam", "uproar", "rest"], KNOWN) == "Modest"
    mixed = rp.choose_nature("electabuzz", ["thunderpunch", "thunderbolt", "swift", "thunderwave"], KNOWN)
    assert mixed in ("Timid", "Jolly")


def test_ability_prefers_an_immunity_to_the_boss_and_the_gym_weather():
    rain = {"weather": "raindance", "ai_permanent": True}
    assert rp.choose_ability("jynx", KNOWN, {"Water", "Ice"}, rain) == "Dry Skin"       # Water immunity vs Misty
    assert rp.choose_ability("jynx", KNOWN, {"Rock"}, {}) == KNOWN["jynx"]["abilities"]["0"]  # nothing to gain: slot 0
    assert rp.choose_ability("gyarados", KNOWN, {"Water"}, {}) == "Intimidate"


def test_sample_box_is_one_encounter_per_open_location():
    import random
    by, known = rp.pools()
    _, cap, pool, t = by["MISTY"]
    box = rp.sample_box(random.Random(3), t, known)
    places = [w.split(" / ")[0] for s, w in box if "/" in w]
    assert len(places) == len(set(places)) and 10 <= len(box) <= 30
    assert box[0][1] == "starter" and box[0][0] in rp.STARTERS
    assert all(s in pool for s, w in box[1:]) and len({s for s, _ in box}) == len(box)  # catchable, no dupes


def test_dump_names_resolve_to_fork_ids():
    """The ROM dump's 12-character names ("Drain Kiss", "None" items) pack as the fork's ids."""
    import rr_probe as rp
    assert rp.dump_id("Drain Kiss") == "drainingkiss"
    assert rp.dump_id("Hidden Power Grass") == "hiddenpowergrass"
    assert rp.dump_id("None") == ""
    assert rp.dump_id("As One", "Calyrex-Shadow") == "asonespectrier"
    assert rp.dump_id("Applite", "Flapple") == "flapplite"
    packed = rp.boss_team(2)[0]  # Lass Anne: Audino holds nothing
    assert "|none|" not in packed and "drainkiss" not in packed
