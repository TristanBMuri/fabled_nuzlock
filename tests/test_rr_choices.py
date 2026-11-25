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


def test_mixed_set_takes_speed_only_when_it_outspeeds_a_boss_mon():
    slow = {"spe": [10], "phys": 1.0}  # outsped neutral anyway: + the stronger attack, - Sp. Def (the boss is physical)
    assert rp.choose_nature("electabuzz", ["thunderpunch", "thunderbolt", "swift", "thunderwave"], KNOWN, slow, 30) == "Rash"
    mine = rp.speed(KNOWN["electabuzz"]["baseStats"]["spe"], 30)
    close = {"spe": [mine + 1], "phys": 1.0}  # +Spe turns a loss of the speed race into a win
    assert rp.choose_nature("electabuzz", ["thunderpunch", "thunderbolt", "swift", "thunderwave"], KNOWN, close, 30) == "Timid"


def test_ability_is_rolled_at_catch_and_kept_by_slot():
    slots = KNOWN["gyarados"]["abilities"]
    a = rp.catch_ability("magikarp", "gyarados", 7)
    assert a == rp.catch_ability("magikarp", "gyarados", 7) and a in (slots["0"], slots.get("1"))
    assert {rp.catch_ability("magikarp", "gyarados", s) for s in range(20)} <= {slots["0"], slots.get("1")}


def test_ability_prefers_an_immunity_to_the_boss_and_the_gym_weather():
    rain = {"weather": "raindance", "ai_permanent": True}
    assert rp.choose_ability("jynx", KNOWN, {"Water", "Ice"}, rain) == "Dry Skin"       # Water immunity vs Misty
    assert rp.choose_ability("jynx", KNOWN, {"Rock"}, {}) == KNOWN["jynx"]["abilities"]["0"]  # nothing to gain: slot 0
    assert rp.choose_ability("gyarados", KNOWN, {"Water"}, {}) == "Intimidate"
    # Greninja prefers Battle Bond (with capsule) / Protean (with patch), and Battle Bond packs as Greninja-Bond
    assert rp.choose_ability("greninja", KNOWN, {"Fire"}, {}, hidden=False) == "Battle Bond"
    assert rp.choose_ability("greninja", KNOWN, {"Fire"}, {}, hidden=True) == "Protean"
    packed_g = gauntlet.Mon("greninja", 50, ["surf"], ability="Battle Bond").pack()
    assert "|Greninja-Bond|" in packed_g
    # Starters no longer default to Overgrow/Blaze/Torrent when stronger abilities are unlocked
    assert rp.choose_ability("skeledirge", KNOWN, {"Poison"}, {}, hidden=True) == "Unaware"
    assert rp.choose_ability("samurotthisui", KNOWN, {"Poison"}, {}, hidden=True) == "Sharpness"
    assert rp.choose_ability("delphox", KNOWN, {"Poison"}, {}, hidden=True) == "Magic Guard"
    assert rp.choose_ability("decidueye", KNOWN, {"Poison"}, {}, hidden=True) == "Tinted Lens"
    assert rp.choose_ability("torterra", KNOWN, {"Poison"}, {}, hidden=False) == "Self Sufficient"
    assert rp.choose_ability("charizard", KNOWN, {"Fire"}, {"weather": "desolateland"}, hidden=True) == "Solar Power"


def test_ability_gates_follow_raids_and_the_saffron_shop():
    capsule, patch = rp.ability_gates()
    import json
    names = [e["name"] for e in json.load(open(rp.RR / "order.json"))["entries"]]
    assert names.index("MISTY") < capsule <= names.index("LT. SURGE")  # Route 5 dens open after Misty
    assert patch < names.index("SABRINA") < names.index("KOGA")      # Saffron's Fanclub before Sabrina


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
