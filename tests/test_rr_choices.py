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
