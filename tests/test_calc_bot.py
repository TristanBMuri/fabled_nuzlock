"""The damage-calc bot's stat model against the engine: for a mon with the bot's
assumed spread (31 IVs / 85 EVs, neutral nature) the estimate must equal the
stats Showdown computes. Needs node + built Showdown."""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_DIST = ROOT / "pokemon-showdown" / "dist" / "sim" / "index.js"
pytestmark = pytest.mark.skipif(not (shutil.which("node") and _DIST.exists()),
                                reason="node + built Showdown required")

from ai_util.calc_bot import ENEMY_EV, ENEMY_IV, max_hp, stat  # noqa: E402
from ai_util.sim import SimBattle, SimBridge  # noqa: E402
from gauntlet import PLAYER_EVS, PLAYER_IVS, Mon, pack_team  # noqa: E402

FMT = "gen8customgamenogimmicks"


def test_enemy_stat_estimate_matches_the_engine():
    assert all(v == ENEMY_EV for v in PLAYER_EVS.values()) and all(v == ENEMY_IV for v in PLAYER_IVS.values())
    us = pack_team([Mon("bidoof", 40, ["tackle"])])  # no Intimidate: the estimate is for stage 0
    them = pack_team([Mon("garchomp", 66, ["earthquake"]), Mon("lucario", 63, ["aurasphere"])])
    bridge = SimBridge()
    try:
        sb = SimBattle(bridge, FMT, us, them)
        opp = sb.mirrors["p1"].opponent_active_pokemon
        real = sb.state()["p2"][0]
        assert opp.species == "garchomp" and real["species"] == "Garchomp"
        assert not any(opp.boosts.values())
        for key in ("atk", "def", "spa", "spd", "spe"):
            assert stat(opp, key, own=False) == real["stats"][key], key
        assert max_hp(opp, own=False) == real["maxhp"]
        me = sb.mirrors["p1"].active_pokemon
        assert stat(me, "atk", own=True) == sb.state()["p1"][0]["stats"]["atk"]  # own stats come from the request
        sb.close()
    finally:
        bridge.close()


def test_full_info_gives_the_bot_the_enemy_set():
    """With full_info the player's mirror knows the enemy's moves, item, ability
    and real stats from turn one; the estimate is replaced by the engine's value
    (a 0-EV Adamant set, where the 31/85 guess is wrong), and a fork keeps it."""
    us = pack_team([Mon("bidoof", 40, ["tackle"])])
    chomp = Mon("garchomp", 66, ["dragonrush", "earthquake", "swordsdance", "flamethrower"], ability="Sand Veil",
                item="Sitrus Berry", nature="Adamant", evs=dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 0),
                ivs=dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 20))
    them = pack_team([chomp, Mon("lucario", 63, ["aurasphere"])])
    bridge = SimBridge()
    try:
        blind = SimBattle(bridge, FMT, us, them)
        opp = blind.mirrors["p1"].opponent_active_pokemon
        assert not opp.moves and stat(opp, "atk", own=False) != blind.state()["p2"][0]["stats"]["atk"]
        blind.close()
        sb = SimBattle(bridge, FMT, us, them, full_info=True)
        opp, real = sb.mirrors["p1"].opponent_active_pokemon, sb.state()["p2"][0]
        assert set(opp.moves) == {"dragonrush", "earthquake", "swordsdance", "flamethrower"}
        assert opp.item == "sitrusberry" and opp.ability == "sandveil" and opp.level == 66
        assert {k: stat(opp, k, own=False) for k in real["stats"]} == real["stats"]
        assert max_hp(opp, own=False) == real["maxhp"]
        assert len(sb.mirrors["p1"].opponent_team) == 2 and len(sb.mirrors["p2"].opponent_team) == 1  # both ways
        assert set(sb.mirrors["p2"].opponent_team["p1: Bidoof"].moves) == {"tackle"}
        f = sb.fork(seed=[1, 2, 3, 4])
        assert set(f.mirrors["p1"].opponent_active_pokemon.moves) == set(opp.moves)
        from ai_util.calc_bot import calc_brain
        from ai_util.evaluate import heuristic_brain
        f.play({"p1": calc_brain(), "p2": heuristic_brain()}, max_turns=60)
        assert f.ended and f.stats["rejected"] == 0
        f.close()
        sb.close()
    finally:
        bridge.close()


def test_ability_immunities_void_a_move():
    """Dry Skin voids Water, Levitate voids Ground: both bots must see zero, not the type chart."""
    from ai_util.calc_bot import effectiveness
    from poke_env.battle import Move, Pokemon
    jynx = Pokemon(gen=9, species="jynx"); jynx._ability = "dryskin"
    gastly = Pokemon(gen=9, species="gastly"); gastly._ability = "levitate"
    scald, quake, psychic = Move("scald", gen=9), Move("earthquake", gen=9), Move("psychic", gen=9)
    assert jynx.damage_multiplier(scald) == 1.0 and effectiveness(scald, jynx) == 0.0
    assert effectiveness(quake, gastly) == 0.0 and effectiveness(psychic, gastly) == 2.0
    assert effectiveness(psychic, jynx) == 0.5


def test_field_effects_in_the_numbers():
    """Rain: Water x1.5, Fire x0.5, Hurricane sure-hit, Swift Swim doubles Speed; Electric
    Terrain: x1.3 for a grounded user only; Tailwind doubles the side's Speed. No engine."""
    from ai_util.calc_bot import dmg_frac, stat
    from poke_env.battle import Battle, Field, Move, Pokemon, SideCondition, Weather
    import logging
    b = Battle("t", "p1", logging.getLogger("t"), gen=9)
    mantine = Pokemon(gen=9, species="mantine"); mantine._ability = "swiftswim"; mantine._level = 27
    jynx = Pokemon(gen=9, species="jynx"); jynx._level = 28
    emolga = Pokemon(gen=9, species="emolga"); emolga._level = 28
    scald, hurricane, ember, spark = (Move(m, gen=9) for m in ("scald", "hurricane", "ember", "spark"))
    dry = {m: dmg_frac(mantine, jynx, m, False, b) for m in (scald, hurricane, ember)}
    assert dry[scald] == dmg_frac(mantine, jynx, scald, False)  # no weather: the plain number
    b._weather = {Weather.RAINDANCE: 1}
    assert dmg_frac(mantine, jynx, scald, False, b) == pytest.approx(dry[scald] * 1.5)
    assert dmg_frac(mantine, jynx, ember, False, b) == pytest.approx(dry[ember] * 0.5)
    assert dmg_frac(mantine, jynx, hurricane, False, b) == pytest.approx(dry[hurricane] / 0.7)  # 70% -> sure hit
    assert stat(mantine, "spe", False, b) == 2 * stat(mantine, "spe", False)
    assert stat(jynx, "spe", True, b) == stat(jynx, "spe", True)
    b._weather = {}
    b._fields = {Field.ELECTRIC_TERRAIN: 1}
    assert dmg_frac(jynx, mantine, spark, True, b) == pytest.approx(dmg_frac(jynx, mantine, spark, True) * 1.3)
    assert dmg_frac(emolga, mantine, spark, True, b) == pytest.approx(dmg_frac(emolga, mantine, spark, True))  # Flying: not grounded
    b._fields = {}
    b._opponent_side_conditions = {SideCondition.TAILWIND: 1}
    assert stat(mantine, "spe", False, b) == 2 * stat(mantine, "spe", False)
    assert stat(jynx, "spe", True, b) == stat(jynx, "spe", True)


def test_resist_berry_halves_the_first_hit():
    from ai_util.calc_bot import dmg_frac
    us = pack_team([Mon("shinx", 28, ["thunderfang", "bite"])])
    them = pack_team([Mon("mantine", 27, ["scald"], item="wacanberry")])
    bridge = SimBridge()
    try:
        sb = SimBattle(bridge, FMT, us, them, full_info=True)
        b = sb.mirrors["p1"]
        me, opp = b.active_pokemon, b.opponent_active_pokemon
        fang = me.moves["thunderfang"]
        held = dmg_frac(me, opp, fang, True, b, with_acc=False)
        opp._item = None
        bare = dmg_frac(me, opp, fang, True, b, with_acc=False)
        assert bare >= 1.0 and abs(held / bare - 0.5) < 0.03  # a 4x one-shot on paper (exactly 96 of 96 HP), half of it through the berry
        bite_bare = dmg_frac(me, opp, me.moves["bite"], True, b, with_acc=False)
        opp._item = "wacanberry"
        assert dmg_frac(me, opp, me.moves["bite"], True, b, with_acc=False) == bite_bare  # the wrong type: no effect
        sb.close()
    finally:
        bridge.close()
