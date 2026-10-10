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
