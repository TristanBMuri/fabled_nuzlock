"""Does the in-process sim build the same battle as the server for a harvest pair?

Prints, per mon, level / max HP / stats / moves / ability / item as seen
in-battle by (a) the sim bridge and (b) a live Showdown server via poke-env.
Any difference here is a team-setup (validator) difference, not a mirror one.

  ./scripts/start_showdown.sh 1 8021
  uv run python ai_util/diff_sets.py --port 8021 --pair 0
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(_here.parent))

import argparse
import asyncio

from ai_util.battler_env import preflight
from ai_util.collect_game import sample_pairs
from ai_util.evaluate import GAME_FORMAT
from ai_util.sim import SimBattle, SimBridge
from poke_env.player import RandomPlayer

STATS = ("atk", "def", "spa", "spd", "spe")


def sim_sets(p1: str, p2: str) -> list[list[dict]]:
    bridge = SimBridge()
    sb = SimBattle(bridge, GAME_FORMAT, p1, p2)
    st = sb.state()
    sb.close()
    bridge.close()
    return [[{"species": m["species"].lower(), "level": m["level"], "maxhp": m["maxhp"],
              "stats": tuple(m["stats"][s] for s in STATS), "moves": sorted(m["moves"]),
              "ability": m["ability"], "item": m["item"]} for m in st[side]] for side in ("p1", "p2")]


def server_sets(p1: str, p2: str, port: int) -> list[list[dict]]:
    server = preflight(port)
    kw = dict(battle_format=GAME_FORMAT, server_configuration=server, log_level=50)
    a, b = RandomPlayer(team=p1, **kw), RandomPlayer(team=p2, **kw)

    async def run():
        await asyncio.wait_for(a.battle_against(b, n_battles=1), timeout=120)
        out = []
        for p in (a, b):
            battle = list(p.battles.values())[-1]
            out.append([{"species": m.species, "level": m.level, "maxhp": m.max_hp,
                         "stats": tuple(m.stats.get(s) for s in STATS), "moves": sorted(m.moves),
                         "ability": m.ability, "item": m.item} for m in battle.team.values()])
        for p in (a, b):
            await p.ps_client.stop_listening()
        return out

    return asyncio.run(run())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default="data/teams/gauntlet_harvest/battles.jsonl")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--port", type=int, default=8021)
    args = ap.parse_args()
    p1, p2 = sample_pairs(args.jsonl, args.pair + 1, 0)[args.pair]
    print("packed p1:", p1)
    print("packed p2:", p2)
    sim, srv = sim_sets(p1, p2), server_sets(p1, p2, args.port)
    diffs = 0
    for side, s_team, v_team in zip(("p1", "p2"), sim, srv):
        key = lambda m: m["species"]
        for s, v in zip(sorted(s_team, key=key), sorted(v_team, key=key)):
            same = s == v
            diffs += not same
            print(f"{side} {s['species']:<12} {'same' if same else 'DIFF'}")
            if not same:
                print("   sim   :", s)
                print("   server:", v)
    print(f"{diffs} differing mons")


if __name__ == "__main__":
    main()
