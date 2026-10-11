"""Radical Red post-mortem: replay a probe's best team against its boss under search (arena
on, legal moves) and attribute every death to the enemy mon on the field — which enemy mons
kill us, which of ours die, which enemy mons we never get through. The first form of threat
detection: what a box would need to add for this boss. `attribution()` is what
rr_team_probe.py uses to name threats and rank counters.

  uv run python scripts/rr_postmortem.py MISTY 12 [results/rr_team_probe_real.jsonl]
"""

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def attribution(history: list[str]) -> tuple[collections.Counter, collections.Counter, collections.Counter]:
    """From one side's protocol lines (SimBattle.history["p1"]): (our deaths by the enemy mon
    on the field, enemy mons KO'd, our deaths by our own mon). The first is the threat list."""
    deaths_by, kos_on, killed = collections.Counter(), collections.Counter(), collections.Counter()
    active = {"p1": None, "p2": None}
    for line in history:
        m = re.match(r"\|(?:switch|drag|replace)\|(p[12])a: ([^|]+)\|([^|,]+)", line)
        if m:
            active[m.group(1)] = m.group(3).strip()
        m = re.match(r"\|faint\|(p[12])a: ([^|]+)", line)
        if m and m.group(1) == "p1":
            deaths_by[active["p2"]] += 1
            killed[active["p1"]] += 1
        elif m:
            kos_on[active["p2"]] += 1
    return deaths_by, kos_on, killed


def main() -> None:
    import rr_probe as rp
    import gauntlet
    from gauntlet import STATS, Mon, encounter_moves, evolve, pack_team
    from poke_env.data import to_id_str
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBattle, SimBridge, player_brain

    boss, n = sys.argv[1], int(sys.argv[2])
    source = sys.argv[3] if len(sys.argv) > 3 else "results/rr_team_probe_real.jsonl"
    gauntlet.BANNED_MOVES = {to_id_str(m) for m in json.load(open(ROOT / "data/rr/hardcore_rules.json"))["banned_moves"]}
    rows = {r["boss"]: r for r in map(json.loads, open(ROOT / source))}
    r = rows[boss]
    by, known = rp.pools()
    block, cap, pool, t = by[boss]
    arena, _ = rp.arena(block.get("battleEffect", ""))
    enemy_packed, dump_cap, species = rp.boss_team(boss)
    party = []
    for sp in r["best_team"]:
        ev = evolve(sp, cap, None, caught_level=5, trainer_idx=None)
        m = Mon(ev, cap, encounter_moves(ev, cap, "best4", progress=t), ability=known[ev]["abilities"].get("0"), caught_level=5)
        m.evs = dict.fromkeys(STATS, 0)
        party.append(m)
    print("team:", [(m.species, m.ability, m.moves) for m in party])
    print("enemy:", [(x.split("|")[0], x.split("|")[4]) for x in enemy_packed.split("]")])
    bridge = SimBridge(showdown=str(rp.FORK))
    enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    searcher = Searcher(None, enemy, k=9, rollouts=8, rollout=calc_brain(), seed=0)
    killer, killed, kos_on, survivors, wins = (collections.Counter() for _ in range(4)), 0
    killer, killed, kos_on, survivors = killer
    for i in range(n):
        sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena, seed=[i + 1, 2, 3, 4])
        sb.play({"p1": searcher, "p2": enemy}, 100)
        st = sb.state()
        wins += st["winner"] == "p1"
        d, k, kd = attribution(sb.history["p1"])
        killer += d
        kos_on += k
        killed += kd
        for mon in st["p2"]:
            if not mon["fainted"]:
                survivors[mon["species"]] += 1
        sb.close()
    print(f"\n{boss}: {wins}/{n} wins")
    print("our deaths by enemy mon on the field:", killer.most_common())
    print("which of ours die:", killed.most_common())
    print("enemy mons we KO:", kos_on.most_common())
    print("enemy mons left standing at the end:", survivors.most_common())
    bridge.close()


if __name__ == "__main__":
    main()
