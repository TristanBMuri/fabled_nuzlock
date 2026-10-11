"""Radical Red matchup matrix with perfect information: for a boss, every (ours, theirs) pair's
hits to KO both ways under the arena, from the calc bot's own damage model on the real sets
at the cap (the enemy's from the dump, ours as the probe builds them). "safe" = ai_util.matchup.safe
(we KO in at most 2 hits, they need 3, 4 if they move first; healers must fall before they heal). Over a team it shows the holes; over the pool it
shows whether a deathless answer to each enemy mon exists at all.

  uv run python scripts/rr_matchups.py BROCK --team applin,sandshrewalola,furret,skwovet,gumshoos,linoonegalar
  uv run python scripts/rr_matchups.py BROCK --pool            # every catchable species: safe answers per enemy mon
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    import rr_probe as rp
    import rr_team_probe as tp
    import gauntlet
    from gauntlet import pack_team
    from poke_env.data import to_id_str
    from ai_util.matchup import pair, safe
    from ai_util.sim import SimBattle, SimBridge

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boss")
    ap.add_argument("--team", help="comma-separated pool species (pre-evolution ids)")
    ap.add_argument("--pool", action="store_true", help="every catchable species, summarised per enemy mon")
    args = ap.parse_args()
    gauntlet.BANNED_MOVES = {to_id_str(m) for m in json.load(open(ROOT / "data/rr/hardcore_rules.json"))["banned_moves"]}
    by, known = rp.pools()
    block, cap, pool, t = by[args.boss]
    arena, _ = rp.arena(block.get("battleEffect", ""))
    enemy_packed, dump_cap, _ = rp.boss_team(int(args.boss[1:]) if args.boss.startswith("#") else args.boss)
    cap = max(cap, dump_cap)
    md = rp.load_game("rr")["moves"]
    boss_types = {md[mv]["type"] for mon in enemy_packed.split("]") for mv in mon.split("|")[4].split(",") if mv in md}
    species = args.team.split(",") if args.team else pool
    mons = {sp: tp.build_mon(sp, cap, known, t, boss_types, arena) for sp in species}
    mons = {sp: m for sp, m in mons.items() if m is not None}
    bridge = SimBridge(showdown=str(rp.FORK))
    rows = {}
    for i in range(0, len(mons), 6):  # six of ours per battle; read both mirrors' real stats, never play
        chunk = list(mons)[i:i + 6]
        sb = SimBattle(bridge, rp.FMT, pack_team([mons[s] for s in chunk]), enemy_packed, full_info=True, arena=arena)
        b = sb.mirrors["p1"]
        ours = {p.species: p for p in b.team.values()}
        theirs = list(b.opponent_team.values())
        for sp in chunk:
            me = next((p for p in ours.values() if p.species == to_id_str(mons[sp].species)), None)
            if me is None:
                continue
            rows[sp] = (mons[sp], {opp.species: pair(me, opp, b) for opp in theirs})
        sb.close()
    bridge.close()
    enemies = [p.split("|")[0] for p in enemy_packed.split("]")]
    ids = [to_id_str(e) for e in enemies]
    print(f"{args.boss}: cap {cap}, arena {arena}. cell = hits we need / hits they need (> = we move first, h = they heal); * = safe (ai_util.matchup.safe)")
    print(f"{'ours':<22}" + "".join(f"{e[:11]:>12}" for e in enemies))
    safe_count = {e: [] for e in ids}
    for sp, (m, row) in rows.items():
        cells = []
        for e in ids:
            p = row.get(e, {"ko_them": 99, "ko_us": 99, "faster": False, "heals": False})
            if safe(p):
                safe_count[e].append(m.species)
            cells.append(f"{'*' if safe(p) else ' '}{min(p['ko_them'], 9)}/{min(p['ko_us'], 9)}{'>' if p['faster'] else ' '}{'h' if p['heals'] else ' '}".rjust(12))
        if not args.pool:
            print(f"{m.species[:21]:<22}" + "".join(cells) + f"   {m.nature} {m.ability} {m.moves}")
    if not args.pool:
        print("\\nsafe answers per enemy mon in this team: " + "; ".join(f"{e}: {', '.join(v) or 'NONE'}" for e, v in safe_count.items()))
    else:
        print(f"{len(rows)} species in the pool. Safe answers per enemy mon (KO in <=2, survive >=3 hits):")
        for e, v in safe_count.items():
            print(f"  {e:<14} {len(v):>3}: {', '.join(v[:14])}{' ...' if len(v) > 14 else ''}")
        # a six that covers every enemy mon with a safe answer?
        cover = set()
        for e, v in sorted(safe_count.items(), key=lambda kv: len(kv[1])):
            if v and not (cover & set(v)):
                cover.add(v[0])
        print(f"  a covering set needs at least {sum(1 for v in safe_count.values() if not v)} enemies left without a safe answer; greedy cover: {sorted(cover)}")


if __name__ == "__main__":
    main()
