"""Radical Red matchup matrix with perfect information: for a boss, every (ours, theirs) pair's
hits to KO both ways under the arena, from the calc bot's own damage model on the real sets
at the cap (the enemy's from the dump, ours as the probe builds them). "safe" = ai_util.matchup.safe
(we KO in at most 2 hits, they need 3, 4 if they move first; healers must fall before they heal). Over a team it shows the holes; over the pool it
shows whether a deathless answer to each enemy mon exists at all.

  uv run python scripts/rr_matchups.py BROCK --team applin,sandshrewalola,furret,skwovet,gumshoos,linoonegalar
  uv run python scripts/rr_matchups.py BROCK --pool            # every catchable species: safe answers per enemy mon

`--json` also writes logs/matrix_<boss>_<team|pool>.json for the dashboard (ui/).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    import rr_probe as rp
    import rr_team_probe as tp
    from poke_env.data import to_id_str
    from ai_util.matchup import safe
    from ai_util.sim import SimBridge

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boss")
    ap.add_argument("--team", help="comma-separated pool species (pre-evolution ids)")
    ap.add_argument("--pool", action="store_true", help="every catchable species, summarised per enemy mon")
    ap.add_argument("--json", action="store_true", help="also write logs/matrix_<boss>_<team|pool>.json")
    args = ap.parse_args()
    rp.hardcore()
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
    grid = tp.cells(bridge, mons, enemy_packed, arena)  # six of ours per dummy battle, never played
    rows = {sp: (mons[sp], grid[sp]) for sp in mons if sp in grid}
    bridge.close()
    if args.json:
        out = ROOT / "logs" / f"matrix_{re.sub(r'[^a-z0-9]+', '_', args.boss.lower()).strip('_')}_{'team' if args.team else 'pool'}.json"
        out.write_text(json.dumps({"boss": args.boss, "cap": cap, "arena": arena, "enemy": tp.enemy_sets(enemy_packed), "team": list(rows),
                                   "sets": [{"species": m.species, "nature": m.nature, "ability": m.ability, "moves": m.moves} for m, _ in rows.values()],
                                   "matrix": grid}))
        print(f"wrote {out}")
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
