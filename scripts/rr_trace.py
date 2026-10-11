"""Radical Red fight trace: one seeded fight of a probe's best team against its boss under
search, turn by turn — both actives with HP, every candidate's rollout value, the pick, and
what the engine did. For judging decisions, not aggregates.

  uv run python scripts/rr_trace.py BROCK [--source results/rr_bench_field.jsonl] [--battle 0] [--preserve]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SHOW = re.compile(r"^\|(move|switch|drag|faint|-damage|-heal|-status|-boost|-unboost|-weather|-ability|-item|-enditem|-crit|-supereffective|-resisted|-immune|-miss|cant|win)\|")


def main() -> None:
    import numpy as np
    import rr_probe as rp
    import rr_team_probe as tp
    import gauntlet
    from gauntlet import pack_team
    from poke_env.data import to_id_str
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBattle, SimBridge, player_brain

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boss")
    ap.add_argument("--source", default="results/rr_bench_field.jsonl")
    ap.add_argument("--team", help="comma-separated pool species instead of the source's best team")
    ap.add_argument("--battle", type=int, default=0, help="which paired seed")
    ap.add_argument("--preserve", action="store_true")
    ap.add_argument("--matchup", action="store_true")
    ap.add_argument("--death-weight", type=float, default=0.5)
    args = ap.parse_args()
    gauntlet.BANNED_MOVES = {to_id_str(m) for m in json.load(open(ROOT / "data/rr/hardcore_rules.json"))["banned_moves"]}
    if args.preserve or args.matchup:
        import ai_util.calc_bot as cb
        cb.PRESERVE, cb.MATCHUP = args.preserve, args.matchup
    by, known = rp.pools()
    block, cap, pool, t = by[args.boss]
    arena, _ = rp.arena(block.get("battleEffect", ""))
    enemy_packed, dump_cap, _ = rp.boss_team(int(args.boss[1:]) if args.boss.startswith("#") else args.boss)
    cap = max(cap, dump_cap)
    md = rp.load_game("rr")["moves"]
    boss_types = {md[mv]["type"] for mon in enemy_packed.split("]") for mv in mon.split("|")[4].split(",") if mv in md}
    team = args.team.split(",") if args.team else {x["boss"]: x for x in map(json.loads, open(ROOT / args.source))}[args.boss]["best_team"]
    party = [tp.build_mon(sp, cap, known, t, boss_types, arena) for sp in team]
    print("ours: " + "; ".join(f"{m.species} {m.nature}/{m.ability} {m.moves}" for m in party))
    print("theirs: " + "; ".join(f"{x.split('|')[0]} L{x.split('|')[10]} @{x.split('|')[2]} {x.split('|')[3]} [{x.split('|')[4]}]" for x in enemy_packed.split("]")))
    bridge = SimBridge(showdown=str(rp.FORK))
    enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    calc = calc_brain()
    searcher = Searcher(None, enemy, k=9, rollouts=8, rollout=calc, seed=0, death_weight=args.death_weight)
    srng = random.Random(f"{args.boss}-seeds-0")
    seeds = [[srng.randrange(1, 2 ** 16) for _ in range(4)] for _ in range(args.battle + 1)]
    seed = seeds[args.battle]
    random.seed(sum(seed))
    searcher.rng = np.random.default_rng(seed)
    sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena, seed=seed)
    shown = 0
    while not sb.ended and sb.turn < 100:
        m = sb.mirrors["p1"]
        me, opp = m.active_pokemon, m.opponent_active_pokemon
        ch = {}
        if sb.pending["p2"]:
            ch["p2"] = sb.choice("p2", enemy)
        if sb.pending["p1"]:
            ch["p1"] = sb.choice("p1", searcher)
            hp = lambda p: f"{p.species} {p.current_hp_fraction:.0%}" + (f" {p.status.name}" if p.status else "") + (f" {dict((k, v) for k, v in p.boosts.items() if v)}" if any(p.boosts.values()) else "")
            bench = ", ".join(f"{p.species} {p.current_hp_fraction:.0%}" for p in m.available_switches)
            print(f"\nT{sb.turn:>2} {hp(me)} vs {hp(opp)}  | bench: {bench}")
            if searcher.searched_last:
                ranked = sorted(searcher.last_scores.items(), key=lambda kv: -kv[1])
                print("     search: " + "  ".join(f"{k.replace('move ', '').replace('switch ', 'sw:')}={v:+.2f}" for k, v in ranked))
            print(f"     pick: {ch['p1']}" + ("  (forced / only option)" if not searcher.searched_last else ""))
        sb.step(ch)
        new = sb.history["p1"][shown:]
        shown = len(sb.history["p1"])
        ev = [l for l in new if SHOW.match(l)]
        print("     " + " | ".join(re.sub(r"\|p([12])a: ([^|]+)", r" \2", l)[1:].replace("|", " ").strip() for l in ev))
    st = sb.state()
    print(f"\nresult: {'WIN' if st['winner'] == 'p1' else 'LOSS'} in {st['turn']} turns; ours left: {[(x['species'], x['hp']) for x in st['p1'] if not x['fainted']]}; theirs left: {[(x['species'], x['hp']) for x in st['p2'] if not x['fainted']]}")
    sb.close()
    bridge.close()


if __name__ == "__main__":
    main()
