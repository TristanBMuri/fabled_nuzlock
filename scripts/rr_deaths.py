"""Radical Red death analysis: replay a probe's best team against its boss under search (arena
on, legal moves, paired seeds) and classify every death from the protocol — which of ours
died, to which enemy mon and move, at what HP before the hit (one-shot from full vs chip),
on the turn it switched in or not, and whether it was our last mon (no choice left). The
question it answers: are deaths decisions or attrition?

  uv run python scripts/rr_deaths.py MISTY 12 [--source results/rr_bench_field.jsonl] [--death-weight 2.0] [--json]

`--json` also writes logs/deaths_<boss>_<variant>.json for the dashboard (ui/).
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def deaths_in(history: list[str]) -> list[dict]:
    """Every death of ours in one side's protocol lines, with its context."""
    out, active, hp, entered, alive = [], {"p1": None, "p2": None}, {}, {}, None
    last_move = None
    for i, line in enumerate(history):
        m = re.match(r"\|turn\|(\d+)", line)
        if m:
            turn = int(m.group(1))
            continue
        m = re.match(r"\|(?:switch|drag|replace)\|(p[12])a: ([^|]+)\|[^|]*\|(\d+)/(\d+)", line)
        if m:
            active[m.group(1)] = m.group(2)
            if m.group(1) == "p1":
                hp[m.group(2)] = int(m.group(3)) / int(m.group(4))
                entered[m.group(2)] = locals().get("turn", 0)
            continue
        m = re.match(r"\|move\|(p[12])a: ([^|]+)\|([^|]+)\|", line)
        if m:
            last_move = (m.group(1), m.group(2), m.group(3))
            continue
        m = re.match(r"\|-damage\|p1a: ([^|]+)\|(\d+)/(\d+)", line)
        if m:
            before = hp.get(m.group(1), 1.0)
            hp[m.group(1)] = int(m.group(2)) / int(m.group(3))
            hp[("before", m.group(1))] = before
            continue
        if re.match(r"\|-damage\|p1a: ([^|]+)\|0 fnt", line):
            name = re.match(r"\|-damage\|p1a: ([^|]+)\|", line).group(1)
            hp[("before", name)] = hp.get(name, 1.0)
            hp[name] = 0.0
            continue
        m = re.match(r"\|faint\|p1a: ([^|]+)", line)
        if m:
            name = m.group(1)
            cause = last_move if last_move and last_move[0] == "p2" else ("p2", active["p2"], "residual/other")
            out.append({"turn": locals().get("turn", 0), "ours": name, "enemy": active["p2"], "move": cause[2],
                        "hp_before": round(hp.get(("before", name), 1.0), 2), "switched_in_this_turn": entered.get(name) == locals().get("turn", 0)})
    return out


def main() -> None:
    import rr_probe as rp
    import gauntlet
    from gauntlet import STATS, Mon, encounter_moves, evolve, pack_team
    from poke_env.data import to_id_str
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBattle, SimBridge, player_brain

    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boss")
    ap.add_argument("n", type=int)
    ap.add_argument("--source", default="results/rr_bench_field.jsonl")
    ap.add_argument("--death-weight", type=float, default=0.5, help="the searcher's death penalty (outcome_value; 0.5 = the probe's)")
    ap.add_argument("--preserve", action="store_true", help="calc bot retreats low / badly poisoned mons (calc_bot.PRESERVE)")
    ap.add_argument("--matchup", action="store_true", help="calc bot switches on the matchup matrix (calc_bot.MATCHUP)")
    ap.add_argument("--blind-field", action="store_true", help="calc bot ignores weather / terrain (calc_bot.FIELD_AWARE = False)")
    ap.add_argument("--json", action="store_true", help="also write logs/deaths_<boss>_<variant>.json")
    args = ap.parse_args()
    boss, n, source = args.boss, args.n, args.source
    gauntlet.BANNED_MOVES = {to_id_str(m) for m in json.load(open(ROOT / "data/rr/hardcore_rules.json"))["banned_moves"]}
    r = {x["boss"]: x for x in map(json.loads, open(ROOT / source))}[boss]
    by, known = rp.pools()
    block, cap, pool, t = by[boss]
    arena, _ = rp.arena(block.get("battleEffect", ""))
    enemy_packed, dump_cap, _ = rp.boss_team(int(boss[1:]) if boss.startswith("#") else boss)
    cap = max(cap, dump_cap)
    md = rp.load_game("rr")["moves"]
    boss_types = {md[mv]["type"] for mon in enemy_packed.split("]") for mv in mon.split("|")[4].split(",") if mv in md}
    party = []
    for sp in r["best_team"]:
        ev = evolve(sp, cap, None, caught_level=5, trainer_idx=None)
        mv = encounter_moves(ev, cap, "best4", progress=t)
        m = Mon(ev, cap, mv, ability=rp.choose_ability(ev, known, boss_types, arena), caught_level=5, nature=rp.choose_nature(ev, mv, known))
        m.evs = dict.fromkeys(STATS, 0)
        party.append(m)
    print("team:", [(m.species, m.nature, m.ability, m.moves) for m in party])
    print("enemy:", [(x.split("|")[0], x.split("|")[2], x.split("|")[3], x.split("|")[4]) for x in enemy_packed.split("]")])
    bridge = SimBridge(showdown=str(rp.FORK))
    enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    calc = calc_brain()
    if args.preserve or args.matchup or args.blind_field:
        import ai_util.calc_bot as cb
        cb.PRESERVE, cb.MATCHUP, cb.FIELD_AWARE = args.preserve, args.matchup, not args.blind_field
    searcher = Searcher(None, enemy, k=9, rollouts=8, rollout=calc, seed=0, death_weight=args.death_weight)
    print(f"death weight {args.death_weight}, preserve {args.preserve}, matchup {args.matchup}, field-aware {not args.blind_field}")
    srng = random.Random(f"{boss}-seeds-0")
    seeds = [[srng.randrange(1, 2 ** 16) for _ in range(4)] for _ in range(n)]
    all_deaths, wins, clean, lengths = [], 0, 0, []
    for i in range(n):
        import numpy as np
        random.seed(sum(seeds[i]))
        searcher.rng = np.random.default_rng(seeds[i])
        sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena, seed=seeds[i])
        sb.play({"p1": searcher, "p2": enemy}, 100)
        st = sb.state()
        wins += st["winner"] == "p1"
        d = deaths_in(sb.history["p1"])
        clean += st["winner"] == "p1" and not d
        for x in d:
            x["battle"], x["won"] = i, st["winner"] == "p1"
        all_deaths += d
        lengths.append(st["turn"])
        sb.close()
    print(f"\n{boss}: {wins}/{n} wins, {clean} deathless, {len(all_deaths)} deaths ({len(all_deaths) / n:.1f} per fight), fights last {sum(lengths) / n:.0f} turns")
    C = collections.Counter
    counters = {
        "deaths by enemy mon": C(x["enemy"] for x in all_deaths).most_common(),
        "deaths by enemy move": C(f"{x['enemy']} {x['move']}" for x in all_deaths).most_common(8),
        "which of ours": C(x["ours"] for x in all_deaths).most_common(),
        "HP before the killing hit": C("full (>=0.9)" if x["hp_before"] >= 0.9 else "half+ (0.5-0.9)" if x["hp_before"] >= 0.5 else "low (<0.5)" for x in all_deaths).most_common(),
        "switched in that turn": C("switched in" if x["switched_in_this_turn"] else "was in" for x in all_deaths).most_common(),
        "fight outcome": C("won" if x["won"] else "lost" for x in all_deaths).most_common(),
        "by turn": sorted(C(min(x["turn"] // 5 * 5, 30) for x in all_deaths).items()),
    }
    for k, v in counters.items():
        print(f"{k}:", v)
    if args.json:
        variant = f"dw{args.death_weight:g}" + "".join(f"_{f}" for f, on in (("preserve", args.preserve), ("matchup", args.matchup), ("blind", args.blind_field)) if on)
        out = ROOT / "logs" / f"deaths_{re.sub(r'[^a-z0-9]+', '_', boss.lower()).strip('_')}_{variant}.json"
        out.write_text(json.dumps({"boss": boss, "variant": variant, "source": source, "team": r["best_team"], "n": n, "wins": wins,
                                   "deathless": clean, "turns": sum(lengths) / n, "deaths": all_deaths,
                                   "counters": {k: [[str(a), b] for a, b in v] for k, v in counters.items()}}))
        print(f"wrote {out}")
    bridge.close()


if __name__ == "__main__":
    main()
