"""Simulated hardcore Nuzlocke runs through Radical Red's trainer order: how far does the battler get?

One run (one process): a random starter; the box grows as locations open (one encounter per location,
drawn when it is first reached, dupes skipped; story gifts as offered). Before each mandatory fight of
data/rr/order.json the probe's chooser picks six from the living box (all of them under six), at the
level cap with the items in hand, and plays ONE searched battle (the box-probe settings). Every mon that
faints dies; a loss ends the run. Healed between fights.

Not modelled: doubles fights (played as singles), optional fights, Celadon eggs, in-game trades,
shops, evolution timing (the probe's cap rule), the player's own catch / route choices (random).

    uv run python scripts/rr_run.py --runs 4 --workers 4 --seed 1 --out results/rr_runs_s1
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import rr_probe as rp  # noqa: E402
import rr_team_probe as tp  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402

ORDER = json.load(open(rp.RR / "order.json"))["entries"]


def probe_args(seed: int, chooser: str) -> argparse.Namespace:
    """The box probe's settings (results/rr_d19_all): greedy chooser, best4 sets, engine search 20000."""
    return argparse.Namespace(
        bosses="", n_solo=3, n_team=4, n_search=6, n_duel=4, search="9,8", seed=seed, workers=1, legal=True, normal=False,
        arena=True, blind_field=False, repair=False, boxes=0, chooser=chooser, cands=24, n_screen=4, matchup=True,
        death_weight=6.0, no_items=False, items=True, engine=20000, engine_p=0.8, enemy="python", states=False,
        engine_opp="ai", sets="best4", alt=0, out=None)


def run(seed: int, chooser: str, out: str | None, max_fights: int = 0) -> dict:
    rng = random.Random(f"run-{seed}")
    starter = rng.choice(rp.STARTERS)
    box, seen_where, caught = [starter], {"starter"}, {starter}
    dead, cleared, log = [], [], []
    by, known = rp.pools()
    fights = [t for t in range(len(ORDER)) if f"#{t}" in by and not ORDER[t]["optional"]][:max_fights or None]
    t0 = time.time()
    for t in fights:
        name = f"#{t}"
        for sp, where in rp.sample_box(random.Random(f"run-{seed}-{t}"), t, known, starter=starter):
            key = where.split(" / ")[0]  # the location (a wild encounter) or the gift
            if key in seen_where:
                continue
            seen_where.add(key)
            if sp not in caught or sp == "eevee":  # dupes clause (the 2nd Eevee gift is allowed)
                caught.add(sp)
                box.append(sp)
        args = probe_args(seed, chooser)
        try:
            ctx = tp.setup(args, name, announce=False)
        except KeyError as e:  # no team data for this fight: skip it
            log.append({"t": t, "name": ORDER[t]["name"], "skipped": str(e)})
            continue
        alive = [sp for sp in box if tp.mon_of(ctx, sp) is not None]
        team = tp.choose(ctx, alive)
        party = tp.team_mons(ctx, team)
        r = tp.one_battle(ctx.bridge, ctx.calc, ctx.enemy, party, ctx.enemy_packed, 0, search=ctx.searcher,
                          arena=ctx.arena, seeds=ctx.seeds, boss=name)
        won, fainted = r[0], set(r[7])
        died = [s for s, m in zip(team, party) if to_id_str(m.species) in fainted]
        for s in died:
            if s.split(">")[0] in box:  # a branched key ("eevee>vaporeon") is its caught mon
                box.remove(s.split(">")[0])
        dead += [tp.mon_of(ctx, s).species for s in died]
        ctx.bridge.close()
        row = {"t": t, "name": ORDER[t]["name"], "cap": ctx.cap, "box": len(alive), "team": [m.species for m in party],
               "won": won, "died": [tp.mon_of(ctx, s).species for s in died], "turn_cap": r[6], "s": round(time.time() - t0)}
        log.append(row)
        print(f"run {seed}: #{t:<3} {ORDER[t]['name']:<20} cap {ctx.cap:>2} box {len(alive):>2} {'WON ' if won else 'LOST'} "
              f"died {row['died'] or '-'}  ({row['s']}s)", flush=True)
        if not won:
            break
        cleared.append(t)
    res = {"seed": seed, "starter": starter, "cleared": len(cleared), "of": len(fights), "last": log[-1] if log else None,
           "deaths": dead, "minutes": round((time.time() - t0) / 60, 1), "fights": log}
    if out:
        with open(f"{out}.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1, help="runs use seeds seed .. seed + runs - 1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chooser", choices=("greedy", "safe", "cover"), default="greedy", help="safe (D20): greedy on survivors after a win")
    ap.add_argument("--out", help="append one JSON line per run to <out>.jsonl")
    ap.add_argument("--max-fights", type=int, default=0, help="stop each run after this many fights (smoke test)")
    args = ap.parse_args()
    jobs = [(s, args.chooser, args.out, args.max_fights) for s in range(args.seed, args.seed + args.runs)]
    with mp.get_context("spawn").Pool(min(args.workers, args.runs)) as pool:
        res = pool.starmap(run, jobs)
    print(f"\n{len(res)} runs, {res[0]['of']} mandatory fights each")
    for r in sorted(res, key=lambda r: -r["cleared"]):
        last = r["last"] or {}
        print(f"  seed {r['seed']:>3} starter {r['starter']:<12} cleared {r['cleared']:>3}/{r['of']}  "
              f"{'lost to' if last and not last.get('won') else 'ended at'} #{last.get('t')} {last.get('name')}  "
              f"deaths {len(r['deaths'])}  {r['minutes']} min")


if __name__ == "__main__":
    main()
