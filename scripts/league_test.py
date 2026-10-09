"""The League alone: replay the no-heal block (Aaron .. Cynthia) from run states
saved at its opener (`plan_gauntlet.py --save-league DIR`), R times each.

Every variant is played from the same boxes, so a lever is compared on the
fight it is meant for, at a fraction of a full-game table's cost, and the
box-to-box spread drops out of the comparison (--vs pairs by snapshot).
Takes the planner's flags; rules flags (--no-items, --evs ...) apply from the
opener on, the box itself is what the saved run caught.

  uv run python scripts/league_test.py --snapshots data/league/tms_items --replays 4 \\
      --slice 405 --policy calc --search 9,8 --search-all --deep 9,16 --workers 10 --out logs/league_base.jsonl
  ... --league 10 --out logs/league_e8.jsonl --vs logs/league_base.jsonl
"""

import json
import multiprocessing as mp
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plan_gauntlet as pg  # noqa: E402
from nuzlocke_env import ORDER  # noqa: E402


def run_jobs(args, jobs: list[tuple[str, int]]) -> list[dict]:
    """One env (one sim) for this worker's (snapshot, replay) jobs."""
    env, rows = pg.make_env(args), []
    for path, j in jobs:
        t0 = time.time()
        for k, v in pickle.loads(Path(path).read_bytes()).items():
            setattr(env, k, v)
        env._risk_cache = {}
        env.rng = random.Random(f"{Path(path).stem}-{j}")  # evolutions / encounters after the opener
        start = env.trainer_idx
        r = pg.play(env, {"action_mask": env._mask()}, args.lam, args.verbose, args.fast,
                    args.parties, args.league, args.moves)
        last = args.slice - 1
        row = {"snapshot": Path(path).stem, "replay": j, "start": start, "reached": r["cleared"],
               "clear": r["slice_clear"], "deaths": r["deaths"],
               "champion_deaths": r["deaths_at"].get(last, 0)}
        rows.append(row)
        print(f"{row['snapshot']:>9} r{j}: reached #{r['cleared']}  deaths {r['deaths']}  "
              f"{'CLEAR' if r['slice_clear'] else ''}  ({time.time() - t0:.0f}s)", flush=True)
    return rows


def by_snapshot(rows: list[dict], f: str) -> dict[str, float]:
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["snapshot"], []).append(float(r[f]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    ap = pg.parser(__doc__)
    ap.add_argument("--snapshots", required=True, help="directory of seed<N>.pkl League-opener states")
    ap.add_argument("--replays", type=int, default=4, help="plays of the block per snapshot")
    ap.add_argument("--out", help="write one JSON row per (snapshot, replay) here")
    ap.add_argument("--vs", help="a previous --out file: print the difference, paired by snapshot")
    args = ap.parse_args()
    snaps = sorted(Path(args.snapshots).glob("seed*.pkl"))
    jobs = [(str(p), j) for p in snaps for j in range(args.replays)]
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.starmap(run_jobs, [(args, jobs[i::args.workers])
                                            for i in range(args.workers) if jobs[i::args.workers]])
    else:
        parts = [run_jobs(args, jobs)]
    rows = [r for p in parts for r in p]
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    def stat(f):  # spread over snapshots (each snapshot's replays averaged first): the box is the unit
        v = list(by_snapshot(rows, f).values())
        return np.mean(v), (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)

    print(f"\n{len(snaps)} snapshots x {args.replays} replays | battler={args.policy or 'SimpleHeuristics'} "
          f"search={args.search}{' (all)' if args.search_all else ''} deep={args.deep} parties={args.parties} "
          f"league={args.league} moves={args.moves} | rules: {pg.rules(args)} | {args.workers} workers, "
          f"{(time.time() - t0) / 60:.0f} min")
    for f, name in [("clear", "block clear rate"), ("deaths", "deaths per block"),
                    ("champion_deaths", "deaths at the champion")]:
        m, se = stat(f)
        print(f"  {name:<24} {m:6.2f} ± {se:.2f}")
    # a run that lost at t ended with trainer_idx = t; a cleared run with trainer_idx = slice
    fights = range(min((r["start"] for r in rows), default=args.slice), args.slice)
    print("  fights won / played: " + ", ".join(
        f"{ORDER[t]['trainer_name'].split()[-1]} {sum(r['reached'] > t for r in rows)}"
        f"/{sum(r['start'] <= t <= r['reached'] for r in rows)}" for t in fights))
    if args.vs:
        old = [json.loads(line) for line in Path(args.vs).read_text().splitlines()]
        for f, name in [("clear", "block clear rate"), ("deaths", "deaths per block")]:
            a, b = by_snapshot(rows, f), by_snapshot(old, f)
            d = [a[k] - b[k] for k in a if k in b]
            if len(d) > 1:
                print(f"  vs {Path(args.vs).name}: {name} {np.mean(d):+.2f} ± {np.std(d, ddof=1) / np.sqrt(len(d)):.2f} "
                      f"(paired over {len(d)} snapshots)")


if __name__ == "__main__":
    main()
