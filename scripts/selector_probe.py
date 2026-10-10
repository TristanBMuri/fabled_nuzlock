"""Selector transfer diagnostic (2026-10-10): does the fast-brain party score
rank League parties the way search play does?

For every saved League opener (`plan_gauntlet.py --save-league DIR`): rank the
planner's whole candidate set (base + single swaps) by its score (the block
estimate under the calc bot, n=8), then play the candidates at a few ranks
through the whole block with the planner's search settings. Reports the
Spearman correlation between fast score and search-play deaths (pooled and
within box), deaths / clears per sampled rank, and the regret of the fast
winner (rank 0, what the planner fields) against the best sampled alternative.

  uv run python scripts/selector_probe.py --snapshots results/league_snapshots --slice 405 \\
      --policy calc --search 9,8 --search-all --deep 9,16 --workers 10 --out logs/selector_probe.jsonl
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
from nuzlocke_env import MAX_PARTY, ORDER  # noqa: E402


def restore(env, path: str, tag: str):
    for k, v in pickle.loads(Path(path).read_bytes()).items():
        setattr(env, k, v)
    env._risk_cache = {}
    env.rng = random.Random(tag)  # the same evolutions on every copy of this opener
    env._apply_level_rule(ORDER[env.trainer_idx])


def run_jobs(args, jobs: list[tuple[str, int]]) -> list[dict]:
    env, rows = pg.make_env(args), []
    for path, j in jobs:
        try:
            rows += run_job(env, args, path, j)
        except Exception as e:  # keep the finished plays; the summary says how many boxes
            print(f"{Path(path).stem} r{j} FAILED: {type(e).__name__}: {e}", flush=True)
    return rows


def run_job(env, args, path: str, j: int) -> list[dict]:
    rows = []
    if True:
        tag, stem = f"{Path(path).stem}-{j}", Path(path).stem
        t0 = time.time()
        restore(env, path, tag)
        assert env._block(), "not at a League opener"
        alive = [i for i, m in enumerate(env.roster) if m.alive]
        base = sorted(alive, key=lambda i: -env.roster[i].level)[:MAX_PARTY]
        cands = pg.swaps(base, alive)
        # the estimate is read here, on the scoring copy: a played block leaves the roster shorter
        scored = sorted(((pg.party_score(env, p, args.lam), p, env.risk(tuple(p))) for p in cands), key=lambda x: -x[0])
        n = len(scored)
        ranks = sorted({min(int(r * (n - 1)) if isinstance(r, float) else r, n - 1) for r in args.ranks})
        print(f"{stem} r{j}: {n} candidates scored in {time.time() - t0:.0f}s, playing ranks {ranks}", flush=True)
        for rank in ranks:
            score, party, (p_win, p_death, progress) = scored[rank]
            t1 = time.time()
            restore(env, path, tag)
            species = [env.roster[i].species for i in party]
            before = sum(m.alive for m in env.roster)
            for i in party:
                env.step(1 + i)
            env.step(0)  # the opener, searched like any breakpoint; the block follows locked
            if not env.done:
                pg.play(env, args.lam, False, False)  # the rest of the block, party locked
            deaths = before - sum(m.alive for m in env.roster)
            row = {"snapshot": stem, "replay": j, "rank": rank, "n_candidates": n, "fast_score": score,
                   "fast_p_win": p_win, "fast_p_death": float(np.mean(p_death)), "fast_progress": progress,
                   "party": species,
                   "reached": env.trainer_idx, "clear": env.trainer_idx >= env.slice_len, "deaths": deaths}
            rows.append(row)
            print(f"{stem} r{j} rank {rank:3d}: fast {score:+.2f} -> reached #{row['reached']} deaths {deaths} "
                  f"{'CLEAR' if row['clear'] else ''} ({time.time() - t1:.0f}s)", flush=True)
    return rows


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # ties: average rank
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def report(rows: list[dict]) -> None:
    fs, de, cl = [r["fast_score"] for r in rows], [r["deaths"] for r in rows], [float(r["clear"]) for r in rows]
    print(f"\n{len(rows)} (box, candidate) plays over {len({r['snapshot'] for r in rows})} boxes")
    print(f"  Spearman(fast score, search-play deaths) pooled: {spearman(fs, de):+.2f}   "
          f"(fast score, clear): {spearman(fs, cl):+.2f}")
    by_box: dict[str, list] = {}
    for r in rows:
        by_box.setdefault(r["snapshot"], []).append(r)
    within = [spearman([r["fast_score"] for r in v], [r["deaths"] for r in v]) for v in by_box.values()]
    within = [w for w in within if not np.isnan(w)]
    print(f"  within-box Spearman(fast score, deaths): mean {np.mean(within):+.2f} ± {np.std(within, ddof=1) / np.sqrt(len(within)):.2f} "
          f"over {len(within)} boxes (negative = the fast score orders parties the way search play does)")
    print("  by sampled slot (slot 0 = rank 0 = what the planner fields; later slots = deeper in the fast ranking):")
    by_slot: dict[int, list] = {}
    for v in by_box.values():
        for slot, r in enumerate(sorted(v, key=lambda r: r["rank"])):
            by_slot.setdefault(slot, []).append(r)
    for k in sorted(by_slot):
        v = by_slot[k]
        ranks = sorted({r["rank"] for r in v})
        print(f"    slot {k}: ranks {ranks[0]}..{ranks[-1]:<3d} fast {np.mean([r['fast_score'] for r in v]):+.2f}  "
              f"deaths {np.mean([r['deaths'] for r in v]):.2f} ± {np.std([r['deaths'] for r in v], ddof=1) / np.sqrt(len(v)):.2f}  "
              f"clear {np.mean([r['clear'] for r in v]):.0%}  (n={len(v)})")
    regret, beat, best_rank = [], 0, []
    for v in by_box.values():
        top = [r for r in v if r["rank"] == 0]
        if not top:
            continue
        best = min(r["deaths"] for r in v)
        regret.append(top[0]["deaths"] - best)
        beat += top[0]["deaths"] == best
        best_rank.append(min(v, key=lambda r: (r["deaths"], not r["clear"]))["rank"])
    print(f"  regret of the fast winner: {np.mean(regret):.2f} ± {np.std(regret, ddof=1) / np.sqrt(len(regret)):.2f} deaths "
          f"vs the best sampled party; the fast winner was (tied) best in {beat}/{len(regret)} boxes; "
          f"best sampled rank median {int(np.median(best_rank))}")


def main() -> None:
    ap = pg.parser(__doc__)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--replays", type=int, default=1, help="plays of the sampled set per box")
    ap.add_argument("--ranks", default="0,1,2,0.125,0.25,0.5,0.75,0.999",
                    help="candidate ranks to play: integers, or fractions of the candidate count")
    ap.add_argument("--limit", type=int, default=0, help="only the first N snapshots (smoke)")
    ap.add_argument("--out", help="jsonl of one row per (box, candidate) play; or an existing one to re-report")
    args = ap.parse_args()
    args.ranks = [float(r) if "." in r else int(r) for r in args.ranks.split(",")]
    if args.out and Path(args.out).exists() and args.limit < 0:
        report([json.loads(l) for l in Path(args.out).read_text().splitlines()])
        return
    snaps = sorted(Path(args.snapshots).glob("seed*.pkl"))[: args.limit or None]
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
    print(f"\nbattler={args.policy} search={args.search}{' (all)' if args.search_all else ''} deep={args.deep} "
          f"full_info={args.full_info} | rules: {pg.rules(args)} | {args.workers} workers, {(time.time() - t0) / 60:.0f} min")
    report(rows)


if __name__ == "__main__":
    main()
