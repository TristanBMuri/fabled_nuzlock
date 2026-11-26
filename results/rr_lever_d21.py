"""Counterfactuals at the fight that ended a logged run (results/rr_runs_d21.jsonl).

Replays the run's box (same sampling as rr_run.run) and its logged deaths up to the losing fight,
then plays the losing fight's box with several (chooser, engine iterations) arms on the same N
battle seeds. Usage: lever_test.py SEED [SEED ...] --n 10 --out file.jsonl
"""
import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path("/home/tristan/CodingProjects/nuzlock_rl")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import rr_probe as rp  # noqa
import rr_team_probe as tp  # noqa
import rr_run  # noqa
from poke_env.data import to_id_str  # noqa

ARMS = [("greedy", 2000), ("greedy", 20000), ("greedy", 200000), ("cover", 20000), ("searched", 20000), ("searched", 200000)]


def replay(seed, log):
    rng = random.Random(f"run-{seed}")
    starter = rng.choice(rp.STARTERS)
    box, seen, caught = [starter], {"starter"}, {starter}
    by, known = rp.pools()
    rows = {f["t"]: f for f in log["fights"] if "won" in f}
    target = log["last"]["t"]
    for t in sorted(rows):
        for sp, where in rp.sample_box(random.Random(f"run-{seed}-{t}"), t, known, starter=starter):
            key = where.split(" / ")[0]
            if key in seen:
                continue
            seen.add(key)
            if sp not in caught or sp == "eevee":
                caught.add(sp)
                box.append(sp)
        if t == target:
            return t, box
        args = rr_run.probe_args(seed, "greedy")
        ctx = tp.setup(args, f"#{t}", announce=False)
        died = set(rows[t]["died"])
        for k in list(box):
            m = tp.mon_of(ctx, k)
            if m is not None and m.species in died:
                box.remove(k)
        ctx.bridge.close()
    raise RuntimeError("target not reached")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logs = {json.loads(l)["seed"]: json.loads(l) for l in open(ROOT / "results/rr_runs_d21.jsonl")}
    for seed in a.seeds:
        log = logs[seed]
        t, box = replay(seed, log)
        bseeds = [[random.Random(f"lever-{seed}-{i}-{j}").randrange(1, 2 ** 16) for j in range(4)] for i in range(a.n)]
        teams = {}
        for chooser, iters in ARMS:
            args = rr_run.probe_args(seed, chooser)
            ctx = tp.setup(args, f"#{t}", announce=False)
            alive = [sp for sp in box if tp.mon_of(ctx, sp) is not None]
            t0 = time.time()
            if chooser not in teams:
                teams[chooser] = tp.choose(ctx, alive)
            choose_s = time.time() - t0
            team = teams[chooser]
            party = tp.team_mons(ctx, team)
            ctx.searcher.iterations = iters
            ctx.searcher.stats.clear()
            t0 = time.time()
            res = [tp.one_battle(ctx.bridge, ctx.calc, ctx.enemy, party, ctx.enemy_packed, i, search=ctx.searcher,
                                 arena=ctx.arena, seeds=bseeds, boss=f"#{t}") for i in range(a.n)]
            row = {"seed": seed, "t": t, "name": log["last"]["name"], "box": len(alive), "logged_team": log["last"]["team"],
                   "chooser": chooser, "iters": iters, "team": [m.species for m in party],
                   "won": [bool(r[0]) for r in res], "deathless": [bool(r[1]) for r in res],
                   "deaths": [round(r[2] * len(party)) for r in res], "capped": [bool(r[6]) for r in res],
                   "battle_s": round((time.time() - t0) / a.n, 1), "choose_s": round(choose_s, 1),
                   "engine": dict(ctx.searcher.stats)}
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            ctx.bridge.close()


if __name__ == "__main__":
    main()
