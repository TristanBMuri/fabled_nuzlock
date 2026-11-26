"""At each logged run's Misty loss: the alive box vs the box with no deaths (everything caught), greedy and cover choosers."""
import json, random, sys, time
sys.path.insert(0, "/tmp/claude-1001/-home-tristan-CodingProjects-nuzlock-rl/136815ab-a6f5-4760-8532-472a1af6a308/scratchpad")
import lever_test as L
from lever_test import tp, rp, rr_run, ROOT

def main():
    seeds, n, out = [int(s) for s in sys.argv[1].split(",")], 8, sys.argv[2]
    logs = {json.loads(l)["seed"]: json.loads(l) for l in open(ROOT / "results/rr_runs_d21.jsonl")}
    for seed in seeds:
        log = logs[seed]
        t, alive_box = L.replay(seed, log)
        # full box: replay sampling only, no removals
        rng = random.Random(f"run-{seed}"); starter = rng.choice(rp.STARTERS)
        full, seen, caught = [starter], {"starter"}, {starter}
        by, known = rp.pools()
        for tt in sorted(f["t"] for f in log["fights"] if "won" in f):
            for sp, where in rp.sample_box(random.Random(f"run-{seed}-{tt}"), tt, known, starter=starter):
                key = where.split(" / ")[0]
                if key in seen: continue
                seen.add(key)
                if sp not in caught or sp == "eevee":
                    caught.add(sp); full.append(sp)
        bseeds = [[random.Random(f"lever-{seed}-{i}-{j}").randrange(1, 2 ** 16) for j in range(4)] for i in range(n)]
        for boxname, box in (("alive", alive_box), ("full", full)):
            for chooser in ("greedy", "cover"):
                ctx = tp.setup(rr_run.probe_args(seed, chooser), f"#{t}", announce=False)
                avail = [sp for sp in box if tp.mon_of(ctx, sp) is not None]
                party = tp.team_mons(ctx, tp.choose(ctx, avail))
                res = [tp.one_battle(ctx.bridge, ctx.calc, ctx.enemy, party, ctx.enemy_packed, i, search=ctx.searcher,
                                     arena=ctx.arena, seeds=bseeds, boss=f"#{t}") for i in range(n)]
                row = {"seed": seed, "t": t, "box": boxname, "size": len(avail), "chooser": chooser,
                       "team": [m.species for m in party], "won": [bool(r[0]) for r in res],
                       "deaths": [round(r[2] * len(party)) for r in res],
                       "dead_before": [m for f in log["fights"] if f.get("won") for m in f["died"]]}
                print(json.dumps(row), flush=True)
                open(out, "a").write(json.dumps(row) + "\n")
                ctx.bridge.close()

main()
