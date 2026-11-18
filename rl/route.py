"""Router: play a turn on poke-engine only where it agrees with Showdown.

Two levels:
  route(matchup)   once per matchup (both packed teams), cached:
                     names  every species / ability / item / move is known to poke-engine
                     probe  --positions random positions of this matchup; each one that
                            turn_gaps lets through is replayed --samples times on Showdown
                            (engine_diff.compare). More than --tolerate disagreements -> Showdown
  turn_gaps(dump)  every turn, microseconds: a known gap (GAPS) on either active mon, or
                   something the converter cannot carry (a two-turn move mid-charge), sends
                   that turn to Showdown. A gap on the bench does not matter until it is in.

Every matchup decision is one line of logs/route.jsonl (its own log, nothing else writes
there); that file is also the cache, so a matchup is probed once.

    uv run python rl/route.py --harvest data/teams/gauntlet_harvest/battles_best4.jsonl --trainers 20
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from ai_util.sim import SimBridge
from rl.convert import to_engine
from rl.engine_diff import compare, positions

FORMAT = "gen8customgamenogimmicks"
LOG = Path("logs/route.jsonl")
# what poke-engine still gets wrong (rl/README.md -> Can we trust it?)
GAPS = {"ability:imposter": "no Transform", "move:transform": "no Transform",
        "move:doomdesire": "no Doom Desire", "ability:soulheart": "Soul-Heart counts only its own KOs",
        "ability:rivalry": "no genders", "move:rollout": "no lock-in / doubling",
        "move:iceball": "no lock-in / doubling"}


def _features(mon: dict) -> set[str]:
    return {f"ability:{mon['ability']}", f"item:{mon['item']}"} | {f"move:{m}" for m, _ in mon["moves"]}


def turn_gaps(dump: dict) -> list[str]:
    """Why this turn must run on Showdown; empty = poke-engine is trusted for it."""
    active = _features(dump["p1"]["pokemon"][0]) | _features(dump["p2"]["pokemon"][0])
    return sorted(f for f in active if f in GAPS)


def load_cache() -> dict:
    return {r["key"]: r for r in map(json.loads, LOG.open())} if LOG.exists() else {}


def route(bridge: SimBridge, p1: str, p2: str, cache: dict, fmt: str = FORMAT, n: int = 50,
          samples: int = 200, tolerate: int = 0, seed: int = 0, **tag) -> dict:
    """{"engine": bool, "gate", "why", "probe"} for this matchup; logged and cached."""
    key = hashlib.sha1(f"{fmt}|{p1}|{p2}|{n}|{samples}|{tolerate}".encode()).hexdigest()[:16]
    if key in cache:
        return cache[key]
    t0 = time.time()
    bridge.call(op="new", id="route", format=fmt, p1={"name": "p1", "team": p1}, p2={"name": "p2", "team": p2})
    d = bridge.call(op="dump", id="route")
    bridge.call(op="close", id="route")
    _, issues = to_engine(d)
    rec = {"key": key, **tag}
    if issues:
        rec |= {"engine": False, "gate": "names", "why": sorted(set(issues))}
    else:
        bad, gapped, skipped = [], Counter(), 0
        for bid, dd, c1, c2 in positions(bridge, fmt, n, random.Random(seed), (p1, p2)):
            if gaps := turn_gaps(dd):
                gapped.update(gaps)
                skipped += 1
                continue
            r = compare(bridge, bid, dd, c1, c2, samples)
            if r.get("flags") or "engine_error" in r:
                bad.append(f"{r['c1']} vs {r['c2']}: {r.get('engine_error', '')[:40] or ','.join(r['flags'])}")
        rec |= {"engine": len(bad) <= tolerate, "gate": "probe", "why": bad,
                "probe": {"positions": n, "engine_turns": n - skipped, "disagree": len(bad),
                          "gaps": dict(gapped)}}
    rec["seconds"] = round(time.time() - t0, 1)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    cache[key] = rec
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True, help="jsonl with trainer_id, player_team, enemy_team")
    ap.add_argument("--trainers", type=int, default=20, help="route the first matchup of this many trainers")
    ap.add_argument("--positions", type=int, default=50)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--tolerate", type=int, default=0, help="disagreements a probe may have and still pass")
    args = ap.parse_args()

    first = {}
    for row in map(json.loads, open(args.harvest)):
        first.setdefault(row["trainer_id"], row)
    bridge, cache, recs = SimBridge(), load_cache(), []
    for tid, row in list(first.items())[:args.trainers]:
        rec = route(bridge, row["player_team"], row["enemy_team"], cache, n=args.positions,
                    samples=args.samples, tolerate=args.tolerate, trainer=tid)
        recs.append(rec)
        share = f"{rec['probe']['engine_turns']:3d}/{args.positions}" if "probe" in rec else "   -  "
        print(f"trainer {tid:4d}  {'engine  ' if rec['engine'] else 'showdown'}  turns on engine {share}"
              f"  {rec['seconds']:5.1f}s  {'; '.join(rec['why'][:2])}", flush=True)
    bridge.close()
    probed = [r for r in recs if "probe" in r]
    turns = sum(r["probe"]["engine_turns"] for r in probed if r["engine"])
    print(f"\n{sum(r['engine'] for r in recs)}/{len(recs)} matchups pass; {turns}/{len(recs) * args.positions}"
          f" probe turns would run on poke-engine; to Showdown by: "
          f"{dict(Counter(r['gate'] for r in recs if not r['engine']))}")


if __name__ == "__main__":
    main()
