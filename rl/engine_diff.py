"""Disagreement test: play the same turn on Showdown and on poke-engine, compare outcomes.

Positions come from random-move gen8 random battles (diverse sets, real mid-battle
states). At each position both sides pick a random legal choice; poke-engine lists every
outcome with its probability, Showdown is forked --samples times with fresh RNG.

    uv run python rl/engine_diff.py --positions 300            # ~5 min
    uv run python rl/engine_diff.py --positions 50 --show 10   # print the worst cases

A position is flagged when the engines disagree beyond sampling noise:
  missing  Showdown outcomes poke-engine gives zero probability   (>= MISS)
  extra    poke-engine outcomes Showdown never produced           (>= MISS)
  tv       total variation distance between the outcome tables    (>= TV)
  hp       mean HP of either active mon differs                   (>= HP of max HP, and >= 1.5 HP:
           poke-engine keeps one damage value per outcome, so ~0.5 HP per hit is its resolution)
           (not checked when the turn can end the battle: Showdown then skips end-of-turn
           effects, poke-engine does not; the outcome table still counts the KO)
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from ai_util.sim import SimBridge
from rl import convert
from rl.convert import outcome_engine, outcome_showdown, to_engine

MISS, TV, HP = 0.10, 0.20, 0.05


def compare(bridge, bid, dump, c1, c2, samples):
    state, issues = to_engine(dump)
    a1, a2 = dump["p1"]["pokemon"][0], dump["p2"]["pokemon"][0]
    rec = {"turn": dump["turn"], "p1": a1["species"], "p2": a2["species"], "c1": c1[1], "c2": c2[1],
           "x1": [a1["ability"], a1["item"]], "x2": [a2["ability"], a2["item"]], "issues": issues}
    if issues:
        return rec
    try:
        branches = convert.pe.generate_instructions(state, convert.engine_choice(dump, "p1", c1, convert.RR),
                                                    convert.engine_choice(dump, "p2", c2, convert.RR))
    except BaseException as e:  # noqa: BLE001 — a Rust panic surfaces as PanicException (a BaseException)
        rec["engine_error"] = str(e)[:200]
        return rec
    eng, eng_hp = defaultdict(float), [0.0, 0.0]
    for b in branches:
        key, hp = outcome_engine(state.apply_instructions(b), dump)
        w = b.percentage / 100
        eng[key] += w
        eng_hp = [eng_hp[i] + w * hp[i] for i in range(2)]
    sd, sd_hp, ended = Counter(), [0.0, 0.0], False
    for i in range(samples):
        fid = f"{bid}f{i}"
        bridge.call(op="fork", id=bid, new=fid, reseed=True)
        bridge.call(op="choose", id=fid, p1=c1[0], p2=c2[0])
        after = bridge.call(op="dump", id=fid)
        ended |= after["ended"]
        key, hp = outcome_showdown(after)
        bridge.call(op="close", id=fid)
        sd[key] += 1 / samples
        sd_hp = [sd_hp[j] + hp[j] / samples for j in range(2)]
    keys = set(eng) | set(sd)
    rec |= {
        "missing": round(sum(p for k, p in sd.items() if eng.get(k, 0) == 0), 3),
        "extra": round(sum(p for k, p in eng.items() if k not in sd), 3),
        "tv": round(0.5 * sum(abs(eng.get(k, 0) - sd.get(k, 0)) for k in keys), 3),
        "dhp": [round(eng_hp[j] - sd_hp[j], 3) for j in range(2)],
        "branches": len(branches),
        "diffs": sorted(([round(eng.get(k, 0), 3), round(sd.get(k, 0), 3), k] for k in keys),
                        key=lambda r: -abs(r[0] - r[1]))[:4],
    }
    # 1.5 HP is the floor: at low levels 5% of max HP is a single hit point
    hp_bad = any(abs(rec["dhp"][j]) >= max(HP, 1.5 / dump[s]["pokemon"][0]["maxhp"]) for j, s in enumerate(("p1", "p2")))
    rec["flags"] = [f for f, bad in [("missing", rec["missing"] >= MISS), ("extra", rec["extra"] >= MISS),
                                      ("tv", rec["tv"] >= TV), ("hp", not ended and hp_bad)] if bad]
    if rec["flags"]:
        rec["dump"] = dump  # random teams cannot be regenerated; keep the position to debug it
    return rec


def positions(bridge, fmt, n, rng, teams=(None, None), arena=None):
    """Yield (battle id, dump, p1 choice, p2 choice) along random-move battles.
    teams: packed teams to replay one matchup; None = the format's random teams.
    arena: the bridge's fixed field effects (Radical Red bosses)."""
    count, game = 0, 0
    while count < n:
        bid = f"g{game}"
        game += 1
        seed = lambda: [rng.randrange(1 << 16) for _ in range(4)]  # noqa: E731
        # a seed per player fixes the random teams too, so --seed fixes the whole position set
        bridge.call(op="new", id=bid, format=fmt, seed=seed(), arena=arena,
                    p1={"name": "p1", "team": teams[0], "seed": seed()},
                    p2={"name": "p2", "team": teams[1], "seed": seed()})
        for _ in range(200):
            d = bridge.call(op="dump", id=bid)
            if d["ended"]:
                break
            c = {s: rng.choice(d[s]["choices"]) for s in ("p1", "p2") if d[s]["choices"]}
            if len(c) == 2 and not d["p1"]["forced"] and not d["p2"]["forced"] and count < n:
                count += 1
                yield bid, d, c["p1"], c["p2"]
            bridge.call(op="choose", id=bid, **{s: ch[0] for s, ch in c.items()})
        bridge.call(op="close", id=bid)


def rr_positions(bridge, n, rng, per=4):
    """Radical Red boss positions: for every hardcore boss in trainer order (scripts/rr_probe.py),
    a sampled player party (box at the cap, best4 moves with TMs) against the boss's sheet set
    under its arena, `per` positions each, round robin until n."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import rr_probe as rp
    from gauntlet import pack_team
    by_boss, known = rp.pools()
    bosses = sorted({row[3]: k for k, row in by_boss.items() if k.startswith("#")}.items())
    count = 0
    while count < n:
        for t, key in bosses:
            block, cap, pool, progress = by_boss[key]
            try:
                enemy, dump_cap, _ = rp.boss_team(t)
                party = rp.player_party(pool, max(cap, dump_cap), rng, known, progress)
            except Exception:  # noqa: BLE001 -- no team data (Lass Relli), an unbuildable box
                continue
            if not party:
                continue
            arena, _ = rp.arena(block.get("battleEffect", ""))
            for bid, d, c1, c2 in positions(bridge, rp.FMT, min(per, n - count), rng, (pack_team(party), enemy), arena):
                d["boss"] = t
                count += 1
                yield bid, d, c1, c2
            if count >= n:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--samples", type=int, default=200, help="Showdown replays per position")
    ap.add_argument("--format", default="gen8randombattle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/engine_diff.jsonl")
    ap.add_argument("--show", type=int, default=5, help="print the N worst flagged positions")
    ap.add_argument("--rr", action="store_true", help="Radical Red boss positions on the RR engine build")
    args = ap.parse_args()

    if args.rr:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import rr_probe as rp
        convert.use_rr()
        args.format = rp.FMT
    bridge = SimBridge(showdown=str(rp.FORK)) if args.rr else SimBridge()
    rng, recs = random.Random(args.seed), []
    gen = rr_positions(bridge, args.positions, rng) if args.rr else positions(bridge, args.format, args.positions, rng)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for bid, d, c1, c2 in gen:
            rec = compare(bridge, bid, d, c1, c2, args.samples)
            rec["boss"] = d.get("boss")
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if len(recs) % 25 == 0:
                print(f"  {len(recs)} positions", flush=True)
    bridge.close()

    n = len(recs)
    issues = [r for r in recs if r["issues"]]
    errors = [r for r in recs if "engine_error" in r]
    clean = [r for r in recs if "flags" in r]
    flagged = [r for r in clean if r["flags"]]
    print(f"\n{n} positions, {args.samples} Showdown samples each ({args.format})")
    print(f"  not convertible: {len(issues)} ({len(issues) / n:.0%})")
    for what, k in Counter(i for r in issues for i in set(r["issues"])).most_common(10):
        print(f"      {k:4d}  {what}")
    print(f"  engine error:    {len(errors)} ({len(errors) / n:.0%})")
    for what, k in Counter(r["engine_error"][:70] for r in errors).most_common(5):
        print(f"      {k:4d}  {what}")
    if clean:
        print(f"  compared:        {len(clean)}; disagree: {len(flagged)} ({len(flagged) / len(clean):.0%})")
        for flag, k in Counter(fl for r in flagged for fl in r["flags"]).most_common():
            print(f"      {k:4d}  {flag}")
        moves = Counter(m for r in flagged for m in (r["c1"], r["c2"]))
        base = Counter(m for r in clean for m in (r["c1"], r["c2"]))
        print("  choices most often in a disagreement (flagged / compared):")
        for m, k in moves.most_common(12):
            print(f"      {m:18s} {k}/{base[m]}")
    worst = sorted(flagged, key=lambda r: -(r["missing"] + r["extra"] + r["tv"]))[:args.show]
    for r in worst:
        print(f"\nturn {r['turn']}: {r['p1']} {r['x1']} {r['c1']}  vs  {r['p2']} {r['x2']} {r['c2']}  "
              f"flags={r['flags']} dhp={r['dhp']}")
        for pe, ps, k in r["diffs"]:
            print(f"   engine {pe:.2f}  showdown {ps:.2f}  {k}")


if __name__ == "__main__":
    main()
