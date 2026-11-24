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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a script from anywhere

from ai_util.sim import SimBridge
from rl import convert
from rl.convert import outcome_engine, outcome_showdown, to_engine

MISS, TV, HP = 0.10, 0.20, 0.05


def _engine_choice(dump, side, c):
    """The engine's name for a dump choice; the RR fork mega-evolves a stone holder on its first
    move (both sides), which the RR build names `<move>-mega`."""
    name = convert.engine_choice(dump, side, c, convert.RR)
    mega = convert.RR and dump[side].get("canMega") and convert._RR_ITEMS.get(dump[side]["pokemon"][0]["item"]) == "mega"
    return name + "-mega" if mega and c[0].startswith("move") else name


def compare(bridge, bid, dump, c1, c2, samples):
    state, issues = to_engine(dump)
    a1, a2 = dump["p1"]["pokemon"][0], dump["p2"]["pokemon"][0]
    rec = {"turn": dump["turn"], "p1": a1["species"], "p2": a2["species"], "c1": c1[1], "c2": c2[1],
           "x1": [a1["ability"], a1["item"]], "x2": [a2["ability"], a2["item"]], "issues": issues}
    if issues:
        return rec
    try:
        branches = convert.pe.generate_instructions(state, _engine_choice(dump, "p1", c1), _engine_choice(dump, "p2", c2))
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
        "_eng": dict(eng), "_sd": dict(sd),  # the outcome tables (from_states reads them; not written)
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


LUCK, CLASSES = 0.15, ("converter", "engine", "ai", "luck", "search")


def _battles(path: str, seed: int):
    """A --states log's battles: (battle line, its decisions, why selected, control?). Selected:
    a loss or >= 2 deaths, a death after a root value >= 0.8, a one-turn root-value drop >= 0.4,
    a timeout; control: a random 5% of all battles."""
    rows, open_, out, rng = [json.loads(x) for x in open(path)], defaultdict(list), [], random.Random(seed)
    key = lambda r: (r.get("boss"), r.get("battle"), str(r.get("battle_seed")), str(r.get("team")))  # noqa: E731
    for r in rows:
        if r["kind"] == "decision":
            open_[key(r)].append(r)
            continue
        ds = [d for d in open_.pop(key(r), []) if d["side"] == "p1"]
        val = [max(d["options"], key=lambda o: o[1])[2] for d in ds]
        alive = [sum(p.hp > 0 for p in convert.pe.State.from_string(d["state"]).side_one.pokemon) for d in ds]
        why = [w for w, bad in [("loss", not r["won"]), ("2+ deaths", sum(r["fainted"]) >= 2), ("timeout", r["timed_out"]),
                                ("death at value>=0.8", any(v >= 0.8 and b < a for v, a, b in zip(val, alive, alive[1:]))),
                                ("drop>=0.4", any(v - w >= 0.4 for v, w in zip(val, val[1:])))] if bad]
        out.append((r, ds, why, rng.random() < 0.05))
    return out


def _match(choices: list, pick: str | None):
    """The dump's [message, id] for a sent choice: "move 2", or a poke-env brain's "move knockoff" / "switch Swalot"."""
    kind, _, what = (pick or "").removesuffix(" mega").partition(" ")
    what = re.sub(r"[^a-z0-9]", "", what.lower())
    return next((x for x in choices if x[0] == pick or (x[0].startswith(kind) and x[1] == what)), None)


def replay(job) -> dict:
    """One logged battle replayed on Showdown from its seed and both sides' logged choices; at
    every turn both sides choose: compare() with the two choices made, the RR AI model's
    options vs the enemy's actual choice, the searcher's converter gaps (why_not), and the
    engine's probability of what actually happened. The turn's class is the first of CLASSES
    that applies (luck: only on a turn that kills one of ours)."""
    path, (r, ds, why, ctl), samples = job
    import rr_probe as rp
    from gauntlet import Mon, pack_team
    from ai_util.sim import SimBattle, SimBridge
    from rl.engine_search import EngineSearcher
    convert.use_rr()
    evs = [json.loads(x) for x in open(Path(path).with_suffix("").with_suffix(".events.jsonl"))]
    cap = next(e["cap"] for e in evs if e["kind"] == "boss_start" and e["boss"] == r["boss"])
    by, _ = rp.pools()
    arena, _ = rp.arena(by[r["boss"]][0].get("battleEffect", ""))
    enemy = rp.boss_team(r["boss"])[0]
    out = {"boss": r["boss"], "battle": r["battle"], "team": r["team"], "won": r["won"], "deaths": sum(r["fainted"]),
           "why": why, "control": ctl, "turns": [], "first": None}
    teams = {pack_team([Mon(s["species"], cap, s["moves"], s["ability"], s["item"] or None, nature=s["nature"],
                            evs=dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 0)) for s in e["sets"]])
             for e in evs if e["kind"] in ("team", "box") and e["boss"] == r["boss"] and [s["species"] for s in e["sets"]] == r["team"]}
    bridge, es = SimBridge(showdown=str(rp.FORK)), EngineSearcher(None)
    for team in sorted(teams):  # several sets for this six: the one whose first position is the logged one
        sb = SimBattle(bridge, rp.FMT, team, enemy, full_info=True, arena=arena, seed=r["battle_seed"])
        if len(teams) == 1 or not ds or convert.to_engine(bridge.call(op="dump", id=sb.id))[0].to_string() == ds[0]["state"]:
            break
        sb.close()
    else:
        out["error"] = "no logged set reproduces the first position"
        bridge.close()
        return out
    chosen, logged = [tuple(c) for c in r["chosen"]], {d["turn"] for d in ds}
    while not sb.ended and chosen:
        if sb.turn > 100:
            break
        picks = {s: chosen.pop(0)[2] for s in ("p1", "p2") if sb.pending[s]}
        d = bridge.call(op="dump", id=sb.id)
        c = {s: _match(d[s]["choices"], picks.get(s)) for s in ("p1", "p2")}
        if not (c["p1"] and c["p2"]) or d["p1"]["forced"] or d["p2"]["forced"]:
            sb.step(picks)
            continue
        t = {"turn": d["turn"], "c1": c["p1"][1], "c2": c["p2"][1], "classes": []}
        gaps = es.why_not(d)
        if len(d["p1"]["choices"]) > 1 and (gaps or d["turn"] not in logged):  # the searcher fell back
            t["classes"].append("converter")
            t["gaps"] = gaps or ["not a logged decision"]
        rec = compare(bridge, sb.id, d, c["p1"], c["p2"], samples)
        if not rec["issues"] and "engine_error" not in rec:
            pred = convert.pe.rr_ai_choice_weighted(convert.to_engine(d)[0], "side_two")
            t["ai"] = [[k.removeprefix("switch ").removesuffix("-mega"), round(w, 3)] for k, w in pred]
            if convert.engine_choice(d, "p2", c["p2"]) not in [k for k, _ in t["ai"]]:
                t["classes"].append("ai")
            if rec["flags"]:
                t["classes"].insert(len(t["classes"]) - ("ai" in t["classes"]), "engine")
            t |= {k: rec[k] for k in ("flags", "tv", "missing", "extra", "dhp", "diffs")}
        before = sum(p["hp"] > 0 for p in d["p1"]["pokemon"])
        sb.step(picks)
        after = bridge.call(op="dump", id=sb.id)
        if "_eng" in rec:
            k = convert.outcome_showdown(after)[0]
            t["p_eng"], t["p_sd"] = round(rec["_eng"].get(k, 0), 3), round(rec["_sd"].get(k, 0), 3)
            t["died"] = before - sum(p["hp"] > 0 for p in after["p1"]["pokemon"])
            if t["died"] and t["p_eng"] < LUCK and not rec["flags"]:
                t["classes"].append("luck")
        out["turns"].append(t)
        if t["classes"] and out["first"] is None:
            out["first"] = [t["turn"], t["classes"][0]]
    out["replayed"] = (sb.state()["winner"] == "p1") == r["won"]
    sb.close()
    bridge.close()
    return out


def from_states(path: str, samples: int, workers: int, out: str, seed: int = 0) -> None:
    """The cross-check of bad battles (rl/RR_PLAN.md phase 1.5): select battles from a --states
    log (rr_team_probe.py --engine --states; the <stem>.events.jsonl beside it has the sets),
    replay each on Showdown and give it one verdict: the class of its first flagged turn."""
    import multiprocessing as mp
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    convert.use_rr()
    picked = [b for b in _battles(path, seed) if (b[2] or b[3]) and b[0].get("battle_seed")]
    print(f"{len(picked)} battles to replay ({sum(bool(b[2]) for b in picked)} selected, {sum(b[3] for b in picked)} control)", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(replay, [(path, b, samples) for b in picked])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(json.dumps(x) + "\n" for x in res))
    for x in res:
        if x["why"]:
            print(f"  {x['boss']} battle {x['battle']}: {', '.join(x['why'])}; deaths {x['deaths']}; "
                  f"first flag {x['first'] or 'none (search)'}{'' if x.get('replayed', False) else '  REPLAY DIVERGED ' + x.get('error', '')}")
    for name, group in (("selected", [x for x in res if x["why"]]), ("control", [x for x in res if x["control"]])):
        n = Counter((x["first"] or [0, "search"])[1] for x in group)
        print(f"{name:9s} n={len(group):3d}  " + "  ".join(f"{c} {n[c]}" for c in CLASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--samples", type=int, default=200, help="Showdown replays per position")
    ap.add_argument("--format", default="gen8randombattle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/engine_diff.jsonl")
    ap.add_argument("--show", type=int, default=5, help="print the N worst flagged positions")
    ap.add_argument("--rr", action="store_true", help="Radical Red boss positions on the RR engine build")
    ap.add_argument("--from-states", help="cross-check the bad battles of a --states log (see from_states)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.from_states:
        return from_states(args.from_states, args.samples, args.workers, args.out)

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
            rec.pop("_eng", None), rec.pop("_sd", None)
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
