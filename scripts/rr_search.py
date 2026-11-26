"""Radical Red: team and set search by battles, no learning (plan v3 phase 3, 2026-11-24).

Sequential halving on paired seeds (Karnin et al. 2013, best-arm identification) picks the six,
lead, moves and items per boss, from a sampled box (--boxes: the fast budget), the pool or the
user's box (--my-box: the wide budget). Candidates: the cover six, rr_team_probe.searched_cands,
the best six per archetype flag (a small quality-diversity archive) and random sixes. Sets (D7):
per mon a shortlist from its legal pool (top-2 attacks by calc damage per boss mon + one move per
utility class), default set by marginal matchup-row gain, items from the bag in hand (copies, one
mega stone), nature per boss, ability by rr_probe.run_ability. A round: every candidate plays the
same fresh seeds, the better half stays, battles double, mutants of the survivors join (a box mon
by row gain, lead, any shortlist move into any slot, item). Score (D8), per battle: progress -
0.5 deaths/mon while the round's best wins < 20%, else U = win (1 - 0.5 deaths/6). The winner
then plays --n-search seeds no round used (box mode: the probe table's seeds, so sprt.py pairs
by box). Writes <out>.events.jsonl ("halving", "box", "plan"), <out>.archive.jsonl (every
evaluated six; the random sixes are the unselected sample), <out>.plan.{txt,json}.

  uv run python scripts/rr_search.py --bosses ERIKA --legal --arena --matchup --boxes 30 --out logs/p3_erika.jsonl
  uv run python scripts/rr_search.py --bosses KOGA,BLAINE --legal --arena --matchup --out logs/p3_pool.jsonl
  uv run python scripts/rr_search.py --bosses KOGA --legal --arena --matchup --my-box box.json --line 200000 --line-check 48
"""

import argparse
import collections
import contextlib
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import rr_team_probe as tp  # noqa: E402  (switches gauntlet to the RR game)
import rr_probe as rp  # noqa: E402
import gauntlet  # noqa: E402
from gauntlet import Mon, base_species, copies, format_species, legal_pool, load_game, mega_stone  # noqa: E402
from poke_env.data import GenData, to_id_str  # noqa: E402
from ai_util.matchup import HEALS, edge, hits_to_ko, race, safe  # noqa: E402

PIVOT = {"uturn", "voltswitch", "flipturn", "partingshot"}
CHANGE = {"simplebeam", "worryseed", "entrainment", "skillswap", "gastroacid", "coreenforcer", "defog", "raindance",
          "sunnyday", "sandstorm", "hail", "snowscape", "grassyterrain", "mistyterrain", "electricterrain", "psychicterrain"}
NEGATE = {"cloudnine", "airlock", "electricsurge", "grassysurge", "mistysurge", "psychicsurge"}


def util_class(mv: str) -> str | None:
    """The utility class of a move (D7's shortlist), None for a plain attack or anything else."""
    e = GenData.from_gen(9).moves.get(mv, {})
    sec = e.get("secondary") or {}
    if mv in PIVOT or mv in ("knockoff", "encore"):
        return "pivot" if mv in PIVOT else mv
    if mv in CHANGE or mv in HEALS:
        return "change" if mv in CHANGE else "recovery"
    if e.get("category") != "Status" and e.get("priority", 0) > 0:
        return "priority"
    st = e.get("status") or (sec.get("status") if sec.get("chance") == 100 else None)
    if st in ("slp", "par", "brn", "tox"):
        return st
    if mv in ("tailwind", "trickroom", "stickyweb") or (e.get("boosts") or {}).get("spe", 0) < 0 \
            or (sec.get("chance") == 100 and (sec.get("boosts") or {}).get("spe", 0) < 0):
        return "speed"
    return None


def schedule(k: int, battles: int) -> tuple[list[int], list[int]]:
    """Sequential halving's rounds for k candidates and a battle budget: (candidates, battles each)
    per round; the field halves to 2, the battles double, every round costs about the same."""
    sizes = [k]
    while sizes[-1] > 2:
        sizes.append(math.ceil(sizes[-1] / 2))
    n0 = max(1, battles // sum(s * 2 ** r for r, s in enumerate(sizes)))
    return sizes, [n0 * 2 ** r for r in range(len(sizes))]


def score(b: tuple, low: bool, size: int, d14_weights: dict[str, float] | None = None) -> float:
    """D8 / D14 for one battle (one_battle's tuple):
    - Default (D8): progress - 0.5 deaths/mon while the best candidate wins < 20% (low),
      else U = win * (1 - 0.5 * deaths / 6).
    - With `d14_weights` ({evolved_species: future_death_cost}, D14): replaces flat 0.5/6
      with the sum of future-boss costs of the specific mons that fainted (b[7]), capped at 0.9
      so any win still beats any loss while strongly preferring to sacrifice expendable filler
      over irreplaceable future-boss carries."""
    won, _, dfrac, prog = b[:4]
    if d14_weights is not None and len(b) >= 8:
        fainted = b[7]
        dead_cost = min(0.9, sum(d14_weights.get(to_id_str(sp), 0.5 / 6) for sp in fainted))
        return prog - dead_cost if low else won * (1.0 - dead_cost)
    return prog - 0.5 * dfrac if low else won * (1 - 0.5 * dfrac * size / 6)


def key(sets) -> tuple:
    """A six's identity: the lead, then the rest in any order; moves in any order."""
    k = lambda s: (s["from"], tuple(sorted(s["moves"])), s["item"], s["nature"], s["ability"])
    return (k(sets[0]),) + tuple(sorted(k(s) for s in sets[1:]))


def info(ctx, species: list[str]) -> dict:
    """{sp: shortlist, calc damage of every legal attack on each boss mon, the matchup pair per
    boss mon} off dummy full-information battles (six per battle; cached per boss)."""
    from poke_env.battle import Move
    from ai_util.calc_bot import dmg_frac
    from ai_util.matchup import pair
    from ai_util.sim import SimBattle
    cache, gd = ctx.__dict__.setdefault("info", {}), GenData.from_gen(9).moves
    todo = [s for s in species if s not in cache and tp.mon_of(ctx, s)]
    for c in range(0, len(todo), 6):
        chunk = todo[c:c + 6]
        sb = SimBattle(ctx.bridge, rp.FMT, gauntlet.pack_team([tp.mon_of(ctx, s) for s in chunk]), ctx.enemy_packed, full_info=True, arena=ctx.arena)
        b = sb.mirrors["p1"]
        for s in chunk:
            m = tp.mon_of(ctx, s)
            me = next((p for p in b.team.values() if p.species == to_id_str(m.species)), None)
            if me is None:
                continue
            pool = [x for x in dict.fromkeys(legal_pool(m.species, m.level, ctx.progress) + getattr(m, "known", []) + m.moves)
                    if x in gd and x not in gauntlet.NO_SLOT_MOVES]
            dmg = {x: {o.species: dmg_frac(me, o, Move(x, 9), True, b) for o in b.opponent_team.values()}
                   for x in pool if gd[x]["category"] != "Status"}
            enemies = [o.species for o in b.opponent_team.values()]
            short = [x for e in enemies for x in sorted(dmg, key=lambda x: -dmg[x][e])[:2]]
            classes = {}
            for x in sorted(pool, key=lambda x: -sum(dmg.get(x, {}).values())):  # the strongest of a class first
                classes.setdefault(util_class(x), x)
            classes.pop(None, None)
            cache[s] = {"short": list(dict.fromkeys(short + list(classes.values()) + [x for x in m.moves if x in pool])),
                        "dmg": dmg, "pair": {o.species: pair(me, o, b) for o in b.opponent_team.values()}, "base": m.moves}
        sb.close()
    return cache


def row(ctx, sp: str, moves) -> dict:
    """{boss mon: cell value} of a set: 2 safe + race + 0.1 edge (ai_util.matchup) with its best hit."""
    i, memo = ctx.info[sp], ctx.__dict__.setdefault("rows", {})
    if (sp, tuple(sorted(moves))) in memo:
        return memo[sp, tuple(sorted(moves))]
    out = memo[sp, tuple(sorted(moves))] = {}
    for e, p in i["pair"].items():
        h = max((i["dmg"].get(x, {}).get(e, 0.0) for x in moves), default=0.0)
        q = dict(p, hit=h, ko_them=hits_to_ko(1.0, h, h) if h > 0 else 99)
        out[e] = 2 * safe(q) + race(q) + 0.1 * edge(q)
    return out


def team_value(ctx, sets) -> float:
    rows = [row(ctx, s["from"], s["moves"]) for s in sets]
    return sum(max(r.get(e, 0.0) for r in rows) for e in (rows[0] if rows else {}))


def make_set(ctx, sp: str, moves: list, item=None) -> dict:
    m = tp.mon_of(ctx, sp)
    return {"from": sp, "species": m.species, "level": m.level, "ability": m.ability, "item": item, "moves": list(moves),
            "nature": rp.choose_nature(m.species, moves, ctx.known, ctx.boss, m.level)}  # natures are free at any Center


def default_set(ctx, sp: str) -> dict:
    """The best4 set's status moves kept (the calc scores damage only: D16's B dropped Thunder Wave /
    Will-O-Wisp / screens), the other slots attacks from the shortlist by marginal row gain while they
    gain, then the best4 set's moves."""
    cache = ctx.__dict__.setdefault("defaults", {})
    if sp not in cache:
        i, moves = ctx.info[sp], []
        keep = [x for x in i["base"] if x not in i["dmg"]]
        val = lambda mv: sum(row(ctx, sp, mv).values())
        while len(moves) < 4 - len(keep):
            best = max((x for x in i["short"] if x in i["dmg"] and x not in moves), key=lambda x: val(moves + [x]), default=None)
            if best is None or val(moves + [best]) <= val(moves) + 1e-9:
                break
            moves.append(best)
        moves += keep
        moves += [x for x in i["base"] + i["short"] if x not in moves][:4 - len(moves)]
        cache[sp] = make_set(ctx, sp, moves)
    return dict(cache[sp])


def bag(ctx) -> dict:
    return {k: v for k, v in load_game()["items"].items() if v["before_trainer"] <= ctx.progress} if ctx.args.items else {}


def with_items(ctx, sets) -> tuple:
    """The bag handed out by gauntlet.assign_items (one per mon, copies respected), one mega stone."""
    mons = [Mon(s["species"], s["level"], s["moves"], s["ability"]) for s in sets]
    gauntlet.assign_items(mons, ctx.progress if ctx.args.items else None)
    out, stone = [], False
    for s, m in zip(sets, mons):
        mega = bool(m.item and "mega" in bag(ctx)[m.item])
        out.append(dict(s, item=None if mega and stone else m.item))
        stone |= mega
    return tuple(out)


def team_of(ctx, species: tuple) -> tuple:
    return with_items(ctx, [default_set(ctx, sp) for sp in species])


def flags(ctx, sp: str) -> set:
    m, cls = tp.mon_of(ctx, sp), {util_class(x) for x in ctx.info[sp]["short"]}
    bs, ab = ctx.known.get(m.species, {}).get("baseStats", {}), to_id_str(m.ability or "")
    return ({"negate"} if ab in NEGATE and (ctx.arena.get("weather") or ctx.arena.get("terrain")) else set()) \
        | ({"speed"} if cls & {"speed", "par"} else set()) | ({"pivot"} if "pivot" in cls else set()) \
        | ({"priority"} if "priority" in cls else set()) | ({"mega"} if mega_stone(m.species, ctx.progress) else set()) \
        | ({"bulk"} if ab == "intimidate" or bs.get("hp", 0) * (bs.get("def", 0) + bs.get("spd", 0)) >= 20000 else set())


def _slot_ok(ctx, sets: list, sp: str, drop_idx: int | None = None) -> bool:
    """True if `sp` can join `sets` (optionally replacing `sets[drop_idx]`) without duplicating an
    evolved species or exceeding per-caught-mon slot limits for branched evolutions."""
    rest = [s for idx, s in enumerate(sets) if idx != drop_idx]
    ev = tp.mon_of(ctx, sp).species
    if any(s["species"] == ev for s in rest):
        return False
    limits = getattr(ctx, "slot_limits", None)
    if limits is not None:
        b = base_species(sp)
        if sum(1 for s in rest if base_species(s["from"]) == b) >= limits.get(b, 1):
            return False
    return True


def fill(ctx, sets: list, species: list) -> list:
    """Greedy to six by team row value (one per evolved species, respecting branch slot limits)."""
    while len(sets) < 6:
        best = max((sp for sp in species if _slot_ok(ctx, sets, sp)),
                   key=lambda sp: team_value(ctx, sets + [default_set(ctx, sp)]), default=None)
        if best is None:
            break
        sets.append(default_set(ctx, best))
    return sets


def archetypes(ctx, species: list) -> list:
    """The quality-diversity seeds: per flag, its best flagged mon (row value), carrying the
    flag's move in its last slot, then the six filled by row value; lead by lead_first."""
    out = []
    for flag in ("negate", "speed", "pivot", "mega", "priority", "bulk"):
        flagged = [sp for sp in species if flag in flags(ctx, sp)]
        if not flagged:
            continue
        sp = max(flagged, key=lambda s: team_value(ctx, [default_set(ctx, s)]))
        seed = default_set(ctx, sp)
        cls = {"speed": ("speed", "par"), "pivot": ("pivot",), "priority": ("priority",)}.get(flag, ())
        mv = next((x for x in ctx.info[sp]["short"] if util_class(x) in cls and x not in seed["moves"]), None)
        if mv:
            seed = make_set(ctx, sp, seed["moves"][:3] + [mv])
        sets = fill(ctx, [seed], species)
        order = tp.lead_first(ctx, tuple(s["from"] for s in sets))
        out.append((flag, with_items(ctx, sorted(sets, key=lambda s: order.index(s["from"])))))
    return out


def mutate(ctx, parent: tuple, species: list, rng: random.Random) -> tuple[str, tuple] | None:
    """One mutation: (kind, sets) or None when it does not apply."""
    sets, kind = list(parent), rng.choice(("swap", "lead", "move", "move", "item"))
    j = rng.randrange(len(sets))
    s = sets[j]
    status = lambda moves: any(GenData.from_gen(9).moves.get(x, {}).get("category") == "Status" for x in moves)
    if kind == "lead" and len(sets) > 1:
        j = rng.randrange(1, len(sets))
        sets[0], sets[j] = sets[j], sets[0]
    elif kind == "move":
        new = [x for x in ctx.info[s["from"]]["short"] if x not in s["moves"]]
        if not new:
            return None
        moves, mv = list(s["moves"]), rng.choice(new)
        if len(moves) < 4:
            moves.append(mv)
        else:
            moves[rng.randrange(4)] = mv
        if s["item"] and bag(ctx)[s["item"]].get("lock") and gauntlet.CHOICE_ALL_ATTACKS and status(moves):
            return None
        sets[j] = make_set(ctx, s["from"], moves, s["item"])
    elif kind == "item":
        others = [x["item"] for k, x in enumerate(sets) if k != j and x["item"]]
        stone = any("mega" in bag(ctx)[x] for x in others)
        opts = [None] + [k for k, v in bag(ctx).items() if others.count(k) < copies(v, ctx.progress) and k != s["item"]
                         and ("mega" not in v or (v["mega"] == to_id_str(s["species"]) and not stone))
                         and not (v.get("lock") and gauntlet.CHOICE_ALL_ATTACKS and status(s["moves"]))]
        sets[j] = dict(s, item=rng.choice(opts))
    elif kind == "swap":
        base = team_value(ctx, sets)
        gains = sorted(((team_value(ctx, sets[:k] + [default_set(ctx, sp)] + sets[k + 1:]) - base, k, sp)
                        for k in range(len(sets)) for sp in species if _slot_ok(ctx, sets, sp, drop_idx=k)), reverse=True)[:3]
        if not gains:
            return None
        _, k, sp = rng.choice(gains)
        item = sets[k]["item"] if sets[k]["item"] and "mega" not in bag(ctx)[sets[k]["item"]] else None
        sets[k] = dict(default_set(ctx, sp), item=item)
    else:
        return None
    return kind, tuple(sets)


def halving(ctx, species: list, tag) -> dict:
    """Sequential halving over the candidates of one box (tag = its index) or pool ("pool"):
    the winner, after every candidate's rounds are emitted ("halving") and archived."""
    args = ctx.args
    rng = random.Random(f"{ctx.name}-search-{args.seed}-{tag}")
    ctx.rng = random.Random(f"{ctx.name}-cands-{args.seed}-{tag}")  # searched_cands' draws, per box (not per worker history)
    info(ctx, species)
    species = [sp for sp in species if sp in ctx.info]
    limits = getattr(ctx, "slot_limits", None)
    small_box = len(species) <= 6 and (limits is None or sum(limits.values()) == len(species))
    raw = [("box", tp.lead_first(ctx, tuple(species)))] if small_box else (
        [("cover", tp.lead_first(ctx, tp.cover_team(ctx, species)))] + [("searched", t) for t in tp.searched_cands(ctx, species, max(2, args.cands // 3))])
    cands, seen = [], set()

    def add(origin, sets, parent=None):
        if key(sets) not in seen:
            seen.add(key(sets))
            cands.append({"id": len(seen) - 1, "origin": origin, "parent": parent, "sets": sets, "scores": [], "fell": None})
    for origin, t in raw:
        add(origin, team_of(ctx, t))
    for origin, sets in archetypes(ctx, species) if len(species) > 6 else []:
        add(origin, sets)
    n_random = max(1, args.cands // 8) if len(species) > 6 else 0
    cands = cands[:args.cands - n_random]
    for _ in range(n_random):
        pick = []
        for sp in rng.sample(species, len(species)):
            if _slot_ok(ctx, [default_set(ctx, x) for x in pick], sp):
                pick.append(sp)
            if len(pick) == 6:
                break
        if pick:
            add("random", team_of(ctx, tp.lead_first(ctx, tuple(pick))))
    sizes, ns = schedule(args.cands, args.battles)

    def mutants(parents, m):
        out = []
        for t in range(20 * m):
            if len(out) >= m:
                break
            p = parents[t % len(parents)]
            mut = mutate(ctx, p["sets"], species, rng)
            if mut and key(mut[1]) not in seen:
                add(mut[0], mut[1], p["id"])
                out.append(cands.pop())
        return out
    everyone = list(cands)
    if len(cands) < args.cands:  # a small box: mutants make up the field
        new = mutants(list(cands), args.cands - len(cands))
        cands += new
        everyone += new
    base, battler = args.n_search, {"engine": args.engine, "engine_opp": args.engine_opp, "engine_p": args.engine_p, "death_weight": args.death_weight}
    d14_w = getattr(ctx, "d14_weights", None)
    for r, n in enumerate(ns):
        res = tp.fan(tp.battle_job, [(args, ctx.name, c["sets"], base + j) for c in cands for j in range(n)])
        for k, c in enumerate(cands):
            c["b"] = res[k * n:(k + 1) * n]
        low = max(sum(b[0] for b in c["b"]) / n for c in cands) < 0.2
        for c in cands:
            c["score"] = sum(score(b, low, len(c["sets"]), d14_w) for b in c["b"]) / n
            c["scores"].append(round(c["score"], 4))
            w, d, _, (db, _, capped), cl = tp.fold(c["b"])
            if args.out:
                tp.append_line(Path(args.out).with_suffix(".archive.jsonl"), {
                    "boss": ctx.name, "box": tag, "round": r, "id": c["id"], "origin": c["origin"], "parent": c["parent"],
                    "sets": c["sets"], "score": c["score"], "low": low, "n": n, "seeds": [base, base + n], "win": w, "deaths_per_mon": d,
                    "deathless": cl, "capped": capped, "deaths_by": dict(db), "battler": battler, "legal": args.legal, "arena": args.arena})
        base += n
        cands.sort(key=lambda c: -c["score"])
        if r == len(ns) - 1:
            break
        m = max(1, sizes[r + 1] // 3)
        for c in cands[sizes[r + 1] - m:]:
            c["fell"] = r
        cands = cands[:sizes[r + 1] - m]
        new = mutants(cands, m)
        cands += new
        everyone += new
    for c in cands[1:]:
        c["fell"] = len(ns) - 1
    tp.emit(args, "halving", boss=ctx.name, box=tag, rounds=[{"n": n, "size": s} for n, s in zip(ns, sizes)],
            cands=[{k: c[k] for k in ("id", "origin", "parent", "scores", "fell")} | {"team": [s["species"] for s in c["sets"]]} for c in everyone],
            winner=cands[0]["id"])
    return cands[0]


def final(ctx, best: dict) -> dict:
    """The winner on --n-search seeds no round used: win, deaths, U with 95% intervals."""
    n = ctx.args.n_search
    d14_w = getattr(ctx, "d14_weights", None)
    bs = tp.fan(tp.battle_job, [(ctx.args, ctx.name, best["sets"], j) for j in range(n)])
    w, d, _, (db, _, capped), cl = tp.fold(bs)
    u = [score(b, False, len(best["sets"]), d14_w) for b in bs]
    mu = sum(u) / n
    ci = lambda p: 1.96 * math.sqrt(p * (1 - p) / n)
    out = {"boss": ctx.name, "team": [s["species"] for s in best["sets"]],
           "sets": [{k: v for k, v in s.items() if k != "from"} | ({"from_base": base_species(s["from"])} if base_species(s["from"]) != s["species"] else {}) for s in best["sets"]],
           "origin": best["origin"], "halving_scores": best["scores"], "n": n, "win": w, "win_ci": ci(w), "deathless": cl, "deathless_ci": ci(cl),
           "deaths_per_mon": d, "U": mu, "U_ci": 1.96 * math.sqrt(sum((x - mu) ** 2 for x in u) / max(n - 1, 1) / n), "capped": capped,
           "deaths_by": dict(db), "cap": ctx.cap, "progress": ctx.progress}
    if d14_w:
        out["d14_team_costs"] = {s["species"]: round(d14_w.get(to_id_str(s["species"]), 0.5 / 6), 3) for s in best["sets"]}
    return out


def box_job(args, name: str, i: int) -> dict | None:
    try:
        ctx = tp.worker_ctx(args, name)
        box = rp.sample_box(random.Random(f"{name}-box-{args.seed}-{i}"), ctx.progress, ctx.known)
        r = final(ctx, halving(ctx, tp.expand_box(ctx, [sp for sp, _ in box]), i))
        tp.emit(args, "box", i=i, of=args.boxes, box=box, **r)
        print(f"{name:<10} box {i + 1:>2}/{args.boxes}: {r['team']} ({r['origin']}) -> win {r['win']:.0%} deaths/mon {r['deaths_per_mon']:.2f} "
              f"deathless {r['deathless']:.0%}  ({time.time() - ctx.t0:.0f}s)", flush=True)
        return r
    except Exception as e:
        import traceback
        print(f"{name} box {i + 1} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
        return None


def load_box(ctx, path: str) -> list[str]:
    """--my-box: [{species, level, nature, ability, moves, evolve?}] as the run has them.
    Already-evolved mons stay as entered; set `evolve: true` (or pass a plain species string)
    to auto-evolve to the boss cap and evaluate all legal evolution branches."""
    capsule, patch = rp.ability_gates()
    out, limits = [], {}
    for x in json.load(open(path)):
        if isinstance(x, str):
            x = {"species": x, "evolve": True}
        sp = to_id_str(x["species"])
        if x.get("evolve"):
            opts = gauntlet.evolve_options(sp, ctx.cap, caught_level=5, trainer_idx=ctx.progress)
            keys = [f"{sp}>{ev}" for ev in opts] if len(opts) > 1 else [sp]
            limits[sp] = limits.get(sp, 0) + 1
            for k in keys:
                if k not in out:
                    out.append(k)
            continue
        lvl = int(x.get("level") or ctx.cap)
        known = [to_id_str(m) for m in x.get("moves", [])]
        ab = x.get("ability") if ctx.progress < capsule else rp.choose_ability(sp, ctx.known, ctx.boss_types, ctx.arena, hidden=ctx.progress >= patch)
        m = Mon(sp, lvl, (known or gauntlet.encounter_moves(sp, lvl, "best4", progress=ctx.progress, dead=rp.dead_types(ctx.arena, ab), ability=ab))[:4], ability=ab, caught_level=5, nature=x.get("nature"))
        m.evs, m.known = dict.fromkeys(gauntlet.STATS, 0), known
        ctx.mons[sp] = m
        limits[sp] = limits.get(sp, 0) + 1
        out.append(sp)
    ctx.slot_limits = limits
    return tp.playable(ctx, out)


def plan_text(r: dict) -> str:
    out = [f"{r['boss']} plan (trainer #{r['progress']}, level cap {r['cap']}; origin {r['origin']}, halving scores {r['halving_scores']})",
           f"  fresh seeds, n = {r['n']}: U {r['U']:.3f} ± {r['U_ci']:.3f}, win {r['win']:.0%} ± {r['win_ci']:.0%}, "
           f"deathless {r['deathless']:.0%} ± {r['deathless_ci']:.0%}, deaths/mon {r['deaths_per_mon']:.2f}, capped {r['capped']:.0%}"]
    out += [f"  {'lead' if k == 0 else f'{k + 1}.':>4} {format_species(s.get('from_base', s['species']), s['species'])} L{s['level']} @ {s['item'] or '-'}, {s['nature']}, {s['ability']}: {', '.join(s['moves'])}"
            for k, s in enumerate(r["sets"])]
    out.append("  riskiest enemy mons (our deaths in the n battles): " + (", ".join(f"{e} {k}" for e, k in collections.Counter(r["deaths_by"]).most_common(3)) or "none"))
    if r.get("d14_team_costs"):
        dc = ", ".join(f"{sp}={w:.3f}" for sp, w in sorted(r["d14_team_costs"].items(), key=lambda kv: -kv[1]))
        out.append(f"  D14 future death costs (higher = protect): {dc}")
    if r.get("line"):
        ln = r["line"]
        out.append(f"  line ({ln['iterations']} iterations): P(win) {ln['win']:.1%}, P(deathless) {ln['deathless']:.1%} (~10 points pessimistic), "
                   f"E[deaths] {ln['deaths']:.2f}, unexplored {ln['unexplored']:.1%}" + (f"; Showdown check: win {ln['check']['win']:.0%}, deathless {ln['check']['deathless']:.0%}" if ln.get("check") else ""))
    return "\n".join(out)


def add_line(args, r: dict) -> None:
    """--line: rl/line.py's policy tree (and --line-check Showdown battles) for the final six."""
    from rl import convert
    from rl.line import Line, check, start
    convert.use_rr()
    sets = [{k: s[k] for k in ("species", "moves", "ability", "item", "nature")} for s in r["sets"]]
    state, *_, bridge = start(r["boss"], r["cap"], sets)
    line = Line(eps=args.line_eps, iterations=args.line)
    line.expand(state.to_string())
    line.estimate_tail(args.line_playouts)
    bridge.close()
    r["line"] = {k: v for k, v in line.summary().items() if k in ("win", "deathless", "deaths", "die", "unexplored", "nodes")} | {"iterations": args.line}
    if args.line_check:
        r["line"]["check"] = check(r["boss"], r["cap"], sets, args.line, args.line_check, args.workers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bosses", default="ERIKA")
    ap.add_argument("--boxes", type=int, default=0, help="N sampled boxes per boss (rr_probe.sample_box), the fast budget; 0 = the pool (wide)")
    ap.add_argument("--my-box", help="the user's box as JSON [{species, level, nature, ability, moves}] (wide budget)")
    ap.add_argument("--battles", type=int, help="halving budget per box / boss (default 120 with --boxes, else 4000)")
    ap.add_argument("--cands", type=int, help="candidates in the first round (default 8 with --boxes, else 48)")
    ap.add_argument("--n-search", type=int, help="the winner's battles on fresh seeds (default 6 with --boxes, as the probe's table; else 48)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--legal", action="store_true")
    ap.add_argument("--arena", action="store_true")
    ap.add_argument("--matchup", action="store_true")
    ap.add_argument("--no-items", action="store_true")
    ap.add_argument("--death-weight", type=float, default=6.0)
    ap.add_argument("--d14-bosses", default="", help="comma-separated future bosses for D14 per-mon death costs (e.g. KOGA,BLAINE)")
    ap.add_argument("--engine", type=int, default=20000)
    ap.add_argument("--engine-opp", choices=("minimax", "ai"), default="ai")
    ap.add_argument("--engine-p", type=float, default=0.8)
    ap.add_argument("--line", type=int, default=0, help="plan the final six with rl/line.py at this many iterations (200000 for the advisor); 0 = off")
    ap.add_argument("--line-eps", type=float, default=0.002)
    ap.add_argument("--line-playouts", type=int, default=200)
    ap.add_argument("--line-check", type=int, default=0, help="--line: also play this many Showdown battles under the line's policy")
    ap.add_argument("--out")
    args = ap.parse_args()
    wide = not args.boxes
    args.battles = args.battles or (4000 if wide else 120)
    args.cands = args.cands or (48 if wide else 8)
    args.n_search = args.n_search or (48 if wide else 6)
    args.items = not args.no_items
    for k, v in dict(normal=False, blind_field=False, n_solo=1, n_team=1, search="9,8", enemy="python", states=False, chooser="cover", n_duel=4).items():
        setattr(args, k, v)
    args.n_screen = sum(schedule(args.cands, args.battles)[1])  # the halving's seeds, after the n_search the table uses
    names, t0 = [b.strip() for b in args.bosses.split(",")], time.time()
    if args.out:
        for suffix in (".events.jsonl", ".archive.jsonl"):
            Path(args.out).with_suffix(suffix).unlink(missing_ok=True)
    tp.emit(args, "start", config=vars(args), bosses=names, mode="boxes" if args.boxes else "pool", pid=__import__("os").getpid())
    tp._MAIN, rows = True, []
    with (mp.get_context("spawn").Pool(args.workers) if args.workers > 1 else contextlib.nullcontext()) as tp._POOL:
        if args.boxes:
            for n in names:
                tp.setup(args, n).bridge.close()  # banner + boss_start
            res = tp.fan(box_job, [(args, n, i) for n in names for i in range(args.boxes)])
            for k, n in enumerate(names):
                bx = [b for b in res[k * args.boxes:(k + 1) * args.boxes] if b]
                m = lambda f: sum(b[f] for b in bx) / max(len(bx), 1)
                rows.append({"boss": n, "mode": "boxes", "boxes": bx, "failed": args.boxes - len(bx), "mean_win": m("win"),
                             "mean_deaths": m("deaths_per_mon"), "mean_deathless": m("deathless"), "mean_U": m("U")})
                tp.emit(args, "boss_done", boss=n, row=rows[-1])
                print(f"{n:<10} {len(bx)} boxes: mean win {m('win'):.0%}, deaths/mon {m('deaths_per_mon'):.2f}, deathless {m('deathless'):.0%}, U {m('U'):.3f}", flush=True)
        else:
            for n in names:
                ctx = tp._CTX[n] = tp.setup(args, n)
                species = load_box(ctx, args.my_box) if args.my_box else tp.expand_box(ctx, sorted(set(ctx.pool) | set(rp.STARTERS)))
                if args.d14_bosses:
                    import rr_route as rt
                    future_names = [b.strip() for b in args.d14_bosses.split(",") if b.strip()]
                    ftables = [rt.build_boss_table(b, seed=args.seed) for b in future_names]
                    ftables = [tb for tb in ftables if tb["progress"] > ctx.progress]
                    if ftables:
                        raw_box = [base_species(sp) for sp in species]
                        costs = rt.death_costs(raw_box, ftables)
                        d14_map = {}
                        for r_c in costs:
                            m = tp.mon_of(ctx, r_c["species"])
                            if m:
                                d14_map[to_id_str(m.species)] = max(d14_map.get(to_id_str(m.species), 0.0), r_c["d14_weight"])
                        ctx.d14_weights = d14_map
                        print(f"{n}: D14 per-mon death costs enabled over {[tb['boss'] for tb in ftables]} ({len(d14_map)} species)", flush=True)
                print(f"{n}: {len(species)} playable species; budget {args.battles} battles, {args.cands} candidates", flush=True)
                rows.append(final(ctx, halving(ctx, species, "pool")) | {"mode": "my-box" if args.my_box else "pool"})
                tp.emit(args, "team", label="search", **rows[-1])
    tp._POOL = None
    if not args.boxes:
        for r in rows:
            if args.line:
                add_line(args, r)
            print(plan_text(r), flush=True)
            tp.emit(args, "plan", **r)
        if args.out:
            Path(args.out).with_suffix(".plan.txt").write_text("\n\n".join(plan_text(r) for r in rows) + "\n")
            Path(args.out).with_suffix(".plan.json").write_text(json.dumps(rows, indent=1))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    tp.emit(args, "end", minutes=round((time.time() - t0) / 60, 1))
    print(f"rr_search: {', '.join(names)} | {'boxes x ' + str(args.boxes) if args.boxes else args.my_box or 'pool'} | {args.battles} battles, "
          f"{args.cands} candidates, n_search {args.n_search} | engine {args.engine} {args.engine_opp} p={args.engine_p} | {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
