"""Radical Red: the best six in the pool (step two, 2026-10-10).

Does *any* team from the encounters available before a hardcore boss win the
fight reliably under search play? Per boss: (A) every species catchable
before it (rr_probe.pools: open locations, rods / Surf in hand, encounter level at or
under the cap), evolved by level at the cap, best4 moves over the TMs and tutors in
hand by then (data/rr/tms.json), fights the boss solo
under the calc bot (n battles) — the progress it makes is its score; (B) the
SHORT best are combined: a greedy build plus random six-subsets, each team
scored by the calc bot's progress (n battles); (C) the TOP teams are played
with calc + search and the best team is reported with its search-play win
rate and deaths per mon. Hardcore = Minimal Grinding Mode: nobody has EVs
(player 0, enemy 0 from the dump), no items (Eviolite / Life Orb are single
late copies). `--legal` removes the 82 moves hardcore bans for the player
(data/rr/hardcore_rules.json) from every moveset in stages A-C; without it the
player has every move (the 2026-10-10 afternoon run also gave the enemy
252/252 EVs and the player items / trained EVs — see NOTEBOOK.md). Full enemy
information; enemy = the RR AI port. `--arena` puts the gym's permanent field effect
(tracker battleEffect: weather / terrain / Tailwind / Trick Room) into the sim and
makes p2's own weather and terrain permanent (hardcore). Not modelled: doubles,
the custom arena rules, natures, the player's ability swaps.

`--boxes N` is the milestone form: N boxes per boss drawn by rarity (one encounter per
open location, a random starter, the gifts), the chooser (stage A ranking + greedy build)
picks six from each box, the team plays under search; per box the killers and the pool
species outside the box that beat them are reported (what the box was missing, where).

  uv run python scripts/rr_team_probe.py --bosses "MISTY,LT. SURGE,ERIKA" --workers 3 --out logs/rr_team_probe.jsonl
  uv run python scripts/rr_team_probe.py --bosses BROCK,MISTY --arena --legal --boxes 30 --workers 2 --out logs/rr_boxes.jsonl
"""

import argparse
import collections
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from poke_env.data import to_id_str  # noqa: E402
import rr_probe as rp  # noqa: E402
from rr_postmortem import attribution  # noqa: E402  (sets gauntlet.GAME = "rr", injects the mod data)
import gauntlet  # noqa: E402
from gauntlet import STATS, Mon, encounter_moves, evolve, pack_team  # noqa: E402

SHORT, TOP = 24, 6
GREEDY_N, SUBSETS = 4, 150


def progress(st: dict) -> float:
    """How far a fight got: enemy mons fainted + the fraction taken off the rest, over
    the enemy team; a win adds the survivors' share (1.0 = clean sweep, 2.0 = deathless)."""
    p2 = st["p2"]
    done = sum(1.0 if m["fainted"] else 1 - m["hp"] / max(m["maxhp"], 1) for m in p2) / len(p2)
    if st["winner"] == "p1":
        return 1.0 + sum(not m["fainted"] for m in st["p1"]) / len(st["p1"])
    return done


def build_mon(species: str, cap: int, known: dict, progress: int, boss_types: set = frozenset(), arena: dict = {}) -> Mon | None:
    """Evolved by level at the cap (no stones), best4 moves over the TMs and tutors in hand
    by trainer `progress` (data/rr/tms.json, minus gauntlet.BANNED_MOVES), the nature and
    ability the planner picks for this boss (rr_probe.choose_nature / choose_ability), no
    item, no EVs (Minimal Grinding Mode)."""
    ev = evolve(species, cap, None, caught_level=5, trainer_idx=None)
    moves = encounter_moves(ev, cap, "best4", progress=progress)
    if not moves or ev not in known:
        return None
    m = Mon(ev, cap, moves, ability=rp.choose_ability(ev, known, boss_types, arena), item=None, caught_level=5,
            nature=rp.choose_nature(ev, moves, known))
    m.evs = dict.fromkeys(STATS, 0)
    return m


def show(m: Mon) -> str:
    return f"{m.species}({m.nature}/{m.ability})"


def play(bridge, brain, enemy, party: list[Mon], enemy_packed: str, n: int, search=None, arena=None, seeds=None):
    """n battles; returns (win rate, deaths per mon, mean progress, (our deaths by enemy mon,
    enemy mons KO'd)). seeds: one engine seed per battle index, the same for every team and
    variant (paired comparisons)."""
    from ai_util.sim import SimBattle
    wins, deaths, prog = 0, 0.0, 0.0
    deaths_by, kos_on = collections.Counter(), collections.Counter()
    for i in range(n):
        sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena,
                       seed=seeds[i] if seeds else None)
        if search is not None and seeds:  # the searcher's rollout seeds pair too
            import numpy as np
            search.rng = np.random.default_rng(seeds[i])
        sb.play({"p1": search or brain, "p2": enemy}, 100)
        st = sb.state()
        wins += st["winner"] == "p1"
        deaths += sum(m["fainted"] for m in st["p1"]) / len(st["p1"])
        prog += progress(st)
        d, k, _ = attribution(sb.history["p1"])
        deaths_by += d
        kos_on += k
        sb.close()
    return wins / n, deaths / n, prog / n, (deaths_by, kos_on)


def run_boss(args, name: str) -> dict:
    try:
        return _run_boxes(args, name) if args.boxes else _run_boss(args, name)
    except Exception as e:
        import traceback
        print(f"{name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
        return {"rows": []}


def setup(args, name: str):
    """Everything a boss run shares: the pool, cap, arena, the boss's sets and attack types,
    the bridge, both brains, the searcher, and the paired battle seeds."""
    from types import SimpleNamespace
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBridge, player_brain
    by, known = rp.pools()
    block, cap, pool, progress = by[name]
    arena, unmodelled = rp.arena(block.get("battleEffect", "")) if args.arena else ({}, [block.get("battleEffect", "")])
    print(f"{name}: trainer #{progress}, cap {cap}, pool {len(pool)} species; arena {arena}; not modelled: {unmodelled}", flush=True)
    if args.legal:
        gauntlet.BANNED_MOVES = {to_id_str(m) for m in json.load(open(rp.RR / "hardcore_rules.json"))["banned_moves"]}
    if args.blind_field:
        import ai_util.calc_bot as cb
        cb.FIELD_AWARE = False
    srng = random.Random(f"{name}-seeds-{args.seed}")
    seeds = [[srng.randrange(1, 2 ** 16) for _ in range(4)] for _ in range(max(args.n_solo, args.n_team, args.n_search))]
    enemy_packed, dump_cap, enemy_species = rp.boss_team(name)
    cap = max(cap, dump_cap)
    md = rp.load_game("rr")["moves"]
    boss_types = {md[mv]["type"] for mon in enemy_packed.split("]") for mv in mon.split("|")[4].split(",") if mv in md}
    enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    calc = calc_brain()
    width = tuple(int(x) for x in args.search.split(","))
    return SimpleNamespace(name=name, args=args, known=known, cap=cap, pool=pool, progress=progress, arena=arena,
                           unmodelled=unmodelled, enemy_packed=enemy_packed, enemy_species=enemy_species,
                           boss_types=boss_types, bridge=SimBridge(showdown=str(rp.FORK)), enemy=enemy, calc=calc,
                           searcher=Searcher(None, enemy, k=width[0], rollouts=width[1], rollout=calc, seed=args.seed),
                           seeds=seeds, rng=random.Random(f"{name}-team-{args.seed}"), team_score={}, mons={}, t0=time.time())


def mon_of(ctx, sp: str) -> Mon | None:
    if sp not in ctx.mons:
        ctx.mons[sp] = build_mon(sp, ctx.cap, ctx.known, ctx.progress, ctx.boss_types, ctx.arena)
    return ctx.mons[sp]


def solo_stage(ctx, species: list[str]) -> tuple[list, dict]:
    """(A) every species solo against the boss under the calc bot: (sorted (progress, sp),
    {sp: (our deaths by enemy mon, enemy mons KO'd)})."""
    solo, solo_vs = [], {}
    for sp in species:
        m = mon_of(ctx, sp)
        if m is None:
            continue
        _, _, pr, solo_vs[sp] = play(ctx.bridge, ctx.calc, ctx.enemy, [m], ctx.enemy_packed, ctx.args.n_solo, arena=ctx.arena, seeds=ctx.seeds)
        solo.append((pr, sp))
    solo.sort(reverse=True)
    return solo, solo_vs


def team_value(ctx, team: tuple) -> float:
    if team not in ctx.team_score:
        ctx.team_score[team] = play(ctx.bridge, ctx.calc, ctx.enemy, [ctx.mons[s] for s in team], ctx.enemy_packed, ctx.args.n_team,
                                    arena=ctx.arena, seeds=ctx.seeds)[2]
    return ctx.team_score[team]


def greedy(ctx, short: list[str], start: tuple = ()) -> tuple:
    """(B) add the species that lifts the team's calc-bot progress most, until six (or the
    box runs out)."""
    team = tuple(sorted(start))
    while len(team) < 6 and any(s not in team for s in short):
        best = max((s for s in short if s not in team), key=lambda s: team_value(ctx, tuple(sorted(team + (s,)))))
        team = tuple(sorted(team + (best,)))
    return team


def searched(ctx, team: tuple):
    """(C) the team under calc + search, paired seeds: (win, deaths/mon, progress, attribution)."""
    return play(ctx.bridge, ctx.calc, ctx.enemy, [ctx.mons[s] for s in team], ctx.enemy_packed, ctx.args.n_search,
                search=ctx.searcher, arena=ctx.arena, seeds=ctx.seeds)


def threats_of(ctx, attr, solo_vs: dict, exclude=()) -> list[dict]:
    """The two enemy mons that killed most, each with up to three species (from solo_vs, not in
    `exclude`) that beat it solo in stage A — KOs on it minus deaths to it — and where to catch them."""
    out = []
    for enemy_mon, k in attr[0].most_common(2):
        ranked = sorted((sp for sp in solo_vs if sp not in exclude), key=lambda sp: solo_vs[sp][1][enemy_mon] - solo_vs[sp][0][enemy_mon], reverse=True)
        ranked = [sp for sp in ranked if solo_vs[sp][1][enemy_mon] > solo_vs[sp][0][enemy_mon]][:3]
        out.append({"enemy": enemy_mon, "deaths": k,
                    "counters": [(sp, solo_vs[sp][1][enemy_mon], solo_vs[sp][0][enemy_mon], rp.where(sp, ctx.progress)) for sp in ranked]})
    return out


def say_threats(ctx, threats: list[dict], tag: str = "") -> None:
    for th in threats:
        print(f"{ctx.name:<10} {tag}threat {th['enemy']} ({th['deaths']} of our deaths): counters "
              + (", ".join(f"{sp} [{ko} KOs / {dd} deaths solo; {wh}]" for sp, ko, dd, wh in th["counters"]) or "none"), flush=True)


def sets_of(ctx, team) -> list[dict]:
    return [{"species": ctx.mons[s].species, "nature": ctx.mons[s].nature, "ability": ctx.mons[s].ability, "moves": ctx.mons[s].moves} for s in team]


def _run_boss(args, name: str) -> dict:
    """The planned-pool probe: the best six in everything catchable."""
    ctx = setup(args, name)
    starter = ctx.rng.choice(rp.STARTERS)  # one per run, at random across every generation
    pool = sorted(set(ctx.pool) | {starter})
    print(f"{name}: starter {starter}; boss attacks with {sorted(ctx.boss_types)}", flush=True)
    solo, solo_vs = solo_stage(ctx, pool)
    short = [sp for _, sp in solo[:SHORT]]
    print(f"{name}: A solo scores for {len(solo)} species in {time.time() - ctx.t0:.0f}s; top: "
          + ", ".join(f"{sp} {pr:.2f}" for pr, sp in solo[:8]), flush=True)
    team = greedy(ctx, short)
    for _ in range(SUBSETS):
        team_value(ctx, tuple(sorted(ctx.rng.sample(short, 6))))
    ranked = sorted((t for t in ctx.team_score if len(t) == 6), key=ctx.team_score.get, reverse=True)  # full teams only
    print(f"{name}: B {len(ctx.team_score)} teams scored in {time.time() - ctx.t0:.0f}s; greedy {team} "
          f"({ctx.team_score[team]:.2f}); best fast {ranked[0]} ({ctx.team_score[ranked[0]]:.2f})", flush=True)
    top = list(dict.fromkeys([team] + ranked))[:TOP]
    results, attr = [], {}
    for t in top:
        w, d, pr, attr[t] = searched(ctx, t)
        results.append((w, -d, t, d))
    results.sort(reverse=True)
    w, _, t, d = results[0]
    print(f"{name:<10} legal={args.legal!s:<5} best team under search: win {w:5.0%} deaths/mon {d:.2f} "
          f"{list(t)}  (others: {', '.join(f'{w_:.0%}' for w_, _, _, _ in results[1:])})  ({time.time() - ctx.t0:.0f}s)", flush=True)
    print(f"{name:<10} sets: " + ", ".join(f"{show(ctx.mons[s])} {ctx.mons[s].moves}" for s in t), flush=True)
    threats = threats_of(ctx, attr[t], solo_vs)
    say_threats(ctx, threats)
    repaired = None
    seed_sp = next((sp for th in threats for sp, *_ in th["counters"] if sp not in t), None)  # a counter the best team lacks
    if args.repair and seed_sp:
        mon_of(ctx, seed_sp)
        rteam = greedy(ctx, short, (seed_sp,))
        rw, rd, _, _ = searched(ctx, rteam)
        repaired = {"team": list(rteam), "win": rw, "deaths_per_mon": rd, "seed": seed_sp}
        print(f"{name:<10} repaired with {seed_sp}: win {rw:5.0%} deaths/mon {rd:.2f} {list(rteam)}  (best was {w:.0%} / {d:.2f})  ({time.time() - ctx.t0:.0f}s)", flush=True)
    rows = [{"boss": name, "mode": "pool", "cap": ctx.cap, "legal": args.legal, "field_aware": not args.blind_field, "arena": ctx.arena,
             "unmodelled": ctx.unmodelled, "best_team": list(t), "win": w, "deaths_per_mon": d, "n": args.n_search,
             "all": [(list(t_), w_, d_) for w_, _, t_, d_ in results], "threats": threats, "repaired": repaired,
             "starter": starter, "sets": sets_of(ctx, t), "enemy": ctx.enemy_species, "pool": len(pool), "progress": ctx.progress, "shortlist": short}]
    ctx.bridge.close()
    return {"rows": rows}


def _run_boxes(args, name: str) -> dict:
    """The sampled-box probe (the milestone): N boxes drawn by rarity (rr_probe.sample_box),
    the chooser picks six from each (greedy on the calc bot), the team plays the boss under
    search; per box the killers and the pool species *outside the box* that beat them — what
    the box was missing and where it is caught."""
    ctx = setup(args, name)
    solo, solo_vs = solo_stage(ctx, ctx.pool + [sp for sp in rp.STARTERS if sp not in ctx.pool])
    print(f"{name}: A solo scores for {len(solo)} species (pool + starters) in {time.time() - ctx.t0:.0f}s", flush=True)
    boxes, missing = [], collections.Counter()
    for i in range(args.boxes):
        box = rp.sample_box(random.Random(f"{name}-box-{args.seed}-{i}"), ctx.progress, ctx.known)
        species = [sp for sp, _ in box if mon_of(ctx, sp) is not None]
        team = greedy(ctx, species)
        w, d, pr, attr = searched(ctx, team)
        threats = threats_of(ctx, attr, solo_vs, exclude=set(species))
        for th in threats:
            for sp, _, _, wh in th["counters"][:1]:
                missing[(th["enemy"], sp, wh)] += 1
        boxes.append({"box": box, "team": list(team), "sets": sets_of(ctx, team), "win": w, "deaths_per_mon": d, "threats": threats})
        print(f"{name:<10} box {i + 1:>2}/{args.boxes}: {len(box)} mons, team {list(team)} -> win {w:5.0%} deaths/mon {d:.2f}; "
              + "; ".join(f"{th['enemy']} killed {th['deaths']}, missing " + (", ".join(f"{sp} ({wh})" for sp, _, _, wh in th["counters"][:1]) or "no counter in the pool") for th in threats)
              + f"  ({time.time() - ctx.t0:.0f}s)", flush=True)
    wins = [b["win"] for b in boxes]
    row = {"boss": name, "mode": "boxes", "cap": ctx.cap, "legal": args.legal, "field_aware": not args.blind_field, "arena": ctx.arena,
           "n": args.n_search, "boxes": boxes, "mean_win": sum(wins) / len(wins), "p_clear": sum(w_ >= 0.5 for w_ in wins) / len(wins),
           "mean_deaths": sum(b["deaths_per_mon"] for b in boxes) / len(boxes), "missing": [(k, v) for k, v in missing.most_common(6)],
           "enemy": ctx.enemy_species, "pool": len(ctx.pool), "progress": ctx.progress}
    print(f"{name:<10} {args.boxes} boxes: mean win {row['mean_win']:.0%}, boxes with a >=50% team {row['p_clear']:.0%}, deaths/mon {row['mean_deaths']:.2f}; "
          f"most missed: " + ", ".join(f"{sp} vs {en} ({wh}) x{v}" for (en, sp, wh), v in row["missing"][:3]), flush=True)
    ctx.bridge.close()
    return {"rows": [row]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bosses", default="BROCK,MISTY,LT. SURGE,ERIKA,KOGA,BLAINE")
    ap.add_argument("--n-solo", type=int, default=3)
    ap.add_argument("--n-team", type=int, default=4)
    ap.add_argument("--n-search", type=int, default=6)
    ap.add_argument("--search", default="9,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--legal", action="store_true", help="drop the moves hardcore bans for the player")
    ap.add_argument("--arena", action="store_true", help="the gym's permanent field effect (tracker battleEffect) in the sim")
    ap.add_argument("--blind-field", action="store_true", help="the calc bot ignores weather / terrain (the pre-2026-10-11 bot)")
    ap.add_argument("--repair", action="store_true", help="rebuild the team around the best counter to the top threat and play it too")
    ap.add_argument("--boxes", type=int, default=0, help="sampled-box mode: N boxes per boss drawn by rarity, the chooser picks six from each")
    ap.add_argument("--out")
    args = ap.parse_args()
    names = [b.strip() for b in args.bosses.split(",")]
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(names))) as pool:
            parts = pool.starmap(run_boss, [(args, n) for n in names])
    else:
        parts = [run_boss(args, n) for n in names]
    rows = [r for p in parts for r in p["rows"]]
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    mode = "boxes" if args.boxes else "pool"
    print(f"\nRadical Red best-six probe ({mode}) | legal moves only: {args.legal} | arena effects: {args.arena} | field-aware calc bot: {not args.blind_field} | seeded | "
          f"search {args.search} x n={args.n_search} | no EVs, no items | enemy = RR AI, full info | {(time.time() - t0) / 60:.0f} min")
    if args.boxes:
        print(f"  {'boss':<10} {'cap':>3} {'mean win':>8} {'>=50%':>6} {'d/mon':>5}  most missed counter (vs)")
        for r in sorted(rows, key=lambda r: r["cap"]):
            top = r["missing"][0] if r["missing"] else ((None, None, None), 0)
            print(f"  {r['boss']:<10} {r['cap']:>3} {r['mean_win']:8.0%} {r['p_clear']:6.0%} {r['mean_deaths']:5.2f}  {top[0][1]} vs {top[0][0]} ({top[0][2]}) x{top[1]}")
    else:
        print(f"  {'boss':<10} {'cap':>3} {'win':>5} {'deaths/mon':>10}  best team  | arena")
        for r in sorted(rows, key=lambda r: r["cap"]):
            ar = {k: v for k, v in r["arena"].items() if k != "ai_permanent"}
            print(f"  {r['boss']:<10} {r['cap']:>3} {r['win']:5.0%} {r['deaths_per_mon']:10.2f}  {', '.join(r['best_team'])}  | {ar or '-'}")


if __name__ == "__main__":
    main()
