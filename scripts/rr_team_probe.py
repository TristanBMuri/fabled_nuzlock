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

  uv run python scripts/rr_team_probe.py --bosses "MISTY,LT. SURGE,ERIKA" --workers 3 --out logs/rr_team_probe.jsonl
"""

import argparse
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
import rr_probe as rp  # noqa: E402  (sets gauntlet.GAME = "rr", injects the mod data)
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


def build_mon(species: str, cap: int, known: dict, progress: int) -> Mon | None:
    """Evolved by level at the cap (no stones), best4 moves over the TMs and tutors in hand
    by trainer `progress` (data/rr/tms.json, minus gauntlet.BANNED_MOVES), ability slot 0,
    no item, no EVs (Minimal Grinding Mode)."""
    ev = evolve(species, cap, None, caught_level=5, trainer_idx=None)
    moves = encounter_moves(ev, cap, "best4", progress=progress)
    if not moves or ev not in known:
        return None
    m = Mon(ev, cap, moves, ability=known[ev].get("abilities", {}).get("0"), item=None, caught_level=5)
    m.evs = dict.fromkeys(STATS, 0)
    return m


def play(bridge, brain, enemy, party: list[Mon], enemy_packed: str, n: int, search=None, arena=None):
    """n battles; returns (win rate, deaths per mon, mean progress)."""
    from ai_util.sim import SimBattle
    wins, deaths, prog = 0, 0.0, 0.0
    for _ in range(n):
        sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena)
        sb.play({"p1": search or brain, "p2": enemy}, 100)
        st = sb.state()
        wins += st["winner"] == "p1"
        deaths += sum(m["fainted"] for m in st["p1"]) / len(st["p1"])
        prog += progress(st)
        sb.close()
    return wins / n, deaths / n, prog / n


def run_boss(args, name: str) -> dict:
    try:
        return _run_boss(args, name)
    except Exception as e:
        import traceback
        print(f"{name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
        return {"rows": []}


def _run_boss(args, name: str) -> dict:
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
    enemy_packed, dump_cap, enemy_species = rp.boss_team(name)
    cap = max(cap, dump_cap)
    bridge = SimBridge(showdown=str(rp.FORK))
    enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    calc = calc_brain()
    rng = random.Random(f"{name}-team-{args.seed}")
    t0 = time.time()
    # A: solo scores (the ranking, not the final strength)
    solo = []
    for sp in pool:
        m = build_mon(sp, cap, known, progress)
        if m is None:
            continue
        _, _, pr = play(bridge, calc, enemy, [m], enemy_packed, args.n_solo, arena=arena)
        solo.append((pr, sp))
    solo.sort(reverse=True)
    short = [sp for _, sp in solo[:SHORT]]
    print(f"{name}: A solo scores for {len(solo)} species in {time.time() - t0:.0f}s; top: "
          + ", ".join(f"{sp} {pr:.2f}" for pr, sp in solo[:8]), flush=True)
    # B: teams from the shortlist, scored by the calc bot
    mons = {sp: build_mon(sp, cap, known, progress) for sp in short}
    team_score = {}

    def score(team: tuple) -> float:
        if team not in team_score:
            team_score[team] = play(bridge, calc, enemy, [mons[s] for s in team], enemy_packed, args.n_team, arena=arena)[2]
        return team_score[team]
    team = ()
    while len(team) < 6:  # greedy: add the species that lifts the team's progress most
        best = max((s for s in short if s not in team), key=lambda s: score(tuple(sorted(team + (s,)))))
        team = tuple(sorted(team + (best,)))
    for _ in range(SUBSETS):
        score(tuple(sorted(rng.sample(short, 6))))
    ranked = sorted((t for t in team_score if len(t) == 6), key=team_score.get, reverse=True)  # full teams only
    print(f"{name}: B {len(team_score)} teams scored in {time.time() - t0:.0f}s; greedy {team} "
          f"({team_score[team]:.2f}); best fast {ranked[0]} ({team_score[ranked[0]]:.2f})", flush=True)
    top = list(dict.fromkeys([team] + ranked))[:TOP]
    # C: the top teams under search
    width = tuple(int(x) for x in args.search.split(","))
    searcher = Searcher(None, enemy, k=width[0], rollouts=width[1], rollout=calc, seed=args.seed)
    results = []
    for t in top:
        w, d, pr = play(bridge, calc, enemy, [mons[s] for s in t], enemy_packed, args.n_search, search=searcher, arena=arena)
        results.append((w, -d, t, d))
    results.sort(reverse=True)
    w, _, t, d = results[0]
    rows = [{"boss": name, "cap": cap, "legal": args.legal, "arena": arena, "unmodelled": unmodelled, "best_team": list(t), "win": w, "deaths_per_mon": d,
             "n": args.n_search, "all": [(list(t_), w_, d_) for w_, _, t_, d_ in results],
             "enemy": enemy_species, "pool": len(pool), "progress": progress, "shortlist": short}]
    print(f"{name:<10} legal={args.legal!s:<5} best team under search: win {w:5.0%} deaths/mon {d:.2f} "
          f"{list(t)}  (others: {', '.join(f'{w_:.0%}' for w_, _, _, _ in results[1:])})  ({time.time() - t0:.0f}s)", flush=True)
    bridge.close()
    return {"rows": rows}


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
    print(f"\nRadical Red best-six probe | legal moves only: {args.legal} | arena effects: {args.arena} | shortlist {SHORT}, top {TOP} teams, "
          f"search {args.search} x n={args.n_search} | no EVs, no items | enemy = RR AI, full info | {(time.time() - t0) / 60:.0f} min")
    print(f"  {'boss':<10} {'cap':>3} {'win':>5} {'deaths/mon':>10}  best team  | arena")
    for r in sorted(rows, key=lambda r: r["cap"]):
        ar = {k: v for k, v in r["arena"].items() if k != "ai_permanent"}
        print(f"  {r['boss']:<10} {r['cap']:>3} {r['win']:5.0%} {r['deaths_per_mon']:10.2f}  {', '.join(r['best_team'])}  | {ar or '-'}")


if __name__ == "__main__":
    main()
