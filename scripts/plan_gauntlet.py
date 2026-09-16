"""Scripted run-level planner: the baseline any learned run policy must beat.

Per trainer: candidate parties = best-known party plus every single swap with
a box mon; each is scored by the sim's death-probability query
(P(win) − λ·mean P(death)); the best is fielded. Encounters are always
caught while there is room. Reports trainers cleared, trainers cleared
without a death, deaths, and slice clears over N episodes.

  uv run python scripts/plan_gauntlet.py --episodes 20
  uv run python scripts/plan_gauntlet.py --episodes 20 --policy models/game_policy_best.pt --search 3,8
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nuzlocke_env import MAX_PARTY, SLICE_LEN, NuzlockeEnv  # noqa: E402


def choose_party(env: NuzlockeEnv, lam: float, verbose: bool, fast: bool = False) -> list[int]:
    alive = [i for i, m in enumerate(env.roster) if m.alive]
    if len(alive) <= MAX_PARTY:
        return alive  # nuzlocke rule: field everything you can
    by_level = sorted(alive, key=lambda i: -env.roster[i].level)
    base = by_level[:MAX_PARTY]
    if fast:  # no risk queries: highest levels, for bulk harvesting
        return base
    box = [i for i in alive if i not in base]
    cands = [tuple(base)] + [tuple(sorted(set(base) - {o} | {b}))
                             for o, b in itertools.product(base, box)]

    def score(p):
        # P(win) − λ·mean P(death) + margin: at n=8 most candidates show zero
        # deaths, and HP left is what separates a safe party from a lucky one
        p_win, p_death, hp = env.risk(p)
        return p_win - lam * float(np.mean(p_death)) + HP_WEIGHT * hp if p_death else p_win

    scored = sorted(((score(p), p) for p in cands), reverse=True)
    if verbose:
        best, second = scored[0], scored[1] if len(scored) > 1 else None
        print(f"    {len(cands)} candidates: best {best[0]:+.2f} "
              f"{[env.roster[i].species for i in best[1]]}"
              + (f", next {second[0]:+.2f}" if second else ""))
    return list(scored[0][1])


HP_WEIGHT = 0.2  # margin tie-breaker weight in the party score (--hp-weight)
DEATHS_AT: dict[int, int] = {}  # trainer index -> deaths (all episodes)
LOSSES_AT: dict[int, int] = {}  # trainer index -> lost battles


def episode(env: NuzlockeEnv, seed: int, lam: float, verbose: bool, fast: bool = False) -> dict:
    obs, info = env.reset(seed=seed)
    deaths, first_death_at = 0, None
    while not env.done:
        if env.phase == "ENCOUNTER":
            env.step(1 if info["action_mask"][1] else 0)
            info = {"action_mask": env._mask()}
            continue
        for i in choose_party(env, lam, verbose, fast):
            if i not in env.party:
                env.step(1 + i)
        alive_before = sum(m.alive for m in env.roster)
        t = env.trainer_idx
        env.step(0)  # confirm -> battle
        died = alive_before - sum(m.alive for m in env.roster)
        deaths += died
        if died:
            DEATHS_AT[t] = DEATHS_AT.get(t, 0) + died
        if env.trainer_idx == t:
            LOSSES_AT[t] = LOSSES_AT.get(t, 0) + 1
        if died and first_death_at is None:
            first_death_at = t
        if verbose:
            print(f"  trainer {t:2d} -> {'won' if env.trainer_idx > t else 'LOST'}, "
                  f"deaths {died} | {env.render()}")
        info = {"action_mask": env._mask()}
    return {"cleared": env.trainer_idx, "deaths": deaths,
            "cleared_no_death": env.trainer_idx if first_death_at is None else first_death_at,
            "slice_clear": env.trainer_idx >= env.slice_len}


def main() -> None:
    global HP_WEIGHT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", help="BC prior for the battler (default SimpleHeuristics)")
    ap.add_argument("--search", help="k,rollouts for boss fights, e.g. 3,8")
    ap.add_argument("--search-all", action="store_true", help="search on every trainer")
    ap.add_argument("--lam", type=float, default=1.0, help="death penalty weight")
    ap.add_argument("--level-margin", type=int, default=0,
                    help="player level = rule level + this (XP proxy)")
    ap.add_argument("--level-rule", choices=["match", "cap"], default="cap",
                    help="cap = next boss's max level (hardcore rule); match = max enemy seen")
    ap.add_argument("--fast", action="store_true",
                    help="no risk queries (highest-level party): bulk harvesting")
    ap.add_argument("--slice", type=int, default=SLICE_LEN,
                    help="trainers per episode (51 = through Roark, 186 = through Gardenia)")
    ap.add_argument("--hp-weight", type=float, default=HP_WEIGHT,
                    help="weight of mean HP left in the party score (0 = old score)")
    ap.add_argument("--harvest", metavar="JSONL",
                    help="record every battle's (party, enemy, won) here (same as GAUNTLET_HARVEST)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    HP_WEIGHT = args.hp_weight
    if args.harvest:
        import gauntlet
        gauntlet.HARVEST = Path(args.harvest)
    search = tuple(int(x) for x in args.search.split(",")) if args.search else None
    # risk_features off: the planner queries env.risk() for its candidates
    # directly and never reads the obs; leaving them on re-queried every
    # intermediate party while the team was being built (+50% cost).
    env = NuzlockeEnv(seed=args.seed, backend="sim", policy=args.policy, search=search,
                      level_margin=args.level_margin, level_rule=args.level_rule,
                      risk_features=False, search_all=args.search_all, slice_len=args.slice)
    t0, rows = time.time(), []
    for e in range(args.episodes):
        r = episode(env, args.seed + e, args.lam, args.verbose, args.fast)
        rows.append(r)
        print(f"ep {e:3d}: cleared {r['cleared']:2d}  no-death {r['cleared_no_death']:2d}  "
              f"deaths {r['deaths']}  {'CLEAR' if r['slice_clear'] else ''}  "
              f"({(time.time() - t0) / (e + 1):.0f}s/ep)", flush=True)
    k = lambda f: np.mean([r[f] for r in rows])
    print(f"\n{args.episodes} episodes | slice {args.slice} | battler={args.policy or 'SimpleHeuristics'} "
          f"search={search}{' (all)' if args.search_all else ''} lam={args.lam} "
          f"level_rule={args.level_rule} "
          f"margin={args.level_margin}\n"
          f"  trainers cleared        {k('cleared'):.2f}\n"
          f"  cleared without a death {k('cleared_no_death'):.2f}\n"
          f"  deaths per run          {k('deaths'):.2f}\n"
          f"  slice clear rate        {k('slice_clear'):.0%}")
    from nuzlocke_env import ORDER, TRAINERS
    print("  deadliest trainers (deaths / lost battles over all episodes):")
    for t, d in sorted(DEATHS_AT.items(), key=lambda kv: -kv[1])[:8]:
        e = ORDER[t]
        team = ", ".join(f"{p['species']}@{p['level']}" for p in TRAINERS[e["trainer_id"]]["party"])
        print(f"    #{t:2d} {e['trainer_name']:<22} {e['breakpoint']:<10} "
              f"deaths {d:3d}  lost {LOSSES_AT.get(t, 0):2d}  [{team}]")


if __name__ == "__main__":
    main()
