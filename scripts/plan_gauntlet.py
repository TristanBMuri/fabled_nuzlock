"""Scripted run-level planner: the baseline any learned run policy must beat.

Per trainer: candidate parties = the base party (top-6 by level) plus every
single swap with a box mon; each is scored by the sim's death-probability
query (P(win) − λ·mean P(death) + HP margin); the best is fielded. At a fight
the best swap still finds hard, `--parties beam|all` widens the candidate set. Encounters are always
caught while there is room. Reports trainers cleared, trainers cleared
without a death, deaths, and slice clears over N episodes.

  uv run python scripts/plan_gauntlet.py --episodes 20
  uv run python scripts/plan_gauntlet.py --episodes 20 --policy models/game_policy_best.pt --search 3,8
"""

import argparse
import functools
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nuzlocke_env import DEEP_HP, EVOS, MAX_PARTY, ORDER, SLICE_LEN, NuzlockeEnv, Rules  # noqa: E402
from gauntlet import load_species  # noqa: E402

def swaps(base, alive):
    """The party plus every single swap with a box mon."""
    box = [i for i in alive if i not in base]
    return [tuple(base)] + [tuple(sorted(set(base) - {o} | {b}))
                            for o, b in itertools.product(base, box)]


def choose_party(env: NuzlockeEnv, lam: float, verbose: bool, fast: bool = False,
                 parties: str = "swap") -> list[int]:
    alive = [i for i, m in enumerate(env.roster) if m.alive]
    if len(alive) <= MAX_PARTY:
        return alive  # nuzlocke rule: field everything you can
    by_level = sorted(alive, key=lambda i: -env.roster[i].level)
    base = by_level[:MAX_PARTY]
    if fast:  # no risk queries: highest levels, for bulk harvesting
        return base
    memo = {}

    def score(p):
        # P(win) − λ·mean P(death) + margin: at n=8 most candidates show zero
        # deaths, and HP left is what separates a safe party from a lucky one
        if p not in memo:
            p_win, p_death, hp = env.risk(p)
            memo[p] = p_win - lam * float(np.mean(p_death)) + HP_WEIGHT * hp if p_death else p_win
        return memo[p]

    def hard(p):  # same trigger as the env's deep search (cached: free)
        p_win, p_death, hp = env.risk(p)
        return p_win < 1.0 or max(p_death, default=0.0) > 0.0 or hp < DEEP_HP

    front = sorted(swaps(base, alive), key=score, reverse=True)
    if parties != "swap" and hard(front[0]):
        # The swap set is the limit at a wall (README 2o): a hard fight may need
        # several changes from the base party. Widen only there (cost).
        if parties == "all":
            front = sorted((tuple(c) for c in itertools.combinations(alive, MAX_PARTY)),
                           key=score, reverse=True)
        else:  # beam: expand the single swaps of the top BEAM parties, BEAM_ROUNDS times
            for _ in range(BEAM_ROUNDS - 1):
                pool = set(front[:BEAM]) | {q for p in front[:BEAM] for q in swaps(p, alive)}
                front = sorted(pool, key=score, reverse=True)
    if verbose:
        print(f"    {len(memo)} candidates: best {score(front[0]):+.2f} "
              f"{[env.roster[i].species for i in front[0]]}"
              + (f", next {score(front[1]):+.2f}" if len(front) > 1 else ""))
    return list(front[0])


BEAM, BEAM_ROUNDS = 3, 3  # --parties beam: top-3 parties expanded three times
HP_WEIGHT = 0.2  # margin tie-breaker weight in the party score (--hp-weight)
DEATHS_AT: dict[int, int] = {}  # trainer index -> deaths (all episodes)
DEATHS_BY_SPECIES: dict[str, int] = {}  # species -> deaths (all episodes)
LOSSES_AT: dict[int, int] = {}  # trainer index -> lost battles


@functools.lru_cache(maxsize=None)
def final_bst(species: str) -> int:
    """Base stat total of the species' last reachable evolution (first branch,
    trade excluded): the scripted rule for choosing among an area's methods."""
    seen = set()
    while species not in seen:
        seen.add(species)
        nxt = [e["to"] for e in EVOS.get(species, []) if e["type"] != "trade"]
        if not nxt:
            break
        species = nxt[0]
    return sum(load_species().get(species, {}).get("baseStats", {}).values())


def episode(env: NuzlockeEnv, seed: int, lam: float, verbose: bool, fast: bool = False,
            parties: str = "swap") -> dict:
    obs, info = env.reset(seed=seed)
    deaths, first_death_at = 0, None
    while not env.done:
        if env.phase == "ENCOUNTER":
            alts, catch = env.pending[0], bool(info["action_mask"][1])
            pick = max(range(len(alts)), key=lambda i: final_bst(alts[i].species))  # the method choice
            if verbose:
                print(f"  offer {' | '.join(f'{m.species}@{m.level}' for m in alts)}: "
                      f"{'caught ' + alts[pick].species if catch else 'no room'}  box {len(env.roster)}/{env.max_roster}")
            env.step(1 + pick if catch else 0)
            info = {"action_mask": env._mask()}
            continue
        if env.party_locked:  # the League: no heal, no swaps — the fielded party fights on
            if verbose:
                print(f"  [locked] {ORDER[env.trainer_idx]['trainer_name']}: party fights on, "
                      f"PP left {[sum(env.carry[i].get('pp') or []) for i in env.party if i in env.carry]}")
        else:
            for i in choose_party(env, lam, verbose, fast, parties):
                if i not in env.party:
                    env.step(1 + i)
        alive_before = sum(m.alive for m in env.roster)
        t = env.trainer_idx
        env.step(0)  # confirm -> battle
        died = alive_before - sum(m.alive for m in env.roster)
        deaths += died
        if died:
            DEATHS_AT[t] = DEATHS_AT.get(t, 0) + died
            for m in env.graveyard[-died:]:
                DEATHS_BY_SPECIES[m.species] = DEATHS_BY_SPECIES.get(m.species, 0) + 1
        if env.trainer_idx == t:
            LOSSES_AT[t] = LOSSES_AT.get(t, 0) + 1
        if died and first_death_at is None:
            first_death_at = t
        if verbose:
            print(f"  trainer {t:2d} -> {'won' if env.trainer_idx > t else 'LOST'}, "
                  f"deaths {died} | {env.render()}")
    if verbose:
        print(f"  end of run at trainer {env.trainer_idx}:\n{env.tracker()}")
        info = {"action_mask": env._mask()}
    return {"cleared": env.trainer_idx, "deaths": deaths,
            "cleared_no_death": env.trainer_idx if first_death_at is None else first_death_at,
            "slice_clear": env.trainer_idx >= env.slice_len}


def rules(args: argparse.Namespace) -> Rules:
    return Rules(level_rule=args.level_rule, level_margin=args.level_margin, box=args.roster,
                 dupes_clause=not args.no_dupes, moveset=args.moveset)


def run_episodes(args: argparse.Namespace, seeds: list[int]) -> tuple[list[dict], dict, dict, int]:
    """One env, the given episode seeds (a worker). Returns the per-episode rows,
    the death / loss histograms and the number of deep-search fights."""
    global HP_WEIGHT
    import torch
    torch.set_num_threads(1)  # batch-1 MLP: the intra-op pool only adds overhead
    HP_WEIGHT = args.hp_weight
    if args.harvest:
        import gauntlet
        gauntlet.HARVEST = Path(args.harvest)
    width = lambda s: tuple(int(x) for x in s.split(",")) if s else None
    # risk_features off: the planner queries env.risk() for its candidates
    # directly and never reads the obs; leaving them on re-queried every
    # intermediate party while the team was being built (+50% cost).
    env = NuzlockeEnv(seed=args.seed, backend="sim", policy=args.policy, search=width(args.search),
                      rules=rules(args), risk_features=False, search_all=args.search_all,
                      slice_len=args.slice, deep=width(args.deep))
    t0, rows = time.time(), []
    for n, s in enumerate(seeds):
        r = episode(env, s, args.lam, args.verbose, args.fast, args.parties)
        rows.append(r)
        print(f"ep seed {s:4d}: cleared {r['cleared']:3d}  no-death {r['cleared_no_death']:3d}  "
              f"deaths {r['deaths']:2d}  {'CLEAR' if r['slice_clear'] else ''}  "
              f"({(time.time() - t0) / (n + 1):.0f}s/ep)", flush=True)
    return rows, DEATHS_AT, LOSSES_AT, env.n_deep, DEATHS_BY_SPECIES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", help="BC prior for the battler (default SimpleHeuristics), or "
                                     "'calc' for the scripted damage-calc bot (ai_util/calc_bot.py)")
    ap.add_argument("--search", help="k,rollouts for boss fights, e.g. 3,8")
    ap.add_argument("--search-all", action="store_true", help="search on every trainer")
    ap.add_argument("--deep", help="k,rollouts at breakpoints and fights the risk query "
                                   "flags as hard (a loss, a death or HP < DEEP_HP at n=8), e.g. 4,16")
    ap.add_argument("--parties", choices=["swap", "beam", "all"], default="swap",
                    help="candidate parties when the best single swap is still flagged hard: "
                         "beam = top-3 parties' swaps expanded 3 rounds (~10 s), "
                         "all = every 6-of-roster party (~2 min at 12 mons)")
    ap.add_argument("--lam", type=float, default=1.0, help="death penalty weight")
    ap.add_argument("--level-margin", type=int, default=0,
                    help="player level = rule level + this (XP proxy)")
    ap.add_argument("--level-rule", choices=["match", "cap"], default="cap",
                    help="cap = next boss's max level (hardcore rule); match = max enemy seen")
    ap.add_argument("--fast", action="store_true",
                    help="no risk queries (highest-level party): bulk harvesting")
    ap.add_argument("--slice", type=int, default=SLICE_LEN,
                    help="trainers per episode (21 = through Roark, 59 = through Gardenia, 167 = Maylene)")
    ap.add_argument("--roster", type=int, default=Rules.box,
                    help="box slots for living mons (default 30 = unbounded on the 186 slice, "
                         "starter + 21 areas; the old 12-slot box dropped 5-7 encounters per run)")
    ap.add_argument("--no-dupes", action="store_true", help="plain first encounter (default: dupes clause)")
    ap.add_argument("--moveset", choices=["last4", "best4"], default=Rules.moveset,
                    help="last4 = the last four level-up moves; best4 = strongest STAB, strongest "
                         "other-type attack, next strongest, one status move (see gauntlet.encounter_moves)")
    ap.add_argument("--hp-weight", type=float, default=HP_WEIGHT,
                    help="weight of mean HP left in the party score (0 = old score)")
    ap.add_argument("--harvest", metavar="JSONL",
                    help="record every battle's (party, enemy, won) here (same as GAUNTLET_HARVEST)")
    ap.add_argument("--workers", type=int, default=1,
                    help="episodes run in this many processes (one env + one sim each, ~1 GB; "
                         "check free -g first)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    seeds = [args.seed + e for e in range(args.episodes)]
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.starmap(run_episodes, [(args, seeds[i::args.workers])
                                                for i in range(args.workers) if seeds[i::args.workers]])
    else:
        parts = [run_episodes(args, seeds)]
    rows = [r for p in parts for r in p[0]]
    n_deep = sum(p[3] for p in parts)
    deaths_at, losses_at, by_species = {}, {}, {}  # merged histograms (the globals belong to the workers)
    for p in parts:
        for t, d in p[1].items():
            deaths_at[t] = deaths_at.get(t, 0) + d
        for t, d in p[2].items():
            losses_at[t] = losses_at.get(t, 0) + d
        for sp, d in (p[4] if len(p) > 4 else {}).items():
            by_species[sp] = by_species.get(sp, 0) + d
    k = lambda f: np.mean([r[f] for r in rows])
    print(f"\n{args.episodes} episodes | slice {args.slice} | battler={args.policy or 'SimpleHeuristics'} "
          f"search={args.search}{' (all)' if args.search_all else ''} deep={args.deep} "
          f"parties={args.parties} lam={args.lam} | rules: {rules(args)} "
          f"| {args.workers} workers, {(time.time() - t0) / 60:.0f} min\n"
          f"  trainers cleared        {k('cleared'):.2f}\n"
          f"  cleared without a death {k('cleared_no_death'):.2f}\n"
          f"  deaths per run          {k('deaths'):.2f}\n"
          f"  slice clear rate        {k('slice_clear'):.0%}"
          + (f"\n  deep-search fights      {n_deep / args.episodes:.1f} per run" if args.deep else ""))
    from nuzlocke_env import ORDER, TRAINERS
    print("  deadliest trainers (deaths / lost battles over all episodes):")
    for t, d in sorted(deaths_at.items(), key=lambda kv: -kv[1])[:8]:
        e = ORDER[t]
        team = ", ".join(f"{p['species']}@{p['level']}" for p in TRAINERS[e["trainer_id"]]["party"])
        print(f"    #{t:2d} {e['trainer_name']:<22} {e['breakpoint']:<10} "
              f"deaths {d:3d}  lost {losses_at.get(t, 0):2d}  [{team}]")
    if by_species:
        print("  deaths by species: " + ", ".join(
            f"{sp} {d}" for sp, d in sorted(by_species.items(), key=lambda kv: -kv[1])[:12]))


if __name__ == "__main__":
    main()
