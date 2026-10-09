"""Scripted run-level planner: the baseline any learned run policy must beat.

Per trainer: candidate parties = the base party (top-6 by level) plus every
single swap with a box mon; each is scored by the sim's death-probability
query (P(win) − λ·mean P(death) + HP margin); the best is fielded. At a fight
the best swap still finds hard, `--parties beam|all` widens the candidate set. At the
League opener `--league K` builds the party from the whole box (E8, null). Encounters are
always caught while there is room. Reports trainers cleared, trainers cleared
without a death, deaths, and slice clears over N episodes.

  uv run python scripts/plan_gauntlet.py --episodes 20
  uv run python scripts/plan_gauntlet.py --episodes 20 --policy models/game_policy_best.pt --search 3,8
"""

import argparse
import functools
import itertools
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nuzlocke_env import DEEP_HP, EVOS, MAX_PARTY, ORDER, RISK_BATTLES, SLICE_LEN, NuzlockeEnv, Rules  # noqa: E402
from gauntlet import encounter_moves, load_species  # noqa: E402


def party_score(env: NuzlockeEnv, p, lam: float, n: int = RISK_BATTLES) -> float:
    # P(win) − λ·mean P(death) + margin: at n=8 most candidates show zero
    # deaths, and HP left is what separates a safe party from a lucky one
    # (env.risk caches per party, moves, trainer and n)
    p_win, p_death, hp = env.risk(tuple(p), n=n)
    return p_win - lam * float(np.mean(p_death)) + HP_WEIGHT * hp if p_death else p_win


def hard(env: NuzlockeEnv, p) -> bool:
    """Same trigger as the env's deep search (cached: free)."""
    p_win, p_death, hp = env.risk(tuple(p))
    return p_win < 1.0 or max(p_death, default=0.0) > 0.0 or hp < DEEP_HP


def swaps(base, alive):
    """The party plus every single swap with a box mon."""
    box = [i for i in alive if i not in base]
    return [tuple(base)] + [tuple(sorted(set(base) - {o} | {b}))
                            for o, b in itertools.product(base, box)]


def choose_party(env: NuzlockeEnv, lam: float, verbose: bool, fast: bool = False,
                 parties: str = "swap", league: int = 0) -> list[int]:
    alive = [i for i, m in enumerate(env.roster) if m.alive]
    if len(alive) <= MAX_PARTY:
        return alive  # nuzlocke rule: field everything you can
    by_level = sorted(alive, key=lambda i: -env.roster[i].level)
    base = by_level[:MAX_PARTY]
    if fast:  # no risk queries: highest levels, for bulk harvesting
        return base
    score = lambda p, n=RISK_BATTLES: party_score(env, p, lam, n)
    front = sorted(swaps(base, alive), key=score, reverse=True)
    if league and env._block():
        # E8: the League party from the whole box. Every mon is at the cap, so the
        # base is roster order and carries filler. Shortlist the mons by the best
        # swap they appear in, score every 6-subset of the shortlist on the block,
        # re-score the top few with more samples (n=8 picks on noise among hundreds).
        best_with = lambda i: max(score(p) for p in front if i in p)
        short = sorted(alive, key=best_with, reverse=True)[:league]
        pool = sorted((tuple(sorted(c)) for c in itertools.combinations(short, MAX_PARTY)),
                      key=score, reverse=True)
        front = sorted(pool[:LEAGUE_TOP], key=lambda p: score(p, LEAGUE_N), reverse=True)
    elif parties != "swap" and hard(env, front[0]):
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
        print(f"    {len(front)} candidates: best {score(front[0]):+.2f} "
              f"{[env.roster[i].species for i in front[0]]}"
              + (f", next {score(front[1]):+.2f}" if len(front) > 1 else ""))
    return list(front[0])


def choose_moves(env: NuzlockeEnv, party: list[int], lam: float, verbose: bool) -> int:
    """E7(d): at a hard fight the sim picks each party mon's four moves. Candidates =
    the current set with one slot swapped for one of the next MOVE_ALTS moves of the
    ranked pool (level-up + TMs in hand); a variant replaces the set when it beats it
    at n=8 and again at 2n (n=8 alone accepts noise). Greedy, one pass over the party;
    the choice persists (Mon.chosen) while the pool holds it. Returns the sets changed."""
    n2, changed = 2 * RISK_BATTLES, 0
    base = party_score(env, party, lam)
    for i in party:
        m = env.roster[i]
        cur, prev = list(m.moves), m.chosen  # the level rule inside risk() rewrites m.moves: vary m.chosen
        pool = encounter_moves(m.species, m.level, env.rules.moveset, env._progress, k=None)
        alts = [x for x in pool if x not in cur][:MOVE_ALTS]
        best, best_s = None, base
        for v in (cur[:j] + [a] + cur[j + 1:] for a in alts for j in range(len(cur))):
            m.chosen = v
            s = party_score(env, party, lam)
            if s > best_s:
                best, best_s = v, s
        m.chosen = prev
        if best:  # confirm at 2n against the current set
            s_cur = party_score(env, party, lam, n2)
            m.chosen = best
            if party_score(env, party, lam, n2) > s_cur:
                base, changed = best_s, changed + 1
                if verbose:
                    print(f"    moves {m.species}: {cur} -> {best} ({base:+.2f})")
            else:
                m.chosen, m.moves = prev, cur
        else:
            m.moves = cur
    return changed


MOVE_ALTS = 3  # --moves sim: the next 3 ranked moves are tried in every slot (12 variants per mon)
BEAM, BEAM_ROUNDS = 3, 3  # --parties beam: top-3 parties expanded three times
LEAGUE_TOP, LEAGUE_N = 8, 32  # --league: the top-8 subsets re-scored at n=32
HP_WEIGHT = 0.2  # margin tie-breaker weight in the party score (--hp-weight)
DEATHS_AT: dict[int, int] = {}  # trainer index -> deaths (all episodes)
DEATHS_BY_SPECIES: dict[str, int] = {}  # species -> deaths (all episodes)
LOSSES_AT: dict[int, int] = {}  # trainer index -> lost battles
MOVE_CHANGES = [0]  # movesets the sim changed (--moves sim, all episodes)


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


# Run state at the League opener (league_test.py replays the block from it). The
# runner, the risk cache and the settings stay with the env that restores it.
SNAPSHOT = ("run_id", "trainer_idx", "roster", "graveyard", "party", "_rival_family", "max_enemy_level",
            "phase", "pending", "_queue", "carry", "dropped", "_route_cursor", "done", "rng")


def episode(env: NuzlockeEnv, seed: int, lam: float, verbose: bool, fast: bool = False,
            parties: str = "swap", league: int = 0, moves: str = "rule", save: Path | None = None) -> dict:
    obs, info = env.reset(seed=seed)
    return play(env, info, lam, verbose, fast, parties, league, moves, save)


def play(env: NuzlockeEnv, info: dict, lam: float, verbose: bool, fast: bool = False,
         parties: str = "swap", league: int = 0, moves: str = "rule", save: Path | None = None) -> dict:
    """The planner from the env's current state to the end of the run. save: pickle the
    run state here at the League opener (once)."""
    deaths, first_death_at, deaths_at = 0, None, {}
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
            if save and env._block():
                save.parent.mkdir(parents=True, exist_ok=True)
                save.write_bytes(pickle.dumps({k: getattr(env, k) for k in SNAPSHOT}))
                save = None
            party = choose_party(env, lam, verbose, fast, parties, league)
            for i in party:
                if i not in env.party:
                    env.step(1 + i)
            # not at the League opener: the fast brain's block ranking does not transfer to
            # search play at Cynthia (E8; moves chosen there took clears 70% -> 20%, README)
            if moves == "sim" and not fast and len(party) > 1 and not env._block() and hard(env, party):
                MOVE_CHANGES[0] += choose_moves(env, party, lam, verbose)
        alive_before = sum(m.alive for m in env.roster)
        t = env.trainer_idx
        env.step(0)  # confirm -> battle
        died = alive_before - sum(m.alive for m in env.roster)
        deaths += died
        if died:
            deaths_at[t] = died
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
            "slice_clear": env.trainer_idx >= env.slice_len, "deaths_at": deaths_at}


def rules(args: argparse.Namespace) -> Rules:
    return Rules(level_rule=args.level_rule, level_margin=args.level_margin, box=args.roster,
                 dupes_clause=not args.no_dupes, moveset=args.moveset, tms=not args.no_tms, evs=args.evs,
                 items=not args.no_items)


def make_env(args: argparse.Namespace) -> NuzlockeEnv:
    global HP_WEIGHT
    import torch
    torch.set_num_threads(1)  # batch-1 MLP: the intra-op pool only adds overhead
    HP_WEIGHT = args.hp_weight
    width = lambda s: tuple(int(x) for x in s.split(",")) if s else None
    # risk_features off: the planner queries env.risk() for its candidates
    # directly and never reads the obs; leaving them on re-queried every
    # intermediate party while the team was being built (+50% cost).
    return NuzlockeEnv(seed=args.seed, backend="sim", policy=args.policy, search=width(args.search),
                       rules=rules(args), risk_features=False, search_all=args.search_all,
                       slice_len=args.slice, deep=width(args.deep))


def run_episodes(args: argparse.Namespace, seeds: list[int]) -> tuple[list[dict], dict, dict, int]:
    """One env, the given episode seeds (a worker). Returns the per-episode rows,
    the death / loss histograms and the number of deep-search fights."""
    if args.harvest:
        import gauntlet
        gauntlet.HARVEST = Path(args.harvest)
    env = make_env(args)
    t0, rows = time.time(), []
    for n, s in enumerate(seeds):
        save = Path(args.save_league) / f"seed{s}.pkl" if args.save_league else None
        r = episode(env, s, args.lam, args.verbose, args.fast, args.parties, args.league, args.moves, save)
        rows.append(r)
        print(f"ep seed {s:4d}: cleared {r['cleared']:3d}  no-death {r['cleared_no_death']:3d}  "
              f"deaths {r['deaths']:2d}  {'CLEAR' if r['slice_clear'] else ''}  "
              f"({(time.time() - t0) / (n + 1):.0f}s/ep)", flush=True)
    return rows, DEATHS_AT, LOSSES_AT, env.n_deep, DEATHS_BY_SPECIES, MOVE_CHANGES[0]


def parser(doc: str = __doc__) -> argparse.ArgumentParser:
    """The planner and rules flags (league_test.py takes the same ones)."""
    ap = argparse.ArgumentParser(description=doc)
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
    ap.add_argument("--league", type=int, default=0, metavar="K",
                    help="League party from the whole box: every 6-subset of the K mons with the best "
                         "swap score, on the block, top-8 re-scored at n=32 (~3 min; null at K=10, "
                         "README 2026-09-23; 0 = plain swaps)")
    ap.add_argument("--lam", type=float, default=1.0, help="death penalty weight")
    ap.add_argument("--moves", choices=["rule", "sim"], default="rule",
                    help="sim = at hard fights the sim re-picks each party mon's four moves from its "
                         "pool, one slot swap at a time (E7d); rule = best4 as computed")
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
    ap.add_argument("--no-tms", action="store_true",
                    help="level-up moves only (default: best4 also over the TMs handed out so far, "
                         "E7: reusable, data/bdsp/tms.json; 20%% -> 70%% clears)")
    ap.add_argument("--no-items", action="store_true",
                    help="no held items (default: items found so far, one per mon, by gauntlet.assign_items; "
                         "E9, data/bdsp/items.json)")
    ap.add_argument("--evs", choices=["flat", "trained"], default=Rules.evs,
                    help="player EVs: flat 85 each, or trained = 252 in the better attacking stat, 252 Spe, 4 HP")
    ap.add_argument("--hp-weight", type=float, default=HP_WEIGHT,
                    help="weight of mean HP left in the party score (0 = old score)")
    ap.add_argument("--harvest", metavar="JSONL",
                    help="record every battle's (party, enemy, won) here (same as GAUNTLET_HARVEST)")
    ap.add_argument("--workers", type=int, default=1,
                    help="episodes run in this many processes (one env + one sim each, ~1 GB; "
                         "check free -g first)")
    ap.add_argument("--save-league", metavar="DIR",
                    help="pickle each episode's run state at the League opener to DIR/seed<N>.pkl "
                         "(the input of scripts/league_test.py)")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
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
    n_moves = sum(p[5] for p in parts)
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
          f"parties={args.parties} league={args.league} moves={args.moves} lam={args.lam} | rules: {rules(args)} "
          f"| {args.workers} workers, {(time.time() - t0) / 60:.0f} min\n"
          f"  trainers cleared        {k('cleared'):.2f}\n"
          f"  cleared without a death {k('cleared_no_death'):.2f}\n"
          f"  deaths per run          {k('deaths'):.2f}\n"
          f"  slice clear rate        {k('slice_clear'):.0%}"
          + (f"\n  deep-search fights      {n_deep / args.episodes:.1f} per run" if args.deep else "")
          + (f"\n  movesets changed        {n_moves / args.episodes:.1f} per run" if args.moves == "sim" else ""))
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
