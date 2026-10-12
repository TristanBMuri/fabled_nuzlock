"""Scripted run-level planner: the run layer.

Per trainer: candidate parties = the base party (top-6 by level) plus every
single swap with a box mon; each is scored by the sim's death-probability
query (P(win) − λ·mean P(death) + HP margin); the best is fielded. Encounters
are always caught while there is room; among an area's methods the one with the
highest final base-stat total. Reports trainers cleared, trainers cleared
without a death, deaths and slice clears over N episodes, with the death
histogram by trainer and by species.

  uv run python scripts/plan_gauntlet.py --episodes 20
  # the full-game benchmark:
  uv run python -u scripts/plan_gauntlet.py --episodes 20 --seed 300 --slice 405 \\
      --policy calc --search 9,8 --search-all --deep 9,16 --workers 10

Measured null and removed (results in the notebook): --refine / --climb,
--parties beam|all, --league K (E8), --moves sim (E7d), --hp-weight.
"""

import argparse
import functools
import itertools
import json
import os
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nuzlocke_env import DEEP_HP, EVOS, MAX_PARTY, ORDER, RISK_BATTLES, SLICE_LEN, TRAINERS, NuzlockeEnv, Rules  # noqa: E402
from gauntlet import load_species, pack_team, trainer_mons  # noqa: E402

HP_WEIGHT = 0.2  # margin tie-breaker weight in the party score (most candidates show 0 deaths at n=8)
SWAPS = "all"  # --swaps: score every single swap, or (hard) only when the cover six's own query is not clean
DEATHS_AT: dict[int, int] = {}  # trainer index -> deaths (all episodes of this process)
DEATHS_BY_SPECIES: dict[str, int] = {}  # species -> deaths
LOSSES_AT: dict[int, int] = {}  # trainer index -> lost battles
EVENTS: Path | None = None  # --out X.jsonl -> X.events.jsonl: one JSON line per battle / episode (ui/)


def emit(kind: str, **data) -> None:
    """One dashboard event (ui/build.py folds them); workers append to the same file."""
    if EVENTS is not None:
        with EVENTS.open("a") as f:
            f.write(json.dumps({"kind": kind, "t": time.time(), **data}) + "\n")


def party_score(env: NuzlockeEnv, p, lam: float, n: int = RISK_BATTLES) -> float:
    # P(win) − λ·mean P(death) + margin: at n=8 most candidates show zero
    # deaths, and HP left is what separates a safe party from a lucky one
    # (env.risk caches per party, trainer and n)
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


def cover_base(env: NuzlockeEnv, alive: list[int]) -> list[int]:
    """The six by matchup cover (ai_util.matchup.cover: safe answers first, then race-winners,
    then total edge) from the whole box against the next trainer's team, or against every
    member of the locked block it opens (the League: that party is locked)."""
    from ai_util.matchup import cover, rows_for
    from runner import BATTLE_FORMAT
    mons = {i: env.roster[i] for i in alive}
    rows: dict = {i: {} for i in alive}
    for k in env._block() or [env.trainer_idx]:
        enemy = pack_team(trainer_mons(TRAINERS[ORDER[k]["trainer_id"]]))
        for i, d in rows_for(env.runner.bridge, BATTLE_FORMAT, mons, enemy, None).items():
            rows[i].update({f"{k}:{sp}": p for sp, p in d.items()})
    enemies = sorted({e for d in rows.values() for e in d})
    return sorted(cover({i: d for i, d in rows.items() if d}, enemies, MAX_PARTY))


def choose_party(env: NuzlockeEnv, lam: float, verbose: bool, fast: bool = False) -> list[int]:
    alive = [i for i, m in enumerate(env.roster) if m.alive]
    if len(alive) <= MAX_PARTY:
        return alive  # nuzlocke rule: field everything you can
    if fast:  # no risk queries: highest levels, for bulk harvesting (ties: catch order)
        return sorted(alive, key=lambda i: -env.roster[i].level)[:MAX_PARTY]
    # 2026-10-12: the base was top-6-by-level, i.e. the first six caught once the level rule
    # puts everyone at the cap, and single swaps only ever moved one slot of it
    base = cover_base(env, alive)
    score = lambda p: party_score(env, p, lam)
    # 2026-10-12: the swap scoring was 3/4 of an episode (~6 x box candidates x 8 battles per
    # trainer); hard() is the deep-search trigger on the base party's own (cached) query
    cands = swaps(base, alive) if SWAPS == "all" or hard(env, base) else [tuple(base)]
    front = sorted(cands, key=score, reverse=True)
    if verbose:
        print(f"    {len(front)} candidates: best {score(front[0]):+.2f} "
              f"{[env.roster[i].species for i in front[0]]}"
              + (f", next {score(front[1]):+.2f}" if len(front) > 1 else ""))
    return list(front[0])


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
            save: Path | None = None) -> dict:
    env.reset(seed=seed)
    if EVENTS is not None:
        env.trace_tag, env.trace_seed = f"{EVENTS.name.removesuffix('.events.jsonl')}_s{seed}", seed
    return play(env, lam, verbose, fast, save, seed=seed)


def play(env: NuzlockeEnv, lam: float, verbose: bool, fast: bool = False, save: Path | None = None,
         seed: int | None = None) -> dict:
    """The planner from the env's current state to the end of the run. save: pickle the
    run state here at the League opener (once). seed: names the episode in the events."""
    deaths, first_death_at, deaths_at, graves = 0, None, {}, []
    while not env.done:
        if env.phase == "ENCOUNTER":
            alts, catch = env.pending[0], env.can_catch
            pick = max(range(len(alts)), key=lambda i: final_bst(alts[i].species))  # the method choice
            if verbose:
                print(f"  offer {' | '.join(f'{m.species}@{m.level}' for m in alts)}: "
                      f"{'caught ' + alts[pick].species if catch else 'no room'}  box {len(env.roster)}/{env.max_roster}")
            env.step(1 + pick if catch else 0)
            continue
        if env.party_locked:  # the League: no swaps, full heal between fights — the fielded party fights on
            if verbose:
                print(f"  [locked] {ORDER[env.trainer_idx]['trainer_name']}: party fights on, "
                      f"healed")
        else:
            if save and env._block():
                save.parent.mkdir(parents=True, exist_ok=True)
                save.write_bytes(pickle.dumps({k: getattr(env, k) for k in SNAPSHOT}))
                save = None
            for i in choose_party(env, lam, verbose, fast):
                env.step(1 + i)
        alive_before = sum(m.alive for m in env.roster)
        t = env.trainer_idx
        fielded = [f"{env.roster[i].species}@{env.roster[i].level}" for i in env.party]
        env.step(0)  # confirm -> battle
        died = alive_before - sum(m.alive for m in env.roster)
        deaths += died
        if died:
            deaths_at[t] = died
            DEATHS_AT[t] = DEATHS_AT.get(t, 0) + died
            for m in env.graveyard[-died:]:
                DEATHS_BY_SPECIES[m.species] = DEATHS_BY_SPECIES.get(m.species, 0) + 1
                graves.append([t, m.species])
        trace = f"trace_{env.trace_tag}_{t}.json" if ORDER[t]["breakpoint"] == "champion" else None
        emit("battle", seed=seed, i=t, won=env.trainer_idx > t, party=fielded,
             died=[m.species for m in env.graveyard[-died:]] if died else [], box=sum(m.alive for m in env.roster),
             **({"trace": f"{EVENTS.parent.name}/{trace}"} if trace and EVENTS is not None else {}))
        if env.trainer_idx == t:
            LOSSES_AT[t] = LOSSES_AT.get(t, 0) + 1
        if died and first_death_at is None:
            first_death_at = t
        if verbose:
            print(f"  trainer {t:2d} -> {'won' if env.trainer_idx > t else 'LOST'}, "
                  f"deaths {died} | {env.render()}")
    if verbose:
        print(f"  end of run at trainer {env.trainer_idx}:\n{env.tracker()}")
    r = {"cleared": env.trainer_idx, "deaths": deaths,
         "cleared_no_death": env.trainer_idx if first_death_at is None else first_death_at,
         "slice_clear": env.trainer_idx >= env.slice_len, "deaths_at": deaths_at}
    emit("episode", seed=seed, cleared=r["cleared"], no_death=r["cleared_no_death"], deaths=deaths,
         clear=r["slice_clear"], graves=graves, party=[f"{env.roster[i].species}@{env.roster[i].level}" for i in env.party])
    return r


def rules(args: argparse.Namespace) -> Rules:
    return Rules(level_rule=args.level_rule, level_margin=args.level_margin, box=args.roster,
                 dupes_clause=not args.no_dupes, moveset=args.moveset, tms=not args.no_tms, evs=args.evs,
                 items=not args.no_items)


def make_env(args: argparse.Namespace) -> NuzlockeEnv:
    width = lambda s: tuple(int(x) for x in s.split(",")) if s else None
    return NuzlockeEnv(seed=args.seed, policy=args.policy, search=width(args.search), rules=rules(args),
                       search_all=args.search_all, slice_len=args.slice, deep=width(args.deep),
                       full_info=args.full_info)


def run_episodes(args: argparse.Namespace, seeds: list[int]) -> tuple:
    """One env, the given episode seeds (a worker). Returns the per-episode rows,
    the death / loss histograms, the number of deep-search fights and the
    deaths by species."""
    import runner
    global EVENTS
    apply(args)
    EVENTS = Path(args.out).with_suffix(".events.jsonl") if args.out else None
    runner.TRACES = EVENTS.parent if EVENTS is not None else None
    if args.harvest:
        runner.HARVEST = Path(args.harvest)
    env = make_env(args)
    t0, rows = time.time(), []
    for n, s in enumerate(seeds):
        save = Path(args.save_league) / f"seed{s}.pkl" if args.save_league else None
        r = episode(env, s, args.lam, args.verbose, args.fast, save)
        rows.append(r)
        print(f"ep seed {s:4d}: cleared {r['cleared']:3d}  no-death {r['cleared_no_death']:3d}  "
              f"deaths {r['deaths']:2d}  {'CLEAR' if r['slice_clear'] else ''}  "
              f"({(time.time() - t0) / (n + 1):.0f}s/ep)", flush=True)
    return rows, DEATHS_AT, LOSSES_AT, env.n_deep, DEATHS_BY_SPECIES


def apply(args: argparse.Namespace) -> None:
    """The planner flags that live in module state (a worker calls it once; league_test.py too)."""
    import runner
    global SWAPS
    runner.ENEMY = args.enemy
    SWAPS = args.swaps


def parser(doc: str = __doc__) -> argparse.ArgumentParser:
    """The planner and rules flags (league_test.py takes the same ones)."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", help="'calc' for the damage-calc bot (ai_util/calc_bot.py), a BC prior's "
                                     ".pt path, or omitted for SimpleHeuristics")
    ap.add_argument("--search", help="k,rollouts for boss fights, e.g. 9,8")
    ap.add_argument("--search-all", action="store_true", help="search on every trainer")
    ap.add_argument("--deep", help="k,rollouts at breakpoints and fights the risk query "
                                   "flags as hard (a loss, a death or HP < DEEP_HP at n=8), e.g. 9,16")
    ap.add_argument("--full-info", action="store_true",
                    help="the battler knows every enemy set, item, ability and real stat from turn one "
                         "(the full-docs regime, project_overview.md 2); default: sets revealed in play, "
                         "stats estimated at 31 IV / 85 EV")
    ap.add_argument("--lam", type=float, default=1.0, help="death penalty weight in the party score")
    ap.add_argument("--swaps", choices=["all", "hard"], default="all",
                    help="score every single swap of the cover six (all), or only when the six's own risk "
                         "query shows a loss, a death or HP < DEEP_HP (hard): 3/4 of an episode is swap scoring")
    ap.add_argument("--level-rule", choices=["match", "cap"], default="cap",
                    help="cap = next boss's max level (hardcore rule); match = max enemy seen")
    ap.add_argument("--level-margin", type=int, default=0,
                    help="player level = rule level + this (XP proxy)")
    ap.add_argument("--fast", action="store_true",
                    help="no risk queries (highest-level party): bulk harvesting")
    ap.add_argument("--slice", type=int, default=SLICE_LEN,
                    help="trainers per episode (21 = through Roark, 59 = Gardenia, 167 = Maylene, 405 = the game)")
    ap.add_argument("--roster", type=int, default=Rules.box,
                    help="box slots for living mons (default 50 = unbounded: starter + 45 areas)")
    ap.add_argument("--no-dupes", action="store_true", help="plain first encounter (default: dupes clause)")
    ap.add_argument("--moveset", choices=["last4", "best4"], default=Rules.moveset,
                    help="last4 = the last four level-up moves; best4 = strongest STAB, strongest "
                         "other-type attack, next strongest, one status move (gauntlet.encounter_moves)")
    ap.add_argument("--no-tms", action="store_true",
                    help="level-up moves only (default: best4 also over the TMs handed out so far, "
                         "E7: reusable, data/bdsp/tms.json)")
    ap.add_argument("--no-items", action="store_true",
                    help="no held items (default: items found so far, one per mon, gauntlet.assign_items; "
                         "E9, data/bdsp/items.json)")
    ap.add_argument("--evs", choices=["flat", "trained"], default=Rules.evs,
                    help="player EVs: flat 85 each, or trained = 252 in the better attacking stat, 252 Spe, 4 HP")
    ap.add_argument("--harvest", metavar="JSONL",
                    help="record every battle's (party, enemy, won) here (same as GAUNTLET_HARVEST)")
    ap.add_argument("--workers", type=int, default=1,
                    help="episodes run in this many processes (one env + one sim each, ~1 GB; "
                         "check free -g first)")
    ap.add_argument("--save-league", metavar="DIR",
                    help="pickle each episode's run state at the League opener to DIR/seed<N>.pkl "
                         "(the input of scripts/league_test.py)")
    ap.add_argument("--enemy", choices=["heuristic", "bdsp"], default="heuristic",
                    help="enemy AI: SimpleHeuristics or the Gen 4 trainer AI (ai_util/bdsp_ai.py)")
    ap.add_argument("--out", help="X.jsonl: per-episode rows there, dashboard events in X.events.jsonl (ui/)")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> None:
    args = parser().parse_args()
    seeds = [args.seed + e for e in range(args.episodes)]
    t0 = time.time()
    global EVENTS
    if args.out:
        EVENTS = Path(args.out).with_suffix(".events.jsonl")
        EVENTS.unlink(missing_ok=True)
        emit("start", mode="gauntlet", config=vars(args), bosses=[], pid=os.getpid(), seeds=seeds,
             order=[[e["trainer_name"], e["location"], e["breakpoint"]] for e in ORDER[:args.slice]])
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
        for sp, d in p[4].items():
            by_species[sp] = by_species.get(sp, 0) + d
    k = lambda f: np.mean([r[f] for r in rows])
    print(f"\n{args.episodes} episodes | slice {args.slice} | battler={args.policy or 'SimpleHeuristics'} "
          f"search={args.search}{' (all)' if args.search_all else ''} deep={args.deep} lam={args.lam} "
          f"full_info={args.full_info} "
          f"| rules: {rules(args)} | {args.workers} workers, {(time.time() - t0) / 60:.0f} min\n"
          f"  trainers cleared        {k('cleared'):.2f}\n"
          f"  cleared without a death {k('cleared_no_death'):.2f}\n"
          f"  deaths per run          {k('deaths'):.2f}\n"
          f"  slice clear rate        {k('slice_clear'):.0%}"
          + (f"\n  deep-search fights      {n_deep / args.episodes:.1f} per run" if args.deep else ""))
    print("  deadliest trainers (deaths / lost battles over all episodes):")
    for t, d in sorted(deaths_at.items(), key=lambda kv: -kv[1])[:8]:
        e = ORDER[t]
        team = ", ".join(f"{p['species']}@{p['level']}" for p in TRAINERS[e["trainer_id"]]["party"])
        print(f"    #{t:2d} {e['trainer_name']:<22} {e['breakpoint']:<10} "
              f"deaths {d:3d}  lost {losses_at.get(t, 0):2d}  [{team}]")
    if by_species:
        print("  deaths by species: " + ", ".join(
            f"{sp} {d}" for sp, d in sorted(by_species.items(), key=lambda kv: -kv[1])[:12]))
    if args.out:
        Path(args.out).write_text("".join(json.dumps({k: v for k, v in r.items() if k != "deaths_at"}) + "\n" for r in rows))
        emit("end", minutes=round((time.time() - t0) / 60, 1), cleared=k("cleared"), no_death=k("cleared_no_death"),
             deaths=k("deaths"), clear=k("slice_clear"))


if __name__ == "__main__":
    main()
