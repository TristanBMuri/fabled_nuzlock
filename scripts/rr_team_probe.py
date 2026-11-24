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
(player 0, enemy 0 from the dump); the final six hold the items the care packages and the
overworld have handed out by that boss (data/rr/items.json, one per mon; 2026-10-14, C1:
Brock 74 -> 83%, Misty 41 -> 53% on the same sixes; `--no-items` for the old regime). `--legal` applies the player's
hardcore legality (data/rr/hardcore_rules.json, rr_probe.hardcore: the 82 banned moves, Toxic /
Leech Seed only on the listed species, the ability swaps) in stages A-C; without it the
player has every move (the 2026-10-10 afternoon run also gave the enemy
252/252 EVs and the player items / trained EVs — see NOTEBOOK.md). Full enemy
information; enemy = the RR AI port. `--arena` puts the gym's permanent field effect
(tracker battleEffect: weather / terrain / Tailwind / Trick Room) into the sim and
makes p2's own weather and terrain permanent (hardcore). Not modelled: doubles,
the custom arena rules. `--normal` swaps in normal mode's boss sets and caps (the easier
game; run it without --legal to compare the settings).

`--boxes N` is the milestone form: N boxes per boss drawn by rarity (one encounter per
open location, a random starter, the gifts), the chooser (stage A ranking + greedy build)
picks six from each box, the team plays under search; per box the killers and the pool
species outside the box that beat them are reported (what the box was missing, where).
The unit of work is seeded on its own and `--workers` spreads it: a box (--boxes), else a
boss's stages A + B, then every (team, battle), then the threat duels (one bridge per worker
per boss). 2026-10-12: 165 -> 22 min (boxes), 151 -> 25 min (pool) on 20 workers, identical.

  uv run python scripts/rr_team_probe.py --bosses "MISTY,LT. SURGE,ERIKA" --workers 3 --out logs/rr_team_probe.jsonl
  uv run python scripts/rr_team_probe.py --bosses BROCK,MISTY --arena --legal --boxes 30 --workers 20 --out logs/rr_boxes.jsonl
"""

import argparse
import collections
import contextlib
import copy
import json
import multiprocessing as mp
import os
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
from gauntlet import STATS, Mon, assign_items, encounter_moves, evolve, mega_stone, pack_team  # noqa: E402

SHORT, TOP = 24, 6
SCREEN_KEEP = 3  # --chooser searched: the screen's best sixes replayed on the table's seeds
GREEDY_N, SUBSETS = 4, 150


def events_path(out: str) -> Path:
    return Path(out).with_suffix(".events.jsonl")


def append_line(path, obj) -> None:
    """One JSON line, a single O_APPEND write, so the workers' lines never interleave."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(obj) + "\n").encode())
    finally:
        os.close(fd)


def emit(args, kind: str, **data) -> None:
    """One structured event for the dashboard (ui/, UI_TODO.md) next to --out. No --out, no events."""
    if args.out:
        append_line(events_path(args.out), {"t": round(time.time(), 1), "kind": kind, **data})


def enemy_sets(packed: str) -> list[dict]:
    return [{"species": f[0], "id": to_id_str(f[0]), "item": f[2], "ability": f[3], "moves": f[4].split(","), "level": int(f[10] or 0)}
            for f in (x.split("|") for x in packed.split("]"))]


def cells(bridge, mons: dict, enemy_packed: str, arena: dict | None) -> dict:
    """{ours: {enemy id: pair + safe / race}}: the matchup matrix as data (ai_util.matchup)."""
    from ai_util.matchup import race, rows_for, safe
    rows = rows_for(bridge, rp.FMT, mons, enemy_packed, arena)
    return {s: {e: {**p, "hit": round(p["hit"], 3), "take": round(p["take"], 3), "safe": safe(p), "race": race(p)} for e, p in r.items()}
            for s, r in rows.items()}


def progress(st: dict) -> float:
    """How far a fight got: enemy mons fainted + the fraction taken off the rest, over
    the enemy team; a win adds the survivors' share (1.0 = clean sweep, 2.0 = deathless)."""
    p2 = st["p2"]
    done = sum(1.0 if m["fainted"] else 1 - m["hp"] / max(m["maxhp"], 1) for m in p2) / len(p2)
    if st["winner"] == "p1":
        return 1.0 + sum(not m["fainted"] for m in st["p1"]) / len(st["p1"])
    return done


def build_mon(species: str, cap: int, known: dict, progress: int, boss: dict | None = None, seed=0,
              boss_types: set = frozenset(), arena: dict = {}) -> Mon | None:
    """Evolved at the cap (stones from the Celadon store on, gauntlet.STONE_GATE), best4 moves over the TMs and tutors in hand
    by trainer `progress` (data/rr/tms.json, minus gauntlet.BANNED_MOVES), the nature the
    planner picks for this boss (rr_probe.choose_nature; `boss` = rr_probe.boss_profile), the
    ability the run has by then (rr_probe.run_ability: rolled at catch, then capsules), its mega
    stone if one is in hand (so stages A / B see the mega; team_mons re-assigns items), no EVs
    (Minimal Grinding Mode)."""
    ev = evolve(species, cap, None, caught_level=5, trainer_idx=progress)
    moves = encounter_moves(ev, cap, "best4", progress=progress)
    if not moves or ev not in known:
        return None
    m = Mon(ev, cap, moves, ability=rp.run_ability(species, ev, seed, progress, known, boss_types, arena), item=mega_stone(ev, progress), caught_level=5,
            nature=rp.choose_nature(ev, moves, known, boss, cap))
    m.evs = dict.fromkeys(STATS, 0)
    return m


def show(m: Mon) -> str:
    return f"{m.species}({m.nature}/{m.ability}{'/' + m.item if m.item else ''})"


def team_mons(ctx, team) -> list[Mon]:
    """The six as played: copies of the cached mons, holding the items in hand by this boss under
    --items (gauntlet.assign_items: one per mon, greedy by item_value, deterministic in the team)."""
    mons = [copy.copy(mon_of(ctx, s)) for s in team]
    if ctx.args.items:
        assign_items(mons, ctx.progress)
    return mons


def play(bridge, brain, enemy, party: list[Mon], enemy_packed: str, n: int, search=None, arena=None, seeds=None):
    """n battles; returns (win rate, deaths per mon, mean progress, (our deaths by enemy mon,
    enemy mons KO'd, share lost at the 100-turn cap), deathless-win rate). seeds: one engine seed per battle index, the same
    for every team and variant (paired comparisons); it also seeds the RR AI's tie-breaks
    (it uses the global `random`; found 2026-10-11 when two "paired" runs ranked stage A
    differently) and the searcher's rollouts."""
    return fold([one_battle(bridge, brain, enemy, party, enemy_packed, i, search, arena, seeds) for i in range(n)])


def one_battle(bridge, brain, enemy, party: list[Mon], enemy_packed: str, i: int, search=None, arena=None, seeds=None,
               boss: str = "") -> tuple:
    """Battle i of play(): (won, deathless win, fainted fraction, progress, deaths by enemy mon,
    enemy mons KO'd, lost at the turn cap). Seeded on its own, so it is the unit of work the pool spreads."""
    from ai_util.sim import SimBattle
    sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena,
                   seed=seeds[i] if seeds else None)
    if seeds:  # pair everything: the engine (above), the RR AI's tie-breaks (global random), the searcher's rollouts
        random.seed(sum(seeds[i]))
        if search is not None:
            import numpy as np
            search.rng = np.random.default_rng(seeds[i])
    log = getattr(search, "log", None)
    if log:  # --states: the searcher's decision lines carry this tag; the battle line closes them
        search.tag = {"boss": boss, "battle": i, "battle_seed": seeds[i] if seeds else None, "team": [m.species for m in party]}
    sb.play({"p1": search or brain, "p2": enemy}, 100)
    st = sb.state()
    won = st["winner"] == "p1"
    d, k, _ = attribution(sb.history["p1"])
    sb.close()
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({"kind": "battle", **search.tag, "won": won, "timed_out": sb.timed_out, "turns": sb.turn,
                                "chosen": sb.chosen, "fainted": [m["fainted"] for m in st["p1"]]}) + "\n")
    return (won, won and not any(m["fainted"] for m in st["p1"]), sum(m["fainted"] for m in st["p1"]) / len(st["p1"]),
            progress(st), d, k, sb.timed_out)


def fold(battles: list[tuple]) -> tuple:
    """play()'s summary of one_battle() results, in battle order."""
    n, wins, deaths, prog, clean, capped = len(battles), 0, 0.0, 0.0, 0, 0
    deaths_by, kos_on = collections.Counter(), collections.Counter()
    for won, cl, dfrac, pr, d, k, to in battles:
        capped += to
        wins += won
        clean += cl
        deaths += dfrac
        prog += pr
        deaths_by += d
        kos_on += k
    return wins / n, deaths / n, prog / n, (deaths_by, kos_on, capped / n), clean / n


def setup(args, name: str, announce: bool = True):
    """Everything a boss run shares: the pool, cap, arena, the boss's sets and attack types,
    the bridge, both brains, the searcher, and the paired battle seeds. announce=False: a
    worker's copy (no boss_start event, no banner)."""
    from types import SimpleNamespace
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBridge, player_brain
    by, known = rp.pools()
    if name not in by:
        raise KeyError(f"{name}: no team data (not a tracker boss block nor a dump id)")
    block, cap, pool, progress = by[name]
    if args.normal:  # the easier game: normal mode's boss set and cap; the pool, items and TMs stay the hardcore run's
        block, enemy_packed, cap, enemy_species = rp.normal_boss(name)
    arena, unmodelled = rp.arena(block.get("battleEffect", "")) if args.arena else ({}, [block.get("battleEffect", "")])
    if announce:
        print(f"{name}: trainer #{progress}, cap {cap}, pool {len(pool)} species; arena {arena}; not modelled: {unmodelled}", flush=True)
    if args.legal:
        rp.hardcore()
    if args.blind_field or args.matchup:
        import ai_util.calc_bot as cb
        cb.FIELD_AWARE = not args.blind_field
        cb.MATCHUP = args.matchup
    srng = random.Random(f"{name}-seeds-{args.seed}")
    seeds = [[srng.randrange(1, 2 ** 16) for _ in range(4)] for _ in range(max(args.n_solo, args.n_team, args.n_search) + getattr(args, "n_screen", 0))]
    if not args.normal:
        enemy_packed, dump_cap, enemy_species = rp.boss_team(int(name[1:]) if name.startswith("#") else name)
        cap = max(cap, dump_cap)
    md = rp.load_game("rr")["moves"]
    boss_types = {md[mv]["type"] for mon in enemy_packed.split("]") for mv in mon.split("|")[4].split(",") if mv in md}
    py_enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    enemy = py_enemy
    if getattr(args, "enemy", "python") == "rust":  # phase 1's oracle: the Rust port plays the enemy
        from rl.engine_search import RustEnemy
        enemy = RustEnemy(py_enemy)
    calc = calc_brain()
    width = tuple(int(x) for x in args.search.split(","))
    if announce:
        emit(args, "boss_start", boss=name, trainer=progress, cap=cap, pool=len(pool), arena=arena, unmodelled=unmodelled,
             enemy=enemy_sets(enemy_packed), boss_types=sorted(boss_types))
    return SimpleNamespace(name=name, args=args, known=known, cap=cap, pool=pool, progress=progress, arena=arena,
                           unmodelled=unmodelled, enemy_packed=enemy_packed, enemy_species=enemy_species,
                           boss_types=boss_types, boss=rp.boss_profile(enemy_packed, known, arena), bridge=SimBridge(showdown=str(rp.FORK)), enemy=enemy, calc=calc,
                           searcher=searcher(args, Searcher(py_enemy, k=width[0], rollouts=width[1], rollout=calc, seed=args.seed, death_weight=args.death_weight)),
                           seeds=seeds, rng=random.Random(f"{name}-team-{args.seed}"), team_score={}, mons={}, t0=time.time())


def searcher(args, showdown):
    """--engine: the Rust engine's search with the Showdown search as its fallback."""
    if not getattr(args, "engine", 0):
        return showdown
    from rl.engine_search import EngineSearcher
    es = EngineSearcher(showdown, iterations=args.engine, opponent=getattr(args, "engine_opp", "minimax"),
                        p_model=getattr(args, "engine_p", 1.0))
    if getattr(args, "states", False) and args.out:
        es.log = str(Path(args.out).with_suffix(".states.jsonl"))
    return es


def mon_of(ctx, sp: str) -> Mon | None:
    if sp not in ctx.mons:
        ctx.mons[sp] = build_mon(sp, ctx.cap, ctx.known, ctx.progress, ctx.boss, ctx.args.seed, ctx.boss_types, ctx.arena)
    return ctx.mons[sp]


def solo_stage(ctx, species: list[str]) -> tuple[list, dict]:
    """(A) every species solo against the boss under the calc bot: (sorted (progress, sp),
    {sp: (our deaths by enemy mon, enemy mons KO'd)})."""
    solo, solo_vs = [], {}
    for sp in species:
        m = mon_of(ctx, sp)
        if m is None:
            continue
        _, _, pr, solo_vs[sp], _ = play(ctx.bridge, ctx.calc, ctx.enemy, [m], ctx.enemy_packed, ctx.args.n_solo, arena=ctx.arena, seeds=ctx.seeds)
        solo.append((pr, sp))
    solo.sort(reverse=True)
    return solo, solo_vs


def team_value(ctx, team: tuple) -> float:
    if team not in ctx.team_score:
        ctx.team_score[team] = play(ctx.bridge, ctx.calc, ctx.enemy, [mon_of(ctx, s) for s in team], ctx.enemy_packed, ctx.args.n_team,
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


def cover_team(ctx, species: list[str]) -> tuple:
    """(B') the six by matchup cover (ai_util.matchup.cover): rows read off dummy battles.
    --chooser duel: every cell is judged by playing the one-on-one instead (duel(), n_duel
    paired seeds; ~4 ms each) — the paper's tiers miss the enemy AI's free first turn, status
    and items (TODO.md B3); the duel sees all of them."""
    from poke_env.data import to_id_str
    from ai_util.matchup import cover, rows_for
    mons = {s: m for s in species if (m := mon_of(ctx, s)) is not None}
    rows = rows_for(ctx.bridge, rp.FMT, mons, ctx.enemy_packed, ctx.arena)
    names = [x.split("|")[0] for x in ctx.enemy_packed.split("]")]
    if ctx.args.chooser == "duel":
        for s in mons:
            for name in names:
                w, h = duel(ctx, s, name, ctx.args.n_duel)
                rows.setdefault(s, {}).setdefault(to_id_str(name), {}).update(win=w, hp=h)
    trace = []
    team = tuple(sorted(cover(rows, [to_id_str(n) for n in names], 6, trace)))
    emit(ctx.args, "draft", boss=ctx.name, picks=trace)  # the draft view: why each pick, and the runner-up
    return team


def playable(ctx, species: list[str]) -> list[str]:
    """The run rules every stage applies: buildable, one per evolved species (dupes clause), no
    Truant, no Ditto / Smeargle (poke-env: Transform / Sketch)."""
    out, seen = [], set()
    for sp in species:
        m = mon_of(ctx, sp)
        if m is None or m.ability == "Truant" or m.species in seen or sp in ("ditto", "smeargle"):
            continue
        seen.add(m.species)
        out.append(sp)
    return out


def lead_first(ctx, team: tuple) -> tuple:
    """The six with the best answer to the boss's lead first (the tuple's order is the team's
    order; sorted sixes led alphabetically until 2026-11-24). Best = matchup tiers (safe, race,
    edge) of the cell against the boss's first mon."""
    from poke_env.data import to_id_str
    from ai_util.matchup import edge, race, rows_for, safe
    rows = rows_for(ctx.bridge, rp.FMT, {s: mon_of(ctx, s) for s in team}, ctx.enemy_packed, ctx.arena)
    e = to_id_str(ctx.enemy_packed.split("]")[0].split("|")[0])
    score = lambda c: (safe(rows[c][e]), race(rows[c][e]), edge(rows[c][e])) if e in rows.get(c, {}) else (False, False, -9.0)
    lead = max(sorted(team), key=score)
    return (lead,) + tuple(sorted(c for c in team if c != lead))


def choose(ctx, species: list[str]) -> tuple:
    """--chooser's six from a box, lead first. searched: the candidate sixes each play
    n_screen battles on seeds the table does not use; the best (win, -deaths) is the six."""
    species = playable(ctx, species)
    if ctx.args.chooser in ("cover", "duel"):
        return lead_first(ctx, cover_team(ctx, species))
    if ctx.args.chooser == "greedy":
        return lead_first(ctx, greedy(ctx, species))
    if ctx.args.chooser == "searched":
        n_s, n_c = ctx.args.n_search, ctx.args.n_screen
        scored = []
        for t in searched_cands(ctx, species, ctx.args.cands):
            w, d, *_ = fold([battle_job(ctx.args, ctx.name, t, n_s + i) for i in range(n_c)])
            scored.append((w, -d, t))
        emit(ctx.args, "screen", boss=ctx.name, scored=[(list(t), w, -d) for w, d, t in sorted(scored, reverse=True)])
        return max(scored)[2]
    raise ValueError(f"unknown chooser {ctx.args.chooser}")


def searched_cands(ctx, species: list[str], n: int, k: int = 5) -> list[tuple]:
    """(B'') --chooser searched: candidate sixes for the screen. The shortlist is the k best
    answers (matchup tiers: safe, race, edge) to *each* boss mon, so no enemy is ignored (stage
    A's solo score only measured the boss's lead). Run rules: one per evolved species (dupes
    clause), no Truant, at most one mega stone. The cover six over the shortlist, then random
    sixes that draw one answer per boss mon. The lead is the best answer to the boss's lead
    (the tuple's order is the team's order)."""
    from poke_env.data import to_id_str
    from ai_util.matchup import cover, edge, race, rows_for, safe
    mons = {sp: mon_of(ctx, sp) for sp in playable(ctx, species)}
    rows = rows_for(ctx.bridge, rp.FMT, mons, ctx.enemy_packed, ctx.arena)
    enemies = [to_id_str(x.split("|")[0]) for x in ctx.enemy_packed.split("]")]
    score = lambda c, e: (safe(rows[c][e]), race(rows[c][e]), edge(rows[c][e])) if e in rows.get(c, {}) else (False, False, -9.0)
    answers = {e: sorted(rows, key=lambda c: score(c, e), reverse=True)[:k] for e in enemies}
    short = list(dict.fromkeys(c for e in enemies for c in answers[e]))
    total = lambda c: sum(score(c, e)[2] for e in enemies)

    def fits(team, c):
        stone = lambda x: mega_stone(mons[x].species, ctx.progress)  # by the stone, not "holds an item"
        return c not in team and not (stone(c) and any(stone(t) for t in team))  # one mega stone

    def finish(team):
        for c in sorted(short, key=total, reverse=True):
            if len(team) >= 6:
                break
            if fits(team, c):
                team.append(c)
        lead = max(team, key=lambda c: score(c, enemies[0]))
        return (lead,) + tuple(sorted(c for c in team if c != lead))

    first = []
    for c in cover({c: rows[c] for c in short}, enemies, 6):
        if fits(first, c):
            first.append(c)
    cands = [finish(first)]
    for _ in range(50 * n):
        if len(cands) >= n:
            break
        team = []
        for e in ctx.rng.sample(enemies, len(enemies)):
            c = ctx.rng.choice(answers[e])
            if fits(team, c):
                team.append(c)
        t = finish(team)
        if t not in cands:
            cands.append(t)
    return cands


def searched(ctx, team: tuple):
    """(C) the team under calc + search, paired seeds: (win, deaths/mon, progress, attribution,
    deathless-win rate)."""
    return searched_many(ctx, [team])[team]


def searched_many(ctx, teams: list[tuple]) -> dict:
    """searched() for several teams at once: every (team, battle) is a pool job when the main
    process holds the pool (the stage-C fan-out), else they play here in order."""
    n = ctx.args.n_search
    battles = fan(battle_job, [(ctx.args, ctx.name, t, i) for t in teams for i in range(n)])
    return {t: fold(battles[j * n:(j + 1) * n]) for j, t in enumerate(teams)}


# The worker pool (main process only; a worker's is None, so fan() runs its jobs in place)
# and the workers' boss contexts: a worker builds a boss's (bridge, brains, sets) once.
_POOL = None
_CTX: dict = {}
_MAIN = False  # True in the main process (set in main; a spawned worker starts fresh)


def fan(fn, jobs: list[tuple]) -> list:
    return _POOL.starmap(fn, jobs) if _POOL is not None and len(jobs) > 1 else [fn(*j) for j in jobs]


def worker_ctx(args, name: str):
    """A worker's context for boss `name`: one at a time (a bridge is ~200 MB of Node; the jobs
    are ordered by boss, so a worker switches rarely). The main process keeps its own per boss."""
    ctx = _CTX.get(name)
    if ctx is None:
        for other in [k for k in _CTX if k != name] if not _MAIN else []:  # main keeps every boss's
            _CTX.pop(other).bridge.close()
        ctx = _CTX[name] = setup(args, name, announce=False)
    return ctx


def battle_job(args, name: str, team: tuple, i: int) -> tuple:
    """Searched battle i of `team` against boss `name` (a pool job). team: species ids (the
    builder's sets), or set dicts (sets_of / scripts/rr_search.py: explicit moves, item, nature)."""
    ctx = worker_ctx(args, name)
    party = team_mons(ctx, team) if isinstance(team[0], str) else [set_mon(s) for s in team]
    return one_battle(ctx.bridge, ctx.calc, ctx.enemy, party, ctx.enemy_packed, i,
                      search=ctx.searcher, arena=ctx.arena, seeds=ctx.seeds, boss=name)


def duel(ctx, sp: str, enemy_mon: str, n: int) -> tuple[float, float]:
    """One of ours against one enemy mon alone (its real set), the calc bot playing, paired
    seeds: (win rate, mean HP fraction left on a win)."""
    from ai_util.sim import SimBattle
    packed = next((m for m in ctx.enemy_packed.split("]") if m.split("|")[0] == enemy_mon), None)
    mon = mon_of(ctx, sp)
    if packed is None or mon is None:
        return 0.0, 0.0
    wins, hp = 0, 0.0
    for i in range(n):
        sb = SimBattle(ctx.bridge, rp.FMT, pack_team([mon]), packed, full_info=True, arena=ctx.arena, seed=ctx.seeds[i % len(ctx.seeds)])
        random.seed(sum(ctx.seeds[i % len(ctx.seeds)]))
        sb.play({"p1": ctx.calc, "p2": ctx.enemy}, 100)
        st = sb.state()
        if st["winner"] == "p1":
            wins += 1
            hp += st["p1"][0]["hp"] / max(st["p1"][0]["maxhp"], 1)
        sb.close()
    return wins / n, hp / max(wins, 1)


def threats_of(ctx, attr, candidates, exclude=()) -> list[dict]:
    """The two enemy mons that killed most, each with up to three `candidates` (not in
    `exclude`) that beat it one on one under the arena — duel win rate, HP left — and where
    to catch them. (Solo records against the whole team found no counters to mid-team
    threats such as Surge's Electrode-Hisui; the duel asks the right question.)"""
    out = []
    for enemy_mon, k in attr[0].most_common(2):
        scored = [(duel(ctx, sp, enemy_mon, ctx.args.n_duel), sp) for sp in candidates if sp not in exclude]
        ranked = sorted(((w, h, sp) for (w, h), sp in scored if w > 0.5), reverse=True)[:3]
        out.append({"enemy": enemy_mon, "deaths": k,
                    "counters": [(sp, w, h, rp.where(sp, ctx.progress)) for w, h, sp in ranked]})
    return out


def say_threats(ctx, threats: list[dict], tag: str = "") -> None:
    for th in threats:
        print(f"{ctx.name:<10} {tag}threat {th['enemy']} ({th['deaths']} of our deaths): counters "
              + (", ".join(f"{sp} [wins the duel {w:.0%}, {h:.0%} HP left; {wh}]" for sp, w, h, wh in th["counters"]) or "none"), flush=True)


def sets_of(ctx, team) -> list[dict]:
    return [{"species": m.species, "level": m.level, "nature": m.nature, "ability": m.ability, "item": m.item, "moves": m.moves}
            for m in team_mons(ctx, team)]


def set_mon(s: dict) -> Mon:
    """The Mon of a set dict (sets_of's form), no EVs (Minimal Grinding Mode)."""
    m = Mon(s["species"], s["level"], list(s["moves"]), s["ability"], s["item"] or None, caught_level=5, nature=s["nature"])
    m.evs = dict.fromkeys(STATS, 0)
    return m


def matrix(ctx, team) -> dict:
    return cells(ctx.bridge, dict(zip(team, team_mons(ctx, team))), ctx.enemy_packed, ctx.arena)


# The planned-pool probe: the best six in everything catchable. Three fan-outs over all the
# bosses at once (2026-10-12; bosses ran one after another before, each stage C serial):
# prelude_job (stages A + B, one job per boss), battle_job (every team x battle index),
# threats_job (the duels, one job per boss); the main process folds, prints and emits.


def prelude_job(args, name: str) -> dict | None:
    """Stages A and B for one boss: the starter, the solo ranking, the shortlist, the greedy /
    fast / cover teams and the TOP list to play under search. None on failure."""
    try:
        ctx = worker_ctx(args, name)
        starter = ctx.rng.choice(rp.STARTERS)  # one per run, at random across every generation
        pool = playable(ctx, sorted(set(ctx.pool) | {starter}))
        print(f"{name}: starter {starter}; boss attacks with {sorted(ctx.boss_types)}", flush=True)
        solo, _ = solo_stage(ctx, pool)
        short = [sp for _, sp in solo[:SHORT]]
        print(f"{name}: A solo scores for {len(solo)} species in {time.time() - ctx.t0:.0f}s; top: "
              + ", ".join(f"{sp} {pr:.2f}" for pr, sp in solo[:8]), flush=True)
        emit(args, "stage_a", boss=name, starter=starter, scored=len(solo), top=[(sp, round(pr, 3)) for pr, sp in solo[:SHORT]])
        team = greedy(ctx, short)
        for _ in range(SUBSETS):
            team_value(ctx, tuple(sorted(ctx.rng.sample(short, 6))))
        ranked = sorted((t for t in ctx.team_score if len(t) == 6), key=ctx.team_score.get, reverse=True)  # full teams only
        print(f"{name}: B {len(ctx.team_score)} teams scored in {time.time() - ctx.t0:.0f}s; greedy {team} "
              f"({ctx.team_score[team]:.2f}); best fast {ranked[0]} ({ctx.team_score[ranked[0]]:.2f})", flush=True)
        top = list(dict.fromkeys([team] + ranked))[:TOP]
        cteam = None
        if args.chooser in ("cover", "duel"):
            cteam = cover_team(ctx, short + [sp for sp in pool if sp not in short])
            print(f"{name}: {args.chooser} team {cteam}", flush=True)
            top = list(dict.fromkeys([cteam] + top))[:TOP]
        emit(args, "stage_b", boss=name, scored=len(ctx.team_score), greedy=[list(team), round(ctx.team_score[team], 3)],
             fast=[list(ranked[0]), round(ctx.team_score[ranked[0]], 3)], cover=list(cteam) if cteam else None)
        cands = searched_cands(ctx, pool, args.cands) if args.chooser == "searched" else []
        if cands:
            print(f"{name}: searched chooser: {len(cands)} candidate sixes to screen, e.g. {list(cands[0])}", flush=True)
        labels = {team: "greedy", ranked[0]: "fast"}
        if cteam is not None:
            labels[cteam] = args.chooser
        return {"starter": starter, "pool": pool, "short": short, "top": top, "labels": labels, "cands": cands}
    except Exception as e:
        import traceback
        print(f"{name} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
        return None


def threats_job(args, name: str, deaths_by: dict, candidates: list[str], exclude: tuple = ()) -> list[dict]:
    """threats_of() for one boss's best team (the duels; a pool job)."""
    return threats_of(worker_ctx(args, name), (collections.Counter(deaths_by), None), candidates, exclude)


def pool_row(args, ctx, pre: dict, played: dict, threats: list[dict]) -> dict:
    """One boss's stage-C report and row from the fanned-out results (the main process)."""
    name, top, labels, short, pool = ctx.name, pre["top"], pre["labels"], pre["short"], pre["pool"]
    results, attr, clean = [], {}, {}
    for t in top:
        w, d, pr, attr[t], clean[t] = played[t]
        results.append((w, -d, t, d))
        print(f"{name}: C {labels.get(t, 'top'):<6} {list(t)}: win {w:.0%} deaths/mon {d:.2f} deathless {clean[t]:.0%}", flush=True)
        emit(args, "team", boss=name, label=labels.get(t, "top"), team=list(t), sets=sets_of(ctx, t), win=w, deaths_per_mon=d,
             deathless=clean[t], deaths_by=dict(attr[t][0]), kos_on=dict(attr[t][1]), matrix=matrix(ctx, t))
    results.sort(reverse=True)
    w, _, t, d = results[0]
    print(f"{name:<10} legal={args.legal!s:<5} best team under search: win {w:5.0%} deaths/mon {d:.2f} deathless {clean[t]:.0%} "
          f"{list(t)}  (others: {', '.join(f'{w_:.0%}' for w_, _, _, _ in results[1:])})  ({time.time() - ctx.t0:.0f}s)", flush=True)
    print(f"{name:<10} sets: " + ", ".join(f"{show(m)} {m.moves}" for m in team_mons(ctx, t)), flush=True)
    say_threats(ctx, threats)
    emit(args, "threats", boss=name, team=list(t), threats=threats)
    repaired = None
    extra = [sp for th in threats for sp, *_ in th["counters"] if sp not in short]  # counters the shortlist lacked
    if args.repair and extra:
        # unforced: the counters join the candidates and must earn a slot on team value (forcing
        # the top counter in made Misty worse, 63% -> 27%, 2026-10-11)
        for sp in extra:
            mon_of(ctx, sp)
        rteam = greedy(ctx, short + extra)
        taken = [sp for sp in extra if sp in rteam]
        rw, rd, _, _, rc = searched(ctx, rteam) if rteam != t else (w, d, None, None, clean[t])
        repaired = {"team": list(rteam), "win": rw, "deaths_per_mon": rd, "deathless": rc, "counters_offered": extra, "counters_taken": taken}
        emit(args, "repaired", boss=name, **repaired, sets=sets_of(ctx, rteam))
        print(f"{name:<10} repaired: counters offered {extra}, taken {taken}: win {rw:5.0%} deaths/mon {rd:.2f} deathless {rc:.0%} {list(rteam)}  (best was {w:.0%} / {d:.2f})  ({time.time() - ctx.t0:.0f}s)", flush=True)
    row = {"boss": name, "mode": "pool", "cap": ctx.cap, "legal": args.legal, "normal": args.normal, "field_aware": not args.blind_field, "chooser": args.chooser, "matchup": args.matchup, "death_weight": args.death_weight, "arena": ctx.arena, "items": args.items,
           "unmodelled": ctx.unmodelled, "best_team": list(t), "win": w, "deaths_per_mon": d, "deathless": clean[t], "n": args.n_search,
           "all": [(list(t_), w_, d_, clean[t_]) for w_, _, t_, d_ in results], "threats": threats, "repaired": repaired,
           "starter": pre["starter"], "sets": sets_of(ctx, t), "enemy": ctx.enemy_species, "pool": len(pool), "progress": ctx.progress, "shortlist": short}
    emit(args, "boss_done", boss=name, row=row)
    ctx.bridge.close()
    return row


def best_of(played: dict, top: list) -> tuple:
    """The best team by (win, -deaths, team) — the order pool_row reports."""
    return max(top, key=lambda t: (played[t][0], -played[t][1], t))


def run_pool(args, names: list[str]) -> list[dict]:
    ctxs = {}
    for n in names:
        try:
            ctxs[n] = _CTX[n] = setup(args, n)  # the banner, boss_start; the fold's context (and the serial path's)
        except KeyError as e:
            print(f"{n} FAILED: {e}", flush=True)
            emit(args, "boss_failed", boss=n, error=str(e))
    names = [n for n in names if n in ctxs]
    pres = dict(zip(names, fan(prelude_job, [(args, n) for n in names])))
    names = [n for n in names if pres[n] is not None]
    for n in set(ctxs) - set(names):
        emit(args, "boss_failed", boss=n, error="prelude failed")
    k, n_s = 0, args.n_search
    if args.chooser == "searched":  # the screen: every candidate on seeds the table does not use
        n_c = args.n_screen
        jobs = [(args, n, t, n_s + i) for n in names for t in pres[n]["cands"] for i in range(n_c)]
        res = fan(battle_job, jobs)
        for n in names:
            scored = []
            for t in pres[n]["cands"]:
                w, d, *_ = fold(res[k:k + n_c])
                k += n_c
                scored.append((w, -d, t))
            scored.sort(reverse=True)
            best = [t for *_, t in scored[:SCREEN_KEEP]]
            print(f"{n}: screen ({n_c} battles each): " + ", ".join(f"{w:.0%}/{-d:.2f} {list(t)}" for w, d, t in scored[:5]), flush=True)
            emit(args, "screen", boss=n, scored=[(list(t), w, -d) for w, d, t in scored])
            pres[n]["labels"].update({t: "screen" for t in best[1:]} | {best[0]: "searched"})
            pres[n]["top"] = list(dict.fromkeys(best + pres[n]["top"]))
        k = 0
    battles = fan(battle_job, [(args, n, t, i) for n in names for t in pres[n]["top"] for i in range(n_s)])
    played = {}
    for n in names:
        for t in pres[n]["top"]:
            played[n, t] = fold(battles[k:k + n_s])
            k += n_s
    best = {n: best_of({t: played[n, t] for t in pres[n]["top"]}, pres[n]["top"]) for n in names}
    threats = fan(threats_job, [(args, n, dict(played[n, best[n]][3][0]), pres[n]["pool"]) for n in names])
    return [pool_row(args, ctxs[n], pres[n], {t: played[n, t] for t in pres[n]["top"]}, th) for n, th in zip(names, threats)]


# The sampled-box probe (the milestone): N boxes per boss drawn by rarity (rr_probe.sample_box),
# the chooser picks six from each, the team plays the boss under search; per box the killers
# and the pool species *outside the box* that beat them — what the box was missing and where
# it is caught. Every box is seeded on its own (sample, engine, AI tie-breaks, rollouts), so
# the boxes are pool jobs spread over the workers (2026-10-12: 30 boxes x 2 bosses ran on 2
# cores, 165 min); a worker builds a boss's context once and keeps it.


def box_job(args, name: str, i: int) -> dict | None:
    """One box: the chooser's six, the searched play, the threats. None on failure."""
    try:
        ctx = worker_ctx(args, name)
        box = rp.sample_box(random.Random(f"{name}-box-{args.seed}-{i}"), ctx.progress, ctx.known)
        species = [sp for sp, _ in box if mon_of(ctx, sp) is not None]
        team = choose(ctx, species)
        w, d, pr, attr, cl = searched(ctx, team)
        threats = threats_of(ctx, attr, ctx.pool, exclude=set(species))
        out = {"box": box, "team": list(team), "sets": sets_of(ctx, team), "win": w, "deaths_per_mon": d, "deathless": cl,
               "capped": attr[2], "threats": threats}
        if hasattr(ctx.searcher, "stats"):  # --engine: decisions on the engine / on Showdown, and why (cumulative per worker)
            out["engine"] = dict(ctx.searcher.stats)
        if hasattr(ctx.enemy, "stats"):  # --enemy rust: decisions by the Rust port / the Python AI (cumulative per worker)
            out["enemy_brain"] = dict(ctx.enemy.stats)
        emit(args, "box", boss=name, i=i, of=args.boxes, **out, deaths_by=dict(attr[0]), matrix=matrix(ctx, team))
        print(f"{name:<10} box {i + 1:>2}/{args.boxes}: {len(box)} mons, team {list(team)} -> win {w:5.0%} deaths/mon {d:.2f} deathless {cl:.0%}; "
              + "; ".join(f"{th['enemy']} killed {th['deaths']}, missing " + (", ".join(f"{sp} ({wh})" for sp, _, _, wh in th["counters"][:1]) or "no counter in the pool") for th in threats)
              + f"  ({time.time() - ctx.t0:.0f}s)", flush=True)
        return out
    except Exception as e:
        import traceback
        print(f"{name} box {i + 1} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
        return None


def boxes_start(args, name: str) -> dict | None:
    """Per boss, before the box jobs: the banner, the boss_start event and the solo stage (A, a
    few seconds; the dashboard's ranking). The row's header fields, or None without team data."""
    try:
        ctx = setup(args, name)
    except KeyError as e:
        print(f"{name} FAILED: {e}", flush=True)
        emit(args, "boss_failed", boss=name, error=str(e))
        return None
    solo, _ = solo_stage(ctx, playable(ctx, ctx.pool + [sp for sp in rp.STARTERS if sp not in ctx.pool]))
    print(f"{name}: A solo scores for {len(solo)} species (pool + starters) in {time.time() - ctx.t0:.0f}s", flush=True)
    emit(args, "stage_a", boss=name, scored=len(solo), top=[(sp, round(pr, 3)) for pr, sp in solo[:SHORT]])
    ctx.bridge.close()
    return {"boss": name, "mode": "boxes", "cap": ctx.cap, "legal": args.legal, "normal": args.normal, "field_aware": not args.blind_field, "chooser": args.chooser,
            "matchup": args.matchup, "death_weight": args.death_weight, "arena": ctx.arena, "n": args.n_search, "items": args.items,
            "enemy": ctx.enemy_species, "pool": len(ctx.pool), "progress": ctx.progress}


def boxes_row(args, head: dict, boxes: list[dict | None]) -> dict:
    """The boss's row from its boxes' results (failed boxes dropped and counted)."""
    failed = sum(b is None for b in boxes)
    boxes = [b for b in boxes if b is not None]
    missing = collections.Counter()
    for b in boxes:
        for th in b["threats"]:
            for sp, _, _, wh in th["counters"][:1]:
                missing[(th["enemy"], sp, wh)] += 1
    wins = [b["win"] for b in boxes]
    row = {**head, "boxes": boxes, "failed": failed, "mean_win": sum(wins) / max(len(wins), 1), "p_clear": sum(w_ >= 0.5 for w_ in wins) / max(len(wins), 1),
           "mean_deaths": sum(b["deaths_per_mon"] for b in boxes) / max(len(boxes), 1), "mean_deathless": sum(b["deathless"] for b in boxes) / max(len(boxes), 1),
           "missing": [(k, v) for k, v in missing.most_common(6)]}
    print(f"{head['boss']:<10} {len(boxes)} boxes{f' ({failed} failed)' if failed else ''}: mean win {row['mean_win']:.0%}, boxes with a >=50% team {row['p_clear']:.0%}, deaths/mon {row['mean_deaths']:.2f}, deathless wins {row['mean_deathless']:.0%}; "
          f"most missed: " + ", ".join(f"{sp} vs {en} ({wh}) x{v}" for (en, sp, wh), v in row["missing"][:3]), flush=True)
    emit(args, "boss_done", boss=head["boss"], row=row)
    return row


def run_boxes(args, names: list[str]) -> list[dict]:
    heads = [h for h in (boxes_start(args, n) for n in names) if h]
    results = fan(box_job, [(args, h["boss"], i) for h in heads for i in range(args.boxes)])
    return [boxes_row(args, h, results[k * args.boxes:(k + 1) * args.boxes]) for k, h in enumerate(heads)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bosses", default="BROCK,MISTY,LT. SURGE,ERIKA,KOGA,BLAINE")
    ap.add_argument("--n-solo", type=int, default=3)
    ap.add_argument("--n-team", type=int, default=4)
    ap.add_argument("--n-search", type=int, default=6)
    ap.add_argument("--n-duel", type=int, default=4, help="duels per candidate when ranking counters to a threat (and per cell under --chooser duel)")
    ap.add_argument("--search", default="9,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--legal", action="store_true", help="the player's hardcore legality: banned moves, the Toxic / Leech Seed list, ability swaps (rr_probe.hardcore)")
    ap.add_argument("--normal", action="store_true", help="normal mode's boss sets and caps (tracker 'default', rr_probe.normal_boss): the easier setting, for comparison")
    ap.add_argument("--arena", action="store_true", help="the gym's permanent field effect (tracker battleEffect) in the sim")
    ap.add_argument("--blind-field", action="store_true", help="the calc bot ignores weather / terrain (the pre-2026-10-11 bot)")
    ap.add_argument("--repair", action="store_true", help="offer the counters to the greedy build and play the result too")
    ap.add_argument("--boxes", type=int, default=0, help="sampled-box mode: N boxes per boss drawn by rarity, the chooser picks six from each")
    ap.add_argument("--chooser", choices=("greedy", "cover", "duel", "searched"), default="greedy",
                    help="six by calc-bot progress (greedy), by matchup cover on the paper matrix (cover), by cover with every cell judged by n-duel one-on-ones (duel), or by screening --cands sixes built from each boss mon's best answers with searched battles (searched; pool mode)")
    ap.add_argument("--cands", type=int, default=24, help="--chooser searched: candidate sixes screened per boss")
    ap.add_argument("--n-screen", type=int, default=4, help="--chooser searched: searched battles per candidate in the screen (own seeds)")
    ap.add_argument("--matchup", action="store_true", help="the calc bot (and so the rollouts) switches on the matchup matrix")
    ap.add_argument("--death-weight", type=float, default=0.5, help="search value: win +-1 minus this x the fainted fraction (6 = one death costs a win)")
    ap.add_argument("--no-items", action="store_true", help="no held items (default: the final six hold the items in hand by the boss — care packages + overworld, data/rr/items.json; one per mon, gauntlet.assign_items; C1, 2026-10-14)")
    ap.add_argument("--engine", type=int, default=0, help="stage C's search runs poke-engine MCTS with this many iterations where the position converts "
                    "(rl/engine_search.py), the Showdown search (--search) everywhere else; 0 = Showdown only")
    ap.add_argument("--engine-p", type=float, default=0.8, help="--engine-opp ai: follow the AI model with this probability per enemy decision, else UCB (phase 1 mixture; 0.8 = D4, 2026-11-24)")
    ap.add_argument("--enemy", choices=("python", "rust"), default="python", help="rust: the Rust AI port plays the enemy (phase 1's exact-model oracle)")
    ap.add_argument("--states", action="store_true", help="--engine: log every search decision and battle to <out>.states.jsonl")
    ap.add_argument("--engine-opp", choices=("minimax", "ai"), default="ai",
                    help="--engine's model of the boss: both sides search, or the RR AI in Rust mixed with search "
                         "(--engine-p; the default since D4, 2026-11-24: +11 points, -0.13 deaths/mon on fresh seeds)")
    ap.add_argument("--out")
    args = ap.parse_args()
    args.items = not args.no_items
    names = [b.strip() for b in args.bosses.split(",")]
    t0 = time.time()
    if args.out:
        events_path(args.out).unlink(missing_ok=True)
    emit(args, "start", config=vars(args), bosses=names, mode="boxes" if args.boxes else "pool", pid=os.getpid())
    global _POOL, _MAIN
    _MAIN = True
    # bosses run in the main process one after another; the pool spreads their boxes (--boxes)
    # or their stage-C battles (teams x n_search), the units that are seeded on their own
    with (mp.get_context("spawn").Pool(args.workers) if args.workers > 1 else contextlib.nullcontext()) as _POOL:
        rows = run_boxes(args, names) if args.boxes else run_pool(args, names)
    _POOL = None
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    emit(args, "end", minutes=round((time.time() - t0) / 60, 1))
    mode = "boxes" if args.boxes else "pool"
    print(f"\nRadical Red best-six probe ({mode}) | {'normal' if args.normal else 'hardcore'} bosses | hardcore legality: {args.legal} | arena effects: {args.arena} | field-aware calc bot: {not args.blind_field} | chooser {args.chooser} | matchup play {args.matchup} | death weight {args.death_weight} | seeded | "
          f"search {args.search} x n={args.n_search} | no EVs, {'items in hand' if args.items else 'no items'} | enemy = RR AI, full info | {(time.time() - t0) / 60:.0f} min")
    if args.boxes:
        print(f"  {'boss':<10} {'cap':>3} {'mean win':>8} {'>=50%':>6} {'d/mon':>5} {'clean':>5}  most missed counter (vs)")
        for r in sorted(rows, key=lambda r: r["cap"]):
            (en, sp, wh), v = r["missing"][0] if r["missing"] else ((None, None, None), 0)
            print(f"  {r['boss']:<10} {r['cap']:>3} {r['mean_win']:8.0%} {r['p_clear']:6.0%} {r['mean_deaths']:5.2f} {r['mean_deathless']:5.0%}  "
                  + (f"{sp} vs {en} ({wh}) x{v}" if v else "-"))
    else:
        print(f"  {'boss':<10} {'cap':>3} {'win':>5} {'deaths/mon':>10} {'clean':>5}  best team  | arena")
        for r in sorted(rows, key=lambda r: r["cap"]):
            ar = {k: v for k, v in r["arena"].items() if k != "ai_permanent"}
            print(f"  {r['boss']:<10} {r['cap']:>3} {r['win']:5.0%} {r['deaths_per_mon']:10.2f} {r['deathless']:5.0%}  {', '.join(r['best_team'])}  | {ar or '-'}")


if __name__ == "__main__":
    main()
