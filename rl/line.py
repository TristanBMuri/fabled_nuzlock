"""Battle line: a boss fight as a flowchart of our plan and the chance events that branch it.

Top nuzlocke players write a "line" for a boss: what to click, and what to do if the enemy
crits, misses, or the AI picks its other option. The enemy here is a known script (the RR AI;
its Rust port agrees with the real one 98% of the time on-policy), so given our policy the fight
is a single-agent problem with chance events, a Markov decision process. The line is that
policy written out as a tree (a *policy tree*):

  our node     the action the phase-1 search picks (MCTS vs the AI model, p_model = 0.8)
  chance       every enemy choice the AI model weighs (its ties and 25 / 50% rolls) x every
               outcome the engine lists for the turn (crit, miss, KO roll, secondary effect,
               speed tie), each with its exact probability

Expanding the tree and backing probabilities up is *policy evaluation by tree expansion*: P(win),
P(deathless win), expected deaths and each mon's P(death) come out exactly for the explored part,
instead of from sampled battles (48 battles cannot tell a 2% death risk from 6%). A branch
whose reach probability falls below `eps` is not expanded; its mass is reported as
**unexplored**, so every number is a bound: [explored value, explored + unexplored].
Identical states merge (by the engine's state string), so the tree is a graph.

Known approximation: the engine keeps one (mean) damage value for a hit that does not KO, so a
roll-dependent 2HKO is mispriced; `check()` replays the line on Showdown against the real
(Python) AI and reports how often the fight leaves the tree and the observed vs predicted rates.

    uv run python rl/line.py --events results/rr_p1b_ceiling.events.jsonl --boss ERIKA
    uv run python rl/line.py --events ... --boss ERIKA --check 200 --out logs/line_erika.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from rl import convert  # noqa: E402


def _alive(side) -> list[bool]:
    return [p.hp > 0 for p in side.pokemon if p.id.upper() != "NONE"]


def _name(choice: str) -> str:
    """A search / AI option as generate_instructions parses it (a switch is the bare species)."""
    return "none" if choice == "No Move" else choice.removeprefix("switch ")


def _over(st) -> str | None:
    if not any(_alive(st.side_two)):
        return "win"
    if not any(_alive(st.side_one)):
        return "loss"
    return None


def _label(before, after) -> str:
    """What the turn did, in HP terms: each active's HP before -> after, and who fainted."""
    out = []
    for name, b, a in (("us", before.side_one, after.side_one), ("them", before.side_two, after.side_two)):
        mb, ma = b.pokemon[int(b.active_index)], a.pokemon[int(a.active_index)]
        if int(b.active_index) != int(a.active_index):
            out.append(f"{name}: {mb.id.lower()} -> {ma.id.lower()} in")
        else:
            out.append(f"{name}: {mb.id.lower()} {100 * mb.hp // max(mb.maxhp, 1)}->{100 * ma.hp // max(ma.maxhp, 1)}%")
        out += [f"{p.id.lower()} faints" for p, q in zip(b.pokemon, a.pokemon) if p.hp > 0 and q.hp <= 0]
    return ", ".join(out)


class Line:
    """The policy tree from one state. nodes[i]: state string, our action, the search's top
    options, children [(p, enemy choice, outcome label, child id)], and the backed-up bounds."""

    def __init__(self, eps: float = 0.005, iterations: int = 3000, max_turns: int = 40, p_model: float = 0.8, seed: int = 0):
        self.eps, self.iterations, self.max_turns, self.p_model, self.seed = eps, iterations, max_turns, p_model, seed
        os.environ.setdefault("POKE_ENGINE_DEATH", "60")  # the battler's death weight (engine_search.py), so the line plays its policy
        self.nodes: list[dict] = []
        self.memo: dict[str, int] = {}
        self.searches = 0

    def our_action(self, st) -> tuple[str, list]:
        pe = convert.pe
        res = pe.monte_carlo_tree_search_vs_ai(st, 60_000, self.iterations, seed=self.seed, p_model=self.p_model).side_one
        self.searches += 1
        ranked = sorted(((r.visits, r.total_score / max(r.visits, 1), r.move_choice) for r in res), reverse=True)
        return ranked[0][2], [[c, n, round(v, 3)] for n, v, c in ranked[:3]]

    def enemy(self, st) -> list[tuple[str, float]]:
        opts = convert.pe.rr_ai_choice_weighted(st, "side_two")
        return opts or [("none", 1.0)]

    def leaf(self, s: str, st, reach: float, kind: str) -> int:
        self.nodes.append({"id": len(self.nodes), "state": s, "reach": reach, "leaf": kind,
                           "dead": [not a for a in _alive(st.side_one)]})
        return len(self.nodes) - 1

    def expand(self, s: str, reach: float = 1.0, turn: int = 0) -> int:
        """Build the policy tree (structure only; values come from backup())."""
        pe = convert.pe
        st = pe.State.from_string(s)
        if s in self.memo:  # merged; a state still being expanded is a cycle back up the line: unexplored
            return self.memo[s] if self.nodes[self.memo[s]].get("done") else self.leaf(s, st, reach, "open")
        over = _over(st)
        if over or reach < self.eps or turn >= self.max_turns:
            return self.leaf(s, st, reach, over or "open")
        action, top = self.our_action(st)
        node = {"id": len(self.nodes), "state": s, "reach": reach, "turn": turn, "action": action, "top": top, "children": []}
        self.nodes.append(node)
        self.memo[s] = node["id"]
        for e, w in self.enemy(st):
            try:
                outcomes = pe.generate_instructions(st, _name(action), _name(e))
            except ValueError as err:
                raise ValueError(f"{err}; options {pe.get_all_options(st)}; search top {top}") from err
            for si in outcomes:
                p = w * si.percentage / 100
                if p <= 0:
                    continue
                child = pe.State.from_string(s).apply_instructions(si)  # returns the new state
                c = self.expand(child.to_string(), reach * p, turn + 1)
                node["children"].append([round(p, 6), e, _label(st, child), c])
        node["done"] = True
        return node["id"]

    def playout(self, s: str, rng, turns: int = 60) -> tuple[bool, list[bool]]:
        """One fight from state `s` to its end under the same policy (our search, the AI's weighted
        choices, the engine's chance outcomes, all sampled); not over after `turns`: a loss (the cap)."""
        pe = convert.pe
        st = pe.State.from_string(s)
        for _ in range(turns):
            if _over(st):
                break
            action, _ = self.our_action(st)
            opts = self.enemy(st)
            e = rng.choices([c for c, _ in opts], weights=[w for _, w in opts])[0]
            outs = pe.generate_instructions(st, _name(action), _name(e))
            st = st.apply_instructions(rng.choices(outs, weights=[o.percentage for o in outs])[0])
        return _over(st) == "win", [not a for a in _alive(st.side_one)]

    def estimate_tail(self, n: int, seed: int = 0) -> None:
        """Monte Carlo policy evaluation of the unexplored leaves: n playouts from leaves drawn in
        proportion to their reach; a leaf drawn k times gets its k-playout mean, an undrawn one the
        mean of all playouts. The exact part of the tree stays exact."""
        import random
        rng = random.Random(seed)
        open_ = [x for x in self.nodes if x.get("leaf") == "open"]
        if not open_ or n <= 0:
            return
        drawn = rng.choices(open_, weights=[x["reach"] for x in open_], k=n)
        runs = []
        for x in drawn:
            won, dead = self.playout(x["state"], rng)
            x.setdefault("runs", []).append((won, dead))
            runs.append((won, dead))
        self.tail = runs

    def backup(self, i: int = 0, memo: dict | None = None) -> dict:
        """Values of node i: the exact lower bounds (win_lo, clean_lo, die_lo: unexplored counts as
        neither), the unexplored mass `open`, and estimates (win, clean, deaths, die) where the
        unexplored part takes its playout values."""
        memo = {} if memo is None else memo
        if i in memo:
            return memo[i]
        n = self.nodes[i]
        if "leaf" in n:
            dead = n["dead"]
            if n["leaf"] == "open":
                runs = n.get("runs") or getattr(self, "tail", None) or [(False, dead)]
                won = sum(r[0] for r in runs) / len(runs)
                clean = sum(r[0] and not any(r[1]) for r in runs) / len(runs)
                die = [sum(r[1][j] for r in runs) / len(runs) for j in range(len(dead))]
                v = {"win_lo": 0.0, "clean_lo": 0.0, "die_lo": [float(d) for d in dead], "open": 1.0,
                     "win": won, "clean": clean, "die": die}
            else:
                won = n["leaf"] == "win"
                v = {"win_lo": float(won), "clean_lo": float(won and not any(dead)), "die_lo": [float(d) for d in dead],
                     "open": 0.0, "win": float(won), "clean": float(won and not any(dead)), "die": [float(d) for d in dead]}
        else:
            kids = [(p, self.backup(c, memo)) for p, _, _, c in n["children"]]
            total = sum(p for p, _ in kids) or 1.0
            v = {k: sum(p * x[k] for p, x in kids) / total for k in ("win_lo", "clean_lo", "open", "win", "clean")}
            for k in ("die_lo", "die"):
                v[k] = [sum(p * x[k][j] for p, x in kids) / total for j in range(len(kids[0][1][k]))]
        v["deaths"] = sum(v["die"])
        n.update(v)
        memo[i] = v
        return v

    def summary(self) -> dict:
        r = self.backup()
        tail = getattr(self, "tail", [])
        return {"win": r["win"], "deathless": r["clean"], "deaths": r["deaths"], "die": r["die"],
                "exact": {"win": r["win_lo"], "deathless": r["clean_lo"]}, "unexplored": r["open"],
                "playouts": len(tail), "nodes": len(self.nodes), "searches": self.searches}

    def text(self, show: float = 0.02, fold: float = 0.02) -> str:
        """The line as an indented tree: nodes reached with probability >= `show`; sibling
        outcomes after which we play the same action with deathless-win odds within `fold`
        are folded into one row (the human line only branches where the plan or the risk changes)."""
        out, seen = [], set()

        def rows(i: int, depth: int, reach: float):
            n = self.nodes[i]
            if "action" not in n or i in seen or reach < show:
                return
            seen.add(i)
            groups: dict = {}
            for p, e, lab, c in n["children"]:
                k = self.nodes[c]
                key = (k.get("action") or k["leaf"], round(k["clean"] / fold))
                g = groups.setdefault(key, [0.0, [], c, e])
                g[0] += p
                g[1].append(lab)
            for (act, _), (p, labs, c, e) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
                k = self.nodes[c]
                what = labs[0] if len(labs) == 1 else f"{len(labs)} outcomes, e.g. {labs[0]}"
                out.append(f"{'  ' * depth}{p:5.1%}  enemy {e}: {what}  ->  {act}   "
                           f"win {k['win']:.0%} deathless {k['clean']:.0%}" + (f"  ({k['open']:.0%} estimated)" if k["open"] > 0.005 else ""))
                rows(c, depth + 1, reach * p)

        r = self.nodes[0]
        out.append(f"T1 {r['action']}   win {r['win']:.1%}  deathless {r['clean']:.1%}  "
                   f"(exact {1 - r['open']:.1%} of the mass; search's top: {r['top']})")
        rows(0, 1, 1.0)
        return "\n".join(out)


def logged_sets(events: str, boss: str, label: str | None) -> tuple[int, list[dict]]:
    """(cap, sets) of a logged six: the first `team` / `box` event of `boss` in a probe run's
    events (with `label`, the first so labelled)."""
    evs = [json.loads(x) for x in open(events)]
    cap = next(e["cap"] for e in evs if e["kind"] == "boss_start" and e["boss"] == boss)
    row = next(e for e in evs if e["kind"] in ("team", "box") and e["boss"] == boss and (label is None or e.get("label") == label))
    return cap, row["sets"]


def reference_sets(name: str) -> tuple[str, int, list[dict]]:
    """(boss, cap, sets) of a player's reference team (data/rr/reference_teams.json): what the
    source states, the rest from our set builder (moves: best4 over the legal pool after the
    stated ones; nature: choose_nature; ability: slot 0; items: assign_items' picks for the mons
    the source gives none, from the items in hand by then)."""
    import rr_probe as rp
    from rr_postmortem import attribution  # noqa: F401  (switches gauntlet to the RR game)
    from gauntlet import Mon, assign_items, encounter_moves
    ref = next(t for t in json.load(open(rp.RR / "reference_teams.json"))["teams"] if t["name"] == name)
    boss = ref["boss"]
    by, known = rp.pools()
    block, cap, _, progress = by[boss]
    packed, dump_cap, _ = rp.boss_team(boss)
    cap = max(cap, dump_cap)
    rp.hardcore()
    profile = rp.boss_profile(packed, known, rp.arena(block.get("battleEffect", ""))[0])
    sets = []
    for m in ref["mons"]:
        moves = list(m.get("moves", []))
        moves += [x for x in encounter_moves(m["species"], cap, "best4", progress=progress) if x not in moves][:4 - len(moves)]
        sets.append({"species": m["species"], "moves": moves, "item": m.get("item"),
                     "ability": m.get("ability") or known[m["species"]]["abilities"]["0"],
                     "nature": m.get("nature") or rp.choose_nature(m["species"], moves, known, profile, cap)})
    mons = [Mon(x["species"], cap, x["moves"], x["ability"]) for x in sets]
    assign_items(mons, progress)
    taken = {x["item"] for x in sets if x["item"]}
    for x, m in zip(sets, mons):
        if not x["item"] and m.item and m.item not in taken:
            x["item"] = m.item
    return boss, cap, sets


def start(boss: str, cap: int, sets: list[dict]):
    """The first decision of a six against its boss: (engine state, party, enemy team, arena, bridge)."""
    import rr_probe as rp
    from gauntlet import Mon, pack_team
    from ai_util.sim import SimBattle, SimBridge
    by, _ = rp.pools()
    arena, _ = rp.arena(by[boss][0].get("battleEffect", ""))
    party = [Mon(s["species"], cap, s["moves"], s["ability"], s["item"] or None, nature=s["nature"],
                 evs=dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 0)) for s in sets]
    enemy_packed = rp.boss_team(boss)[0]
    bridge = SimBridge(showdown=str(rp.FORK))
    sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena)
    dump = bridge.call(op="dump", id=sb.id)
    state, issues = convert.to_engine(dump)
    if issues:
        print(f"note: the start does not convert for the engine: {issues}")
    return state, party, enemy_packed, arena, bridge


def _check_job(boss: str, cap: int, sets: list, iterations: int, seed: int, rust_enemy: bool = False) -> tuple[bool, list[bool], bool]:
    """One Showdown battle of the line's six: our search (the line's policy) against the real
    (Python) RR AI. (won, our mons dead, hit the turn cap)."""
    import random
    import numpy as np
    import rr_probe as rp
    from gauntlet import pack_team
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBattle, player_brain
    from rl.engine_search import EngineSearcher
    convert.use_rr()
    _, party, enemy_packed, arena, bridge = start(boss, cap, sets)
    enemy = player_brain(RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50))
    py_enemy = enemy
    if rust_enemy:  # the diagnostic: the Rust port plays the enemy, so only the engine's mechanics differ from the line
        from rl.engine_search import RustEnemy
        enemy = RustEnemy(py_enemy)
    search = EngineSearcher(Searcher(py_enemy, k=9, rollouts=8, rollout=calc_brain(), seed=seed), iterations=iterations)
    search.fixed_seed = 0  # the line's policy: Line.our_action searches with seed 0
    rng = random.Random(seed)
    bseed = [rng.randrange(1, 2 ** 16) for _ in range(4)]
    random.seed(sum(bseed))
    search.rng = np.random.default_rng(bseed)
    sb = SimBattle(bridge, rp.FMT, pack_team(party), enemy_packed, full_info=True, arena=arena, seed=bseed)
    sb.play({"p1": search, "p2": enemy}, 100)
    st = sb.state()
    sb.close()
    bridge.close()
    return st["winner"] == "p1", [m["fainted"] for m in st["p1"]], sb.timed_out


def check(boss: str, cap: int, sets: list, iterations: int, n: int, workers: int, rust_enemy: bool = False) -> dict:
    """n Showdown battles of the six under the line's policy: observed P(win), P(deathless), deaths."""
    import multiprocessing as mp
    jobs = [(boss, cap, sets, iterations, 1000 + i, rust_enemy) for i in range(n)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.starmap(_check_job, jobs)
    won = [r[0] for r in res]
    clean = [r[0] and not any(r[1]) for r in res]
    se = lambda xs: 1.96 * (sum(xs) / n * (1 - sum(xs) / n) / n) ** 0.5
    return {"n": n, "win": sum(won) / n, "win_ci": se(won), "deathless": sum(clean) / n, "deathless_ci": se(clean),
            "deaths": sum(sum(r[1]) for r in res) / n, "capped": sum(r[2] for r in res) / n,
            "die": [sum(r[1][j] for r in res) / n for j in range(len(res[0][1]))]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", help="a probe run's <out>.events.jsonl (with --boss)")
    ap.add_argument("--boss")
    ap.add_argument("--reference", help="a player team from data/rr/reference_teams.json by name (K1, B1 ...) instead")
    ap.add_argument("--no-line", action="store_true", help="skip the line (only --check)")
    ap.add_argument("--label", help="the team event's label (cover, top, searched ...); default the first")
    ap.add_argument("--eps", type=float, default=0.0005, help="expand a branch while its reach probability >= eps")
    ap.add_argument("--iterations", type=int, default=3000, help="search iterations per decision")
    ap.add_argument("--playouts", type=int, default=400, help="playouts that value the unexplored branches")
    ap.add_argument("--show", type=float, default=0.02, help="print nodes reached with probability >= show")
    ap.add_argument("--check", type=int, default=0, help="play this many Showdown battles of the six under the same policy and compare")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rust-enemy", action="store_true", help="--check with the Rust AI port as the enemy (separates the AI model from the engine)")
    ap.add_argument("--out", help="write the nodes as JSON (the dashboard's Line view)")
    args = ap.parse_args()
    convert.use_rr()
    if args.reference:
        args.boss, cap, sets = reference_sets(args.reference)
    else:
        cap, sets = logged_sets(args.events, args.boss, args.label)
    state, party, *_ = start(args.boss, cap, sets)
    print("sets: " + "; ".join(f"{x['species']} {x['nature']} {x['ability']} {x['item']} {x['moves']}" for x in sets))
    t0 = time.time()
    line = Line(eps=2.0 if args.no_line else args.eps, iterations=args.iterations)
    line.expand(state.to_string())
    line.estimate_tail(0 if args.no_line else args.playouts)
    s = line.summary()
    print(f"{args.boss}: {[m.species for m in party]}  ({s['nodes']} nodes, {s['playouts']} playouts, "
          f"{s['searches']} searches, {time.time() - t0:.0f}s)")
    print(f"P(win) {s['win']:.1%}  P(deathless) {s['deathless']:.1%}  E[deaths] {s['deaths']:.2f}   "
          f"exact part: {1 - s['unexplored']:.1%} of the probability (win >= {s['exact']['win']:.1%}, "
          f"deathless >= {s['exact']['deathless']:.1%}); the rest from playouts")
    print("P(dies): " + ", ".join(f"{m.species} {d:.0%}" for m, d in zip(party, s["die"])))
    print("  note: P(deathless) may be ~10 points low (Erika check, 2026-11-24): the engine branches a damage roll"
          " only when it KOs directly, not when it decides whether poison / chip damage finishes a mon later")
    if not args.no_line:
        print(line.text(args.show))
    if args.check:
        c = check(args.boss, cap, sets, args.iterations, args.check, args.workers, args.rust_enemy)
        print(f"Showdown check, {c['n']} battles vs the {'Rust' if args.rust_enemy else 'Python'} AI: P(win) {c['win']:.1%} ± {c['win_ci']:.1%} "
              f"(line {s['win']:.1%}), P(deathless) {c['deathless']:.1%} ± {c['deathless_ci']:.1%} (line {s['deathless']:.1%}), "
              f"E[deaths] {c['deaths']:.2f} (line {s['deaths']:.2f}), capped {c['capped']:.1%}")
        print("  P(dies) on Showdown: " + ", ".join(f"{m.species} {d:.0%} (line {x:.0%})" for m, d, x in zip(party, c["die"], s["die"])))
        s["check"] = c
    if args.out:
        Path(args.out).write_text(json.dumps({"boss": args.boss, "team": [m.species for m in party], "summary": s,
                                              "nodes": [{k: v for k, v in n.items() if k != "state"} for n in line.nodes]}))


if __name__ == "__main__":
    main()
