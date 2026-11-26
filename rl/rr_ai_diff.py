"""Compare Rust `rr_ai_choice` against Python `RadicalRedAI` on real Radical Red boss battles.

Plays hardcore RR boss battles on Showdown (SimBattle + poke-env mirror, full info,
arena on). At every non-trivial decision where the enemy (p2) has >1 legal option,
compares Python `RadicalRedAI`'s pick with `poke_engine_rr.rr_ai_choice(state, "side_two")`
on the converted `poke-engine` state.

    uv run python rl/rr_ai_diff.py --positions 600 --seed 0 --out logs/aidiff_off.jsonl

On-policy (--onpolicy): the box teams of a probe run, played as the probe plays them, p1 = the
engine searcher following the Rust AI model (p = 0.8), so the positions are the ones real
searched battles reach. --rescore re-runs the current Rust build on a saved file's states
(seconds, no battles): the loop for fixing rr_ai.rs.

    uv run python rl/rr_ai_diff.py --onpolicy results/rr_p1_mix0.8.events.jsonl --boxes 10 --out logs/aidiff_on.jsonl
    uv run python rl/rr_ai_diff.py --rescore logs/aidiff_on.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_util.calc_bot import calc_brain  # noqa: E402
from ai_util.rr_ai import RadicalRedAI  # noqa: E402
from ai_util.sim import SimBattle, SimBridge, order_to_choice  # noqa: E402
from gauntlet import pack_team  # noqa: E402
from poke_env.battle.move import Move as PyMove  # noqa: E402
from poke_env.battle.pokemon import Pokemon as PyPokemon  # noqa: E402
from poke_env.data import GenData, to_id_str  # noqa: E402
from rl import convert  # noqa: E402
import rr_probe as rp  # noqa: E402



def _canon_py(dump: dict, py_order) -> tuple[str, str]:
    """Map poke-env BattleOrder -> (canonical option matching rust, kind)."""
    obj = getattr(py_order, "order", None)
    if isinstance(obj, PyMove):
        if obj.id == "recharge":
            return "No Move", "move"
        mid = "hiddenpower" if obj.id.startswith("hiddenpower") else obj.id  # typed or not: the engine's typed id
        return convert._move_id(mid, dump["p2"]["pokemon"][0]), "move"
    if isinstance(obj, PyPokemon):
        return f"switch {to_id_str(obj.species)}", "switch"
    msg = order_to_choice(py_order)
    return msg, "switch" if msg.startswith("switch") else "move"


def _canon_rust(opt: str) -> str:
    return opt.removesuffix("-mega").removesuffix("-tera")


def _classify_diff(rec: dict) -> str:
    py_k = rec["py_kind"]
    rust_k = rec["rust_kind"]
    if rec["forced"]:
        return "forced_switch:different_mon"
    if py_k == "switch" and rust_k == "move":
        return "py_switch_vs_rust_move"
    if py_k == "move" and rust_k == "switch":
        return "py_move_vs_rust_switch"
    if py_k == "switch" and rust_k == "switch":
        return "voluntary_switch:different_mon"
    return f"move:{rec['py']}_vs_{rec['rust'][0]}"


def rust_pred(state) -> tuple[list[str], str, dict[str, float]]:
    """The Rust AI on an engine state: its tied top options, their kind, and the weighted
    choice (every random roll, ties split) as {option: probability}."""
    rust = [_canon_rust(x) for x in convert.pe.rr_ai_choice(state, "side_two")]
    weighted: dict[str, float] = {}
    for c, w in convert.pe.rr_ai_choice_weighted(state, "side_two"):
        weighted[_canon_rust(c)] = weighted.get(_canon_rust(c), 0.0) + w
    return rust, "switch" if rust and rust[0].startswith("switch ") else "move", weighted


def score(rec: dict, state) -> dict:
    rec["rust"], rec["rust_kind"], w = rust_pred(state)
    rec["agree"] = rec["py"] in rec["rust"]
    rec["p_py"] = round(w.get(rec["py"], 0.0), 4)
    rec.pop("pattern", None)
    if not rec["agree"]:
        rec["pattern"] = _classify_diff(rec)
    return rec


def compare(dump: dict, py_order, boss) -> dict | None:
    """One p2 decision: the Python AI's pick against the Rust AI's on the converted state (None
    when the position does not convert). Keeps the engine state, so --rescore can re-run it."""
    state, issues = convert.to_engine(dump)
    if issues:
        return None
    py_canon, py_kind = _canon_py(dump, py_order)
    a1, a2 = dump["p1"]["pokemon"][0], dump["p2"]["pokemon"][0]
    rec = {
        "boss": boss, "turn": dump["turn"], "forced": bool(dump["p2"]["forced"]),
        "p1": a1["species"], "p2": a2["species"],
        "p1_hp": round(a1["hp"] / max(a1["maxhp"], 1), 2), "p2_hp": round(a2["hp"] / max(a2["maxhp"], 1), 2),
        "p1_ab": a1["ability"], "p2_ab": a2["ability"], "p1_item": a1["item"], "p2_item": a2["item"],
        "weather": dump["weather"], "terrain": dump["terrain"],
        "py": py_canon, "py_kind": py_kind, "legal": [c[1] for c in dump["p2"]["choices"]],
        "state": state.to_string(),
    }
    return score(rec, state)


def py_view(ai, b) -> dict:
    """What the Python AI sees, for reading a disagreement: the bench ranking (species, score,
    flags), each move's score and hit (fraction of the foe's HP), the foe's hits on us. Leaves the global random state as it found it."""
    a, f = b.active_pokemon, b.opponent_active_pokemon
    if f is None:
        return {}
    st = random.getstate()
    try:
        out = {"ranked": [(m.species, s, sorted(fl)) for m, s, fl in ai._ranked_switches(b)],
               "bench": {m.species: ({x.id: round(ai._hit(b, m, f, x, True), 3) for x in ai._attacks(m)},
                                     {x.id: round(ai._hit(b, f, m, x, False), 3) for x in ai._attacks(f)})
                         for m in b.available_switches}} if b.available_switches else {}
        if b.force_switch or a is None:
            return out
        return {**out, "score": {m.id: ai._score_move(b, a, f, m) for m in b.available_moves},
                "hit": {m.id: round(ai._hit(b, a, f, m, True), 3) for m in ai._attacks(a)},
                "foe_hit": {m.id: round(ai._hit(b, f, a, m, False), 3) for m in ai._attacks(f)}}
    finally:
        random.setstate(st)


# --- on-policy: p1 is the real searcher (phase 1's EngineSearcher vs the Rust AI model, p = 0.8) ---

def onpolicy_job(config: dict, boss: str, team: list, i: int) -> list[dict]:
    """Battle i of `team` against `boss` as the probe plays it (rr_team_probe.one_battle, the run's
    settings from its start event), p1 = EngineSearcher, p2 = the Python AI, every non-trivial p2
    decision compared."""
    import rr_team_probe as rtp
    from types import SimpleNamespace
    args = SimpleNamespace(**config)
    ctx = rtp.worker_ctx(args, boss)
    ai = getattr(ctx, "py_ai", None) or RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50)
    ctx.py_ai, recs = ai, []

    def enemy(sb, side):
        if sb.mirrors[side].wait:
            return "default"
        dump = sb.bridge.call(op="dump", id=sb.id)
        why = py_view(ai, sb.mirrors[side])
        order = ai.choose_move(sb.mirrors[side])
        if len(dump[side]["choices"]) > 1 and (rec := compare(dump, order, boss)) is not None:
            recs.append({**rec, "battle": i, "team": team, "why": why})
        return order_to_choice(order)

    rtp.one_battle(ctx.bridge, ctx.calc, enemy, rtp.team_mons(ctx, team), ctx.enemy_packed, i,
                   search=ctx.searcher, arena=ctx.arena, seeds=ctx.seeds, boss=boss)
    return recs


def collect_onpolicy(events: str, bosses: list[str] | None, boxes: int, battles: int, iterations: int,
                     workers: int) -> list[dict]:
    import multiprocessing as mp
    lines = [json.loads(x) for x in open(events)]
    config = next(r["config"] for r in lines if r["kind"] == "start")
    config.update(engine=iterations, engine_opp="ai", engine_p=0.8, enemy="python", out=None, states=False)
    teams = sorted((r["boss"], r["i"], r["team"]) for r in lines if r["kind"] == "box"
                   and (not bosses or r["boss"] in bosses) and r["i"] < boxes)
    jobs = [(config, b, team, i) for b, _, team in teams for i in range(battles)]
    with mp.get_context("spawn").Pool(workers) as pool:
        return [r for recs in pool.starmap(onpolicy_job, jobs, chunksize=1) for r in recs]


def collect_diffs(n_positions: int, seed: int, max_per_boss: int, bosses_filter: list[str] | None):
    convert.use_rr()
    bridge = SimBridge(showdown=str(rp.FORK))
    rng = random.Random(seed)
    by_boss, known = rp.pools()
    bosses = sorted({row[3]: k for k, row in by_boss.items() if k.startswith("#")}.items())
    if bosses_filter:
        wanted = {b.strip().upper() for b in bosses_filter}
        bosses = [(t, k) for t, k in bosses if k.upper() in wanted or by_boss[k][0].get("name", "").upper() in wanted]

    p1_brain = calc_brain()
    p2_ai = RadicalRedAI(battle_format=rp.FMT, start_listening=False, log_level=50)

    recs: list[dict] = []
    skipped_issues = Counter()
    trivial = 0
    battles = 0

    try:
        while len(recs) < n_positions:
            progressed = False
            for t, key in bosses:
                if len(recs) >= n_positions:
                    break
                block, cap, pool, progress = by_boss[key]
                try:
                    enemy, dump_cap, _ = rp.boss_team(t)
                    party = rp.player_party(pool, max(cap, dump_cap), rng, known, progress)
                except Exception:  # noqa: BLE001
                    continue
                if not party:
                    continue
                arena, _ = rp.arena(block.get("battleEffect", ""))
                bseed = [rng.randrange(1, 1 << 16) for _ in range(4)]
                random.seed(sum(bseed))
                try:
                    sb = SimBattle(
                        bridge,
                        rp.FMT,
                        pack_team(party),
                        enemy,
                        full_info=True,
                        arena=arena,
                        seed=bseed,
                    )
                except Exception:  # noqa: BLE001 -- poke-env lacks some custom formes (e.g. aegislashshield)
                    continue
                battles += 1
                boss_added = 0
                try:
                    for _ in range(80):
                        if sb.ended or len(recs) >= n_positions or boss_added >= max_per_boss:
                            break
                        choices_to_send: dict[str, str] = {}
                        if sb.pending["p1"]:
                            choices_to_send["p1"] = sb.choice("p1", p1_brain)
                        if sb.pending["p2"]:
                            dump = bridge.call(op="dump", id=sb.id)
                            legal_p2 = dump["p2"]["choices"]
                            # Ask Python RadicalRedAI first (advances its RNG as in real play)
                            py_order = p2_ai.choose_move(sb.mirrors["p2"])
                            py_choice = order_to_choice(py_order)
                            choices_to_send["p2"] = py_choice

                            if len(legal_p2) <= 1:
                                trivial += 1
                            else:
                                rec = compare(dump, py_order, t)
                                if rec is None:
                                    skipped_issues["convert"] += 1
                                else:
                                    recs.append(rec)
                                    boss_added += 1
                                    progressed = True
                        if not choices_to_send:
                            break
                        sb.step(choices_to_send)
                finally:
                    sb.close()
            if not progressed:
                break
    finally:
        bridge.close()

    return recs, skipped_issues, trivial, battles


def report(recs: list[dict], show: int) -> None:
    def pct(xs: list[dict]) -> str:
        if not xs:
            return "n/a"
        k = sum(r["agree"] for r in xs)
        return f"{k}/{len(xs)} ({100.0 * k / len(xs):.1f}%), weighted p(py) {sum(r['p_py'] for r in xs) / len(xs):.2f}"

    n = len(recs)
    forced = [r for r in recs if r["forced"]]
    normal = [r for r in recs if not r["forced"]]
    print(f"  OVERALL agreement:                {pct(recs)}")
    print(f"  Move decisions (py chose move):   {pct([r for r in normal if r['py_kind'] == 'move'])}")
    print(f"  Voluntary switches (py switch):   {pct([r for r in normal if r['py_kind'] == 'switch'])}")
    print(f"  Forced switches (post-KO/pivot):  {pct(forced)}")
    for b in sorted({r["boss"] for r in recs}, key=str):
        print(f"  boss {b!s:<28} {pct([r for r in recs if r['boss'] == b])}")

    diffs = [r for r in recs if not r["agree"]]
    if diffs:
        cat = Counter("move:different_move" if r["pattern"].startswith("move:") else r["pattern"] for r in diffs)
        print(f"\nDisagreements by category ({len(diffs)} total, % of all decisions):")
        for pat, cnt in cat.most_common():
            print(f"  {cnt:4d} {100.0 * cnt / n:5.1f}%  {pat}")
        print("\nTop patterns:")
        for pat, cnt in Counter(r["pattern"] for r in diffs).most_common(12):
            print(f"  {cnt:4d}  {pat}")
        print(f"\nSample disagreements (first {min(show, len(diffs))}):")
        for r in diffs[:show]:
            tag = "FORCED" if r["forced"] else "NORMAL"
            print(
                f"  [{tag} boss#{r['boss']} t={r['turn']}] "
                f"p2={r['p2']}({r['p2_hp']},{r['p2_ab']},{r['p2_item']}) vs "
                f"p1={r['p1']}({r['p1_hp']},{r['p1_ab']},{r['p1_item']}) "
                f"w={r['weather'] or '-'} tr={r['terrain'] or '-'} | "
                f"py={r['py']} vs rust={r['rust']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=400, help="non-trivial p2 decisions to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-boss", type=int, default=12, help="max decisions per boss battle")
    ap.add_argument("--bosses", default="", help="comma-separated boss keys (default: all)")
    ap.add_argument("--show", type=int, default=15, help="print up to N sample disagreements")
    ap.add_argument("--out", default="", help="optional jsonl path to save all records")
    ap.add_argument("--onpolicy", default="", help="a probe run's events.jsonl: replay its box teams with p1 = the "
                    "engine searcher (on-policy positions) instead of random-team calc-bot play")
    ap.add_argument("--boxes", type=int, default=10, help="--onpolicy: first N boxes per boss")
    ap.add_argument("--battles", type=int, default=2, help="--onpolicy: battles per box (the probe's seeds)")
    ap.add_argument("--iterations", type=int, default=2000, help="--onpolicy: the searcher's MCTS iterations")
    ap.add_argument("--workers", type=int, default=10, help="--onpolicy: processes (~1 GB each)")
    ap.add_argument("--rescore", default="", help="re-run the current Rust build on a saved --out file's states")
    args = ap.parse_args()

    bosses_filter = [b.strip() for b in args.bosses.split(",") if b.strip()] or None
    convert.use_rr()
    if args.rescore:
        recs = [json.loads(x) for x in open(args.rescore)]
        recs = [score(r, convert.pe.State.from_string(r["state"])) for r in recs]
        print(f"RR AI diff, rescored {args.rescore}: {len(recs)} decisions")
    elif args.onpolicy:
        recs = collect_onpolicy(args.onpolicy, bosses_filter, args.boxes, args.battles, args.iterations, args.workers)
        print(f"RR AI diff, on-policy ({args.onpolicy}, {args.boxes} boxes x {args.battles} battles, "
              f"{args.iterations} iterations): {len(recs)} non-trivial p2 decisions")
    else:
        recs, skipped, trivial, battles = collect_diffs(args.positions, args.seed, args.per_boss, bosses_filter)
        print(f"RR AI diff: {len(recs)} non-trivial p2 decisions across {battles} boss battles (seed={args.seed})")
        print(f"  trivial (1 legal option) skipped: {trivial}; non-convertible skipped: {sum(skipped.values())}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
    if recs:
        report(recs, args.show)


if __name__ == "__main__":
    main()
