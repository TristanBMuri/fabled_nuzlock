"""Compare Rust `rr_ai_choice` against Python `RadicalRedAI` on real Radical Red boss battles.

Plays hardcore RR boss battles on Showdown (SimBattle + poke-env mirror, full info,
arena on). At every non-trivial decision where the enemy (p2) has >1 legal option,
compares Python `RadicalRedAI`'s pick with `poke_engine_rr.rr_ai_choice(state, "side_two")`
on the converted `poke-engine` state.

    PYTHONPATH=<scratch_wheel_dir> uv run python rl/rr_ai_diff.py --positions 500 --seed 0
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

# mod_data.json stores explicit `"self": null` on some moves; poke-env's Move.self_boost
# checks `"self" in entry and "boosts" in entry["self"]`. Strip null `"self"` in memory.
for _mv in GenData.from_gen(9).moves.values():
    if "self" in _mv and _mv["self"] is None:
        del _mv["self"]


def _canon_py(dump: dict, py_order) -> tuple[str, str]:
    """Map poke-env BattleOrder -> (canonical option matching rust, kind)."""
    obj = getattr(py_order, "order", None)
    if isinstance(obj, PyMove):
        if obj.id == "recharge":
            return "No Move", "move"
        return convert._move_id(obj.id, dump["p2"]["pokemon"][0]), "move"
    if isinstance(obj, PyPokemon):
        return f"switch {to_id_str(obj.species)}", "switch"
    msg = order_to_choice(py_order)
    return msg, "switch" if msg.startswith("switch") else "move"


def _with_force_switch(state, dump: dict):
    """Carry dump['p1'/'p2']['forced'] into poke-engine Side.force_switch when the active mon
    is still alive (Volt Switch / U-turn / Flip Turn pivot). When hp == 0 poke-engine already
    treats the side as force-switching (including simultaneous double-KOs)."""
    f1 = bool(dump["p1"]["forced"] and dump["p1"]["pokemon"][0]["hp"] > 0)
    f2 = bool(dump["p2"]["forced"] and dump["p2"]["pokemon"][0]["hp"] > 0)
    if not f1 and not f2:
        return state
    pe = convert.pe

    def copy_side(sd, forced: bool):
        if not forced or sd.force_switch:
            return sd
        return pe.Side(
            pokemon=list(sd.pokemon),
            side_conditions=sd.side_conditions,
            active_index=sd.active_index,
            baton_passing=sd.baton_passing,
            shed_tailing=sd.shed_tailing,
            volatile_status_durations=sd.volatile_status_durations,
            wish=sd.wish,
            future_sight=sd.future_sight,
            force_switch=True,
            force_trapped=sd.force_trapped,
            slow_uturn_move=sd.slow_uturn_move,
            volatile_statuses=set(sd.volatile_statuses),
            substitute_health=sd.substitute_health,
            attack_boost=sd.attack_boost,
            defense_boost=sd.defense_boost,
            special_attack_boost=sd.special_attack_boost,
            special_defense_boost=sd.special_defense_boost,
            speed_boost=sd.speed_boost,
            accuracy_boost=sd.accuracy_boost,
            evasion_boost=sd.evasion_boost,
            last_used_move=sd.last_used_move,
            switch_out_move_second_saved_move=sd.switch_out_move_second_saved_move,
        )

    return pe.State(
        side_one=copy_side(state.side_one, f1),
        side_two=copy_side(state.side_two, f2),
        weather=state.weather,
        weather_turns_remaining=state.weather_turns_remaining,
        terrain=state.terrain,
        terrain_turns_remaining=state.terrain_turns_remaining,
        trick_room=state.trick_room,
        trick_room_turns_remaining=state.trick_room_turns_remaining,
        team_preview=state.team_preview,
    )


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
                                state, issues = convert.to_engine(dump)
                                if issues:
                                    for iss in set(issues):
                                        skipped_issues[iss] += 1
                                else:
                                    state = _with_force_switch(state, dump)
                                    rust_raw = convert.pe.rr_ai_choice(state, "side_two")
                                    rust_canon = [_canon_rust(x) for x in rust_raw]
                                    py_canon, py_kind = _canon_py(dump, py_order)
                                    rust_kind = (
                                        "switch"
                                        if rust_canon and rust_canon[0].startswith("switch ")
                                        else "move"
                                    )
                                    agree = py_canon in rust_canon
                                    a1, a2 = dump["p1"]["pokemon"][0], dump["p2"]["pokemon"][0]
                                    rec = {
                                        "boss": t,
                                        "turn": dump["turn"],
                                        "forced": bool(dump["p2"]["forced"]),
                                        "p1": a1["species"],
                                        "p2": a2["species"],
                                        "p1_hp": round(a1["hp"] / max(a1["maxhp"], 1), 2),
                                        "p2_hp": round(a2["hp"] / max(a2["maxhp"], 1), 2),
                                        "p1_ab": a1["ability"],
                                        "p2_ab": a2["ability"],
                                        "p1_item": a1["item"],
                                        "p2_item": a2["item"],
                                        "weather": dump["weather"],
                                        "terrain": dump["terrain"],
                                        "py": py_canon,
                                        "py_kind": py_kind,
                                        "rust": rust_canon,
                                        "rust_kind": rust_kind,
                                        "agree": agree,
                                        "legal": [c[1] for c in legal_p2],
                                    }
                                    if not agree:
                                        rec["pattern"] = _classify_diff(rec)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=400, help="non-trivial p2 decisions to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-boss", type=int, default=12, help="max decisions per boss battle")
    ap.add_argument("--bosses", default="", help="comma-separated boss keys (default: all)")
    ap.add_argument("--show", type=int, default=15, help="print up to N sample disagreements")
    ap.add_argument("--out", default="", help="optional jsonl path to save all records")
    args = ap.parse_args()

    bosses_filter = [b for b in args.bosses.split(",") if b.strip()] if args.bosses else None
    recs, skipped, trivial, battles = collect_diffs(
        args.positions, args.seed, args.per_boss, bosses_filter
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in recs))

    n = len(recs)
    if n == 0:
        print("No comparable positions collected.")
        return

    agree_all = sum(r["agree"] for r in recs)
    forced = [r for r in recs if r["forced"]]
    normal = [r for r in recs if not r["forced"]]
    py_moves = [r for r in normal if r["py_kind"] == "move"]
    py_vol_sw = [r for r in normal if r["py_kind"] == "switch"]
    all_sw = [r for r in recs if r["py_kind"] == "switch" or r["rust_kind"] == "switch"]

    def pct(xs: list[dict]) -> str:
        if not xs:
            return "n/a"
        k = sum(r["agree"] for r in xs)
        return f"{k}/{len(xs)} ({100.0 * k / len(xs):.1f}%)"

    print(f"RR AI diff: {n} non-trivial p2 decisions across {battles} boss battles (seed={args.seed})")
    print(f"  trivial (1 legal option) skipped: {trivial}")
    if skipped:
        print(f"  non-convertible skipped:          {sum(skipped.values())} {skipped.most_common(5)}")
    print(f"  OVERALL agreement:                {pct(recs)}")
    print(f"  Move decisions (py chose move):   {pct(py_moves)}")
    print(f"  Voluntary switches (py switch):   {pct(py_vol_sw)}")
    print(f"  Forced switches (post-KO/pivot):  {pct(forced)}")
    print(f"  All switch-involved decisions:    {pct(all_sw)}")

    diffs = [r for r in recs if not r["agree"]]
    if diffs:
        print(f"\nTop disagreement patterns ({len(diffs)} total):")
        buckets = Counter(r["pattern"] for r in diffs)
        for pat, cnt in buckets.most_common(12):
            print(f"  {cnt:4d}  {pat}")

        # Group move-vs-move differences by coarse cause
        print(f"\nSample disagreements (first {min(args.show, len(diffs))}):")
        for r in diffs[: args.show]:
            tag = "FORCED" if r["forced"] else "NORMAL"
            print(
                f"  [{tag} boss#{r['boss']} t={r['turn']}] "
                f"p2={r['p2']}({r['p2_hp']},{r['p2_ab']},{r['p2_item']}) vs "
                f"p1={r['p1']}({r['p1_hp']},{r['p1_ab']},{r['p1_item']}) "
                f"w={r['weather'] or '-'} tr={r['terrain'] or '-'} | "
                f"py={r['py']} vs rust={r['rust']}"
            )


if __name__ == "__main__":
    main()
