"""Radical Red: calibrate the matchup matrix against duels (TODO.md B1, 2026-10-12).

The cover chooser and the calc bot's switching trust `ai_util.matchup.pair()`: a one-turn
damage calc per (candidate, enemy mon) labelled safe / race / none. Nothing had checked
those labels against play. Here every candidate the pool offers before a boss (stage A's
list: the pool plus the starters, built as the probes build them) duels every enemy mon
alone, the calc bot playing under the arena, N paired seeds (the probe's seeds), and the
label is scored against the duel: win rate, HP left, turns. The report per boss is the
confusion table (cells / duel win / HP left per tier), the false safes (safe cells that
lose the duel at least half the time) and the false nones (none cells that win it almost
always) by name with the paper numbers behind them. Measures only; changes nothing.

  uv run python scripts/rr_matrix_calib.py --bosses BROCK,MISTY --legal --arena --seeds 20 --out results/rr_matrix_calib.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from poke_env.data import to_id_str  # noqa: E402
import rr_probe as rp  # noqa: E402
import rr_team_probe as tp  # noqa: E402
from gauntlet import pack_team  # noqa: E402

FALSE_SAFE, FALSE_NONE = 0.5, 0.9


def first_hits(history: list[str], paper: dict) -> dict:
    """From one duel's protocol (p1's view): the first hit each side lands — (move id, the
    fraction of max HP it took, the formula's fraction for that move if it hits) — skipping
    misses and Protect; an immune hit counts as 0. HP is tracked through every line (p1
    cur/max, p2 percent). Multi-hit moves sum their -damage lines."""
    hp = {"p1a": 1.0, "p2a": 1.0}
    out, cur = {}, None  # cur = (side, move id, hp before, taken so far)

    def flush():
        nonlocal cur
        if cur and cur[0] not in out and cur[3] is not None:
            out[cur[0]] = (cur[1], round(cur[3], 3), paper.get(cur[0], {}).get(cur[1]))
        cur = None
    for line in history:
        f = line.split("|")
        if len(f) < 3:
            continue
        if f[1] in ("move", "turn"):
            flush()
            if f[1] == "move" and len(f) > 3:
                cur = (f[2][:3], to_id_str(f[3]), hp[f[2][:3]], None)
        elif f[1] in ("-damage", "-heal", "-sethp"):
            side, val = f[2][:3], f[3].split()[0]
            new = 0.0 if val == "0" else int(val.split("/")[0]) / int(val.split("/")[1])
            direct = f[1] == "-damage" and not any(x.startswith("[from]") for x in f[4:])
            if direct and cur and side != cur[0]:
                cur = (cur[0], cur[1], cur[2], (cur[3] or 0.0) + hp[side] - new)
            hp[side] = new
        elif f[1] == "-immune" and cur and f[2][:3] != cur[0]:
            cur = (cur[0], cur[1], cur[2], cur[3] if cur[3] is not None else 0.0)
        elif f[1] in ("-miss", "-activate", "-fail") and cur and ("Protect" in line or f[1] != "-activate"):
            cur = None
    flush()
    return out


def duels(ctx, mon, packed: str, n: int) -> dict:
    """mon against one packed enemy mon, n paired seeds: per seed won, HP fraction left (0 on
    a loss) and turns; plus each side's first hit per seed against the formula (first_hits;
    the seeds share one first roll, so a single seed reads ~8% low across the board)."""
    from ai_util.calc_bot import dmg_frac
    from ai_util.sim import SimBattle
    won, hp, turns, first = [], [], [], []
    for i in range(n):
        seed = ctx.seeds[i]
        sb = SimBattle(ctx.bridge, rp.FMT, pack_team([mon]), packed, full_info=True, arena=ctx.arena, seed=seed)
        if i == 0:
            b = sb.mirrors["p1"]
            me, opp = next(iter(b.team.values())), next(iter(b.opponent_team.values()))
            paper = {"p1a": {mv.id: round(dmg_frac(me, opp, mv, True, b, with_acc=False), 3) for mv in me.moves.values()},
                     "p2a": {mv.id: round(dmg_frac(opp, me, mv, False, b, with_acc=False), 3) for mv in opp.moves.values()}}
        random.seed(sum(seed))
        sb.play({"p1": ctx.calc, "p2": ctx.enemy}, 100)
        st = sb.state()
        w = st["winner"] == "p1"
        won.append(int(w))
        hp.append(st["p1"][0]["hp"] / max(st["p1"][0]["maxhp"], 1) if w else 0.0)
        turns.append(st.get("turn", 0))
        first.append(first_hits(sb.history["p1"], paper))
        sb.close()
    return {"wins": won, "hp": hp, "turns": turns, "first": first}


def calibrate(args, name: str) -> list[dict]:
    """Every (candidate, enemy mon) cell at boss `name`: the paper pair and its duels."""
    pargs = SimpleNamespace(n_solo=args.seeds, n_team=args.seeds, n_search=args.seeds, search="9,8", seed=args.seed,
                            death_weight=0.5, legal=args.legal, arena=args.arena, blind_field=False, matchup=False, items=False, out=None)
    ctx = tp.setup(pargs, name, announce=False)
    cands = ctx.pool + [s for s in rp.STARTERS if s not in ctx.pool]
    mons = {s: m for s in cands if (m := tp.mon_of(ctx, s)) is not None}
    paper = tp.cells(ctx.bridge, mons, ctx.enemy_packed, ctx.arena)
    enemies = [(m.split("|")[0], m) for m in ctx.enemy_packed.split("]")]
    print(f"{name}: trainer #{ctx.progress}, cap {ctx.cap}, {len(mons)} candidates x {len(enemies)} enemies, "
          f"{args.seeds} seeds, arena {ctx.arena or '-'}", flush=True)
    rows = []
    if missing := [s for s in mons if s not in paper]:  # formes the mirror reads under another id (rows_for skips them)
        print(f"{name}: no paper row for {missing}", flush=True)
    for s, mon in mons.items():
        for ename, packed in enemies:
            p = paper.get(s, {}).get(to_id_str(ename))
            if p is None:
                continue
            d = duels(ctx, mon, packed, args.seeds)
            ef = packed.split("|")
            rows.append({"boss": name, "cap": ctx.cap, "cand": s, "set": tp.show(mon), "moves": mon.moves, "enemy": ename,
                         "our_ability": mon.ability, "their_ability": ef[3], "their_item": ef[2],
                         "tier": "safe" if p["safe"] else "race" if p["race"] else "none", **{k: p[k] for k in ("ko_them", "ko_us", "faster", "heals", "hit", "take")},
                         **d, "win": sum(d["wins"]) / args.seeds})
    ctx.bridge.close()
    return rows


def report(name: str, rows: list[dict]) -> None:
    print(f"\n{name}: {len(rows)} cells")
    print(f"  {'tier':<5} {'cells':>5} {'duel win':>8} {'HP left':>8} {'win>=90%':>8} {'win<=10%':>8}  (HP left: mean over won duels)")
    for tier in ("safe", "race", "none"):
        r = [x for x in rows if x["tier"] == tier]
        if not r:
            continue
        wins = sum(x["win"] for x in r) / len(r)
        hp = [h for x in r for h in x["hp"] if h > 0]
        print(f"  {tier:<5} {len(r):>5} {wins:8.0%} {(sum(hp) / max(len(hp), 1)):8.0%} {sum(x['win'] >= 0.9 for x in r) / len(r):8.0%} {sum(x['win'] <= 0.1 for x in r) / len(r):8.0%}")

    def line(x):
        return f"{x['cand']} vs {x['enemy']}: duel {x['win']:.0%}, paper {x['ko_them']}/{x['ko_us']} {'faster' if x['faster'] else 'slower'}{' healer' if x['heals'] else ''}"
    fs = sorted((x for x in rows if x["tier"] == "safe" and x["win"] < FALSE_SAFE), key=lambda x: x["win"])
    fn = sorted((x for x in rows if x["tier"] == "none" and x["win"] >= FALSE_NONE), key=lambda x: -x["win"])
    print(f"  false safes (safe, duel < {FALSE_SAFE:.0%}): {len(fs)}")
    for x in fs[:12]:
        print(f"    {line(x)}")
    print(f"  false nones (none, duel >= {FALSE_NONE:.0%}): {len(fn)}")
    for x in fn[:12]:
        print(f"    {line(x)}")
    formula_report(rows)


def formula_report(rows: list[dict]) -> None:
    """The first hit each side landed, averaged over the seeds that opened with the same
    move, against the formula's fraction for that move: the share within a damage roll of
    it, and the groups (attacker ability / defender item / move) whose median ratio is
    outside it — the terms the formula lacks, by name."""
    import collections
    import statistics
    for side, who in (("p1a", "ours"), ("p2a", "theirs")):
        hits = []
        for x in rows:
            per = [f[side] for f in x.get("first", []) if side in f and f[side][2]]
            if not per:
                continue
            mv = collections.Counter(f[0] for f in per).most_common(1)[0][0]
            same = [f for f in per if f[0] == mv]
            hits.append((x, mv, statistics.mean(f[1] for f in same), same[0][2]))
        if not hits:
            continue
        ratios = [a / pp for _, _, a, pp in hits]
        ok = sum(0.8 <= r <= 1.2 for r in ratios) / len(ratios)
        print(f"  formula vs engine, {who} first hit (mean over seeds): {len(hits)} cells, median engine/paper {statistics.median(ratios):.2f}, "
              f"within +-20%: {ok:.0%}, engine < 0.8x: {sum(r < 0.8 for r in ratios) / len(ratios):.0%}, > 1.2x: {sum(r > 1.2 for r in ratios) / len(ratios):.0%}")
        groups: dict = {}
        for x, mv, a, pp in hits:
            att_ab, dfn_item = (x["our_ability"], x["their_item"]) if side == "p1a" else (x["their_ability"], None)
            for key in ((f"ability {att_ab}" if att_ab else None), (f"item {dfn_item}" if dfn_item else None), f"move {mv}"):
                if key:
                    groups.setdefault(key, []).append(a / pp)
        off = sorted(((statistics.median(v), len(v), k) for k, v in groups.items() if len(v) >= 3 and not 0.8 <= statistics.median(v) <= 1.2),
                     key=lambda t: -t[1])
        for med, n, k in off[:14]:
            print(f"    {k:<28} n={n:<4} engine/paper median {med:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bosses", default="BROCK,MISTY")
    ap.add_argument("--seeds", type=int, default=20, help="paired duels per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legal", action="store_true", help="drop the moves hardcore bans for the player")
    ap.add_argument("--arena", action="store_true", help="the gym's permanent field effect in the sim")
    ap.add_argument("--out")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for name in [b.strip() for b in args.bosses.split(",")]:
        rows += calibrate(args, name)
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name in dict.fromkeys(r["boss"] for r in rows):
        report(name, [r for r in rows if r["boss"] == name])
    if len({r["boss"] for r in rows}) > 1:
        report("ALL", rows)
    print(f"\nmatrix calibration | legal {args.legal} | arena {args.arena} | {args.seeds} seeds | calc bot vs RR AI, full info | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
