"""Do entry hazards pay at a boss? (2026-11-25, the user's question; hardcore bans Stealth Rock and
Spikes, but Stone Axe and Ceaseless Edge set them as a side effect.) Paired 6v6 searched battles on
starter-less routed boxes: the box's duel-picked team (base) against the same team with Kleavor
(Stone Axe: rocks) / Samurott-Hisui (Ceaseless Edge: Spikes) leading in place of its weakest
member(s) by duel value. Each hazard arm has a control that swaps the hazard move for a similar
attack without the hazard, so a gain splits into "the mon" and "the hazard".

The forced mons are the run's own (tp.mon_of: level cap, nature, items) with the hazard move put
first, and Sharpness once the Dream Patch is in hand: the ability picker likes Sheer Force on
Kleavor, which cancels Stone Axe's rocks (Showdown: `!move.hasSheerForce`).

--states <path> also logs every searched position (fit_eval.py's format) for D15, the refit of
the leaf evaluation with hazards in the data; battle ids become "<arm>-<box>-<i>" so paired
battles of different arms never share a key.

    uv run python -u scripts/rr_hazard_probe.py --bosses BLAINE --boxes 40 --battles 6 --workers 6
    uv run python -u scripts/rr_hazard_probe.py --bosses "#26,#34,#38,#44,#46,#80,#90" \\
        --arms kleavor,samurott,both --states logs/d15_hazard_data.states.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import gauntlet  # noqa: E402
import rr_probe as rp  # noqa: E402
import rr_route as rt  # noqa: E402
import rr_search as rs  # noqa: E402
import rr_team_probe as tp  # noqa: E402
from gauntlet import assign_items, base_species, load_game  # noqa: E402

SETTERS = {"kleavor": ("scyther>kleavor", "stoneaxe", "stoneedge"),  # species: (box key, hazard, control)
           "samurotthisui": ("oshawott>samurotthisui", "ceaselessedge", "knockoff")}
ARMS = {"base": [], "kleavor": [("kleavor", 1)], "kleavor_ctrl": [("kleavor", 0)],
        "samurott": [("samurotthisui", 1)], "samurott_ctrl": [("samurotthisui", 0)],
        "both": [("kleavor", 1), ("samurotthisui", 1)]}  # name: [(species, with hazard?)], leads first


def setter(ctx, sp: str, hazard: bool):
    """The run's own mon with the hazard (or control) move first; None if not legal here."""
    key, hz, ctrl = SETTERS[sp]
    m = tp.mon_of(ctx, key)
    move = hz if hazard else ctrl
    if m is None or m.species != sp or move not in gauntlet.legal_pool(sp, ctx.cap, ctx.progress):
        return None
    m = copy.copy(m)
    m.moves = [move] + [x for x in m.moves if x not in (hz, ctrl)][:3]
    if ctx.progress >= rp.ability_gates()[1]:
        m.ability = "Sharpness"
    m.nature = rp.choose_nature(sp, m.moves, ctx.known, ctx.boss, m.level)
    return m


def job(args, boss: str, box_i: int, box: list[str], duel_cells: dict, enemy_order: list[str], arm: str,
        states: str | None) -> dict | None:
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    ctx = tp.worker_ctx(args, boss)
    forced = [setter(ctx, sp, hz) for sp, hz in ARMS[arm]]
    if any(m is None for m in forced):
        return None
    species = [sp for sp in box if tp.mon_of(ctx, sp) is not None]
    team = [k for k in rt._duel_pick_team(ctx, species, duel_cells, enemy_order)
            if base_species(k) not in {base_species(SETTERS[m.species][0]) for m in forced}]
    value = lambda k: sum(c["win"] for c in duel_cells.get(k, {}).values())
    while len(team) > 6 - len(forced):
        team.remove(min(team, key=value))
    mons = forced + [copy.copy(tp.mon_of(ctx, k)) for k in team]
    assign_items(mons, ctx.progress)
    log = None
    if states:
        log = Path(states).with_suffix(f".{boss.strip('#')}-{arm}-{box_i}.part")
        log.unlink(missing_ok=True)
        ctx.searcher.log = str(log)
    bs = [tp.one_battle(ctx.bridge, ctx.calc, ctx.enemy, mons, ctx.enemy_packed, j,
                        search=ctx.searcher, arena=ctx.arena, seeds=ctx.seeds, boss=boss) for j in range(args.n_search)]
    ctx.searcher.log = None
    w, d, _, _, _ = tp.fold(bs)
    return {"win": w, "d": d, "U": sum(rs.score(b, False, len(mons)) for b in bs) / len(bs),
            "team": [m.species for m in mons], "part": str(log) if log else None}


def merge(parts: list[str], out: str) -> None:
    """Concatenate the per-job state logs, battle ids made unique across arms and boxes."""
    with open(out, "w") as f:
        for p in parts:
            tag = Path(p).suffixes[-2].lstrip(".") if len(Path(p).suffixes) > 1 else Path(p).stem
            for line in open(p):
                r = json.loads(line)
                r["battle"] = f"{tag}-{r['battle']}"
                f.write(json.dumps(r) + "\n")
            Path(p).unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bosses", default="BLAINE")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--boxes", type=int, default=40)
    ap.add_argument("--battles", type=int, default=6)
    ap.add_argument("--engine", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--states", help="log searched positions here (fit_eval.py format)")
    a = ap.parse_args()
    rp.hardcore()
    t0 = time.time()
    bosses, arms = [b.strip() for b in a.bosses.split(",")], a.arms.split(",")
    known = load_game("rr")["species"]
    by_boss = rp.pools()[0]
    args = rt._base_args(seed=a.seed, workers=a.workers, engine=a.engine, n_search=a.battles)
    jobs = []
    for boss in bosses:
        table = rt.build_boss_table(boss, extra_species=list(rp.STARTERS) + ["scyther"], seed=a.seed, workers=4)
        boxes = [[sp for sp, wh in rt.routed_box(f"haz-{boss}-{a.seed}-{i}", by_boss[boss][3], known, [table],
                                                 starter="__none__", route_methods=True, include_celadon_egg=False,
                                                 include_choices=False) if wh != "starter"] for i in range(a.boxes)]
        jobs += [(args, boss, i, box, table["duel_cells"], table["enemy_order"], arm, a.states)
                 for arm in arms for i, box in enumerate(boxes)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.starmap(job, jobs, chunksize=1)
    if a.states:
        merge([r["part"] for r in res if r and r["part"]], a.states)
    print(f"\n=== hazard probe: {a.boxes} boxes x {a.battles} battles @ {a.engine} iters, paired on box + seed "
          f"({time.time() - t0:.0f}s) ===")
    se = lambda xs: math.sqrt(sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / max(len(xs) - 1, 1) / len(xs))
    for boss in bosses:
        by = {arm: [r for (_, b, _, _, _, _, ar, _), r in zip(jobs, res) if b == boss and ar == arm] for arm in arms}
        print(f"\n{boss}\n  {'arm':<14} {'win':>6} {'dWin':>7} {'+-se':>5} {'d/mon':>6} {'U':>6} {'dU':>7} {'+-se':>6}")
        for arm, rows in by.items():
            if any(r is None for r in rows):
                print(f"  {arm:<14} not legal here")
                continue
            mean = lambda k: sum(r[k] for r in rows) / len(rows)
            line = f"  {arm:<14} {mean('win'):6.1%}"
            if "base" in by and arm != "base":
                dw = [r["win"] - b["win"] for r, b in zip(rows, by["base"])]
                du = [r["U"] - b["U"] for r, b in zip(rows, by["base"])]
                line += f" {sum(dw) / len(dw):+6.1%} {se(dw):5.1%} {mean('d'):6.2f} {mean('U'):6.3f} {sum(du) / len(du):+7.3f} {se(du):6.3f}"
            else:
                line += f" {'':>7} {'':>5} {mean('d'):6.2f} {mean('U'):6.3f}"
            print(line)
        print("  example teams:", {arm: rows[0]["team"] for arm, rows in by.items() if rows and rows[0]})


if __name__ == "__main__":
    main()
