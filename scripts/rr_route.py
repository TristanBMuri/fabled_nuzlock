"""Radical Red: encounter routing, gift/starter-egg advice, and D14 per-mon death costs (plan v3 phase 6).

Pure decision-theoretic expected value on top of the matchup-cover scorer (no RL):
  - Precomputes each legal species' best-set matchup row against each upcoming boss (including
    Mega forms once their Mega Stone is in hand and weather/terrain negation abilities).
  - Box value for boss B: sum over enemy mons of (best answer + 0.35 * second-best answer) plus
    arena-negation bonus. Adaptive boss weights focus routing on bosses with uncovered threats.
  - Expected gain of an encounter table (a route's grass_day/grass_night/rod/surfing table, or a
    Celadon Starter Egg shard color, or a choice gift): dupes-filtered, renormalised rarity
    weights times marginal box-value gain over all upcoming bosses.
  - D14 per-mon death cost: drop in future-boss box value when a mon dies, naming the exact
    upcoming boss threats that lose their answer.

Examples:
  uv run python scripts/rr_route.py --starters --bosses BROCK,MISTY,LT.SURGE,ERIKA,KOGA,BLAINE
  uv run python scripts/rr_route.py --celadon-eggs --bosses ERIKA,KOGA,BLAINE
  uv run python scripts/rr_route.py --advise --boss ERIKA --seed 3
  uv run python scripts/rr_route.py --benchmark --bosses BROCK,MISTY,ERIKA,KOGA,BLAINE --boxes 30 --seed 5 --workers 10
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import gauntlet  # noqa: E402
import rr_probe as rp  # noqa: E402
import rr_search as rs  # noqa: E402
import rr_team_probe as tp  # noqa: E402
from gauntlet import (Mon, base_species, evo_line, evolve_options, format_species,  # noqa: E402
                      load_game, mega_stone, split_evo_key)
from poke_env.data import GenData, to_id_str  # noqa: E402

RR = ROOT / "data" / "rr"
SECOND_WEIGHT = 0.45
THIRD_WEIGHT = 0.20
NEGATE_BONUS = 2.0
SPECIES_ALIASES = {
    "taurosblaze": "taurospaldeablaze",
    "taurosaqua": "taurospaldeaaqua",
    "ursalunabm": "ursalunabloodmoon",
    "flabébé": "flabebe",
    "anycappikachu": "pikachuoriginal",
}


def norm_sp(name: str) -> str:
    s = to_id_str(name)
    return SPECIES_ALIASES.get(s, s)


def cell_score(q: dict) -> float:
    """Unsaturated continuous matchup score (higher = cleaner, safer 1v1 win against this boss mon).
    Avoids the binary 3.1 ceiling of 2*safe+race+0.1*edge so routing keeps distinguishing
    dominant counters / Megas from marginal ones even on 40-70 mon boxes."""
    out_h = min(float(q.get("hit", 0.0)), 1.5)
    in_h = min(float(q.get("take", 1.0)), 2.0)
    ko_out = min(int(q.get("ko_them", 99)), 6)
    ko_in = min(int(q.get("ko_us", 1)), 6)
    faster = bool(q.get("faster", False))
    win_1v1 = 1.0 if (ko_out < ko_in or (ko_out == ko_in and faster)) else 0.0
    hits_taken = max(0, ko_out - (1 if faster else 0))
    hp_left = max(0.0, 1.0 - hits_taken * in_h) if win_1v1 else 0.0
    margin = max(0.0, (ko_in - ko_out) / 4.0)
    return max(0.0, 1.2 * win_1v1 + 0.8 * hp_left + 0.5 * out_h - 0.35 * in_h + 0.4 * margin + 0.2 * faster)

# Choice gifts / fossils / eggs not yet auto-enrolled as single fixed gifts:
# Each entry: (name, trainer_index, {option_label: [species_ids]})
def choice_encounters() -> list[tuple[str, int, dict[str, list[str]]]]:
    te = json.load(open(RR / "tracker_encounters.json"))
    out = []
    # Celadon Mansion Old Man starter egg (pre-Erika, t=33): 1 egg total, player chooses shard pool
    eggs = {shard: [norm_sp(s) for s in sps] for shard, sps in te.get("eggVendor", {}).items()}
    if eggs:
        out.append(("CELADON EGG VENDOR", 33, eggs))
    # Mt. Moon fossil (t=10)
    fossils = te.get("fossils", {})
    if "MT. MOON" in fossils:
        out.append(("MT. MOON FOSSIL", 10, {s: [norm_sp(s)] for s in fossils["MT. MOON"]}))
    # Celadon shard fossils (t=33)
    shard_fossils = {k: [norm_sp(s) for s in v] for k, v in fossils.items() if "SHARD" in k}
    if shard_fossils:
        out.append(("CELADON SHARD FOSSIL", 33, shard_fossils))
    # Saffron Fighting Dojo choice (t=44): Tauros-Paldea-Blaze vs Tauros-Paldea-Aqua
    out.append(("SAFFRON DOJO", 44, {"Tauros-Paldea-Blaze": ["taurospaldeablaze"], "Tauros-Paldea-Aqua": ["taurospaldeaaqua"]}))
    return out


def _base_args(seed: int = 0, workers: int = 1, engine: int = 20000, n_search: int = 6,
               normal: bool = False) -> argparse.Namespace:
    """Probe args; normal = normal mode's boss sets and caps with the easier player rules (no
    hardcore bans or ability swaps); otherwise tp.setup applies rp.hardcore() (legal=True)."""
    return argparse.Namespace(
        bosses="", n_solo=3, n_team=4, n_search=n_search, n_duel=4, search="9,8", seed=seed,
        workers=workers, legal=not normal, normal=normal, arena=True, blind_field=False, repair=False,
        boxes=0, chooser="cover", cands=24, n_screen=4, matchup=True, death_weight=6.0,
        no_items=False, items=True, engine=engine, engine_p=0.8, enemy="python", states=False,
        engine_opp="ai", out=None,
    )


def _mega_species(base_sp: str, known: dict, progress: int) -> str | None:
    """If `base_sp` has an obtainable Mega Stone by `progress`, return the best Mega forme id."""
    stone = mega_stone(base_sp, progress)
    if not stone:
        return None
    cands = [f"{base_sp}mega", f"{base_sp}megax", f"{base_sp}megay"]
    found = [c for c in cands if c in known]
    if not found:
        return None
    return max(found, key=lambda c: sum(known[c]["baseStats"].values()))


DUEL_CACHE_DIR = ROOT / "logs" / "duel_cache"
DUEL_CACHE_VERSION = 5  # 4: no dead Water moves under Desolate Land, Palafin duels as Hero; 5: D16 ability-aware sets + calc


def _duel_chunk_worker(boss: str, keys: list[str], enemy_names: list[str], n_duel: int, seed: int,
                       normal: bool = False) -> dict[str, dict[str, list[float]]]:
    """Run real Showdown 1v1 duels (calc_bot vs RadicalRedAI, with Mega Stones / items / arena)
    for a chunk of candidate species/branch keys on one boss."""
    import logging
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    args = _base_args(seed=seed, normal=normal)
    args.n_duel = n_duel
    ctx = tp.setup(args, boss, announce=False)
    out: dict[str, dict[str, list[float]]] = {}
    for k in keys:
        row = {}
        for en in enemy_names:
            w, hp = tp.duel(ctx, k, en, n_duel)
            row[to_id_str(en)] = [round(w, 4), round(hp, 4)]
        out[k] = row
    ctx.bridge.close()
    return out


def _load_or_run_duels(boss: str, ctx, eval_keys: list[str], n_duel: int = 2, seed: int = 0, workers: int = 4,
                       normal: bool = False) -> dict[str, dict[str, list[float]]]:
    """Compute or load cached Showdown duel matrix `{key: {enemy_id: [win, hp]}}` for `boss`."""
    import hashlib
    enemy_names = [x.split("|")[0] for x in ctx.enemy_packed.split("]")]
    sig_payload = json.dumps({
        "v": DUEL_CACHE_VERSION, "boss": boss, "progress": ctx.progress, "cap": ctx.cap,
        "arena": ctx.arena, "n_duel": n_duel, "seed": seed, "enemies": enemy_names, "normal": normal,
    }, sort_keys=True)
    sig = hashlib.sha1(sig_payload.encode()).hexdigest()[:12]
    DUEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DUEL_CACHE_DIR / f"{to_id_str(boss)}_{sig}.json"
    cached: dict[str, dict[str, list[float]]] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = {}
    missing = [k for k in eval_keys if k not in cached]
    if missing:
        n_w = max(1, min(workers, 4, len(missing)))
        if n_w > 1 and len(missing) >= 12:
            chunks = [missing[i::n_w] for i in range(n_w)]
            with mp.get_context("spawn").Pool(n_w) as pool:
                parts = pool.starmap(_duel_chunk_worker, [(boss, ch, enemy_names, n_duel, seed, normal) for ch in chunks if ch])
            for part in parts:
                cached.update(part)
        else:
            cached.update(_duel_chunk_worker(boss, missing, enemy_names, n_duel, seed, normal))
        cache_path.write_text(json.dumps(cached))
    return cached


def build_boss_table(
    boss: str,
    extra_species: list[str] | None = None,
    seed: int = 0,
    scorer: str = "duel",
    n_duel: int = 2,
    workers: int = 4,
    normal: bool = False,
) -> dict:
    """Precompute per-species matchup rows & flags for `boss`, expanding all legal branched
    evolutions (e.g. Eevee -> 8 Eeveelutions, Dartrix -> Decidueye / Decidueye-Hisui,
    Charcadet -> Armarouge / Ceruledge) and Mega forms once their Mega Stone is in hand.
    - `scorer="duel"` (default): runs real Showdown 1v1 duels (cached in `logs/duel_cache/`)
      combining real win rate & HP left with a 0.15x continuous paper tie-break.
    - `scorer="paper"`: fast static 1v1 damage heuristic (`cell_score`)."""
    args = _base_args(seed=seed, normal=normal)
    ctx = tp.setup(args, boss, announce=False)
    base_list = list(dict.fromkeys(list(ctx.pool) + list(rp.STARTERS) + list(extra_species or [])))
    dex9 = GenData.from_gen(9).pokedex
    base_list = [s for s in base_list if s in ctx.known and s in dex9]

    # Expand branched evolutions into `'base>ev'` keys while keeping single-path species as `'base'`
    eval_keys: list[str] = []
    branches: dict[str, list[str]] = {}
    for sp in base_list:
        if sp in ("ditto", "smeargle"):
            continue
        opts = evolve_options(sp, ctx.cap, caught_level=5, trainer_idx=ctx.progress)
        if len(opts) > 1:
            bkeys = [f"{sp}>{ev}" for ev in opts if ev in ctx.known and ev in dex9]
            if len(bkeys) > 1:
                branches[sp] = bkeys
                eval_keys.extend(bkeys)
            else:
                eval_keys.append(sp)
        else:
            eval_keys.append(sp)
    eval_keys = [k for k in dict.fromkeys(eval_keys)
                 if (m := tp.mon_of(ctx, k)) is not None and m.ability != "Truant"]
    rs.info(ctx, eval_keys)

    # Also evaluate Mega forms where a Mega Stone is obtainable by ctx.progress
    mega_map = {}
    for k in eval_keys:
        if k not in ctx.info:
            continue
        m = tp.mon_of(ctx, k)
        ms = _mega_species(to_id_str(m.species), ctx.known, ctx.progress)
        if ms and ms in dex9:
            mega_map[k] = ms
    if mega_map:
        mega_list = sorted(set(mega_map.values()))
        rs.info(ctx, [s for s in mega_list if s not in ctx.info])

    duels = _load_or_run_duels(boss, ctx, eval_keys, n_duel=n_duel, seed=seed, workers=workers, normal=normal) if scorer == "duel" else {}

    def _cont_row(sp_id: str, moves: list[str]) -> dict[str, float]:
        inf = ctx.info[sp_id]
        out = {}
        for e, p in inf["pair"].items():
            h = max((inf["dmg"].get(x, {}).get(e, 0.0) for x in moves), default=0.0)
            q = dict(p, hit=h, ko_them=rs.hits_to_ko(1.0, h, h) if h > 0 else 99)
            out[e] = cell_score(q)
        return out

    rows, duel_cells, negate_set, mega_set, evolved_map = {}, {}, set(), set(), {}
    enemies = [to_id_str(e) for e in ctx.enemy_species]
    for k in eval_keys:
        if k not in ctx.info:
            continue
        ds = rs.default_set(ctx, k)
        r = _cont_row(k, ds["moves"])
        evolved_map[k] = ds["species"]
        if k in mega_map and mega_map[k] in ctx.info:
            ms = mega_map[k]
            mds = rs.default_set(ctx, ms)
            mr = _cont_row(ms, mds["moves"])
            r = {e: max(r.get(e, 0.0), mr.get(e, 0.0)) for e in set(r) | set(mr)}
            mega_set.add(k)
        if scorer == "duel" and k in duels:
            drow = duels[k]
            duel_cells[k] = {e: {"win": float(drow[e][0]), "hp": float(drow[e][1])} for e in drow}
            r = {e: 2.0 * float(drow[e][0]) + 1.0 * float(drow[e][0]) * float(drow[e][1]) + 0.15 * r.get(e, 0.0)
                 for e in drow}
        rows[k] = r
        fl = rs.flags(ctx, k)
        if "negate" in fl:
            negate_set.add(k)

    # For unpinned branched base species (e.g. 'eevee'), point `rows[base]` at its single best
    # legal branch for this boss (avoiding illegally double-counting two branches from one mon)
    for base, bkeys in branches.items():
        valid = [bk for bk in bkeys if bk in rows]
        if not valid:
            continue
        best_bk = max(valid, key=lambda bk: sum(rows[bk].values()) + (NEGATE_BONUS if bk in negate_set else 0.0))
        rows[base] = rows[best_bk]
        if best_bk in duel_cells:
            duel_cells[base] = duel_cells[best_bk]
        evolved_map[base] = evolved_map[best_bk]
        if best_bk in negate_set:
            negate_set.add(base)
        if best_bk in mega_set:
            mega_set.add(base)

    ctx.bridge.close()
    return {
        "boss": boss,
        "progress": ctx.progress,
        "cap": ctx.cap,
        "arena": ctx.arena,
        "scorer": scorer,
        "enemies": sorted(next(iter(rows.values())).keys()) if rows else enemies,
        "enemy_order": [to_id_str(e) for e in enemies],
        "rows": rows,
        "duel_cells": duel_cells,
        "negate": negate_set,
        "mega": mega_set,
        "evolved": evolved_map,
        "branches": branches,
    }


def label_sp(sp: str, tables: list[dict]) -> str:
    """Format a box/branch key with its evolved species (from the last table that knows it)."""
    ev = None
    for tb in reversed(tables):
        ev = tb.get("evolved", {}).get(sp)
        if ev:
            break
    return format_species(sp, ev)


def _row_for(sp: str, table: dict) -> dict[str, float]:
    """Matchup row for a plain or branch-pinned key ('eevee>jolteon' falls back to 'eevee' on
    early bosses before the branch is legal)."""
    rows = table["rows"]
    if sp in rows:
        return rows[sp]
    base, _ = split_evo_key(sp)
    return rows.get(base, {})


def _in_flag(sp: str, flag_set: set[str], table: dict) -> bool:
    if sp in flag_set:
        return True
    base, _ = split_evo_key(sp)
    return sp not in table["rows"] and base in flag_set


def box_boss_value(box_species: set[str] | list[str], table: dict) -> float:
    """Scalar value of `box_species` against one boss: sum over enemy mons of
    (best + 0.45 * 2nd + 0.20 * 3rd best) + arena-negation bonus."""
    enemies = table["enemies"]
    total = 0.0
    for e in enemies:
        b1 = b2 = b3 = 0.0
        for sp in box_species:
            v = _row_for(sp, table).get(e, 0.0)
            if v > b1:
                b1, b2, b3 = v, b1, b2
            elif v > b2:
                b2, b3 = v, b2
            elif v > b3:
                b3 = v
        total += b1 + SECOND_WEIGHT * b2 + THIRD_WEIGHT * b3
    n_neg = sum(1 for sp in box_species if _in_flag(sp, table["negate"], table))
    if n_neg >= 1:
        total += NEGATE_BONUS * (1.0 + (SECOND_WEIGHT if n_neg >= 2 else 0.0))
    return total


def adaptive_boss_weights(box_species: set[str] | list[str], tables: list[dict]) -> dict[str, float]:
    """Higher weight on bosses where the current box still has weak / uncovered enemy threats."""
    out = {}
    for tb in tables:
        n_e = max(len(tb["enemies"]), 1)
        # Max possible per-enemy score is roughly 3.1 * (1 + SECOND_WEIGHT) ~ 4.2
        cur = box_boss_value(box_species, tb) / n_e
        out[tb["boss"]] = 1.0 / (0.6 + cur)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def marginal_gain(sp: str, box_species: set[str], tables: list[dict], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Expected weighted gain of adding `sp` to `box_species` across `tables`, plus per-boss breakdown."""
    if sp in box_species:
        return 0.0, {}
    per = {}
    total = 0.0
    with_sp = box_species | {sp}
    for tb in tables:
        if sp not in tb["rows"]:
            continue
        dv = box_boss_value(with_sp, tb) - box_boss_value(box_species, tb)
        if dv > 1e-9:
            per[tb["boss"]] = dv
            total += weights[tb["boss"]] * dv
    return total, per


def expected_table_gain(
    entries: list[tuple[str, float]],
    box_species: set[str],
    tables: list[dict],
    weights: dict[str, float],
) -> tuple[float, list[tuple[str, float, float]]]:
    """Dupes-filtered expected marginal gain of a rarity table [(species, weight)].
    Returns (expected_gain, top_hits [(species, prob, gain)])."""
    avail = [(sp, w) for sp, w in entries if sp not in box_species and any(sp in tb["rows"] for tb in tables)]
    if not avail:
        return 0.0, []
    # Aggregate duplicate species in the same table
    agg: dict[str, float] = defaultdict(float)
    for sp, w in avail:
        agg[sp] += max(w, 0.0)
    tot_w = sum(agg.values())
    if tot_w <= 0:
        return 0.0, []
    exp_gain = 0.0
    hits = []
    for sp, w in agg.items():
        p = w / tot_w
        g, _ = marginal_gain(sp, box_species, tables, weights)
        exp_gain += p * g
        hits.append((sp, p, g))
    hits.sort(key=lambda x: -(x[1] * x[2]))
    return exp_gain, hits[:3]


def resolve_box_branches(box_species: list[str], tables: list[dict]) -> list[str]:
    """Replace unpinned branched species in `box_species` (including multi-copy gifts like 2 Eevees)
    with their top recommended `'base>evolved'` branch(es) across `tables`."""
    recs = {r["base"]: r for r in advise_evolutions(box_species, tables)}
    out: list[str] = []
    used: dict[str, int] = defaultdict(int)
    for sp in box_species:
        base, forced = split_evo_key(sp)
        if forced is not None or base not in recs:
            out.append(sp)
            continue
        r = recs[base]
        idx = used[base]
        used[base] += 1
        if r["copies"] >= 2 and r["best_pair"] and idx < 2:
            out.append(f"{base}>{r['best_pair']['evolved'][idx]}")
        elif idx < len(r["ranked"]):
            out.append(r["ranked"][idx]["key"])
        else:
            out.append(sp)
    return out


def death_costs(
    box_species: list[str],
    future_tables: list[dict],
    base_death_scale: float = 0.5 / 6,
) -> list[dict]:
    """D14: Per-mon future death cost over `future_tables`.
    Resolves unpinned multi-copy branched species (e.g. 2 Eevees) into their recommended
    distinct branches before computing leave-one-out drops.
    Returns one dict per mon sorted from highest cost (irreplaceable) to lowest (expendable):
      species, raw_cost, d14_weight, hurts [(boss, delta_v, threats_lost)]
    """
    resolved = resolve_box_branches(box_species, future_tables)
    box_set = set(resolved)
    weights = adaptive_boss_weights(box_set, future_tables) if future_tables else {}
    raw_rows = []
    for sp in list(dict.fromkeys(resolved)):
        without = box_set - {sp}
        raw = 0.0
        hurts = []
        for tb in future_tables:
            v_with = box_boss_value(box_set, tb)
            v_without = box_boss_value(without, tb)
            drop = v_with - v_without
            if drop > 0.05:
                # Identify which enemy mons lost their top answer
                lost = []
                for e in tb["enemies"]:
                    top_with = max((_row_for(s, tb).get(e, 0.0) for s in box_set), default=0.0)
                    top_without = max((_row_for(s, tb).get(e, 0.0) for s in without), default=0.0)
                    if top_with - top_without > 0.2:
                        lost.append(e)
                if _in_flag(sp, tb["negate"], tb) and not any(_in_flag(s, tb["negate"], tb) for s in without):
                    lost.append("arena-negate")
                hurts.append((tb["boss"], round(drop, 2), lost))
                raw += weights[tb["boss"]] * drop
        raw_rows.append({"species": sp, "raw_cost": raw, "hurts": hurts})

    mean_raw = sum(r["raw_cost"] for r in raw_rows) / max(len(raw_rows), 1)
    for r in raw_rows:
        if mean_raw > 1e-9:
            # Normalize so mean weight matches D1's 0.5/6, capped at [0, 3x]
            r["d14_weight"] = min(3.0 * base_death_scale, base_death_scale * (r["raw_cost"] / mean_raw))
        else:
            r["d14_weight"] = base_death_scale
    raw_rows.sort(key=lambda r: -r["raw_cost"])
    return raw_rows


def routed_box(
    seed_key: str | random.Random,
    t: int,
    known: dict,
    tables: list[dict],
    starter: str | None = None,
    route_methods: bool = True,
    include_celadon_egg: bool = False,
    include_choices: bool = False,
    allow_trades: bool = True,
) -> list[tuple[str, str]]:
    """Simulate a run's box up to trainer `t`, using independent per-location RNGs (so random,
    routed, and routed+egg stay tightly paired on every shared route/method).
    Run rules:
      - Starter: chosen freely out of all 27 when `starter` is given, else random.
      - Story gifts: accepted (including 2nd Eevee gift at Celadon 4F; Saffron Dojo chooses
        between Tauros-Paldea-Blaze and Tauros-Paldea-Aqua when `include_choices=True`).
      - In-game trades: disabled unless the box actually owns a member of the `give` line;
        when owned and `allow_trades=True`, executed iff trading improves future-boss value
        (removing the traded-away mon).
      - Celadon Starter Egg (`include_celadon_egg=True`): 1 egg total, choose best shard pool,
        roll 1 random unowned mon from that shard pool."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))
    e = order["entries"][t]
    dex9 = GenData.from_gen(9).pokedex
    base_seed = seed_key if isinstance(seed_key, str) else str(seed_key.randrange(1 << 60))
    st = starter or random.Random(f"{base_seed}-starter").choice(rp.STARTERS)
    box = [(st, "starter")]
    have = {base_species(st)}

    # Accepted story gifts (not trades — trades require actually owning the `give` mon)
    for g in order.get("gifts", []):
        if g["t"] > t or (g["level"] or 0) > e["cap"]:
            continue
        kind = g.get("kind") or ("trade" if g.get("how", "").startswith("trade:") else "gift")
        if kind != "gift":
            continue
        if include_choices and g.get("choice") == "SAFFRON DOJO":
            continue  # handled by choice_encounters() below
        opts = [s for s in g.get("choice_options", [g["species"]]) if s in known and s in dex9]
        if not opts:
            continue
        sp = random.Random(f"{base_seed}-gift-{g.get('choice', g['species'])}").choice(opts) if len(opts) > 1 else opts[0]
        if sp not in have or sp == "eevee":
            have.add(sp)
            box.append((sp, g["how"]))

    # Causal schedule: first trainer index whose `reach` opens each location
    entries = order["entries"]
    locs = [x for x in enc["locations"] if not x["postgame"]][: e["reach"] + 1]
    loc_open_t = [next((idx for idx, en in enumerate(entries) if en["reach"] >= li), 0) for li in range(len(locs))]

    # Walk open wild locations in game order, routing causally over bosses still ahead when each route opens
    for li, loc in enumerate(locs):
        open_t = loc_open_t[li]
        ahead = [tb for tb in tables if tb["progress"] >= open_t] or tables
        methods = {
            m: [x for x in tb if int(x["levels"].split("-")[0]) <= e["cap"]
                and norm_sp(x["species"]) in known and norm_sp(x["species"]) in dex9]
            for m, tb in loc["methods"].items()
            if (0 if m.startswith("grass") else order["gates"].get(m, 999)) <= t
        }
        methods = {m: tb for m, tb in methods.items() if tb}
        if not methods:
            continue
        if route_methods:
            weights = adaptive_boss_weights(have, ahead)
            best_m = max(
                sorted(methods),
                key=lambda m: expected_table_gain(
                    [(norm_sp(x["species"]), float(x["rarity"].rstrip("%")) or 1.0) for x in methods[m]],
                    have, ahead, weights,
                )[0],
            )
            meth = best_m
        else:
            meth = random.Random(f"{base_seed}-meth-{loc['name']}").choice(sorted(methods))
        tb = methods[meth]
        roll_rng = random.Random(f"{base_seed}-roll-{loc['name']}-{meth}")
        pick = None
        for _ in range(4):
            cand = roll_rng.choices(tb, weights=[float(x["rarity"].rstrip("%")) or 1.0 for x in tb])[0]
            if norm_sp(cand["species"]) not in have:
                pick = cand
                break
        if pick is not None:
            sp = norm_sp(pick["species"])
            have.add(sp)
            box.append((sp, f"{loc['name']} / {meth} {pick['rarity']}"))

    # Optional choice encounters (1 Celadon Starter Egg total, fossils, Saffron Dojo) on their own RNG
    if include_celadon_egg or include_choices:
        for cname, ct, opts in choice_encounters():
            if ct > t:
                continue
            if cname == "CELADON EGG VENDOR" and not include_celadon_egg:
                continue
            if cname != "CELADON EGG VENDOR" and not include_choices:
                continue
            ahead_c = [tb for tb in tables if tb["progress"] >= ct] or tables
            weights = adaptive_boss_weights(have, ahead_c)
            best_opt, best_g = None, -1.0
            for label, sps in opts.items():
                valid = [(sp, 1.0) for sp in sps if sp in known and sp in dex9]
                g, _ = expected_table_gain(valid, have, ahead_c, weights)
                if g > best_g:
                    best_opt, best_g = label, g
            if best_opt is not None:
                pool_sps = [sp for sp in opts[best_opt] if sp in known and sp in dex9 and sp not in have]
                if pool_sps:
                    pick = random.Random(f"{base_seed}-choice-{cname}-{best_opt}").choice(pool_sps)
                    have.add(pick)
                    box.append((pick, f"{cname} / {best_opt}"))

    # In-game trades: ONLY eligible if the box actually owns a member of the `give` line.
    # Execute a trade iff swapping out the owned mon for `receive` strictly raises future-boss value.
    if allow_trades:
        for tr in rp.eligible_trades([sp for sp, _ in box], t, known):
            ahead_tr = [tb for tb in tables if tb["progress"] >= tr["t"]] or tables
            weights = adaptive_boss_weights(have, ahead_tr)
            without = have - {tr["owned_match"]}
            v_keep = sum(weights[tb["boss"]] * box_boss_value(have, tb) for tb in ahead_tr)
            v_trade = sum(weights[tb["boss"]] * box_boss_value(without | {tr["species"]}, tb) for tb in ahead_tr)
            if v_trade > v_keep + 1e-9:
                idx = next((k for k, (sp, _) in enumerate(box) if base_species(sp) == tr["owned_match"]), None)
                if idx is not None:
                    box.pop(idx)
                    have.discard(tr["owned_match"])
                    have.add(tr["species"])
                    box.append((tr["species"], tr["how"]))
    return box


def rank_starters(tables: list[dict]) -> list[dict]:
    """Rank all 27 starters across `tables`, evaluating every legal committed evolution branch
    (e.g. Rowlet -> Decidueye vs Decidueye-Hisui, Oshawott -> Samurott vs Samurott-Hisui,
    Cyndaquil -> Typhlosion vs Typhlosion-Hisui) and Mega Stones once obtainable."""
    weights = {tb["boss"]: 1.0 / len(tables) for tb in tables}
    out = []
    for sp in rp.STARTERS:
        # Collect all committed branch keys observed for this starter across tables
        bkeys: list[str] = []
        for tb in tables:
            for bk in tb.get("branches", {}).get(sp, []):
                if bk not in bkeys:
                    bkeys.append(bk)
        cands = bkeys or [sp]
        best_rec = None
        for cand in cands:
            per = {}
            total = 0.0
            megas = []
            for tb in tables:
                v = box_boss_value({cand}, tb)
                per[tb["boss"]] = round(v, 2)
                total += weights[tb["boss"]] * v
                if _in_flag(cand, tb["mega"], tb):
                    megas.append(tb["boss"])
            rec = {"starter": sp, "key": cand, "label": label_sp(cand, tables), "score": total, "per_boss": per, "mega_at": megas}
            if best_rec is None or rec["score"] > best_rec["score"]:
                best_rec = rec
        out.append(best_rec)
    out.sort(key=lambda r: -r["score"])
    return out


def advise_evolutions(box_species: list[str], tables: list[dict]) -> list[dict]:
    """For every unpinned branched species in `box_species` (e.g. Eevee x1 or x2, Charcadet,
    Kirlia, Slowpoke, Scyther, Kubfu, Hisuian starters), rank its committed evolution branches
    across upcoming `tables` (or the best distinct pair when owning 2 copies, like 2 Eevees)."""
    from collections import Counter
    from itertools import combinations
    counts = Counter(base_species(s) for s in box_species)
    out = []
    for base, cnt in sorted(counts.items()):
        # Gather all distinct legal branches across upcoming tables
        bkeys: list[str] = []
        for tb in tables:
            for bk in tb.get("branches", {}).get(base, []):
                if bk not in bkeys:
                    bkeys.append(bk)
        if len(bkeys) < 2:
            continue
        rest = {s for s in box_species if base_species(s) != base}
        weights = adaptive_boss_weights(rest, tables) if tables else {}
        scored = []
        for bk in bkeys:
            per = {}
            tot = 0.0
            for tb in tables:
                dv = box_boss_value(rest | {bk}, tb) - box_boss_value(rest, tb)
                per[tb["boss"]] = round(dv, 2)
                tot += weights[tb["boss"]] * dv
            scored.append({"key": bk, "evolved": split_evo_key(bk)[1], "gain": tot, "per_boss": per})
        scored.sort(key=lambda r: -r["gain"])
        best_pair = None
        if cnt >= 2:
            pairs = []
            for a, b in combinations(bkeys, 2):
                tot = sum(weights[tb["boss"]] * (box_boss_value(rest | {a, b}, tb) - box_boss_value(rest, tb)) for tb in tables)
                pairs.append((tot, split_evo_key(a)[1], split_evo_key(b)[1]))
            pairs.sort(reverse=True)
            if pairs:
                best_pair = {"evolved": [pairs[0][1], pairs[0][2]], "gain": pairs[0][0]}
        out.append({"base": base, "copies": cnt, "ranked": scored, "best_pair": best_pair})
    return out


def rank_celadon_eggs(box_species: set[str], tables: list[dict]) -> list[dict]:
    """Rank the 4 Celadon Mansion Shard Eggs (pre-Erika, t=33; 1 egg total, choose shard pool)
    by expected marginal gain over `tables`."""
    te = json.load(open(RR / "tracker_encounters.json"))
    eggs = te.get("eggVendor", {})
    weights = adaptive_boss_weights(box_species, tables)
    out = []
    for shard, sps in eggs.items():
        ids = [norm_sp(s) for s in sps]
        g, hits = expected_table_gain([(s, 1.0) for s in ids], box_species, tables, weights)
        out.append({"shard": shard, "expected_gain": g, "n_unowned": sum(s not in box_species for s in ids), "top_hits": hits})
    out.sort(key=lambda r: -r["expected_gain"])
    return out


def advise_next_catches(box_species: set[str], t: int, tables: list[dict], top_k: int = 12) -> list[dict]:
    """Rank open locations (and choice gifts) by expected marginal gain for `box_species` across `tables`."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))
    e = order["entries"][t]
    known = load_game("rr")["species"]
    dex9 = GenData.from_gen(9).pokedex
    weights = adaptive_boss_weights(box_species, tables)
    recs = []
    for loc in [x for x in enc["locations"] if not x["postgame"]][: e["reach"] + 1]:
        best = None
        for m, tb in loc["methods"].items():
            if (0 if m.startswith("grass") else order["gates"].get(m, 999)) > t:
                continue
            entries = [
                (norm_sp(x["species"]), float(x["rarity"].rstrip("%")) or 1.0)
                for x in tb
                if int(x["levels"].split("-")[0]) <= e["cap"] and norm_sp(x["species"]) in known and norm_sp(x["species"]) in dex9
            ]
            g, hits = expected_table_gain(entries, box_species, tables, weights)
            if best is None or g > best["expected_gain"]:
                best = {"where": loc["name"], "method": m, "expected_gain": g, "top_hits": hits}
        if best and best["expected_gain"] > 1e-6:
            recs.append(best)
    for cname, ct, opts in choice_encounters():
        if ct > t:
            continue
        best = None
        for label, sps in opts.items():
            g, hits = expected_table_gain([(s, 1.0) for s in sps if s in known and s in dex9], box_species, tables, weights)
            if best is None or g > best["expected_gain"]:
                best = {"where": cname, "method": label, "expected_gain": g, "top_hits": hits}
        if best and best["expected_gain"] > 1e-6:
            recs.append(best)
    recs.sort(key=lambda r: -r["expected_gain"])
    return recs[:top_k]


def _duel_pick_team(ctx, species: list[str], duel_cells: dict, enemy_order: list[str],
                    force_base: str | None = None) -> tuple[str, ...]:
    """Build a 6-mon team from `species` using cached Showdown duel cells + branch expansion.
    If `force_base` is set (e.g. testing a starter in 6v6), includes the starter's best legal
    branch first, then fills the remaining 5 slots with `cover()` on uncovered enemy threats."""
    from ai_util.matchup import cover, edge, race, safe
    exp = tp.expand_box(ctx, species)
    sub_rows = {k: duel_cells[k] for k in exp if k in duel_cells}
    enemies = enemy_order or [to_id_str(x.split("|")[0]) for x in ctx.enemy_packed.split("]")]
    if not sub_rows:
        return tp.choose(ctx, species)
    limits = dict(getattr(ctx, "slot_limits", {}) or {})
    picked: list[str] = []
    rem_enemies = list(enemies)
    if force_base is not None:
        st_cands = [k for k in sub_rows if base_species(k) == force_base]
        if st_cands:
            best_st = max(st_cands, key=lambda c: (
                sum(safe(sub_rows[c][e]) for e in enemies if e in sub_rows[c]),
                sum(race(sub_rows[c][e]) for e in enemies if e in sub_rows[c]),
                sum(edge(sub_rows[c][e]) for e in enemies if e in sub_rows[c]),
            ))
            picked.append(best_st)
            rem_enemies = [e for e in enemies if not (e in sub_rows[best_st] and race(sub_rows[best_st][e]))] or list(enemies)
            limits[force_base] = max(0, limits.get(force_base, 1) - 1)
    rem_rows = {k: v for k, v in sub_rows.items() if k not in picked and limits.get(base_species(k), 1) > 0}
    if len(picked) < 6 and rem_rows:
        picked += cover(rem_rows, rem_enemies, 6 - len(picked), limits=limits, group_fn=base_species)
    lead_e = enemies[0]
    lead = max(sorted(picked), key=lambda c: (
        safe(sub_rows[c][lead_e]), race(sub_rows[c][lead_e]), edge(sub_rows[c][lead_e])
    ) if lead_e in sub_rows.get(c, {}) else (False, False, -9.0))
    return (lead,) + tuple(sorted(c for c in picked if c != lead))


def _eval_6v6_team_job(args, boss: str, box_species: list[str], duel_cells: dict, enemy_order: list[str],
                       force_base: str | None) -> dict:
    """Play `args.n_search` paired 6v6 MCTS battles for one (boss, box, optional starter) team."""
    import logging
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    ctx = tp.worker_ctx(args, boss)
    species = [sp for sp in box_species if tp.mon_of(ctx, sp) is not None]
    team = _duel_pick_team(ctx, species, duel_cells, enemy_order, force_base=force_base)
    n = args.n_search
    bs = [tp.battle_job(args, boss, team, j) for j in range(n)]
    w, d, _, _, cl = tp.fold(bs)
    u = sum(rs.score(b, False, len(team)) for b in bs) / n
    return {"win": w, "deaths_per_mon": d, "deathless": cl, "U": u, "team": tp.evolved_names(ctx, team)}


def rank_starters_6v6(args, bosses: list[str], tables_by_boss: dict[str, dict]) -> list[dict]:
    """True 6v6 MCTS starter ranking across all 27 starters via 3-round sequential halving.
    For every (boss, box_i), evaluates a starter-less routed base box ONCE (`U_base`), then
    each surviving starter on the EXACT same paired box & battle seeds (`U_with`).
    Per-boss lift = `max(0, mean_boxes[U_with - U_base])`, capturing true 6v6 synergy
    (Battle Bond sweeps, Unaware, Intimidate, Megas, Hisuian branches) while allowing
    benching a starter on gyms where it doesn't help."""
    known = load_game("rr")["species"]
    by_boss, _ = rp.pools()
    all_tables = [tables_by_boss[b] for b in bosses]

    # 3-round halving schedule: (n_starters_keep_next, n_boxes, n_battles, mcts_iters)
    rounds = [
        ("R1 (27->14)", 14, 6, 2, 4000),
        ("R2 (14->7)", 7, 10, 4, 10000),
        ("R3 (final 7)", 7, 16, 6, 20000),
    ]
    if args.fast_6v6:
        rounds = [
            ("R1 (27->10)", 10, 4, 2, 3000),
            ("R2 (final 10)", 10, 8, 4, 10000),
        ]

    # Pre-generate strictly paired starter-less base boxes for every (boss, box_i)
    max_boxes = max(r[2] for r in rounds)
    base_boxes: dict[tuple[str, int], list[str]] = {}
    for boss in bosses:
        t = by_boss[boss][3]
        for i in range(max_boxes):
            # Generate routed box with a non-species sentinel starter, then strip slot 0 so all
            # 27 starters share identical wild catches + ALL accepted story gifts (incl. Magikarp)
            b = routed_box(f"s6v6-{boss}-{args.seed}-{i}", t, known, all_tables, starter="__none__",
                           route_methods=True, include_celadon_egg=False, include_choices=False)
            base_boxes[boss, i] = [sp for sp, wh in b if wh != "starter"]

    alive = list(rp.STARTERS)
    history: dict[str, dict] = {}
    t_start = time.time()

    for rname, keep, n_boxes, n_bat, iters in rounds:
        rt0 = time.time()
        probe_args = _base_args(seed=args.seed, workers=args.workers, engine=iters, n_search=n_bat, normal=args.normal)
        # 1. Evaluate starter-less baseline once per (boss, box_i)
        base_jobs = [
            (probe_args, boss, base_boxes[boss, i], tables_by_boss[boss].get("duel_cells", {}),
             tables_by_boss[boss].get("enemy_order", []), None)
            for boss in bosses for i in range(n_boxes)
        ]
        # 2. Evaluate every surviving starter on those exact same boxes & seeds
        st_jobs = [
            (probe_args, boss, [st] + [sp for sp in base_boxes[boss, i] if sp != st],
             tables_by_boss[boss].get("duel_cells", {}), tables_by_boss[boss].get("enemy_order", []), st)
            for st in alive for boss in bosses for i in range(n_boxes)
        ]
        all_jobs = base_jobs + st_jobs
        if args.workers > 1:
            with mp.get_context("spawn").Pool(min(args.workers, len(all_jobs))) as pool:
                all_res = pool.starmap(_eval_6v6_team_job, all_jobs)
        else:
            all_res = [_eval_6v6_team_job(*j) for j in all_jobs]

        base_res = {}
        idx = 0
        for boss in bosses:
            for i in range(n_boxes):
                base_res[boss, i] = all_res[idx]
                idx += 1

        ranked = []
        for st in alive:
            per_boss_lift, per_boss_raw, per_boss_win = {}, {}, {}
            for boss in bosses:
                d_u, d_w = [], []
                for i in range(n_boxes):
                    r = all_res[idx]
                    idx += 1
                    b0 = base_res[boss, i]
                    d_u.append(r["U"] - b0["U"])
                    d_w.append(r["win"] - b0["win"])
                raw_u = sum(d_u) / len(d_u)
                raw_w = sum(d_w) / len(d_w)
                per_boss_raw[boss] = raw_u
                per_boss_lift[boss] = max(0.0, raw_u)
                per_boss_win[boss] = max(0.0, raw_w)
            mean_lift = sum(per_boss_lift.values()) / len(bosses)
            mean_raw = sum(per_boss_raw.values()) / len(bosses)
            mean_wlift = sum(per_boss_win.values()) / len(bosses)
            rec = {
                "starter": st, "label": label_sp(st, all_tables), "round": rname,
                "lift_U": mean_lift, "raw_U": mean_raw, "lift_win": mean_wlift,
                "per_boss": per_boss_lift, "per_boss_raw": per_boss_raw,
            }
            ranked.append(rec)
            history[st] = rec

        ranked.sort(key=lambda r: (-r["lift_U"], -r["raw_U"]))
        print(f"\n=== 6v6 STARTER HALVING {rname} ({n_boxes} boxes x {n_bat} battles @ {iters} iters, {time.time()-rt0:.0f}s) ===", flush=True)
        hdr = "  " + f"{'starter':<26} {'liftU':>6} {'liftW':>6}  " + " ".join(f"{b[:6]:>6}" for b in bosses)
        print(hdr, flush=True)
        for r in ranked:
            cols = " ".join(f"{r['per_boss'][b]:+6.3f}" for b in bosses)
            print(f"  {r['label']:<26} {r['lift_U']:+6.3f} {r['lift_win']:+5.1%}  {cols}", flush=True)
        alive = [r["starter"] for r in ranked[:keep]]

    final_ranked = sorted(history.values(), key=lambda r: (r["round"] != rounds[-1][0], -r["lift_U"], -r["raw_U"]))
    print(f"\n6v6 starter ranking finished in {(time.time() - t_start)/60:.1f} min.", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(final_ranked, indent=2))
    return final_ranked


def _eval_box_job(args, boss: str, i: int, box: list[tuple[str, str]], duel_cells: dict | None = None,
                  enemy_order: list[str] | None = None) -> dict:
    """Evaluate one pre-built box on one boss using duel-judged cover (when `duel_cells` is
    provided) or paper cover (with branch expansion) + default Phase 4 R2 battler."""
    ctx = tp.worker_ctx(args, boss)
    species = [sp for sp, _ in box if tp.mon_of(ctx, sp) is not None]
    if duel_cells:
        team = _duel_pick_team(ctx, species, duel_cells, enemy_order or [])
    else:
        team = tp.choose(ctx, species)
    w, d, _, _, cl = tp.searched(ctx, team)
    return {"win": w, "deaths_per_mon": d, "deathless": cl, "team": tp.evolved_names(ctx, team), "raw_team": list(team)}


def run_benchmark(args, bosses: list[str], tables_by_boss: dict[str, dict]) -> None:
    """Phase 6 gate: tightly paired 30-box comparison of random vs routed vs routed+egg vs routed+starter+egg."""
    known = load_game("rr")["species"]
    by_boss, _ = rp.pools()
    all_tables = [tables_by_boss[b] for b in bosses]
    top_rec = rank_starters(all_tables)[0]
    top_starter = top_rec["starter"]
    print(f"Top routed starter across {bosses} (scorer={args.scorer}, chooser={args.bench_chooser}): {top_rec['label']}", flush=True)
    variants = [
        ("random", False, False, False),
        ("routed", True, False, False),
        ("routed+egg", True, True, False),
        ("routed+full", True, True, True),
    ]
    probe_args = _base_args(seed=args.seed, workers=args.workers, engine=args.engine, n_search=args.n_search, normal=args.normal)
    summary = {v[0]: {} for v in variants}

    for boss in bosses:
        t = by_boss[boss][3]
        tb_b = tables_by_boss[boss]
        dcells = tb_b.get("duel_cells") if args.bench_chooser == "duel" and tb_b.get("duel_cells") else None
        eorder = tb_b.get("enemy_order")
        starters = [random.Random(f"starter-{boss}-{args.seed}-{i}").choice(rp.STARTERS) for i in range(args.boxes)]
        for vname, do_route, do_egg, pick_starter in variants:
            boxes = []
            for i in range(args.boxes):
                st = top_starter if pick_starter else starters[i]
                b = routed_box(f"box-{boss}-{args.seed}-{i}", t, known, all_tables, starter=st,
                               route_methods=do_route, include_celadon_egg=do_egg, include_choices=pick_starter)
                boxes.append(b)
            jobs = [(probe_args, boss, i, boxes[i], dcells, eorder) for i in range(args.boxes)]
            if args.workers > 1:
                with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
                    res = pool.starmap(_eval_box_job, jobs)
            else:
                res = [_eval_box_job(*j) for j in jobs]
            summary[vname][boss] = res
            mw = sum(r["win"] for r in res) / len(res)
            md = sum(r["deaths_per_mon"] for r in res) / len(res)
            ge50 = sum(r["win"] >= 0.5 for r in res) / len(res)
            print(f"{boss:<8} {vname:<12} ({args.boxes} boxes): win {mw:5.1%}  >=50% {ge50:4.0%}  d/mon {md:.2f}", flush=True)
            if args.out:
                ev_path = Path(args.out).with_name(f"{Path(args.out).stem}_{vname.replace('+', '_')}.events.jsonl")
                for i, r in enumerate(res):
                    with open(ev_path, "a") as f:
                        f.write(json.dumps({"kind": "box", "boss": boss, "i": i, "win": r["win"],
                                            "deaths_per_mon": r["deaths_per_mon"], "deathless": r.get("deathless", 0.0),
                                            "team": r["team"]}) + "\n")

    print("\n=== PAIRED SUMMARY VS RANDOM-METHOD BOXES ===", flush=True)
    for vname in ("routed", "routed+egg", "routed+full"):
        dw_all, dd_all = [], []
        for boss in bosses:
            base_r, cur_r = summary["random"][boss], summary[vname][boss]
            dw = [c["win"] - b["win"] for b, c in zip(base_r, cur_r)]
            dd = [c["deaths_per_mon"] - b["deaths_per_mon"] for b, c in zip(base_r, cur_r)]
            dw_all += dw
            dd_all += dd
            mw = sum(dw) / len(dw)
            sw = 1.96 * (sum((x - mw) ** 2 for x in dw) / max(len(dw) - 1, 1) / len(dw)) ** 0.5
            md = sum(dd) / len(dd)
            sd = 1.96 * (sum((x - md) ** 2 for x in dd) / max(len(dd) - 1, 1) / len(dd)) ** 0.5
            print(f"  {vname:<12} {boss:<8}: win {mw:+6.1%} ± {sw:5.1%}  |  d/mon {md:+.3f} ± {sd:.3f}")
        n = len(dw_all)
        mw = sum(dw_all) / n
        sw = 1.96 * (sum((x - mw) ** 2 for x in dw_all) / max(n - 1, 1) / n) ** 0.5
        md = sum(dd_all) / n
        sd = 1.96 * (sum((x - md) ** 2 for x in dd_all) / max(n - 1, 1) / n) ** 0.5
        print(f"  {vname:<12} OVERALL : win {mw:+6.1%} ± {sw:5.1%}  |  d/mon {md:+.3f} ± {sd:.3f}  (n={n})\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bosses", default="BROCK,MISTY,LT. SURGE,ERIKA,KOGA,BLAINE")
    ap.add_argument("--boss", default="ERIKA", help="current boss for --advise")
    ap.add_argument("--starters", action="store_true", help="rank all 27 starters across --bosses (1v1 duel/paper table)")
    ap.add_argument("--starters-6v6", action="store_true", help="rank all 27 starters by real 6v6 MCTS sequential halving")
    ap.add_argument("--fast-6v6", action="store_true", help="faster 2-round halving schedule for --starters-6v6")
    ap.add_argument("--celadon-eggs", action="store_true", help="rank the 4 Celadon Mansion Shard Eggs")
    ap.add_argument("--advise", action="store_true", help="print 'what to catch next' + D14 death costs for a box")
    ap.add_argument("--my-box", help="JSON file of your current box species (list of strings or dicts with 'species')")
    ap.add_argument("--benchmark", action="store_true", help="run Phase 6 gate: random vs routed vs routed+egg boxes")
    ap.add_argument("--scorer", choices=("duel", "paper"), default="duel",
                    help="routing cell scorer: real Showdown 1v1 duels (default, cached) or static paper matrix")
    ap.add_argument("--n-duel", type=int, default=2, help="Showdown duels per (species, enemy) cell when --scorer duel")
    ap.add_argument("--bench-chooser", choices=("duel", "cover"), default="duel",
                    help="team chooser in --benchmark: duel-judged cover (default) or paper cover")
    ap.add_argument("--boxes", type=int, default=30)
    ap.add_argument("--n-search", type=int, default=6)
    ap.add_argument("--engine", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", help="output prefix for benchmark events")
    ap.add_argument("--normal", action="store_true",
                    help="normal mode: its boss sets and caps, no hardcore bans or ability swaps (default: hardcore)")
    args = ap.parse_args()

    if not args.normal:
        rp.hardcore()
    bosses = [b.strip() for b in args.bosses.split(",") if b.strip()]
    t0 = time.time()
    extra = list(rp.STARTERS)
    for _, _, opts in choice_encounters():
        for sps in opts.values():
            extra += sps
    tables_by_boss = {
        b: build_boss_table(b, extra_species=extra, seed=args.seed, scorer=args.scorer, n_duel=args.n_duel, workers=args.workers, normal=args.normal)
        for b in bosses
    }
    tables = [tables_by_boss[b] for b in bosses]
    print(f"Built {args.scorer} routing tables for {len(bosses)} bosses in {time.time() - t0:.1f}s", flush=True)

    if args.starters:
        ranked = rank_starters(tables)
        print("\n=== STARTER RANKING (free choice of 27, with branched evos & Mega Stones) ===")
        hdr = "  " + f"{'starter':<26} {'mean':>6}  " + " ".join(f"{b[:6]:>6}" for b in bosses) + "  mega_at"
        print(hdr)
        for r in ranked:
            cols = " ".join(f"{r['per_boss'][b]:6.2f}" for b in bosses)
            m = ",".join(r["mega_at"]) if r["mega_at"] else "-"
            print(f"  {r['label']:<26} {r['score']:6.2f}  {cols}  {m}")

    if args.celadon_eggs:
        erika_and_after = [tb for tb in tables if tb["progress"] >= 33] or tables
        box_set = set()
        if args.my_box:
            raw = json.load(open(args.my_box))
            box_set = {norm_sp(x["species"] if isinstance(x, dict) else x) for x in raw}
        ranked = rank_celadon_eggs(box_set, erika_and_after)
        print("\n=== CELADON STARTER EGG (1 total pre-Erika, t=33 — choose shard pool) ===")
        for r in ranked:
            hits = ", ".join(f"{label_sp(sp, erika_and_after)} ({p:.0%}, +{g:.2f})" for sp, p, g in r["top_hits"])
            print(f"  {r['shard']:<13} E[gain]={r['expected_gain']:.3f} ({r['n_unowned']} unowned)  top hits: {hits}")

    if args.advise:
        by_boss, known = rp.pools()
        t = by_boss[args.boss][3]
        if args.my_box:
            raw = json.load(open(args.my_box))
            box_sps = [norm_sp(x["species"] if isinstance(x, dict) else x) for x in raw]
        else:
            box_sps = [sp for sp, _ in rp.sample_box(random.Random(args.seed), t, known)]
        future = [tb for tb in tables if tb["progress"] >= t]
        after = [tb for tb in tables if tb["progress"] > t]
        print(f"\n=== WHAT TO CATCH NEXT (at {args.boss}, box={len(box_sps)} mons) ===")
        for r in advise_next_catches(set(box_sps), t, future, top_k=12):
            hits = ", ".join(f"{label_sp(sp, future)} ({p:.0%}, +{g:.2f})" for sp, p, g in r["top_hits"])
            print(f"  {r['where']:<24} [{r['method']:<12}] E[gain]={r['expected_gain']:.3f}  top: {hits}")
        evo_recs = advise_evolutions(box_sps, future)
        if evo_recs:
            print(f"\n=== BRANCHED EVOLUTION CHOICES (across {[tb['boss'] for tb in future]}) ===")
            for er in evo_recs:
                top3 = ", ".join(f"{x['evolved']} (+{x['gain']:.2f})" for x in er["ranked"][:3])
                pair_s = f" | 2 copies -> {' + '.join(er['best_pair']['evolved'])} (+{er['best_pair']['gain']:.2f})" if er["best_pair"] else ""
                print(f"  {er['base']:<14} (x{er['copies']}): {top3}{pair_s}")
        elig_tr = rp.eligible_trades(box_sps, t, known)
        if elig_tr:
            print(f"\n=== ELIGIBLE IN-GAME TRADES (owned required mon) ===")
            for tr in elig_tr:
                print(f"  give {tr['owned_match']} -> receive {label_sp(tr['species'], future)} ({tr['how']})")
        if after:
            print(f"\n=== D14 PER-MON DEATH COSTS (future bosses: {[tb['boss'] for tb in after]}) ===")
            costs = death_costs(box_sps, after)
            for r in costs[:10]:
                why = "; ".join(f"{b} (-{d}: {','.join(ls[:3])})" for b, d, ls in r["hurts"]) or "redundant"
                print(f"  {label_sp(r['species'], after):<24} d14_w={r['d14_weight']:.3f} (raw={r['raw_cost']:.3f})  {why}")
            exp = [label_sp(r["species"], after) for r in costs if r["raw_cost"] < 1e-6]
            print(f"  Expendable (0 future cost, {len(exp)} mons): {', '.join(exp[:12])}")

    if args.starters_6v6:
        rank_starters_6v6(args, bosses, tables_by_boss)

    if args.benchmark:
        run_benchmark(args, bosses, tables_by_boss)


if __name__ == "__main__":
    main()
