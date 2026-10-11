"""Radical Red boss probe (step one of the RR question, 2026-10-10).

Can the battler win hardcore-mode boss fights at the level cap? For each boss:
sample a Nuzlocke-shaped box from the encounter pool available before it
(the tracker's docs export), field the six strongest after evolution at the
cap with best4 movesets from the gen9rr mod, and play the boss's documented
set (IVs 31, EVs assumed 252/252/4 from the nature — the hardcore docs carry
no spreads) under the Radical Red AI port, on the RadicalRedShowdown fork.
Player: flat 85 EVs, no items, ability slot 0/1, full enemy information.
Not modelled: the gyms' permanent field effects (favours the player), the
player's TMs / items / EV training unless --tms (an upper bound), doubles.

  uv run python scripts/rr_probe.py --bosses BROCK,MISTY --boxes 30 --n 2 --search 9,8 --workers 8
"""

import argparse
import json
import re
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import gauntlet  # noqa: E402

gauntlet.GAME = "rr"
from gauntlet import Mon, encounter_moves, evolve, load_game, pack_team  # noqa: E402
from poke_env.data import GenData, to_id_str  # noqa: E402


def inject_mod_data() -> dict:
    """poke-env's gen 9 tables gain the mod's own moves, species, items and abilities
    (data/rr/mod_data.json, dumped from the fork by scripts/build_rr.py), so the battle
    mirrors can hold what the engine plays (Dark Hole, Centiskorch-Sevii, ...).
    Entries poke-env already has keep poke-env's version. Returns the counts added."""
    mod = json.load(open(ROOT / "data" / "rr" / "mod_data.json"))
    g = GenData.from_gen(9)
    added = {}
    for table, mine in (("moves", g.moves), ("pokedex", g.pokedex)):
        src = mod["moves" if table == "moves" else "species"]
        new = {k: v for k, v in src.items() if k not in mine}
        mine.update(new)
        added[table] = len(new)
    return added


MOD_ADDED = inject_mod_data()

FMT = "gen9rrcustomgamenogimmicks"
FORK = ROOT / "pokemon-showdown-rr"
RR = ROOT / "data" / "rr"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
PLUS = {"Lonely": "atk", "Brave": "atk", "Adamant": "atk", "Naughty": "atk", "Bold": "def", "Relaxed": "def", "Impish": "def",
        "Lax": "def", "Timid": "spe", "Hasty": "spe", "Jolly": "spe", "Naive": "spe", "Modest": "spa", "Mild": "spa",
        "Quiet": "spa", "Rash": "spa", "Calm": "spd", "Gentle": "spd", "Sassy": "spd", "Careful": "spd"}
BOX = 10  # catches a run holds by a boss (one per area: ~8 areas by Brock, more later; capped for later bosses too)


def boss_evs(nature: str, moves: list[str], md: dict) -> dict:
    """The docs carry no hardcore spreads; assume a competitive 252/252/4 from the nature:
    speed nature -> Spe + the attacking stat the set uses; attacking nature -> that + Spe;
    defensive nature -> that + HP; neutral -> attacking stat + Spe."""
    cats = [md.get(to_id_str(m), {}).get("category", "Status") for m in moves]
    atk = "atk" if cats.count("Physical") >= cats.count("Special") else "spa"
    plus = PLUS.get(nature)
    ev = dict.fromkeys(STATS, 0)
    if plus == "spe":
        ev["spe"], ev[atk], ev["hp"] = 252, 252, 4
    elif plus in ("atk", "spa"):
        ev[plus], ev["spe"], ev["hp"] = 252, 252, 4
    elif plus in ("def", "spd"):
        ev[plus], ev["hp"], ev[atk] = 252, 252, 4
    else:
        ev[atk], ev["spe"], ev["hp"] = 252, 252, 4
    return ev


# first fights in the ROM dump (data/rr/rr41_hardcore_trainers.txt, ids): the docs export
# abbreviates move names ("Pow-Up Punch"), the dump's were fixed to Showdown's
DUMP_ID = {"BROCK": "0x19e", "MISTY": "0x19f", "LT. SURGE": "0x1a0", "ERIKA": "0x1a1", "SABRINA": "0x1a4",
           "KOGA": "0x1a2", "BLAINE": "0x1a3", "GIOVANNI": "0x45", "LORELEI": "0x4d", "BRUNO": "0x50",
           "AGATHA": "0x53", "LANCE": "0x56"}


def boss_team(name: str) -> tuple[str, int, list[str]]:
    """Packed team, max level and species for a boss's first fight, from the dump
    (data/rr/trainers_hardcore.json: parsed by scripts/build_rr.py, EVs assumed from the nature)."""
    dump = {t["id"]: t for t in json.load(open(RR / "trainers_hardcore.json"))}
    t = dump[DUMP_ID[name]]
    return t["packed"], t["cap"], [m["species"] for m in t["mons"]]


ARENA = {"DESOLATE LAND": ("weather", "desolateland"), "PRIMORDIAL SEA": ("weather", "primordialsea"),
         "SANDSTORM": ("weather", "sandstorm"), "RAIN": ("weather", "raindance"), "SUN": ("weather", "sunnyday"),
         "SNOW": ("weather", "snow"), "HAIL": ("weather", "hail"),
         "ELECTRIC TERRAIN": ("terrain", "electricterrain"), "GRASSY TERRAIN": ("terrain", "grassyterrain"),
         "MISTY TERRAIN": ("terrain", "mistyterrain"), "PSYCHIC TERRAIN": ("terrain", "psychicterrain"),
         "TRICK ROOM": ("pseudo", "trickroom"), "TAILWIND": ("p2", "tailwind")}


def arena(effect: str) -> tuple[dict, list[str]]:
    """The tracker's battleEffect ("PERMANENT SANDSTORM", "DOUBLES + PERMANENT TRICK ROOM") as
    the bridge's arena (weather / terrain / pseudo-weather / p2 side conditions, all permanent,
    plus the hardcore rule that p2's own weather and terrain never run out) and the parts the
    sim does not model (doubles, Brock's -33% on super-effective hits, Swamp, Magma Storm ...)."""
    out, rest = {"ai_permanent": True}, []
    for part in re.split(r"\s*[+&]\s*|\n", effect.upper()):
        part = part.replace("PERMANENT ", "").strip()
        hit = next((v for k, v in ARENA.items() if re.search(rf"\b{k}\b", part)), None)  # RAIN is not in TERRAIN
        if hit and hit[0] in ("pseudo", "p2"):
            out.setdefault(hit[0], []).append(hit[1])
        elif hit:
            out[hit[0]] = hit[1]
        elif part:
            rest.append(part)
    return out, rest


def pools() -> tuple[dict, dict]:
    """{boss name: (block, cap, [species ids catchable before it], trainer index)} from the
    tracker export and data/rr/order.json: every encounter at a location open by that fight
    (order.json `reach`) whose method is in hand (grass always; rods and Surf from `gates`)
    and whose lowest level is at or under the cap. Gifts, statics, fossils, trades and the
    starter are not in it; neither is rarity or the one-encounter-per-route rule."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    bosses = json.load(open(RR / "tracker_bosses.json"))["hardcore"]
    order = json.load(open(RR / "order.json"))
    locs = [l for l in enc["locations"] if not l["postgame"]]
    known = load_game("rr")["species"]
    dex9 = GenData.from_gen(9).pokedex
    # the first fight of each boss: leaders from "Kanto Leaders" (not "Kanto Rematch"), the League from
    # "Indigo League", rivals from "Rivals"; the first block of a title within its category
    by_title = {}
    for c in bosses["categories"]:
        if c["name"] not in ("Kanto Leaders", "Indigo League", "Rivals", "Team Rocket"):
            continue
        for x in c["bosses"]:
            by_title.setdefault(x["title"].split("·")[-1].strip(), x)
    out = {}
    for t, e in enumerate(order["entries"]):
        if e["name"] not in by_title or e["name"] in out or not e["cap"]:
            continue
        pool = set()
        for l in locs[:e["reach"] + 1]:
            for meth, tb in l["methods"].items():
                if (0 if meth.startswith("grass") else order["gates"].get(meth, 999)) > t:
                    continue
                pool |= {to_id_str(x["species"]) for x in tb if int(x["levels"].split("-")[0]) <= e["cap"]}
        out[e["name"]] = (by_title[e["name"]], e["cap"], sorted(s for s in pool if s in known and s in dex9), t)
    return out, known


def where(species: str, t: int) -> str:
    """The first open location and method (by trainer t) that offers `species` at or under
    the cap: the encounter-routing answer for a counter."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))
    e = order["entries"][t]
    for l in [x for x in enc["locations"] if not x["postgame"]][:e["reach"] + 1]:
        for meth, tb in l["methods"].items():
            if (0 if meth.startswith("grass") else order["gates"].get(meth, 999)) > t:
                continue
            for x in tb:
                if to_id_str(x["species"]) == species and int(x["levels"].split("-")[0]) <= e["cap"]:
                    return f"{l['name']} / {meth} {x['rarity']}"
    return "?"


def player_party(pool: list[str], cap: int, rng: random.Random, known: dict, progress: int | None) -> list[Mon]:
    """A run's box: BOX species drawn from the pool, each evolved by level at the cap;
    the party = the six with the highest base-stat total (the fast rule's pick)."""
    box = []
    for sp in rng.sample(pool, min(BOX, len(pool))):
        ev = evolve(sp, cap, rng, caught_level=5, trainer_idx=None)
        slots = known.get(ev, {}).get("abilities", {})
        ability = rng.choice([slots[k] for k in ("0", "1") if k in slots]) if slots else None
        moves = encounter_moves(ev, cap, "best4", progress=progress)  # None = level-up only
        if not moves:
            continue
        box.append(Mon(ev, cap, moves, ability=ability, caught_level=5))
    bst = lambda m: sum(known[m.species]["baseStats"].values())
    return sorted(box, key=bst, reverse=True)[:6]


def run_boss(args, name: str, tms: bool) -> dict:
    try:
        return _run_boss(args, name, tms)
    except Exception as e:  # the pool would only raise at the end, with every other job's output lost
        import traceback
        print(f"{name} tms={tms} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}", flush=True)
        return {"rows": []}


def _run_boss(args, name: str, tms: bool) -> dict:
    from ai_util.calc_bot import calc_brain
    from ai_util.evaluate import Searcher, evaluate
    from ai_util.rr_ai import RadicalRedAI
    from ai_util.sim import SimBridge, player_brain
    by_boss, known = pools()
    block, cap, pool, progress = by_boss[name]
    enemy_packed, dump_cap, enemy_species = boss_team(name)
    cap = max(cap, dump_cap)
    rng = random.Random(f"{name}-{args.seed}")
    parties = [player_party(pool, cap, rng, known, progress if tms else None) for _ in range(args.boxes)]
    bridge = SimBridge(showdown=str(FORK))
    enemy = player_brain(RadicalRedAI(battle_format=FMT, start_listening=False, log_level=50))
    calc = calc_brain()
    width = tuple(int(x) for x in args.search.split(",")) if args.search else None
    brains = {"calc": calc}
    if width:
        brains["calc+search"] = Searcher(None, enemy, k=width[0], rollouts=width[1], rollout=calc, seed=args.seed)
    rows = []
    t0 = time.time()
    for tag, brain in brains.items():
        wins, deaths, mons, turns, both = 0.0, 0.0, 0, 0.0, 0
        for party in parties:
            r = evaluate(bridge, pack_team(party), enemy_packed, brain, enemy, n=args.n, fmt=FMT, full_info=True)
            wins += r["p_win"]; deaths += sum(r["p_death"]); mons += len(r["p_death"]); turns += r["turns"]
            both += r["p_win"] == 1.0
            if r["stats"].get("rejected") or r["stats"].get("brain_default"):
                print(f"  {name} {tag}: choices {r['stats']}", flush=True)
        rows.append({"boss": name, "cap": cap, "tms": tms, "brain": tag, "boxes": len(parties), "n": args.n,
                     "win": wins / len(parties), "deaths_per_mon": deaths / max(mons, 1),
                     "boxes_always_win": both / len(parties), "turns": turns / len(parties),
                     "pool": len(pool), "sample_party": [m.species for m in parties[0]],
                     "enemy": enemy_species, "effect": block["battleEffect"]})
        print(f"{name:<10} cap {cap:<3} tms={tms!s:<5} {tag:<12} win {wins / len(parties):5.1%}  deaths/mon {deaths / max(mons, 1):.2f}  "
              f"always-win boxes {both / len(parties):4.0%}  turns {turns / len(parties):4.1f}  ({time.time() - t0:.0f}s)", flush=True)
    bridge.close()
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bosses", default="BROCK,MISTY,LT. SURGE,ERIKA")
    ap.add_argument("--boxes", type=int, default=30, help="sampled boxes (parties) per boss")
    ap.add_argument("--n", type=int, default=2, help="battles per party")
    ap.add_argument("--search", default="9,8", help="k,rollouts for the search brain ('' = calc alone only)")
    ap.add_argument("--tms", choices=["no", "yes", "both"], default="both", help="player movesets over every TM / tutor move too")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="jsonl of the summary rows")
    args = ap.parse_args()
    jobs = [(args, b.strip(), t) for b in args.bosses.split(",") for t in ({"no": [False], "yes": [True], "both": [False, True]}[args.tms])]
    t0 = time.time()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            parts = pool.starmap(run_boss, jobs)
    else:
        parts = [run_boss(*j) for j in jobs]
    rows = [r for p in parts for r in p["rows"]]
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nRadical Red boss probe | {args.boxes} boxes x n={args.n} per boss | search {args.search} | "
          f"enemy = RR AI port, full info, player flat EVs / no items | {(time.time() - t0) / 60:.0f} min")
    print(f"  {'boss':<10} {'cap':>3} {'TMs':<5} {'brain':<12} {'win':>6} {'deaths/mon':>10} {'always-win':>10}  enemy")
    for r in sorted(rows, key=lambda r: (r["cap"], r["tms"], r["brain"])):
        print(f"  {r['boss']:<10} {r['cap']:>3} {str(r['tms']):<5} {r['brain']:<12} {r['win']:6.1%} {r['deaths_per_mon']:10.2f} "
              f"{r['boxes_always_win']:10.0%}  {', '.join(r['enemy'])}  [{r['effect']}]")


if __name__ == "__main__":
    main()
