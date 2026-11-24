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
import functools
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
    """poke-env's gen 9 tables become the mod's (data/rr/mod_data.json, the fork's merged
    gen9rr Dex dumped by scripts/build_rr.py), so the battle mirrors hold what the engine
    plays: the mod's own moves and species (Dark Hole, Centiskorch-Sevii, ...) and its
    changes to existing ones (2026-10-12, matrix calibration: 40 species with other types,
    250 with other base stats, 134 moves with other power / type / accuracy; entries
    poke-env already had kept poke-env's vanilla version before, so Arbok was Poison, not
    Poison/Dark, in every damage calc). Returns {table: (added, replaced)}."""
    mod = json.load(open(ROOT / "data" / "rr" / "mod_data.json"))
    g = GenData.from_gen(9)
    counts = {}
    for table, mine in (("moves", g.moves), ("pokedex", g.pokedex)):
        src = mod["moves" if table == "moves" else "species"]
        counts[table] = (len(set(src) - set(mine)), len(set(src) & set(mine)))
        mine.update(src)
    return counts


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


# the ROM's 12-character names the docs export keeps ("Drain Kiss", "Soupercell Slam") -> the
# fork's ids; "None" and consumables are no held item; As One / Applite depend on the species
DUMP_NAMES = {"drainkiss": "drainingkiss", "soupercellslam": "supercellslam", "disarmcry": "disarmingvoice",
              "craftyguard": "craftyshield", "none": "", "xattack": "", "hyperpotion": "", "abilitypill": "",
              "blastoisnite": "blastoisinite", "houndoomnite": "houndoominite", "necroziumz": "ultranecroziumz",
              "rustysword": "rustedsword", "rustyshield": "rustedshield", "aerodactlite": "aerodactylite",
              "kangaskanite": "kangaskhanite", "neutralizegas": "neutralizinggas", "valiantshield": "dauntlessshield",
              "wanderingsoul": "wanderingspirit"}


def dump_id(name: str, species: str = "") -> str:
    x = to_id_str(name or "")
    if x == "asone":
        return "asonespectrier" if "shadow" in species.lower() else "asoneglastrier"
    if x == "applite":
        return "flapplite" if "flapple" in species.lower() else "appletunite"
    return DUMP_NAMES.get(x, x)


def pack_trainer(t: dict, cap: int | None) -> str:
    """A dump trainer's packed team; a mon with a relative level ("Max Level - n") gets cap + n."""
    spread = lambda e: ",".join(str(e[k]) for k in ("hp", "atk", "def", "spa", "spd", "spe"))
    parts = []
    for d in t["mons"]:
        lv = d["level"] if d["level"] is not None else (cap or 0) + (d.get("offset") or 0)
        parts.append("|".join([d["species"], "", dump_id(d["item"]), dump_id(d["ability"], d["species"]),
                               ",".join(dump_id(x) for x in d["moves"]), d["nature"], spread(d["evs"]), d["gender"],
                               spread(d["ivs"]), "", str(lv), ""]))
    return "]".join(parts)


def boss_team(name: str | int, sheet: bool = True) -> tuple[str, int, list[str]]:
    """Packed team, max level and species for a trainer-order entry (by name = first entry
    with that name, or by index) from the dump: the twelve big bosses by dump id, every
    other entry by matching the tracker block's species against dump trainers of that
    class name (the rival's variant = the one the tracker shows); relative levels resolve
    against the entry's cap. sheet=True (default since 2026-11-23): every mon the official sheet
    lists takes the sheet's level, item, ability, nature and moves (the dump reads abilities from
    slot 0 where the sheet has the hidden one: Brock's Varoom Overcoat vs Filter)."""
    dump = json.load(open(RR / "trainers_hardcore.json"))
    order = json.load(open(RR / "order.json"))["entries"]
    t_idx = name if isinstance(name, int) else next(i for i, e in enumerate(order) if e["name"] == name)
    entry = order[t_idx]
    if entry["name"] in DUMP_ID:
        tr = next(x for x in dump if x["id"] == DUMP_ID[entry["name"]])
    else:
        block = block_of(entry)
        want = {to_id_str(p["species"]) for p in block["pokemon"]} if block else set()
        last = to_id_str(entry["name"].replace("(REMATCH)", "").split("&")[0].split()[-1])
        cands = [x for x in dump if want and {to_id_str(m["species"]) for m in x["mons"]} == want] or \
                [x for x in dump if {"relli": "reli"}.get(last, last) in to_id_str(x["name"])]  # the tracker's "Relli" is the dump's "Reli"
        if not cands:
            raise KeyError(f"no dump trainer for #{t_idx} {entry['name']}")
        tr = max(cands, key=lambda x: (len(want & {to_id_str(m["species"]) for m in x["mons"]}), x["relative"]))
    cap = entry["cap"] if tr["relative"] else max(entry["cap"], tr["cap"] or 0)
    block = block_of(entry) if sheet else None
    if block:  # the official sheet is the truth (user, 2026-11-23); the dump keeps IVs, gender, unlisted mons
        by_sp = {tracker_id(p["species"], "species").removesuffix("mega"): p for p in block["pokemon"]}
        mons = []
        for d in tr["mons"]:
            p = by_sp.get(to_id_str(d["species"]).removesuffix("mega"))
            if p:
                d = {**d, "level": tracker_level(p["level"], entry["cap"]), "item": tracker_id(p["item"], "items"),
                     "ability": tracker_id(p["ability"], "abilities"), "nature": p["nature"],
                     "moves": [m for m in (tracker_id(x, "moves") for x in p["moves"]) if m]}
            mons.append(d)
        tr = {**tr, "mons": mons}
        cap = max([entry["cap"]] + [d["level"] for d in mons if d["level"] is not None])
    return pack_trainer(tr, entry["cap"]), cap, [m["species"] for m in tr["mons"]]


_MOD: dict = {}
TYPES = {"bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost", "grass", "ground", "ice",
         "poison", "psychic", "rock", "steel", "water"}
FORM_ALIASES = {"bm": "bloodmoon", "i": "", "c": "complete", "p": "primal", "t": "therian", "a": "alola", "g": "galar", "h": "hisui"}


def tracker_id(name: str, kind: str) -> str:
    """A tracker-export name -> the fork's id (kind: moves / items / abilities / species). The export
    abbreviates long names ("Headlon. Rush", "Weakness Pol.", "HP Ground", "Ursaluna-BM"): each of
    its words must start the full name's word. Unresolved names come back as their plain id."""
    if not _MOD:
        _MOD.update(json.load(open(RR / "mod_data.json")))
    x, table = to_id_str(name or ""), _MOD[kind]
    if x in ("", "none", "noitem") or x in table:
        return "" if x in ("none", "noitem") else x
    if kind == "moves" and x.startswith("hp") and x[2:] in TYPES:  # Showdown's alias, as the dump's names
        return "hiddenpower" + x[2:]
    words = [w for w in re.split(r"[ \-.]+", name.strip().lower()) if w]
    if kind == "species" and len(words) == 2 and words[1] in FORM_ALIASES:
        x = to_id_str(words[0] + FORM_ALIASES[words[1]])
        return x if x in table else to_id_str(words[0])
    for k, e in table.items():
        full = [w for w in re.split(r"[ \-]+", e["name"].lower()) if w]
        if len(full) == len(words) and all(f.startswith(w) for w, f in zip(words, full)):
            return k
    return dump_id(name)


def tracker_level(lv: str, cap: int) -> int:
    """A tracker level: "13", or relative to the cap ("Highest Lv -2", "Highest Lv", "Player Max Level")."""
    m = re.search(r"-\s*(\d+)$", lv)
    return int(lv) if lv.isdigit() else cap - (int(m.group(1)) if m else 0)


def normal_boss(name: str) -> tuple[dict, str, int, list[str]]:
    """Normal mode (the easier setting): the boss's first tracker block under `default` and its cap in
    the normal trainer order -> (block, packed team, cap, species). Levels "Highest Lv - n" resolve
    against the cap; the sets' own EVs (mostly none early), IVs 31."""
    d = json.load(open(RR / "tracker_bosses.json"))["default"]
    cap = next(int(e["levelCap"].rstrip("+").split("/")[0]) for e in d["trainerOrder"] if e["name"] == name)
    block = next(x for c in d["categories"] if c["name"] not in ("Kanto Rematch", "Postgame")
                 for x in c["bosses"] if x["title"].split("·")[-1].strip().upper() == name)
    parts, species = [], []
    for p in block["pokemon"]:
        lv = tracker_level(p["level"], cap)
        ev = {k.lower(): int(v) for k, v in p.get("evs", {}).items()}
        sp = tracker_id(p["species"], "species")
        species.append(sp)
        parts.append("|".join([sp, "", tracker_id(p["item"], "items"), tracker_id(p["ability"], "abilities"),
                               ",".join(tracker_id(m, "moves") for m in p["moves"]), p["nature"],
                               ",".join(str(ev.get(k, 0)) for k in STATS), "", ",".join(["31"] * 6), "", str(lv), ""]))
    return block, "]".join(parts), cap, species


def block_of(entry: dict) -> dict | None:
    """The tracker block for a trainer-order entry: its name's first block in a first-fight
    category (not rematches / post-game), preferring one whose title names the location."""
    bosses = json.load(open(RR / "tracker_bosses.json"))["hardcore"]
    name = entry["name"].replace("(REMATCH)", "").strip().upper()
    words = name.replace("&", " ").split()
    hits = []
    for c in bosses["categories"]:
        if c["name"] in ("Kanto Rematch", "Postgame"):
            continue
        for x in c["bosses"]:
            segs = [t.strip().upper() for t in x["title"].split("·")]  # "VIRIDIAN FOREST · LASS · ANNE", "GYM LEADER · BROCK"
            tail = " ".join(segs[1:]) if len(segs) > 1 else segs[0]
            if tail == name or (segs[-1] == words[-1] and (len(words) == 1 or words[0] in tail)):
                hits.append(x)
    loc = entry["location"].split()[0].upper()
    return next((x for x in hits if loc in x["title"].upper()), hits[0] if hits else None)


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


# every generation's starters: the run draws one at random (user decision 2026-10-11)
STARTERS = ["bulbasaur", "charmander", "squirtle", "chikorita", "cyndaquil", "totodile", "treecko", "torchic",
            "mudkip", "turtwig", "chimchar", "piplup", "snivy", "tepig", "oshawott", "chespin", "fennekin",
            "froakie", "rowlet", "litten", "popplio", "grookey", "scorbunny", "sobble", "sprigatito", "fuecoco", "quaxly"]
NATURE = {("atk", "spa"): "Adamant", ("spa", "atk"): "Modest", ("spe", "atk"): "Timid", ("spe", "spa"): "Jolly",
          ("atk", "def"): "Lonely", ("atk", "spd"): "Naughty", ("spa", "def"): "Mild", ("spa", "spd"): "Rash"}


def speed(base: int, level: int, iv: int = 31, ev: int = 0, mult: float = 1.0) -> int:
    return int(((2 * base + iv + ev // 4) * level // 100 + 5) * mult)


def boss_profile(enemy_packed: str, known: dict, arena: dict) -> dict:
    """What the nature choice reads off the boss: each mon's speed (doubled under its permanent
    Tailwind) and the share of its attacks that are physical."""
    from poke_env.data import GenData
    natures, md = GenData.from_gen(9).natures, load_game("rr")["moves"]
    tail = 2 if "tailwind" in arena.get("p2", []) else 1
    spe, cats = [], []
    for x in enemy_packed.split("]"):
        f = x.split("|")
        base = known.get(to_id_str(f[1] or f[0]), {}).get("baseStats", {}).get("spe", 0)
        ivs, evs = (f[8] or "31,31,31,31,31,31").split(","), (f[6] or "0,0,0,0,0,0").split(",")
        spe.append(speed(base, int(f[10] or 100), int(ivs[5] or 31), int(evs[5] or 0),
                         natures.get(to_id_str(f[5]), {}).get("spe", 1)) * tail)
        cats += [md[m]["category"] for m in f[4].split(",") if m in md and md[m]["category"] != "Status"]
    return {"spe": spe, "phys": sum(c == "Physical" for c in cats) / max(len(cats), 1)}


def choose_nature(species: str, moves: list[str], known: dict, boss: dict | None = None, level: int = 100) -> str:
    """Mints are free (hardcore sells them): + the attacking stat the set uses, - the other. A
    mixed set takes +Spe (dropping the weaker attacking stat) only when it outspeeds a boss mon
    that it would not outspeed neutral (`boss` = boss_profile); otherwise + the stronger
    attacking stat, - the defence the boss hits less (2026-11-24: +Spe on every mixed set made
    a Jolly base-30 Coalossal)."""
    md = load_game("rr")["moves"]
    cats = [md[m]["category"] for m in moves if m in md and md[m]["category"] != "Status"]
    bs = known.get(species, {}).get("baseStats", {"atk": 1, "spa": 1, "spe": 1})
    if "Physical" in cats and "Special" in cats:
        weak, strong = ("atk", "spa") if bs["atk"] < bs["spa"] else ("spa", "atk")
        mine = speed(bs.get("spe", 1), level)
        if boss is None or any(mine <= e < int(mine * 1.1) for e in boss["spe"]):
            return NATURE[("spe", weak)]
        return NATURE[(strong, "def" if boss["phys"] < 0.5 else "spd")]
    return NATURE[("atk", "spa")] if "Physical" in cats or not cats else NATURE[("spa", "atk")]


ABILITY_SWAPS: dict = {}  # hardcore only (hardcore()): ability id -> what the player's mon gets instead
SPECIES_SWAPS: dict = {}  # species id -> its own replacement (Tyranitar: Intimidate, not Sand Force)
KEEPS: dict = {}  # species id -> the ability it keeps despite the swap (Spinda: Contrary)


def hardcore() -> None:
    """The player's hardcore legality from data/rr/hardcore_rules.json: the 82 banned moves, Toxic /
    Leech Seed only on the 91 listed species (gauntlet.SPECIES_MOVES), and the ability swaps
    (Drizzle -> Adaptability, Speed Boost -> Infiltrator, ...; player_ability). Without it the
    player plays the easier rules: every move and ability the mod has."""
    r = json.load(open(RR / "hardcore_rules.json"))
    gauntlet.BANNED_MOVES = {to_id_str(m) for m in r["banned_moves"]}
    allowed = {to_id_str(s) for s in r["toxic_leech_seed_allowed_species"]}
    gauntlet.SPECIES_MOVES = {"toxic": allowed, "leechseed": allowed}
    # set rules (2026-11-24): Rest only with Sleep Talk, which best4 never pairs, so no Rest;
    # no Choice item on a set with a status move
    gauntlet.NO_SLOT_MOVES = gauntlet.NO_SLOT_MOVES | {"rest"}
    gauntlet.STATUS_MOVES = [m for m in gauntlet.STATUS_MOVES if m != "rest"]
    gauntlet.CHOICE_ALL_ATTACKS = True
    ABILITY_SWAPS.update({to_id_str(a): b for a, b in r["ability_changes"].items()})
    for name, ab in r["species_ability_changes"].items():  # "Mega Tyranitar" -> tyranitarmega
        SPECIES_SWAPS[to_id_str(name[5:]) + "mega" if name.startswith("Mega ") else to_id_str(name)] = ab
    for line in r["keeps"]:  # "Spinda Keeps Contrary"
        sp, ab = line.split(" Keeps ")
        KEEPS[to_id_str(sp)] = to_id_str(ab)


def player_ability(species: str, ability: str) -> str:
    """The ability the player's mon has under the hardcore swaps (unchanged when hardcore() is off)."""
    a = to_id_str(ability)
    if a not in ABILITY_SWAPS or KEEPS.get(species) == a:
        return ability
    return SPECIES_SWAPS.get(species, ABILITY_SWAPS[a])


def choose_ability(species: str, known: dict, boss_types: set, arena: dict, hidden: bool = True) -> str | None:
    """Capsules (and with `hidden` the Dream Patch) make the slots available: an immunity to a
    type the boss attacks with, a speed ability for the gym's weather / terrain, Cloud Nine /
    Air Lock against the gym's weather (2026-11-24, phase 3: Blaine's Desolate Land; it chose
    Golduck's Swift Swim before), Intimidate; else slot 0. Each slot is scored as what the player actually gets (player_ability)."""
    from ai_util.calc_bot import ABILITY_IMMUNE_TYPE, SPEED_ABILITY
    slots = {k: player_ability(species, v) for k, v in known.get(species, {}).get("abilities", {}).items()}
    if not slots:
        return None
    weather = {"raindance": "swiftswim", "sunnyday": "chlorophyll", "sandstorm": "sandrush", "snow": "slushrush",
               "desolateland": "chlorophyll", "primordialsea": "swiftswim"}.get(arena.get("weather", ""), "")

    def score(ab: str) -> int:
        a = to_id_str(ab)
        return (2 * (ABILITY_IMMUNE_TYPE.get(a) in boss_types) + 2 * (a == weather or (a == "surgesurfer" and arena.get("terrain") == "electricterrain"))
                + 2 * (a in ("cloudnine", "airlock") and bool(arena.get("weather"))) + (a == "intimidate") + (a in SPEED_ABILITY and SPEED_ABILITY[a] and False))
    return max((slots[k] for k in ("0", "1", "H")[:3 if hidden else 2] if k in slots), key=score)


@functools.cache
def ability_gates() -> tuple[int, int]:
    """(trainer index from which Ability Capsules are in hand, from which the Dream Patch is).
    The user's rule (2026-11-24, from the official sheets): capsules (slot 0 / 1) drop from raid
    dens, which need a badge and sit on Route 5 / 6 / 10 / 11, Diglett Cave, Rock Tunnel;
    the Saffron City Fanclub sells the Ability Pill and the Dream Patch (hidden ability)."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))["entries"]
    locs = [l["name"] for l in enc["locations"] if not l["postgame"]]
    dens = {l["location"].upper() for l in enc["raids"]["locations"]
            if any("Ability Capsule" in str(d.get("drops")) for d in l["dens"])}
    first_den = min(i for i, name in enumerate(locs) if any(name.startswith(d) for d in dens))
    brock = next(t for t, e in enumerate(order) if e["name"] == "BROCK")
    capsule = next(t for t, e in enumerate(order) if t > brock and e["reach"] >= first_den)
    patch = next(t for t, e in enumerate(order) if e["location"].startswith(("SAFFRON", "SILPH")))
    return capsule, patch


def run_ability(species: str, evolved: str, seed, progress: int, known: dict, boss_types: set, arena: dict) -> str | None:
    """The ability the player's mon has at trainer `progress`: rolled at catch before capsules
    are in hand, the best slot for this boss after, the hidden slot too once the patch is."""
    capsule, patch = ability_gates()
    if progress < capsule:
        return catch_ability(species, evolved, seed)
    return choose_ability(evolved, known, boss_types, arena, hidden=progress >= patch)


def catch_ability(species: str, evolved: str, seed) -> str | None:
    """Abilities are fixed at catch in the user's hardcore run (no capsules; user, 2026-11-24):
    slot 0 or 1 at even odds, rolled once per (run seed, caught species), kept by slot on
    evolution; what the player gets under the hardcore swaps (player_ability)."""
    ab = gauntlet.roll_ability(species, random.Random(f"{seed}-{species}-ability"))
    if ab is None:
        return None
    if evolved != species:
        ab = gauntlet.evolved_ability(species, ab, evolved)
    return player_ability(evolved, ab) if ab else None


def pools() -> tuple[dict, dict]:
    """{boss name or "#index": (block, cap, [species ids catchable before it], trainer index)} from the
    tracker export and data/rr/order.json: every encounter at a location open by that fight
    (order.json `reach`) whose method is in hand (grass always; rods and Surf from `gates`)
    and whose lowest level is at or under the cap, plus the gifts and in-game trades in hand
    (order.json `gifts`). The starter is drawn by the caller (STARTERS); statics, fossils,
    rarity and the one-encounter-per-route rule are not in it."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))
    locs = [l for l in enc["locations"] if not l["postgame"]]
    known = load_game("rr")["species"]
    dex9 = GenData.from_gen(9).pokedex
    # the first fight of each boss: leaders from "Kanto Leaders" (not "Kanto Rematch"), the League from
    # "Indigo League", rivals from "Rivals"; the first block of a title within its category
    out = {}
    for t, e in enumerate(order["entries"]):
        block = block_of(e)
        if (block is None and e["name"] not in DUMP_ID) or not e["cap"]:
            continue
        block = block or {"battleEffect": e.get("effect", ""), "pokemon": []}
        pool = {g["species"] for g in order.get("gifts", []) if g["t"] <= t and (g["level"] or 0) <= e["cap"]}
        for l in locs[:e["reach"] + 1]:
            for meth, tb in l["methods"].items():
                if (0 if meth.startswith("grass") else order["gates"].get(meth, 999)) > t:
                    continue
                pool |= {to_id_str(x["species"]) for x in tb if int(x["levels"].split("-")[0]) <= e["cap"]}
        row = (block, e["cap"], sorted(s for s in pool if s in known and s in dex9), t)
        out.setdefault(e["name"], row)
        out[f"#{t}"] = row
    return out, known


def sample_box(rng: random.Random, t: int, known: dict) -> list[tuple[str, str]]:
    """A run's box at trainer t (user decision 2026-10-11: random, one encounter per open
    location, drawn by the tracker's rarity): per location one method a run would use
    (grass by day or night, a rod or Surf once in hand, equal odds) and one species from it
    by rarity among those at or under the cap; a species already in the box is re-rolled
    up to three times (dupes clause); plus one random starter and the gifts / trades in
    hand. Returns [(species id, where)]."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))
    e = order["entries"][t]
    dex9 = GenData.from_gen(9).pokedex
    box = [(rng.choice(STARTERS), "starter")]
    have = {box[0][0]}
    for l in [x for x in enc["locations"] if not x["postgame"]][:e["reach"] + 1]:
        methods = {m: [x for x in tb if int(x["levels"].split("-")[0]) <= e["cap"]
                       and to_id_str(x["species"]) in known and to_id_str(x["species"]) in dex9]
                   for m, tb in l["methods"].items()
                   if (0 if m.startswith("grass") else order["gates"].get(m, 999)) <= t}
        methods = {m: tb for m, tb in methods.items() if tb}
        if not methods:
            continue
        for _ in range(4):
            meth = rng.choice(sorted(methods))
            tb = methods[meth]
            pick = rng.choices(tb, weights=[float(x["rarity"].rstrip("%")) or 1 for x in tb])[0]
            sp = to_id_str(pick["species"])
            if sp not in have:
                break
        if sp not in have:
            have.add(sp)
            box.append((sp, f"{l['name']} / {meth} {pick['rarity']}"))
    for g in order.get("gifts", []):
        if g["t"] <= t and (g["level"] or 0) <= e["cap"] and g["species"] in known and g["species"] not in have:
            have.add(g["species"])
            box.append((g["species"], g["how"]))
    return box


def where(species: str, t: int) -> str:
    """The first open location and method (by trainer t) that offers `species` at or under
    the cap: the encounter-routing answer for a counter."""
    if species in STARTERS:
        return "starter"
    enc = json.load(open(RR / "tracker_encounters.json"))
    order = json.load(open(RR / "order.json"))
    e = order["entries"][t]
    for g in order.get("gifts", []):
        if g["species"] == species and g["t"] <= t:
            return g["how"]
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
        brains["calc+search"] = Searcher(enemy, k=width[0], rollouts=width[1], rollout=calc, seed=args.seed)
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
