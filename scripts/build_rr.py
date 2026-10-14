"""data/rr/: the Radical Red game folder for the boss probe, from the tracker's
docs export (encounters, bosses, items) and the RadicalRedShowdown fork's
gen9rr mod (learnsets, evolutions, species, moves). Minimal: what
gauntlet.encounter_moves / evolve / the probe need. Not a full gauntlet yet
(no route trainers in order.json).

  uv run python scripts/build_rr.py            # needs pokemon-showdown-rr/ built
"""

import json
import re
import subprocess
from pathlib import Path

from poke_env.data import to_id_str

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "data" / "rr"
FORK = ROOT / "pokemon-showdown-rr"

NODE = """
const {Dex} = require(process.argv[1] + '/dist/sim');
const d = Dex.mod('gen9rr');
const want = new Set(process.argv.slice(2));
const evos = {}, queue = [...want];
while (queue.length) {                       // evolution closure, every method
  const sp = queue.pop();
  if (evos[sp]) continue;
  const s = d.species.get(sp);
  if (!s.exists) { evos[sp] = []; continue; }
  evos[sp] = s.evos.map(e => d.species.get(e)).filter(e => e.exists)
    .map(e => ({to: e.id, type: e.evoType || "level", level: e.evoLevel || null, item: e.evoItem || null, move: e.evoMove || null}));
  for (const e of evos[sp]) { want.add(e.to); queue.push(e.to); }
}
const learnsets = {}, species = {}, moves = {};
for (const sp of want) {
  const s = d.species.get(sp);
  if (!s.exists) continue;
  species[s.id] = {name: s.name, types: s.types, abilities: s.abilities, baseStats: s.baseStats};
  const ls = (d.data.Learnsets[s.id] || d.data.Learnsets[s.baseSpecies ? d.species.get(s.baseSpecies).id : s.id] || {}).learnset || {};
  const levelup = [], tm = [];
  for (const [m, srcs] of Object.entries(ls)) {
    const lv = srcs.filter(x => /^9L\\d+$/.test(x)).map(x => parseInt(x.slice(2)));
    if (lv.length) levelup.push([m, Math.min(...lv)]);
    if (srcs.some(x => x === "9M" || x === "9T")) tm.push(m);
  }
  levelup.sort((a, b) => a[1] - b[1]);
  learnsets[s.id] = {levelup, tm};
  for (const m of [...levelup.map(x => x[0]), ...tm]) {
    const mv = d.moves.get(m);
    moves[mv.id] = {type: mv.type, power: mv.basePower, category: mv.category, accuracy: mv.accuracy};
  }
}
console.log(JSON.stringify({learnsets, evos, species, moves}));
"""


STATS = ("hp", "atk", "def", "spa", "spd", "spe")
ITEM_ALIASES = {"Charzardite X": "Charizardite X", "Charzardite Y": "Charizardite Y"}  # typos in the dump


def trainer_evs(trainer: str) -> dict:
    """Hardcore auto-enables Minimal Grinding Mode: no trainer has EVs (official sheet, Main
    tab; its speed stats match EV 0 for 84/92 boss mons, 252 for none). The one exception the
    sheet names is the Ghost Marowak with 252 HP. (Until 2026-10-10 this guessed 252/252/4.)"""
    ev = dict.fromkeys(STATS, 0)
    if trainer.startswith("Ghost"):
        ev["hp"] = 252
    return ev


def parse_dump() -> None:
    """data/rr/trainers_hardcore.json from the ROM dump: every trainer with its mons
    (species, item, ability, level, nature, moves, IVs, EVs 0) and a packed team."""
    import re
    txt = open(RR / "rr41_hardcore_trainers.txt", encoding="utf-8").read()
    out = []
    for b in re.split(r"\n(?=\S.*\(id: 0x[0-9a-f]+\)\n)", txt):
        m = re.match(r"(.+?) \(id: (0x[0-9a-f]+)\)", b)
        if not m:
            continue
        mons = []
        for sec in re.split(r"\n\n", b.split("\n", 1)[1]):  # the header shares its paragraph with the lead
            lines = [l for l in sec.strip().splitlines() if l.strip()]
            hm = lines and re.match(r"(.+?)(?: \((M|F|Gender unknown)\))?(?: @ (.+))?$", lines[0])
            if not hm:
                continue
            item = (hm.group(3) or "").strip()
            d = {"species": hm.group(1).strip(), "gender": {"M": "M", "F": "F"}.get(hm.group(2) or "", ""),
                 "item": ITEM_ALIASES.get(item, item), "ability": "", "level": None, "nature": "Serious", "moves": [],
                 "ivs": dict.fromkeys(STATS, 31)}
            for l in lines[1:]:
                if l.startswith("Ability: "):
                    d["ability"] = l[9:].strip()
                elif l.startswith("Level: "):  # "17" or "Max Level - 2" (hardcore: the segment's cap minus n)
                    lv = l[7:].strip()
                    d["level"] = int(lv) if lv.isdigit() else None
                    d["offset"] = -int(lv.split("-")[1]) if lv.startswith("Max Level -") else (0 if lv == "Max Level" else None)
                elif l.endswith(" Nature"):
                    d["nature"] = l[:-7].strip()
                elif l.startswith("- "):
                    d["moves"].append(l[2:].strip())
                elif l.startswith("IVs: "):
                    for part in l[5:].split("/"):
                        v, k = part.strip().split(" ", 1)
                        d["ivs"][{"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}[k]] = int(v)
            if (d["level"] is None and d.get("offset") is None) or not d["moves"]:
                continue
            d["evs"] = trainer_evs(m[1])
            mons.append(d)
        if not mons:
            continue
        spread = lambda e: ",".join(str(e[k]) for k in STATS)
        fixed = [d["level"] for d in mons if d["level"] is not None]
        packed = "]".join("|".join([d["species"], "", to_id_str(d["item"]), to_id_str(d["ability"]),
                                    ",".join(to_id_str(x) for x in d["moves"]), d["nature"], spread(d["evs"]), d["gender"],
                                    spread(d["ivs"]), "", str(d["level"] or 0), ""]) for d in mons)  # 0 = relative: see rr_probe.pack_trainer
        out.append({"name": m[1].strip(), "id": m[2], "cap": max(fixed) if fixed else None, "relative": len(fixed) < len(mons),
                    "n": len(mons), "mons": mons, "packed": packed})
    (RR / "trainers_hardcore.json").write_text(json.dumps(out))
    print(f"data/rr/trainers_hardcore.json: {len(out)} trainers from the dump ({sum(t['relative'] for t in out)} with levels relative to the cap)")
    mod = json.loads(subprocess.run(["node", "-e", "const {Dex}=require(process.argv[1]+'/dist/sim');const d=Dex.mod('gen9rr');"
                                     "console.log(JSON.stringify({items:d.items.all().map(x=>x.id),moves:d.moves.all().map(x=>x.id),species:d.species.all().map(x=>x.id)}))",
                                     str(FORK)], capture_output=True, text=True, check=True).stdout)
    known = {k: set(v) for k, v in mod.items()}
    bad = {"items": set(), "moves": set(), "species": set()}
    for tr in out:
        for d in tr["mons"]:
            if d["item"] and to_id_str(d["item"]) not in known["items"]:
                bad["items"].add(d["item"])
            bad["moves"] |= {m for m in d["moves"] if to_id_str(m) not in known["moves"]}
            if to_id_str(d["species"]) not in known["species"]:
                bad["species"].add(d["species"])
    for k, v in bad.items():
        if v:
            print(f"  dump {k} unknown to the mod ({len(v)}): {sorted(v)[:15]}")

# --- progression: the tracker's trainer order as the gauntlet index, locations / items / TMs on it
ALIASES = [  # places the item sheets name that are not encounter locations -> the location reached with them
    (r"SAFARI.*(WEST|FAR WEST)", "SAFARI WEST (ZONE 4)"), (r"SAFARI.*EAST", "SAFARI EAST (ZONE 2)"),
    (r"SAFARI.*NORTH", "SAFARI NORTH (ZONE 3)"), (r"SAFARI", "SAFARI CENTER (ZONE 1)"),
    (r"SILPH|SAFFRON|DOJO|MR PSYCHIC", "PKMN TOWER 7F"), (r"ROCKET HIDEOUT", "PKMN TOWER 3&5F"),
    (r"POKEMON TOWER|PKMN TOWER", "PKMN TOWER 3&5F"), (r"LAVENDER", "ROCK TUNNEL B1F"),
    (r"INDIGO PLATEAU", "VICTORY ROAD 3F"), (r"VICTORY ROAD", "VICTORY ROAD 1F"), (r"SEAFOAM", "SEAFOAM 1F"),
    (r"UNDERGROUND PATH|ROUTE 7/8", "ROUTE 8"), (r"DIGLETT", "DIGLETT CAVE"), (r"ROCK TUNNEL", "ROCK TUNNEL 1F"),
    (r"MT MOON", "MT MOON 1F"), (r"DEPT STORE|GAME CORNER|CELADON", "CELADON CITY"), (r"MANSION", "MANSION 1F"),
    (r"CINNABAR", "CINNABAR ISLAND"), (r"CERULEAN CAVE", "CERULEAN CAVE 1F"), (r"POWER PLANT", "POWER PLANT"),
    (r"SS ANNE", "S.S. ANNE"), (r"VIRIDIAN FOREST", "VIRIDIAN FOREST")]
NEVER = r"POST GAME|CANNOT BE FOUND|REMATCH|ONE ISLAND|ISLAND ONE|TWO ISLAND|THREE ISLAND|SEVII|EMBER"
TUTORS_GONE = "FFEFEFEF"  # grey font on a tutor's location in the sheet: not there in hardcore
NOT_HARDCORE = {"FFFFC2C2", "FFD2EFFF"}  # care-package fills: red = not in hardcore, blue = not in MGM + hardcore
# The held items the care packages and the overworld hand out that gauntlet.item_value can price
# (2026-10-14, TODO.md C1): a type booster is x1.2 on its type, a heal berry / Leftovers / Shell
# Bell a flat value, Eviolite x1.5 for a mon that still evolves, Choice items a locked x1.5.
# Not priced (left out): resist berries (the matrix knows them, the item model does not), Choice
# Scarf, Light Ball / Thick Club, Black Sludge, Air Balloon and the weather rocks (not in hardcore).
HELD = {**{to_id_str(n): {"type": t, "mult": 1.2} for n, t in (
    ("Mystic Water", "Water"), ("Never-Melt Ice", "Ice"), ("Poison Barb", "Poison"), ("Sharp Beak", "Flying"),
    ("Silver Powder", "Bug"), ("Soft Sand", "Ground"), ("Spell Tag", "Ghost"), ("Twisted Spoon", "Psychic"),
    ("Magnet", "Electric"), ("Hard Stone", "Rock"), ("Miracle Seed", "Grass"), ("Dragon Fang", "Dragon"),
    ("Black Glasses", "Dark"), ("Black Belt", "Fighting"), ("Charcoal", "Fire"), ("Silk Scarf", "Normal"),
    ("Metal Coat", "Steel"), ("Pixie Plate", "Fairy"))},
    "oranberry": {"flat": 1.05}, "sitrusberry": {"flat": 1.15}, "berryjuice": {"flat": 1.15}, "shellbell": {"flat": 1.08},
    "leftovers": {"flat": 1.25}, "expertbelt": {"flat": 1.1}, "lifeorb": {"flat": 1.2}, "eviolite": {"flat": 1.5, "nfe": True},
    "choiceband": {"category": "Physical", "mult": 1.5, "lock": True}, "choicespecs": {"category": "Special", "mult": 1.5, "lock": True}}


MOVES_KNOWN = set(json.load(open(RR / "moves.json"))) if (RR / "moves.json").exists() else set()


def _norm(s: str) -> str:
    s = s.upper().replace("VERMILLION", "VERMILION").replace("POKÉMON", "POKEMON").replace("'S", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9&/ ]", " ", s)).strip()


def progression() -> None:
    """data/rr/order.json: the hardcore trainer order (name, location, cap, optional, arena
    effect) with `reach` = the index of the last encounter location open at that fight, and
    the method gates (rods / Surf) as trainer indices; data/rr/tms.json: every TM and hardcore
    tutor move with before_trainer = the trainer index at which it is in hand (a boss's TM:
    after that boss; a place: the first trainer that place is open at; post-game, rematch
    rewards and the Sevii Islands never). Money is not modelled (Game Corner TMs cost up to
    30k). Unmatched item texts are printed for review."""
    enc = json.load(open(RR / "tracker_encounters.json"))
    bosses = json.load(open(RR / "tracker_bosses.json"))["hardcore"]
    items = json.load(open(RR / "tracker_items.json"))
    locs = [_norm(l["name"]) for l in enc["locations"] if not l["postgame"]]

    def loc_index(text: str) -> int | None:
        t = _norm(text)
        if re.search(NEVER, t):
            return None
        hits = [i for i, n in enumerate(locs) if re.search(rf"\b{re.escape(n)}\b", t)]
        for pat, name in ALIASES:
            if not hits and re.search(pat, t):
                hits = [locs.index(_norm(name))]
        return min(hits) if hits else None

    entries, reach, cap = [], 0, 0
    for t in bosses["trainerOrder"]:
        li = loc_index(t["location"].replace(" GYM", ""))
        reach = max(reach, li if li is not None else 0)
        digits = "".join(ch for ch in str(t["levelCap"]).split("/")[0] if ch.isdigit())
        cap = int(digits) if digits else cap
        entries.append({"name": t["name"], "location": t["location"], "cap": cap, "optional": t["optional"], "reach": reach})
    by_title = {}
    for c in bosses["categories"]:
        for x in c["bosses"]:
            by_title.setdefault(x["title"].split("·")[-1].strip(), x.get("battleEffect", ""))
    for e in entries:
        e["effect"] = by_title.get(e["name"], "")

    def first_trainer(li: int | None) -> int | None:
        return None if li is None else next((i for i, e in enumerate(entries) if e["reach"] >= li), None)

    def when(text: str) -> int | None:
        """The trainer index at which an item described by `text` is in hand."""
        m = re.search(r"from (?:the )?(.+?) after defeating", text, re.I)
        if m and not re.search(NEVER, _norm(text)):
            name = _norm(m.group(1))
            li = loc_index(text)
            for i, e in enumerate(entries):
                if _norm(e["name"]) in name and (li is None or e["reach"] >= li):
                    return i + 1
        w = first_trainer(loc_index(text))
        hm = re.search(r"\(([A-Za-z ]+) HM required\)", text)
        if w is not None and hm:  # the place is open, the item behind an HM comes with the HM
            hw = next((when(h["location"]) for h in items["hms"] if h["move"].lower() == hm.group(1).strip().lower()), None)
            w = max(w, hw) if hw is not None else None
        return w

    gates = {}
    for area in items["overworld"]:
        for x in area["items"]:
            key = {"Old Rod": "old_rod", "Good Rod": "good_rod", "Super Rod": "super_rod", "HM 03 - Surf": "surfing"}.get(x["name"])
            if key:
                gates[key] = when(f"{area['area']}, {x['location']}")
    tms, unmatched = {}, []
    for t in items["tms"] + items["hms"]:
        w = when(t["location"])
        if w is None and not re.search(NEVER, _norm(t["location"])):
            unmatched.append((t["move"], t["location"][:80]))
        tms[to_id_str(t["move"])] = {"name": t["move"], "before_trainer": w if w is not None else 999, "from": t["location"]}
    # tutors: the sheet's Move Tutors tab (location row, then one row per move; grey = gone in hardcore)
    import zipfile
    z = zipfile.ZipFile(RR / "sheets" / "rr_items_tms_tutors.xlsx")
    ss = [re.sub(r"<[^>]+>", "", m) for m in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S)]
    st = z.read("xl/styles.xml").decode()
    font = [(re.search(r'<color rgb="([0-9A-F]+)"', f) or [None, None])[1] for f in re.findall(r"<font>(.*?)</font>", st, re.S)]
    xf_font = [int(m) for m in re.findall(r'<xf [^>]*fontId="(\d+)"', re.search(r"<cellXfs[^>]*>(.*?)</cellXfs>", st, re.S).group(1))]
    sheets = dict(re.findall(r'<sheet [^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', z.read("xl/workbook.xml").decode()))
    rels = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', z.read("xl/_rels/workbook.xml.rels").decode()))
    x = z.read("xl/" + rels[sheets["Move Tutors"]].replace("/xl/", "").lstrip("/")).decode()
    place, gone = None, False
    for row in re.findall(r"<row[^>]*>(.*?)</row>", x, re.S):
        cells = []
        for attrs, body in re.findall(r"<c ([^>]*?)>(.*?)</c>", re.sub(r"<c [^>]*/>", "", row), re.S):
            tt, sidx, v = re.search(r'\bt="([^"]+)"', attrs), re.search(r'\bs="(\d+)"', attrs), re.search(r"<v>(.*?)</v>", body, re.S)
            val = (ss[int(v.group(1))] if tt and tt.group(1) == "s" else v.group(1)) if v else ""
            if val.strip():
                cells.append((val.strip(), font[xf_font[int(sidx.group(1))]] if sidx else None))
        if not cells or cells[0][0] == "SPECIAL MOVE TUTORS":
            if cells:
                place = None
            continue
        if len(cells) >= 2 and to_id_str(cells[0][0]) not in MOVES_KNOWN:  # a location row
            place, gone = cells[0][0], cells[0][1] == TUTORS_GONE
            continue
        mid = to_id_str(cells[0][0])
        if place is None or gone or mid not in MOVES_KNOWN:
            continue
        w = when(place)
        if mid not in tms or (w or 999) < tms[mid]["before_trainer"]:
            tms[mid] = {"name": cells[0][0], "before_trainer": w if w is not None else 999, "from": f"tutor: {place}"}
    # gifts and in-game trades: species, level (a trade keeps the given mon's level: None), the trainer
    # index at which they are in hand (rematch rewards and the Sevii Islands never), how
    gifts = []
    for g in enc["gifts"]:
        w = when(f"{g['location']}, {g['requirements']}")
        lv = re.search(r"Lv\. ?(\d+)", g.get("info", "") + g.get("requirements", ""))
        if w is not None and lv:
            gifts.append({"species": to_id_str(g["species"]), "level": int(lv.group(1)), "t": w,
                          "how": f"gift: {g['location']} — {g['requirements']}"})
    for tr in enc["trades"]:
        w = when(tr["location"])
        if w is not None and tr["receive"] != "?":
            gifts.append({"species": to_id_str(tr["receive"]), "level": None, "t": w,
                          "how": f"trade: give {tr['give']} at {tr['location']}"})
    (RR / "order.json").write_text(json.dumps({"entries": entries, "gates": gates, "gifts": gifts}, indent=0))
    (RR / "tms.json").write_text(json.dumps(tms))
    # held items (data/rr/items.json, the BDSP format: gauntlet.assign_items): the care packages
    # (the sheet's tab, stage headers in play order, red / blue fills left out, x10 = six holders)
    # and the overworld's single held items (a boss's reward: after that boss; nothing EV-gated)
    held, prev = {}, 0

    def stage_when(header: str) -> int | None:
        """The trainer index a care-package stage is in hand at: "DEFEATING X" / "... AFTER X" is the
        first trainer so named from the previous stage on (the grunts repeat), plus one; a place is
        the first trainer it is open at ("EXITING" it: the next place; "AFTER FIRST MANDATORY TRAINER":
        plus one). Stages are in play order, so the index never goes back."""
        h = _norm(header)
        if re.search(NEVER, h):
            return None
        left, _, cond = h.partition(" AFTER ")
        name = h.removeprefix("DEFEATING ").split(" IN ")[0].strip() if h.startswith("DEFEATING ") else \
            cond.removeprefix("DEFEATING ").strip() if cond and not cond.startswith(("EXITING", "FIRST")) else None
        if name:
            i = next((i for i, e in enumerate(entries) if _norm(e["name"]) == name and i >= prev), None)
            return None if i is None else i + 1
        li = loc_index(re.sub(r"^(BEFORE )?(ENTERING|EXITING) ", "", left))
        if li is None:
            return None
        w = first_trainer(li + 1 if left.startswith("EXITING") else li)
        return None if w is None else w + 1 if cond.startswith("FIRST") else w

    def add(name: str, w: int | None, src: str) -> None:
        q = re.search(r"\[x(\d+)\]", name)
        iid = to_id_str(re.sub(r"\s*\[.*\]", "", name))
        if iid in HELD and w is not None and (iid not in held or w < held[iid]["before_trainer"]):
            held[iid] = {"name": re.sub(r"\s*\[.*\]", "", name).strip(), "before_trainer": w, "from": src,
                         "copies": min(int(q.group(1)), 6) if q else 1, **HELD[iid]}
    fill = [(re.search(r'fgColor rgb="([0-9A-F]+)"', f) or [None, None])[1] for f in re.findall(r"<fill>(.*?)</fill>", st, re.S)]
    xf_fill = [int(m) for m in re.findall(r'<xf [^>]*fillId="(\d+)"', re.search(r"<cellXfs[^>]*>(.*?)</cellXfs>", st, re.S).group(1))]
    x = z.read("xl/" + rels[sheets["Care Packages"]].replace("/xl/", "").lstrip("/")).decode()
    stage = None
    for row in re.findall(r"<row[^>]*>(.*?)</row>", x, re.S):
        cells = []
        for attrs, body in re.findall(r"<c ([^>]*?)>(.*?)</c>", re.sub(r"<c [^>]*/>", "", row), re.S):
            tt, sidx, v = re.search(r'\bt="([^"]+)"', attrs), re.search(r'\bs="(\d+)"', attrs), re.search(r"<v>(.*?)</v>", body, re.S)
            val = (ss[int(v.group(1))] if tt and tt.group(1) == "s" else v.group(1)) if v else ""
            if val.strip():
                cells.append((val.strip(), fill[xf_fill[int(sidx.group(1))]] if sidx else None))
        if len(cells) == 1 and "[x" not in cells[0][0] and cells[0][0].isupper():  # a stage header
            w = stage_when(cells[0][0])
            stage = (cells[0][0], w)
            prev = max(prev, w or 0)
        elif stage and stage[1] is not None:
            for name, fc in cells:
                if fc not in NOT_HARDCORE:
                    add(name, stage[1], f"care package: {stage[0].title()}")
    for area in items["overworld"]:
        for it in area["items"]:
            if "EV" not in it["location"]:
                add(it["name"], when(f"{area['area']}, {it['location']}"), f"{area['area']}: {it['location'][:60]}")
    (RR / "items.json").write_text(json.dumps(dict(sorted(held.items(), key=lambda kv: kv[1]["before_trainer"])), indent=0))
    stages = sorted({(v["before_trainer"], v["from"]) for v in held.values() if v["from"].startswith("care")})
    print(f"data/rr/items.json: {len(held)} held items; care-package stages at trainers {[t for t, _ in stages]}")
    n_early = {b: sum(1 for v in tms.values() if v["before_trainer"] <= i) for i, b in
               [(6, "Brock"), (15, "Misty"), (26, "Surge"), (34, "Erika")]}
    print(f"data/rr/order.json: {len(entries)} trainers, gates {gates}, {len(gifts)} gifts/trades; tms.json: {len(tms)} TM/tutor moves, in hand by {n_early}")
    for u in unmatched:
        print("  unmatched item text:", u)


def main() -> None:
    enc = json.load(open(RR / "tracker_encounters.json"))
    bosses = json.load(open(RR / "tracker_bosses.json"))
    names = {e["species"] for l in enc["locations"] for t in l["methods"].values() for e in t}
    names |= {p["species"] for mode in bosses.values() for c in mode["categories"] for x in c["bosses"] for p in x["pokemon"]}
    # the player's possible starters (every generation; drawn at random per run) are not wild encounters
    starters = ["bulbasaur", "charmander", "squirtle", "chikorita", "cyndaquil", "totodile", "treecko", "torchic", "mudkip",
                "turtwig", "chimchar", "piplup", "snivy", "tepig", "oshawott", "chespin", "fennekin", "froakie", "rowlet",
                "litten", "popplio", "grookey", "scorbunny", "sobble", "sprigatito", "fuecoco", "quaxly"]
    ids = sorted({to_id_str(n) for n in names} | set(starters))
    raw = subprocess.run(["node", "-e", NODE, str(FORK), *ids], capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    missing = [i for i in ids if i not in data["species"]]
    (RR / "learnsets.json").write_text(json.dumps(data["learnsets"]))
    (RR / "evolutions.json").write_text(json.dumps(data["evos"]))
    (RR / "species.json").write_text(json.dumps(data["species"]))
    (RR / "moves.json").write_text(json.dumps(data["moves"]))
    (RR / "trainers.json").write_text(json.dumps({"trainers": []}))
    (RR / "encounters.json").write_text(json.dumps({"starter": [], "routes": []}))
    print(f"data/rr: {len(data['species'])} species, {len(data['moves'])} moves; {len(missing)} ids unknown to the mod: {missing[:30]}")
    parse_dump()
    progression()
    # the mod's full tables, for rr_probe.inject_mod_data (poke-env's mirrors must know the mod's own moves / species)
    dump = subprocess.run(["node", "-e", """
const {Dex} = require(process.argv[1] + '/dist/sim'); const d = Dex.mod('gen9rr'); const out = {};
for (const [t, all] of [['moves', d.moves.all()], ['species', d.species.all()], ['items', d.items.all()], ['abilities', d.abilities.all()]]) {
  out[t] = {}; for (const x of all) { const o = {}; for (const k of Object.keys(x)) if (typeof x[k] !== 'function') o[k] = x[k]; out[t][x.id] = o; } }
console.log(JSON.stringify(out));""", str(FORK)], capture_output=True, text=True, check=True).stdout
    (RR / "mod_data.json").write_text(dump)
    print("data/rr/mod_data.json: the mod's moves / species / items / abilities")


if __name__ == "__main__":
    main()
