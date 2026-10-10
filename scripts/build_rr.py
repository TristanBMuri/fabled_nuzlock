"""data/rr/: the Radical Red game folder for the boss probe, from the tracker's
docs export (encounters, bosses, items) and the RadicalRedShowdown fork's
gen9rr mod (learnsets, evolutions, species, moves). Minimal: what
gauntlet.encounter_moves / evolve / the probe need. Not a full gauntlet yet
(no route trainers in order.json).

  uv run python scripts/build_rr.py            # needs pokemon-showdown-rr/ built
"""

import json
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
                elif l.startswith("Level: "):
                    d["level"] = int(l[7:]) if l[7:].strip().isdigit() else None
                elif l.endswith(" Nature"):
                    d["nature"] = l[:-7].strip()
                elif l.startswith("- "):
                    d["moves"].append(l[2:].strip())
                elif l.startswith("IVs: "):
                    for part in l[5:].split("/"):
                        v, k = part.strip().split(" ", 1)
                        d["ivs"][{"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}[k]] = int(v)
            if d["level"] is None or not d["moves"]:
                continue
            d["evs"] = trainer_evs(m[1])
            mons.append(d)
        if not mons:
            continue
        spread = lambda e: ",".join(str(e[k]) for k in STATS)
        packed = "]".join("|".join([d["species"], "", to_id_str(d["item"]), to_id_str(d["ability"]),
                                    ",".join(to_id_str(x) for x in d["moves"]), d["nature"], spread(d["evs"]), d["gender"],
                                    spread(d["ivs"]), "", str(d["level"]), ""]) for d in mons)
        out.append({"name": m[1].strip(), "id": m[2], "cap": max(d["level"] for d in mons), "n": len(mons),
                    "mons": mons, "packed": packed})
    (RR / "trainers_hardcore.json").write_text(json.dumps(out))
    print(f"data/rr/trainers_hardcore.json: {len(out)} trainers from the dump")
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


def main() -> None:
    enc = json.load(open(RR / "tracker_encounters.json"))
    bosses = json.load(open(RR / "tracker_bosses.json"))
    names = {e["species"] for l in enc["locations"] for t in l["methods"].values() for e in t}
    names |= {p["species"] for mode in bosses.values() for c in mode["categories"] for x in c["bosses"] for p in x["pokemon"]}
    ids = sorted({to_id_str(n) for n in names})
    raw = subprocess.run(["node", "-e", NODE, str(FORK), *ids], capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    missing = [i for i in ids if i not in data["species"]]
    (RR / "learnsets.json").write_text(json.dumps(data["learnsets"]))
    (RR / "evolutions.json").write_text(json.dumps(data["evos"]))
    (RR / "species.json").write_text(json.dumps(data["species"]))
    (RR / "moves.json").write_text(json.dumps(data["moves"]))
    # every TM / tutor move is "in hand" from the start: the probe's upper bound (gates come with the gauntlet)
    (RR / "tms.json").write_text(json.dumps({m: {"tm": 0, "name": m, "before_trainer": 0, "from": "any"}
                                             for ls in data["learnsets"].values() for m in ls["tm"]}))
    (RR / "items.json").write_text("{}")
    (RR / "trainers.json").write_text(json.dumps({"trainers": []}))
    (RR / "order.json").write_text(json.dumps({"entries": []}))
    (RR / "encounters.json").write_text(json.dumps({"starter": [], "routes": []}))
    print(f"data/rr: {len(data['species'])} species, {len(data['moves'])} moves; {len(missing)} ids unknown to the mod: {missing[:30]}")
    parse_dump()
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
