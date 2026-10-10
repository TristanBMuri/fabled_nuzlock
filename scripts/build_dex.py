"""Rebuild the dex-derived files of the game folder from the Showdown clone (node, once):

  uv run python scripts/build_dex.py --build-moves     # learnsets.json (level-up + TM compatibility),
                                                       # evolutions.json (every method), moves.json
  uv run python scripts/build_dex.py --build-species   # species.json (name, types, ability slots, base stats)
  uv run python scripts/build_dex.py --check           # trainer / encounter species against the learnsets

TM gates: scripts/build_tms.py; trainers / order / encounters: scripts/build_order.py;
held items: scripts/build_items.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gauntlet import DATA, GAME, ROOT, load_encounters, load_trainers, species_name  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402


# --------------------------------------------------------------- moves builder

LEARNSET_SCRIPT = """
const path = require('path');
const {Learnsets} = require(path.join(process.argv[1], 'dist', 'data', 'learnsets.js'));
// TM compatibility from the BDSP mod ("8M" = a machine move in BDSP)
const BDSP = require(path.join(process.argv[1], 'dist', 'data', 'mods', 'gen8bdsp', 'learnsets.js')).Learnsets;
const {Dex} = require(path.join(process.argv[1], 'dist', 'sim'));
const maxLevel = Number(process.argv[2]);
// closure over level-up evolutions: evolved forms need learnsets too
const species = new Set(process.argv.slice(3));
// every evolution method (gen 4 dex, no regional formes); the closure adds evolved forms
const evos = {};
const queue = [...species];
while (queue.length) {
  const sp = queue.pop();
  if (evos[sp]) continue;
  evos[sp] = Dex.species.get(sp).evos.map(e => Dex.species.get(e))
    .filter(e => e.exists && e.num <= 493 && !e.forme)
    .map(e => ({to: e.id, type: e.evoType || "level", level: e.evoLevel || null,
                item: e.evoItem || null, move: e.evoMove || null}));
  for (const e of evos[sp]) { species.add(e.to); queue.push(e.to); }
}
const out = {};
for (const sp of species) {
  const ls = Learnsets[sp];
  if (!ls) continue;
  const levelup = Object.entries(ls.learnset)
    .filter(([m, gens]) => gens.some(g => /^4L\\d+$/.test(g)))
    .map(([m, gens]) => [m, Math.max(...gens.filter(g => /^4L\\d+$/.test(g)).map(g => parseInt(g.slice(2))))]);
  levelup.sort((a, b) => a[1] - b[1]);
  const tm = Object.entries((BDSP[sp] || {}).learnset || {}).filter(([, g]) => g.includes("8M")).map(([m]) => m);
  out[sp] = {levelup: levelup.filter(([, l]) => l <= maxLevel), tm};
}
const moves = {};
for (const ls of Object.values(out)) for (const m of [...ls.levelup.map(([m]) => m), ...ls.tm]) {
  const mv = Dex.moves.get(m);
  moves[mv.id] = {type: mv.type, power: mv.basePower, category: mv.category, accuracy: mv.accuracy};
}
console.log(JSON.stringify({moves: out, evos, movedata: moves}));
"""


def build_moves(max_level: int = 100) -> None:
    """learnsets.json (species -> {levelup: [[move, level], ...] up to max_level,
    sorted by level; tm: moves the species takes from a BDSP TM}) and evolutions.json (species -> [[evolved id,
    method fields: type level|levelFriendship|useItem|levelHold|levelMove|trade,
    level, item, move]) in the game folder, for the encounter species and
    their evolutions (every method, gen 4 dex)."""
    enc = json.load(open(DATA / GAME / "encounters.json"))
    species = sorted({e["species"] for r in enc["routes"] for e in r["encounters"]}
                     | set(enc["starter"]))
    raw = subprocess.run(
        ["node", "-e", LEARNSET_SCRIPT, str(ROOT / "pokemon-showdown"),
         str(max_level), *species],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(raw)
    missing = [s for s in species if s not in data["moves"]]
    if missing:
        print(f"WARN: no learnset data for {missing}")
    (DATA / GAME / "learnsets.json").write_text(json.dumps(data["moves"]))
    (DATA / GAME / "evolutions.json").write_text(json.dumps(data["evos"], indent=1))
    (DATA / GAME / "moves.json").write_text(json.dumps(data["movedata"], indent=1))
    print(f"wrote {GAME}/learnsets.json for {len(data['moves'])} species (max level "
          f"{max_level}), evolutions.json for {len(data['evos'])} species, "
          f"moves.json for {len(data['movedata'])} moves")


SPECIES_SCRIPT = """
const path = require('path');
const {Dex} = require(path.join(process.argv[1], 'dist', 'sim'));
const out = {};
for (const sp of process.argv.slice(2)) {
  const s = Dex.species.get(sp);
  if (!s.exists) continue;
  out[s.id] = {name: s.name, types: s.types, abilities: s.abilities, baseStats: s.baseStats};
}
console.log(JSON.stringify(out));
"""


def build_species() -> None:
    """species.json in the game folder: name, types, ability slots ({0, 1, H})
    and base stats for every species a run can hold or face (encounters,
    their evolutions, trainer parties). The env rolls a catch's ability slot
    from it."""
    ids = set(json.load(open(DATA / GAME / "learnsets.json")))
    trainers, _ = load_trainers()
    ids |= {to_id_str(species_name(p["species"])) for t in trainers.values() for p in t["party"]}
    raw = subprocess.run(["node", "-e", SPECIES_SCRIPT, str(ROOT / "pokemon-showdown"), *sorted(ids)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    (DATA / GAME / "species.json").write_text(json.dumps(data, indent=1))
    print(f"wrote species.json for {len(data)} species ({len(ids) - len(data)} unknown)")


# --------------------------------------------------------------------- checks

def check(slice_len: int = 21) -> None:
    trainers, order = load_trainers()
    enc = load_encounters()
    learnsets = json.loads(subprocess.run(
        ["node", "-e",
         "const {Learnsets} = require(require('path').join(process.argv[1], 'dist', 'data', 'learnsets.js'));"
         "console.log(JSON.stringify(Object.keys(Learnsets)))",
         str(ROOT / "pokemon-showdown")],
        capture_output=True, text=True, check=True).stdout)
    known = set(learnsets)

    problems = 0
    prev_level = 0
    for e in order[:slice_len]:
        if e["max_level"] < prev_level:
            print(f"WARN: level drop at {e['trainer_name']}")
        prev_level = max(prev_level, e["max_level"])
        t = trainers[e["trainer_id"]]
        for p in t["party"]:
            if to_id_str(species_name(p["species"])) not in known:
                print(f"BAD species {p['species']} ({e['trainer_name']})")
                problems += 1
    enc_species = ({e["species"] for r in enc["routes"] for e in r["encounters"]}
                   | set(enc["starter"]))
    for s in sorted(enc_species):
        if s not in known:
            print(f"BAD encounter species {s}")
            problems += 1
    for r in enc["routes"]:
        total = sum(e["weight"] for e in r["encounters"])
        if abs(total - 1.0) > 0.05:
            print(f"WARN weights sum {total:.2f} on {r['name']}")
    print(f"check done: {problems} hard problems over slice {slice_len}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build-moves", action="store_true")
    ap.add_argument("--build-species", action="store_true", help="data/species.json from the dex")
    ap.add_argument("--slice", type=int, default=21)
    ap.add_argument("--max-level", type=int, default=100)
    args = ap.parse_args()
    if args.check:
        check(args.slice)
    if args.build_moves:
        build_moves(args.max_level)
    if args.build_species:
        build_species()
