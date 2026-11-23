"""Radical Red data for the Rust engine: diff the gen9rr mod against the gen9 it inherits from
(both in the RR fork, pokemon-showdown-rr) and write what can be data into
poke-engine/src/rr_data.rs (built only with the `rr` feature; poke_engine_rr in Python).

  uv run python rl/rr_engine_data.py            # writes rr_data.rs, prints the worklist
  uv run python rl/rr_engine_data.py --json logs/rr_engine_gaps.json

Data (generated): an existing move's power, accuracy, type, category, priority, drain,
recoil, flags, secondary chances / effects, self / target boosts, status.
Not data (printed as the worklist, done by hand in Rust): new moves / abilities / items /
species, and every move / ability / item / condition whose handler code differs
(a handler = any function on the entry; compared by source text).
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "pokemon-showdown-rr"
OUT = ROOT / "poke-engine" / "src" / "rr_data.rs"

# both dexes in one node call: plain fields as JSON, handlers as their source text
NODE = r"""
const {Dex} = require(process.argv[1] + '/dist/sim');
const dump = (e) => { const o = {}; for (const k in e) { const v = e[k];
  o[k] = typeof v === 'function' ? 'fn:' + v.toString() : v; } return o; };
const out = {};
for (const mod of ['gen9', 'gen9rr']) {
  const d = Dex.mod(mod); out[mod] = {};
  for (const [t, key] of [['moves', 'Moves'], ['abilities', 'Abilities'], ['items', 'Items'], ['species', 'Species'], ['conditions', 'Conditions']]) {
    out[mod][t] = {};
    const all = t === 'conditions' ? Object.keys(d.data.Conditions).map(id => d.conditions.get(id)) : d[t].all();
    for (const e of all) if (e.exists) out[mod][t][e.id] = dump(e);
  }
}
process.stdout.write(JSON.stringify(out));
"""

TYPES = {t: t.upper() for t in ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground",
                                 "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]}
TYPES["???"] = "TYPELESS"
CATEGORY = {"Physical": "Physical", "Special": "Special", "Status": "Status"}
FLAGS = ["bite", "bullet", "charge", "contact", "heal", "powder", "protect", "pulse", "punch", "recharge",
         "reflectable", "slicing", "sound", "wind"]
STATUS = {"brn": "BURN", "par": "PARALYZE", "slp": "SLEEP", "frz": "FREEZE", "psn": "POISON", "tox": "TOXIC"}
BOOST = {"atk": "attack", "def": "defense", "spa": "special_attack", "spd": "special_defense", "spe": "speed",
         "accuracy": "accuracy"}
VOLATILE = {"flinch": "FLINCH", "confusion": "CONFUSION"}
# statements the diff cannot derive: poke-engine models vanilla Chloroblast's self-damage as a
# negative heal, which the mod's plain recoil replaces (found by the conformance diff)
EXTRA = {"chloroblast": ["m.heal = None;"]}
# plain fields that the generated patch covers, or that do not matter to the engine
DATA_FIELDS = {"basePower", "accuracy", "type", "category", "priority", "drain", "recoil", "flags", "secondary",
               "secondaries", "self", "boosts", "status", "desc", "shortDesc", "num", "pp", "name", "gen",
               "isNonstandard", "contestType", "zMove", "maxMove", "fullname", "noPPBoosts", "hasSheerForce",
               "baseMoveType", "realMove", "isViable", "viable", "hasCrashDamage", "rating", "noTutor", "noTM"}


def load() -> dict:
    out = subprocess.run(["node", "-e", NODE, str(FORK)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def handlers(e: dict) -> dict:
    return {k: v for k, v in e.items() if isinstance(v, str) and v.startswith("fn:")}


def handler_diff(old: dict, new: dict) -> list[str]:
    ho, hn = handlers(old), handlers(new)
    return sorted(k for k in set(ho) | set(hn) if ho.get(k) != hn.get(k))


def boosts(b: dict) -> str:
    return "StatBoosts { " + ", ".join(f"{v}: {int(b.get(k, 0))}" for k, v in BOOST.items()) + " }"


def secondary(s: dict) -> str | None:
    """One Showdown secondary as a poke-engine Secondary, None if it has no such shape."""
    chance = f"{float(s.get('chance', 100)):.1f}"
    if set(s) - {"chance", "status", "volatileStatus", "boosts", "self"}:
        return None
    if "self" in s:
        if set(s["self"]) != {"boosts"}:
            return None
        return f"Secondary {{ chance: {chance}, target: MoveTarget::User, effect: Effect::Boost({boosts(s['self']['boosts'])}) }}"
    if "status" in s and s["status"] in STATUS:
        return f"Secondary {{ chance: {chance}, target: MoveTarget::Opponent, effect: Effect::Status(PokemonStatus::{STATUS[s['status']]}) }}"
    if "volatileStatus" in s and s["volatileStatus"] in VOLATILE:
        return (f"Secondary {{ chance: {chance}, target: MoveTarget::Opponent, "
                f"effect: Effect::VolatileStatus(PokemonVolatileStatus::{VOLATILE[s['volatileStatus']]}) }}")
    if "boosts" in s:
        return f"Secondary {{ chance: {chance}, target: MoveTarget::Opponent, effect: Effect::Boost({boosts(s['boosts'])}) }}"
    return None


def move_patch(mid: str, old: dict, new: dict, todo: list) -> list[str]:
    """Rust statements that turn poke-engine's gen9 entry for `mid` into the mod's."""
    lines, v = [], "m"
    if new["basePower"] != old["basePower"]:
        lines.append(f"{v}.base_power = {float(new['basePower']):.1f};")
    if new["accuracy"] != old["accuracy"]:
        lines.append(f"{v}.accuracy = {100.0 if new['accuracy'] is True else float(new['accuracy']):.1f};")
    if new["type"] != old["type"]:
        lines.append(f"{v}.move_type = PokemonType::{TYPES[new['type']]};")
    if new["category"] != old["category"]:
        lines.append(f"{v}.category = MoveCategory::{CATEGORY[new['category']]};")
    if new["priority"] != old["priority"]:
        lines.append(f"{v}.priority = {int(new['priority'])};")
    if new.get("drain") != old.get("drain"):
        d = new.get("drain")
        lines.append(f"{v}.drain = {'None' if not d else f'Some({d[0] / d[1]:.4f})'};")
    if new.get("recoil") != old.get("recoil"):
        r = new.get("recoil")
        lines.append(f"{v}.recoil = {'None' if not r else f'Some({r[0] / r[1]:.4f})'};")
    fo, fn = old.get("flags", {}), new.get("flags", {})
    for f in FLAGS:
        if bool(fo.get(f)) != bool(fn.get(f)):
            lines.append(f"{v}.flags.{f} = {'true' if fn.get(f) else 'false'};")
    so, sn = old.get("secondaries") or [], new.get("secondaries") or []
    if so != sn:
        conv = [secondary(s) for s in sn]
        if all(conv):
            lines.append(f"{v}.secondaries = {'None' if not conv else 'Some(vec![' + ', '.join(conv) + '])'};")
        else:
            todo.append(f"move {mid}: secondaries {sn}")
    bo, bn = old.get("boosts"), new.get("boosts")
    selfo, selfn = (old.get("self") or {}).get("boosts"), (new.get("self") or {}).get("boosts")
    if bo != bn or selfo != selfn:
        if bn and selfn:
            todo.append(f"move {mid}: boosts on both sides")
        elif bn or selfn:
            tgt = "Opponent" if bn else "User"
            lines.append(f"{v}.boost = Some(Boost {{ target: MoveTarget::{tgt}, boosts: {boosts(bn or selfn)} }});")
        else:
            lines.append(f"{v}.boost = None;")
    if new.get("status") != old.get("status"):
        s = new.get("status")
        if s and s in STATUS:
            lines.append(f"{v}.status = Some(Status {{ target: MoveTarget::Opponent, status: PokemonStatus::{STATUS[s]} }});")
        elif not s:
            lines.append(f"{v}.status = None;")
        else:
            todo.append(f"move {mid}: status {s}")
    other = sorted(k for k in set(old) | set(new) if k not in DATA_FIELDS and not k.startswith("on")
                   and not (isinstance(new.get(k), str) and new[k].startswith("fn:"))
                   and json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True))
    if other:
        todo.append(f"move {mid}: fields {other}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the worklist here")
    args = ap.parse_args()
    d = load()
    base, rr = d["gen9"], d["gen9rr"]
    todo, work = [], {}
    body = []
    for mid in sorted(rr["moves"]):
        new = rr["moves"][mid]
        if mid.startswith("hiddenpower") or new.get("isZ") or new.get("isMax"):
            continue
        old = base["moves"].get(mid)
        if old is None:
            work.setdefault("new_moves", []).append(mid)
            continue
        lines = move_patch(mid, old, new, todo) + EXTRA.get(mid, [])
        if lines:
            body.append(f"    if let Some(m) = moves.get_mut(&Choices::{mid.upper()}) {{\n        "
                        + "\n        ".join(lines) + "\n    }")
        if h := handler_diff(old, new):
            work.setdefault("move_handlers", []).append(f"{mid}: {','.join(h)}")
    for t in ("abilities", "items", "conditions"):
        for k in sorted(rr[t]):
            if k not in base[t]:
                work.setdefault(f"new_{t}", []).append(k)
            elif h := handler_diff(base[t][k], rr[t][k]):
                work.setdefault(f"{t}_handlers", []).append(f"{k}: {','.join(h)}")
    for k in sorted(rr["species"]):
        o, n = base["species"].get(k), rr["species"][k]
        if o is None:
            work.setdefault("new_species", []).append(k)
        elif o.get("weightkg") != n.get("weightkg"):
            work.setdefault("species_weight", []).append(k)
    work["move_data_todo"] = todo
    OUT.write_text(
        "// Generated by rl/rr_engine_data.py from the gen9rr mod (pokemon-showdown-rr) -- do not edit by hand.\n"
        "// The mod's changes to existing moves that are plain data; handler changes are hand-written.\n"
        "#![allow(unused_imports)]\n"
        "use crate::choices::{Boost, Choice, Choices, Effect, MoveCategory, MoveTarget, Secondary, StatBoosts, Status};\n"
        "use crate::engine::state::PokemonVolatileStatus;\n"
        "use crate::state::{PokemonStatus, PokemonType};\n"
        "use std::collections::HashMap;\n\n"
        "#[allow(unused_imports, unused_variables, clippy::all)]\n"
        "pub fn patch_moves(moves: &mut HashMap<Choices, Choice>) {\n" + "\n".join(body) + "\n}\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(body)} moves patched")
    for k, v in work.items():
        print(f"\n{k} ({len(v)}):\n  " + "\n  ".join(v))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(work, indent=1))


if __name__ == "__main__":
    main()
