"""data/bdsp/items.json: held items the player can own, before which trainer of
the route order each is in hand, and what it does (E9). Source: Serebii's BDSP
item list (one row per item: locations, Pickup, wild holders, shops), cached in
/tmp/serebii_cache. Rules: an item counts from its earliest fixed pre-League
source — an overworld pickup or an NPC gift — one copy per source. Not counted:
the Grand Underground, Battle Park / Tower prizes, shops, Pickup, wild holders,
post-game areas. Consumables (Focus Sash, berries, herbs) stay out: a battle
rebuilt from packed teams would hand them back every fight. The effect model is
the assignment rule's (gauntlet.assign_items): a multiplier on the mon's best
attack, or a flat value every mon gets.

  uv run python scripts/build_items.py
"""

import html
import json
import re
import subprocess
from pathlib import Path

from poke_env.data import to_id_str

ROOT = Path(__file__).resolve().parents[1]
ORDER = json.load(open(ROOT / "data/bdsp/order.json"))["entries"]
URL = "https://www.serebii.net/brilliantdiamondshiningpearl/items.shtml"
CACHE = Path("/tmp/serebii_cache")

# Serebii location -> the order.json location it is reachable from (first trainer of that
# label); conservative where an item needs a later HM or a second visit.
GATES = {
    "Route 204": "Route 204 North", "Floaroma Meadow": "Floaroma Meadow (Galactic)",
    "Eterna Forest": "Eterna Forest", "Eterna City": "Eterna Gym", "Old Chateau": "Eterna Gym",
    "Route 206": "Route 206 (Cycling Road)", "Wayward Cave": "Wayward Cave (optional)",
    "Hearthome City": "Hearthome City", "Amity Square": "Hearthome City", "Route 212": "Route 212 (optional detour)",
    "Solaceon Ruins": "Solaceon Ruins (optional)", "Route 210": "Route 210 North", "Route 215": "Route 215",
    "Veilstone City": "Veilstone City (Galactic)", "Route 214": "Route 214", "Great Marsh": "Pastoria Gym",
    "Pastoria City": "Pastoria City", "Celestic Town": "Celestic Town", "Iron Island": "Iron Island (optional)",
    "Fuego Ironworks": "Fuego Ironworks (optional)", "Route 220": "Route 220 (optional)",
    "Route 221": "Route 221 (optional)", "Oreburgh Gate": "Oreburgh Gate B1F (optional)",  # Earth Plate: Surf
    "Route 217": "Route 217", "Mt. Coronet": "Mt. Coronet (ascent)", "Sunyshore City": "Sunyshore Gym",
    "Victory Road": "Victory Road", "Pokemon League": "Pokemon League", "Jubilife City": "Jubilife City (Galactic)",
}
# item -> effect: {"type": T, "mult"} boosts attacks of type T; {"category": C, "mult"} attacks of
# that category ("lock": Choice); {"flat": v} the same value for every holder (Leftovers).
ITEMS = {
    "Leftovers": {"flat": 1.25}, "Shell Bell": {"flat": 1.08}, "Quick Claw": {"flat": 1.05},
    "Metronome": {"flat": 1.1}, "Wide Lens": {"flat": 1.05}, "Razor Claw": {"flat": 1.05},
    "Expert Belt": {"flat": 1.1}, "Wise Glasses": {"category": "Special", "mult": 1.1},
    "Choice Specs": {"category": "Special", "mult": 1.5, "lock": True},
    "Silk Scarf": {"type": "Normal", "mult": 1.2}, "Black Glasses": {"type": "Dark", "mult": 1.2},
    "Mystic Water": {"type": "Water", "mult": 1.2}, "Sea Incense": {"type": "Water", "mult": 1.2},
    "Wave Incense": {"type": "Water", "mult": 1.2}, "Splash Plate": {"type": "Water", "mult": 1.2},
    "Miracle Seed": {"type": "Grass", "mult": 1.2}, "Rose Incense": {"type": "Grass", "mult": 1.2},
    "Meadow Plate": {"type": "Grass", "mult": 1.2}, "Magnet": {"type": "Electric", "mult": 1.2},
    "Zap Plate": {"type": "Electric", "mult": 1.2}, "Black Belt": {"type": "Fighting", "mult": 1.2},
    "Fist Plate": {"type": "Fighting", "mult": 1.2}, "Poison Barb": {"type": "Poison", "mult": 1.2},
    "Toxic Plate": {"type": "Poison", "mult": 1.2}, "Soft Sand": {"type": "Ground", "mult": 1.2},
    "Earth Plate": {"type": "Ground", "mult": 1.2}, "SilverPowder": {"type": "Bug", "mult": 1.2},
    "Insect Plate": {"type": "Bug", "mult": 1.2}, "Spell Tag": {"type": "Ghost", "mult": 1.2},
    "Spooky Plate": {"type": "Ghost", "mult": 1.2}, "Metal Coat": {"type": "Steel", "mult": 1.2},
    "Iron Plate": {"type": "Steel", "mult": 1.2}, "Icicle Plate": {"type": "Ice", "mult": 1.2},
    "Sky Plate": {"type": "Flying", "mult": 1.2}, "Mind Plate": {"type": "Psychic", "mult": 1.2},
    "Odd Incense": {"type": "Psychic", "mult": 1.2}, "Stone Plate": {"type": "Rock", "mult": 1.2},
    "Rock Incense": {"type": "Rock", "mult": 1.2}, "Draco Plate": {"type": "Dragon", "mult": 1.2},
    "Dread Plate": {"type": "Dark", "mult": 1.2},
}


def fetch() -> str:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "bdsp_items.html"
    if not f.exists() or f.stat().st_size == 0:
        subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0 nuzlock_rl research", URL, "-o", str(f)], check=True)
    return f.read_text(encoding="utf-8", errors="replace")


def rows(page: str) -> dict[str, list[str]]:
    """item name -> fixed locations (the links before the first Pickup / Shop / Wild line)."""
    out = {}
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(tds) < 4:
            continue
        name = html.unescape(re.sub(r"<[^>]+>", "", tds[1])).strip()
        fixed = re.split(r"<br\s*/?>", tds[3])[0]  # later lines: Pickup, Shop, Wild Encounter
        out[name] = [html.unescape(x).strip() for x in re.findall(r"<a [^>]*>([^<]+)</a>", fixed)]
    return out


def main() -> None:
    first = {}
    for e in ORDER:
        first.setdefault(e["location"], e["order_index"])
    table, skipped = {}, []
    for name, locs in rows(fetch()).items():
        if name not in ITEMS:
            continue
        gates = [(first[GATES[l]], l) for l in locs if l in GATES]
        if not gates:
            skipped.append(f"{name} [{', '.join(locs) or 'no fixed source'}]")
            continue
        idx, src = min(gates)
        table[to_id_str(name)] = {"name": name, "before_trainer": idx, "from": src, "copies": len(gates), **ITEMS[name]}
    out = ROOT / "data/bdsp/items.json"
    table = dict(sorted(table.items(), key=lambda kv: kv[1]["before_trainer"]))
    out.write_text(json.dumps(table, indent=1))
    for it in table.values():
        print(f"  #{it['before_trainer']:3d} {it['name']:<14} x{it['copies']} {it['from']}")
    print(f"wrote {out.relative_to(ROOT)}: {len(table)} items; no pre-League source: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
