"""data/bdsp/tms.json: which TM hands out which move, and before which trainer
of the route order it is in hand (E7). Source: each TM's Bulbapedia page
(TM001..TM100), the BDSP row of its acquisition table, cached by
build_order.fetch. Rules: TMs are reusable (decided 2026-09-19); a TM counts
from its earliest fixed source — an overworld pickup, an NPC gift (gym
leaders after their fight), the Veilstone Department Store (all its stock on
arrival), the Route 212 shard collector. Not counted: Grand Underground
merchants (random stock, sphere farming), the Battle Tower and everything
post-game (Routes 224+, Survival Area, Stark Mountain).

  uv run python scripts/build_tms.py        # then gauntlet.py --build-moves (TM compatibility)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_order import fetch  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDER = json.load(open(ROOT / "data/bdsp/order.json"))["entries"]

# Bulbapedia location -> the order.json location it is reachable from (first
# trainer of that label); a gym leader's gift comes after their fight.
GATES = {
    "Trainers' School": "Jubilife Trainers' School (optional)", "Oreburgh Gate": "Oreburgh Gate",
    "Ravaged Path": "Route 204 North", "Floaroma Town": "Floaroma Meadow (Galactic)",
    "Valley Windworks": "Valley Windworks", "Eterna Forest": "Eterna Forest",
    "Eterna City": "Eterna Gym", "Eterna Condominiums": "Eterna Gym", "Old Chateau": "Eterna Gym",
    "Wayward Cave": "Wayward Cave (optional)", "Mount Coronet": "Mt. Coronet (ascent)",
    "Amity Square": "Hearthome City", "Lost Tower": "Lost Tower (optional)",
    "Maniac Tunnel": "Solaceon Ruins (optional)", "Veilstone Department Store": "Veilstone City (Galactic)",
    "Veilstone City": "Veilstone City (Galactic)", "Team Galactic HQ": "Galactic HQ (Veilstone)",
    "Valor Lakefront": "Valor Lakefront", "Hotel Grand Lake": "Valor Lakefront", "Great Marsh": "Pastoria Gym",
    "Celestic Town": "Celestic Town", "Canalave City": "Canalave City", "Iron Island": "Iron Island (optional)",
    "Fuego Ironworks": "Fuego Ironworks (optional)", "Lake Valor": "Lake Valor", "Lake Verity": "Lake Verity",
    "Lake Acuity": "Snowpoint Gym", "Pokémon Mansion": "Route 212 (optional detour)",
    "Sunyshore City": "Sunyshore Gym", "Victory Road": "Victory Road",
}
LEADERS = {"Roark": "Oreburgh Gym", "Gardenia": "Eterna Gym", "Maylene": "Veilstone Gym",
           "Crasher Wake": "Pastoria Gym", "Fantina": "Hearthome Gym", "Byron": "Canalave Gym",
           "Candice": "Snowpoint Gym", "Volkner": "Sunyshore Gym"}
NEVER = {"Grand Underground", "Battle Tower", "Survival Area", "Stark Mountain"}


def bdsp_row(page: str) -> tuple[str, str]:
    """(move name, raw location field) from the BDSP row of a TM page."""
    i = page.find("{{gameabbrev8|BDSP}}")
    move = re.findall(r"\{\{TMtable/header\|([^|}]+)\|", page[:i])[-1].strip()
    rest = page[i + len("{{gameabbrev8|BDSP}}|"):].split("\n")[0]
    depth, out = 0, ""
    for ch in rest:  # up to the first top-level '|' (the price column)
        depth += (ch in "{[") - (ch in "}]")
        if ch == "|" and depth == 0:
            break
        out += ch
    return move, out


def gate(field: str) -> tuple[int | None, str | None]:
    """(before_trainer, source) = the earliest fixed source in the field, or (None, None)."""
    first = {}
    for e in ORDER:
        first.setdefault(e["location"], e["order_index"])
    leader = {e["location"]: e["order_index"] + 1 for e in ORDER if e["breakpoint"] == "gym_leader"}
    names = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", field) + re.findall(r"\{\{si\|([^|}]+)\}\}", field)
    names += [f"Route {n}" for n in re.findall(r"\{\{rtn?\|(\d+)\|Sinnoh\}\}", field)]
    best = (None, None)
    for name in names:
        if name in LEADERS:
            idx = leader[LEADERS[name]]
        elif name.startswith("Route "):
            label = next((loc for loc in first if loc.startswith(name + " ") or loc == name), None)
            idx = first[label] if label else None
        else:
            idx = first[GATES[name]] if name in GATES else None
        if idx is not None and (best[0] is None or idx < best[0]):
            best = (idx, name)
    return best


def main() -> None:
    tms, skipped = {}, []
    for n in range(1, 101):
        move, field = bdsp_row(fetch(f"TM{n:03d}"))
        idx, src = gate(field)
        if idx is None:
            skipped.append((n, move, re.sub(r"\{\{DL\|[^}]*\}\}", "GU", field)))
            continue
        tms[to_id_str(move)] = {"tm": n, "name": move, "before_trainer": idx, "from": src}
    out = ROOT / "data/bdsp/tms.json"
    out.write_text(json.dumps(dict(sorted(tms.items(), key=lambda kv: kv[1]["before_trainer"])), indent=1))
    for m, t in sorted(tms.items(), key=lambda kv: kv[1]["before_trainer"]):
        print(f"  #{t['before_trainer']:3d} TM{t['tm']:03d} {t['name']:<16} {t['from']}")
    print(f"wrote {out.relative_to(ROOT)}: {len(tms)} TMs; not obtainable before the League: "
          + ", ".join(f"TM{n:03d} {mv} [{f}]" for n, mv, f in skipped))


if __name__ == "__main__":
    main()
