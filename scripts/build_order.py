"""Build data/bdsp/order.json (trainers in story order) and the encounter gates in
encounters.json from Bulbapedia's BDSP trainer tables.

  uv run python scripts/build_order.py            # fetches pages into /tmp/bulbapedia_cache

Each story visit is a Bulbapedia page plus a level band, so a location visited
twice (Route 208, 210, 211, 213, Oreburgh Gate, the lakes) is split by the
levels of the trainers there. Story battles (rival, commanders, leaders, E4)
are inserted by trainer_id from trainers.json. `mandatory` is per location for
the Galactic plot fights and per record for story battles; route trainers are
marked optional because the tables do not say who blocks the path.
"""
import json, os, re, subprocess, sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "bdsp"
CACHE = Path("/tmp/bulbapedia_cache")
UA = "nuzlock_rl research"


def fetch(title: str) -> str:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / (title + ".txt")
    if not f.exists() or f.stat().st_size == 0:
        subprocess.run(["curl", "-sL", "-A", UA, f"https://bulbapedia.bulbagarden.net/w/index.php?title={title}&action=raw",
                        "-o", str(f)], check=True)
    return f.read_text()


def split_top(s):
    out, depth, cur = [], 0, ""
    i = 0
    while i < len(s):
        if s.startswith("{{", i) or s.startswith("[[", i): depth += 1; cur += s[i:i+2]; i += 2; continue
        if s.startswith("}}", i) or s.startswith("]]", i): depth -= 1; cur += s[i:i+2]; i += 2; continue
        if s[i] == "|" and depth == 0: out.append(cur); cur = ""; i += 1; continue
        cur += s[i]; i += 1
    out.append(cur); return out
def clean(s):
    s = re.sub(r"\{\{tt\|([^|}]*)\|[^}]*\}\}", r"\1", s)   # {{tt|shown|tip}} -> shown
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    return s.strip()
def entries(text):
    out = []
    for m in re.finditer(r"\{\{[Tt]rainerentry\|(.*)\}\}\s*$", text, flags=re.M):
        raw = m.group(1)
        if "game=BDSP" not in raw: continue
        if re.search(r"rematch", raw, re.I): continue
        f = split_top(raw)
        f = [x for x in f if not x.startswith("game=")]
        pos = [x for x in f if "=" not in x.split("{{")[0] or x.startswith("{{")]
        # pos: img, class, name, money, n, then groups of 5 (dex, species, gender, level, item)
        cls, name = clean(pos[1]), clean(pos[2])
        rest = [clean(x) for x in pos[5:]]
        mons, i = [], 0
        isdex = lambda x: x.isdigit() and 3 <= len(x) <= 4
        while i + 3 < len(rest):
            if not isdex(rest[i]): i += 1; continue
            sp, lvl = rest[i + 1], rest[i + 3]
            i += 4
            if i < len(rest) and not isdex(rest[i]): i += 1  # item slot present
            try: mons.append((sp, int(lvl.split("/")[0].replace(",", ""))))
            except ValueError: pass
        out.append((cls, name, mons))
def entries(text):
    out = []
    for m in re.finditer(r"\{\{[Tt]rainerentry\|(.*)\}\}\s*$", text, flags=re.M):
        raw = m.group(1)
        if "game=BDSP" not in raw: continue
        if re.search(r"rematch", raw, re.I): continue
        f = split_top(raw)
        f = [x for x in f if not x.startswith("game=")]
        pos = [x for x in f if "=" not in x.split("{{")[0] or x.startswith("{{")]
        # pos: img, class, name, money, n, then groups of 5 (dex, species, gender, level, item)
        cls, name = clean(pos[1]), clean(pos[2])
        rest = [clean(x) for x in pos[5:]]
        mons, i = [], 0
        isdex = lambda x: x.isdigit() and 3 <= len(x) <= 4
        while i + 3 < len(rest):
            if not isdex(rest[i]): i += 1; continue
            sp, lvl = rest[i + 1], rest[i + 3]
            i += 4
            if i < len(rest) and not isdex(rest[i]): i += 1  # item slot present
            try: mons.append((sp, int(lvl.split("/")[0].replace(",", ""))))
            except ValueError: pass
        out.append((cls, name, mons))
    return out

# (page, lo, hi, location label, mandatory-by-location)  or ("ID", trainer_id, location, breakpoint)
V = [
 ("Sinnoh_Route_202", 0, 6, "Route 202", False),
 ("Sinnoh_Route_204", 0, 6, "Route 204 South", False),
 ("Jubilife_City_School", 0, 6, "Jubilife Trainers' School (optional)", False),
 ("ID", 176, "Route 203", "rival"), ("ID", 177, "Route 203", "rival"), ("ID", 178, "Route 203", "rival"),
 ("Sinnoh_Route_203", 0, 8, "Route 203", False),
 ("Oreburgh_Gate", 0, 10, "Oreburgh Gate", False),
 ("Oreburgh_Mine", 0, 10, "Oreburgh Mine", False),
 ("Oreburgh_Gym", 0, 12, "Oreburgh Gym", False),
 ("ID", 175, "Oreburgh Gym", "gym_leader"),
 ("Jubilife_City", 0, 12, "Jubilife City (Galactic)", True),
 ("Sinnoh_Route_204", 7, 12, "Route 204 North", False),
 ("Floaroma_Meadow", 0, 12, "Floaroma Meadow (Galactic)", True),
 ("Sinnoh_Route_205", 0, 12, "Route 205 South", False),
 ("Valley_Windworks", 0, 13, "Valley Windworks (Galactic)", True),
 ("ID", 220, "Valley Windworks", "commander"),
 ("Sinnoh_Route_205", 13, 16, "Route 205 North", False),
 ("Eterna_Forest", 0, 16, "Eterna Forest", False),
 ("Eterna_Gym", 0, 18, "Eterna Gym", False),
 ("ID", 240, "Eterna Gym", "gym_leader"),
 ("Team_Galactic_Eterna_Building", 0, 18, "Galactic Eterna Building", True),
 ("ID", 311, "Galactic Eterna Building", "commander"),
 ("Sinnoh_Route_211", 0, 22, "Route 211 West (optional)", False),
 ("Sinnoh_Route_206", 0, 18, "Route 206 (Cycling Road)", False),
 ("Wayward_Cave", 0, 20, "Wayward Cave (optional)", False),
 ("Sinnoh_Route_207", 0, 20, "Route 207", False),
 ("Sinnoh_Route_208", 0, 22, "Route 208", False),
 ("ID", 351, "Hearthome City", "rival"), ("ID", 352, "Hearthome City", "rival"), ("ID", 353, "Hearthome City", "rival"),
 ("Sinnoh_Route_212", 0, 27, "Route 212 (optional detour)", False),
 ("Sinnoh_Route_209", 0, 22, "Route 209", False),
 ("Lost_Tower", 0, 22, "Lost Tower (optional)", False),
 ("Solaceon_Ruins", 0, 22, "Solaceon Ruins (optional)", False),
 ("Sinnoh_Route_210", 0, 22, "Route 210 South", False),
 ("Caf%C3%A9_Cabin", 0, 24, "Cafe Cabin (optional)", False),
 ("Sinnoh_Route_215", 0, 26, "Route 215", False),
 ("Veilstone_City", 0, 26, "Veilstone City (Galactic)", True),
 ("Veilstone_Gym", 0, 29, "Veilstone Gym", False),
 ("ID", 242, "Veilstone Gym", "gym_leader"),
 ("Sinnoh_Route_214", 0, 27, "Route 214", False),
 ("Valor_Lakefront", 0, 27, "Valor Lakefront", False),
 ("Seven_Stars_Restaurant", 0, 27, "Seven Stars Restaurant (optional)", False),
 ("Sinnoh_Route_213", 0, 27, "Route 213", False),
 ("Pastoria_Gym", 0, 27, "Pastoria Gym", False),
 ("ID", 241, "Pastoria Gym", "gym_leader"),
 ("ID", 354, "Pastoria City", "rival"), ("ID", 355, "Pastoria City", "rival"), ("ID", 356, "Pastoria City", "rival"),
 ("Hearthome_Gym", 0, 31, "Hearthome Gym", False),
 ("ID", 243, "Hearthome Gym", "gym_leader"),
 ("Sinnoh_Route_210", 23, 99, "Route 210 North", False),
 ("Sinnoh_Route_211", 23, 99, "Route 211 East", False),
 ("Celestic_Town", 0, 99, "Celestic Town", True),
 ("Sinnoh_Route_218", 0, 99, "Route 218", False),
 ("ID", 357, "Canalave City", "rival"), ("ID", 358, "Canalave City", "rival"), ("ID", 359, "Canalave City", "rival"),
 ("Canalave_Gym", 0, 99, "Canalave Gym", False),
 ("ID", 179, "Canalave Gym", "gym_leader"),
 ("Iron_Island", 0, 99, "Iron Island (optional)", False),
 ("ID", 483, "Iron Island (optional)", "none"),
 ("Fuego_Ironworks", 0, 99, "Fuego Ironworks (optional)", False),
 ("Sinnoh_Route_219", 0, 99, "Route 219 (optional)", False),
 ("Sinnoh_Route_220", 0, 99, "Route 220 (optional)", False),
 ("Sinnoh_Route_221", 0, 99, "Route 221 (optional)", False),
 ("Oreburgh_Gate", 11, 99, "Oreburgh Gate B1F (optional)", False),
 ("Sinnoh_Route_208", 23, 99, "Route 208 (Surf, optional)", False),
 ("Sinnoh_Route_213", 28, 99, "Route 213 (Surf, optional)", False),
 ("Lake_Valor", 0, 99, "Lake Valor (Galactic)", True),
 ("ID", 313, "Lake Valor", "commander"),
 ("Lake_Verity", 30, 99, "Lake Verity (Galactic)", True),
 ("ID", 310, "Lake Verity", "commander"),
 ("Sinnoh_Route_216", 0, 99, "Route 216", False),
 ("Sinnoh_Route_217", 0, 99, "Route 217", False),
 ("Snowpoint_Gym", 0, 99, "Snowpoint Gym", False),
 ("ID", 244, "Snowpoint Gym", "gym_leader"),
 ("Team_Galactic_HQ", 0, 99, "Galactic HQ (Veilstone)", True),
 ("ID", 314, "Galactic HQ (Veilstone)", "commander"),
 ("ID", 308, "Galactic HQ (Veilstone)", "boss"),
 ("Mount_Coronet", 0, 99, "Mt. Coronet (ascent)", True),
 ("Spear_Pillar", 0, 99, "Spear Pillar", True),
 ("ID", 401, "Spear Pillar (tag battle)", "commander"), ("ID", 312, "Spear Pillar (tag battle)", "commander"),
 ("ID", 309, "Spear Pillar", "boss"),
 ("Sunyshore_Gym", 0, 99, "Sunyshore Gym", False),
 ("ID", 245, "Sunyshore Gym", "gym_leader"),
 ("Sinnoh_Route_222", 0, 99, "Route 222", False),
 ("Sinnoh_Route_223", 0, 99, "Route 223", False),
 ("Victory_Road_(Sinnoh)", 0, 52, "Victory Road", False),
 ("ID", 360, "Pokemon League", "rival"), ("ID", 361, "Pokemon League", "rival"), ("ID", 362, "Pokemon League", "rival"),
 ("ID", 185, "Pokemon League", "elite_four"), ("ID", 186, "Pokemon League", "elite_four"),
 ("ID", 187, "Pokemon League", "elite_four"), ("ID", 188, "Pokemon League", "elite_four"),
 ("ID", 191, "Pokemon League", "champion"),
]
# Encounter areas with no trainers of their own, or sub-areas of a page that
# belong to a different visit: (page, sub-area keyword or None, area name, opens_at location)
ENC_EXTRA = [
 ("Sinnoh_Route_201", None, "Route 201", None),
 ("Ravaged_Path", None, "Ravaged Path", "Route 204 North"),
 ("Old_Chateau", None, "Old Chateau (optional)", "Eterna Gym"),
 ("Mount_Coronet", "Southern", "Mt. Coronet South", "Route 208"),
 ("Mount_Coronet", "room 1", "Mt. Coronet North (Celestic side)", "Route 211 West (optional)"),
 ("Mount_Coronet", "room 2", "Mt. Coronet North (Snowpoint side)", "Route 216"),
 ("Mount_Coronet", "Mountainside", "Mt. Coronet North (Snowpoint side)", "Route 216"),
 ("Mount_Coronet", "Tunnel", "Mt. Coronet North (Snowpoint side)", "Route 216"),
 ("Lake_Verity", None, "Lake Verity", None),
 ("Acuity_Lakefront", None, "Acuity Lakefront", "Snowpoint Gym"),
 ("Sinnoh_Route_207", None, "Route 207", "Oreburgh Mine"),  # its grass is reachable from Oreburgh
 ("Twinleaf_Town", None, "Twinleaf Town", None),          # water only: opens with the Old Rod
 ("Hearthome_City", None, "Hearthome City", "Route 209"),  # Happiny egg
 ("Pastoria_City", None, "Pastoria City", "Pastoria Gym"),
 ("Canalave_City", None, "Canalave City", "Canalave Gym"),
 ("Sunyshore_City", None, "Sunyshore City", "Sunyshore Gym"),
]
ENC_SKIP = {"Sinnoh_Route_207"}  # visits whose encounter area is listed in ENC_EXTRA instead
# encounter methods -> the trainer gate where the player gains the method (None = always)
METHODS = {"Grass": "walk", "Cave": "walk", "Building": "walk", "Walking": "walk",
           "Surf": "surf", "Fish Old": "old", "Fish Good": "good", "Gift": "gift", "Egg": "gift"}
METHOD_GATES = {"walk": None, "gift": None, "old": "Route 203",  # Old Rod: Jubilife gate house
                "good": "Route 209", "surf": "Route 218"}          # Good Rod: Route 209; HM Surf: Celestic
NOT_GIFTS = {"mew", "jirachi", "turtwig", "chimchar", "piplup", "eevee"}  # save bonuses, starters, postgame
# visits whose page is one Nuzlocke area with an earlier visit (one encounter per route)
SKIP_LABELS = {"Route 204 North", "Route 205 North", "Route 210 North", "Route 211 East",
               "Route 208 (Surf, optional)", "Route 213 (Surf, optional)", "Oreburgh Gate B1F (optional)"}
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def catch_tables(text: str) -> list[tuple[str, list]]:
    """BDSP walking encounters (BD column) per sub-area of a page:
    [(sub-area heading, [(species, lo, hi, rate%)])]."""
    sec = re.search(r"^==Pokémon==\s*$", text, flags=re.M)  # city pages have other Generation VIII headings
    m = re.search(r"^(={2,4})Generation VIII=+\s*$", text[sec.end():] if sec else text, flags=re.M)
    if not m:
        return []
    depth = len(m.group(1))
    text = text[sec.end():] if sec else text
    out, name, rows = [], "", []
    for line in text[m.end():].split("\n"):
        h = re.fullmatch(r"(=+)(.+?)=+\s*", line)
        if h:
            if len(h.group(1)) <= depth:
                break
            if rows: out.append((name, rows))
            name, rows = h.group(2).strip(), []
            continue
        m = re.match(r"\{\{[Cc]atch/entrybdsp\|(.*)\}\}", line.strip())
        if not m:
            continue
        f = [clean(x) for x in split_top(m.group(1))]
        method = METHODS.get(re.sub(r"^\dF$", "Building", f[4])) if len(f) >= 7 else None
        if not method or f[2] != "yes" or (method == "gift" and norm(f[1]) in NOT_GIFTS):
            continue
        lv = [int(x) for x in re.findall(r"\d+", f[5])]
        # rate: "all=10%" or morning|day|night columns ("0%|0%|10%": take the max,
        # a player can pick the time of day); gifts are "One"
        rates = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", "|".join(f[6:9]))] or [100.0]
        if not lv:
            continue
        rows.append((norm(f[1]), min(lv), max(lv), max(rates), method))  # Showdown id: "mimejr"
    if rows: out.append((name, rows))
    return out


def area_rows(tables, keyword, lo, hi, multi=False):
    """Sub-tables of a page for one visit: by keyword, else by level band; a
    page with one visit takes every table, a twice-visited page only matches."""
    if keyword:
        return [r for n, rows in tables if keyword.lower() in n.lower() for r in rows]
    lvl = lambda rows: max(h for _, _, h, _, meth in rows if meth == "walk") if any(r[4] == "walk" for r in rows) else max(h for _, _, h, _, _ in rows)
    picked = [rows for n, rows in tables if rows and lo <= lvl(rows) <= hi]
    if not picked and not multi:
        picked = [rows for _, rows in tables]
    return [r for rows in picked for r in rows]


STORY_CLASSES = {"Gym Leader", "Commander", "Team Galactic Boss", "Elite Four", "Champion", "Pokémon Trainer"}
def key(mons): return tuple(sorted((norm(s), int(l)) for s, l in mons))


def main() -> None:
    T = json.load(open(DATA / "trainers.json"))["trainers"]
    W = {}
    for v in V:
        if v[0] != "ID" and v[0] not in W and v[0] != "Jubilife_City_School":
            W[v[0]] = [{"class": c, "name": n, "mons": m} for c, n, m in entries(fetch(v[0]))]
    byid = {t["trainer_id"]: t for t in T}
    pool = {}  # (label, teamkey) -> [ids]
    for t in T:
        pool.setdefault((t["trainer_label"], key((m["species"], m["level"]) for m in t["party"])), []).append(t["trainer_id"])
    bylabel = {}
    for t in T: bylabel.setdefault(t["trainer_label"], []).append(t["trainer_id"])
    used, order, unmatched = set(), [], []
    W["Jubilife_City_School"] = [{"class": "School Kid", "name": n, "mons": [(m["species"], m["level"]) for m in byid[i]["party"]]}
                                 for i, n in ((264, "Harrison"), (267, "Christine"))]
    def take(tid, loc, bp, mandatory, wiki=None):
        t = byid[tid]; used.add(tid)
        lv = [m["level"] for m in t["party"]]
        order.append({"order_index": len(order), "trainer_id": tid, "trainer_name": t["trainer_label"],
                      "max_level": max(lv), "avg_level": round(sum(lv) / len(lv), 2), "team_size": len(lv),
                      "breakpoint": bp, "location": loc, "mandatory": mandatory})
    for v in V:
        if v[0] == "ID":
            _, tid, loc, bp = v
            take(tid, loc, "none" if bp in ("rival", "commander", "boss") else bp, True)
            continue
        page, lo, hi, loc, mand = v
        for e in W.get(page, []):
            mx = max(l for _, l in e["mons"])
            if not (lo <= mx <= hi): continue
            name = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", e["name"]).strip()
            names = [n.strip() for n in re.split(r"\s+(?:and|&)\s+", name)] if re.search(r"\s(and|&)\s", name) else [name]
            for n in names:  # a double battle is two trainer records in the game data; played as two singles
                label = f"{e['class']} {n}".strip()
                k = key(e["mons"])
                ids = [i for i in pool.get((label, k), []) if i not in used]
                if not ids and len(names) > 1:  # each partner owns a subset of the listed mons
                    ids = [i for i in bylabel.get(label, []) if i not in used
                           and set(key((m["species"], m["level"]) for m in byid[i]["party"])) <= set(k)]
                if not ids and label == "Team Galactic Grunt":
                    ids = [i for i in bylabel.get(label, []) if i not in used
                           and key((m["species"], m["level"]) for m in byid[i]["party"]) == k]
                if not ids:  # same label and species, levels differ between wiki and sheet
                    sp = sorted(x for x, _ in k)
                    ids = [i for i in bylabel.get(label, []) if i not in used
                           and sorted(norm(m["species"]) for m in byid[i]["party"]) == sp]
                if not ids:  # label-only fallback when unique and unused
                    cand = [i for i in bylabel.get(label, []) if i not in used]
                    if len(cand) == 1: ids = cand
                if not ids:
                    unmatched.append((page, label, e["mons"])); continue
                take(ids[0], loc, "none", mand)
    first = {}
    for e in order: first.setdefault(e["location"], e["order_index"])
    gate = lambda loc: 0 if loc is None else first[loc]
    pages = {v[0] for v in V if v[0] != "ID"} | {e[0] for e in ENC_EXTRA}
    tables = {pg: catch_tables(fetch(pg)) for pg in pages}
    multi = Counter([v[0] for v in V if v[0] != "ID"] + [e[0] for e in ENC_EXTRA])
    areas = []
    for v in V:
        if v[0] == "ID" or v[0] == "Jubilife_City_School": continue
        page, lo, hi, loc, _ = v
        if loc in SKIP_LABELS: continue
        kw = None
        if multi[page] > 1:  # twice-visited page: sub-area by keyword, else level band
            kw = next((k for k in ("B1F", "South", "North", "West", "East") if k in loc), None)
        rows = area_rows(tables[page], kw, lo, hi, multi[page] > 1) if page not in ENC_SKIP | {"Mount_Coronet"} else []
        if rows: areas.append((loc, loc, rows))
    for page, kw, name, opens in ENC_EXTRA:
        rows = area_rows(tables[page], kw, 0, 99)
        if rows: areas.append((name, opens, rows))
    enc = json.load(open(DATA / "encounters.json"))
    enc["_comment"] = ("BDSP encounters (BD column) per story visit from Bulbapedia's Generation VIII tables "
                       "(scripts/build_order.py): methods walk (Grass/Cave/Building), old / good (Old and Good "
                       "Rod), surf, gift (gift/egg Pokémon); Super Rod, radar, swarm, honey, Underground, "
                       "trades excluded. One encounter per area: the env rolls one pick per method available "
                       "when the area opens and the run layer takes at most one. level = [min,max] uniform; "
                       "weights are the table rates normalized per method. 'before_trainer' = the area opens "
                       "before this index of the route-ordered gauntlet (order.json); 'opens_at' names the "
                       "visit (a water-only area waits for the Old Rod at Route 203, Good Rod Route 209, Surf Route 218).")
    routes, byname = [], {}
    for name, opens, rows in areas:
        if name in byname:  # a second sub-area of the same visit: merge
            byname[name].extend(rows); continue
        byname[name] = list(rows)
        routes.append((name, opens))
    routes_out = []
    for name, opens in routes:
        # the area fires at its first visit if any method is available then, else
        # when the first water method arrives; the pool = methods available at that point
        avail = {meth: max(gate(opens), gate(METHOD_GATES[meth]) if METHOD_GATES[meth] else 0)
                 for meth in {r[4] for r in byname[name]}}
        opening = min(avail.values())
        merged = {}
        for sp, lo, hi, rate, meth in byname[name]:
            if avail[meth] > opening: continue
            m = merged.setdefault((meth, sp), [lo, hi, 0.0])
            m[0], m[1], m[2] = min(m[0], lo), max(m[1], hi), m[2] + rate
        tot = {meth: sum(m[2] for (mt, _), m in merged.items() if mt == meth) or 1.0 for meth in avail}
        routes_out.append({"name": name, "before_trainer": opening, "opens_at": opens or "start",
                           "encounters": [{"species": sp, "level": [m[0], m[1]], "weight": round(m[2] / tot[meth], 4),
                                           "method": meth} for (meth, sp), m in sorted(merged.items())]})
    routes = sorted(routes_out, key=lambda r: r["before_trainer"])
    enc["routes"] = routes
    print(f"{len(routes)} encounter areas, {sum(len(r['encounters']) for r in routes)} rows")
    print(f"ordered {len(order)} trainers; unmatched wiki entries {len(unmatched)}")
    for e in order:
        if e["breakpoint"] != "none" or e["trainer_name"].startswith(("Pokémon Trainer Cedric", "Commander", "Team Galactic Boss")):
            print(f"   #{e['order_index']:3d} {e['trainer_name']:28} L{e['max_level']:2d} {e['location']}")
    print("encounter gates:", [(r["name"], r["before_trainer"], len(r["encounters"])) for r in enc["routes"]])
    for u in unmatched: print("  UNMATCHED", u)
    # level-cap sanity: max level so far vs next boss
    cap, drops = 0, []
    for e in order:
        if e["max_level"] < cap - 8: drops.append((e["order_index"], e["trainer_name"], e["max_level"], cap))
        cap = max(cap, e["max_level"])
    print("big level drops (expected: Galactic building after Gardenia, optional detours) (idx, name, lvl, cap-so-far):", drops[:20])
    unused = [t for t in T if t["trainer_id"] not in used]
    print(f"trainers.json entries not placed: {len(unused)}")
    print(Counter(t["trainer_label"].split(" ")[0] for t in unused).most_common(12))
    lowunused = [(t["trainer_id"], t["trainer_label"], max(m["level"] for m in t["party"])) for t in unused if max(m["level"] for m in t["party"]) <= 30]
    print("unplaced with max level <= 30:", lowunused)
    json.dump(enc, open(DATA / "encounters.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"metadata": {"source": "Bulbapedia BDSP trainer tables in story order (scripts/build_order.py)",
                            "trainer_count": len(order)}, "entries": order},
              open(DATA / "order.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
