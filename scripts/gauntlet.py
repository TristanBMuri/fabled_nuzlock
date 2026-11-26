"""The game as data, and the player's side of a run as pure logic: the game
folder (load_game), Mon / packed teams, movesets (best4 / last4 over level-up
moves and the TMs in hand), held items, ability slots, evolution behind
progression gates. Offline-testable; nothing here touches the simulator.
Battles: scripts/runner.py. The dex-derived files: scripts/build_dex.py.
"""

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from poke_env.data import to_id_str

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------------------------------------------------------------- data loading

GAME = "bdsp"  # the game folder under data/: everything the env knows about a game


def load_game(name: str | None = None) -> dict:
    """The game folder (default: GAME, read at call time so a script can switch folders)."""
    return _load_game(name or GAME)


@functools.lru_cache(maxsize=None)
def _load_game(name: str) -> dict:
    """The game folder as one dict: trainers (id -> record), order (gauntlet
    entries), encounters (starter + areas), learnsets (species -> {levelup:
    [[move, level], ...] sorted by level, tm: [moves the species takes from a
    TM]}), evolutions (species -> [[to, level], ...]), species (id -> name,
    types, ability slots, base stats), tms (move -> {tm, before_trainer, from}:
    scripts/build_tms.py), items (held items -> {before_trainer, from, copies,
    effect}: scripts/build_items.py). A second game is a second folder with the
    same files and no code change."""
    d = DATA / name
    j = lambda f: json.load(open(d / f))
    return {"trainers": {t["trainer_id"]: t for t in j("trainers.json")["trainers"]},
            "order": j("order.json")["entries"],
            "encounters": j("encounters.json"),
            "learnsets": j("learnsets.json"),
            "evolutions": j("evolutions.json"),
            "species": j("species.json"),
            "moves": j("moves.json"),
            "tms": j("tms.json"),
            "items": j("items.json")}


def load_trainers():
    """{trainer_id: trainer_dict} and the ordered gauntlet entries."""
    g = load_game()
    return g["trainers"], g["order"]


def load_encounters():
    return load_game()["encounters"]


def species_name(species: str) -> str:
    """Showdown's display name ('mrmime' -> 'Mr. Mime', 'shellos (east sea)' ->
    'Shellos-East'). The nickname must equal it exactly: Showdown replaces a
    nickname that matches the species with the canonical name, and poke-env keys
    the team by nickname — 'Mr. mime' vs 'Mr. Mime' made three Mr. Mimes collapse
    (AssertionError at order 158); a literal '(East Sea)' parsed as species
    'eastsea' (KeyError at order 162). Formes in the source data are
    'Base (Forme Words)'; Showdown ids are base + forme, or base + first word."""
    from poke_env.data import GenData, to_id_str
    dex = GenData.from_gen(9).pokedex  # a superset of gen 8's (Radical Red adds gen 9 species)
    m = re.fullmatch(r"\s*(.+?)\s*\((.+?)\)\s*", species)
    candidates = ([m.group(1) + m.group(2), m.group(1) + m.group(2).split()[0], m.group(1)]
                  if m else [species])
    for c in candidates:
        if to_id_str(c) in dex:
            return dex[to_id_str(c)]["name"]
    return species.capitalize()


def base_name(species: str) -> str:
    """The base species' display name ('Raticate' for 'Raticate-Alola'): what Showdown
    makes the nickname of a forme anyway, and never over the 18-character limit a
    long forme name ('Dudunsparce-Three-Segment') would break."""
    from poke_env.data import GenData, to_id_str
    dex = GenData.from_gen(9).pokedex
    name = species_name(species)
    base = to_id_str(dex.get(to_id_str(name), {}).get("baseSpecies", name))
    return dex.get(base, {}).get("name", name)


STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_NAMES = dict(zip(STATS, ("HP", "Atk", "Def", "SpA", "SpD", "Spe")))
# Player mons: 31 IVs and a flat 85 EVs per stat (the 510 total a trained mon
# carries, spread evenly). The format has no EV limit, so an omitted spread
# would be filled with 252 per stat by the validator (README: the EV bug).
PLAYER_EVS = {k: 85 for k in STATS}
BANNED_MOVES: set[str] = set()  # moves the player may not use (Radical Red hardcore: data/rr/hardcore_rules.json)
SPECIES_MOVES: dict[str, set[str]] = {}  # move -> the only species that may use it (hardcore: Toxic, Leech Seed)
PLAYER_IVS = {k: 31 for k in STATS}


def player_evs(species: str, rule: str = "flat") -> dict:
    """The player's EV spread (Rules.evs): flat = 85 in every stat (the old
    fill); trained = 252 in the better attacking stat, 252 Spe, 4 HP — the
    best legal spread on the Cynthia probe (README 2026-09-19: calc bot
    2.7% -> 6.0% wins from full HP). Natures stay neutral (no mints in BDSP)."""
    if rule == "flat":
        return PLAYER_EVS
    bs = load_game()["species"].get(species, {}).get("baseStats", {"atk": 1, "spa": 0})
    return {**dict.fromkeys(STATS, 0), "hp": 4, "atk" if bs["atk"] >= bs["spa"] else "spa": 252, "spe": 252}


@dataclass(eq=False)  # identity, not value: two twins at the cap are two mons (League party / carry defects)
class Mon:
    species: str
    level: int
    moves: list[str]
    ability: str | None = None
    item: str | None = None
    alive: bool = True
    caught_level: int | None = None    # friendship evolutions fire FRIENDSHIP_LEVELS above it
    nature: str | None = None          # None = neutral (Serious)
    evs: dict | None = None            # None = PLAYER_EVS; trainers carry the sheet's spread
    ivs: dict | None = None            # None = PLAYER_IVS

    def pack(self) -> str:
        """Showdown packed string (nick|species|item|ability|moves|nature|evs|gender|
        ivs|shiny|level|happiness). Written directly: poke-env's text roundtrip
        drops an all-zero EV spread, which the validator then fills with 252s.
        The nickname is the base species (what Showdown names a forme anyway)."""
        spread = lambda d: ",".join(str(d[k]) for k in STATS)
        name, nick = species_name(self.species), base_name(self.species)
        if to_id_str(self.species) == "greninja" and to_id_str(self.ability or "") == "battlebond":
            name = "Greninja-Bond"  # gen9rr Battle Bond only triggers Ash-Greninja transform on greninjabond
        return "|".join([nick, name if name != nick else "", to_id_str(self.item or ""),
                         to_id_str(self.ability or ""), ",".join(to_id_str(m) for m in self.moves),
                         self.nature or "Serious", spread(self.evs or PLAYER_EVS), "M",
                         spread(self.ivs or PLAYER_IVS), "", str(self.level), ""])


def pack_team(mons: list[Mon]) -> str:
    return "]".join(m.pack() for m in mons)


def trainer_mons(trainer: dict) -> list[Mon]:
    mons = []
    for p in trainer["party"]:
        moves = p["moves"]
        if not moves:  # 2 records in source data have empty movesets;
            # server rejects moveless mons — learnset fallback
            moves = encounter_moves(p["species"].lower(), p["level"]) or ["tackle"]
        mons.append(Mon(p["species"].lower(), p["level"], moves, p.get("ability"), p.get("item"),
                        nature=p.get("nature"), evs=p.get("evs") or dict.fromkeys(STATS, 0),
                        ivs=p.get("ivs") or dict.fromkeys(STATS, 0)))
    return mons


def load_species() -> dict:
    return load_game()["species"]


def ability_slots(species: str) -> dict:
    """{'0': ..., '1': ..., 'H': ...} for a species id ('' if unknown)."""
    return load_species().get(species, {}).get("abilities", {})


def roll_ability(species: str, rng) -> str | None:
    """A wild catch has ability slot 0 or 1 (even odds), never the hidden one."""
    slots = ability_slots(species)
    return rng.choice([slots[k] for k in ("0", "1") if k in slots]) if slots else None


def evolved_ability(species: str, ability: str | None, evolved: str) -> str | None:
    """Evolution keeps the ability *slot*: Starly Keen Eye (0) -> Staravia
    Intimidate (0); Shinx Intimidate (1) -> Luxio Intimidate (1)."""
    slot = next((k for k, v in ability_slots(species).items() if v == ability), "0")
    new = ability_slots(evolved)
    return new.get(slot, new.get("0")) if new else None


# Rules.moveset: best4 (default) or last4; a rule change is a progression change.
NO_SLOT_MOVES = {"selfdestruct", "explosion", "memento", "finalgambit", "healingwish",  # the user faints
                 "lunardance", "mistyexplosion",
                 "hyperbeam", "gigaimpact", "solarbeam", "skyattack", "skullbash", "razorwind",  # a turn lost
                 "focuspunch", "dreameater", "snore", "fling", "bide", "spitup", "counter",  # conditional
                 "mirrorcoat", "lastresort"}  # never a slot under best4 (the calc bot cannot value them)
STATUS_MOVES = ["swordsdance", "dragondance", "nastyplot", "calmmind", "bulkup", "growth",
                "agility", "recover", "roost", "morningsun", "synthesis", "moonlight",
                "softboiled", "milkdrink", "slackoff", "rest", "spore", "sleeppowder",
                "hypnosis", "sing", "thunderwave", "stunspore", "willowisp", "toxic",
                "leechseed", "reflect", "lightscreen", "substitute", "protect"]


def legal_pool(species: str, level: int, progress: int | None = None) -> list[str]:
    """Every move `species` may know at `level`: the level-up moves learned so far (a relearner
    teaches any of them), then the TMs and tutors in hand at trainer index `progress` (tms.json;
    None = none; later = preferred on ties), minus BANNED_MOVES and the SPECIES_MOVES limits.
    No egg moves and no pre-evolution-only moves (the user's RR hardcore run, 2026-11-24)."""
    game = load_game()
    ls = game["learnsets"].get(species, {})
    legal = lambda m: m not in BANNED_MOVES and species in SPECIES_MOVES.get(m, (species,))
    pool = list(dict.fromkeys(m for m, lvl in ls.get("levelup", []) if lvl <= level and legal(m)))
    if progress is not None:
        pool += [m for m in ls.get("tm", []) if m not in pool and legal(m)
                 and m in game["tms"] and game["tms"][m]["before_trainer"] <= progress]
    return pool


def encounter_moves(species: str, level: int, rule: str = "best4",
                    progress: int | None = None, dead: set[str] = frozenset(),
                    ability: str | None = None) -> list[str]:
    """A mon's moves at `level` under the moveset rule; progress = the trainer
    index reached (None = no TMs): under best4 the pool also holds every TM
    handed out before it that the species can learn (tms.json; reusable).
    last4 — the last four level-up moves learned (every new move replaced the
            oldest: what a player gets by never declining a move);
    best4 — from every level-up move learned so far: the strongest STAB
            attack, the strongest attack of another type, the next strongest
            attack of a third type, one status move from STATUS_MOVES; slots
            left are filled with the next attacks, then the latest moves.
            Strength = power x accuracy x STAB x category fit (a physical
            move on a special attacker counts atk / spa of it; 2026-09-23,
            with the TM pool raw power chose Hyper Beam / Hydro Pump for
            everyone). Variable-damage
            attacks (Seismic Toss, Low Kick) count as 50 power; NO_SLOT_MOVES
            (self-KO, recharge, conditional) never get a slot. `dead`: attack
            types that fail in this fight (rr_probe.dead_types: Water under
            Desolate Land), never picked under best4. `ability`: the mon's, so
            its move boosts count (ai_util.calc_bot.ability_power: Sharpness,
            Iron Fist, -ate, Adaptability ...; D16, 2026-11-25); Foul Play
            counts as off a base-100 Attack (it uses the target's).
    strength — best4's attacks' summed power (rr_probe.run_ability's tie-break)."""
    game = load_game()
    pool = legal_pool(species, level)
    if rule == "last4" or not pool:
        return pool[-4:]
    pool = legal_pool(species, level, progress)
    from ai_util.calc_bot import ability_power
    from poke_env.data import GenData
    md, sp, sd = game["moves"], game["species"].get(species, {}), GenData.from_gen(9).moves
    stab, bs = set(sp.get("types", [])), sp.get("baseStats", {"atk": 1, "spa": 1})
    ab = to_id_str(ability or "")
    fit = {"Physical": bs["atk"] * (2 if ab in ("hugepower", "purepower") else 1), "Special": bs["spa"]}
    eff = {m: ability_power(ab, {**sd.get(m, {}), "type": md[m]["type"]}, md[m]["power"] or 50) for m in pool}
    mtype = lambda m: eff[m][1]  # noqa: E731  (-ate abilities change Normal moves' type)

    def power(m):  # expected damage scale: power x accuracy x STAB x category fit
        acc = md[m]["accuracy"]
        bp, t, stab_x = eff[m]
        user = 100 if sd.get(m, {}).get("overrideOffensivePokemon") == "target" else fit[md[m]["category"]]
        return bp * (1.0 if acc is True else acc / 100) * (stab_x if t in stab else 1.0) * user / max(fit.values())
    attacks = sorted((m for m in pool if md[m]["category"] != "Status" and m not in NO_SLOT_MOVES
                      and mtype(m) not in dead),
                     key=lambda m: (-power(m), -pool.index(m)))
    picks = []
    for cond in (lambda m: mtype(m) in stab,
                 lambda m: not picks or mtype(m) != mtype(picks[0]),
                 lambda m: mtype(m) not in {mtype(p) for p in picks}):  # coverage
        nxt = next((m for m in attacks if m not in picks and cond(m)), None)
        if nxt:
            picks.append(nxt)
    picks += [m for m in STATUS_MOVES if m in pool][:1]
    picks += [m for m in attacks if m not in picks]
    if rule == "strength":
        return sum(power(m) for m in picks[:4] if m in attacks)
    picks += [m for m in reversed(pool) if m not in picks and m not in NO_SLOT_MOVES and mtype(m) not in dead]
    return picks[:4]


CHOICE_ALL_ATTACKS = False  # RR hardcore (rr_probe.hardcore): no Choice item on a set with a status move


def item_value(m: Mon, item: dict, md: dict, types: set[str]) -> float:
    """What holding `item` is worth to m: the multiplier on its attacks of the
    boosted type (1.2 STAB / 1.1 off-type) or category (Choice locks: x0.85), or
    the item's flat value (Leftovers 1.25); an `nfe` item (Eviolite) only for a mon that
    still evolves; a mega stone only for its own species (3.0: always first). 1.0 = nothing."""
    if "mega" in item:
        return 3.0 if to_id_str(m.species) == item["mega"] else 1.0
    if item.get("nfe") and not load_game()["evolutions"].get(m.species):
        return 1.0
    if "flat" in item:
        return item["flat"]
    attacks = [md[x] for x in m.moves if x in md and md[x]["category"] != "Status"]
    if "type" in item:
        hit = [a for a in attacks if a["type"] == item["type"]]
        return item["mult"] if any(a["type"] in types for a in hit) else (1.0 + (item["mult"] - 1) / 2 if hit else 1.0)
    if item.get("lock") and CHOICE_ALL_ATTACKS and any(md[x]["category"] == "Status" for x in m.moves if x in md):
        return 1.0  # locked into one move, a status move is dead weight
    if any(a["category"] == item["category"] for a in attacks[:1]):  # the strongest attack's category
        return item["mult"] * (0.85 if item.get("lock") else 1.0)
    return 1.0


def mega_stone(species: str, progress: int | None) -> str | None:
    """The mega stone of `species` in hand at trainer `progress` (items.json), if any."""
    if progress is None:
        return None
    sid = to_id_str(species)
    return next((k for k, v in load_game()["items"].items() if v.get("mega") == sid and v["before_trainer"] <= progress), None)


def copies(item: dict, progress: int) -> int:
    """Holders of `item` at trainer `progress`: the most any source in hand by then gives
    (`steps` [[trainer, copies], ...]; a later care package can raise one to six)."""
    return max([n for t, n in item.get("steps", []) if t <= progress] or [item["copies"]])


def assign_items(mons: list[Mon], progress: int | None) -> None:
    """E9: hand the items in hand at trainer `progress` (None = no items) to the
    party, greedily by item_value, one item per mon and `copies` holders per item.
    Deterministic in (party, progress): the risk cache needs no item key."""
    for m in mons:
        m.item = None
    if progress is None:
        return
    game = load_game()
    md, sp = game["moves"], game["species"]
    left = {k: copies(v, progress) for k, v in game["items"].items() if v["before_trainer"] <= progress}
    cands = sorted(((item_value(m, game["items"][k], md, set(sp.get(m.species, {}).get("types", []))), m.level, k, i)
                    for i, m in enumerate(mons) for k in left), reverse=True)
    for value, _, k, i in cands:
        if value > 1.0 and left[k] > 0 and mons[i].item is None:
            mons[i].item, left[k] = k, left[k] - 1


def load_evolutions() -> dict:
    return load_game()["evolutions"]


# Evolution methods without an in-game mechanic here, as progression rules:
FRIENDSHIP_LEVELS = 10   # a friendship evolution fires this many levels after the catch
# every evolution stone: BDSP's Grand Underground opens in Eterna City; RR's Celadon Dept. Store
# sells them all (official sheet, Shops tab; RR has no place / held-item gates here: all = stones)
STONE_GATE = {"bdsp": "Eterna Gym", "rr": "CELADON CITY"}
EVO_GATES = {"leafeon": "Eterna Forest", "glaceon": "Route 217",            # Moss Rock, Ice Rock
             "magnezone": "Route 208", "probopass": "Route 208"}            # Mt. Coronet's field
ITEM_GATES = {"Razor Claw": "Victory Road", "Razor Fang": "Victory Road"}  # Weavile, Gliscor
# trade evolutions never happen (no trading in a Nuzlocke)


@functools.lru_cache(maxsize=None)
def first_index(location: str) -> int:
    """order_index of the first trainer whose location starts with `location` (RR's order has
    no order_index: its position is the trainer index)."""
    return next(e.get("order_index", i) for i, e in enumerate(load_game()["order"]) if e["location"].startswith(location))


def _evo_due(species: str, e: dict, level: int, caught_level, trainer_idx) -> bool:
    t = e["type"]
    if t == "level":
        return e["level"] <= level
    if t == "levelMove":  # Bonsly / Mime Jr.: when the move is learned
        learned = [l for m, l in load_game()["learnsets"].get(species, {}).get("levelup", []) if m == to_id_str(e["move"])]
        return bool(learned) and min(learned) <= level
    if t == "levelFriendship":
        return caught_level is not None and level >= caught_level + FRIENDSHIP_LEVELS
    if t in ("useItem", "levelHold", "levelExtra"):
        gate = (EVO_GATES.get(e["to"]) or ITEM_GATES.get(e["item"] or "") if GAME == "bdsp" else None) or STONE_GATE[GAME]
        return trainer_idx is not None and trainer_idx >= first_index(gate)
    return False


def split_evo_key(species: str) -> tuple[str, str | None]:
    """Split a branch-pinned species key ('eevee>jolteon' -> ('eevee', 'jolteon'); 'bronzor' -> ('bronzor', None))."""
    if ">" in species:
        base, target = species.split(">", 1)
        return base, target or None
    return species, None


def base_species(species: str) -> str:
    """Caught/base species id from a plain or branch-pinned key ('eevee>jolteon' -> 'eevee')."""
    return species.split(">", 1)[0]


def format_species(sp: str, evolved: str | None = None) -> str:
    """Human-readable label showing evolved form when it differs from caught base species."""
    base, target = split_evo_key(sp)
    ev = evolved or target or base
    return f"{base}→{ev}" if ev != base else ev


def evolve_options(species: str, level: int, caught_level: int | None = None,
                   trainer_idx: int | None = None, distinct: bool = True) -> list[str]:
    """All terminal (maximally evolved) species legally reachable from `species` at
    `(level, caught_level, trainer_idx)`. Single-path species return `[ev]`; branched
    species (Eevee, Charcadet, Kirlia, Slowpoke, Hisuian starters, Kubfu, ...) return
    every legal branch. `distinct=True` collapses purely cosmetic forms with identical
    types, base stats, and abilities (e.g. Maushold-Four, Dudunsparce-Three-Segment)."""
    base, forced = split_evo_key(species)
    evos = load_evolutions()
    terminals: list[str] = []
    seen_nodes: set[str] = set()

    def walk(sp: str) -> None:
        if sp in seen_nodes:
            return
        seen_nodes.add(sp)
        due = [e["to"] for e in evos.get(sp, []) if _evo_due(sp, e, level, caught_level, trainer_idx)]
        if not due:
            if sp not in terminals:
                terminals.append(sp)
            return
        for nxt in due:
            walk(nxt)

    walk(base)
    if distinct and len(terminals) > 1:
        sp_meta = load_game()["species"]
        deduped, sigs = [], set()
        for t in terminals:
            info = sp_meta.get(t)
            if not info:
                deduped.append(t)
                continue
            bs = info.get("baseStats", {})
            sig = (tuple(info.get("types", ())),
                   tuple(bs.get(k, 0) for k in STATS),
                   tuple(sorted(info.get("abilities", {}).items())))
            if sig not in sigs:
                sigs.add(sig)
                deduped.append(t)
        terminals = deduped
    if forced and forced in terminals:
        return [forced]
    return terminals


def evo_line(species: str) -> set[str]:
    """Full connected evolution family (ancestors + descendants) of `species`."""
    evos = load_evolutions()
    base = base_species(species)
    parents: dict[str, list[str]] = {}
    for src, lst in evos.items():
        for e in lst:
            parents.setdefault(e["to"], []).append(src)
    line, stack = {base}, [base]
    while stack:
        cur = stack.pop()
        for nxt in [e["to"] for e in evos.get(cur, [])] + parents.get(cur, []):
            if nxt not in line:
                line.add(nxt)
                stack.append(nxt)
    return line


def expand_branches(species_list: list[str], level: int, caught_level: int | None = None,
                    trainer_idx: int | None = None) -> tuple[list[str], dict[str, int]]:
    """Expand any unpinned branched species in `species_list` into `'base>target'` keys.
    Returns `(expanded_keys, slot_limits)` where `slot_limits[base]` is how many copies of
    `base` the box owns (e.g. 1 for a single Eevee, 2 if both Route 5 & Celadon Eevees are owned)."""
    limits: dict[str, int] = {}
    expanded: list[str] = []
    seen: set[str] = set()
    for sp in species_list:
        base, forced = split_evo_key(sp)
        limits[base] = limits.get(base, 0) + 1
        if forced is not None:
            key = sp
            if key not in seen:
                seen.add(key)
                expanded.append(key)
            continue
        opts = evolve_options(base, level, caught_level=caught_level, trainer_idx=trainer_idx)
        keys = [f"{base}>{ev}" for ev in opts] if len(opts) > 1 else [base]
        for k in keys:
            if k not in seen:
                seen.add(k)
                expanded.append(k)
    return expanded, limits


def evolve(species: str, level: int, rng=None, caught_level: int | None = None,
           trainer_idx: int | None = None) -> str:
    """Species after every evolution due at `level`: level-up always; friendship
    needs caught_level, stones / items / places need trainer_idx (the gates
    above); trade never. Branches (Wurmple, Eevee) pick with rng, first without.
    Branch-pinned keys ('eevee>jolteon') evolve directly to the pinned target if legal."""
    base, forced = split_evo_key(species)
    if forced is not None:
        opts = evolve_options(base, level, caught_level=caught_level, trainer_idx=trainer_idx, distinct=False)
        if forced in opts:
            return forced
    evos = load_evolutions()
    sp = base
    while True:
        due = [e["to"] for e in evos.get(sp, []) if _evo_due(sp, e, level, caught_level, trainer_idx)]
        if not due:
            return sp
        sp = rng.choice(due) if rng else due[0]
