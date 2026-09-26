"""Gauntlet data + battle logic for the nuzlocke env.

Pure logic (offline-testable) + BattleRunner (needs a Showdown server).
Modes:
  --check        validate BDSP slice + encounters against Showdown learnsets
  --build-moves  generate data/<game>/learnsets.json (level-up + TM compatibility) + evolutions.json
                 + moves.json from the dex (node, once); TM gates: scripts/build_tms.py
"""

import argparse
import functools
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from poke_env.data import to_id_str
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from battle_env import BATTLE_FORMAT, server_config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# GAUNTLET_HARVEST=path records every (player party, enemy team, won) the env
# plays, for BC collection. Unset = no recording: the battler's data must follow
# the progression rule the run layer plays under, so a harvest is always an
# explicit, fresh file (evaluation runs used to append to the old one silently).
HARVEST = Path(os.environ["GAUNTLET_HARVEST"]) if os.environ.get("GAUNTLET_HARVEST") else None
BATTLE_FORMAT = "gen8customgamenogimmicks"

# ---------------------------------------------------------------- data loading

GAME = "bdsp"  # the game folder under data/: everything the env knows about a game


@functools.lru_cache(maxsize=None)
def load_game(name: str = GAME) -> dict:
    """The game folder as one dict: trainers (id -> record), order (gauntlet
    entries), encounters (starter + areas), learnsets (species -> {levelup:
    [[move, level], ...] sorted by level, tm: [moves the species takes from a
    TM]}), evolutions (species -> [[to, level], ...]), species (id -> name,
    types, ability slots, base stats), tms (move -> {tm, before_trainer, from}:
    scripts/build_tms.py). A second game is a second folder with the same
    files and no code change."""
    d = DATA / name
    j = lambda f: json.load(open(d / f))
    return {"trainers": {t["trainer_id"]: t for t in j("trainers.json")["trainers"]},
            "order": j("order.json")["entries"],
            "encounters": j("encounters.json"),
            "learnsets": j("learnsets.json"),
            "evolutions": j("evolutions.json"),
            "species": j("species.json"),
            "moves": j("moves.json"),
            "tms": j("tms.json")}


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
    dex = GenData.from_gen(8).pokedex
    m = re.fullmatch(r"\s*(.+?)\s*\((.+?)\)\s*", species)
    candidates = ([m.group(1) + m.group(2), m.group(1) + m.group(2).split()[0], m.group(1)]
                  if m else [species])
    for c in candidates:
        if to_id_str(c) in dex:
            return dex[to_id_str(c)]["name"]
    return species.capitalize()


STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_NAMES = dict(zip(STATS, ("HP", "Atk", "Def", "SpA", "SpD", "Spe")))
# Player mons: 31 IVs and a flat 85 EVs per stat (the 510 total a trained mon
# carries, spread evenly). The format has no EV limit, so an omitted spread
# would be filled with 252 per stat by the validator (README: the EV bug).
PLAYER_EVS = {k: 85 for k in STATS}
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


@dataclass
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
    chosen: list[str] | None = None    # moves the sim picked (plan --moves sim); kept while the pool holds them

    def pack(self) -> str:
        """Showdown packed string (nick|species|item|ability|moves|nature|evs|gender|
        ivs|shiny|level|happiness). Written directly: poke-env's text roundtrip
        drops an all-zero EV spread, which the validator then fills with 252s."""
        spread = lambda d: ",".join(str(d[k]) for k in STATS)
        return "|".join([species_name(self.species), "", to_id_str(self.item or ""),
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


# ------------------------------------------------------------- battle running

class BattleRunner:
    """Persistent heuristic-vs-heuristic battler on one server port.
    Policy pluggable: p1 can be any poke-env Player subclass."""

    def __init__(self, port: int = 8000, policy_cls=SimpleHeuristicsPlayer,
                 info_mode: str = "full"):
        self.port, self.policy_cls, self.info_mode = port, policy_cls, info_mode
        self._make_players()

    def _make_players(self):
        # NOTE: accept_open_team_sheet is deliberately NOT used — it hangs the
        # challenge (server waits for an interactive accept-button click that
        # bots never send; verified 2026-08-19). Full info comes self-sourced
        # instead: the env knows both teams it constructed.
        # Don't pass loop=: poke-env waits for login in __init__, which deadlocks
        # if the loop isn't already running.
        placeholder = pack_team([Mon("bidoof", 5, ["tackle", "growl"])])
        kw = dict(battle_format=BATTLE_FORMAT,
                  server_configuration=server_config(self.port),
                  log_level=logging.WARNING)
        self.p1 = self.policy_cls(team=ConstantTeambuilder(placeholder), **kw)
        self.p2 = SimpleHeuristicsPlayer(team=ConstantTeambuilder(placeholder),
                                         **kw)

    def run(self, player_packed: str, enemy_packed: str, timeout: float = 60.0,
            boss: bool = False, search: tuple[int, int] | None = None):
        """One battle. Returns (won, alive_mask) aligned to player_packed
        order, or None on failure (players rebuilt). `boss` / `search` are
        ignored here (server backend has no search).

        Timeout matters: a rejected/empty team makes poke-env wait forever
        on the challenge. Combined with per-env rollouts (train_nuzlocke),
        one hung battle no longer stalls sibling envs."""
        import asyncio
        if not player_packed or not enemy_packed:
            print("BattleRunner: empty packed team")
            return None
        self.p1._team = ConstantTeambuilder(player_packed)
        self.p2._team = ConstantTeambuilder(enemy_packed)
        before = set(self.p1.battles)
        try:
            async def _one():
                await asyncio.wait_for(
                    self.p1.battle_against(self.p2, n_battles=1),
                    timeout=timeout)
            asyncio.run(_one())
        except Exception as e:
            print(f"BattleRunner: rebuild after {type(e).__name__}: {e}")
            self._make_players()
            return None
        new = [b for tag, b in self.p1.battles.items() if tag not in before]
        if not new:
            print("BattleRunner: no new battle object")
            return None
        battle = max(new, key=lambda b: b.turn)
        won = bool(battle.won)
        # Alive mask aligned to packed order (nickname holds the species).
        from collections import Counter
        alive_counts = Counter(
            m.species for m in battle.team.values() if not m.fainted)
        alive = []
        for pm in Teambuilder.parse_packed_team(player_packed):
            sid = (pm.nickname or "").lower()
            if alive_counts.get(sid, 0) > 0:
                alive_counts[sid] -= 1
                alive.append(True)
            else:
                alive.append(False)
        return won, alive


class SimBattleRunner:
    """Same contract as BattleRunner, but battles run in-process through
    ai_util.sim (no Showdown server, ~100 battles/s). Player brain is
    SimpleHeuristics (default) or the BC prior at `policy`, optionally with
    top-k rollout search `search=(k, rollouts)` for boss fights."""

    def __init__(self, policy: str | None = None, search: tuple[int, int] | None = None,
                 max_turns: int = 100, info_mode: str = "full"):
        import sys
        sys.path.insert(0, str(ROOT))
        from ai_util.evaluate import heuristic_brain, prior_brain
        from ai_util.sim import SimBridge
        self.info_mode, self.max_turns = info_mode, max_turns
        self.bridge = SimBridge()
        self.enemy = heuristic_brain()
        self.search = search  # default (k, rollouts) for boss fights; the compute dial
        self.policy, self.calc = None, policy == "calc"
        self._searchers: dict[tuple[int, int], object] = {}  # one Searcher per width
        if self.calc:  # the scripted damage-calc bot; search=(k, r) = all legal actions (up to k) as candidates
            from ai_util.calc_bot import calc_brain
            self.player = calc_brain()
        elif policy:
            from ai_util.battler_policy import load_policy
            self.policy = load_policy(policy, device="cpu")
            self.player = prior_brain(self.policy)
        else:
            self.player = heuristic_brain()

    def searcher(self, width: tuple[int, int] | None):
        """Search brain at `width` = (k, rollouts); None or no prior = the fast player."""
        if width is None or (self.policy is None and not self.calc):
            return self.player
        if width not in self._searchers:
            from ai_util.evaluate import Searcher
            self._searchers[width] = Searcher(self.policy, self.enemy, k=width[0], rollouts=width[1],
                                              max_turns=self.max_turns,
                                              rollout=self.player if self.calc else None)
        return self._searchers[width]

    def estimate(self, player_packed: str, enemy_packed: str, n: int = 8, carry: list | None = None):
        """Death-probability query: (P(win), [P(death_i)] in packed order, mean
        HP fraction left) under the fast player brain, n fresh-seed battles.
        None on failure."""
        from ai_util.evaluate import evaluate
        try:
            r = evaluate(self.bridge, player_packed, enemy_packed, self.player, self.enemy,
                         n=n, max_turns=self.max_turns, carry=carry)
        except Exception as e:
            import traceback
            print(f"SimBattleRunner.estimate failed: {type(e).__name__}: {e}\n"
                  + "".join(traceback.format_exc().splitlines(True)[-4:]))
            return None
        return r["p_win"], r["p_death"], r["hp"]

    def estimate_block(self, player_packed: str, enemies_packed: list[str], n: int = 8):
        """estimate() over the League block: the same party through every fight,
        PP carried, HP / status restored between fights, the dead stay dead.
        (P(all won), [P(death_i)], progress) — progress = fights won plus the
        lost fight's KOs, over the block (1.0 = cleared): the HP slot. When
        every party wipes (the fast brain at Cynthia) deaths are all 1 and
        how far a party gets is the only ordering left."""
        from ai_util.sim import SimBattle
        wins, deaths, progress = 0, None, 0.0
        try:
            for _ in range(n):
                carry, won, st, fights = None, True, None, 0.0
                for enemy in enemies_packed:
                    sb = SimBattle(self.bridge, BATTLE_FORMAT, player_packed, enemy, carry=carry)
                    sb.play({"p1": self.player, "p2": self.enemy}, self.max_turns)
                    st = sb.state()
                    sb.close()
                    # between fights: HP / status restored (items), PP carried, the dead stay dead
                    carry = [{"hp": 0 if m["fainted"] else None, "pp": m.get("pp")} for m in st["p1"]]
                    if st["winner"] != "p1":
                        won = False
                        fights += sum(m["fainted"] for m in st["p2"]) / len(st["p2"])
                        break
                    fights += 1
                wins += won
                dead = [m["fainted"] for m in st["p1"]]
                deaths = [d + float(x) for d, x in zip(deaths, dead)] if deaths else [float(x) for x in dead]
                progress += fights / len(enemies_packed)
        except Exception as e:
            print(f"SimBattleRunner.estimate_block failed: {type(e).__name__}: {e}")
            return None
        return wins / n, [d / n for d in deaths], progress / n

    def run(self, player_packed: str, enemy_packed: str, timeout: float = 0.0,
            boss: bool = False, search: tuple[int, int] | None = None, carry: list | None = None):
        """One battle. Returns (won, alive_mask, state) aligned to player_packed
        order — state = per-mon {hp, status, pp} after the fight, what the next
        no-heal fight starts from — or None on failure (bridge rebuilt).
        boss=True searches at the default width; search=(k, r) sets the width
        for this fight (deeper at hard ones); carry = state to start from."""
        from ai_util.sim import SimBattle, SimBridge
        if not player_packed or not enemy_packed:
            print("SimBattleRunner: empty packed team")
            return None
        brain = self.searcher(search or (self.search if boss else None))
        try:
            sb = SimBattle(self.bridge, BATTLE_FORMAT, player_packed, enemy_packed, carry=carry)
            sb.play({"p1": brain, "p2": self.enemy}, self.max_turns)
            st = sb.state()
            won, dead = st["winner"] == "p1", [m["fainted"] for m in st["p1"]]
            state = [{"hp": m["hp"], "status": m.get("status", ""), "pp": m.get("pp")} for m in st["p1"]]
            sb.close()
        except Exception as e:
            import traceback
            print(f"SimBattleRunner: rebuild after {type(e).__name__}: {e}\n"
                  + "".join(traceback.format_exc().splitlines(True)[-4:]))
            try:
                self.bridge.close()
            except Exception:
                pass
            self.bridge = SimBridge()
            return None
        return won, [not d for d in dead], state


_HARVEST_LOCK = None


def record_battle(run_id: str, trainer_id: int | None, info_mode: str,
                  won: bool, player_packed: str, enemy_packed: str):
    global _HARVEST_LOCK
    if HARVEST is None:
        return
    if _HARVEST_LOCK is None:
        import threading
        _HARVEST_LOCK = threading.Lock()
    HARVEST.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "run_id": run_id, "trainer_id": trainer_id, "info": info_mode,
        "won": won, "player_team": player_packed,
        "enemy_team": enemy_packed,
    }) + "\n"
    with _HARVEST_LOCK:
        with open(HARVEST, "a") as f:
            f.write(line)


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
    from poke_env.data import to_id_str
    ids = set(json.load(open(DATA / GAME / "learnsets.json")))
    trainers, _ = load_trainers()
    ids |= {to_id_str(species_name(p["species"])) for t in trainers.values() for p in t["party"]}
    raw = subprocess.run(["node", "-e", SPECIES_SCRIPT, str(ROOT / "pokemon-showdown"), *sorted(ids)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    (DATA / GAME / "species.json").write_text(json.dumps(data, indent=1))
    print(f"wrote species.json for {len(data)} species ({len(ids) - len(data)} unknown)")


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


MOVESET_RULE = "best4"  # or "last4" (plan_gauntlet --moveset); a rule change is a progression change
# status moves worth a slot under best4, in preference order
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


def encounter_moves(species: str, level: int, rule: str | None = None,
                    progress: int | None = None, k: int | None = 4) -> list[str]:
    """A mon's moves at `level` under the moveset rule; progress = the trainer
    index reached (None = no TMs): under best4 the pool also holds every TM
    handed out before it that the species can learn (tms.json; reusable).
    k=None returns the whole eligible pool ranked (the first four are the
    moveset): the candidates when the sim chooses the moves (plan --moves sim).
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
            (self-KO, recharge, conditional) never get a slot."""
    game = load_game()
    ls = game["learnsets"].get(species, {})
    pool = list(dict.fromkeys(m for m, lvl in ls.get("levelup", []) if lvl <= level))
    if (rule or MOVESET_RULE) == "last4" or not pool:
        return pool[-4:] if k else pool
    if progress is not None:  # TMs in hand, after the level-up moves (later = preferred on ties)
        pool += [m for m in ls.get("tm", []) if m not in pool
                 and m in game["tms"] and game["tms"][m]["before_trainer"] <= progress]
    md, sp = game["moves"], game["species"].get(species, {})
    stab, bs = set(sp.get("types", [])), sp.get("baseStats", {"atk": 1, "spa": 1})
    fit = {"Physical": bs["atk"], "Special": bs["spa"]}

    def power(m):  # expected damage scale: power x accuracy x STAB x category fit
        acc = md[m]["accuracy"]
        return ((md[m]["power"] or 50) * (1.0 if acc is True else acc / 100) * (1.5 if md[m]["type"] in stab else 1.0)
                * fit[md[m]["category"]] / max(bs["atk"], bs["spa"]))
    attacks = sorted((m for m in pool if md[m]["category"] != "Status" and m not in NO_SLOT_MOVES),
                     key=lambda m: (-power(m), -pool.index(m)))
    picks = []
    for cond in (lambda m: md[m]["type"] in stab,
                 lambda m: not picks or md[m]["type"] != md[picks[0]]["type"],
                 lambda m: md[m]["type"] not in {md[p]["type"] for p in picks}):  # coverage
        nxt = next((m for m in attacks if m not in picks and cond(m)), None)
        if nxt:
            picks.append(nxt)
    picks += [m for m in STATUS_MOVES if m in pool][:1]
    picks += [m for m in attacks if m not in picks]
    picks += [m for m in reversed(pool) if m not in picks and m not in NO_SLOT_MOVES]
    return picks[:k] if k else picks


def load_evolutions() -> dict:
    return load_game()["evolutions"]


# Evolution methods without an in-game mechanic here, as progression rules:
FRIENDSHIP_LEVELS = 10   # a friendship evolution fires this many levels after the catch
STONE_GATE = "Eterna Gym"  # the Grand Underground (every evolution stone) opens in Eterna City
EVO_GATES = {"leafeon": "Eterna Forest", "glaceon": "Route 217",            # Moss Rock, Ice Rock
             "magnezone": "Route 208", "probopass": "Route 208"}            # Mt. Coronet's field
ITEM_GATES = {"Razor Claw": "Victory Road", "Razor Fang": "Victory Road"}  # Weavile, Gliscor
# trade evolutions never happen (no trading in a Nuzlocke)


@functools.lru_cache(maxsize=None)
def first_index(location: str) -> int:
    """order_index of the first trainer whose location starts with `location`."""
    return next(e["order_index"] for e in load_game()["order"] if e["location"].startswith(location))


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
        gate = EVO_GATES.get(e["to"]) or ITEM_GATES.get(e["item"] or "") or STONE_GATE
        return trainer_idx is not None and trainer_idx >= first_index(gate)
    return False


def evolve(species: str, level: int, rng=None, caught_level: int | None = None,
           trainer_idx: int | None = None) -> str:
    """Species after every evolution due at `level`: level-up always; friendship
    needs caught_level, stones / items / places need trainer_idx (the gates
    above); trade never. Branches (Wurmple, Eevee) pick with rng, first without."""
    evos = load_evolutions()
    while True:
        due = [e["to"] for e in evos.get(species, []) if _evo_due(species, e, level, caught_level, trainer_idx)]
        if not due:
            return species
        species = rng.choice(due) if rng else due[0]


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
            from poke_env.data import to_id_str
            if to_id_str(species_name(p["species"])) not in known:
                print(f"BAD species {p['species']} ({e['trainer_name']})")
                problems += 1
    enc_species = ({e["species"] for r in enc["routes"] for e in r["encounters"]}
                   | set(enc["starter"]))
    for s in sorted(enc_species):
        if s not in known:
            print(f"BAD encounter species {s}")
            problems += 1
        else:
            w = sum(e["weight"] for e in enc["routes"][0]["encounters"])
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
