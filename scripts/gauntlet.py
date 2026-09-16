"""Gauntlet data + battle logic for the nuzlocke env.

Pure logic (offline-testable) + BattleRunner (needs a Showdown server).
Modes:
  --check        validate BDSP slice + encounters against Showdown learnsets
  --build-moves  generate data/encounter_moves.json from learnsets (node, once)
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

def load_trainers():
    """Return {trainer_id: trainer_dict} and the ordered gauntlet entries."""
    trainers = json.load(open(DATA / "bdsp_trainers.json"))["trainers"]
    order = json.load(open(DATA / "bdsp_gauntlet_order.json"))["entries"]
    return {t["trainer_id"]: t for t in trainers}, order


def load_encounters():
    return json.load(open(DATA / "encounters.json"))


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


@dataclass
class Mon:
    species: str
    level: int
    moves: list[str]
    ability: str | None = None
    item: str | None = None
    alive: bool = True

    def pack(self) -> str:
        """Showdown text -> packed string. Level line survives the roundtrip."""
        lines = [f"{species_name(self.species)} (M)"]
        if self.ability:
            lines.append(f"Ability: {self.ability}")
        if self.item:
            lines[0] += f" @ {self.item}"
        lines.append(f"Level: {self.level}")
        lines += [f"- {m}" for m in self.moves]
        return lines


def pack_team(mons: list[Mon]) -> str:
    text = "\n\n".join("\n".join(m.pack()) for m in mons)
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(text))


def trainer_mons(trainer: dict) -> list[Mon]:
    mons = []
    for p in trainer["party"]:
        moves = p["moves"]
        if not moves:  # 2 records in source data have empty movesets;
            # server rejects moveless mons — learnset fallback
            moves = encounter_moves(p["species"].lower(), p["level"]) or ["tackle"]
        mons.append(Mon(p["species"].lower(), p["level"], moves,
                        p.get("ability"), p.get("item")))
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
        self.policy = None
        self._searchers: dict[tuple[int, int], object] = {}  # one Searcher per width
        if policy:
            from ai_util.battler_policy import load_policy
            self.policy = load_policy(policy, device="cpu")
            self.player = prior_brain(self.policy)
        else:
            self.player = heuristic_brain()

    def searcher(self, width: tuple[int, int] | None):
        """Search brain at `width` = (k, rollouts); None or no prior = the fast player."""
        if width is None or self.policy is None:
            return self.player
        if width not in self._searchers:
            from ai_util.evaluate import Searcher
            self._searchers[width] = Searcher(self.policy, self.enemy, k=width[0],
                                              rollouts=width[1], max_turns=self.max_turns)
        return self._searchers[width]

    def estimate(self, player_packed: str, enemy_packed: str, n: int = 8):
        """Death-probability query: (P(win), [P(death_i)] in packed order, mean
        HP fraction left) under the fast player brain, n fresh-seed battles.
        None on failure."""
        from ai_util.evaluate import evaluate
        try:
            r = evaluate(self.bridge, player_packed, enemy_packed, self.player, self.enemy,
                         n=n, max_turns=self.max_turns)
        except Exception as e:
            import traceback
            print(f"SimBattleRunner.estimate failed: {type(e).__name__}: {e}\n"
                  + "".join(traceback.format_exc().splitlines(True)[-4:]))
            return None
        return r["p_win"], r["p_death"], r["hp"]

    def run(self, player_packed: str, enemy_packed: str, timeout: float = 0.0,
            boss: bool = False, search: tuple[int, int] | None = None):
        """One battle. Returns (won, alive_mask) aligned to player_packed order,
        or None on failure (bridge rebuilt). boss=True searches at the default
        width; search=(k, r) sets the width for this fight (deeper at hard ones)."""
        from ai_util.sim import SimBattle, SimBridge
        if not player_packed or not enemy_packed:
            print("SimBattleRunner: empty packed team")
            return None
        brain = self.searcher(search or (self.search if boss else None))
        try:
            sb = SimBattle(self.bridge, BATTLE_FORMAT, player_packed, enemy_packed)
            sb.play({"p1": brain, "p2": self.enemy}, self.max_turns)
            won, dead = sb.outcome()
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
        return won, [not d for d in dead]


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
const {Dex} = require(path.join(process.argv[1], 'dist', 'sim'));
const maxLevel = Number(process.argv[2]);
// closure over level-up evolutions: evolved forms need learnsets too
const species = new Set(process.argv.slice(3));
const evos = {};
for (const sp of species) {
  evos[sp] = Dex.species.get(sp).evos
    .map(e => Dex.species.get(e)).filter(e => e.evoLevel && !e.evoType)
    .map(e => [e.id, e.evoLevel]);
  for (const [id] of evos[sp]) species.add(id);
}
const out = {};
for (const sp of species) {
  const ls = Learnsets[sp];
  if (!ls) continue;
  const levelup = Object.entries(ls.learnset)
    .filter(([m, gens]) => gens.some(g => /^4L\\d+$/.test(g)))
    .map(([m, gens]) => [m, Math.max(...gens.filter(g => /^4L\\d+$/.test(g)).map(g => parseInt(g.slice(2))))]);
  levelup.sort((a, b) => a[1] - b[1]);
  for (let lvl = 1; lvl <= maxLevel; lvl++) {
    (out[sp] = out[sp] || {})[lvl] = levelup.filter(([, l]) => l <= lvl).slice(-4).map(([m]) => m);
  }
}
console.log(JSON.stringify({moves: out, evos}));
"""


def build_moves(max_level: int = 20) -> None:
    """data/encounter_moves.json (last 4 level-up moves per species and level) and
    data/evolutions.json (species -> [[evolved id, level]]; level evolutions only —
    item / friendship / trade evolutions are not modelled)."""
    enc = load_encounters()
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
    (DATA / "encounter_moves.json").write_text(json.dumps(data["moves"], indent=1))
    (DATA / "evolutions.json").write_text(json.dumps(data["evos"], indent=1))
    print(f"wrote encounter_moves.json for {len(data['moves'])} species (max level "
          f"{max_level}) and evolutions.json for {len(data['evos'])} species")


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
    """data/species.json: name, types, ability slots ({0, 1, H}) and base stats
    for every species a run can hold or face (encounters, their evolutions,
    trainer parties). The env rolls a catch's ability slot from it."""
    from poke_env.data import to_id_str
    ids = set(json.load(open(DATA / "encounter_moves.json")))
    trainers, _ = load_trainers()
    ids |= {to_id_str(species_name(p["species"])) for t in trainers.values() for p in t["party"]}
    raw = subprocess.run(["node", "-e", SPECIES_SCRIPT, str(ROOT / "pokemon-showdown"), *sorted(ids)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    (DATA / "species.json").write_text(json.dumps(data, indent=1))
    print(f"wrote species.json for {len(data)} species ({len(ids) - len(data)} unknown)")


@functools.lru_cache(maxsize=None)
def load_species() -> dict:
    return json.load(open(DATA / "species.json"))


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


def encounter_moves(species: str, level: int) -> list[str]:
    moves = json.load(open(DATA / "encounter_moves.json"))
    by_level = moves.get(species, {})
    return by_level.get(str(min(level, max(int(k) for k in by_level))), []) \
        if by_level else []


def load_evolutions() -> dict:
    return json.load(open(DATA / "evolutions.json"))


def evolve(species: str, level: int, rng=None) -> str:
    """Species after every level evolution reachable at `level` (branching
    evolutions such as Wurmple pick with rng, first branch without)."""
    evos = load_evolutions()
    while True:
        due = [e for e, lvl in evos.get(species, []) if lvl <= level]
        if not due:
            return species
        species = rng.choice(due) if rng else due[0]


# --------------------------------------------------------------------- checks

def check(slice_len: int = 50) -> None:
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
    ap.add_argument("--slice", type=int, default=50)
    ap.add_argument("--max-level", type=int, default=20)
    args = ap.parse_args()
    if args.check:
        check(args.slice)
    if args.build_moves:
        build_moves(args.max_level)
    if args.build_species:
        build_species()
