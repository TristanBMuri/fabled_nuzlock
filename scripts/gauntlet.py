"""Gauntlet data + battle logic for the nuzlocke env.

Pure logic (offline-testable) + BattleRunner (needs a Showdown server).
Modes:
  --check        validate BDSP slice + encounters against Showdown learnsets
  --build-moves  generate data/encounter_moves.json from learnsets (node, once)
"""

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from battle_env import BATTLE_FORMAT, server_config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HARVEST = DATA / "teams" / "gauntlet_harvest" / "battles.jsonl"
BATTLE_FORMAT = "gen8customgamenogimmicks"

# ---------------------------------------------------------------- data loading

def load_trainers():
    """Return {trainer_id: trainer_dict} and the ordered gauntlet entries."""
    trainers = json.load(open(DATA / "bdsp_trainers.json"))["trainers"]
    order = json.load(open(DATA / "bdsp_gauntlet_order.json"))["entries"]
    return {t["trainer_id"]: t for t in trainers}, order


def load_encounters():
    return json.load(open(DATA / "encounters.json"))


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
        lines = [f"{self.species.capitalize()} (M)"]
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

    def __init__(self, port: int = 8020, policy_cls=SimpleHeuristicsPlayer,
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

    def run(self, player_packed: str, enemy_packed: str, timeout: float = 60.0):
        """One battle. Returns (won, alive_mask) aligned to player_packed
        order, or None on failure (players rebuilt).

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


_HARVEST_LOCK = None


def record_battle(run_id: str, trainer_id: int | None, info_mode: str,
                  won: bool, player_packed: str, enemy_packed: str):
    global _HARVEST_LOCK
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
const {Learnsets} = require(require('path').join(process.argv[1], 'dist', 'data', 'learnsets.js'));
const maxLevel = Number(process.argv[2]);
const out = {};
for (const sp of process.argv.slice(3)) {
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
console.log(JSON.stringify(out));
"""


def build_moves(max_level: int = 20) -> None:
    enc = load_encounters()
    species = sorted({e["species"] for r in enc["routes"] for e in r["encounters"]}
                     | set(enc["starter"]))
    raw = subprocess.run(
        ["node", "-e", LEARNSET_SCRIPT, str(ROOT / "pokemon-showdown"),
         str(max_level), *species],
        capture_output=True, text=True, check=True,
    ).stdout
    moves = json.loads(raw)
    missing = [s for s in species if s not in moves]
    if missing:
        print(f"WARN: no learnset data for {missing}")
    (DATA / "encounter_moves.json").write_text(json.dumps(moves, indent=1))
    print(f"wrote encounter_moves.json for {len(moves)} species "
          f"(max level {max_level})")


def encounter_moves(species: str, level: int) -> list[str]:
    moves = json.load(open(DATA / "encounter_moves.json"))
    by_level = moves.get(species, {})
    return by_level.get(str(min(level, max(int(k) for k in by_level))), []) \
        if by_level else []


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
            if p["species"].lower() not in known:
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
    ap.add_argument("--slice", type=int, default=50)
    ap.add_argument("--max-level", type=int, default=20)
    args = ap.parse_args()
    if args.check:
        check(args.slice)
    if args.build_moves:
        build_moves(args.max_level)
