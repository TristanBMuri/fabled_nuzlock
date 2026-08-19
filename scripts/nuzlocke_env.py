"""Nuzlocke gauntlet env: run-level decisions (party selection, encounters)
around battles played by a pluggable policy (heuristic by default).

One env.step = one phase decision (ENCOUNTER or PRE_BATTLE); the battle runs
to completion inside the PRE_BATTLE confirm step. Gymnasium API.
"""

import random
import uuid

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from battle_env import SPECIES  # pool species (82) — union below with BDSP
from gauntlet import (Mon, encounter_moves, load_encounters, load_trainers,
                      pack_team, record_battle, trainer_mons)
from gauntlet import BattleRunner

MAX_ROSTER = 12
MAX_PARTY = 6
N_META_ACTIONS = 1 + MAX_ROSTER
BREAKPOINTS = ["none", "gym_leader", "elite_four", "champion"]
SLICE_LEN = 50  # trainers per episode (through gym 1+ for v1)

TRAINERS, ORDER = load_trainers()


def build_vocab():
    species = set(SPECIES)
    for e in ORDER[:SLICE_LEN]:
        for p in TRAINERS[e["trainer_id"]]["party"]:
            species.add(p["species"].lower())
    enc = load_encounters()
    species |= {e["species"] for r in enc["routes"] for e in r["encounters"]}
    species |= set(enc["starter"])
    return sorted(species)


SPECIES_V2 = build_vocab()
V = len(SPECIES_V2)
IDX = {s: i for i, s in enumerate(SPECIES_V2)}
# obs: roster(12 x [V+3]) + trainer([4+6]) + per-opp-mon(6 x [V+1]) + phase(2) + 2
OBS_DIM = 12 * (V + 3) + 10 + 6 * (V + 1) + 2 + 2


class NuzlockeEnv(Env):
    def __init__(self, port: int = 8020, info_mode: str = "full",
                 seed: int | None = None):
        self.info_mode = info_mode
        self.rng = random.Random(seed)
        self.runner = BattleRunner(port=port, info_mode=info_mode)
        self.encounters = load_encounters()["routes"]
        self.observation_space = Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = Discrete(N_META_ACTIONS)
        self._reset_state()

    # ------------------------------------------------------------- lifecycle
    def _reset_state(self):
        self.run_id = uuid.uuid4().hex[:8]
        enc = load_encounters()
        starter = self.rng.choice(enc["starter"])
        ability = enc.get("abilities", {}).get(starter)
        self.roster = [Mon(starter, 5, encounter_moves(starter, 5),
                           ability=ability)]
        self.party = []  # indices into roster
        self.trainer_idx = 0
        self.max_enemy_level = 0
        self.phase = "ENCOUNTER"  # first route unlocks before trainer 0
        self.pending_encounter: Mon | None = None
        self._route_cursor = 0  # routes consumed
        self.done = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_state()
        self._maybe_sample_encounter()
        return self._obs(), {"action_mask": self._mask()}

    # ------------------------------------------------------------ encounters
    def _maybe_sample_encounter(self):
        """Fire the next route's encounter if its trainer gate has arrived."""
        self.pending_encounter = None
        abilities = load_encounters().get("abilities", {})
        while (self._route_cursor < len(self.encounters)
               and self.encounters[self._route_cursor]["before_trainer"]
               <= self.trainer_idx):
            route = self.encounters[self._route_cursor]
            self._route_cursor += 1
            table = route["encounters"]
            pick = self.rng.choices(
                table, weights=[e["weight"] for e in table])[0]
            lvl = self.rng.randint(*pick["level"])
            mon = Mon(pick["species"], lvl, encounter_moves(pick["species"], lvl),
                      ability=abilities.get(pick["species"]))
            if len(self.roster) < MAX_ROSTER:
                self.pending_encounter = mon  # offer it (last one wins)
                self.phase = "ENCOUNTER"

    # ---------------------------------------------------------------- battle
    def _run_battle(self):
        party = [self.roster[i] for i in self.party]
        entry = ORDER[self.trainer_idx]
        enemy = trainer_mons(TRAINERS[entry["trainer_id"]])
        self.max_enemy_level = max(self.max_enemy_level, entry["max_level"])
        for m in self.roster:  # player level rule: match max, never decrease
            if m.level < self.max_enemy_level:
                m.level = self.max_enemy_level
                m.moves = encounter_moves(m.species, m.level) or m.moves
        player_packed = pack_team(party)
        enemy_packed = pack_team(enemy)
        result = self.runner.run(player_packed, enemy_packed)
        if result is None:  # rebuild happened; retry once
            result = self.runner.run(player_packed, enemy_packed)
        if result is None:
            raise RuntimeError("battle failed twice")
        won, alive = result
        record_battle(self.run_id, entry["trainer_id"], self.info_mode,
                      won, player_packed, enemy_packed)
        return won, alive

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        v = np.zeros(OBS_DIM, dtype=np.float32)
        o = 0
        for i, mon in enumerate(self.roster[:MAX_ROSTER]):
            v[o + IDX.get(mon.species, 0)] = 1.0
            v[o + V] = min(mon.level, 100) / 100.0
            v[o + V + 1] = 1.0 if mon.alive else 0.0
            v[o + V + 2] = 1.0 if i in self.party else 0.0
            o += V + 3
        entry = ORDER[self.trainer_idx] if self.trainer_idx < SLICE_LEN else None
        if entry is not None:
            v[o] = entry["max_level"] / 100.0
            v[o + 1] = entry["avg_level"] / 100.0
            v[o + 2] = entry["team_size"] / 6.0
            v[o + 3 + BREAKPOINTS.index(entry["breakpoint"])] = 1.0
        o += 10
        if entry is not None and self.info_mode in ("revealed", "full"):
            for j, pm in enumerate(TRAINERS[entry["trainer_id"]]["party"][:6]):
                v[o + IDX.get(pm["species"].lower(), 0)] = 1.0
                v[o + V] = pm["level"] / 100.0
                o += V + 1
        else:
            o += 6 * (V + 1)
        v[o] = 1.0 if self.phase == "ENCOUNTER" else 0.0
        v[o + 1] = 1.0 if self.phase == "PRE_BATTLE" else 0.0
        v[o + 2] = self.trainer_idx / SLICE_LEN
        v[o + 3] = 1.0  # bias
        return v

    def _mask(self) -> np.ndarray:
        m = np.zeros(N_META_ACTIONS, dtype=np.int64)
        if self.phase == "ENCOUNTER":
            m[0] = 1  # skip
            if (self.pending_encounter is not None
                    and len(self.roster) < MAX_ROSTER):
                m[1] = 1  # catch
        else:  # PRE_BATTLE
            if self.party:
                m[0] = 1  # done -> fight
            if len(self.party) < MAX_PARTY:
                for i, mon in enumerate(self.roster[:MAX_ROSTER]):
                    if mon.alive and i not in self.party:
                        m[1 + i] = 1
        return m

    # ------------------------------------------------------------------ step
    def step(self, action):
        reward = 0.0
        if self.phase == "ENCOUNTER":
            if action == 1 and self.pending_encounter is not None:
                self.roster.append(self.pending_encounter)
                self.pending_encounter = None
            self.phase = "PRE_BATTLE"
        else:  # PRE_BATTLE
            if action == 0:  # confirm -> battle
                won, alive = self._run_battle()
                party = [self.roster[i] for i in self.party]
                for mon, a in zip(party, alive):
                    if not a:
                        mon.alive = False
                        reward -= 1.0
                if won:
                    reward += 1.0
                    self.trainer_idx += 1
                    if self.trainer_idx >= SLICE_LEN:
                        reward += 10.0
                        self.done = True
                    else:
                        self.party = []
                        self._maybe_sample_encounter()
                        if self.pending_encounter is None:
                            self.phase = "PRE_BATTLE"
                else:
                    # A lost battle ends the run (nuzlocke: you'd reload a
                    # save; for RL, retry loops would dilute credit). Wipe
                    # penalty only when everyone died.
                    if not any(m.alive for m in self.roster):
                        reward -= 5.0
                    self.done = True
            else:
                i = action - 1
                if i < len(self.roster) and self.roster[i].alive:
                    self.party.append(i)
        return (self._obs(), reward, self.done, False,
                {"action_mask": self._mask()})

    def render(self):
        roster = ", ".join(f"{m.species}@{m.level}{'†' if not m.alive else ''}"
                           for m in self.roster)
        entry = (ORDER[self.trainer_idx]["trainer_name"]
                 if self.trainer_idx < SLICE_LEN else "CLEARED")
        return (f"[{self.phase}] trainer {self.trainer_idx}/{SLICE_LEN} "
                f"({entry}) | roster: {roster} | party: {self.party}")
