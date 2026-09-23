"""Nuzlocke gauntlet env: run-level decisions (party selection, encounters)
around battles played by a pluggable policy (heuristic by default).

One env.step = one phase decision (ENCOUNTER or PRE_BATTLE); the battle runs
to completion inside the PRE_BATTLE confirm step. Gymnasium API.
"""

import random
import uuid
from dataclasses import dataclass

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from battle_env import SPECIES  # pool species (82) — union below with BDSP
from gauntlet import (Mon, ability_slots, encounter_moves, evolve, evolved_ability,
                      load_encounters, load_evolutions, load_trainers, pack_team,
                      player_evs, record_battle, roll_ability, trainer_mons)
from gauntlet import BattleRunner, SimBattleRunner

MAX_ROSTER = 12  # obs/mask slots for the legacy PPO layout; Rules.box sets the real box
MAX_PARTY = 6
N_META_ACTIONS = 1 + MAX_ROSTER


@dataclass(frozen=True)
class Rules:
    """The Nuzlocke ruleset, as a player would state it. Printed in every table
    header; a change to any field is a progression change (regenerate the
    harvest and re-clone the prior)."""
    level_rule: str = "cap"      # cap: every mon at the next boss's max level (hardcore); match: max enemy seen
    level_margin: int = 0        # levels above the rule (XP proxy)
    box: int = 50                # living mons the box holds (50 > starter + 45 areas = unbounded)
    dupes_clause: bool = True    # an area's encounter is rerolled among families never owned (none left: no catch)
    moveset: str = "best4"       # best4: strongest STAB / other-type / next attack + a status move; last4: last learned
    tms: bool = True             # best4 also over the TMs handed out so far (reusable; data/bdsp/tms.json)
    evs: str = "flat"            # player EVs: flat 85 each, or trained (252 attack / 252 Spe / 4 HP)

    def __str__(self):
        return (f"level_rule={self.level_rule} margin={self.level_margin} box={self.box} "
                f"dupes={self.dupes_clause} moveset={self.moveset} tms={self.tms} evs={self.evs}")
BREAKPOINTS = ["none", "gym_leader", "elite_four", "champion"]
SLICE_LEN = 21  # trainers per episode: through Gym Leader Roark (order_index 20 in the route order)
EVOS = load_evolutions()
STARTERS = {"chimchar", "piplup", "turtwig"}
STARTERS |= {e["to"] for s in list(STARTERS) for e in EVOS.get(s, [])}
STARTERS |= {e["to"] for s in list(STARTERS) for e in EVOS.get(s, [])}  # second stage

TRAINERS, ORDER = load_trainers()


def build_vocab():
    species = set(SPECIES)
    for e in ORDER[:SLICE_LEN]:
        for p in TRAINERS[e["trainer_id"]]["party"]:
            species.add(p["species"].lower())
    enc = load_encounters()
    species |= {e["species"] for r in enc["routes"] for e in r["encounters"]}
    species |= set(enc["starter"])
    species |= set(EVOS)  # evolved forms the roster can reach
    return sorted(species)


SPECIES_V2 = build_vocab()
V = len(SPECIES_V2)
IDX = {s: i for i, s in enumerate(SPECIES_V2)}
# obs: roster(12 x [V+3]) + trainer([4+6]) + per-opp-mon(6 x [V+1]) + phase(2) + 2
#      + risk(7): P(win) + P(death) per party slot for the current party vs the
#        next trainer, from the sim (zeros when no party / server backend)
N_RISK = 1 + MAX_PARTY
RISK_BATTLES = 8
# A fight is "hard" (deep search width) when the fielded party's risk query shows
# a loss, a death, or less than this fraction of the party's HP left on average.
DEEP_HP = 0.5
obs_dim = lambda max_roster: max_roster * (V + 3) + 10 + 6 * (V + 1) + 2 + 2 + N_RISK
OBS_DIM = obs_dim(MAX_ROSTER)


class NuzlockeEnv(Env):
    def __init__(self, port: int = 8000, info_mode: str = "full",
                 seed: int | None = None, backend: str = "sim",
                 policy: str | None = None, search: tuple[int, int] | None = None,
                 rules: Rules = Rules(),
                 risk_features: bool = True, search_all: bool = False,
                 slice_len: int = SLICE_LEN, deep: tuple[int, int] | None = None):
        """backend 'sim' (in-process, default) or 'server' (Showdown on `port`).
        slice_len: trainers per episode (21 = through Roark, 59 = through Gardenia, 167 = Maylene).
        policy selects the sim battler (None = SimpleHeuristics); search=(k, r)
        switches boss fights (breakpoint trainers) to rollout search, search_all
        every fight. deep=(k, r): wider search at breakpoints and at any fight
        the risk query flags as hard (see DEEP_HP) — the compute dial per trainer.
        rules: the Nuzlocke ruleset (see Rules)."""
        self.info_mode = info_mode
        self.risk_features = risk_features  # off = no sim queries in _obs (harvesting)
        self.search_all = search_all  # search on every trainer, not just breakpoints
        self.deep = deep
        self.n_deep = 0  # fights played at the deep width (cost accounting)
        self.slice_len = slice_len
        self.rules = rules
        self.max_roster = rules.box  # living mons the box holds; the obs and mask are sized by it
        self.rng = random.Random(seed)
        if backend == "sim":
            self.runner = SimBattleRunner(policy=policy, search=search, info_mode=info_mode)
        else:
            self.runner = BattleRunner(port=port, info_mode=info_mode)
        self.encounters = load_encounters()["routes"]
        self.observation_space = Box(-np.inf, np.inf, (obs_dim(self.max_roster),), np.float32)
        self.action_space = Discrete(1 + self.max_roster)
        self._reset_state()

    # ------------------------------------------------------------- lifecycle
    def _reset_state(self):
        self.run_id = uuid.uuid4().hex[:8]
        self.trainer_idx = 0  # before the starter: its moves depend on the TMs in hand
        starter = self.rng.choice(load_encounters()["starter"])
        # starters come with ability slot 0; wild catches roll slot 0 or 1
        # (roll_ability) and evolution keeps the slot (evolved_ability)
        self.roster = [Mon(starter, 5, encounter_moves(starter, 5, self.rules.moveset, self._progress),
                           ability=ability_slots(starter).get("0"), caught_level=5)]
        self.graveyard: list[Mon] = []  # dead mons leave the roster (box slots free up)
        self.party = []  # indices into roster
        rival = {"turtwig": "chimchar", "chimchar": "piplup", "piplup": "turtwig"}[starter]
        self._rival_family = {rival} | {e["to"] for e in EVOS.get(rival, [])}
        self._rival_family |= {e["to"] for s in list(self._rival_family) for e in EVOS.get(s, [])}
        self._skip_other_rivals()
        self.max_enemy_level = 0
        self.phase = "ENCOUNTER"  # first route unlocks before trainer 0
        self.pending: list[list[Mon]] = []  # the offer being answered: one pick per method
        self._queue: list[dict] = []  # areas whose gate has arrived, rolled one at a time
        self.carry: dict[int, dict] | None = None  # roster index -> {hp, status, pp}: no heal since the last fight
        self.dropped: list[str] = []  # offers that found no box slot (species), this run
        self._route_cursor = 0  # routes consumed
        self.done = False
        self._risk_cache: dict = {}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_state()
        self._maybe_sample_encounter()
        return self._obs(), {"action_mask": self._mask()}

    # ------------------------------------------------------------ encounters
    def _owned_families(self) -> set[str]:
        """Species families ever owned (alive or dead), for the dupes clause."""
        fam = set()
        stack = [m.species for m in self.roster + self.graveyard]
        while stack:
            sp = stack.pop()
            if sp not in fam:
                fam.add(sp)
                stack += [e["to"] for e in EVOS.get(sp, [])]
        # an evolved catch also claims its pre-evolutions, all the way down
        # (a Crobat used to leave Zubat catchable: one step only)
        pre = {p for p, evs in EVOS.items() if any(e["to"] in fam for e in evs)} - fam
        while pre:
            fam |= pre
            pre = {p for p, evs in EVOS.items() if any(e["to"] in fam for e in evs)} - fam
        return fam

    def _skip_other_rivals(self):
        """The trainer table has one rival record per starter choice; only the
        one whose starter counters ours is fought (Turtwig -> Chimchar rival...)."""
        while self.trainer_idx < self.slice_len:
            e = ORDER[self.trainer_idx]
            if e["trainer_name"] != "Pokémon Trainer Cedric":
                break
            party = {p["species"].lower() for p in TRAINERS[e["trainer_id"]]["party"]}
            if party & self._rival_family:
                break
            self.trainer_idx += 1

    def _maybe_sample_encounter(self):
        """Queue every area whose trainer gate has arrived; the ENCOUNTER phase
        answers them one at a time."""
        while (self._route_cursor < len(self.encounters)
               and self.encounters[self._route_cursor]["before_trainer"]
               <= self.trainer_idx):
            route = self.encounters[self._route_cursor]
            self._route_cursor += 1
            if len(self.roster) + len(self.pending) + len(self._queue) < self.max_roster:
                self._queue.append(route)
            else:
                self.dropped.append(route["name"])
        self._roll_next()
        if self.pending:
            self.phase = "ENCOUNTER"

    def _roll_next(self):
        """Roll the next queued area's first encounter once the previous offer is
        answered, so the dupes clause sees that catch (two water areas opening
        at one trainer both rolled Magikarp when rolled together)."""
        while self._queue and not self.pending:
            route = self._queue.pop(0)
            owned = self._owned_families() if self.rules.dupes_clause else set()
            alts = []  # one roll per method (grass, rod, surf, gift): the run layer picks the method
            methods = sorted({e.get("method", "walk") for e in route["encounters"]})
            tables = {m: [e for e in route["encounters"] if e.get("method", "walk") == m] for m in methods}
            if self.rules.dupes_clause:  # a method whose whole pool is owned is not offered;
                tables = {m: t for m, t in ((m, [e for e in t if e["species"] not in owned])
                                            for m, t in tables.items()) if t}
                if not tables:  # every family here is owned: no catch in this area
                    continue
            for method, table in tables.items():
                pick = self.rng.choices(table, weights=[e["weight"] for e in table])[0]
                lvl = self.rng.randint(*pick["level"])
                alts.append(Mon(pick["species"], lvl, encounter_moves(pick["species"], lvl, self.rules.moveset, self._progress),
                                ability=roll_ability(pick["species"], self.rng), caught_level=lvl))
            self.pending.append(alts)

    # ---------------------------------------------------------------- battle
    def _apply_level_rule(self, entry):
        self.max_enemy_level = max(self.max_enemy_level, entry["max_level"])
        cap = self.max_enemy_level
        # a no-heal block (the League) has one cap: its highest level (Cynthia's
        # 66), reached before Aaron — there is no leveling between the E4 fights
        j = self.trainer_idx
        while j + 1 < self.slice_len and ORDER[j + 1].get("no_heal"):
            j += 1
        block_cap = max(ORDER[k]["max_level"] for k in range(self.trainer_idx, j + 1))
        self.max_enemy_level = max(self.max_enemy_level, block_cap)
        cap = self.max_enemy_level
        if self.rules.level_rule == "cap":
            boss = next((e for e in ORDER[self.trainer_idx:self.slice_len]
                         if e["breakpoint"] != "none"), ORDER[self.slice_len - 1])
            cap = max(cap, boss["max_level"])
        target = min(cap + self.rules.level_margin, 100)
        for m in self.roster:  # player level rule: match max (+margin), never decrease
            if m.level < target:
                m.level = target
                evolved = evolve(m.species, m.level, self.rng, m.caught_level, self.trainer_idx)
                if evolved != m.species:
                    m.ability = evolved_ability(m.species, m.ability, evolved)
                    m.species = evolved
            # moves follow the level and, with TMs, the trainer reached (a TM lands the fight after)
            m.moves = encounter_moves(m.species, m.level, self.rules.moveset, self._progress) or m.moves
            m.evs = player_evs(m.species, self.rules.evs)

    @property
    def _progress(self) -> int | None:
        """Trainer index for the TM pool (None = TMs off)."""
        return self.trainer_idx if self.rules.tms else None

    @property
    def party_locked(self) -> bool:
        """Inside a no-heal block (the League after Aaron): the party stays as
        fielded and starts each fight where the last one left it."""
        return self.carry is not None

    def _carry_list(self, party_idx):
        return [self.carry.get(i) for i in party_idx] if self.carry else None

    def _packed_matchup(self, party_idx):
        """(player_packed, enemy_packed, entry) for a candidate party vs the
        next trainer, with the level rule applied."""
        entry = ORDER[self.trainer_idx]
        self._apply_level_rule(entry)
        party = [self.roster[i] for i in party_idx]
        enemy = trainer_mons(TRAINERS[entry["trainer_id"]])
        return pack_team(party), pack_team(enemy), entry

    def _block(self) -> list[int] | None:
        """Indices of the no-heal block the next fight opens (Aaron + the rest of
        the League), or None: the party chosen here fights all of them."""
        i = self.trainer_idx
        if self.party_locked or i + 1 >= self.slice_len or not ORDER[i + 1].get("no_heal"):
            return None
        j = i + 1
        while j < self.slice_len and ORDER[j].get("no_heal"):
            j += 1
        return list(range(i, j))

    def risk(self, party_idx=None, n: int = RISK_BATTLES):
        """(P(win), [P(death_i)], mean HP fraction left) for a party (default:
        current) vs the next trainer — or vs the whole no-heal block it opens —
        from n battles under the fast brain. Cached per (party, trainer, n)."""
        party_idx = tuple(self.party if party_idx is None else party_idx)
        key = (party_idx, self.trainer_idx, n)
        if key not in self._risk_cache:
            est = None
            if party_idx and self.trainer_idx < self.slice_len and hasattr(self.runner, "estimate"):
                pp, ep, _ = self._packed_matchup(party_idx)
                block = self._block()
                if pp and ep and block:
                    enemies = [pack_team(trainer_mons(TRAINERS[ORDER[k]["trainer_id"]])) for k in block]
                    est = self.runner.estimate_block(pp, enemies, n=n)
                elif pp and ep:
                    est = self.runner.estimate(pp, ep, n=n, carry=self._carry_list(party_idx))
            self._risk_cache[key] = est or (0.0, [0.0] * len(party_idx), 0.0)
        return self._risk_cache[key]

    def _hard(self) -> bool:
        """Deep-search trigger: the fielded party's risk query (free when the
        planner already scored this party) shows a loss, a death, or a close fight."""
        p_win, p_death, hp = self.risk()
        return p_win < 1.0 or max(p_death, default=0.0) > 0.0 or hp < DEEP_HP

    def _run_battle(self):
        player_packed, enemy_packed, entry = self._packed_matchup(self.party)
        if not player_packed or not enemy_packed:
            return None
        boss = self.search_all or entry["breakpoint"] != "none"
        width = None
        if self.deep and (entry["breakpoint"] != "none" or self._hard()):
            width = self.deep
            self.n_deep += 1
        carry = self._carry_list(self.party)
        result = self.runner.run(player_packed, enemy_packed, boss=boss, search=width, carry=carry)
        if result is None:  # rebuild happened; retry once
            result = self.runner.run(player_packed, enemy_packed, boss=boss, search=width, carry=carry)
        if result is None:
            return None
        won, alive, state = result
        record_battle(self.run_id, entry["trainer_id"], self.info_mode,
                      won, player_packed, enemy_packed)
        return won, alive, state

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        v = np.zeros(obs_dim(self.max_roster), dtype=np.float32)
        o = 0
        for i, mon in enumerate(self.roster[:self.max_roster]):
            v[o + IDX.get(mon.species, 0)] = 1.0
            v[o + V] = min(mon.level, 100) / 100.0
            v[o + V + 1] = 1.0 if mon.alive else 0.0
            v[o + V + 2] = 1.0 if i in self.party else 0.0
            o += V + 3
        entry = ORDER[self.trainer_idx] if self.trainer_idx < self.slice_len else None
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
        v[o + 2] = self.trainer_idx / self.slice_len
        v[o + 3] = 1.0  # bias
        o += 4
        if self.risk_features and self.phase == "PRE_BATTLE" and self.party and not self.done:
            p_win, p_death, _ = self.risk()
            v[o] = p_win
            for j, p in enumerate(p_death[:MAX_PARTY]):
                v[o + 1 + j] = p
        return v

    def _mask(self) -> np.ndarray:
        m = np.zeros(1 + self.max_roster, dtype=np.int64)
        if self.phase == "ENCOUNTER":
            m[0] = 1  # skip
            if self.pending and len(self.roster) < self.max_roster:
                m[1:1 + len(self.pending[0])] = 1  # catch alternative k
        else:  # PRE_BATTLE — nuzlocke rule: field everything you can.
            alive_idx = [i for i, mon in enumerate(self.roster[:self.max_roster])
                         if mon.alive]
            if not alive_idx:
                # No mons left — only "confirm", which ends the run (no battle).
                # Without this, min(0, 6)==0 made empty-party confirm legal and
                # crashed workers on pack_team([]) (seen run5 update ~330).
                m[0] = 1
            elif self.party_locked:
                m[0] = 1  # the League block: the fielded party fights on as it stands
            else:
                need = min(len(alive_idx), MAX_PARTY)
                if len(self.party) >= need and len(self.party) > 0:
                    m[0] = 1
                if len(self.party) < MAX_PARTY:
                    for i in alive_idx:
                        if i not in self.party:
                            m[1 + i] = 1
        return m

    # ------------------------------------------------------------------ step
    def step(self, action):
        reward = 0.0
        if self.phase == "ENCOUNTER":
            alts = self.pending.pop(0) if self.pending else []
            if 1 <= action <= len(alts):
                self.roster.append(alts[action - 1])
            self._roll_next()
            self.phase = "ENCOUNTER" if self.pending else "PRE_BATTLE"
        else:  # PRE_BATTLE
            if action == 0:  # confirm -> battle (or end if no alive mons)
                alive_left = any(m.alive for m in self.roster)
                if not self.party or not alive_left:
                    reward -= 5.0
                    self.done = True
                else:
                    result = self._run_battle()
                    if result is None:
                        # Showdown reject / timeout — end run, don't crash worker.
                        reward -= 5.0
                        self.done = True
                    else:
                        won, alive, state = result
                        party = [self.roster[i] for i in self.party]
                        for mon, a in zip(party, alive):
                            if not a:
                                mon.alive = False
                                reward -= 1.0
                        self.graveyard += [m for m in self.roster if not m.alive]
                        self.roster = [m for m in self.roster if m.alive]
                        # by identity: two value-equal mons (twins at the cap) made a 7-mon League party
                        self.party = [i for i, m in enumerate(self.roster) if any(m is p for p in party)]
                        if won:
                            reward += 1.0
                            self.trainer_idx += 1
                            self._skip_other_rivals()
                            nxt = ORDER[self.trainer_idx] if self.trainer_idx < self.slice_len else {}
                            # the League block: the party is locked (no box) and PP carries over;
                            # HP and status are restored between fights (Full Restores — items and
                            # money are out of scope, so healing between fights is free)
                            self.carry = ({self.roster.index(m): {"pp": st["pp"]} for m, st in zip(party, state) if m.alive}
                                          if nxt.get("no_heal") else None)
                            if self.trainer_idx >= self.slice_len:
                                reward += 10.0
                                self.done = True
                            elif not any(m.alive for m in self.roster):
                                # Won on a wipe — can't continue the gauntlet. <-- Shouldnt be possible but just in case
                                reward -= 5.0
                                self.done = True
                            else:
                                if not self.party_locked:
                                    self.party = []
                                self._maybe_sample_encounter()
                                if not self.pending:
                                    self.phase = "PRE_BATTLE"
                        else:
                            # A lost battle ends the run (nuzlocke: you'd reload a
                            # save; for RL, retry loops would dilute credit). Wipe
                            # penalty only when everyone died.
                            if not any(m.alive for m in self.roster):
                                reward -= 5.0
                            self.done = True
                if self.done:
                    # End-of-run shaping: deeper runs and living catches are
                    # worth something even on a loss — counters the learned
                    # "solo the starter, skip everything" degenerate policy.
                    reward += 0.5 * self.trainer_idx / self.slice_len * 10
                    reward += 0.5 * sum(m.alive for m in self.roster
                                        if m.species not in STARTERS)
            else:
                i = action - 1
                if i < len(self.roster) and self.roster[i].alive:
                    if i not in self.party:
                        self.party.append(i)
        return (self._obs(), reward, self.done, False,
                {"action_mask": self._mask()})

    def render(self):
        roster = ", ".join(f"{m.species}@{m.level}" for m in self.roster)
        if self.graveyard:
            roster += " | †" + ", ".join(m.species for m in self.graveyard)
        entry = (ORDER[self.trainer_idx]["trainer_name"]
                 if self.trainer_idx < self.slice_len else "CLEARED")
        return (f"[{self.phase}] trainer {self.trainer_idx}/{self.slice_len} "
                f"({entry}) | roster: {roster} | party: {self.party}")

    def tracker(self) -> str:
        """The box as a Nuzlocke tracker would show it: party first, then the
        bench, then the graveyard, with level, ability and moves."""
        party = [self.roster[i] for i in self.party]
        rows = [("party", m) for m in party] + [("box", m) for m in self.roster if not any(m is p for p in party)]
        rows += [("dead", m) for m in self.graveyard]
        lines = [f"    {tag:<5} {m.species:<11} L{m.level:<3} {m.ability or '-':<12} {', '.join(m.moves)}"
                 for tag, m in rows]
        if self.dropped:
            lines.append(f"    dropped (no room): {', '.join(self.dropped)}")
        return "\n".join(lines)
