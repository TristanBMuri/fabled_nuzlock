"""Nuzlocke gauntlet env: the run as a state machine around battles played by
a pluggable brain (scripts/runner.py).

Two phases. ENCOUNTER answers the pending offer: step(0) skips, step(k)
catches alternative k (one per encounter method). PRE_BATTLE builds the party:
step(k) adds roster mon k-1, step(0) confirms — the fight runs, deaths apply,
the next trainer opens (and its areas queue offers). A lost battle or a wipe
ends the run. The scripted planner on top is scripts/plan_gauntlet.py.
"""

import random
import uuid
from dataclasses import dataclass

from gauntlet import (Mon, ability_slots, assign_items, encounter_moves, evolve, evolved_ability,
                      load_encounters, load_evolutions, load_trainers, pack_team, player_evs,
                      roll_ability, trainer_mons)
import runner
from runner import SimBattleRunner, record_battle

MAX_PARTY = 6


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
    items: bool = True           # held items found so far, assigned by gauntlet.assign_items (E9; data/bdsp/items.json)
    evs: str = "flat"            # player EVs: flat 85 each, or trained (252 attack / 252 Spe / 4 HP)

    def __str__(self):
        return (f"level_rule={self.level_rule} margin={self.level_margin} box={self.box} "
                f"dupes={self.dupes_clause} moveset={self.moveset} tms={self.tms} evs={self.evs} items={self.items}")


SLICE_LEN = 21  # trainers per episode: through Gym Leader Roark (order_index 20 in the route order)
EVOS = load_evolutions()
TRAINERS, ORDER = load_trainers()
RISK_BATTLES = 8
# A fight is "hard" (deep search width) when the fielded party's risk query shows
# a loss, a death, or less than this fraction of the party's HP left on average.
DEEP_HP = 0.5


class NuzlockeEnv:
    def __init__(self, seed: int | None = None, policy: str | None = None,
                 search: tuple[int, int] | None = None, rules: Rules = Rules(),
                 search_all: bool = False, slice_len: int = SLICE_LEN,
                 deep: tuple[int, int] | None = None, full_info: bool = False):
        """slice_len: trainers per episode (21 = through Roark, 59 = through Gardenia,
        167 = Maylene, 405 = the whole game). policy selects the battler (None =
        SimpleHeuristics, 'calc' = the damage-calc bot, a path = a BC prior);
        search=(k, r) plays boss fights with rollout search, search_all every fight.
        deep=(k, r): wider search at breakpoints and at any fight the risk query
        flags as hard (DEEP_HP) — the compute dial per trainer. full_info: the
        battler knows every enemy set from turn one (the full-docs regime). rules: see Rules."""
        self.search_all = search_all
        self.deep = deep
        self.n_deep = 0  # fights played at the deep width (cost accounting)
        self.slice_len = slice_len
        self.rules = rules
        self.max_roster = rules.box  # living mons the box holds
        self.rng = random.Random(seed)
        self.runner = SimBattleRunner(policy=policy, search=search, full_info=full_info)
        self.trace_tag, self.trace_seed = "run", 0  # names the recorded champion fights (runner.TRACES)
        self.encounters = load_encounters()["routes"]
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
        self.phase = "ENCOUNTER"
        self.pending: list[list[Mon]] = []  # the offer being answered: one pick per method
        self._queue: list[dict] = []  # areas whose gate has arrived, rolled one at a time
        self.carry: dict[int, dict] | None = None  # roster index -> {}: the League's locked party
        self.dropped: list[str] = []  # offers that found no box slot (species), this run
        self._route_cursor = 0  # routes consumed
        self.done = False
        self._risk_cache: dict = {}

    def reset(self, seed=None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self._reset_state()
        self._maybe_sample_encounter()

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
        answers them one at a time, then the fight opens."""
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
        self._next_phase()

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

    def _next_phase(self):
        """More offers to answer, or the fight: entering PRE_BATTLE applies the
        level rule once per trainer (levels, evolutions, moves, EVs) — after the
        offers, so a catch and its evolution draw from the rng in that order."""
        if self.pending:
            self.phase = "ENCOUNTER"
        else:
            self.phase = "PRE_BATTLE"
            if self.trainer_idx < self.slice_len:
                self._apply_level_rule(ORDER[self.trainer_idx])

    @property
    def can_catch(self) -> bool:
        return bool(self.pending) and len(self.roster) < self.max_roster

    # ---------------------------------------------------------------- battle
    def _apply_level_rule(self, entry):
        self.max_enemy_level = max(self.max_enemy_level, entry["max_level"])
        # the locked block (the League) has one cap: its highest level (Cynthia's
        # 66), reached before Aaron — there is no leveling between the E4 fights
        j = self.trainer_idx
        while j + 1 < self.slice_len and ORDER[j + 1].get("locked"):
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
        """Inside the locked block (the League after Aaron): the party stays as
        fielded and starts each fight where the last one left it."""
        return self.carry is not None

    def _carry_list(self, party_idx):
        return None  # the League heals fully between fights: nothing carries into a battle

    def _packed_matchup(self, party_idx):
        """(player_packed, enemy_packed, entry) for a candidate party vs the next
        trainer. Items are handed out here (deterministic in party x progress)."""
        entry = ORDER[self.trainer_idx]
        party = [self.roster[i] for i in party_idx]
        assign_items(party, self.trainer_idx if self.rules.items else None)
        enemy = trainer_mons(TRAINERS[entry["trainer_id"]])
        return pack_team(party), pack_team(enemy), entry

    def _block(self) -> list[int] | None:
        """Indices of the locked block the next fight opens (Aaron + the rest of
        the League), or None: the party chosen here fights all of them."""
        i = self.trainer_idx
        if self.party_locked or i + 1 >= self.slice_len or not ORDER[i + 1].get("locked"):
            return None
        j = i + 1
        while j < self.slice_len and ORDER[j].get("locked"):
            j += 1
        return list(range(i, j))

    def risk(self, party_idx=None, n: int = RISK_BATTLES):
        """(P(win), [P(death_i)], mean HP fraction left) for a party (default:
        current) vs the next trainer — or vs the whole locked block it opens —
        from n battles under the fast brain. Read-only; cached per (party, trainer, n)."""
        party_idx = tuple(self.party if party_idx is None else party_idx)
        key = (party_idx, self.trainer_idx, n)
        if key not in self._risk_cache:
            est = None
            if party_idx and self.trainer_idx < self.slice_len:
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
        record = None
        if runner.TRACES is not None and entry["breakpoint"] == "champion":  # watchable in ui/
            record = {"path": runner.TRACES / f"trace_{self.trace_tag}_{self.trainer_idx}.json",
                      "boss": f"#{self.trainer_idx} {entry['trainer_name']}", "battle": self.trace_seed,
                      "flags": {"run": self.trace_tag, "enemy": runner.ENEMY}, "source": self.trace_tag}
        result = self.runner.run(player_packed, enemy_packed, boss=boss, search=width, carry=carry, record=record)
        if result is None:  # rebuild happened; retry once
            result = self.runner.run(player_packed, enemy_packed, boss=boss, search=width, carry=carry, record=record)
        if result is None:
            return None
        won, alive, state = result
        record_battle(self.run_id, entry["trainer_id"], won, player_packed, enemy_packed)
        return won, alive, state

    # ------------------------------------------------------------------ step
    def step(self, action: int) -> None:
        if self.done:
            return
        if self.phase == "ENCOUNTER":
            alts = self.pending.pop(0) if self.pending else []
            if 1 <= action <= len(alts) and len(self.roster) < self.max_roster:
                self.roster.append(alts[action - 1])
            self._roll_next()
            self._next_phase()
            return
        if action != 0:  # add roster mon action-1 to the party
            i = action - 1
            if (i < len(self.roster) and self.roster[i].alive and i not in self.party
                    and len(self.party) < MAX_PARTY and not self.party_locked):
                self.party.append(i)
            return
        # confirm -> battle (or end if nothing to field)
        if not self.party or not any(m.alive for m in self.roster):
            self.done = True
            return
        result = self._run_battle()
        if result is None:  # runner failure: the run ends as a loss (check the log for 'rebuild after')
            self.done = True
            return
        won, alive, state = result
        party = [self.roster[i] for i in self.party]
        for mon, a in zip(party, alive):
            if not a:
                mon.alive = False
        self.graveyard += [m for m in self.roster if not m.alive]
        self.roster = [m for m in self.roster if m.alive]
        self.party = [i for i, m in enumerate(self.roster) if any(m is p for p in party)]  # by identity
        if not won or not self.roster:
            # a lost battle ends the run (nuzlocke: you'd reload a save); so does a wipe
            self.done = True
            return
        self.trainer_idx += 1
        self._skip_other_rivals()
        nxt = ORDER[self.trainer_idx] if self.trainer_idx < self.slice_len else {}
        # the League block: the party is locked (no box swaps) and fully healed between
        # fights, HP, status and PP (2026-10-12, the user's rule: full heal, no swaps);
        # the dead stay dead. `carry` marks the lock; its entries are empty (nothing carries).
        # Keyed by identity (Mon eq=False).
        self.carry = ({self.roster.index(m): {} for m, st in zip(party, state) if m.alive}
                      if nxt.get("locked") else None)
        if self.trainer_idx >= self.slice_len:
            self.done = True
            return
        if not self.party_locked:
            self.party = []
        self._maybe_sample_encounter()

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
