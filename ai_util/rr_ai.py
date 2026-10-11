"""Radical Red / CFRU trainer AI as a poke-env Player.

Singles only. Each move starts at 100 (CFRU convention), then:
  negatives — dump immunities, redundant status, already-up field effects
  positives — KO, damage, STAB, setup, hazards, status, recovery, priority
Switching is CFRU's (ai_switching.c): absorb a predicted move, dodge a predicted KO,
Toxic counter, -3 offence, save the sweeper; switch-ins scored by damage calc.
Not a 1:1 port of the C engine (~700KB); the decision skeleton is the same.
"""

from __future__ import annotations

import random

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import STACKABLE_CONDITIONS, SideCondition
from poke_env.battle.status import Status
from poke_env.battle.target import Target
from poke_env.battle.weather import Weather
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from ai_util.calc_bot import dmg_frac, effectiveness, stat

# CFRU starts every legal move at 100. Dumped moves go to 0.
BASE_SCORE = 100
KO_BONUS = 30

HAZARDS = {
    "stealthrock": SideCondition.STEALTH_ROCK,
    "spikes": SideCondition.SPIKES,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
    "stickyweb": SideCondition.STICKY_WEB,
}
HAZARD_REMOVAL = {"rapidspin", "defog", "courtchange", "tidyup", "mortalspin"}
SCREENS = {
    "reflect": SideCondition.REFLECT,
    "lightscreen": SideCondition.LIGHT_SCREEN,
    "auroraveil": SideCondition.AURORA_VEIL,
}
RECOVERY = {
    "recover",
    "softboiled",
    "roost",
    "slackoff",
    "moonlight",
    "morningsun",
    "synthesis",
    "shoreup",
    "milkdrink",
    "healorder",
    "rest",
    "wish",
    "strengthsap",
}
ABSORB = {  # the abilities an absorb switch looks for, by the predicted move's type
    PokemonType.FIRE: {"flashfire"},
    PokemonType.ELECTRIC: {"voltabsorb", "lightningrod", "motordrive"},
    PokemonType.WATER: {"waterabsorb", "dryskin", "stormdrain"},
    PokemonType.GRASS: {"sapsipper"},
}
PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport"}
FIRST_TURN_ONLY = {"fakeout", "firstimpression"}
WEATHER_BOOST = {
    Weather.SUNNYDAY: PokemonType.FIRE,
    Weather.DESOLATELAND: PokemonType.FIRE,
    Weather.RAINDANCE: PokemonType.WATER,
    Weather.PRIMORDIALSEA: PokemonType.WATER,
}
WEATHER_CUT = {
    Weather.SUNNYDAY: PokemonType.WATER,
    Weather.DESOLATELAND: PokemonType.WATER,
    Weather.RAINDANCE: PokemonType.FIRE,
    Weather.PRIMORDIALSEA: PokemonType.FIRE,
}
STATUS_BLOCK_ABILITIES = {
    Status.PAR: {"limber", "comatose"},
    Status.BRN: {"waterveil", "waterbubble", "comatose", "thermalexchange"},
    Status.PSN: {"immunity", "comatose", "pastelveil"},
    Status.TOX: {"immunity", "comatose", "pastelveil"},
    Status.SLP: {"insomnia", "vitalspirit", "comatose", "sweetveil", "purifyingsalt"},
    Status.FRZ: {"magmaarmor", "comatose"},
}


class RadicalRedAI(Player):
    """CFRU-style singles heuristics player. `ko_bonus` / `switching` are per instance so the
    sensitivity check (TODO.md A2.3) can perturb the real enemy and keep the search's model."""

    ko_bonus = KO_BONUS
    switching = True

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        if isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        if not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        return self._choose_singles(battle)

    def _choose_singles(self, battle: Battle) -> BattleOrder:
        if battle.force_switch:
            return self._best_switch(battle) or self.choose_random_move(battle)

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if any(m.id == "recharge" for m in battle.available_moves):  # the Hyper Beam turn: no choice,
            return self.create_order(battle.available_moves[0])  # and poke-env's stub entry has no priority

        scored: list[tuple[Move, int]] = []
        for move in battle.available_moves:
            scored.append((move, self._score_move(battle, active, opponent, move)))

        best_move: Move | None = None
        best_score = -1
        if scored:
            top = max(s for _, s in scored)
            tied = [m for m, s in scored if s == top]
            best_move = random.choice(tied)
            best_score = top

        target = self._switch_target(battle, active, opponent, best_score)
        if target is not None:
            return self.create_order(target)

        if best_move is not None:  # CFRU clicks its least-bad move rather than switch for nothing
            return self.create_order(
                best_move,
                mega=bool(battle.can_mega_evolve),
                dynamax=self._should_dynamax(battle),
                terastallize=self._should_tera(battle, best_move),
            )

        if battle.available_switches:
            return self._best_switch(battle) or self.choose_random_move(battle)
        return self.choose_random_move(battle)

    # --- scoring (CFRU: negatives then positives) ---

    def _score_move(
        self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move
    ) -> int:
        score = BASE_SCORE
        dumped = self._negatives(battle, active, opponent, move)
        if dumped is not None:
            return dumped
        score += self._positives(battle, active, opponent, move)
        return score

    def _negatives(
        self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move
    ) -> int | None:
        """Return 0 if the move should never be used, else None."""
        if move.id in FIRST_TURN_ONLY and not active.first_turn:
            return 0
        if move.is_protect_move and active.protect_counter > 0:
            return 0

        if move.category != MoveCategory.STATUS:
            if effectiveness(move, opponent) == 0:
                return 0
        else:
            if self._status_fails(battle, active, opponent, move):
                return 0

        if move.self_destruct and (
            not self._would_ko(battle, active, opponent, move)
            or self._n_remaining(battle.team) <= 1
        ):
            return 0

        boosts = self._self_boosts(move)
        positives = [s for s, v in (boosts or {}).items() if v > 0]
        if positives and all(active.boosts.get(s, 0) >= 6 for s in positives):
            return 0

        if move.weather and move.weather in battle.weather:
            return 0
        if move.terrain and move.terrain in battle.fields:
            return 0
        if move.id in SCREENS and SCREENS[move.id] in battle.side_conditions:
            return 0
        if move.id in HAZARDS:
            cond = HAZARDS[move.id]
            stacks = battle.opponent_side_conditions.get(cond, 0)
            cap = STACKABLE_CONDITIONS.get(cond, 1)
            if stacks >= cap or (
                cond not in STACKABLE_CONDITIONS
                and cond in battle.opponent_side_conditions
            ):
                return 0
        if move.id in HAZARD_REMOVAL and not battle.side_conditions:
            return 0

        vs = move.volatile_status
        if vs is not None and vs in opponent.effects:
            return 0
        return None

    def _positives(
        self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move
    ) -> int:
        bonus = 0
        dmg = self._damage_fraction(battle, active, opponent, move)
        hp_left = max(opponent.current_hp_fraction, 0.01)
        our_hp = active.current_hp_fraction
        matchup = self._matchup(active, opponent)

        if move.category != MoveCategory.STATUS:
            if dmg >= hp_left * 0.92:
                bonus += self.ko_bonus
                if move.priority > 0:
                    bonus += 4
            else:
                bonus += int(min(dmg / hp_left, 1.5) * 18)
            if effectiveness(move, opponent) > 1:
                bonus += 4
            elif 0 < effectiveness(move, opponent) < 1:
                bonus -= 4
            if move.recoil > 0:
                bonus -= 2
            if move.drain > 0 and our_hp < 0.7:
                bonus += 4
        else:
            bonus += self._status_bonus(battle, active, opponent, move)

        boosts = self._self_boosts(move)
        if boosts and our_hp > 0.5 and matchup >= 0:
            total = sum(v for v in boosts.values() if v > 0)
            bonus += 6 + 3 * total
            if battle.turn <= 2 or active.first_turn:
                bonus += 4
        elif boosts and our_hp < 0.4:
            bonus -= 8

        if move.id in HAZARDS:
            n_opp = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
            if n_opp >= 3:
                bonus += 10
            elif n_opp >= 2:
                bonus += 4
            else:
                bonus -= 8

        if move.id in HAZARD_REMOVAL and battle.side_conditions:
            n_us = self._n_remaining(battle.team)
            bonus += 8 if n_us >= 2 else -6

        if move.id in SCREENS and our_hp > 0.4:
            bonus += 6

        heal = self._heal_fraction(move)
        if heal > 0 or move.id in RECOVERY:
            if our_hp < 0.35:
                bonus += 14
            elif our_hp < 0.55:
                bonus += 6
            else:
                bonus -= 10
            if move.id == "rest" and our_hp > 0.5:
                bonus -= 12

        if move.id in PIVOTS and matchup < 0 and battle.available_switches:
            bonus += 8

        if move.priority > 0 and self._incoming_ko_risk(active, opponent) and dmg > 0:
            bonus += 6

        if move.id in FIRST_TURN_ONLY and active.first_turn:
            bonus += 12

        acc = move.accuracy
        if acc < 1:
            bonus -= int((1 - acc) * 8)
        return bonus

    def _status_bonus(
        self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move
    ) -> int:
        if move.status is None:
            drops = move.boosts if move.target != Target.SELF else None
            if drops and any(v < 0 for v in drops.values()):
                return 5 if opponent.current_hp_fraction > 0.5 else 1
            return 2
        damaging = [m for m in battle.available_moves if m.base_power > 0]
        if damaging:
            best = max(
                damaging,
                key=lambda m: self._damage_fraction(battle, active, opponent, m),
            )
            if self._would_ko(battle, active, opponent, best) and self._faster(
                active, opponent
            ):
                return -6
        if opponent.current_hp_fraction < 0.25:
            return -4
        bonus = 8
        if move.status in (Status.SLP, Status.TOX):
            bonus += 3
        if move.status == Status.PAR and not self._faster(active, opponent):
            bonus += 4
        if move.status == Status.BRN and opponent.base_stats["atk"] >= opponent.base_stats["spa"]:
            bonus += 3
        return bonus

    def _status_fails(
        self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move
    ) -> bool:
        status = move.status
        if status is None:
            return False
        if opponent.status is not None:
            return True
        types = opponent.types
        ability = (opponent.ability or "").replace(" ", "").lower()
        if ability in STATUS_BLOCK_ABILITIES.get(status, set()):
            return True
        if status in (Status.PSN, Status.TOX) and (
            PokemonType.STEEL in types or PokemonType.POISON in types
        ):
            return True
        if status == Status.BRN and PokemonType.FIRE in types:
            return True
        if status == Status.FRZ and PokemonType.ICE in types:
            return True
        if status == Status.PAR:
            if PokemonType.ELECTRIC in types:
                return True
            if move.type == PokemonType.ELECTRIC and PokemonType.GROUND in types:
                return True
        if Field.MISTY_TERRAIN in battle.fields:
            return True
        return status == Status.SLP and Field.ELECTRIC_TERRAIN in battle.fields

    # --- switching (CFRU src/Battle_AI/ai_switching.c, singles, smart AI; data/rr/AI_DIFF.md) ---
    # Damage is the calc bot's model at its mean roll, against the player's real moves (full
    # info = "the AI knows your team"). Not ported: Wish / Perish / locked moves / Yawn / sleep /
    # Wonder Guard / Natural Cure / hazard removers, the fighting-style classes (a "damager" here
    # = three or more attacking moves), the absorb-switch memory (cooldown = just switched in).

    def _hit(self, battle, att, dfn, move, att_own: bool) -> float:
        """Fraction of dfn's max HP one hit of move removes."""
        return dmg_frac(att, dfn, move, att_own, battle, with_acc=False)

    def _attacks(self, mon) -> list:
        return [m for m in mon.moves.values() if m.category != MoveCategory.STATUS]

    def _can_ko(self, battle, att, dfn, att_own: bool, hits: int = 1) -> bool:
        hp = dfn.current_hp_fraction
        return any(hits * self._hit(battle, att, dfn, m, att_own) >= hp for m in self._attacks(att))

    def _predict_foe_move(self, battle, active, foe):
        """The player's move as the AI predicts it: its strongest hit on our active (CFRU runs its
        own scorer on the player's moves; strongest-hit is the attack-or-nothing reading of it)."""
        atk = self._attacks(foe)
        return max(atk, key=lambda m: self._hit(battle, foe, active, m, False)) if atk else None

    def _switch_in(self, battle, mon, foe) -> tuple[int, set]:
        """CalcMostSuitableMonToSwitchInto for one bench mon: (score, flags)."""
        score, flags = 0, set()
        if stat(mon, "spe", True, battle) >= stat(foe, "spe", False, battle):
            score, flags = score + 14, flags | {"outspeeds"}
        if self._can_ko(battle, mon, foe, True):
            score, flags = score + 31, flags | {"ko"}
            if any(m.priority > 0 and self._hit(battle, mon, foe, m, True) >= foe.current_hp_fraction for m in self._attacks(mon)):
                score, flags = score + 8, flags | {"revenge"}
                if "outspeeds" not in flags:
                    score, flags = score + 14, flags | {"outspeeds"}
        elif self._can_ko(battle, mon, foe, True, hits=2):
            score, flags = score + 2, flags | {"2hko"}
        elif not self._attacks(mon):
            return -1, flags
        hp = mon.current_hp_fraction
        faints = weak = normal = 0
        for m in sorted(self._attacks(foe), key=lambda m: -m.priority):
            d = self._hit(battle, foe, mon, m, False)
            if d >= hp:
                faints = 1
                if "outspeeds" in flags and m.priority > 0:
                    flags.discard("outspeeds")
                break
            if 2 * d >= hp:
                weak = 1
            elif 3 * d >= hp:
                normal += 1
        if faints and "outspeeds" not in flags:
            score, flags = max(score - 39, 0), flags | {"faints"}
        elif faints and "ko" not in flags:
            score = max(score - 15, 0)
        elif faints or weak:
            score = max(score - 1, 0)
        elif not normal:
            score, flags = score + 17, flags | {"resists"}
        else:
            phys = any(m.category == MoveCategory.PHYSICAL for m in self._attacks(foe))
            spec = any(m.category == MoveCategory.SPECIAL for m in self._attacks(foe))
            if not (phys and stat(mon, "def", True) <= stat(foe, "atk", False)) and \
                    not (spec and stat(mon, "spd", True) <= stat(foe, "spa", False)):
                score, flags = score + 2, flags | {"walls"}
        return score, flags

    def _ranked_switches(self, battle) -> list[tuple]:
        """Bench mons best first as (mon, score, flags); equal scores: the one that does not faint,
        else a coin flip (CFRU's 50%)."""
        foe = battle.opponent_active_pokemon
        rows = [(m, *self._switch_in(battle, m, foe)) for m in battle.available_switches]
        rows = [(m, s, f, "faints" in f, random.random()) for m, s, f in rows]
        rows.sort(key=lambda r: (-r[1], r[3], r[4]))
        return [r[:3] for r in rows]

    def _wont_take_much(self, battle, mon, flags, active, foe, move) -> bool:
        """PredictedMoveWontDoTooMuchToMon."""
        if "faints" in flags:
            return False
        if move is None or "resists" in flags or move.id == "suckerpunch":
            return True
        d, hp = self._hit(battle, foe, mon, move, False), mon.current_hp_fraction
        if d >= hp:
            return False
        if 2 * d < hp:
            return True
        return 2 * d >= 1 and any(m.id in RECOVERY or self._heal_fraction(m) for m in mon.moves.values()) \
            and stat(mon, "spe", True, battle) > stat(foe, "spe", False, battle)

    def _switch_target(self, battle: Battle, active: Pokemon, foe: Pokemon, best_score: int):
        """ShouldSwitch: the bench mon to switch to, or None to stay in."""
        if battle.trapped or not battle.available_switches or not self.switching:
            return None
        ranked = self._ranked_switches(battle)
        top2 = ranked[:2]
        move = self._predict_foe_move(battle, active, foe)
        fresh = active.first_turn
        ability = (active.ability or "").replace(" ", "").lower()
        we_ko = self._can_ko(battle, active, foe, True)
        they_ko = move is not None and self._hit(battle, foe, active, move, False) >= active.current_hp_fraction
        faster = stat(active, "spe", True, battle) > stat(foe, "spe", False, battle)
        boosted = any(v > 0 for v in active.boosts.values())

        # absorb the predicted move (Flash Fire, Volt / Water Absorb, Dry Skin, Sap Sipper ...)
        absorbers = ABSORB.get(move.type) if move is not None else None
        if absorbers and not fresh and ability not in absorbers and not (faster and we_ko) \
                and not (not they_ko and we_ko and boosted):
            for mon, _, _ in ranked:
                if (mon.ability or "").replace(" ", "").lower() in absorbers:
                    if mon.current_hp_fraction >= 1 and mon.ability in ("voltabsorb", "waterabsorb", "dryskin") \
                            and 2 * self._hit(battle, foe, active, move, False) < active.current_hp_fraction:
                        continue  # a full-HP healer gains nothing from a hit the active takes twice
                    return mon
        # only bad moves left: go to a mon that outspeeds or walls and takes the hit
        if best_score <= 0 and not fresh:
            for mon, _, f in top2:
                if f & {"outspeeds", "walls", "resists"} and self._wont_take_much(battle, mon, f, active, foe, move):
                    return mon
        # annoying residual damage: Toxic past six turns (urgent: the best mon, whatever it is)
        if active.status == Status.TOX and active.status_counter > 6 and ability != "poisonheal" and not we_ko:
            return ranked[0][0]
        # avoid death: slower, the predicted hit KOs, not nearly dead already
        if not fresh and they_ko and not faster and not (active.status == Status.PAR and active.current_hp_fraction < 1 / 3) \
                and (active.current_hp_fraction > 0.2 or ranked[0][1] >= 70):
            for mon, _, f in ranked:
                if (f & {"outspeeds", "walls", "resists"} and effectiveness(move, mon) == 0) or \
                        ((f & {"walls", "resists"} or {"ko", "outspeeds"} <= f) and self._wont_take_much(battle, mon, f, active, foe, move)):
                    return mon
        damager = len(self._attacks(active)) >= 3
        # offensive stats at -3 with only that kind of attack, and the foe cannot KO us
        if not fresh and damager:
            cats = {m.category for m in self._attacks(active)}
            low = (active.boosts.get("atk", 0) <= -3 and cats == {MoveCategory.PHYSICAL}) or \
                (active.boosts.get("spa", 0) <= -3 and cats == {MoveCategory.SPECIAL}) or \
                (active.boosts.get("atk", 0) <= -3 and active.boosts.get("spa", 0) <= -3)
            if low and not self._can_ko(battle, foe, active, False):
                mon, _, f = ranked[0]
                if f & {"walls", "resists"} or ("outspeeds" in f and self._wont_take_much(battle, mon, f, active, foe, move)):
                    return mon
        # save the sweeper for later: it cannot KO, and it would die or is doing nothing useful
        if not fresh and damager and not we_ko and active.status not in (Status.FRZ, Status.PAR):
            heals = any(m.id in RECOVERY or self._heal_fraction(m) for m in active.moves.values())
            setup = any(self._self_boosts(m) for m in active.moves.values())
            r = random.randrange(4)
            if (self._can_ko(battle, foe, active, False) and not heals) or (
                    not boosted and not setup and ("ko" in ranked[0][2] or (
                        r & 1 and (not self._can_ko(battle, active, foe, True, hits=2) or r & 2)
                        and not self._resists_all(battle, active, foe)))):
                if self._can_ko(battle, foe, active, False):
                    for mon, _, f in top2:
                        if {"ko", "outspeeds"} <= f and self._wont_take_much(battle, mon, f, active, foe, move):
                            return mon
                for mon, _, f in ranked:
                    if "resists" in f:
                        return mon
        return None

    def _resists_all(self, battle, mon, foe) -> bool:
        return all(3 * self._hit(battle, foe, mon, m, False) < mon.current_hp_fraction for m in self._attacks(foe))

    def _best_switch(self, battle: Battle) -> BattleOrder | None:
        if not battle.available_switches:
            return None
        if battle.opponent_active_pokemon is None:
            return self.create_order(random.choice(battle.available_switches))
        return self.create_order(self._ranked_switches(battle)[0][0])

    # --- damage / matchup ---

    def _damage_fraction(
        self, battle: Battle, attacker: Pokemon, defender: Pokemon, move: Move
    ) -> float:
        if move.category == MoveCategory.STATUS or move.base_power <= 0:
            if isinstance(move.damage, int) and move.damage > 0:
                max_hp = defender.max_hp or self._est_hp(defender)
                return move.damage / max(max_hp, 1)
            return 0.0
        power = move.base_power * move.expected_hits
        if move.category == MoveCategory.PHYSICAL:
            ratio = self._stat(attacker, "atk") / max(self._stat(defender, "def"), 1.0)
        else:
            ratio = self._stat(attacker, "spa") / max(self._stat(defender, "spd"), 1.0)
        level = attacker.level or 100
        base = ((2 * level / 5 + 2) * power * ratio) / 50 + 2
        stab = attacker.stab_multiplier if move.type in attacker.types else 1.0
        typ = effectiveness(move, defender)
        weather_mult = self._weather_mult(battle, move)
        acc = move.accuracy if move.accuracy else 1.0
        dmg = base * stab * typ * weather_mult * acc
        max_hp = defender.max_hp or self._est_hp(defender)
        return dmg / max(max_hp, 1)

    def _would_ko(
        self, battle: Battle, attacker: Pokemon, defender: Pokemon, move: Move
    ) -> bool:
        return self._damage_fraction(battle, attacker, defender, move) >= (
            defender.current_hp_fraction * 0.92 or 0.01
        )

    def _matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        our_off = max((opponent.damage_multiplier(t) for t in mon.types if t), default=1.0)
        their_off = max((mon.damage_multiplier(t) for t in opponent.types if t), default=1.0)
        score = our_off - their_off
        if self._faster(mon, opponent):
            score += 0.15
        else:
            score -= 0.15
        score += 0.3 * (mon.current_hp_fraction - opponent.current_hp_fraction)
        return score

    def _incoming_ko_risk(self, active: Pokemon, opponent: Pokemon) -> bool:
        their_off = max(
            (active.damage_multiplier(t) for t in opponent.types if t), default=1.0
        )
        if their_off >= 2 and not self._faster(active, opponent):
            return active.current_hp_fraction < 0.55
        return their_off >= 2 and active.current_hp_fraction < 0.35

    @staticmethod
    def _stat(mon: Pokemon, stat: str) -> float:
        raw = (mon.stats or {}).get(stat)
        if isinstance(raw, (int, float)) and raw > 0:
            val = float(raw)
        else:
            level = mon.level or 100
            val = ((2 * mon.base_stats[stat] + 31) * level) / 100 + 5
        boost = mon.boosts.get(stat, 0)
        if boost >= 0:
            return val * (2 + boost) / 2
        return val * 2 / (2 - boost)

    @staticmethod
    def _est_hp(mon: Pokemon) -> float:
        level = mon.level or 100
        return ((2 * mon.base_stats["hp"] + 31) * level) / 100 + level + 10

    @staticmethod
    def _faster(a: Pokemon, b: Pokemon) -> bool:
        return RadicalRedAI._stat(a, "spe") > RadicalRedAI._stat(b, "spe")

    @staticmethod
    def _weather_mult(battle: Battle, move: Move) -> float:
        for w in battle.weather:
            if WEATHER_BOOST.get(w) == move.type:
                return 1.5
            if WEATHER_CUT.get(w) == move.type:
                return 0.5
        return 1.0

    @staticmethod
    def _self_boosts(move: Move) -> dict | None:
        if move.self_boost:
            return move.self_boost
        if move.boosts and move.target == Target.SELF:
            return move.boosts
        return None

    @staticmethod
    def _heal_fraction(move: Move) -> float:
        heal = move.entry.get("heal")
        if heal:
            return heal[0] / heal[1]
        return 0.0

    @staticmethod
    def _n_remaining(team: dict) -> int:
        return sum(1 for m in team.values() if not m.fainted)

    def _should_dynamax(self, battle: Battle) -> bool:
        if not battle.can_dynamax or battle.active_pokemon is None:
            return False
        alive = self._n_remaining(battle.team)
        if alive == 1:
            return True
        return battle.active_pokemon.current_hp_fraction == 1 and alive <= 3

    def _should_tera(self, battle: Battle, move: Move) -> bool:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not battle.can_tera or active is None or opponent is None:
            return False
        if active.tera_type is None:
            return False
        return effectiveness(move, opponent) >= 2


RadicalRedPlayer = RadicalRedAI
