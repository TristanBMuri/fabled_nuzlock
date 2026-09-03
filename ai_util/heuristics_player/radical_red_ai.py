"""Radical Red / CFRU trainer AI as a poke-env Player.

Singles only. Each move starts at 100 (CFRU convention), then:
  negatives — dump immunities, redundant status, already-up field effects
  positives — KO, damage, STAB, setup, hazards, status, recovery, priority
Smart-switch when the matchup is losing and a better bench mon exists.
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

# CFRU starts every legal move at 100. Dumped moves go to 0.
BASE_SCORE = 100
KO_BONUS = 30
SWITCH_MATCHUP_THRESHOLD = -1.5

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
    """CFRU-style singles heuristics player."""

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

        if (
            self._should_switch(battle, active, opponent, best_score)
            and battle.available_switches
        ):
            return self._best_switch(battle) or self.choose_random_move(battle)

        if best_move is not None and best_score > 0:
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
            if opponent.damage_multiplier(move) == 0:
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
                bonus += KO_BONUS
                if move.priority > 0:
                    bonus += 4
            else:
                bonus += int(min(dmg / hp_left, 1.5) * 18)
            if opponent.damage_multiplier(move) > 1:
                bonus += 4
            elif 0 < opponent.damage_multiplier(move) < 1:
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

    # --- switching ---

    def _should_switch(
        self, battle: Battle, active: Pokemon, opponent: Pokemon, best_score: int
    ) -> bool:
        if battle.trapped or not battle.available_switches:
            return False
        if best_score >= BASE_SCORE + KO_BONUS:
            return False
        if any(active.boosts.get(s, 0) <= -3 for s in ("atk", "spa", "def", "spd")):
            return True
        matchup = self._matchup(active, opponent)
        if matchup >= SWITCH_MATCHUP_THRESHOLD:
            return False
        better = [
            m
            for m in battle.available_switches
            if self._matchup(m, opponent) > matchup + 0.5
        ]
        if not better:
            return False
        if self._incoming_ko_risk(active, opponent) and best_score < BASE_SCORE + 10:
            return True
        return matchup < SWITCH_MATCHUP_THRESHOLD

    def _best_switch(self, battle: Battle) -> BattleOrder | None:
        opponent = battle.opponent_active_pokemon
        switches = battle.available_switches
        if not switches:
            return None
        if opponent is None:
            return self.create_order(random.choice(switches))
        best = max(switches, key=lambda m: self._switch_score(m, opponent, battle))
        return self.create_order(best)

    def _switch_score(self, mon: Pokemon, opponent: Pokemon, battle: Battle) -> float:
        score = self._matchup(mon, opponent) * 10
        score += mon.current_hp_fraction * 4
        for cond, stacks in battle.side_conditions.items():
            if cond == SideCondition.STEALTH_ROCK:
                score -= 2 * mon.damage_multiplier(PokemonType.ROCK)
            elif cond == SideCondition.SPIKES:
                score -= 0.8 * stacks
        return score

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
        typ = defender.damage_multiplier(move)
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
        return opponent.damage_multiplier(move) >= 2


RadicalRedPlayer = RadicalRedAI
