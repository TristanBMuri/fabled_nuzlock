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
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player

from ai_util.calc_bot import dmg_frac, effectiveness, stat

# CFRU starts every legal move at 100. Dumped moves go to 0.
BASE_SCORE = 100
KO_BONUS = 9  # CFRU: a KO that goes first

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

    # --- scoring (CFRU ai_negatives.c / ai_positives.c / ai_advanced.c, singles) ---
    # Every move starts at 100; negatives are soft (-10, -15 immune, -20 absorbed) so a bad move
    # is still clicked when nothing else is; positives are small tiers keyed by the mon's
    # fighting-style class: KO first +9, slow KO +8 / +6, strongest attack +2, other attacks 0,
    # status / setup / recovery +4..+7 when their damage-calc check passes. Not ported: the
    # predicted-switch exceptions, Destiny Bond / Sucker Punch interplay in the KO tier, the
    # long tail of rare effects (screens, weather, Trick, Taunt ...: they score 100).

    def _class(self, mon) -> str:
        """PredictFightingStyle, singles."""
        attacks = status = screens = hazards = 0
        boosting = healing = phazing = leech = protect = False
        for m in mon.moves.values():
            if m.id == "batonpass":
                return "batonpass"
            if m.id in ("haze", "clearsmog"):
                return "phazer"
            if m.id in ("wish", "healbell", "aromatherapy"):
                return "cleric"
            if m.id in ("meanlook", "block", "spiderweb", "bind", "wrap", "firespin", "whirlpool", "sandtomb",
                        "clamp", "infestation", "magmastorm", "snaptrap", "thundercage"):
                return "stall"
            if mon.item in ("choiceband", "choicespecs", "choicescarf", "assaultvest"):
                return "kill"
            phazing |= m.id in ("roar", "whirlwind", "dragontail", "circlethrow")
            screens += m.id in ("reflect", "lightscreen") or 2 * (m.id == "auroraveil")
            leech |= m.id == "leechseed"
            protect |= m.is_protect_move
            healing |= m.id in RECOVERY or bool(m.heal) or m.id == "swallow"
            if m.category != MoveCategory.STATUS:
                attacks += 1
            elif m.id in HAZARDS:
                hazards += 1
            elif self._self_boosts(m) or m.id == "stockpile":
                boosting = True
            else:
                status += 1
        if screens >= 2:
            return "screens_sweeper" if attacks >= 2 else "screens"
        if hazards:
            return "phazer" if phazing else "hazards"
        if attacks >= 3:
            return "setup_stats" if boosting else "setup_status" if status or phazing else "kill"
        if leech and protect:
            return "stall"
        if attacks >= 2 and (boosting or status or phazing):
            return "setup_stats" if boosting else "setup_status"
        return "stall"

    def _status_viability(self, cls: str, boost: int, battle, active, foe) -> int:
        """IncreaseStatusViability."""
        if cls == "setup_stats":
            return 3 if not self._can_ko(battle, foe, active, False, hits=2) else 0
        return {"setup_status": 4 + boost, "stall": 3 + boost, "cleric": 3 + boost, "phazer": 4 + boost,
                "screens": 2 + boost, "screens_sweeper": 2 + boost, "hazards": 3,
                "batonpass": int(boost >= 3)}.get(cls, 0)

    def _goes_first(self, battle, move, active, foe, foe_move) -> bool:
        p, q = move.priority, foe_move.priority if foe_move is not None else 0
        return p > q or (p == q and stat(active, "spe", True, battle) > stat(foe, "spe", False, battle))

    def _foe_best(self, battle, active, foe) -> float:
        return max((self._hit(battle, foe, active, m, False) for m in self._attacks(foe)), default=0.0)

    def _should_recover(self, battle, active, foe, move, foe_move) -> bool:
        """ShouldRecover: heal when it beats the foe's damage, by damage calc and speed order."""
        hp = active.current_hp_fraction
        heal = min(1.0 if move.id == "rest" else (move.heal or 0.5), 1 - hp)
        d = self._foe_best(battle, active, foe)
        if self._goes_first(battle, move, active, foe, foe_move):
            if d >= hp and d < hp + heal:
                return True
            return 2 * d >= hp and heal > hp and random.random() < 0.5
        return d < hp and 2 * d >= hp and hp - d + heal > d

    def _setup_ok(self, battle, active, foe, move, foe_move, key: str, limit: int) -> bool:
        """ShouldTryToSetUpStat (singles)."""
        if (foe.ability or "") == "unaware":
            return False
        d = self._foe_best(battle, active, foe)
        hp = active.current_hp_fraction
        if self._goes_first(battle, move, active, foe, foe_move):
            if d >= hp:
                return False
        elif 2 * d >= hp:
            phys = foe_move is not None and foe_move.category == MoveCategory.PHYSICAL
            return d < hp and ((key == "def" and phys) or (key == "spd" and not phys and foe_move is not None))
        return active.boosts.get(key, 0) < limit

    def _stat_viability(self, cls: str, battle, active, foe, move, foe_move, key: str, limit: int = 2, boost: int = 2) -> int:
        """IncreaseStatViability."""
        ok = self._setup_ok(battle, active, foe, move, foe_move, key, limit)
        if cls in ("kill", "setup_stats"):
            if ok:
                return 3 + boost
            return 1 if cls == "kill" and move.category != MoveCategory.STATUS and key in ("atk", "spa") else 0
        if cls == "batonpass":
            return 5 if ok else 0
        return self._status_viability(cls, 1, battle, active, foe) if ok else 0

    def _good_to_raise(self, key: str, active, foe, battle) -> bool:
        """GoodIdeaToRaise*Against: the boosted stat has a use."""
        cats = {m.category for m in self._attacks(active)}
        foe_phys = sum(m.category == MoveCategory.PHYSICAL for m in self._attacks(foe))
        foe_spec = sum(m.category == MoveCategory.SPECIAL for m in self._attacks(foe))
        return {"atk": MoveCategory.PHYSICAL in cats, "spa": MoveCategory.SPECIAL in cats,
                "def": foe_phys > foe_spec, "spd": foe_spec > foe_phys,
                "spe": stat(active, "spe", True, battle) <= stat(foe, "spe", False, battle)}.get(key, False)

    def _bad_status(self, status, foe, active) -> bool:
        """BadIdeaTo{Poison,Paralyze,Burn,PutToSleep}: abilities and moves that profit from it."""
        ab = (foe.ability or "").replace(" ", "").lower()
        moves = set(foe.moves)
        phys = any(m.category == MoveCategory.PHYSICAL for m in self._attacks(foe))
        if ab in ("shedskin", "quickfeet") or "facade" in moves or "psychoshift" in moves:
            return True
        if ab == "guts" and phys or ab == "naturalcure" or ab == "hydration":
            return True
        if ab == "marvelscale" and any(m.category == MoveCategory.PHYSICAL for m in self._attacks(active)):
            return True
        if status in (Status.PSN, Status.TOX):
            return ab in ("poisonheal", "magicguard") or (ab == "toxicboost" and phys)
        if status == Status.PAR:
            return "rest" in moves
        if status == Status.BRN:
            return ab == "magicguard" or (ab == "flareboost" and any(m.category == MoveCategory.SPECIAL for m in self._attacks(foe)))
        if status == Status.SLP:
            return ab == "earlybird"
        return False

    def _score_move(self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move) -> int:
        return BASE_SCORE + self._negatives(battle, active, opponent, move) + self._positives(battle, active, opponent, move)

    def _negatives(self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move) -> int:
        """AIScript_Negatives, the cases that matter in singles."""
        hp = active.current_hp_fraction
        if move.id in FIRST_TURN_ONLY and not active.first_turn:
            return -10
        if move.is_protect_move and active.protect_counter > 0:
            return -10 if active.protect_counter >= 2 or random.random() < 0.5 else 0
        if move.category != MoveCategory.STATUS and effectiveness(move, opponent) == 0:
            ab = (opponent.ability or "").replace(" ", "").lower()
            return -20 if any(ab in s for s in ABSORB.values()) else -15
        if move.category == MoveCategory.STATUS and self._status_fails(battle, active, opponent, move):
            return -10
        if move.id == "suckerpunch":
            foe_move = self._predict_foe_move(battle, active, opponent)
            if foe_move is None:
                return -10
        if move.self_destruct and self._n_remaining(battle.team) <= 1:
            return -10
        boosts = self._self_boosts(move)
        positives = [k for k, v in (boosts or {}).items() if v > 0]
        if positives and all(active.boosts.get(k, 0) >= 6 for k in positives):
            return -10
        if move.id == "stockpile" and active.effects.get("stockpile3"):
            return -10
        if move.id in RECOVERY or move.heal:
            if hp >= 1:
                return -10
            if hp >= 0.9:
                return -9
        if move.weather and move.weather in battle.weather or move.terrain and move.terrain in battle.fields:
            return -10
        if move.id in SCREENS and SCREENS[move.id] in battle.side_conditions:
            return -10
        if move.id in HAZARDS:
            cond = HAZARDS[move.id]
            if battle.opponent_side_conditions.get(cond, 0) >= STACKABLE_CONDITIONS.get(cond, 1):
                return -10
        if move.id in HAZARD_REMOVAL and not battle.side_conditions:
            return -10
        if move.volatile_status is not None and move.volatile_status in opponent.effects:
            return -10
        return 0

    def _positives(self, battle: Battle, active: Pokemon, opponent: Pokemon, move: Move) -> int:
        """AIScript_Positives + DamageMoveViabilityIncrease."""
        cls = self._class(active)
        foe_move = self._predict_foe_move(battle, active, opponent)
        sv = lambda b: self._status_viability(cls, b, battle, active, opponent)
        bonus = 0
        sec = (move.secondary or [{}])[0]
        status = move.status
        if status is None and sec.get("status") and sec.get("chance", 0) >= 75:
            status = Status[sec["status"].upper()]
        if status is not None and not self._bad_status(status, opponent, active):
            if status in (Status.PSN, Status.TOX):
                bonus += sv(2 if {"venoshock", "barbbarrage"} & set(active.moves) else 1)
            elif status == Status.PAR:
                me, them = stat(active, "spe", True, battle), stat(opponent, "spe", False, battle)
                bonus += sv(2 if them >= me > them / 2 else 1)
            elif status == Status.BRN:
                phys = foe_move is not None and foe_move.category == MoveCategory.PHYSICAL
                if phys and self._hit(battle, opponent, active, foe_move, False) >= active.current_hp_fraction:
                    bonus += sv(3)
                else:
                    bonus += sv(2 if any(m.category == MoveCategory.PHYSICAL for m in self._attacks(opponent)) else 1)
            elif status == Status.SLP:
                if cls in ("setup_stats", "batonpass"):
                    bonus += 9 if (move.accuracy or 1) >= 0.8 else sv(3)
                elif cls == "setup_status":
                    bonus += 8 if (move.accuracy or 1) >= 0.8 else sv(3)
                else:
                    bonus += sv(3)
        heals = move.id in RECOVERY or move.heal or (move.drain and move.category != MoveCategory.STATUS)
        if heals and self._should_recover(battle, active, opponent, move, foe_move):
            if move.drain:
                bonus += sv(2) if cls == "stall" else 3
            else:
                bonus += 8 if cls in ("stall", "phazer") else sv(3)
        if move.is_protect_move and self._should_protect(battle, active, opponent, foe_move):
            bonus += (7 if battle.opponent_active_pokemon and battle.trapped else 3) if cls == "stall" else sv(3)
        boosts = self._self_boosts(move)
        if move.id == "stockpile":
            if {"swallow", "spitup"} & set(active.moves):
                bonus += sv(2)
            else:
                boosts = {"def": 1, "spd": 1}
        if move.category != MoveCategory.STATUS and sec.get("self", {}).get("boosts") and sec.get("chance", 0) >= 75:
            boosts = sec["self"]["boosts"]  # Power-Up Punch and kin
        if boosts:
            ups = [k for k, v in boosts.items() if v > 0]
            key = next((k for k in ("atk", "spa", "spe", "def", "spd") if k in ups and self._good_to_raise(k, active, opponent, battle)), None)
            if key is not None:
                bonus += self._stat_viability(cls, battle, active, opponent, move, foe_move, key)
        if move.id in HAZARDS and cls not in ("kill",):
            left = sum(1 for m in battle.opponent_team.values() if not m.fainted)
            bonus += sv(2) if left > 2 else (sv(1) if cls == "stall" else 0)
        if move.entry.get("selfSwitch") and move.id != "batonpass":
            pivot = self._should_pivot(battle, active, opponent, move, foe_move, cls)
            bonus += (3 if cls == "kill" else 9) if pivot == "pivot" else (-9 if pivot == "dont" else 0)
        if move.category != MoveCategory.STATUS:
            bonus += self._damage_tier(battle, active, opponent, move, foe_move, cls)
        return bonus

    def _damage_tier(self, battle, active, foe, move, foe_move, cls) -> int:
        """DamageMoveViabilityIncrease (singles)."""
        hp = foe.current_hp_fraction
        d = self._hit(battle, active, foe, move, True)
        kos = d >= hp
        first = self._goes_first(battle, move, active, foe, foe_move)
        has_protect = any(m.is_protect_move for m in active.moves.values())
        dies = foe_move is not None and self._hit(battle, foe, active, foe_move, False) >= active.current_hp_fraction
        if kos and first and ((move.accuracy or 1) >= 0.7 or dies):
            return self.ko_bonus
        if kos and not first and not has_protect:
            return {"setup_stats": 6}.get(cls, 8)
        attacks = [m for m in battle.available_moves if m.category != MoveCategory.STATUS]  # usable this turn
        fast = [m for m in attacks if self._goes_first(battle, m, active, foe, foe_move)]
        if dies and not has_protect and fast and cls not in ("stall", "phazer", "hazards") and \
                move.id == max(fast, key=lambda m: min(self._hit(battle, active, foe, m, True), hp)).id:
            return self.ko_bonus
        strongest = max(attacks, key=lambda m: (min(self._hit(battle, active, foe, m, True), hp), m.priority, m.accuracy or 1))
        if move.id == strongest.id:
            if kos and cls in ("stall", "phazer"):
                return 8
            return 4 if kos and cls == "hazards" else 2
        return 0

    def _should_protect(self, battle, active, foe, foe_move) -> bool:
        """ShouldProtect == USE_PROTECT (singles): the foe dies to residual damage, or would KO /
        2HKO us while it takes residual damage."""
        hurt = foe.status in (Status.PSN, Status.TOX, Status.BRN) or "leechseed" in str(foe.effects).lower()
        if foe.status in (Status.PSN, Status.TOX, Status.BRN) and foe.current_hp_fraction <= 1 / 8:
            return True
        d = self._foe_best(battle, active, foe)
        hp = active.current_hp_fraction
        return hurt and 2 * d >= hp

    def _should_pivot(self, battle, active, foe, move, foe_move, cls) -> str:
        """ShouldPivot (singles), the branches that need no prediction of a switch: 'pivot',
        'try' (no change) or 'dont' (-9)."""
        if not battle.available_switches:
            return "try"
        damager = cls in ("kill", "setup_stats", "setup_status", "screens_sweeper")
        flags = self._ranked_switches(battle)[0][2]
        we_ko = self._can_ko(battle, active, foe, True)
        we_2hko = self._can_ko(battle, active, foe, True, hits=2)
        d, hp = self._foe_best(battle, active, foe), active.current_hp_fraction
        boosted = damager and (active.boosts.get("atk", 0) > 0 or active.boosts.get("spa", 0) > 0)
        if self._goes_first(battle, move, active, foe, foe_move):
            if we_ko:
                return "try"
            if damager and we_2hko:
                return "pivot" if d >= hp else "dont"
            if not boosted and flags & {"walls", "resists"}:
                return "try"
            return "dont"
        if d >= hp:
            return "try"
        if 2 * d >= hp:
            if we_ko:
                return "try"
            return "pivot" if damager and "ko" in flags and flags & {"outspeeds", "walls", "resists"} else "dont"
        if we_ko:
            return "try" if not boosted else "dont"
        if we_2hko:
            return "pivot" if damager and "ko" in flags and flags & {"outspeeds", "walls", "resists"} else "dont"
        if damager and {"ko", "outspeeds"} <= flags:
            return "pivot"
        return "try" if not boosted and flags & {"outspeeds", "walls", "resists"} else "dont"

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
        return 2 * d >= 1 and any(m.id in RECOVERY or m.heal for m in mon.moves.values()) \
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
        if best_score < BASE_SCORE and not fresh:
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
            heals = any(m.id in RECOVERY or m.heal for m in active.moves.values())
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

    @staticmethod
    def _self_boosts(move: Move) -> dict | None:
        if move.self_boost:
            return move.self_boost
        if move.boosts and move.target == Target.SELF:
            return move.boosts
        return None

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
