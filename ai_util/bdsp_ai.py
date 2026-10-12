"""Pokémon Platinum (Gen 4) trainer AI as the enemy AI of the BDSP Nuzlocke sim (singles).

BDSP reimplements the Gen 4 AI; the reference is pret/pokeplatinum (`tr_ai/script.s`,
`trainer_ai.c`, BattleAI_PostKOSwitchIn in `battle_lib.c`). Per turn, as TrainerAI_PickCommand:
the switch check, then every usable move starts at 100 and the flag scripts run in bit order
over all moves: Basic (CheckBadMove), EvalAttack (TryToFaint), Expert (CheckViability), mask
`BDSPAI.flags`. Top score wins, ties at random. Rolls: global `random`, the engine's 0..255.
Damage: the calc bot's model at the max roll, one hit of a multi-hit move, no accuracy.
The enemy knows the player's team, as in rr_ai.py.

Ported, quirks kept (Dry Skin never penalised, Recovery -8 when *faster*, U-turn's last-mon
exit, Substitute's inverted status test, the post-KO u8 overflow):
  Basic: immunities (chart, Levitate, Wonder Guard, absorb abilities, Soundproof, no-calc
    moves), sleep, poison/Toxic, paralysis, burn, confusion/Swagger/Flatter, stat ups/downs
    (incl. Bulk Up, Calm Mind, Cosmic Power, Dragon Dance, Tickle), recovery, screens,
    Substitute, Leech Seed, hazards, weather, Baton Pass, Fake Out, Explosion, Dream Eater.
  EvalAttack: all (KO +4, +6 for the +1-priority effect; not strongest -1; 4x +2 ...).
  Expert: the above minus burn and Sandstorm (no Gen 4 entry), plus drain, Icy Wind/Rock Tomb/Mud Shot,
    Rest, Protect, U-turn, Sucker Punch. Multi-hit and plain priority have no Expert entry.
  Switching: Perish Song, Wonder Guard, only-immune moves, absorb, Natural Cure, the
    super-effective / +4 vetoes, immune / resist switch-ins; the post-KO replacement.
Not ported (no score change): the rest of Expert (Haze, Roar, Counter, Encore, Curse, Belly
Drum, Trick, Pursuit, OHKO, charge/recharge, recoil ...) and of Basic (Attract, Disable,
Encore, Curse, OHKO, Stockpile, Defense Curl, Charge, Captivate, Memento, Yawn ...), Mold
Breaker, trainer items (TrainerAI_ShouldUseItem).
poke-env limits: "the move that last hit us" and the foe's last move class are the player's
`last_move` (none: no class); speed ties count as faster; invalid moves and the engine's
switch bookkeeping are `available_moves` / `available_switches`; Return, Low Kick, Flail ...
(no base power in Showdown data) count as no-calc; super-effective checks use damaging moves.
"""

from __future__ import annotations

import random

from poke_env.battle import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from ai_util.calc_bot import ROLL, dmg_frac, max_hp, stat

AI_FLAG_BASIC, AI_FLAG_EVAL_ATTACK, AI_FLAG_EXPERT = 1, 2, 4

# effects the engine scores without its damage calc (sNoDamageCalcMoveEffects)
NO_CALC = {"explosion", "selfdestruct", "dreameater", "razorwind", "skyattack", "skullbash",
           "solarbeam", "hyperbeam", "gigaimpact", "blastburn", "frenzyplant", "hydrocannon",
           "rockwrecker", "roaroftime", "spitup", "focuspunch", "superpower", "eruption",
           "waterspout", "suckerpunch", "headsmash"}
NOT_IN_BASIC = {"dreameater", "suckerpunch", "headsmash", "eruption", "waterspout", "solarbeam", "skyattack", "spitup"}
PRIORITY_1 = {"quickattack", "machpunch", "bulletpunch", "aquajet", "iceshard", "shadowsneak",
              "vacuumwave", "extremespeed"}  # BATTLE_EFFECT_PRIORITY_1 in Gen 4
SOUND = {"growl", "roar", "sing", "supersonic", "screech", "snore", "uproar", "metalsound",
         "grasswhistle", "bugbuzz", "chatter"}
NAMED = {"growth": ("up", ("spa",))}  # move id -> its Gen 4 effect as named here (None: unported)
for _e, _ms in {"explode": "explosion selfdestruct", "protect": "protect detect", "rain": "raindance",
                "sun": "sunnyday", "sand": "sandstorm", "recover": "recover softboiled milkdrink slackoff healorder roost",
                "synthesis": "moonlight morningsun synthesis", "speeddownhit": "icywind rocktomb mudshot",
                None: "defensecurl charge captivate"}.items():
    NAMED.update(dict.fromkeys(_ms.split(), _e))
NAMED.update({m: m for m in ("dreameater", "fakeout", "uturn", "suckerpunch", "substitute", "leechseed", "spikes",
                             "toxicspikes", "stealthrock", "hail", "batonpass", "reflect", "lightscreen", "rest",
                             "swagger", "flatter")})
RESTORE_HALF = {"recover", "softboiled", "milkdrink", "slackoff", "healorder"}
STATUS_EFFECT = {Status.SLP: "sleep", Status.PSN: "poison", Status.TOX: "toxic",
                 Status.PAR: "paralyze", Status.BRN: "burn"}
STAT_ORDER = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
UP_PAIRS = {("atk", "def"), ("spa", "spd"), ("def", "spd"), ("atk", "spe")}  # Bulk Up, Calm Mind, Cosmic Power, DD
SNAPPED = (Effect.CURSE, Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3,
           Effect.ATTRACT, Effect.LEECH_SEED, Effect.YAWN)  # Expert_Protect's residual effects
WEATHER = {"rain": Weather.RAINDANCE, "sun": Weather.SUNNYDAY, "hail": Weather.HAIL, "sand": Weather.SANDSTORM}
HAZARDS = {"spikes": (SideCondition.SPIKES, 3), "toxicspikes": (SideCondition.TOXIC_SPIKES, 2),
           "stealthrock": (SideCondition.STEALTH_ROCK, 1)}
ABSORB = {PokemonType.FIRE: "flashfire", PokemonType.WATER: "waterabsorb", PokemonType.ELECTRIC: "voltabsorb"}


def _odds(n: int) -> bool:
    """The fall-through of `IfRandomLessThan n`: True with probability (256 - n) / 256."""
    return random.randrange(256) >= n


def _ab(mon) -> str:
    return (mon.ability or "").replace(" ", "").lower()


def _pct(mon) -> int:
    """curHP * 100 / maxHP, as the scripts compare it."""
    return int(mon.current_hp_fraction * 100 + 1e-6)


def _eff(move: Move, dfn) -> float:
    """ApplyTypeChart's verdict: the type chart, Levitate on Ground, Wonder Guard on a damaging
    non-super-effective move (absorbing abilities are separate script checks)."""
    if move.type == PokemonType.GROUND and _ab(dfn) == "levitate":
        return 0.0
    m = dfn.damage_multiplier(move)
    return 0.0 if _ab(dfn) == "wonderguard" and m < 2 and move.base_power else m


def _damaging(mon) -> list[Move]:
    return [m for m in mon.moves.values() if m.base_power]


def _effect(move: Move):
    """The Gen 4 battle effect of a move, for the effects ported here: a name, ("up", stats)
    / ("down", stats) for stat moves, or None."""
    if move.id in NAMED:
        return NAMED[move.id]
    if move.category != MoveCategory.STATUS:
        return "drain" if move.drain else None
    if move.status in STATUS_EFFECT:
        return STATUS_EFFECT[move.status]
    if move.volatile_status == Effect.CONFUSION:
        return "confuse"
    b = move.boosts or {}
    stats = tuple(sorted(b, key=STAT_ORDER.index))
    if b and all(v > 0 for v in b.values()) and (len(stats) == 1 or stats in UP_PAIRS):
        return ("up", stats)
    if b and all(v < 0 for v in b.values()) and (len(stats) == 1 or stats == ("atk", "def")):  # Tickle
        return ("down", stats)
    return None


class BDSPAI(Player):
    """Gen 4 trainer AI. `flags` is the trainer's AI mask; BDSP trainers here get all three."""

    flags = AI_FLAG_BASIC | AI_FLAG_EVAL_ATTACK | AI_FLAG_EXPERT

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        if isinstance(battle, DoubleBattle) or not isinstance(battle, Battle):
            return self.choose_random_move(battle)
        me, foe = battle.active_pokemon, battle.opponent_active_pokemon
        if foe is None or (me is None and not battle.force_switch):
            return self.choose_random_move(battle)
        if battle.force_switch:
            mon = self._post_ko(battle, battle.available_switches, foe, me)
            return self.create_order(mon) if mon else self.choose_random_move(battle)
        if any(m.id == "recharge" for m in battle.available_moves):
            return self.create_order(battle.available_moves[0])
        target = self._should_switch(battle, me, foe)
        if target is not None:
            return self.create_order(target)
        if not battle.available_moves:
            return self.choose_random_move(battle)
        scores = self.score_moves(battle, me, foe)
        top = max(scores.values())
        return self.create_order(random.choice([m for m in battle.available_moves if scores[m.id] == top]))

    def score_moves(self, battle: Battle, me: Pokemon, foe: Pokemon) -> dict[str, int]:
        """TrainerAI_MainSingles: each flag script over every move, in bit order."""
        trick_room = Field.TRICK_ROOM in battle.fields
        a, b = stat(me, "spe", True, battle), stat(foe, "spe", False, battle)
        self._slower = a > b if trick_room else a < b  # COMPARE_SPEED_SLOWER (ties: faster)
        scores = {m.id: 100 for m in battle.available_moves}
        for bit, script in ((AI_FLAG_BASIC, self._basic), (AI_FLAG_EVAL_ATTACK, self._eval_attack),
                            (AI_FLAG_EXPERT, self._expert)):
            if self.flags & bit:
                for m in battle.available_moves:
                    scores[m.id] = max(0, scores[m.id] + script(battle, me, foe, m))
        return scores

    # --- damage (TrainerAI_CalcDamage at the max roll) ---

    @staticmethod
    def _dmg(battle, att, dfn, move: Move, raw: bool = False) -> float | None:
        """Max-roll fraction of dfn's max HP one hit of `move` removes; None where the engine
        makes no damage comparison (status, power <= 1, sNoDamageCalcMoveEffects unless `raw`)."""
        if (move.id in NO_CALC and not raw) or (move.base_power <= 1 and not move.damage):
            return None
        hits = getattr(move, "expected_hits", 1) or 1
        return dmg_frac(att, dfn, move, True, battle, with_acc=False) / ROLL / hits

    def _strongest(self, battle, me, foe, move) -> bool:
        """FlagMoveDamageScore == AI_MOVE_IS_HIGHEST_DAMAGE (ties count as highest)."""
        d = self._dmg(battle, me, foe, move)
        return all((self._dmg(battle, me, foe, m) or 0) <= d for m in me.moves.values())

    @staticmethod
    def _bench(team: dict, active) -> list:
        return [m for m in team.values() if not m.fainted and m is not active and not m.active]

    def _has_se(self, me, foe) -> bool:
        return any(_eff(m, foe) >= 2 for m in _damaging(me))

    # --- Basic_Main (CheckBadMove) ---

    def _basic(self, battle: Battle, me, foe, move: Move) -> int:
        e, ab = _effect(move), _ab(foe)
        if self._dmg(battle, me, foe, move) is not None:  # Basic_CheckForImmunity
            if _eff(move, foe) == 0:
                return -10
            if (ab in ("voltabsorb", "motordrive") and move.type == PokemonType.ELECTRIC) or \
                    (ab == "waterabsorb" and move.type == PokemonType.WATER) or \
                    (ab == "flashfire" and move.type == PokemonType.FIRE):
                return -12  # (Dry Skin is never checked: the script tests Levitate twice)
        elif move.category != MoveCategory.STATUS and e != "explode" and move.id not in NOT_IN_BASIC:
            if _eff(move, foe) == 0:  # Basic_CheckNonStandardDamageOrChargeTurn
                return -10
        if ab == "soundproof" and move.id in SOUND:
            return -10
        foe_side, my_side = battle.opponent_side_conditions, battle.side_conditions
        safeguard = SideCondition.SAFEGUARD in foe_side
        weather = set(battle.weather)
        foe_left = len(self._bench(battle.opponent_team, foe))
        if e == "sleep":
            return -10 if foe.status or safeguard or ab in ("insomnia", "vitalspirit") else 0
        if e == "explode":
            if _eff(move, foe) == 0 or ab == "damp":
                return -10
            return 0 if self._bench(battle.team, me) else (-10 if foe_left else -1)
        if e == "dreameater":
            return -8 if foe.status != Status.SLP else (-10 if _eff(move, foe) == 0 else 0)
        if isinstance(e, tuple):
            return (self._basic_up if e[0] == "up" else self._basic_down)(battle, me, foe, e[1])
        if e in ("recover", "synthesis"):
            return -8 if _pct(me) == 100 else 0
        if e in ("toxic", "poison"):
            return -10 if {PokemonType.STEEL, PokemonType.POISON} & {foe.type_1, foe.type_2} or \
                ab in ("immunity", "magicguard", "poisonheal") or foe.status or safeguard or \
                (ab == "leafguard" and Weather.SUNNYDAY in weather) or \
                (ab == "hydration" and Weather.RAINDANCE in weather) else 0
        if e in ("reflect", "lightscreen"):
            cond = SideCondition.REFLECT if e == "reflect" else SideCondition.LIGHT_SCREEN
            return -8 if cond in my_side else 0
        if e in ("confuse", "swagger", "flatter"):
            return -5 if Effect.CONFUSION in foe.effects else (-10 if ab == "owntempo" or safeguard else 0)
        if e == "paralyze":
            return -10 if _eff(move, foe) == 0 or ab in ("limber", "magicguard") or foe.status or safeguard or \
                (move.id == "thunderwave" and ab in ("motordrive", "voltabsorb")) else 0
        if e == "burn":
            fire = PokemonType.FIRE in (foe.type_1, foe.type_2)
            return -10 if ab in ("waterveil", "magicguard") or foe.status or fire or safeguard else 0
        if e == "substitute":
            return -8 if Effect.SUBSTITUTE in me.effects else (-10 if _pct(me) < 26 else 0)
        if e == "leechseed":
            grass = PokemonType.GRASS in (foe.type_1, foe.type_2)
            return -10 if Effect.LEECH_SEED in foe.effects or grass or ab == "magicguard" else 0
        if e in HAZARDS:
            cond, full = HAZARDS[e]
            return -10 if foe_side.get(cond, 0) >= full or not foe_left else 0
        if e in WEATHER:
            skip = {"rain": ("swiftswim", "hydration"), "sun": ("flowergift", "leafguard", "solarpower")}
            if e in skip and _ab(me) not in skip[e] and ab == "hydration" and foe.status:
                return -8 if e == "rain" else -10
            if WEATHER[e] in weather:
                return -8
            return (-8 + 8 * (_ab(me) == "icebody")) if e == "hail" and ab == "icebody" else 0
        if e == "batonpass":
            return 0 if self._bench(battle.team, me) else -10
        if e == "fakeout":
            return 0 if me.first_turn else -10
        return 0

    @staticmethod
    def _basic_up(battle, me, foe, stats) -> int:
        """Basic_CheckHighStatStage_* and the two-stat variants (second stat at +6: -8)."""
        if ("spe" in stats and Field.TRICK_ROOM in battle.fields) or \
                (stats[0] in ("accuracy", "evasion") and "noguard" in (_ab(me), _ab(foe))):
            return -10
        if _ab(me) == "simple":
            return -10 if any(me.boosts.get(s, 0) > 2 for s in stats) else 0
        for s, pen in zip(stats, (-10, -8)):
            if me.boosts.get(s, 0) == 6:
                return pen
        return 0

    @staticmethod
    def _basic_down(battle, me, foe, stats) -> int:
        """Basic_CheckLowStatStage_*, Basic_CheckTickle."""
        ab, s = _ab(foe), stats[0]
        if foe.boosts.get(s, 0) == -6 or (s == "atk" and ab == "hypercutter") or \
                (s == "spe" and (Field.TRICK_ROOM in battle.fields or ab == "speedboost")) or \
                (s in ("accuracy", "evasion") and "noguard" in (_ab(me), ab)) or \
                (s == "accuracy" and ab == "keeneye") or ab in ("clearbody", "whitesmoke"):
            return -10
        return -8 if len(stats) > 1 and foe.boosts.get(stats[1], 0) == -6 else 0

    # --- EvalAttack_Main (TryToFaint) ---

    def _eval_attack(self, battle: Battle, me, foe, move: Move) -> int:
        d = self._dmg(battle, me, foe, move)
        if d is not None and d >= foe.current_hp_fraction:  # IfCurrentMoveKills USE_MAX_DAMAGE
            if move.id in ("futuresight", "doomdesire"):
                return 4 if _odds(170) else 0
            return 6 if move.id in PRIORITY_1 else 4
        if d is not None and not self._strongest(battle, me, foe, move):
            return -1
        s = -2 if move.id in ("explosion", "selfdestruct", "focuspunch", "suckerpunch") and _odds(51) else 0
        return s + (2 if _eff(move, foe) == 4 and _odds(80) else 0)

    # --- Expert_Main (CheckViability) ---

    def _expert(self, battle: Battle, me, foe, move: Move) -> int:
        e, hp, fhp, slower = _effect(move), _pct(me), _pct(foe), self._slower
        weather = set(battle.weather)
        mine = set(me.moves)
        if e == "sleep":
            return 1 if mine & {"dreameater", "nightmare"} and _odds(128) else 0
        if e == "drain":
            return -3 if _eff(move, foe) < 1 and _odds(50) else 0
        if e == "explode":
            s, ev = 0, foe.boosts.get("evasion", 0)
            if ev >= 1:
                s -= 1 + (ev >= 4 and _odds(128))
            if hp >= 80 and not slower:
                return s - 3 if _odds(50) else s
            if hp > 50:
                return s - 1 if _odds(50) else s
            return s + _odds(128) + (hp <= 30 and _odds(50))
        if e == "dreameater":
            return -1 if _eff(move, foe) < 1 else 3 * (foe.status == Status.SLP and _odds(51))
        if isinstance(e, tuple):
            return (self._expert_up if e[0] == "up" else self._expert_down)(me, foe, e[1], hp, fhp, slower)
        if e == "speeddownhit":
            return 0 if _eff(move, foe) < 1 else ((2 if _odds(70) else 0) if slower else -3)
        if e in ("recover", "synthesis"):
            s = -2 if e == "synthesis" and weather & {Weather.HAIL, Weather.RAINDANCE, Weather.SANDSTORM} else 0
            if hp == 100:
                return s - 3
            if not slower:
                return s - 8
            if hp >= 70 and _odds(30):
                return s - 3
            return s + 2 * (("snatch" not in foe.moves or _odds(100)) and _odds(20))
        if e == "rest":
            if not slower:
                if hp == 100:
                    return -8
                if hp >= 40 and (hp > 50 or _odds(70)):
                    return -3
            elif hp >= 60 and (hp > 70 or _odds(50)):
                return -3
            return 3 * (("snatch" not in foe.moves or _odds(50)) and _odds(10))
        if e in ("toxic", "leechseed"):
            s = -3 * (hp <= 50 and _odds(50)) - 3 * (fhp <= 50 and _odds(50)) if _damaging(me) else 0
            # (or a +1-SpDef move: none exist)
            return s + 2 * (any(_effect(m) == "protect" for m in me.moves.values()) and _odds(60))
        if e in ("reflect", "lightscreen"):
            if hp < 50:
                return -2
            s = hp >= 90 and _odds(128)
            cat = MoveCategory.PHYSICAL if e == "reflect" else MoveCategory.SPECIAL
            return s + (foe.last_move is not None and foe.last_move.category == cat and _odds(64))
        if e in ("confuse", "swagger", "flatter"):
            if e == "swagger" and "psychup" in mine:
                return -5 if foe.boosts.get("atk", 0) >= -2 else 3 + 2 * (battle.turn <= 1)
            s = int(e != "confuse" and _odds(128))  # Flatter / Swagger's coin flip
            return s if fhp > 70 else s - _odds(128) - (fhp <= 50) - (fhp <= 30)
        if e == "poison":
            return -1 if hp < 50 or fhp <= 50 else 0
        if e == "paralyze":
            return (3 if _odds(20) else 0) if slower else (-1 if hp <= 70 else 0)
        if e == "substitute":
            s = 1 if "focuspunch" in mine and _odds(96) else 0
            s -= sum(_odds(100) for _ in range(0 if hp > 90 else 1 if hp > 70 else 2 if hp > 50 else 3))
            last = _effect(foe.last_move) if foe.last_move is not None and not slower else None
            if (last in ("sleep", "toxic", "poison", "paralyze", "burn") and foe.status is None) or \
                    (last == "confuse" and Effect.CONFUSION not in foe.effects) or \
                    (last == "leechseed" and Effect.LEECH_SEED not in foe.effects):
                s += _odds(100)
            return s
        if e == "protect":
            return self._expert_protect(me, foe)
        if e in HAZARDS:
            return _odds(128) and 1 + (bool(mine & {"roar", "whirlwind"}) and _odds(64))
        if e == "batonpass":
            keys = ("atk", "def", "spa", "spd", "evasion")
            low = hp <= 70 if slower else hp <= 60
            if any(me.boosts.get(k, 0) >= 3 for k in keys):
                return 2 if low and _odds(80) else 0
            if any(me.boosts.get(k, 0) >= 2 for k in keys):
                return 0 if (hp < 70 if slower else hp <= 60) else -2
            return -2
        if e in ("rain", "sun", "hail"):
            if e == "rain" and slower and _ab(me) == "swiftswim":
                return 1
            if hp < 40:
                return -1
            if weather & ({Weather.HAIL, Weather.RAINDANCE, Weather.SUNNYDAY, Weather.SANDSTORM} - {WEATHER[e]}):
                return 1 + (e == "hail") * (2 * ("blizzard" in mine) + 2 * (_ab(me) == "icebody"))
            good, if_statused = {"rain": ("raindish", "hydration"), "sun": ("flowergift", "leafguard")}.get(e, ("", ""))
            return int(_ab(me) == good or (_ab(me) == if_statused and me.status is not None))
        if e == "fakeout":
            return 2
        if e == "suckerpunch":
            return -1 if _eff(move, foe) < 1 else int(_odds(64))
        if e == "uturn":
            return self._expert_uturn(battle, me, foe, move, fhp, slower)
        return 0

    @staticmethod
    def _expert_up(me, foe, stats, hp, fhp, slower) -> int:
        """Expert_Status{Attack,Defense,Speed,SpAttack,SpDefense,Accuracy,Evasion}Up, DragonDance."""
        if stats == ("atk", "spe"):
            return int(_odds(128)) if slower else -(hp <= 50 and _odds(70))
        if stats == ("spe",):
            return (3 if _odds(70) else 0) if slower else -3
        key = {("atk", "def"): "def", ("spa", "spd"): "spd", ("def", "spd"): "spd"}.get(stats, stats[0])
        b = me.boosts.get(key, 0)
        if key == "accuracy":
            return -2 * (b >= 3 and _odds(50)) - 2 * (hp <= 70)
        if key == "evasion":
            s = 3 * (hp >= 90 and _odds(100)) - (b >= 3 and _odds(128))
            if foe.status == Status.TOX and (hp > 50 or _odds(80)) and _odds(50):
                s += 3
            s += 3 * (Effect.LEECH_SEED in foe.effects and _odds(70))
            s += 2 * ((Effect.INGRAIN in me.effects or Effect.AQUA_RING in me.effects) and _odds(128))
            s += 3 * (Effect.CURSE in foe.effects and _odds(70))
            return s if hp > 70 or b == 0 else s - 2 * (hp < 40 or fhp < 40 or _odds(70))
        s = -1 if b >= 3 and _odds(100) else (2 if b < 3 and hp == 100 and _odds(128) else 0)
        if key in ("atk", "spa"):
            return s if hp > 70 else s - 2 * (hp < 40 or _odds(40 if key == "atk" else 70))
        if hp >= 70 and not _odds(200):
            return s
        if hp < 40:
            return s - 2
        last, wrong = foe.last_move, MoveCategory.SPECIAL if key == "def" else MoveCategory.PHYSICAL
        if last is not None and last.base_power and last.category == wrong:
            return s - 2
        return s - 2 if _odds(60) else s

    @staticmethod
    def _expert_down(me, foe, stats, hp, fhp, slower) -> int:
        """Expert_Status{Attack,Defense,Speed,SpAttack,SpDefense,Accuracy,Evasion}Down (Tickle: Defense)."""
        key = "def" if len(stats) > 1 else stats[0]
        b, s = foe.boosts.get(key, 0), 0
        if key == "spe":
            return (2 if _odds(70) else 0) if slower else -3
        if key in ("atk", "spa"):
            if b != 0:
                s -= 1 + (hp <= 90) + 2 * (b <= -3 and _odds(50))
            s -= 2 * (fhp <= 70)
            pointless = MoveCategory.SPECIAL if key == "atk" else MoveCategory.PHYSICAL
            last = foe.last_move
            return s - 2 * (last is not None and last.category == pointless and _odds(128))
        if key == "accuracy":
            if (hp < 70 or fhp <= 70) and _odds(100):
                s -= 1
            s -= 2 * (me.boosts.get("accuracy", 0) <= -2 and _odds(80))
            s += 2 * (foe.status == Status.TOX and _odds(70))
            s += 2 * (Effect.LEECH_SEED in foe.effects and _odds(70))
            s += (Effect.INGRAIN in me.effects or Effect.AQUA_RING in me.effects) and _odds(128)
            s += 2 * (Effect.CURSE in foe.effects and _odds(70))
            return s if hp > 70 or b == 0 else s - 2 * (hp < 40 or fhp < 40 or _odds(70))
        if (hp < 70 or b <= -3) and _odds(50):  # Defense, SpDefense, Evasion
            s -= 2
        return s - 2 * (fhp <= 70)

    @staticmethod
    def _expert_protect(me, foe) -> int:
        s = -2 if foe.moves.keys() & {"feint", "shadowforce"} and _odds(128) else 0
        chain = me.protect_counter
        if chain > 1:
            return s - 2
        locked = Effect.LOCK_ON in me.effects
        if me.status == Status.TOX or any(x in me.effects for x in SNAPPED) or \
                any(m in RESTORE_HALF or m == "defensecurl" for m in foe.moves):
            return s if locked else s - 2
        if foe.status == Status.TOX or any(x in foe.effects for x in SNAPPED) or locked or not _odds(85):
            s += 2
        s -= _odds(128)
        if chain:
            s -= 1 + _odds(128)
        return s

    def _expert_uturn(self, battle, me, foe, move, fhp, slower) -> int:
        if _eff(move, foe) < 1:
            return -1
        bench = self._bench(battle.team, me)
        if not bench:
            return 0
        s = -2 if self._has_se(me, foe) and _odds(64) else 0
        mine = max((self._dmg(battle, me, foe, m) or 0) for m in me.moves.values())
        # IfPartyMemberDealsMoreDamage: the bench mons' moves, run with the active mon's stats
        more = any((self._dmg(battle, me, foe, m) or 0) > mine for b in bench for m in b.moves.values())
        if not more and _odds(64):
            return s - 2
        if fhp > 70:
            s += _odds(64) + _odds(128)
        elif fhp > 30:
            s += _odds(128)
        else:
            s += _odds(128) and _odds(128)
        return s + (1 if not slower else _odds(128))

    # --- switching (TrainerAI_ShouldSwitch, BattleAI_PostKOSwitchIn) ---

    def _should_switch(self, battle: Battle, me, foe):
        """The bench mon to switch to this turn, or None to fight."""
        bench = battle.available_switches
        if battle.trapped or not bench:
            return None
        if Effect.PERISH1 in me.effects:  # AI_PerishSongKO: post-KO logic
            return self._post_ko(battle, bench, foe, me)
        if _ab(foe) == "wonderguard" and not self._has_se(me, foe):  # AI_CannotDamageWonderGuard
            for b in bench:
                for m in _damaging(b):
                    if _eff(m, foe) >= 2 and random.randrange(3) < 2:
                        return b
        atk = _damaging(me)  # AI_OnlyIneffectiveMoves
        if len(atk) >= 2 and all(foe.damage_multiplier(m) == 0 for m in atk):
            # singles: each move is rolled twice (once per "opposing slot", both the same mon)
            for b in bench:
                for m in _damaging(b):
                    if _eff(m, foe) >= 2 and (random.randrange(3) < 2 or random.randrange(3) < 2):
                        return b
            for b in bench:
                for m in _damaging(b):
                    if _eff(m, foe) == 1 and (random.randrange(2) == 0 or random.randrange(2) == 0):
                        return b
        hit = foe.last_move  # the move that last hit us (moveHit), as poke-env can read it
        if not (self._has_se(me, foe) and random.randrange(3) != 0) and hit is not None and \
                hit.base_power and hit.type in ABSORB and _ab(me) != ABSORB[hit.type]:
            for b in bench:  # AI_HasAbsorbAbilityInParty
                if _ab(b) == ABSORB[hit.type] and random.randrange(2):
                    return b
        if me.status == Status.SLP and _ab(me) == "naturalcure" and _pct(me) >= 50:
            if hit is None and random.randrange(2):
                return self._post_ko(battle, bench, foe, me)
            if (hit is None or not hit.base_power) and random.randrange(2):
                return self._post_ko(battle, bench, foe, me)
            mon = self._party_se(bench, foe, hit, True, 1) or self._party_se(bench, foe, hit, False, 1)
            if mon is not None:
                return mon
            if random.randrange(2):
                return self._post_ko(battle, bench, foe, me)
        if any(_eff(m, foe) >= 2 and random.randrange(10) != 0 for m in _damaging(me)):
            return None  # a super-effective move: stay (90% per such move)
        if sum(v for v in me.boosts.values() if v > 0) >= 4:
            return None
        return self._party_se(bench, foe, hit, True, 2) or self._party_se(bench, foe, hit, False, 3)

    @staticmethod
    def _party_se(bench, foe, hit, immune: bool, n: int):
        """AI_HasPartyMemberWithSuperEffectiveMove: a bench mon immune to (or resisting) the last
        hit with a super-effective move on the foe, each such move switching 1/n of the time."""
        if hit is None or not hit.base_power:
            return None
        for b in bench:
            e = _eff(hit, b)
            if (e == 0) if immune else (0 < e < 1):
                for m in _damaging(b):
                    if _eff(m, foe) >= 2 and random.randrange(n) == 0:
                        return b
        return None

    def _post_ko(self, battle, bench, foe, fainted):
        """BattleAI_PostKOSwitchIn. Stage 1: the best offensive type matchup (a u8: two x4 types
        overflow) that also has a super-effective move; stage 2: the most damage (u8), as dealt
        by the fainted active mon with the bench mon's moves. Ties: party order."""
        if not bench:
            return None

        def type_score(b):
            return sum(int(40 * foe.damage_multiplier(t)) for t in (b.type_1, b.type_2 or b.type_1)) % 256

        left = list(bench)
        while left and type_score(best := max(left, key=type_score)) > 0:
            if any(_eff(m, foe) >= 2 for m in _damaging(best)):
                return best
            left.remove(best)
        hp = max_hp(foe, False)
        return max(bench, key=lambda b: max(
            (int((self._dmg(battle, fainted or b, foe, m, raw=True) or 0) * hp) % 256 for m in b.moves.values()),
            default=0))
