"""Observation encoder for the battler: full battle state as a flat vector.

12 mon slots (ours in team order = switch-action order, then opponent slots
canonicalized by species), then a field block. All values in [-1, 4].
"""

import numpy as np
from poke_env.battle import AbstractBattle, Pokemon
from poke_env.battle.field import Field
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

BATTLE_FORMAT = "gen9randombattle"
_DATA = GenData.from_format(BATTLE_FORMAT)
_TYPE_CHART = _DATA.type_chart

_TYPES = list(PokemonType)
_TYPE_INDEX = {t: i for i, t in enumerate(_TYPES)}
N_TYPES = len(_TYPES)
_STATUSES = [Status.BRN, Status.PAR, Status.PSN, Status.TOX, Status.SLP, Status.FRZ]
_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "evasion")
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_WEATHERS = (Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM, Weather.SNOW, Weather.HAIL)
_TERRAINS = (Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN, Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN)
_HAZARDS = (
    (SideCondition.STEALTH_ROCK, 1),
    (SideCondition.SPIKES, 3),
    (SideCondition.TOXIC_SPIKES, 2),
    (SideCondition.STICKY_WEB, 1),
)
_FLAGS, _N_STATUS, _N_BOOST, _N_MOVES, _N_STATS, _N_SPEED = 4, len(_STATUSES), len(_BOOST_KEYS), 40, len(_STAT_KEYS), 1
_N_REAL = 1 + len(_STAT_KEYS)  # level and the actual stats (own from the request, theirs estimated)
_PER_MON = _FLAGS + _N_STATUS + _N_BOOST + N_TYPES + _N_MOVES + _N_STATS + _N_SPEED + _N_REAL
_TYPES_OFF = _FLAGS + _N_STATUS + _N_BOOST
_MOVES_OFF = _TYPES_OFF + N_TYPES
_STATS_OFF = _MOVES_OFF + _N_MOVES
_FIELD = len(_WEATHERS) + len(_TERRAINS) + 1 + 2 * len(_HAZARDS)
N_FEATURES = 12 * _PER_MON + _FIELD


def _damage_multiplier(move, types) -> float:
    t1, t2 = types
    if move.category == MoveCategory.STATUS or t1 is None:
        return 1.0
    return float(
        np.clip(move.type.damage_multiplier(t1, t2, type_chart=_TYPE_CHART), 0.0, 4.0)
    )


def _base_stats(mon: Pokemon) -> dict:
    bs = mon.base_stats
    if bs:
        return bs
    entry = _DATA.pokedex.get(to_id_str(mon.species), {})
    return entry.get("baseStats", {})


def _speed_factor(base: float, boost: int) -> float:
    """Showdown's speed stages: x(2+boost)/2 above 0, x2/(2-boost) below."""
    return base * ((2 + boost) / 2 if boost > 0 else 2 / (2 - boost))


def _mon_block(mon: Pokemon | None, *, target_types, active, cmp_speed: float) -> np.ndarray:
    block = np.zeros(_PER_MON, dtype=np.float32)
    if mon is None:  # unrevealed opponent slot
        return block
    block[0] = mon.current_hp_fraction
    block[1] = float(mon.fainted)
    block[2] = float(mon is active)
    block[3] = 1.0  # present in battle.team / opponent_team => known
    if mon.status in _STATUSES:
        block[_FLAGS + _STATUSES.index(mon.status)] = 1.0
    for i, key in enumerate(_BOOST_KEYS):
        block[_FLAGS + _N_STATUS + i] = mon.boosts.get(key, 0) / 6.0
    for t in (mon.type_1, mon.type_2):
        if t is not None:
            block[_TYPES_OFF + _TYPE_INDEX[t]] = 1.0
    for j, move in enumerate(list(mon.moves.values())[:4]):
        off = _MOVES_OFF + 10 * j
        block[off] = move.base_power / 100.0
        block[off + 1] = _damage_multiplier(move, target_types)
        block[off + 2] = float(move.category == MoveCategory.STATUS)
        block[off + 3] = float(move.category == MoveCategory.PHYSICAL)
        block[off + 4] = float(move.category == MoveCategory.SPECIAL)
        block[off + 5] = float(move.priority > 0)
        block[off + 6] = (move.accuracy or 100) / 100.0
        # the terms of the heuristic scoring function: STAB, multi-hit
        # expectation, self-boost detection
        mon_types = [t for t in (mon.type_1, mon.type_2) if t is not None]
        block[off + 7] = float(move.type in mon_types)
        block[off + 8] = min(getattr(move, "expected_hits", 1) or 1, 5) / 5.0
        block[off + 9] = float(bool(move.boosts) and move.target == "self")
    for i, key in enumerate(_STAT_KEYS):
        block[_STATS_OFF + i] = _base_stats(mon).get(key, 0) / 255.0
    if cmp_speed > 0:
        block[_PER_MON - _N_REAL - 1] = float(_speed_factor(_base_stats(mon).get("spe", 0),
                                                            mon.boosts.get("spe", 0)) > cmp_speed)
    # level and real stats: what the damage formula needs and base stats cannot say
    off = _PER_MON - _N_REAL
    block[off] = mon.level / 100.0
    for i, key in enumerate(_STAT_KEYS):
        block[off + 1 + i] = min(_real_stat(mon, key), 999) / 500.0
    return block


def _real_stat(mon: Pokemon, key: str) -> float:
    """Own mons: the stat from the request (HP = max HP); others: estimated from
    base stats at the mon's level with the calc bot's enemy IV/EV guess."""
    if key == "hp":
        if mon.stats and mon.max_hp and mon.max_hp > 100:
            return mon.max_hp
    elif mon.stats and mon.stats.get(key):
        return mon.stats[key]
    base = _base_stats(mon).get(key, 0)
    v = ((2 * base + 31 + 85 // 4) * mon.level) // 100
    return v + mon.level + 10 if key == "hp" else v + 5


def _field_block(battle: AbstractBattle) -> np.ndarray:
    block = np.zeros(_FIELD, dtype=np.float32)
    for i, w in enumerate(_WEATHERS):
        block[i] = float(w in battle.weather)
    for i, t in enumerate(_TERRAINS):
        block[len(_WEATHERS) + i] = float(t in battle.fields)
    block[len(_WEATHERS) + len(_TERRAINS)] = min(battle.turn, 20) / 20.0
    off = len(_WEATHERS) + len(_TERRAINS) + 1
    for conds in (battle.side_conditions, battle.opponent_side_conditions):
        for j, (cond, stack) in enumerate(_HAZARDS):
            block[off + j] = min(conds.get(cond, 0), stack) / stack
        off += len(_HAZARDS)
    return block


def embed_battle(battle: AbstractBattle) -> np.ndarray:
    our_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    our_types = (our_active.type_1, our_active.type_2) if our_active else (None, None)
    opp_types = (opp_active.type_1, opp_active.type_2) if opp_active else (None, None)
    our_speed = (
        _speed_factor(_base_stats(our_active).get("spe", 0), our_active.boosts.get("spe", 0))
        if our_active else 0.0
    )
    opp_speed = (
        _speed_factor(_base_stats(opp_active).get("spe", 0), opp_active.boosts.get("spe", 0))
        if opp_active else 0.0
    )
    ours = list(battle.team.values())  # NOT sorted: switch actions index this order
    # opponent slots are canonicalized by species: reveal order varies across
    # battles and would present identical states as different inputs
    opps = sorted(battle.opponent_team.values(), key=lambda m: to_id_str(m.base_species))
    blocks = [
        _mon_block(ours[i] if i < len(ours) else None,
                   target_types=opp_types, active=our_active, cmp_speed=opp_speed)
        for i in range(6)
    ] + [
        _mon_block(opps[i] if i < len(opps) else None,
                   target_types=our_types, active=opp_active, cmp_speed=our_speed)
        for i in range(6)
    ]
    return np.concatenate(blocks + [_field_block(battle)])
