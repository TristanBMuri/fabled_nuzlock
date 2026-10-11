"""A scripted battler that runs the real damage formula (roadmap 1a, 2026-09-18).

Own stats come from the request; enemy stats and moves are known when the
battle runs with full info (SimBattle(full_info=True): the env built the
enemy's sets), else estimated from base stats at the enemy's level with
ENEMY_IV / ENEMY_EV (a boss-safe guess: route trainers in BDSP have 0/0,
bosses 31/510) and unrevealed moves guessed as a 60-power STAB. Decisions, in order: forced
switch -> best bench mon; a KO in hand -> the most accurate KO move (priority
first if slower) unless we are slower and die first; the next hit kills us
and a bench mon survives it -> switch; safe -> set up; else the strongest
expected hit. Beats SimpleHeuristics and the search line on the Edwin /
rival probes (NOTEBOOK.md); the teacher for the next prior.
"""

from poke_env.battle import Field, MoveCategory, PokemonType, SideCondition, Status, Weather
from poke_env.data import GenData
from poke_env.player import Player

GAME_FORMAT = "gen8customgamenogimmicks"
_CHART = GenData.from_format(GAME_FORMAT).type_chart
ROLL = 0.925          # mean damage roll
ENEMY_IV, ENEMY_EV = 31, 85
FIELD_AWARE = True    # weather / terrain / Tailwind in the numbers (False = the pre-2026-10-11 bot)

# field effects (2026-10-11: Radical Red's gyms fight under permanent weather / terrain)
RAIN, SUN = {Weather.RAINDANCE, Weather.PRIMORDIALSEA}, {Weather.SUNNYDAY, Weather.DESOLATELAND}
SAND, SNOW = {Weather.SANDSTORM}, {Weather.HAIL, Weather.SNOWSCAPE}
SPEED_ABILITY = {"swiftswim": RAIN, "chlorophyll": SUN, "sandrush": SAND, "slushrush": SNOW}
TERRAIN_BOOST = {Field.ELECTRIC_TERRAIN: PokemonType.ELECTRIC, Field.GRASSY_TERRAIN: PokemonType.GRASS,
                 Field.PSYCHIC_TERRAIN: PokemonType.PSYCHIC}


def _ability(mon) -> str:
    return (mon.ability or "").replace(" ", "").lower()


def grounded(mon) -> bool:
    return PokemonType.FLYING not in mon.types and _ability(mon) != "levitate" and mon.item != "airballoon"


def stat(mon, key: str, own: bool, battle=None) -> float:
    """The real stat when known (ours from the request, theirs under full info),
    else estimated; then boost stages, paralysis, and with `battle` the field:
    weather speed abilities, Surge Surfer, the side's Tailwind."""
    v = (mon.stats or {}).get(key)
    if not v:
        v = ((2 * mon.base_stats[key] + ENEMY_IV + ENEMY_EV // 4) * mon.level) // 100 + 5
    b = mon.boosts.get(key, 0)
    v *= (2 + b) / 2 if b >= 0 else 2 / (2 - b)
    if key == "spe":
        if mon.status == Status.PAR:
            v *= 0.5
        if battle is not None and FIELD_AWARE:
            ab = _ability(mon)
            if SPEED_ABILITY.get(ab, set()) & set(battle.weather) or \
                    (ab == "surgesurfer" and Field.ELECTRIC_TERRAIN in battle.fields):
                v *= 2
            if SideCondition.TAILWIND in (battle.side_conditions if own else battle.opponent_side_conditions):
                v *= 2
    return v


def max_hp(mon, own: bool) -> int:
    if own and mon.max_hp and mon.stats:
        return mon.max_hp
    if (mon.stats or {}).get("hp"):  # theirs, under full info (max_hp is a percentage for them)
        return mon.stats["hp"]
    return ((2 * mon.base_stats["hp"] + ENEMY_IV + ENEMY_EV // 4) * mon.level) // 100 + mon.level + 10


# abilities that void a move: by the move's type, or by a move flag (sound, bullet)
ABILITY_IMMUNE_TYPE = {"levitate": "Ground", "eartheater": "Ground",
                       "waterabsorb": "Water", "dryskin": "Water", "stormdrain": "Water",
                       "voltabsorb": "Electric", "lightningrod": "Electric", "motordrive": "Electric",
                       "flashfire": "Fire", "wellbakedbody": "Fire", "sapsipper": "Grass"}
ABILITY_IMMUNE_FLAG = {"soundproof": "sound", "bulletproof": "bullet"}


def effectiveness(move, dfn) -> float:
    """Type multiplier of `move` on `dfn`, zero when dfn's ability voids it (known under
    full info; the game's AI never clicks Scald into Dry Skin, so neither may the model of it)."""
    mult = dfn.damage_multiplier(move)
    ab = (dfn.ability or "").replace(" ", "").lower()
    if ab in ABILITY_IMMUNE_TYPE and move.type is not None and move.type.name.capitalize() == ABILITY_IMMUNE_TYPE[ab]:
        return 0.0
    if ab in ABILITY_IMMUNE_FLAG and (move.entry.get("flags") or {}).get(ABILITY_IMMUNE_FLAG[ab]):
        return 0.0
    if ab == "wonderguard" and mult < 2:
        return 0.0
    return mult


def field_mod(att, dfn, move, battle) -> tuple[float, float | None]:
    """(damage multiplier, accuracy override) from the battle's weather and terrain:
    rain / sun on Water and Fire (the primal weathers void the opposed type), Hurricane
    and Thunder sure-hit in rain and 50% in sun, Blizzard sure-hit in snow, sandstorm
    x1.5 SpD for Rock, snow x1.5 Def for Ice, Electric / Grassy / Psychic Terrain x1.3
    for a grounded user, Grassy halves Earthquake / Bulldoze on a grounded target,
    Misty halves Dragon moves on one."""
    if battle is None or not FIELD_AWARE:
        return 1.0, None
    mod, acc = 1.0, None
    w = set(battle.weather)
    if w & RAIN:
        if move.type == PokemonType.WATER:
            mod *= 1.5
        elif move.type == PokemonType.FIRE:
            mod *= 0.0 if Weather.PRIMORDIALSEA in w else 0.5
        if move.id in ("hurricane", "thunder"):
            acc = 1.0
    elif w & SUN:
        if move.type == PokemonType.FIRE:
            mod *= 1.5
        elif move.type == PokemonType.WATER:
            mod *= 0.0 if Weather.DESOLATELAND in w else 0.5
        if move.id in ("hurricane", "thunder"):
            acc = 0.5
    elif w & SAND and move.category == MoveCategory.SPECIAL and PokemonType.ROCK in dfn.types:
        mod /= 1.5
    elif w & SNOW:
        if move.category == MoveCategory.PHYSICAL and PokemonType.ICE in dfn.types:
            mod /= 1.5
        if move.id == "blizzard":
            acc = 1.0
    for terrain, typ in TERRAIN_BOOST.items():
        if terrain in battle.fields and move.type == typ and grounded(att):
            mod *= 1.3
    if Field.GRASSY_TERRAIN in battle.fields and move.id in ("earthquake", "bulldoze", "magnitude") and grounded(dfn):
        mod *= 0.5
    if Field.MISTY_TERRAIN in battle.fields and move.type == PokemonType.DRAGON and grounded(dfn):
        mod *= 0.5
    return mod, acc


def dmg_frac(att, dfn, move, att_own: bool, battle=None) -> float:
    """Expected fraction of dfn's max HP one use of move removes (accuracy included);
    with `battle`, under its weather and terrain."""
    if move.category == MoveCategory.STATUS:
        return 0.0
    hp = max_hp(dfn, not att_own)
    if move.damage:  # Seismic Toss / Night Shade / fixed damage
        d = att.level if move.damage == "level" else float(move.damage)
        return min(d, hp) / hp * (move.accuracy or 1)
    if not move.base_power:
        return 0.0
    phys = move.category == MoveCategory.PHYSICAL
    a = stat(att, "atk" if phys else "spa", att_own, battle)
    d = stat(dfn, "def" if phys else "spd", not att_own, battle)
    base = ((2 * att.level / 5 + 2) * move.base_power * a / d) / 50 + 2
    mod = ROLL * (1.5 if move.type in att.types else 1.0) * effectiveness(move, dfn)
    mod *= getattr(move, "expected_hits", 1) or 1
    if att.status == Status.BRN and phys:
        mod *= 0.5
    fmod, acc = field_mod(att, dfn, move, battle)
    return base * mod * fmod / hp * (move.accuracy or 1 if acc is None else acc)


def threat(opp, mon, battle=None) -> float:
    """Best expected fraction the enemy's active takes off `mon` this turn. Unknown
    moves: a 60-power STAB of each of its types in its better category."""
    moves = list(opp.moves.values())
    if moves:
        return max(dmg_frac(opp, mon, m, False, battle) for m in moves)
    phys = opp.base_stats["atk"] >= opp.base_stats["spa"]
    a = stat(opp, "atk" if phys else "spa", False, battle)
    d = stat(mon, "def" if phys else "spd", True, battle)
    best = 0.0
    for t in opp.types:
        if t is None:
            continue
        base = ((2 * opp.level / 5 + 2) * 60 * a / d) / 50 + 2
        eff = t.damage_multiplier(mon.type_1, mon.type_2, type_chart=_CHART)
        best = max(best, base * ROLL * 1.5 * eff / max_hp(mon, True))
    return best


class CalcPlayer(Player):
    def choose_move(self, battle):
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if me is None or opp is None:
            return self.choose_random_move(battle)
        thr = threat(opp, me, battle) if not me.fainted else 0.0
        opp_prio = max((m.priority for m in opp.moves.values()), default=0)
        faster = stat(me, "spe", True, battle) > stat(opp, "spe", False, battle)

        def switch_score(m):
            t = threat(opp, m, battle)
            hit = max((dmg_frac(m, opp, mv, True, battle) for mv in m.moves.values()), default=0)
            survive = 2.0 if t < m.current_hp_fraction / 2 else (1.0 if t < m.current_hp_fraction else 0.0)
            return survive + hit - t

        if battle.force_switch or not battle.available_moves:
            if battle.available_switches:
                return self.create_order(max(battle.available_switches, key=switch_score))
            return self.choose_random_move(battle)

        scored = [(dmg_frac(me, opp, mv, True, battle), mv) for mv in battle.available_moves]
        best_frac, best_mv = max(scored, key=lambda x: x[0])
        # a KO in hand: the mean roll covers the HP left; P(KO) ~ accuracy
        kos = [(mv.accuracy or 1, mv) for f, mv in scored if f / (mv.accuracy or 1) >= opp.current_hp_fraction]
        we_die = thr >= me.current_hp_fraction * ROLL  # a near-max roll kills us
        if kos:
            acc, mv = max(kos, key=lambda x: (x[1].priority > opp_prio, x[0]))
            if faster or mv.priority > opp_prio or not we_die:
                return self.create_order(mv)
        if we_die and not (kos and faster) and battle.available_switches:
            cand = max(battle.available_switches, key=switch_score)
            if switch_score(cand) >= 1.0:  # someone survives the hit
                return self.create_order(cand)
        if thr < 0.2 and me.current_hp_fraction > 0.6:  # safe: set up
            for mv in battle.available_moves:
                if mv.boosts and mv.target == "self" and sum(mv.boosts.values()) >= 2 \
                        and max(me.boosts.get(s, 0) for s in mv.boosts) < 2:
                    return self.create_order(mv)
        return self.create_order(best_mv)


def calc_brain():
    from ai_util.sim import player_brain
    return player_brain(CalcPlayer(battle_format=GAME_FORMAT, start_listening=False, log_level=50))
