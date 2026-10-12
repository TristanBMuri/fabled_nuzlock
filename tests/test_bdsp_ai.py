"""The Gen 4 (Platinum / BDSP) trainer AI on hand-built poke-env battles. No engine."""

import logging
import random

from poke_env.battle import Battle, Pokemon

from ai_util.bdsp_ai import BDSPAI


def _mon(species, moves, level=30, hp=1.0, active=False):
    mon = Pokemon(gen=8, species=species)
    mon._level, mon._active = level, active
    for m in moves:
        mon._add_move(m)
    mon._current_hp, mon._max_hp = int(100 * hp), 100
    return mon


def _battle(me, foe, bench=()):
    b = Battle("t", "p2", logging.getLogger("t"), gen=8)
    b._team = {f"p2: {m.species}": m for m in (me, *bench)}
    b._opponent_team = {f"p1: {foe.species}": foe}
    b._available_moves = list(me.moves.values())
    b._available_switches = list(bench)
    return b


AI = BDSPAI(battle_format="gen8customgame", start_listening=False, log_level=50)


def _picks(battle, n=40):
    out = []
    for seed in range(n):
        random.seed(seed)
        order = AI.choose_move(battle)
        out.append(getattr(order.order, "id", None) or getattr(order.order, "species", None))
    return out


def test_never_clicks_into_an_immunity():
    """Basic: Tackle into a Ghost is -10; Bite works, so Bite every time."""
    me = _mon("raticate", ["tackle", "bite"], active=True)
    foe = _mon("haunter", ["lick"], active=True)
    assert set(_picks(_battle(me, foe))) == {"bite"}


def test_takes_the_ko():
    """EvalAttack: a KO is +4, beating paralysis (+3 when slower) and setup (+2 at full HP)."""
    me = _mon("raticate", ["tackle", "thunderwave", "swordsdance"], active=True)
    foe = _mon("electrike", ["quickattack"], hp=0.05, active=True)
    assert set(_picks(_battle(me, foe))) == {"tackle"}


def test_only_immune_moves_switch_to_a_super_effective_bench_mon():
    """ShouldSwitch: two Normal attacks into a Ghost -> the bench mon with a Dark move, 8/9 of the
    time (two 2/3 rolls); otherwise it stays and clicks a -10 move, as the engine does."""
    me = _mon("raticate", ["tackle", "quickattack"], active=True)
    foe = _mon("haunter", ["lick"], active=True)
    dark, grass = _mon("houndour", ["bite"]), _mon("oddish", ["absorb"])
    picks = _picks(_battle(me, foe, bench=[grass, dark]))
    assert picks.count("houndour") >= 30 and "oddish" not in picks


def test_post_ko_replacement_prefers_type_matchup_with_a_super_effective_move():
    """PostKOSwitchIn stage 1: Rock vs Fire/Flying (x4) with Rock Throw beats everyone."""
    fainted = _mon("raticate", ["tackle"], hp=0.0, active=True)
    foe = _mon("charmeleon", ["ember"], active=True)
    rock, grass = _mon("geodude", ["rockthrow"]), _mon("oddish", ["absorb"])
    b = _battle(fainted, foe, bench=[grass, rock])
    assert AI._post_ko(b, b.available_switches, foe, fainted) is rock
