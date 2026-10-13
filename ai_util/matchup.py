"""Perfect-information matchups (2026-10-11): how many hits each side needs, from the calc
bot's damage model on the real sets under the field. `pair()` is what the calc bot switches
on (MATCHUP), `cover()` is what the planner picks a six with (--chooser cover), `rows_for()`
reads a candidate list's rows off one dummy battle per six species (never played).

A pair is *safe* when we KO in at most two hits and they need at least three on us; its
*edge* is their hits minus ours, plus a half for moving first. 2026-10-13 (the matrix
calibration, scripts/rr_matrix_calib.py): a healer costs one more hit unless it falls before
it heals (it disqualified every slower answer before); Air Balloon pops on the first hit, so
a Ground move lands from the second.
"""

import copy
import math

from ai_util.calc_bot import dmg_frac, stat
from poke_env.battle import PokemonType

SAFE_OURS, SAFE_THEIRS = 2, 3
HEALS = {"roost", "recover", "slackoff", "softboiled", "milkdrink", "synthesis", "moonlight", "morningsun",
         "shoreup", "rest", "wish", "lunarblessing", "strengthsap"}


def pair(me, opp, battle) -> dict:
    """{ko_them, ko_us, faster, heals, hit, take}: hits to KO each way (99 = never), who moves
    first, whether they carry a recovery move, and the best fraction each side takes per hit."""
    hit = max((dmg_frac(me, opp, mv, True, battle) for mv in me.moves.values()), default=0.0)
    take = max((dmg_frac(opp, me, mv, False, battle) for mv in opp.moves.values()), default=0.0)
    their_hp = opp.current_hp_fraction if opp.current_hp else 1.0  # a benched enemy reads as unknown HP: full
    our_hp = me.current_hp_fraction if me.current_hp else 1.0
    ko_them = math.ceil(their_hp / hit) if hit > 0 else 99
    if opp.item == "airballoon" and hit > 0:  # the balloon pops on the first hit; Ground works from the second
        bare = copy.copy(opp)
        bare._item = None
        ground = max((dmg_frac(me, bare, mv, True, battle) for mv in me.moves.values() if mv.type == PokemonType.GROUND), default=0.0)
        if ground > hit:
            ko_them = min(ko_them, 1 + math.ceil(max(their_hp - hit, 0) / ground))
    return {"ko_them": ko_them,
            "ko_us": math.ceil(our_hp / take) if take > 0 else 99,
            "faster": stat(me, "spe", True, battle) > stat(opp, "spe", False, battle),
            "heals": any(mv.id in HEALS for mv in opp.moves.values()), "hit": hit, "take": take}


def ours(p: dict) -> int:
    """Hits to KO them; a healer (Roost / Recover / ...) costs one more unless it falls before
    it heals: one hit, or two while moving first (2026-10-11: Gyarados "2/3" vs Roost Mantine
    was no answer)."""
    k = p["ko_them"]
    if p.get("heals") and not (k <= 1 or (k <= 2 and p["faster"])):
        k += 1
    return k


def safe(p: dict) -> bool:
    """We KO in at most two hits and survive the hits we absorb before our last one with one
    to spare (we absorb ko_them hits moving second, one fewer moving first; never fewer than
    three to KO us)."""
    k = ours(p)
    if k > SAFE_OURS:
        return False
    absorbed = k - (1 if p["faster"] else 0)
    return p["ko_us"] >= max(SAFE_THEIRS, absorbed + 2)


def race(p: dict) -> bool:
    """Looser than safe: we land the KO before they land theirs (one hit of margin, not two).
    A one-shot that absorbs one hit counts."""
    k = ours(p)
    if k > SAFE_OURS + 1:
        return False
    absorbed = k - (1 if p["faster"] else 0)
    return p["ko_us"] >= absorbed + 1


def edge(p: dict) -> float:
    return min(p["ko_us"], 9) - min(p["ko_them"], 9) + (0.5 if p["faster"] else 0.0)


def cover(rows: dict, enemies: list[str], k: int = 6) -> list:
    """rows: {candidate: {enemy: pair}}. Greedy set cover in two tiers: the candidate that is
    the safe answer to most uncovered enemies, then the race-winner to most, ties by total
    edge, until every enemy has one or none is left; enemies nobody answers get the best
    edge; the remaining slots go to total edge."""
    team, uncovered = [], set(enemies)
    while len(team) < k and uncovered:
        def gain(c):
            return (sum(safe(rows[c][e]) for e in uncovered if e in rows[c]),
                    sum(race(rows[c][e]) for e in uncovered if e in rows[c]),
                    sum(edge(rows[c][e]) for e in enemies if e in rows[c]))
        best = max((c for c in rows if c not in team), key=gain, default=None)
        if best is None or gain(best)[:2] == (0, 0):
            break
        team.append(best)
        uncovered -= {e for e in uncovered if e in rows[best] and race(rows[best][e])}
    for e in sorted(uncovered):  # no safe answer in the pool: the least bad one
        if len(team) >= k:
            break
        best = max((c for c in rows if c not in team and e in rows[c]), key=lambda c: edge(rows[c][e]), default=None)
        if best is not None:
            team.append(best)
    while len(team) < k and len(team) < len(rows):  # second answers: total edge
        team.append(max((c for c in rows if c not in team), key=lambda c: sum(edge(rows[c][e]) for e in enemies if e in rows[c])))
    return team


def rows_for(bridge, fmt: str, mons: dict, enemy_packed: str, arena: dict | None) -> dict:
    """{candidate: {enemy species: pair}} for Mon objects, read off dummy full-information battles
    (six candidates per battle; the mirrors carry both sides' real stats and the arena)."""
    from gauntlet import pack_team
    from poke_env.data import to_id_str
    from ai_util.sim import SimBattle
    out, names = {}, list(mons)
    for i in range(0, len(names), 6):
        chunk = names[i:i + 6]
        sb = SimBattle(bridge, fmt, pack_team([mons[s] for s in chunk]), enemy_packed, full_info=True, arena=arena)
        b = sb.mirrors["p1"]
        ours = list(b.team.values())
        theirs = list(b.opponent_team.values())
        for s in chunk:
            me = next((p for p in ours if p.species == to_id_str(mons[s].species)), None)
            if me is not None:
                out[s] = {opp.species: pair(me, opp, b) for opp in theirs}
        sb.close()
    return out
