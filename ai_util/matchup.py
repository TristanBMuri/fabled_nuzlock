"""Perfect-information matchups (2026-10-11): how many hits each side needs, from the calc
bot's damage model on the real sets under the field. `pair()` is what the calc bot switches
on (MATCHUP), `cover()` is what the planner picks a six with (--chooser cover), `rows_for()`
reads a candidate list's rows off one dummy battle per six species (never played).

A pair is *safe* when we KO in at most two hits and they need at least three on us; its
*edge* is their hits minus ours, plus a half for moving first. 2026-10-13 (the matrix
calibration, scripts/rr_matrix_calib.py): a healer costs one more hit unless it falls before
it heals (it disqualified every slower answer before); Air Balloon pops on the first hit, so
a Ground move lands from the second; a resist berry halves the first hit of its type and is
gone (hits_to_ko). Not a heal berry: counting Berry Juice / Oran / Iapapa as HP turned 2/6 cells
into 3/6 "nones" that win every duel (false nones 58 -> 105); the tiers' cap on our hits is
the question there, not the berry. A cell may instead be *judged by the duel* (B3,
2026-10-13): `win` / `hp` from playing the one-on-one (rr_team_probe.duel) make its tiers —
the paper discards 3-hit answers that win because the enemy AI opens with Stockpile / Toxic /
Recover, and it cannot know; the duel does.
"""

import copy
import math

from ai_util.calc_bot import RESIST_BERRY, dmg_frac, stat

SAFE_OURS, SAFE_THEIRS = 2, 3
DUEL_SAFE, DUEL_SAFE_HP, DUEL_RACE = 0.9, 0.5, 0.5  # duel-judged cells: win rate (and HP left on a win) per tier
HEALS = {"roost", "recover", "slackoff", "softboiled", "milkdrink", "synthesis", "moonlight", "morningsun",
         "shoreup", "rest", "wish", "lunarblessing", "strengthsap"}


def pair(me, opp, battle) -> dict:
    """{ko_them, ko_us, faster, heals, hit, take}: hits to KO each way (99 = never), who moves
    first, whether they carry a recovery move, and the best fraction each side takes per hit."""
    hit = max((dmg_frac(me, opp, mv, True, battle) for mv in me.moves.values()), default=0.0)
    take = max((dmg_frac(opp, me, mv, False, battle) for mv in opp.moves.values()), default=0.0)
    their_hp = opp.current_hp_fraction if opp.current_hp else 1.0  # a benched enemy reads as unknown HP: full
    our_hp = me.current_hp_fraction if me.current_hp else 1.0
    ko_them = hits_to_ko(their_hp, hit, hit) if hit > 0 else 99
    if opp.item == "airballoon" and hit > 0 or opp.item in RESIST_BERRY:  # spent on the first hit: full hits from the second
        bare = copy.copy(opp)
        bare._item = None
        first = hit if opp.item == "airballoon" else max((dmg_frac(me, opp, mv, True, battle) for mv in me.moves.values()
                                                         if mv.type == RESIST_BERRY[opp.item]), default=0.0)
        rest = max((dmg_frac(me, bare, mv, True, battle) for mv in me.moves.values()), default=0.0)
        if first > 0 and rest > hit:
            ko_them = min(ko_them, hits_to_ko(their_hp, first, rest))
    return {"ko_them": ko_them,
            "ko_us": math.ceil(our_hp / take) if take > 0 else 99,
            "faster": stat(me, "spe", True, battle) > stat(opp, "spe", False, battle),
            "heals": any(mv.id in HEALS for mv in opp.moves.values()), "hit": hit, "take": take}


def hits_to_ko(hp: float, first: float, rest: float) -> int:
    """Hits that take `hp` to zero: `first` on the first, `rest` after it (an item spent on the
    first hit: the balloon pops, the berry is eaten); 99 when nothing lands."""
    if first == rest:
        return math.ceil(hp / first) if first > 0 else 99
    if first >= hp:
        return 1
    return 1 + math.ceil((hp - first) / rest - 1e-9) if rest > 0 else 99  # (1 - 1/6) / (1/6) = 5.000000000000001


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
    three to KO us). Duel-judged: wins almost always and comes out with half its HP."""
    if "win" in p:
        return p["win"] >= DUEL_SAFE and p["hp"] >= DUEL_SAFE_HP
    k = ours(p)
    if k > SAFE_OURS:
        return False
    absorbed = k - (1 if p["faster"] else 0)
    return p["ko_us"] >= max(SAFE_THEIRS, absorbed + 2)


def race(p: dict) -> bool:
    """Looser than safe: we land the KO before they land theirs (one hit of margin, not two).
    A one-shot that absorbs one hit counts. Duel-judged: wins more often than not."""
    if "win" in p:
        return p["win"] > DUEL_RACE
    k = ours(p)
    if k > SAFE_OURS + 1:
        return False
    absorbed = k - (1 if p["faster"] else 0)
    return p["ko_us"] >= absorbed + 1


def edge(p: dict) -> float:
    if "win" in p:  # duel-judged: wins weighted up by the HP kept, losses count against (a liability on the bench)
        return p["win"] * (1 + p["hp"]) - (1 - p["win"])
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
