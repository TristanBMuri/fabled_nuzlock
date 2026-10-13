"""Matchup matrix: safe / edge on pairs, and the set-cover chooser (pure)."""

from ai_util.matchup import cover, edge, hits_to_ko, race, safe


def P(ko_them, ko_us, faster=True, heals=False):
    return {"ko_them": ko_them, "ko_us": ko_us, "faster": faster, "heals": heals, "hit": 1 / ko_them, "take": 1 / ko_us}


def test_safe_and_edge():
    assert safe(P(2, 3)) and not safe(P(3, 3)) and not safe(P(2, 2))
    assert not safe(P(2, 3, faster=False)) and safe(P(2, 4, faster=False))  # moving second costs a hit of margin
    assert not safe(P(2, 3, faster=False, heals=True))  # Gyarados vs Roost Mantine
    assert safe(P(1, 3, faster=False, heals=True)) and safe(P(2, 3, faster=True, heals=True))
    assert race(P(1, 2, faster=False)) and not safe(P(1, 2, faster=False))  # a one-shot that absorbs one hit
    assert not race(P(2, 2, faster=False)) and race(P(2, 2, faster=True)) and not race(P(2, 2, faster=False, heals=True))
    assert edge(P(1, 4)) > edge(P(2, 3)) > edge(P(2, 2))


def test_cover_prefers_answers_over_damage():
    enemies = ["a", "b", "c"]
    rows = {"wall": {"a": P(2, 5), "b": P(2, 4), "c": P(4, 2)},     # answers a and b
            "sweeper": {"a": P(1, 2), "b": P(1, 2), "c": P(1, 2)},  # KOs everything, dies to everything
            "niche": {"a": P(5, 2), "b": P(5, 2), "c": P(2, 3)},    # the only answer to c
            "dud": {"a": P(6, 1), "b": P(6, 1), "c": P(6, 1)}}
    team = cover(rows, enemies, k=3)
    assert team[0] == "wall" and "niche" in team and "dud" not in team
    assert cover(rows, enemies, k=2) == ["wall", "niche"]
    rows["racer"] = {"a": P(1, 2, faster=False), "b": P(6, 1), "c": P(6, 1)}  # race-winner vs a, nothing else
    assert cover({k: v for k, v in rows.items() if k not in ("wall", "sweeper")}, ["a"], k=1) == ["racer"]  # second tier when no safe one


def test_duel_judged_cells():
    D = lambda win, hp: {**P(5, 2), "win": win, "hp": hp}  # a paper none the duel overrules
    assert safe(D(0.95, 0.7)) and not safe(D(0.95, 0.2)) and not safe(D(0.8, 0.9))
    assert race(D(0.6, 0.1)) and not race(D(0.5, 1.0))  # a coin flip is no answer
    assert edge(D(0.6, 0.3)) > edge(D(0.1, 1.0)) and edge(D(1.0, 0.9)) > edge(D(1.0, 0.2)) and edge(D(0.0, 0.0)) < 0 < edge(D(1.0, 0.0))
    rows = {"paper": {"a": P(2, 3)}, "stockpiled": {"a": D(1.0, 0.8)}}  # the 3-hit answer the paper discards wins the duel
    assert cover({k: {"a": {**v["a"], "win": 0.0, "hp": 0.0}} if k == "paper" else v for k, v in rows.items()}, ["a"], k=1) == ["stockpiled"]


def test_hits_to_ko_spends_the_item_on_the_first_hit():
    assert hits_to_ko(1.0, 0.5, 0.5) == 2 and hits_to_ko(1.0, 0.0, 0.0) == 99 and hits_to_ko(1.0, 1.0, 1.0) == 1
    assert hits_to_ko(1.0, 0.3, 0.6) == 3  # Air Balloon / Wacan: the first hit is the small one
    assert hits_to_ko(1.0, 1 / 6, 1 / 6) == 6 and hits_to_ko(1.0, 1 / 6, 1 / 3) == 4  # no floating-point seventh hit
