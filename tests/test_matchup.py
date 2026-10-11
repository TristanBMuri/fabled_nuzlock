"""Matchup matrix: safe / edge on pairs, and the set-cover chooser (pure)."""

from ai_util.matchup import cover, edge, safe


def P(ko_them, ko_us, faster=True, heals=False):
    return {"ko_them": ko_them, "ko_us": ko_us, "faster": faster, "heals": heals, "hit": 1 / ko_them, "take": 1 / ko_us}


def test_safe_and_edge():
    assert safe(P(2, 3)) and not safe(P(3, 3)) and not safe(P(2, 2))
    assert not safe(P(2, 3, faster=False)) and safe(P(2, 4, faster=False))  # moving second costs a hit of margin
    assert not safe(P(2, 3, faster=False, heals=True))  # Gyarados vs Roost Mantine
    assert safe(P(1, 3, faster=False, heals=True)) and safe(P(2, 3, faster=True, heals=True))
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
