"""rl/route.py: a known gap only matters while the mon carrying it is on the field."""

from rl.route import turn_gaps


def _mon(ability, moves):
    return {"ability": ability, "item": "", "moves": [[m, 10] for m in moves]}


def test_turn_gaps_look_at_the_active_mons_only():
    shinx, bidoof = _mon("rivalry", ["tackle"]), _mon("simple", ["doomdesire", "tackle"])
    starly = _mon("keeneye", ["quickattack"])
    dump = lambda p1: {"p1": {"pokemon": p1}, "p2": {"pokemon": [starly]}}  # noqa: E731
    assert turn_gaps(dump([starly, bidoof])) == []                    # Doom Desire on the bench
    assert turn_gaps(dump([bidoof, starly])) == ["move:doomdesire"]  # Doom Desire in
    assert turn_gaps(dump([shinx])) == ["ability:rivalry"]
