"""In-process simulator bridge: plays, mirrors, forks. Needs node + built Showdown."""

import shutil
from pathlib import Path

import pytest

from ai_util.sim import SimBattle, SimBridge, player_brain
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder

FMT = "gen8customgamenogimmicks"
_DIST = Path(__file__).resolve().parents[1] / "pokemon-showdown" / "dist" / "sim" / "index.js"
pytestmark = pytest.mark.skipif(not (shutil.which("node") and _DIST.exists()),
                                reason="node + built Showdown required")


def packed(text: str) -> str:
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(text))


TEAM_A = packed("Starly\nLevel: 12\n- Tackle\n- Quick Attack\n- Wing Attack\n\n"
                "Shinx\nLevel: 11\n- Tackle\n- Spark\n- Bite")
TEAM_B = packed("Bidoof\nLevel: 10\n- Tackle\n- Headbutt\n\n"
                "Geodude\nLevel: 12\n- Tackle\n- Rock Throw")


@pytest.fixture(scope="module")
def bridge():
    b = SimBridge()
    yield b
    b.close()


def brains():
    h = SimpleHeuristicsPlayer(battle_format=FMT, start_listening=False, log_level=50)
    return {"p1": player_brain(h), "p2": player_brain(h)}


def test_play_to_end(bridge):
    sb = SimBattle(bridge, FMT, TEAM_A, TEAM_B)
    assert sb.pending == {"p1": True, "p2": True}
    m = sb.mirrors["p1"]
    assert {p.species for p in m.team.values()} == {"starly", "shinx"}
    assert m.active_pokemon is not None and m.opponent_active_pokemon is not None
    sb.play(brains(), max_turns=60)
    assert sb.ended and sb.winner in ("p1", "p2", "")
    won, dead = sb.outcome()
    assert len(dead) == 2 and (won == (sb.winner == "p1"))
    assert m.finished
    sb.close()


def test_fork_continues_independently(bridge):
    sb = SimBattle(bridge, FMT, TEAM_A, TEAM_B)
    b = brains()
    sb.step({s: b[s](sb, s) for s in ("p1", "p2") if sb.pending[s]})
    turn_at_fork = sb.turn
    f = sb.fork()
    assert f.turn == turn_at_fork and f.pending == sb.pending
    assert f.mirrors["p1"].turn == sb.mirrors["p1"].turn
    assert [p.current_hp for p in f.mirrors["p1"].team.values()] == \
           [p.current_hp for p in sb.mirrors["p1"].team.values()]
    f.play(b, max_turns=60)
    assert f.ended and not sb.ended  # original untouched
    sb.play(b, max_turns=60)
    assert sb.ended
    sb.close()
    f.close()


def test_fork_seed_is_reproducible(bridge):
    """Same seed -> same chance events (common random numbers for search)."""
    sb = SimBattle(bridge, FMT, TEAM_A, TEAM_B)
    b = brains()
    sb.step({s: b[s](sb, s) for s in ("p1", "p2") if sb.pending[s]})
    runs = []
    for _ in range(2):
        f = sb.fork(seed=[1, 2, 3, 4])
        f.play(b, max_turns=60)
        runs.append((f.turn, f.winner, [(m["hp"], m["fainted"]) for m in f.state()["p1"]]))
        f.close()
    assert runs[0] == runs[1]
    sb.close()
