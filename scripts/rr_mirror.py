"""Battler anchor (rl/RR_PLAN.md, the team / battler loop): our search plays the boss's own six
against the RR AI playing the same six, arena as in the real fight (the boss keeps its side's
arena conditions, e.g. Koga's Tailwind). The teams are equal by construction, so the result
measures the player, not the team builder.

  uv run python scripts/rr_mirror.py --boss KOGA --n 12 --engine 20000 --legal --arena --matchup --death-weight 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rr_team_probe as p  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boss", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--engine", type=int, default=20000)
    ap.add_argument("--search", default="9,8")
    ap.add_argument("--death-weight", type=float, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legal", action="store_true")
    ap.add_argument("--arena", action="store_true")
    ap.add_argument("--matchup", action="store_true")
    ap.add_argument("--calc", action="store_true", help="the calc bot plays instead of the search (a weaker player)")
    a = ap.parse_args()
    args = argparse.Namespace(**vars(a), n_solo=1, n_team=1, n_search=a.n, n_screen=0, normal=False, blind_field=False,
                              items=True, engine_opp="minimax", chooser="cover")
    ctx = p.setup(args, a.boss, announce=False)
    from ai_util.sim import SimBattle
    wins = deaths = 0
    for i in range(a.n):
        sb = SimBattle(ctx.bridge, p.rp.FMT, ctx.enemy_packed, ctx.enemy_packed, full_info=True, arena=ctx.arena, seed=ctx.seeds[i])
        p.random.seed(sum(ctx.seeds[i]))
        sb.play({"p1": ctx.calc if a.calc else ctx.searcher, "p2": ctx.enemy}, 100)
        st = sb.state()
        won, dead = st["winner"] == "p1", sum(m["fainted"] for m in st["p1"])
        wins += won
        deaths += dead
        print(f"{a.boss} mirror {i + 1}/{a.n}: {'win' if won else 'loss'}, our deaths {dead}, theirs {sum(m['fainted'] for m in st['p2'])}, turns {sb.turn}", flush=True)
        sb.close()
    print(f"{a.boss} mirror ({'calc bot' if a.calc else f'engine {a.engine}'} vs RR AI, same six): win {wins / a.n:.0%}, deaths/mon {deaths / a.n / 6:.2f}", flush=True)
    ctx.bridge.close()


if __name__ == "__main__":
    main()
