"""Evaluate a trained checkpoint on fresh teams.

Modes:
- --pool N   battle against the first N pool teams (in-distribution check)
- --random   battle against freshly harvested Showdown random teams
             (out-of-distribution — proxies nuzlocke team entropy)

Uses the greedy policy (argmax over masked logits).

Usage: uv run python scripts/eval_teams.py --checkpoint checkpoints/ppo_latest_attention.pt --random --battles 20
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from battle_env import N_ACTIONS, make_env  # noqa: E402
from ppo import AttentionPolicy, MaskedPolicy  # noqa: E402
from harvest_teams import NODE_SCRIPT, SHOWDOWN_DIR  # noqa: E402


def harvest(count: int) -> list[str]:
    """Fresh randombattle teams via Showdown's generator (eval seed)."""
    raw = subprocess.run(
        ["node", "-e", NODE_SCRIPT, str(SHOWDOWN_DIR), "100", str(count),
         "nuzlock-eval"],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(raw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["pool", "random"], default="random")
    p.add_argument("--pool-size", type=int, default=4)
    p.add_argument("--battles", type=int, default=20)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, weights_only=False)
    arch = ck.get("arch", "attention")
    agent = (MaskedPolicy if arch == "mlp" else AttentionPolicy)(N_ACTIONS)
    agent.load_state_dict(ck["model"])
    agent.eval()

    if args.mode == "random":
        teams = harvest(args.battles)
        tmp = Path(tempfile.mkdtemp(prefix="eval_teams_"))
        for i, t in enumerate(teams):
            (tmp / f"team_{i:02d}.txt").write_text(t)
        # Reuse make_env's pool machinery by pointing the pool dir at tmp.
        import battle_env
        battle_env.POOL_FILES = sorted(str(f) for f in tmp.glob("*.txt"))
        env = make_env(team_mode="pool", arch=arch, pool_size=len(teams),
                       port=args.port)
    else:
        env = make_env(team_mode="pool", arch=arch, pool_size=args.pool_size,
                       port=args.port)

    wins = 0
    done = 0
    while done < args.battles:
        try:
            obs, _ = env.reset()
            while True:
                o = torch.as_tensor(obs["observation"]).unsqueeze(0)
                m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent(o, m)
                action = int(logits.argmax(dim=1).item())
                obs, _, term, trunc, _ = env.step(np.int64(action))
                if term or trunc:
                    break
            won = bool(env.env.battle1.won)
        except TimeoutError as e:
            # Reset can time out when the opponent player's state has drifted;
            # rebuilding the whole env is the reliable recovery (no retry infra).
            print(f"battle {done:2d}: reset timeout ({e}), rebuilding env...")
            del env
            env = make_env(team_mode="pool", arch=arch,
                           pool_size=len(teams) if args.mode == "random" else args.pool_size,
                           port=args.port)
            continue
        wins += won
        done += 1
        print(f"battle {done - 1:2d}: won={won} turns={env.env.battle1.turn}")
    print(f"\n=== {args.mode} mode: {wins}/{args.battles} = {wins/args.battles:.0%} ===")


if __name__ == "__main__":
    main()
