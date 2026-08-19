"""Train the nuzlocke meta-policy. Requires Showdown on the given port.

Usage:
  uv run python scripts/train_nuzlocke.py --random            # manual episode
  uv run python scripts/train_nuzlocke.py --updates 20        # PPO smoke
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from nuzlocke_env import N_META_ACTIONS, OBS_DIM, NuzlockeEnv  # noqa: E402
from ppo import MaskedPolicy, compute_gae, explained_variance, ppo_update  # noqa: E402

ROLLOUT = int(os.environ.get("ROLLOUT", 512))
GAMMA, LAM = 0.99, 0.95  # longer horizon: a run is ~50 decisions
CLIP, VF_CLIP, ENTROPY, LR = 0.2, 10.0, 0.05, 2.5e-4
EPOCHS, MINIBATCHES = 4, 4


def manual_episode(env, seed=0):
    """Random legal actions, verbose — the G2 sanity walk."""
    rng = np.random.RandomState(seed)
    obs, info = env.reset(seed=seed)
    steps = 0
    while True:
        print(env.render())
        valid = np.flatnonzero(info["action_mask"])
        a = int(rng.choice(valid))
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            print(env.render(), f"| done, steps={steps}")
            break


def train(updates: int, seed: int, port: int, info_mode: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = NuzlockeEnv(port=port, info_mode=info_mode, seed=seed)
    agent = MaskedPolicy(OBS_DIM, N_META_ACTIONS).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    writer = SummaryWriter(f"runs/{datetime.now():%Y%m%d-%H%M%S}-nuzlocke")
    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)

    ep_r, ep_len, cleared, deaths = deque(maxlen=20), deque(maxlen=20), deque(maxlen=20), deque(maxlen=20)
    cur_r, cur_len, cur_deaths = 0.0, 0, 0

    obs, info = env.reset(seed=seed)
    s = torch.zeros(ROLLOUT, OBS_DIM, device=device)
    masks = torch.zeros(ROLLOUT, N_META_ACTIONS, device=device)
    acts = torch.zeros(ROLLOUT, dtype=torch.long, device=device)
    lps = torch.zeros(ROLLOUT, device=device)
    rews = torch.zeros(ROLLOUT, device=device)
    dones = torch.zeros(ROLLOUT, device=device)
    vals = torch.zeros(ROLLOUT, device=device)

    gstep = 0
    for update in range(1, updates + 1):
        t0 = time.time()
        for t in range(ROLLOUT):
            o = torch.as_tensor(obs, device=device)
            m = torch.as_tensor(info["action_mask"], device=device,
                                dtype=torch.float32)
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(o, m)
            s[t], masks[t], acts[t], lps[t], vals[t] = o, m, a, lp, v
            obs, r, term, trunc, info = env.step(int(a.item()))
            rews[t], dones[t] = r, term or trunc
            cur_r += r
            cur_len += 1
            gstep += 1
            if term or trunc:
                ep_r.append(cur_r)
                ep_len.append(cur_len)
                cleared.append(env.trainer_idx)
                cur_r, cur_len = 0.0, 0
                obs, info = env.reset()
        with torch.no_grad():
            o = torch.as_tensor(obs, device=device)
            m = torch.as_tensor(info["action_mask"], device=device,
                                dtype=torch.float32)
            _, _, _, last_v = agent.get_action_and_value(o, m)
        adv, ret = compute_gae(rews, vals, dones, last_v, GAMMA, LAM)
        stats = ppo_update(agent, opt, s, masks, acts, lps, adv, ret, vals,
                           clip_coef=CLIP, vf_clip=VF_CLIP,
                           entropy_coef=ENTROPY, epochs=EPOCHS,
                           minibatches=MINIBATCHES)
        dt = time.time() - t0
        writer.add_scalar("episode_reward", np.mean(ep_r) if ep_r else 0, gstep)
        writer.add_scalar("trainers_cleared", np.mean(cleared) if cleared else 0, gstep)
        writer.add_scalar("episode_len", np.mean(ep_len) if ep_len else 0, gstep)
        for k, val in stats.items():
            writer.add_scalar(k, val, gstep)
        writer.add_scalar("explained_variance",
                          explained_variance(vals, ret), gstep)
        print(f"update {update:3d}/{updates} R={np.mean(ep_r) if ep_r else 0:+.1f} "
              f"cleared={np.mean(cleared) if cleared else 0:.1f} "
              f"len={np.mean(ep_len) if ep_len else 0:.0f} "
              f"H={stats['entropy']:.2f} {ROLLOUT/dt:.1f} st/s")
        torch.save({"model": agent.state_dict(), "arch": "mlp",
                    "obs_dim": OBS_DIM}, ckpt / "nuzlocke_latest.pt")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--info", choices=["none", "revealed", "full"],
                   default="full")
    p.add_argument("--random", action="store_true")
    args = p.parse_args()
    env = NuzlockeEnv(port=args.port, info_mode=args.info, seed=args.seed)
    if args.random:
        manual_episode(env, args.seed)
    else:
        env.close() if hasattr(env, "close") else None
        train(args.updates, args.seed, args.port, args.info)
