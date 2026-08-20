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
from concurrent.futures import ThreadPoolExecutor
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
CLIP, VF_CLIP, ENTROPY, LR = 0.2, 10.0, 0.1, 2.5e-4  # higher entropy: 5-step
# episodes collapse exploration at 0.05
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


def train(updates: int, seed: int, port: int, info_mode: str, n_envs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    envs = [NuzlockeEnv(port=port + i, info_mode=info_mode, seed=seed + i)
            for i in range(n_envs)]
    pool = ThreadPoolExecutor(max_workers=n_envs) if n_envs > 1 else None
    agent = MaskedPolicy(OBS_DIM, N_META_ACTIONS).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    writer = SummaryWriter(
        f"runs/{datetime.now():%Y%m%d-%H%M%S}-nuzlocke{'x' + str(n_envs) if n_envs > 1 else ''}")
    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)

    ep_r, ep_len, cleared = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    cur_r = np.zeros(n_envs)
    cur_len = np.zeros(n_envs, dtype=int)
    steps_per_env = ROLLOUT // n_envs  # same total rollout per update

    initial = [e.reset(seed=seed + i) for i, e in enumerate(envs)]
    obs = [o for o, _ in initial]
    infos = [i for _, i in initial]
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
        for t in range(steps_per_env):
            # batched forward across envs
            o = torch.as_tensor(np.stack(obs), device=device)
            m = torch.as_tensor(
                np.stack([i["action_mask"] for i in infos]), device=device,
                dtype=torch.float32)
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(o, m)
            base = t * n_envs
            s[base:base + n_envs] = o
            masks[base:base + n_envs] = m
            acts[base:base + n_envs] = a
            lps[base:base + n_envs] = lp
            vals[base:base + n_envs] = v
            actions_np = [int(x) for x in a.tolist()]

            if n_envs > 1:  # concurrent steps (each blocks on its own battle)
                results = list(pool.map(
                    lambda i: envs[i].step(actions_np[i]), range(n_envs)))
            else:
                results = [envs[0].step(actions_np[0])]

            for i, (ob, r, term, trunc, info) in enumerate(results):
                rews[base + i] = r
                dones[base + i] = term or trunc
                cur_r[i] += r
                cur_len[i] += 1
                gstep += 1
                if term or trunc:
                    ep_r.append(cur_r[i])
                    ep_len.append(cur_len[i])
                    cleared.append(envs[i].trainer_idx)
                    cur_r[i], cur_len[i] = 0.0, 0
                    obs[i], infos[i] = envs[i].reset()
                else:
                    obs[i], infos[i] = ob, info
        with torch.no_grad():
            o = torch.as_tensor(np.stack(obs), device=device)
            m = torch.as_tensor(
                np.stack([i["action_mask"] for i in infos]), device=device,
                dtype=torch.float32)
            _, _, _, last_v = agent.get_action_and_value(o, m)
        # GAE per env over its strided slice (scalar bootstrap each), like train.py
        adv = torch.zeros_like(rews)
        for j in range(n_envs):
            sl = slice(j, ROLLOUT, n_envs)
            adv_j, _ = compute_gae(rews[sl], vals[sl], dones[sl],
                                   last_v[j], GAMMA, LAM)
            adv[sl] = adv_j
        ret = adv + vals
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
    p.add_argument("--envs", type=int, default=1,
                   help="parallel runs (one server per env on port..port+N-1)")
    p.add_argument("--info", choices=["none", "revealed", "full"],
                   default="full")
    p.add_argument("--random", action="store_true")
    args = p.parse_args()
    if args.random:
        manual_episode(NuzlockeEnv(port=args.port, info_mode=args.info,
                                   seed=args.seed), args.seed)
    else:
        train(args.updates, args.seed, args.port, args.info, args.envs)
