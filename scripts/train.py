"""Train the battler with masked PPO. Requires the local Showdown server on
port 8000 (see README).

Usage:
  uv run python scripts/train.py --random                       # pool-mode sanity
  uv run python scripts/train.py --arch mlp --team-mode fixed    # legacy baseline
  uv run python scripts/train.py --arch attention --updates 500  # pool A/B run
"""

import argparse
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from battle_env import N_ACTIONS, N_TEAMS_AVAIL, N_TOKENS, OBS_DIM, TOKEN_DIM, make_env
from ppo import AttentionPolicy, MaskedPolicy, compute_gae, explained_variance, ppo_update

# --- config (single place to tune) ---
ROLLOUT = int(os.environ.get("ROLLOUT", 1024))  # env steps per update, across all envs
GAMMA = 0.95
LAM = 0.95
CLIP = 0.2
VF_CLIP = 5.0
ENTROPY = 0.05
LR = 2.5e-4
EPOCHS = 4
MINIBATCHES = 4


def obs_shape(arch):
    return (OBS_DIM,) if arch == "mlp" else (N_TOKENS, TOKEN_DIM)


def make_policy(arch):
    return MaskedPolicy(OBS_DIM, N_ACTIONS) if arch == "mlp" else AttentionPolicy(N_ACTIONS)


def random_battles(env, arch, n=3):
    """Env sanity check: play n battles with uniformly random valid actions."""
    expected = obs_shape(arch)
    for i in range(n):
        obs, _ = env.reset()
        steps = 0
        while True:
            assert obs["observation"].shape == expected, obs["observation"].shape
            valid = np.flatnonzero(obs["action_mask"])
            assert len(valid) > 0
            obs, reward, term, trunc, _ = env.step(np.int64(np.random.choice(valid)))
            steps += 1
            if term or trunc:
                break
        won = env.env.battle1.won
        opp_idx = getattr(env.env.agent2._team, "last_idx", None)
        if arch == "attention":
            species_dim = obs["observation"][0].argmax()
            print(f"battle {i}: turns={env.env.battle1.turn} steps={steps} "
                  f"won={won} opp_team={opp_idx} active_species_token={species_dim}")
        else:
            print(f"battle {i}: turns={env.env.battle1.turn} steps={steps} won={won}")


def train(total_updates: int, seed: int, arch: str, team_mode: str, pool_size,
          n_envs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ports = [8000 + i for i in range(n_envs)]  # one Showdown server per env
    envs = [make_env(team_mode=team_mode, arch=arch, pool_size=pool_size, port=p)
            for p in ports]
    agent = make_policy(arch).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    writer = SummaryWriter(f"runs/{datetime.now():%Y%m%d-%H%M%S}-{arch}-{team_mode}")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=n_envs)

    shape = obs_shape(arch)
    wins = deque(maxlen=20)
    ep_rewards = deque(maxlen=20)
    n_opp = pool_size if (team_mode == "pool" and pool_size) else N_TEAMS_AVAIL
    opp_wins = [deque(maxlen=20) for _ in range(n_opp)]
    steps_per_env = ROLLOUT // n_envs  # same total rollout as single-env runs

    obs = [e.reset()[0] for e in envs]
    ep_reward = np.zeros(n_envs)
    s = torch.zeros((ROLLOUT, *shape), device=device)
    masks = torch.zeros(ROLLOUT, N_ACTIONS, device=device)
    actions = torch.zeros(ROLLOUT, dtype=torch.long, device=device)
    logprobs = torch.zeros(ROLLOUT, device=device)
    rewards = torch.zeros(ROLLOUT, device=device)
    dones = torch.zeros(ROLLOUT, device=device)
    values = torch.zeros(ROLLOUT, device=device)

    global_step = 0
    for update in range(1, total_updates + 1):
        t0 = time.time()
        for t in range(steps_per_env):
            # One batched forward for all envs.
            o = torch.as_tensor(np.stack([ob["observation"] for ob in obs]), device=device)
            m = torch.as_tensor(np.stack([ob["action_mask"] for ob in obs]),
                                device=device, dtype=torch.float32)
            with torch.no_grad():
                acts, lps, _, vals = agent.get_action_and_value(o, m)

            base = t * n_envs
            s[base:base + n_envs] = o
            masks[base:base + n_envs] = m
            actions[base:base + n_envs] = acts
            logprobs[base:base + n_envs] = lps
            values[base:base + n_envs] = vals

            # Step all envs concurrently (each blocks on its own websocket).
            acts_np = [int(a) for a in acts.tolist()]
            results = list(pool.map(lambda i: envs[i].step(np.int64(acts_np[i])),
                                    range(n_envs)))
            global_step += n_envs

            for i, (ob, reward, term, trunc, _) in enumerate(results):
                rewards[base + i] = reward
                dones[base + i] = term or trunc
                ep_reward[i] += reward
                if term or trunc:
                    won = envs[i].env.battle1.won
                    wins.append(1.0 if won else 0.0)
                    ep_rewards.append(ep_reward[i])
                    opp_idx = getattr(envs[i].env.agent2._team, "last_idx", None)
                    if opp_idx is not None and opp_idx < n_opp:
                        opp_wins[opp_idx].append(1.0 if won else 0.0)
                    ep_reward[i] = 0.0
                    obs[i], _ = envs[i].reset()
                else:
                    obs[i] = ob

        # Bootstrap per env (env j's slot in the last vector-step is
        # base + j = steps_per_env * n_envs - n_envs + j).
        with torch.no_grad():
            o = torch.as_tensor(np.stack([ob["observation"] for ob in obs]), device=device)
            m = torch.as_tensor(np.stack([ob["action_mask"] for ob in obs]),
                                device=device, dtype=torch.float32)
            _, _, _, last_vals = agent.get_action_and_value(o, m)

        # GAE per env over its contiguous strided slice, then scatter back.
        advantages = torch.zeros_like(rewards)
        for j in range(n_envs):
            sl = slice(j, ROLLOUT, n_envs)
            adv_j, _ = compute_gae(rewards[sl], values[sl], dones[sl],
                                   last_vals[j], GAMMA, LAM)
            advantages[sl] = adv_j
        returns = advantages + values

        stats = ppo_update(
            agent, optimizer, s, masks, actions, logprobs, advantages, returns,
            values, clip_coef=CLIP, vf_clip=VF_CLIP, entropy_coef=ENTROPY,
            epochs=EPOCHS, minibatches=MINIBATCHES,
        )
        dt = time.time() - t0
        writer.add_scalar("win_rate_20", np.mean(wins) if wins else 0.0, global_step)
        if ep_rewards:
            writer.add_scalar("episode_reward", np.mean(ep_rewards), global_step)
        for k, v in stats.items():
            writer.add_scalar(k, v, global_step)
        writer.add_scalar("explained_variance",
                          explained_variance(values, returns), global_step)
        writer.add_scalar("steps_per_sec", ROLLOUT / dt, global_step)
        for i, dq in enumerate(opp_wins):
            if dq:
                writer.add_scalar(f"win_opp{i:02d}", np.mean(dq), global_step)
        print(f"update {update:3d}/{total_updates} step {global_step} "
              f"win20={np.mean(wins) if wins else 0:.2f} "
              f"R={np.mean(ep_rewards) if ep_rewards else 0:+.1f} "
              f"pi={stats['policy_loss']:+.3f} v={stats['value_loss']:.3f} "
              f"H={stats['entropy']:.2f} kl={stats['approx_kl']:.4f} "
              f"{ROLLOUT / dt:.0f} it/s")

        torch.save({"model": agent.state_dict(), "arch": arch},
                   ckpt_dir / f"ppo_latest_{arch}.pt")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["mlp", "attention"], default="attention")
    p.add_argument("--team-mode", choices=["pool", "fixed"], default="pool")
    p.add_argument("--pool-size", type=int, default=4)
    p.add_argument("--envs", type=int, default=1,
                   help="parallel envs (needs a Showdown server per env on ports 8000..)")
    p.add_argument("--random", action="store_true", help="random-play sanity check")
    args = p.parse_args()
    if args.random:
        random_battles(make_env(team_mode=args.team_mode, arch=args.arch,
                                pool_size=args.pool_size), args.arch)
    else:
        train(args.updates, args.seed, args.arch, args.team_mode, args.pool_size,
              args.envs)
