"""Train the battler with masked PPO. Requires the local Showdown server on
port 8020 (see README).

Usage:
  uv run python scripts/train.py --random                       # pool-mode sanity
  uv run python scripts/train.py --arch mlp --team-mode fixed    # legacy baseline
  uv run python scripts/train.py --arch attention --updates 500  # pool A/B run
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from battle_env import N_ACTIONS, N_TEAMS_AVAIL, N_TOKENS, OBS_DIM, TOKEN_DIM, make_env
from ppo import AttentionPolicy, MaskedPolicy, compute_gae, explained_variance, ppo_update

# --- config (single place to tune) ---
ROLLOUT = 1024
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


def train(total_updates: int, seed: int, arch: str, team_mode: str, pool_size):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(team_mode=team_mode, arch=arch, pool_size=pool_size)
    agent = make_policy(arch).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    writer = SummaryWriter(f"runs/{datetime.now():%Y%m%d-%H%M%S}-{arch}-{team_mode}")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    shape = obs_shape(arch)
    wins = deque(maxlen=20)
    ep_rewards = deque(maxlen=20)
    n_opp = pool_size if (team_mode == "pool" and pool_size) else N_TEAMS_AVAIL
    opp_wins = [deque(maxlen=20) for _ in range(n_opp)]

    obs, _ = env.reset()
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
        episode_reward = 0.0
        for t in range(ROLLOUT):
            o = torch.as_tensor(obs["observation"], device=device)
            m = torch.as_tensor(obs["action_mask"], device=device, dtype=torch.float32)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(o, m)

            s[t], masks[t] = o, m
            actions[t], logprobs[t], values[t] = action, logprob, value
            obs, reward, term, trunc, _ = env.step(np.int64(action.item()))
            rewards[t] = reward
            dones[t] = term or trunc
            episode_reward += reward
            global_step += 1

            if term or trunc:
                won = env.env.battle1.won
                wins.append(1.0 if won else 0.0)
                ep_rewards.append(episode_reward)
                opp_idx = getattr(env.env.agent2._team, "last_idx", None)
                if opp_idx is not None and opp_idx < n_opp:
                    opp_wins[opp_idx].append(1.0 if won else 0.0)
                episode_reward = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            o = torch.as_tensor(obs["observation"], device=device)
            m = torch.as_tensor(obs["action_mask"], device=device, dtype=torch.float32)
            _, _, _, last_value = agent.get_action_and_value(o, m)
        advantages, returns = compute_gae(rewards, values, dones, last_value, GAMMA, LAM)

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
    p.add_argument("--random", action="store_true", help="random-play sanity check")
    args = p.parse_args()
    if args.random:
        random_battles(make_env(team_mode=args.team_mode, arch=args.arch,
                                pool_size=args.pool_size), args.arch)
    else:
        train(args.updates, args.seed, args.arch, args.team_mode, args.pool_size)
