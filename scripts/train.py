"""Train the fixed-team battler with masked PPO. Requires the local Showdown
server on port 8020 (see README).

Usage:
  uv run python scripts/train.py --random           # env sanity: 3 random battles
  uv run python scripts/train.py --updates 8        # mini run
  uv run python scripts/train.py                    # full smoke run (60 updates)
"""

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from battle_env import N_ACTIONS, OBS_DIM, make_env
from ppo import MaskedPolicy, compute_gae, explained_variance, ppo_update

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


def random_battles(env, n=3):
    """Env sanity check: play n battles with uniformly random valid actions."""
    for i in range(n):
        obs, _ = env.reset()
        steps = 0
        while True:
            assert obs["observation"].shape == (OBS_DIM,), obs["observation"].shape
            valid = np.flatnonzero(obs["action_mask"])
            assert len(valid) > 0
            obs, reward, term, trunc, _ = env.step(np.int64(np.random.choice(valid)))
            steps += 1
            if term or trunc:
                break
        won = env.env.battle1.won
        print(f"battle {i}: turns={env.env.battle1.turn} steps={steps} won={won}")


def train(total_updates: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env()
    agent = MaskedPolicy(OBS_DIM, N_ACTIONS).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    writer = SummaryWriter(f"runs/{datetime.now():%Y%m%d-%H%M%S}")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    wins = deque(maxlen=20)
    episode_rewards = deque(maxlen=20)

    obs, _ = env.reset()
    s = torch.zeros(ROLLOUT, OBS_DIM, device=device)
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
                episode_rewards.append(episode_reward)
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
        if episode_rewards:
            writer.add_scalar("episode_reward", np.mean(episode_rewards), global_step)
        for k, v in stats.items():
            writer.add_scalar(k, v, global_step)
        writer.add_scalar("explained_variance",
                          explained_variance(values, returns), global_step)
        writer.add_scalar("steps_per_sec", ROLLOUT / dt, global_step)
        print(f"update {update:3d}/{total_updates} step {global_step} "
              f"win20={np.mean(wins) if wins else 0:.2f} "
              f"R={np.mean(episode_rewards) if episode_rewards else 0:+.1f} "
              f"pi={stats['policy_loss']:+.3f} v={stats['value_loss']:.3f} "
              f"H={stats['entropy']:.2f} kl={stats['approx_kl']:.4f} "
              f"{ROLLOUT / dt:.0f} it/s")

        torch.save({"model": agent.state_dict()}, ckpt_dir / "ppo_latest.pt")
    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random", action="store_true", help="random-play sanity check")
    args = p.parse_args()
    if args.random:
        random_battles(make_env())
    else:
        train(args.updates, args.seed)
