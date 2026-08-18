"""Masked PPO (CleanRL-style, single file): MLP policy, GAE, clipped update.

Hyperparameter defaults encode lessons from the previous project:
grad_clip 5.0, gamma 0.95, entropy 0.05, vf_clip 5.0.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from battle_env import N_TOKENS, TOKEN_DIM


class MaskedPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        """Returns masked logits and state value."""
        h = self.body(obs)
        logits = self.actor(h).masked_fill(mask == 0, -1e9)
        return logits, self.critic(h).squeeze(-1)

    def get_action_and_value(self, obs, mask):
        logits, value = self(obs, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs, mask, actions):
        logits, value = self(obs, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class AttentionPolicy(nn.Module):
    """Token-based policy for varied teams: project tokens, one transformer
    layer, mean-pool, actor/critic heads. No src_mask anywhere (PyTorch
    TransformerEncoder NaN risk with float masks — see CLAUDE.md)."""

    def __init__(self, n_actions: int, d_model: int = 128, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(TOKEN_DIM, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=2 * d_model,
                batch_first=True, norm_first=True, dropout=0.0,
            ),
            num_layers=1, enable_nested_tensor=False,
        )
        self.actor = nn.Linear(d_model, n_actions)
        self.critic = nn.Linear(d_model, 1)

    def forward(self, obs, mask):
        if obs.dim() == 2:  # single step -> batch of 1
            obs = obs.unsqueeze(0)
        h = self.encoder(self.proj(obs))
        pooled = h.mean(dim=1)
        logits = self.actor(pooled).masked_fill(mask == 0, -1e9)
        if torch.isnan(logits).any():
            print("WARN: NaN logits -> nan_to_num applied")
            logits = torch.nan_to_num(logits, nan=-1e9)
        return logits, self.critic(pooled).squeeze(-1)

    def get_action_and_value(self, obs, mask):
        logits, value = self(obs, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs, mask, actions):
        logits, value = self(obs, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages via GAE; returns = advantages + values."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def ppo_update(agent, optimizer, b_obs, b_masks, b_actions, b_logprobs,
               b_advantages, b_returns, b_values, *, clip_coef=0.2, vf_clip=5.0,
               entropy_coef=0.05, vf_coef=0.5, epochs=4, minibatches=4,
               max_grad_norm=5.0):
    """One PPO update over the rollout buffer. Returns metric dict."""
    # Normalize advantages over the whole batch.
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    n = b_obs.shape[0]
    idx = np.arange(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    steps = 0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, n // minibatches):
            mb = idx[start:start + n // minibatches]
            new_logprob, entropy, new_value = agent.evaluate_actions(
                b_obs[mb], b_masks[mb], b_actions[mb]
            )
            logratio = new_logprob - b_logprobs[mb]
            ratio = torch.exp(logratio)

            # Clipped surrogate policy loss.
            pg_loss = -torch.min(
                ratio * b_advantages[mb],
                torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * b_advantages[mb],
            ).mean()

            # Clipped value loss.
            v_loss = 0.5 * torch.max(
                (new_value - b_returns[mb]) ** 2,
                (torch.clamp(new_value, b_values[mb] - vf_clip,
                             b_values[mb] + vf_clip) - b_returns[mb]) ** 2,
            ).mean()

            loss = pg_loss + vf_coef * v_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += pg_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += (ratio - 1 - logratio).mean().item()
            steps += 1
    return {k: v / steps for k, v in stats.items()}


def explained_variance(values, returns):
    """1.0 = perfect value prediction, 0 = no better than mean."""
    var_returns = torch.var(returns)
    return (1 - torch.var(returns - values) / (var_returns + 1e-8)).item()
