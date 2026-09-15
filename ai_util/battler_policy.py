"""Masked policy network and the player that runs it for eval."""

from typing import Any, Awaitable

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    Player,
)

from ai_util.battler_obs import N_FEATURES, embed_battle

N_ACTIONS = SinglesEnv.get_action_space_size(9)
_SWITCH_ACTIONS = 6


def action_mask(battle: AbstractBattle) -> np.ndarray:
    """poke-env's mask with the switch slots decided by identity. poke-env marks
    a slot legal when its *species* is among available switches, so with two of
    a species (common in gauntlet parties) the active one is marked too and the
    resulting order is rejected. Padded to N_ACTIONS (gen8 masks are 22-dim)."""
    mask = np.array(SinglesEnv.get_action_mask(battle))
    if len(mask) < N_ACTIONS:
        mask = np.pad(mask, (0, N_ACTIONS - len(mask)))
    if not battle.wait:
        for i, p in enumerate(list(battle.team.values())[:_SWITCH_ACTIONS]):
            mask[i] = int(not battle.trapped and any(p is s for s in battle.available_switches))
    return mask


def order_to_action(order: BattleOrder, battle: AbstractBattle) -> int:
    """Inverse of SinglesEnv.action_to_order with switches resolved by identity
    (poke-env maps a switch to the first slot of that species)."""
    from poke_env.battle import Pokemon
    from poke_env.player import SingleBattleOrder
    if isinstance(order, SingleBattleOrder) and isinstance(order.order, Pokemon):
        return next(i for i, p in enumerate(battle.team.values()) if p is order.order)
    return int(SinglesEnv.order_to_action(order, battle, fake=True, strict=False))


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, **kwargs: Any):
        # default kept at [256, 256]: existing checkpoints (state-dict
        # shapes) depend on it; pass net_arch explicitly to go bigger
        net_arch = kwargs.pop("net_arch", [256, 256])
        super().__init__(
            *args,
            **kwargs,
            net_arch=net_arch,
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


def load_policy(path: str, net_arch=(256, 256), device: str | None = None) -> MaskedActorCriticPolicy:
    """Rebuild the masked policy from a saved state_dict (BC or PPO checkpoint)."""
    from gymnasium.spaces import Box, Dict, Discrete

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    obs_space = Dict({
        "observation": Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int64),
    })
    policy = MaskedActorCriticPolicy(obs_space, Discrete(N_ACTIONS), lambda _: 3e-4,
                                     net_arch=list(net_arch)).to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy


class PolicyPlayer(Player):
    """Plays embed_battle/action_mask through a policy; used for eval."""

    def __init__(
        self,
        policy: ActorCriticPolicy | None = None,
        deterministic: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.deterministic = deterministic

    def choose_move(self, battle: AbstractBattle) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()
        obs = embed_battle(battle)
        mask = action_mask(battle)
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict, deterministic=self.deterministic)
        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)
