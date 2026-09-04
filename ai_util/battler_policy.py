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
        mask = np.array(SinglesEnv.get_action_mask(battle))
        if len(mask) < N_ACTIONS:  # gen8 battles have a 22-dim action mask
            mask = np.pad(mask, (0, N_ACTIONS - len(mask)))
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict, deterministic=self.deterministic)
        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)
