"""Self-play PPO copy of no_AI_code_here/players/policy/self_play.py with eval/length logs."""

import sys
from pathlib import Path

# `python ai_util/self_play.py` puts this directory on sys.path, so `import poke_env`
# would resolve to the local ai_util/poke_env/ wrapper instead of the installed package.
_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]

import asyncio
import datetime
from typing import Any, Awaitable

import numpy as np
import supersuit as ss
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor

from poke_env.battle import AbstractBattle
from poke_env.data import GenData
from poke_env.environment import SinglesEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

BATTLE_FORMAT = "gen9randombattle"
N_FEATURES = 12
# SinglesEnv: 0-5 switch, 6-9 move (then gimmick repeats of the 4 moves).
_SWITCH_ACTIONS = 6


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            # next: don't widen until eval/win_vs_heuristics moves; obs is 12-d
            net_arch=[64, 64],
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
    """Extracts the observation tensor from the dict obs and declares features_dim
    so SB3 builds the MLP with the right input size."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class PolicyPlayer(Player):
    policy: ActorCriticPolicy | None

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

    def choose_move(
        self, battle: AbstractBattle
    ) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()
        obs = self.embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(
                    obs, device=self.policy.device
                ).unsqueeze(0),
                "action_mask": torch.as_tensor(
                    mask, device=self.policy.device
                ).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(
                obs_dict, deterministic=self.deterministic
            )
        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle):
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(battle.gen).type_chart,
                )
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        our_hp = (
            battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        )
        opp_hp = (
            battle.opponent_active_pokemon.current_hp_fraction
            if battle.opponent_active_pokemon
            else 0.0
        )
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                [our_hp, opp_hp],
            ],
            dtype=np.float32,
        )


class SelfPlayEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            # next: leave shaping until win_vs_heuristics is the tuning target
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            # next: keep, or lower if the critic still explodes after gamma=0.95
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        return PolicyPlayer.embed_battle(battle)


def _win_rate(player: Player) -> float:
    n = player.n_finished_battles
    return 0.0 if n == 0 else player.n_lost_battles / n


def _obs_features(obs) -> np.ndarray:
    if isinstance(obs, dict):
        return np.asarray(obs["observation"])
    return np.asarray(obs)


class SelfPlayLogCallback(BaseCallback):
    """Periodic frozen-bot eval + rollout action mix. Checkpoints at each eval."""

    def __init__(
        self,
        save_dir: Path,
        eval_every: int,
        n_battles: int,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.eval_every = eval_every
        self.n_battles = n_battles
        self._last_eval_idx = -1
        self._agent: PolicyPlayer | None = None
        self._opponents: list[Player] = []
        self._n_act = 0
        self._n_switch = 0
        self._n_move = 0
        self._mult_sum = 0.0

    def _ensure_eval_players(self) -> None:
        if self._agent is not None:
            return
        kw = dict(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=10,
            log_level=40,
        )
        self._agent = PolicyPlayer(policy=self.model.policy, deterministic=True, **kw)
        self._opponents = [
            MaxBasePowerPlayer(**kw),
            SimpleHeuristicsPlayer(**kw),
        ]

    def _log_actions(self) -> None:
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        feats = _obs_features(self.model._last_obs)
        if feats.ndim == 1:
            feats = feats[None, :]
        n = min(len(actions), len(feats))
        actions = actions[:n]
        feats = feats[:n]
        is_switch = (actions >= 0) & (actions < _SWITCH_ACTIONS)
        is_move = actions >= _SWITCH_ACTIONS
        self._n_act += n
        self._n_switch += int(is_switch.sum())
        self._n_move += int(is_move.sum())
        if is_move.any():
            slots = (actions[is_move] - _SWITCH_ACTIONS) % 4
            self._mult_sum += float(feats[is_move, 4 + slots].sum())

    def _flush_action_stats(self) -> None:
        if self._n_act:
            self.logger.record("rollout/switch_rate", self._n_switch / self._n_act)
        if self._n_move:
            self.logger.record(
                "rollout/move_dmg_mult", self._mult_sum / self._n_move
            )
        self._n_act = self._n_switch = self._n_move = 0
        self._mult_sum = 0.0

    def _eval(self) -> None:
        self._ensure_eval_players()
        assert self._agent is not None
        self._agent.reset_battles()
        for opp in self._opponents:
            opp.reset_battles()
        asyncio.run(
            self._agent.battle_against(*self._opponents, n_battles=self.n_battles)
        )
        rates = {}
        for opp in self._opponents:
            wr = _win_rate(opp)
            tag = (
                "eval/win_vs_heuristics"
                if isinstance(opp, SimpleHeuristicsPlayer)
                else "eval/win_vs_maxbp"
            )
            rates[tag] = wr
            self.logger.record(tag, wr)
            if self.verbose:
                print(
                    f"eval step={self.num_timesteps} {opp.username}: {wr:.0%} "
                    f"({opp.n_finished_battles} battles)"
                )
        self.logger.dump(self.num_timesteps)
        path = self.save_dir / f"step_{self.num_timesteps}"
        self.model.save(path)
        if self.verbose:
            print(f"saved {path}")

    async def _stop_eval_players(self) -> None:
        players = [p for p in (self._agent, *self._opponents) if p is not None]
        for p in players:
            await p.ps_client.stop_listening()

    def _on_training_start(self) -> None:
        self._eval()
        self._last_eval_idx = 0

    def _on_step(self) -> bool:
        self._log_actions()
        idx = self.num_timesteps // self.eval_every
        if idx > self._last_eval_idx:
            self._last_eval_idx = idx
            self._eval()
        return True

    def _on_rollout_end(self) -> None:
        self._flush_action_stats()

    def _on_training_end(self) -> None:
        if self._agent is not None:
            asyncio.run(self._stop_eval_players())


def train():
    # --- run / scale ---
    # next: only raise if Showdown/RAM headroom; 12 was fine, fps drop was not env-count
    num_envs = 24
    # 2M: long enough to see if entropy holds past the old ~332k collapse
    total_timesteps = 4_000_000
    # next: keep 1M until you know eval cost; tighten only if the curve is too sparse
    eval_every = 500_000
    # next: 30 is enough to see 33% vs 50%; raise toward 50–100 for a publishable number
    n_eval_battles = 30

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"models/self_play_{stamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = SelfPlayEnv(battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env,
        num_vec_envs=num_envs,
        num_cpus=num_envs,
        base_class="stable_baselines3",
    )
    # ep_len_mean is the useful series; ep_rew_mean is ~0 in self-play — ignore it
    vec_env = VecMonitor(vec_env)

    ppo = PPO(
        MaskedActorCriticPolicy,
        vec_env,
        # next: keep until entropy is stable; don't lower while still exploring
        learning_rate=3e-4,
        # 128. next: 256–512 if KL stays tiny after entropy is fixed
        n_steps=3072 // (2 * num_envs),
        # next: keep; if n_steps grows, 128–256 is fine
        batch_size=128,
        # 0.95: battles are ~25 turns; 0.99 left value loss stuck ~270
        gamma=0.95,
        # 0.05: last two runs H died by ~332k with 0.01
        ent_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        tensorboard_log=f"logs/tensorboard/self_play_{stamp}",
    )

    callback = SelfPlayLogCallback(save_dir, eval_every, n_eval_battles)
    ppo.learn(total_timesteps, callback=callback)
    ppo.save(save_dir / "final")
    vec_env.close()

    # final eval: keep Random here so the number is comparable to yesterday
    agent = PolicyPlayer(
        policy=ppo.policy,
        deterministic=True,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
        log_level=40,
    )
    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=10, log_level=40)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))
    print("--- Win rates vs bots ---")
    for opp in opponents:
        print(f"{opp.username}: {round(100 * _win_rate(opp))}%")


if __name__ == "__main__":
    train()
