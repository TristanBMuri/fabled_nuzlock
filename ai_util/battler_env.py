"""Battle environment stack: poke-env singles battle as a single-agent gym env.

RotatingOpponentWrapper picks a fresh fixed opponent per episode
(SimpleHeuristics 50% / MaxBasePower 25% / Random 25%) — training optimizes
the target matchup directly instead of self-play.
"""

import socket
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3.common.monitor import Monitor

from poke_env import ServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from ai_util.battler_obs import BATTLE_FORMAT, N_FEATURES, embed_battle
from ai_util.battler_policy import N_ACTIONS
from ai_util.poke_env.lazy_poke_env import LazyPokeEnv


class BattlerEnv(SinglesEnv):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle: AbstractBattle) -> float:
        # Shaping must stay small relative to the DISCOUNTED win: battles run
        # ~40 turns, so the win bonus is worth victory_value * gamma^40 at the
        # first move (gamma 0.95 made it ~3.9 — the agent converged to farming
        # dense HP shaping and never learned to close games; see README).
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=0.25,
            status_value=0.1,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return embed_battle(battle)


class SafeSingleAgentWrapper(SingleAgentWrapper):
    """Heuristic opponents can pick a move poke-env cannot map back to an action
    (available_moves / known-moves desync, ~1 per 250k steps): degrade that
    opponent order to a default order instead of killing the worker. The RL
    side keeps strict conversion."""

    def step(self, action):
        try:
            return super().step(action)
        except ValueError:
            obs, rewards, terms, truncs, infos = self.env.step(
                {self.env.agent1.username: action, self.env.agent2.username: np.int64(-2)}
            )
            i = self.env.agent1.username
            return obs[i], rewards[i], terms[i], truncs[i], infos[i]


class RotatingOpponentWrapper(Env):
    """Picks a fresh fixed opponent for every episode.

    Opponents run with start_listening=False: SingleAgentWrapper drives them
    locally, so they never touch the server.
    """

    OPPONENTS = ((SimpleHeuristicsPlayer, 0.5), (MaxBasePowerPlayer, 0.25), (RandomPlayer, 0.25))

    def __init__(self, env: SingleAgentWrapper, server: ServerConfiguration):
        self.env = env
        self._server = server
        self._classes, self._probs = zip(*self.OPPONENTS)
        self._pool = [
            cls(battle_format=BATTLE_FORMAT, server_configuration=server,
                start_listening=False, log_level=40)
            for cls in self._classes
        ]
        self._rng = np.random.default_rng()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        idx = self._rng.choice(len(self._pool), p=self._probs)
        self.env.opponent = self._pool[int(idx)]
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def inner_env(server: ServerConfiguration) -> Monitor:
    """One poke-env battle as a single-agent gym env (shared by both trainers)."""
    env = BattlerEnv(
        battle_format=BATTLE_FORMAT,
        log_level=40,
        server_configuration=server,
    )
    single = SafeSingleAgentWrapper(
        env,
        opponent=SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT, server_configuration=server,
            start_listening=False, log_level=40,
        ),
    )
    return Monitor(RotatingOpponentWrapper(single, server))


def make_env(port: int) -> LazyPokeEnv:
    # Module-level for spawn. Do not open Showdown here — LazyPokeEnv defers that.
    return LazyPokeEnv(
        env_factory=lambda: inner_env(server_conf(port)),
        observation_space=Dict(
            {
                "observation": Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int64),
            }
        ),
        action_space=Discrete(N_ACTIONS),
    )


def server_conf(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        f"http://localhost:{port}/action.php",
    )


def preflight(port: int) -> ServerConfiguration:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
    except OSError:
        print(
            f"No Showdown server on port {port}. Start it first:\n"
            f"  cd pokemon-showdown && node pokemon-showdown start --no-security --port {port}"
        )
        import sys
        sys.exit(1)
    return server_conf(port)
