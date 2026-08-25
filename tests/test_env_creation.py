"""Integration tests for LazyPokeEnv + ExampleEnv / SubprocVecEnv creation."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3.common.vec_env import SubprocVecEnv

from ai_util.poke_env.lazy_poke_env import LazyPokeEnv
from policy_player import (
    N_ACTIONS,
    N_FEATURES,
    SERVER,
    ExampleEnv,
    make_env,
)


def test_lazy_env_defers_factory_until_reset():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1

        class _Dummy:
            observation_space = Box(-1, 1, shape=(2,), dtype=np.float32)
            action_space = Discrete(2)

            def reset(self, *, seed=None, options=None):
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

            def close(self):
                pass

        return _Dummy()

    env = LazyPokeEnv(
        env_factory=factory,
        observation_space=Box(-1, 1, shape=(2,), dtype=np.float32),
        action_space=Discrete(2),
    )
    assert calls["n"] == 0
    env.reset()
    assert calls["n"] == 1
    env.step(0)
    assert calls["n"] == 1
    env.close()


def test_make_env_exposes_expected_spaces():
    env = make_env()
    assert isinstance(env.observation_space, Dict)
    assert env.observation_space["observation"].shape == (N_FEATURES,)
    assert env.observation_space["action_mask"].shape == (N_ACTIONS,)
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == N_ACTIONS
    # still deferred
    assert env._env is None
    env.close()


def test_example_env_reset_and_step(showdown_8000):
    assert SERVER.websocket_url.startswith("ws://localhost:8000")
    env = ExampleEnv.create_env(SERVER)
    obs, info = env.reset()
    assert "observation" in obs and "action_mask" in obs
    assert obs["observation"].shape == (N_FEATURES,)
    assert obs["action_mask"].shape == (N_ACTIONS,)

    legal = np.flatnonzero(obs["action_mask"] == 1)
    action = int(legal[0])
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2["observation"].shape == (N_FEATURES,)
    assert isinstance(reward, float)
    env.close()


def test_subproc_make_env_reset(showdown_8000):
    env = SubprocVecEnv([make_env, make_env], start_method="spawn")
    try:
        obs = env.reset()
        assert obs["observation"].shape == (2, N_FEATURES)
        assert obs["action_mask"].shape == (2, N_ACTIONS)
    finally:
        env.close()
