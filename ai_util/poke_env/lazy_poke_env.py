"""Lazy gym wrapper so SubprocVecEnv can handshake spaces without opening Showdown yet.

Connects on first reset/step. Use with start_method="spawn".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium.spaces import Space


class LazyPokeEnv(gym.Env):
    """Defer env_factory() until first reset/step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        observation_space: Space[Any],
        action_space: Space[Any],
    ):
        self._env_factory = env_factory
        self.observation_space = observation_space
        self.action_space = action_space
        self._env: gym.Env | None = None

    def _ensure(self) -> gym.Env:
        if self._env is None:
            self._env = self._env_factory()
        return self._env

    def reset(self, *, seed=None, options=None):
        return self._ensure().reset(seed=seed, options=options)

    def step(self, action):
        return self._ensure().step(action)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self):
        if self._env is not None:
            return self._env.render()
