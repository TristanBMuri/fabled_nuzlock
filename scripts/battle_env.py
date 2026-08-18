"""Gymnasium battle env: poke-env SinglesEnv + SingleAgentWrapper.

One battle vs SimpleHeuristicsPlayer, fixed team both sides (poke-env gives
both internal players the same team string — accepted mirror match for v1).
Observation is a flat 106-dim vector; action mask comes back automatically
as obs["action_mask"] (22 actions: 0-5 switch, 6-9 move, 10+ gimmicks).
"""

import logging

import numpy as np
from gymnasium.spaces import Box
from poke_env import ServerConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer

HOST, PORT = "localhost", 8020
BATTLE_FORMAT = "gen8customgame"
N_ACTIONS = 22
N_TYPES = 18
STATUSES = ["brn", "par", "psn", "tox", "slp", "frz"]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
WEATHERS = ["raindance", "sundane", "sandstorm", "hail", "snow"]  # none = all-zero
OBS_DIM = 2 * 34 + 6 * 3 + 6 * 2 + 5 + 3  # 106

SERVER = ServerConfiguration(
    websocket_url=f"ws://{HOST}:{PORT}/showdown/websocket",
    authentication_url=f"http://{HOST}:{PORT}/action.php",
)


def _mon_vec(mon) -> np.ndarray:
    """34 dims: hp_frac, status one-hot (6), boosts (7), types (18), level/100."""
    v = np.zeros(34, dtype=np.float32)
    if mon is None:
        return v
    v[0] = mon.current_hp_fraction
    status = mon.status.name.lower() if mon.status is not None else None
    if status in STATUSES:
        v[1 + STATUSES.index(status)] = 1.0
    for i, stat in enumerate(BOOSTS):
        v[7 + i] = np.clip(mon.boosts.get(stat, 0) / 6.0, -1.0, 1.0)
    for i in range(2):
        if i < len(mon.types) and mon.types[i] is not None:
            v[13 + mon.types[i].value] = 1.0  # .value: 1-18 type id
    v[32] = (mon.level or 100) / 100.0
    v[33] = 1.0 if mon.fainted else 0.0
    return v


class BattleEnv(SinglesEnv):
    def __init__(self, team: str, **kwargs):
        super().__init__(
            battle_format=BATTLE_FORMAT,
            team=team,
            server_configuration=SERVER,
            log_level=logging.WARNING,
            **kwargs,
        )
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=1.0, hp_value=1.0, status_value=0.5, victory_value=10.0
        )

    def embed_battle(self, battle) -> np.ndarray:
        v = np.zeros(OBS_DIM, dtype=np.float32)
        v[0:34] = _mon_vec(battle.active_pokemon)
        v[34:68] = _mon_vec(battle.opponent_active_pokemon)

        # Benches in team-slot order. Fixed team => identity is constant, so
        # only condition features are needed. Unrevealed opponent mons are zeros.
        for i, mon in enumerate(battle.team.values()):
            if i >= 6:
                break
            base = 68 + i * 3
            v[base] = mon.current_hp_fraction
            v[base + 1] = 1.0 if mon.fainted else 0.0
            v[base + 2] = 1.0 if mon.status is not None else 0.0
        for i, mon in enumerate(battle.opponent_team.values()):
            if i >= 6:
                break
            base = 86 + i * 2
            v[base] = mon.current_hp_fraction
            v[base + 1] = 1.0 if mon.fainted else 0.0

        for weather in battle.weather:
            name = weather.name.lower()
            if name in WEATHERS:
                v[98 + WEATHERS.index(name)] = 1.0

        v[103] = min(battle.turn, 100) / 100.0
        v[104] = 1.0 if battle.trapped else 0.0
        v[105] = 1.0  # bias
        return v


def make_env(team_path: str = "data/teams/fixed.txt") -> SingleAgentWrapper:
    with open(team_path) as f:
        team = f.read()
    env = BattleEnv(team=team)
    opponent = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT,
        server_configuration=SERVER,
        log_level=logging.WARNING,
    )
    return SingleAgentWrapper(env, opponent)
