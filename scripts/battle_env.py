"""Gymnasium battle env: poke-env SinglesEnv + SingleAgentWrapper.

Two obs modes:
- "mlp":       flat 106-dim vector (legacy, fixed-team baseline)
- "attention": (15, TOKEN_DIM) token matrix — permutation-robust, carries
               species identity + moves, for varied team pools.

Two team modes (set in make_env):
- "fixed": single team both sides (legacy mirror match)
- "pool":  both sides cycle independently through data/teams/pool/*.txt
          via CyclerTeambuilder — all N^2 pairings occur uniformly.

Action mask comes back automatically as obs["action_mask"] (22 actions).
"""

import glob
import logging
import os
import re

import numpy as np
from gymnasium.spaces import Box
from poke_env import ServerConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import Teambuilder
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

HOST, PORT = "localhost", 8020
BATTLE_FORMAT = "gen8customgame"


def server_config(port: int = PORT) -> ServerConfiguration:
    return ServerConfiguration(
        websocket_url=f"ws://{HOST}:{port}/showdown/websocket",
        authentication_url=f"http://{HOST}:{port}/action.php",
    )
N_ACTIONS = 22
N_TYPES = 18
STATUSES = ["brn", "par", "psn", "tox", "slp", "frz"]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
WEATHERS = ["raindance", "sundane", "sandstorm", "hail", "snow"]  # none = all-zero
OBS_DIM = 2 * 34 + 6 * 3 + 6 * 2 + 5 + 3  # 106

SERVER = server_config()

# --- team pool constants (computed once at import) ---
POOL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "teams", "pool")
POOL_FILES = sorted(glob.glob(os.path.join(POOL_DIR, "*.txt")))
N_TEAMS_AVAIL = len(POOL_FILES)


def _species_ids(team_text: str) -> set[str]:
    """Species ids from a showdown export: mon name lines start a block, i.e.
    the line after a blank line (or file start). Strip forme/gender/@ item."""
    ids = set()
    prev_blank = True
    for line in team_text.splitlines():
        stripped = line.strip()
        if not stripped:
            prev_blank = True
            continue
        if prev_blank and stripped:
            name = stripped.split("@")[0].strip()
            name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # drop trailing (M)/(F)
            ids.add("".join(name.split()).lower())
        prev_blank = False
    return ids


SPECIES = sorted(set().union(*(_species_ids(open(f).read()) for f in POOL_FILES)))
N_SPECIES = len(SPECIES)
SPECIES_IDX = {s: i for i, s in enumerate(SPECIES)}

# Token layout: [species N_SPECIES | hp 1 | fainted 1 | status 6 | types 18 |
#                is_ours 1 | is_active 1 | presence 1 | level 1 | boosts 7 |
#                moves 4x21 (our active only)]
TOKEN_DIM = N_SPECIES + 1 + 1 + 6 + 18 + 1 + 1 + 1 + 1 + 7 + 4 * 21
N_TOKENS = 15  # 0=our active, 1=opp active, 2-7 our bench, 8-13 opp bench, 14=global


def _mon_vec(mon) -> np.ndarray:
    """34 dims (mlp mode): hp_frac, status one-hot (6), boosts (7), types (18),
    level/100, fainted."""
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
            v[14 + mon.types[i].value] = 1.0  # .value: 1-18 type id
    v[32] = (mon.level or 100) / 100.0
    v[33] = 1.0 if mon.fainted else 0.0
    return v


def _fill_mon_token(tok: np.ndarray, mon, is_ours: bool, is_active: bool):
    """Fill the shared per-mon fields of a TOKEN_DIM token (moves/boosts only
    for actives; moves only for our active — caller handles via extra args)."""
    if mon is None:
        return
    spec = getattr(mon, "species", None)
    if spec in SPECIES_IDX:
        tok[SPECIES_IDX[spec]] = 1.0
    o = N_SPECIES
    tok[o] = mon.current_hp_fraction
    tok[o + 1] = 1.0 if mon.fainted else 0.0
    status = mon.status.name.lower() if mon.status is not None else None
    if status in STATUSES:
        tok[o + 2 + STATUSES.index(status)] = 1.0
    for i in range(2):
        if i < len(mon.types) and mon.types[i] is not None:
            tok[o + 8 + mon.types[i].value - 1] = 1.0
    tok[o + 26] = 1.0 if is_ours else 0.0
    tok[o + 27] = 1.0 if is_active else 0.0
    tok[o + 28] = 1.0  # presence
    tok[o + 29] = (mon.level or 100) / 100.0


def _move_block(tok: np.ndarray, base: int, moves):
    """4 slots x [type 18, bp/150, phys, spec, status] into tok[base:base+84]."""
    for j, move in enumerate(list(moves)[:4]):
        p = base + j * 21
        t = move.type.value  # 1-18
        tok[p + t - 1] = 1.0
        tok[p + 18] = min(move.base_power or 0, 150) / 150.0
        cat = move.category.name.lower()
        if "physical" in cat:
            tok[p + 19] = 1.0
        elif "special" in cat:
            tok[p + 20] = 1.0
        else:
            tok[p + 21 - 1] = 1.0  # status moves flag


def _boost_block(tok: np.ndarray, base: int, mon):
    for i, stat in enumerate(BOOSTS):
        tok[base + i] = np.clip(mon.boosts.get(stat, 0) / 6.0, -1.0, 1.0)


class CyclerTeambuilder(ConstantTeambuilder):
    """Yields pool teams round-robin; records last index for win logging."""

    def __init__(self, teams: list[str]):  # packed team strings
        self.teams = teams
        self.i = 0
        self.last_idx = 0
        super().__init__(teams[0])

    def yield_team(self) -> str:
        self.last_idx = self.i
        team = self.teams[self.i]
        self.i = (self.i + 1) % len(self.teams)
        return team


class BattleEnv(SinglesEnv):
    def __init__(self, team: str, obs_mode: str = "mlp", port: int = PORT, **kwargs):
        self.obs_mode = obs_mode
        super().__init__(
            battle_format=BATTLE_FORMAT,
            team=team,
            server_configuration=server_config(port),
            log_level=logging.WARNING,
            **kwargs,
        )
        shape = (OBS_DIM,) if obs_mode == "mlp" else (N_TOKENS, TOKEN_DIM)
        self.observation_spaces = {
            agent: Box(-np.inf, np.inf, shape, dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=1.0, hp_value=1.0, status_value=0.5, victory_value=10.0
        )

    def embed_battle(self, battle):
        if self.obs_mode == "attention":
            return self._embed_tokens(battle)
        return self._embed_flat(battle)

    def _embed_flat(self, battle) -> np.ndarray:
        v = np.zeros(OBS_DIM, dtype=np.float32)
        v[0:34] = _mon_vec(battle.active_pokemon)
        v[34:68] = _mon_vec(battle.opponent_active_pokemon)
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

    def _embed_tokens(self, battle) -> np.ndarray:
        toks = np.zeros((N_TOKENS, TOKEN_DIM), dtype=np.float32)
        o = N_SPECIES

        our_active = battle.active_pokemon
        _fill_mon_token(toks[0], our_active, is_ours=True, is_active=True)
        _boost_block(toks[0], o + 30, our_active) if our_active else None
        if our_active is not None:
            _move_block(toks[0], o + 37, our_active.moves.values())

        opp_active = battle.opponent_active_pokemon
        _fill_mon_token(toks[1], opp_active, is_ours=False, is_active=True)
        if opp_active is not None:
            _boost_block(toks[1], o + 30, opp_active)

        bench = [m for m in battle.team.values() if m is not our_active][:6]
        for i, mon in enumerate(bench):
            _fill_mon_token(toks[2 + i], mon, is_ours=True, is_active=False)
        opp_bench = [m for m in battle.opponent_team.values()
                     if m is not opp_active][:6]
        for i, mon in enumerate(opp_bench):
            _fill_mon_token(toks[8 + i], mon, is_ours=False, is_active=False)

        g = toks[14]
        for weather in battle.weather:
            name = weather.name.lower()
            if name in WEATHERS:
                g[WEATHERS.index(name)] = 1.0
        g[5] = min(battle.turn, 100) / 100.0
        g[6] = 1.0 if battle.trapped else 0.0
        g[7] = 1.0  # bias
        return toks


def make_env(team_mode: str = "pool", arch: str = "attention",
             pool_size: int | None = None, port: int = PORT) -> SingleAgentWrapper:
    """Build env + heuristic opponent. team_mode: "fixed" (mirror, needs
    data/teams/fixed.txt) or "pool" (cycle data/teams/pool/*.txt[:pool_size]).
    port selects the Showdown server — use distinct ports for parallel envs."""
    if team_mode == "fixed":
        with open(os.path.join(os.path.dirname(__file__), "..", "data",
                               "teams", "fixed.txt")) as f:
            team = f.read()
        env = BattleEnv(team=team, obs_mode=arch, port=port)
    else:
        files = POOL_FILES if pool_size is None else POOL_FILES[:pool_size]
        packed = [
            Teambuilder.join_team(Teambuilder.parse_showdown_team(open(f).read()))
            for f in files
        ]
        env = BattleEnv(team=packed[0], obs_mode=arch, port=port)
        env.agent1._team = CyclerTeambuilder(packed)  # our side
        env.agent2._team = CyclerTeambuilder(packed)  # opponent side
    opponent = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT,
        server_configuration=server_config(port),
        log_level=logging.WARNING,
    )
    return SingleAgentWrapper(env, opponent)
