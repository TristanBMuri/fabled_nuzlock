"""Evaluation and diagnostics: win-rate eval vs baselines, behavior stats,
obs dump, and the training-time EvalCallback."""

import asyncio
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from poke_env import ServerConfiguration
from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.battle.move_category import MoveCategory
from poke_env.environment import SinglesEnv
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from ai_util.battler_env import BattlerEnv, preflight
from ai_util.battler_obs import (
    BATTLE_FORMAT,
    N_FEATURES,
    _FIELD,
    _PER_MON,
    _damage_multiplier,
)
from ai_util.battler_policy import _SWITCH_ACTIONS, PolicyPlayer


def _win_rate(player: Player) -> float:
    n = player.n_finished_battles
    return 0.0 if n == 0 else player.n_lost_battles / n


class EvalCallback(BaseCallback):
    """Periodic frozen-bot eval vs the two baselines; checkpoints at each eval."""

    def __init__(self, save_dir: Path, server: ServerConfiguration, eval_every: int, n_battles: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_dir = save_dir
        self.server = server
        self.eval_every = eval_every
        self.n_battles = n_battles
        self._last_eval_idx = -1
        self._agent: PolicyPlayer | None = None
        self._opponents: list[Player] = []
        self._n_act = 0
        self._n_switch = 0

    def _ensure_players(self) -> None:
        if self._agent is not None:
            return
        kw = dict(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=10,
            log_level=40,
            server_configuration=self.server,
        )
        self._agent = PolicyPlayer(policy=self.model.policy, deterministic=True, **kw)
        self._opponents = [MaxBasePowerPlayer(**kw), SimpleHeuristicsPlayer(**kw)]

    def _eval(self) -> None:
        self._ensure_players()
        assert self._agent is not None
        self._agent.reset_battles()
        for opp in self._opponents:
            opp.reset_battles()
        asyncio.run(self._agent.battle_against(*self._opponents, n_battles=self.n_battles))
        for opp in self._opponents:
            wr = _win_rate(opp)
            tag = "eval/win_vs_heuristics" if isinstance(opp, SimpleHeuristicsPlayer) else "eval/win_vs_maxbp"
            self.logger.record(tag, wr)
            if self.verbose:
                print(f"eval step={self.num_timesteps} {opp.username}: {wr:.0%} ({opp.n_finished_battles} battles)")
        self.logger.dump(self.num_timesteps)
        self.model.save(self.save_dir / f"step_{self.num_timesteps}")

    def _on_training_start(self) -> None:
        self._eval()
        self._last_eval_idx = 0

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        self._n_act += len(actions)
        self._n_switch += int(((actions >= 0) & (actions < _SWITCH_ACTIONS)).sum())
        idx = self.num_timesteps // self.eval_every
        if idx > self._last_eval_idx:
            self._last_eval_idx = idx
            self._eval()
        return True

    def _on_rollout_end(self) -> None:
        if self._n_act:
            self.logger.record("rollout/switch_rate", self._n_switch / self._n_act)
        self._n_act = self._n_switch = 0

    def _on_training_end(self) -> None:
        if self._agent is not None:
            async def _stop() -> None:
                for p in [self._agent, *self._opponents]:
                    await p.ps_client.stop_listening()
            asyncio.run(_stop())


def _decode(battle: AbstractBattle) -> str:
    def mon_side(m: Pokemon | None) -> str:
        if m is None:
            return "??"
        return f"{m.species} hp={m.current_hp_fraction:.2f} status={m.status}"

    bench = ", ".join(f"{m.species}:{m.current_hp_fraction:.1f}" for m in battle.team.values())
    opp = ", ".join(f"{m.species}:{m.current_hp_fraction:.1f}" for m in battle.opponent_team.values())
    return f"turn {battle.turn}\n  us  : {mon_side(battle.active_pokemon)} | bench {bench}\n  them: {mon_side(battle.opponent_active_pokemon)} | known {opp}"


def eval_checkpoint(path: str, port: int, n_battles: int, deterministic: bool) -> None:
    """Play a saved checkpoint vs the baselines and print behavior stats:
    switch rate and how often the chosen move is the max-effectiveness one."""
    server = preflight(port)
    model = PPO.load(path, device="cuda" if torch.cuda.is_available() else "cpu")
    _eval_policy(model.policy, server, n_battles, deterministic=deterministic)


def _eval_policy(policy: ActorCriticPolicy, server: ServerConfiguration, n_battles: int,
                 deterministic: bool = True) -> None:
    agent = PolicyPlayer(policy, deterministic=deterministic,
                         battle_format=BATTLE_FORMAT, max_concurrent_battles=10,
                         log_level=40, server_configuration=server)
    original_choose_move = agent.choose_move

    stats = {"n": 0, "switch": 0, "move": 0, "best_mult": 0}

    def counting_choose_move(battle):
        order = original_choose_move(battle)
        if order and not battle.wait and getattr(order, "order", None) is not None:
            act = SinglesEnv.order_to_action(order, battle, fake=True, strict=False)
            stats["n"] += 1
            if act < _SWITCH_ACTIONS:
                stats["switch"] += 1
            elif isinstance(order.order, Move):
                stats["move"] += 1
                mults = []
                opp = battle.opponent_active_pokemon
                for mv in battle.available_moves:
                    if mv.category != MoveCategory.STATUS and opp is not None:
                        mults.append(_damage_multiplier(mv, (opp.type_1, opp.type_2)))
                if mults and _damage_multiplier(order.order, (opp.type_1, opp.type_2)) >= max(mults) - 1e-9:
                    stats["best_mult"] += 1
        return order

    agent.choose_move = counting_choose_move
    opponents = [
        cls(battle_format=BATTLE_FORMAT, max_concurrent_battles=10, log_level=40,
            server_configuration=server)
        for cls in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=n_battles))
    print(f"--- eval (deterministic={deterministic}) ---")
    for opp in opponents:
        print(f"{opp.username}: {round(100 * _win_rate(opp))}% (n={opp.n_finished_battles})")
    if stats["n"]:
        print(f"actions: {stats['n']}, switch_rate={stats['switch'] / stats['n']:.0%}, "
              f"best_effectiveness_move={stats['best_mult'] / max(stats['move'], 1):.0%} of moves")


def obs_dump(port: int, seed: int) -> None:
    """One live battle on random actions; prints obs stats + a readable decode."""
    from poke_env.environment import SingleAgentWrapper
    from poke_env.player import SimpleHeuristicsPlayer as _SHP

    server = preflight(port)
    single = SingleAgentWrapper(
        BattlerEnv(battle_format=BATTLE_FORMAT, log_level=40, server_configuration=server),
        opponent=_SHP(
            battle_format=BATTLE_FORMAT, server_configuration=server,
            start_listening=False, log_level=40,
        ),
    )
    rng = np.random.default_rng(seed)
    obs, _ = single.reset(seed=seed)
    battle = single.env.battle1
    assert battle is not None
    pending, done = [1, 8], False
    while (pending or not done) and battle.turn < 100:
        if pending and battle.turn >= pending[0]:
            vec = obs["observation"]
            print(f"--- obs at turn {battle.turn}: shape={vec.shape} "
                  f"min={vec.min():.2f} max={vec.max():.2f} nan={np.isnan(vec).sum()} "
                  f"dim={N_FEATURES} (per-mon={_PER_MON}, field={_FIELD}) "
                  f"active_slots={[i for i in range(12) if vec[i * _PER_MON + 2] > 0]}")
            print(_decode(battle))
            pending.pop(0)
        action = np.int64(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, rew, term, trunc, _ = single.step(action)
        done = term or trunc
        if done:
            print(f"battle finished at turn {battle.turn}, reward={rew:+.1f}")
    single.close()


def _final_eval(policy: ActorCriticPolicy, server: ServerConfiguration, n_battles: int) -> None:
    """Random included so the numbers are comparable to the baseline run."""
    agent = PolicyPlayer(policy=policy, deterministic=True, **{
        "battle_format": BATTLE_FORMAT, "max_concurrent_battles": 10,
        "log_level": 40, "server_configuration": server,
    })
    opponents = [
        cls(battle_format=BATTLE_FORMAT, max_concurrent_battles=10, log_level=40,
            server_configuration=server)
        for cls in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=n_battles))
    print("--- Final win rates ---")
    for opp in opponents:
        print(f"{opp.username}: {round(100 * _win_rate(opp))}% (n={opp.n_finished_battles})")
