"""Battler PPO trainer: BC-init or from-scratch, vs a rotating fixed-opponent mix.

Two tiers:
  --single   one env in the main process, one Showdown server (simplest)
  default    --envs N processes round-robin over --servers M Showdown ports

Needs a Showdown server on the port (default 8020, this project's range —
8000-8007 belong to the old project):
    node pokemon-showdown start --no-security --port 8020

Modules: battler_obs (encoder), battler_policy (net + eval player),
battler_env (env stack), battler_eval (diagnostics/callback). This file is
only the training loop and CLI. The imports below also re-export the policy
classes into __main__ so SB3 checkpoints (pickled against __main__) load.
"""

import sys
from pathlib import Path

# `python ai_util/train_battler.py` puts ai_util/ on sys.path, so `import poke_env`
# would resolve to the local ai_util/poke_env/ wrapper instead of the installed package.
_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(_here.parent))

import argparse
import datetime
import functools

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from ai_util.battler_env import inner_env, make_env, preflight, server_conf
from ai_util.battler_eval import EvalCallback, _final_eval, eval_checkpoint, obs_dump
# the class imports below double as checkpoint compat: SB3 zips pickle the
# policy against __main__, so these names must exist here at load time
from ai_util.battler_policy import MaskedActorCriticPolicy, PolicyPlayer
from ai_util.battler_obs import embed_battle  # noqa: F401 (checkpoint compat)


def _make_ppo(env, stamp: str, num_envs: int) -> PPO:
    return PPO(
        MaskedActorCriticPolicy,
        env,
        # E1: batch 128 + lr 3e-4 gave approx_kl 0.03-0.06 (healthy is <0.02):
        # each update jumped the policy 3-10x too far. Bigger batch + smaller
        # lr is the textbook variance fix.
        learning_rate=1e-4,
        n_steps=3072 // num_envs,
        batch_size=1024,
        gamma=0.99,  # 40-turn episodes: 0.95 discounted the +30 win to ~3.9 by turn 1
        # 0.05 pinned entropy at ~1.2 nats for 8M steps and the policy stayed
        # soft (25% switch rate vs never-switching opponents); 0.01 matches the
        # baseline run that sharpened to entropy 0.31 / 72% vs MaxBP.
        ent_coef=0.01,
        max_grad_norm=5.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        tensorboard_log=f"logs/tensorboard/battler_{stamp}",
    )


def train_single(args: argparse.Namespace) -> None:
    """Single-threaded trainer: one env in the main process, one Showdown server.

    As simple as this task gets — no subprocesses, no eval callback; use
    --eval on the saved model to measure it.
    """
    server = preflight(args.port)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"models/battler_single_{stamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = inner_env(server)
    ppo = _make_ppo(env, stamp, num_envs=1)
    ppo.learn(args.total_timesteps)
    ppo.save(save_dir / "final")
    env.close()
    _final_eval(ppo.policy, server, args.final_battles)
    print(f"saved {save_dir}")


def train(args: argparse.Namespace) -> None:
    """Distributed trainer: one process+env each, round-robin over
    `--servers` Showdown ports starting at --port. Needs those servers up."""
    ports = [args.port + i % args.servers for i in range(args.envs)]
    for p in sorted(set(ports)):
        preflight(p)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"models/battler_{stamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv(
        [functools.partial(make_env, port) for port in ports],
        start_method="spawn",
    )
    ppo = _make_ppo(vec_env, stamp, num_envs=args.envs)
    if args.init_from:
        ppo.policy.load_state_dict(torch.load(args.init_from, map_location=ppo.device))
        print(f"initialized policy from {args.init_from}")
    callback = EvalCallback(save_dir, server_conf(args.port), args.eval_every, args.n_eval_battles)
    ppo.learn(args.total_timesteps, callback=callback)
    ppo.save(save_dir / "final")
    vec_env.close()
    _final_eval(ppo.policy, server_conf(args.port), args.final_battles)
    print(f"saved {save_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", action="store_true",
                   help="single-threaded trainer (1 env, main process, 1 server)")
    p.add_argument("--envs", type=int, default=12, help="distributed: env count")
    p.add_argument("--servers", type=int, default=1,
                   help="distributed: envs round-robin over ports port..port+servers-1")
    p.add_argument("--total-timesteps", type=int, default=6_000_000)
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--n-eval-battles", type=int, default=50)
    p.add_argument("--final-battles", type=int, default=200)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--init-from", help="policy.pt from pretrain_bc.py to start from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-dump", action="store_true", help="one random battle, print obs stats")
    p.add_argument("--eval", metavar="CHECKPOINT", help="play a saved model vs the baselines")
    p.add_argument("--n", type=int, default=100, help="battles per opponent for --eval")
    p.add_argument("--stochastic", action="store_true", help="--eval samples instead of argmax")
    args = p.parse_args()
    if args.obs_dump:
        obs_dump(args.port, args.seed)
    elif args.eval:
        eval_checkpoint(args.eval, args.port, args.n, deterministic=not args.stochastic)
    elif args.single:
        train_single(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
