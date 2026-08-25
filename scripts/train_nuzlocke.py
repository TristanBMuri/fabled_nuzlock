"""Train the nuzlocke meta-policy. Requires Showdown on the given port.

Usage:
  uv run python scripts/train_nuzlocke.py --random            # manual episode
  uv run python scripts/train_nuzlocke.py --updates 20        # PPO smoke
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from nuzlocke_env import N_META_ACTIONS, OBS_DIM, NuzlockeEnv  # noqa: E402
from ppo import MaskedPolicy, compute_gae, explained_variance, ppo_update  # noqa: E402

ROLLOUT = int(os.environ.get("ROLLOUT", 512))
GAMMA, LAM = 0.99, 0.95  # longer horizon: a run is ~50 decisions
CLIP, VF_CLIP, ENTROPY, LR = 0.2, 10.0, 0.1, 2.5e-4  # higher entropy: 5-step
# episodes collapse exploration at 0.05
EPOCHS, MINIBATCHES = 4, 4


def manual_episode(env, seed=0):
    """Random legal actions, verbose — the G2 sanity walk."""
    rng = np.random.RandomState(seed)
    obs, info = env.reset(seed=seed)
    steps = 0
    while True:
        print(env.render())
        valid = np.flatnonzero(info["action_mask"])
        a = int(rng.choice(valid))
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            print(env.render(), f"| done, steps={steps}")
            break


def _worker_main(conn, i: int, port: int, info_mode: str, seed: int):
    """One env in its own process — bypasses the GIL (SimpleHeuristics is
    Python-CPU-bound; threads capped us at ~80 st/s on one core)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from nuzlocke_env import N_META_ACTIONS, OBS_DIM, NuzlockeEnv
    from ppo import MaskedPolicy

    env = NuzlockeEnv(port=port + i, info_mode=info_mode, seed=seed + i)
    agent = MaskedPolicy(OBS_DIM, N_META_ACTIONS)  # CPU
    obs, info = env.reset(seed=seed + i)
    cr, cl = 0.0, 0
    try:
        while True:
            msg = conn.recv()
            if msg[0] == "close":
                break
            assert msg[0] == "collect"
            weights, n_steps = msg[1], msg[2]
            agent.load_state_dict(weights)
            out_s = np.zeros((n_steps, OBS_DIM), np.float32)
            out_m = np.zeros((n_steps, N_META_ACTIONS), np.float32)
            out_a = np.zeros(n_steps, np.int64)
            out_lp = np.zeros(n_steps, np.float32)
            out_v = np.zeros(n_steps, np.float32)
            out_r = np.zeros(n_steps, np.float32)
            out_d = np.zeros(n_steps, np.float32)
            finished = []
            for t in range(n_steps):
                o = torch.as_tensor(obs).unsqueeze(0)
                m = torch.as_tensor(info["action_mask"],
                                   dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    a, lp, _, v = agent.get_action_and_value(o, m)
                a_i = int(a.item())
                out_s[t] = obs
                out_m[t] = info["action_mask"]
                out_a[t] = a_i
                out_lp[t] = float(lp.item())
                out_v[t] = float(v.item())
                try:
                    ob, r, term, trunc, info = env.step(a_i)
                except Exception as e:
                    # Don't kill the whole train job on one bad battle
                    # (empty party / Showdown reject used to raise here).
                    print(f"worker {i}: step failed ({type(e).__name__}: {e}); "
                          f"forcing episode end", flush=True)
                    r, term, trunc = -5.0, True, False
                    info = {"action_mask": np.zeros(N_META_ACTIONS, np.int64)}
                    info["action_mask"][0] = 1
                    ob = obs
                out_r[t] = r
                out_d[t] = float(term or trunc)
                cr += r
                cl += 1
                if term or trunc:
                    finished.append((cr, cl, env.trainer_idx))
                    cr, cl = 0.0, 0
                    obs, info = env.reset()
                else:
                    obs = ob
            with torch.no_grad():
                o = torch.as_tensor(obs).unsqueeze(0)
                m = torch.as_tensor(info["action_mask"],
                                   dtype=torch.float32).unsqueeze(0)
                _, _, _, last_v = agent.get_action_and_value(o, m)
            conn.send((out_s, out_m, out_a, out_lp, out_v, out_r, out_d,
                       float(last_v.item()), finished))
    finally:
        conn.close()


def train(updates: int, seed: int, port: int, info_mode: str, n_envs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = MaskedPolicy(OBS_DIM, N_META_ACTIONS).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    latest = ckpt / "nuzlocke_latest.pt"
    if latest.exists():
        agent.load_state_dict(
            torch.load(latest, map_location=device)["model"])
        print(f"resumed weights from {latest}", flush=True)
    writer = SummaryWriter(
        f"runs/{datetime.now():%Y%m%d-%H%M%S}-nuzlocke{'x' + str(n_envs) if n_envs > 1 else ''}")

    # spawn workers (one Showdown port + one process each)
    ctx = get_context("spawn")
    pipes, procs = [], []
    for i in range(n_envs):
        parent, child = ctx.Pipe()
        p = ctx.Process(
            target=_worker_main,
            args=(child, i, port, info_mode, seed),
            daemon=True,
        )
        p.start()
        child.close()
        pipes.append(parent)
        procs.append(p)
    print(f"spawned {n_envs} env processes on ports {port}..{port + n_envs - 1}",
          flush=True)

    ep_r, ep_len, cleared = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    steps_per_env = ROLLOUT // n_envs
    s = torch.zeros(ROLLOUT, OBS_DIM, device=device)
    masks = torch.zeros(ROLLOUT, N_META_ACTIONS, device=device)
    acts = torch.zeros(ROLLOUT, dtype=torch.long, device=device)
    lps = torch.zeros(ROLLOUT, device=device)
    rews = torch.zeros(ROLLOUT, device=device)
    dones = torch.zeros(ROLLOUT, device=device)
    vals = torch.zeros(ROLLOUT, device=device)

    gstep = 0
    try:
        for update in range(1, updates + 1):
            t0 = time.time()
            weights = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
            for pipe in pipes:
                pipe.send(("collect", weights, steps_per_env))
            results = [pipe.recv() for pipe in pipes]

            last_v = torch.zeros(n_envs, device=device)
            for i, (out_s, out_m, out_a, out_lp, out_v, out_r, out_d,
                    lv, finished) in enumerate(results):
                for t in range(steps_per_env):
                    idx = t * n_envs + i
                    s[idx] = torch.as_tensor(out_s[t], device=device)
                    masks[idx] = torch.as_tensor(out_m[t], device=device)
                    acts[idx] = int(out_a[t])
                    lps[idx] = float(out_lp[t])
                    vals[idx] = float(out_v[t])
                    rews[idx] = float(out_r[t])
                    dones[idx] = float(out_d[t])
                    gstep += 1
                last_v[i] = lv
                for er, el, tc in finished:
                    ep_r.append(er)
                    ep_len.append(el)
                    cleared.append(tc)

            adv = torch.zeros_like(rews)
            for j in range(n_envs):
                sl = slice(j, ROLLOUT, n_envs)
                adv_j, _ = compute_gae(rews[sl], vals[sl], dones[sl],
                                       last_v[j], GAMMA, LAM)
                adv[sl] = adv_j
            ret = adv + vals
            stats = ppo_update(agent, opt, s, masks, acts, lps, adv, ret, vals,
                               clip_coef=CLIP, vf_clip=VF_CLIP,
                               entropy_coef=ENTROPY, epochs=EPOCHS,
                               minibatches=MINIBATCHES)
            dt = time.time() - t0
            writer.add_scalar("episode_reward",
                              np.mean(ep_r) if ep_r else 0, gstep)
            writer.add_scalar("trainers_cleared",
                              np.mean(cleared) if cleared else 0, gstep)
            writer.add_scalar("episode_len",
                              np.mean(ep_len) if ep_len else 0, gstep)
            for k, val in stats.items():
                writer.add_scalar(k, val, gstep)
            writer.add_scalar("explained_variance",
                              explained_variance(vals, ret), gstep)
            print(f"update {update:3d}/{updates} "
                  f"R={np.mean(ep_r) if ep_r else 0:+.1f} "
                  f"cleared={np.mean(cleared) if cleared else 0:.1f} "
                  f"len={np.mean(ep_len) if ep_len else 0:.0f} "
                  f"H={stats['entropy']:.2f} {ROLLOUT / dt:.1f} st/s",
                  flush=True)
            torch.save({"model": agent.state_dict(), "arch": "mlp",
                        "obs_dim": OBS_DIM}, ckpt / "nuzlocke_latest.pt")
    finally:
        for pipe in pipes:
            try:
                pipe.send(("close",))
            except Exception:
                pass
            pipe.close()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--envs", type=int, default=1,
                   help="parallel runs (one server+process per env on port..)")
    p.add_argument("--info", choices=["none", "revealed", "full"],
                   default="full")
    p.add_argument("--random", action="store_true")
    args = p.parse_args()
    if args.random:
        manual_episode(NuzlockeEnv(port=args.port, info_mode=args.info,
                                   seed=args.seed), args.seed)
    else:
        train(args.updates, args.seed, args.port, args.info, args.envs)
