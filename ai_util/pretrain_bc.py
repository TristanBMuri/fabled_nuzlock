"""E2 phase 1+2: behavior-cloning pretraining from SimpleHeuristics.

collect:  battles where SimpleHeuristics plays OUR side of the training env
          (opponent side rotates 50/25/25 heuristics/MaxBP/Random); records
          (obs, action_mask, action) for every real decision.
pretrain: masked cross-entropy into the exact policy net PPO uses; saves a
          state_dict that train_battler.py --init-from can load.

Pre-registered diagnostics (see README): masked-CE vs uniform baseline,
accuracy split by move/switch, teacher-behavior context stats.
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(_here.parent))

import argparse
import datetime
import json
import multiprocessing as mp

import numpy as np
import torch
from gymnasium.spaces import Box, Dict, Discrete

from ai_util.battler_env import (
    BattlerEnv,
    RotatingOpponentWrapper,
    SafeSingleAgentWrapper,
    preflight,
    server_conf,
)
from ai_util.battler_eval import _eval_policy
from ai_util.battler_obs import N_FEATURES, _damage_multiplier
from ai_util.battler_policy import N_ACTIONS, MaskedActorCriticPolicy, PolicyPlayer
from poke_env.battle import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.environment import SinglesEnv
from poke_env.player import DefaultBattleOrder, SimpleHeuristicsPlayer

GAME_FORMAT = "gen8customgamenogimmicks"


def collect_worker(worker: int, n_battles: int, port: int, out_dir: Path) -> None:
    server = preflight(port)
    env = BattlerEnv(battle_format="gen9randombattle", log_level=40,
                     server_configuration=server)
    inner = SafeSingleAgentWrapper(env, opponent=SimpleHeuristicsPlayer(
        battle_format="gen9randombattle", server_configuration=server,
        start_listening=False, log_level=40))
    rot = RotatingOpponentWrapper(inner, server)
    teacher = SimpleHeuristicsPlayer(
        battle_format="gen9randombattle", server_configuration=server,
        start_listening=False, log_level=40)

    obs_buf, mask_buf, act_buf = [], [], []
    ctx = {"switch": 0, "move": 0, "max_mult": 0}
    rng = np.random.default_rng(1000 + worker)
    battles_done = 0
    while battles_done < n_battles:
        obs, _ = rot.reset()
        battle = inner.env.battle1
        done = False
        while not done and battle is not None and not battle.finished:
            if battle.wait:
                action = np.int64(-2)
            else:
                order = teacher.choose_move(battle)
                if isinstance(order, DefaultBattleOrder) or order.order is None:
                    action = np.int64(-2)
                else:
                    action = SinglesEnv.order_to_action(order, battle, fake=True, strict=False)
                    obs_buf.append(obs["observation"].astype(np.float16))
                    mask_buf.append(obs["action_mask"].astype(np.int8))
                    act_buf.append(np.int16(action))
                    if action < 6:
                        ctx["switch"] += 1
                    elif isinstance(order.order, Move):
                        ctx["move"] += 1
                        opp = battle.opponent_active_pokemon
                        mults = [_damage_multiplier(m, (opp.type_1, opp.type_2))
                                 for m in battle.available_moves
                                 if m.category != MoveCategory.STATUS and opp is not None]
                        if mults and _damage_multiplier(order.order, (opp.type_1, opp.type_2)) >= max(mults) - 1e-9:
                            ctx["max_mult"] += 1
            obs, _, term, trunc, _ = rot.step(np.int64(action))
            done = term or trunc
        battles_done += 1
        if battles_done % 50 == 0:
            print(f"worker {worker}: {battles_done}/{n_battles} battles, {len(act_buf)} pairs", flush=True)

    np.savez_compressed(
        out_dir / f"shard_{worker}.npz",
        obs=np.asarray(obs_buf), mask=np.asarray(mask_buf), action=np.asarray(act_buf))
    total = max(ctx["move"], 1)
    print(f"worker {worker} done: {len(act_buf)} pairs, "
          f"teacher switch={ctx['switch'] / max(len(act_buf), 1):.1%}, "
          f"teacher max-mult move={ctx['max_mult'] / total:.0%}", flush=True)
    rot.close()


def collect(args: argparse.Namespace) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"models/bc_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=collect_worker, args=(w, args.battles // args.workers, args.port, out_dir))
        for w in range(args.workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    shards = sorted(out_dir.glob("shard_*.npz"))
    obs = np.concatenate([np.load(s)["obs"] for s in shards])
    mask = np.concatenate([np.load(s)["mask"] for s in shards])
    action = np.concatenate([np.load(s)["action"] for s in shards])
    np.savez_compressed(out_dir / "data.npz", obs=obs, mask=mask, action=action)
    for s in shards:
        s.unlink()
    n = len(action)
    print(f"collected {n} pairs ({n / args.battles:.0f}/battle avg) -> {out_dir / 'data.npz'}")
    return out_dir


def _batch(arrays, idx, device):
    mask = arrays["mask"][idx].astype(np.int64)
    if mask.shape[1] < N_ACTIONS:  # gen8 battles have a 22-dim mask
        mask = np.pad(mask, ((0, 0), (0, N_ACTIONS - mask.shape[1])))
    return {
        "observation": torch.as_tensor(arrays["obs"][idx].astype(np.float32), device=device),
        "action_mask": torch.as_tensor(mask, device=device),
    }


def _gauntlet_winrate(policy, pairs: list, server, n: int) -> float:
    """Policy plays the player side of held-out gauntlet pairs vs SimpleHeuristics,
    in the in-process sim (`server` is unused; kept for the call signature)."""
    from ai_util.evaluate import evaluate, heuristic_brain, prior_brain
    from ai_util.sim import SimBridge

    bridge, me, foe = SimBridge(), prior_brain(policy), heuristic_brain()
    wins = 0
    for p_team, e_team in pairs[:n]:
        wins += evaluate(bridge, p_team, e_team, me, foe, n=1)["p_win"]
    bridge.close()
    return wins / max(1, len(pairs[:n]))


def pretrain(args: argparse.Namespace) -> Path:
    paths = args.data if isinstance(args.data, list) else [args.data]
    loaded = [np.load(p) for p in paths]
    # "value" (battle outcome per decision, collect_game >= 2026-09-15) trains the
    # critic; datasets without it train the actor only
    keys = ("obs", "mask", "action") + (("value",) if all("value" in d for d in loaded) else ())
    arrays = {k: np.concatenate([d[k] for d in loaded]) for k in keys}
    if len(paths) > 1:
        print("datasets: " + ", ".join(f"{p} ({len(d['action'])})" for p, d in zip(paths, loaded)))
    if args.value_only and "value" not in arrays:
        raise SystemExit("--value-only needs datasets with a value column")
    n = len(arrays["action"])
    perm = np.random.default_rng(0).permutation(n)
    n_val = max(n // 20, 100)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_space = Dict({
        "observation": Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int64),
    })
    policy = MaskedActorCriticPolicy(obs_space, Discrete(N_ACTIONS), lambda _: 3e-4,
                                     net_arch=[int(a) for a in args.net_arch.split(",")]).to(device)
    if args.init:
        policy.load_state_dict(torch.load(args.init, map_location=device))
    params = sum(p.numel() for p in policy.parameters())
    print(f"pretraining {params / 1e3:.0f}k-param policy on {len(train_idx)} pairs "
          f"(val {len(val_idx)}), device={device}"
          + (f", init {args.init}" if args.init else "")
          + (", value head" + (" only" if args.value_only else "") if "value" in arrays else ""))

    def _value_target(chunk):
        return torch.as_tensor(arrays["value"][chunk].astype(np.float32), device=device)

    def _metrics(idx, bs):
        losses, vlosses, correct, switch_n, switch_c = [], [], 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                chunk = idx[i:i + bs]
                obs = _batch(arrays, chunk, device)
                if not args.value_only:  # leaf datasets carry no action labels (-1)
                    act = torch.as_tensor(arrays["action"][chunk].astype(np.int64), device=device)
                    policy._mask = obs["action_mask"]
                    dist = policy.get_distribution(obs)
                    losses.append(-dist.log_prob(act).mean().item())
                    pred = dist.distribution.logits.argmax(-1)
                    ok = (pred == act).cpu().numpy()
                    correct += int(ok.sum())
                    is_switch = (arrays["action"][chunk] < 6)
                    switch_n += int(is_switch.sum())
                    switch_c += int(ok[is_switch].sum())
                if "value" in arrays:
                    v = policy.predict_values(obs).squeeze(-1)
                    vlosses.append(((v - _value_target(chunk)) ** 2).mean().item())
        return (float(np.mean(losses)) if losses else float("nan"), correct / len(idx),
                (switch_c / switch_n if switch_n else float("nan")),
                float(np.mean(vlosses)) if vlosses else float("nan"))

    valid_counts = arrays["mask"][val_idx].sum(1)
    uniform_ce = float(np.log(valid_counts.mean()))
    print(f"baseline: uniform-over-valid CE = {uniform_ce:.3f} (avg {valid_counts.mean():.1f} valid actions)")
    if "value" in arrays:
        vv = arrays["value"][val_idx].astype(np.float32)
        print(f"baseline: value MSE = {vv.var():.3f} (predicting the mean {vv.mean():+.3f})")

    opt = policy.optimizer
    eval_pairs = None
    if args.eval_pairs:
        eval_pairs = json.loads(Path(args.eval_pairs).read_text())
    best_wr, patience_left, last_out = -1.0, args.patience, None
    out_dir = Path(args.out).parent if args.out else Path("models")
    tag = Path(args.out).stem if args.out else "policy"

    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(len(train_idx))
        for i in range(0, len(order), args.batch):
            chunk = train_idx[order[i:i + args.batch]]
            obs = _batch(arrays, chunk, device)
            act = torch.as_tensor(arrays["action"][chunk].astype(np.int64), device=device)
            policy._mask = obs["action_mask"]
            loss = 0.0 if args.value_only else -policy.get_distribution(obs).log_prob(act).mean()
            if "value" in arrays:
                v = policy.predict_values(obs).squeeze(-1)
                loss = loss + args.value_coef * ((v - _value_target(chunk)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        tl, ta, ts, tv = _metrics(train_idx[:20000], 4096)
        vl, va, vs, vv = _metrics(val_idx, 4096)
        msg = (f"epoch {epoch}: train CE {tl:.3f} acc {ta:.0%} (switch {ts:.0%}) | "
               f"val CE {vl:.3f} acc {va:.0%} (switch {vs:.0%})")
        if "value" in arrays:
            msg += f" | value MSE train {tv:.3f} val {vv:.3f}"
        if eval_pairs is not None and (epoch + 1) % args.eval_every == 0:
            wr = _gauntlet_winrate(policy, eval_pairs, server_conf(args.port), args.eval_battles)
            last_out = out_dir / f"{tag}_epoch{epoch}.pt"
            torch.save(policy.state_dict(), last_out)
            if wr > best_wr:
                best_wr, patience_left = wr, args.patience
                best_out = out_dir / f"{tag}_best.pt"
                torch.save(policy.state_dict(), best_out)
                msg += f" | gauntlet win rate {wr:.1%} * (best, saved)"
            else:
                patience_left -= 1
                msg += f" | gauntlet win rate {wr:.1%} (patience {patience_left}/{args.patience})"
        print(msg, flush=True)
        if eval_pairs is not None and patience_left <= 0:
            print(f"early stop: no improvement for {args.patience} evals (best {best_wr:.1%})")
            break

    out = Path(args.out) if args.out else Path(f"models/policy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(policy.state_dict(), out)
    print(f"saved {out}")
    return out


def eval_bc(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_space = Dict({
        "observation": Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32),
        "action_mask": Box(low=0, high=1, shape=(N_ACTIONS,), dtype=np.int64),
    })
    policy = MaskedActorCriticPolicy(obs_space, Discrete(N_ACTIONS), lambda _: 3e-4,
                                     net_arch=[int(a) for a in args.net_arch.split(",")]).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    server = preflight(args.port)
    _eval_policy(policy, server, args.n, deterministic=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["collect", "pretrain", "eval"])
    p.add_argument("--battles", type=int, default=4000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--port", type=int, default=8022)
    p.add_argument("--data", nargs="+", help="data.npz file(s) for pretrain (concatenated)")
    p.add_argument("--policy", help="policy.pt for eval")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--net-arch", default="256,256", help="comma-separated MLP widths")
    p.add_argument("--out", help="where to save policy.pt (default: timestamped in models/)")
    p.add_argument("--eval-pairs", help="json of held-out (player, enemy) gauntlet pairs; "
                                        "enables win-rate checkpointing + early stop")
    p.add_argument("--eval-battles", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5, help="epochs between win-rate evals")
    p.add_argument("--patience", type=int, default=3, help="evals without improvement before stop")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--init", help="state_dict to start from (e.g. to add a value head to a prior)")
    p.add_argument("--value-only", action="store_true",
                   help="train only the value head (actor untouched; needs --init to be useful)")
    p.add_argument("--value-coef", type=float, default=1.0, help="weight of the value MSE loss")
    p.add_argument("--n", type=int, default=100, help="battles per opponent for eval")
    args = p.parse_args()
    if args.phase == "collect":
        collect(args)
    elif args.phase == "pretrain":
        pretrain(args)
    else:
        eval_bc(args)


if __name__ == "__main__":
    main()
