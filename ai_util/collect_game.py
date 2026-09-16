"""Prototype BC collector for the gauntlet (game-team) distribution.

Samples (player_team, enemy_team) pairs from the gauntlet harvest jsonl —
real BDSP matchups, game levels/movesets — and plays them in
gen8customgamenogimmicks with SimpleHeuristics on the player side,
recording (obs, action_mask, action) per decision.

Modes:
  collect  harvest BC pairs from sampled matchups (in-process sim by default;
           --backend server for the old websocket path)
  holdout  write a json of held-out (player, enemy) pairs for pretrain --eval-pairs
  baseline play SimpleHeuristics-vs-SimpleHeuristics on holdout pairs on a server
           (the sim equivalent is `evaluate.py --check --from-harvest`)
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

from ai_util.battler_env import preflight, server_conf
from ai_util.battler_obs import embed_battle
from ai_util.battler_policy import action_mask, order_to_action
from poke_env.player import ForfeitBattleOrder, SimpleHeuristicsPlayer

GAME_FORMAT = "gen8customgamenogimmicks"


def sample_pairs(jsonl: str, n: int, seed: int, trainer_ids: set | None = None) -> list:
    """(player_team, enemy_team) pairs; trainer_ids restricts to those enemies."""
    with open(jsonl) as f:
        rows = [json.loads(line) for line in f]
    if trainer_ids:
        rows = [r for r in rows if r.get("trainer_id") in trainer_ids]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [(rows[i]["player_team"], rows[i]["enemy_team"]) for i in idx]


class RecordingHeuristics(SimpleHeuristicsPlayer):
    """SimpleHeuristics that records (obs, mask, action) for every decision."""

    def __init__(self, *args, record: list, ctx: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._record = record
        self._ctx = ctx

    def choose_move(self, battle):
        if not battle.wait:
            # some low-level BDSP pairs cannot damage each other; without a
            # cap those battles run forever (both sides are heuristics)
            if battle.turn > 100:
                return ForfeitBattleOrder()
            mask = action_mask(battle)
            if mask.sum() > 0:
                order = super().choose_move(battle)
                try:
                    action = order_to_action(order, battle)
                except (ValueError, StopIteration):
                    return order
                self._record.append(
                    (embed_battle(battle).astype(np.float16), mask.astype(np.int8), np.int16(action)))
                self._ctx["n"] += 1
                if action < 6:
                    self._ctx["switch"] += 1
                return order
        return super().choose_move(battle)


def collect_worker(worker: int, pairs: list, port: int, out_dir: Path) -> None:
    import asyncio

    record, ctx, failures = [], {"n": 0, "switch": 0}, 0
    kw = dict(battle_format=GAME_FORMAT, server_configuration=server_conf(port), log_level=40)

    async def _stop(p) -> None:
        try:
            await asyncio.wait_for(p.ps_client.stop_listening(), timeout=10)
        except Exception:
            pass

    async def run() -> None:
        nonlocal failures
        # stagger logins: a simultaneous 16-connection burst hung workers
        await asyncio.sleep(worker * 3)
        for i, (p_team, e_team) in enumerate(pairs):
            me = RecordingHeuristics(team=p_team, record=record, ctx=ctx, **kw)
            foe = SimpleHeuristicsPlayer(team=e_team, **kw)
            try:
                # hard cap: hung handshakes / unprogressable battles must not
                # stall the worker forever
                await asyncio.wait_for(me.battle_against(foe, n_battles=1), timeout=180)
            except Exception:
                failures += 1
            finally:
                await _stop(me)
                await _stop(foe)
            if (i + 1) % 100 == 0:
                print(f"worker {worker}: {i + 1}/{len(pairs)} battles, {ctx['n']} pairs, "
                      f"{failures} failures", flush=True)

    asyncio.run(run())
    np.savez_compressed(
        out_dir / f"shard_{worker}.npz",
        obs=np.asarray([r[0] for r in record]),
        mask=np.asarray([r[1] for r in record]),
        action=np.asarray([r[2] for r in record]))
    print(f"worker {worker} done: {ctx['n']} pairs, switch={ctx['switch'] / max(ctx['n'], 1):.1%}, "
          f"{failures} failed battles", flush=True)


def recording_brain(record: list, ctx: dict):
    """SimpleHeuristics as a sim Brain that records (obs, mask, action) per decision."""
    from ai_util.sim import order_to_choice
    teacher = SimpleHeuristicsPlayer(battle_format=GAME_FORMAT, start_listening=False, log_level=50)

    def brain(sb, side):
        battle = sb.mirrors[side]
        if battle.wait:
            return "default"
        order = teacher.choose_move(battle)
        mask = action_mask(battle)
        if mask.sum() > 0:
            try:
                action = order_to_action(order, battle)
            except (ValueError, StopIteration):
                return order_to_choice(order)
            record.append(
                (embed_battle(battle).astype(np.float16), mask.astype(np.int8), np.int16(action)))
            ctx["n"] += 1
            if action < 6:
                ctx["switch"] += 1
        return order_to_choice(order)

    return brain


KEYS = ("obs", "mask", "action", "value")


def _save_shard(path: Path, record: list, values: list) -> None:
    """record rows are (obs, mask, action); value = the battle's outcome_value
    (win - 0.5 * fainted fraction), the Monte Carlo target for the value head."""
    np.savez_compressed(path, obs=np.asarray([r[0] for r in record]),
                        mask=np.asarray([r[1] for r in record]),
                        action=np.asarray([r[2] for r in record]),
                        value=np.asarray(values, dtype=np.float16))


def collect_worker_sim(worker: int, pairs: list, out_dir: Path) -> None:
    from ai_util.evaluate import heuristic_brain, outcome_value
    from ai_util.sim import SimBattle, SimBridge

    record, values, ctx, failures = [], [], {"n": 0, "switch": 0}, 0
    bridge, me, foe = SimBridge(), None, heuristic_brain()
    me = recording_brain(record, ctx)
    for i, (p_team, e_team) in enumerate(pairs):
        start = len(record)
        try:
            sb = SimBattle(bridge, GAME_FORMAT, p_team, e_team)
            sb.play({"p1": me, "p2": foe}, max_turns=100)
            values += [outcome_value(sb.state(), "p1")] * (len(record) - start)
            sb.close()
        except Exception as e:
            failures += 1
            del record[start:]
            print(f"worker {worker}: battle failed ({type(e).__name__}: {e})", flush=True)
            bridge = SimBridge()
        if (i + 1) % 500 == 0:
            print(f"worker {worker}: {i + 1}/{len(pairs)} battles, {ctx['n']} pairs, "
                  f"{failures} failures", flush=True)
    bridge.close()
    _save_shard(out_dir / f"shard_{worker}.npz", record, values)
    print(f"worker {worker} done: {len(record)} pairs, switch={ctx['switch'] / max(ctx['n'], 1):.1%}, "
          f"{failures} failed battles", flush=True)


def collect_worker_search(worker: int, pairs: list, out_dir: Path, policy: str,
                          k: int, rollouts: int) -> None:
    """Expert iteration labels: the search's chosen action on each decision."""
    from ai_util.battler_policy import load_policy
    from ai_util.evaluate import Searcher, heuristic_brain, outcome_value
    from ai_util.sim import SimBattle, SimBridge

    bridge, foe = SimBridge(), heuristic_brain()
    searcher = Searcher(load_policy(policy, device="cpu"), foe, k=k, rollouts=rollouts)
    searcher.leaf_record = leaves = []  # (leaf obs, mask, -1, rollout outcome) per rollout
    record, values, ctx, failures = [], [], {"n": 0, "switch": 0, "searched": 0, "changed": 0}, 0

    def me(sb, side):
        m = sb.mirrors[side]
        if m.wait:
            return "default"
        obs, mask = embed_battle(m), action_mask(m)
        choice = searcher(sb, side)
        if searcher.last_action is not None and mask.sum() > 0:
            record.append((obs.astype(np.float16), mask.astype(np.int8),
                           np.int16(searcher.last_action)))
            ctx["n"] += 1
            ctx["searched"] += searcher.searched_last
            if searcher.last_action < 6:
                ctx["switch"] += 1
        return choice

    for i, (p_team, e_team) in enumerate(pairs):
        start = len(record)
        try:
            sb = SimBattle(bridge, GAME_FORMAT, p_team, e_team)
            sb.play({"p1": me, "p2": foe}, max_turns=100)
            values += [outcome_value(sb.state(), "p1")] * (len(record) - start)
            sb.close()
        except Exception as e:
            failures += 1
            del record[start:]
            print(f"worker {worker}: battle failed ({type(e).__name__}: {e})", flush=True)
            bridge = SimBridge()
            searcher.enemy = foe
        if (i + 1) % 50 == 0:
            print(f"worker {worker}: {i + 1}/{len(pairs)} battles, {ctx['n']} pairs "
                  f"({ctx['searched']} searched), {failures} failures", flush=True)
    bridge.close()
    _save_shard(out_dir / f"shard_{worker}.npz", record, values)
    _save_shard(out_dir / f"leaf_{worker}.npz", [r[:3] for r in leaves], [r[3] for r in leaves])
    print(f"worker {worker} done: {len(record)} pairs, {len(leaves)} leaves, searched {ctx['searched']}, "
          f"switch={ctx['switch'] / max(ctx['n'], 1):.1%}, {failures} failed battles", flush=True)


def collect_search(args: argparse.Namespace) -> None:
    """Search labels on matchups vs the given trainers (the deadliest fights)."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"models/bc_search_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {int(t) for t in args.trainers.split(",")} if args.trainers else None
    pairs = sample_pairs(args.jsonl, args.battles, args.seed, ids)
    print(f"{len(pairs)} matchups vs trainers {sorted(ids) if ids else 'all'}", flush=True)
    chunks = [pairs[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=collect_worker_search,
                         args=(w, chunks[w], out_dir, args.policy, args.k, args.rollouts))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for kind, name in (("shard", "data"), ("leaf", "leaves")):
        shards = sorted(out_dir.glob(f"{kind}_*.npz"))
        data = {k: np.concatenate([np.load(s)[k] for s in shards]) for k in KEYS}
        np.savez_compressed(out_dir / f"{name}.npz", **data)
        for s in shards:
            s.unlink()
        print(f"collected {len(data['action'])} {name} -> {out_dir / name}.npz")


def holdout(args: argparse.Namespace) -> None:
    """Held-out pairs (seed offset so they don't overlap a seed-0 collect sample)."""
    pairs = sample_pairs(args.jsonl, args.n, args.seed + 1)
    out = Path(args.out or "models/gauntlet_holdout_cap.json")
    out.write_text(json.dumps(pairs))
    print(f"wrote {len(pairs)} holdout pairs -> {out}")


def collect(args: argparse.Namespace) -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"models/bc_game_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {int(t) for t in args.trainers.split(",")} if args.trainers else None
    pairs = sample_pairs(args.jsonl, args.battles, args.seed, ids)
    chunks = [pairs[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    if args.backend == "sim":
        procs = [ctx.Process(target=collect_worker_sim, args=(w, chunks[w], out_dir))
                 for w in range(args.workers)]
    else:
        procs = [ctx.Process(target=collect_worker, args=(w, chunks[w], args.port, out_dir))
                 for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    shards = sorted(out_dir.glob("shard_*.npz"))
    data = {k: np.concatenate([np.load(s)[k] for s in shards]) for k in KEYS}
    np.savez_compressed(out_dir / "data.npz", **data)
    for s in shards:
        s.unlink()
    print(f"collected {len(data['action'])} pairs from {args.battles} matchups -> {out_dir / 'data.npz'}")


def baseline(args: argparse.Namespace) -> None:
    """SimpleHeuristics-vs-SimpleHeuristics on holdout pairs: the reference line."""
    import asyncio

    server = preflight(args.port)
    pairs = sample_pairs(args.jsonl, args.n, args.seed)

    async def run() -> None:
        kw = dict(battle_format=GAME_FORMAT, server_configuration=server, log_level=40)
        wins = failures = 0
        for i, (p_team, e_team) in enumerate(pairs):
            a = SimpleHeuristicsPlayer(team=p_team, **kw)
            b = SimpleHeuristicsPlayer(team=e_team, **kw)
            try:
                await a.battle_against(b, n_battles=1)
                wins += bool(a.n_won_battles)
            except Exception:
                failures += 1
            finally:
                for p in (a, b):
                    try:
                        await p.ps_client.stop_listening()
                    except Exception:
                        pass
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(pairs)}: {wins / (i + 1 - failures):.1%}", flush=True)
        print(f"baseline SimpleHeuristics-vs-SimpleHeuristics: "
              f"{wins / (len(pairs) - failures):.1%} ({failures} failures)")

    asyncio.run(run())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["collect", "collect_search", "holdout", "baseline"])
    p.add_argument("--policy", help="prior for collect_search")
    p.add_argument("--trainers", help="comma-separated trainer_ids to restrict collect/collect_search to")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--jsonl", default="data/teams/gauntlet_harvest/battles.jsonl")
    p.add_argument("--battles", type=int, default=20000)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--port", type=int, default=8021)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["sim", "server"], default="sim")
    p.add_argument("--out", help="holdout json path")
    args = p.parse_args()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "collect_search":
        collect_search(args)
    elif args.mode == "holdout":
        holdout(args)
    else:
        baseline(args)


if __name__ == "__main__":
    main()
