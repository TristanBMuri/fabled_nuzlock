"""Prototype BC collector for the gauntlet (game-team) distribution.

Samples (player_team, enemy_team) pairs from the gauntlet harvest jsonl —
real BDSP matchups, game levels/movesets — and plays them in
gen8customgamenogimmicks with SimpleHeuristics on the player side,
recording (obs, action_mask, action) per decision.

Two modes:
  collect  harvest BC pairs from sampled matchups
  baseline play SimpleHeuristics-vs-SimpleHeuristics on holdout pairs
           (the reference line the prototype policy is judged against)
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
from poke_env.environment import SinglesEnv
from poke_env.player import ForfeitBattleOrder, SimpleHeuristicsPlayer

GAME_FORMAT = "gen8customgamenogimmicks"


def sample_pairs(jsonl: str, n: int, seed: int) -> list:
    rows = []
    with open(jsonl) as f:
        rows = [json.loads(line) for line in f]
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
            mask = np.array(SinglesEnv.get_action_mask(battle))
            if mask.sum() > 0:
                order = super().choose_move(battle)
                try:
                    action = SinglesEnv.order_to_action(order, battle, fake=True, strict=False)
                except ValueError:
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


def collect(args: argparse.Namespace) -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"models/bc_game_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sample_pairs(args.jsonl, args.battles, args.seed)
    chunks = [pairs[i::args.workers] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=collect_worker, args=(w, chunks[w], args.port, out_dir))
        for w in range(args.workers)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    shards = sorted(out_dir.glob("shard_*.npz"))
    data = {k: np.concatenate([np.load(s)[k] for s in shards]) for k in ("obs", "mask", "action")}
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
    p.add_argument("mode", choices=["collect", "baseline"])
    p.add_argument("--jsonl", default="data/teams/gauntlet_harvest/battles.jsonl")
    p.add_argument("--battles", type=int, default=20000)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--port", type=int, default=8021)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.mode == "collect":
        collect(args)
    else:
        baseline(args)


if __name__ == "__main__":
    main()
