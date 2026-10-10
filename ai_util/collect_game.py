"""Prototype BC collector for the gauntlet (game-team) distribution.

Samples (player_team, enemy_team) pairs from the gauntlet harvest jsonl —
real BDSP matchups, game levels/movesets — and plays them in
gen8customgamenogimmicks with SimpleHeuristics on the player side,
recording (obs, action_mask, action) per decision.

Modes:
  collect  harvest BC pairs from sampled matchups in the in-process sim
  holdout  write a json of held-out (player, enemy) pairs for pretrain --eval-pairs
  (the heuristic-vs-heuristic reference line is `evaluate.py --check --from-harvest`)
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


def recording_brain(record: list, ctx: dict, teacher: str = "heuristic"):
    """The teacher (SimpleHeuristics or the calc bot) as a sim Brain that records
    (obs, mask, action) per decision."""
    from ai_util.sim import order_to_choice
    if teacher == "calc":
        from ai_util.calc_bot import CalcPlayer
        teacher = CalcPlayer(battle_format=GAME_FORMAT, start_listening=False, log_level=50)
    else:
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


def collect_worker_sim(worker: int, pairs: list, out_dir: Path, teacher: str = "heuristic") -> None:
    from ai_util.evaluate import heuristic_brain, outcome_value
    from ai_util.sim import SimBattle, SimBridge

    record, values, ctx, failures = [], [], {"n": 0, "switch": 0}, 0
    bridge, me, foe = SimBridge(), None, heuristic_brain()
    me = recording_brain(record, ctx, teacher)
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


def synth(args: argparse.Namespace) -> None:
    """Synthetic matchups for targeted teacher data: late-game player parties
    from the harvest (fought at gauntlet order >= --min-order) x the teams of
    --trainers (the deadliest fights). Writes a harvest-shaped jsonl for
    `collect --jsonl`. This is how battles_hard_synth.jsonl was made."""
    sys.path.insert(0, str(_here.parent / "scripts"))
    from gauntlet import load_trainers, pack_team, trainer_mons
    trainers, order = load_trainers()
    late = {e["trainer_id"] for e in order[args.min_order:]}
    with open(args.jsonl) as f:
        parties = sorted({r["player_team"] for r in map(json.loads, f)
                          if r.get("trainer_id") in late})
    rng = np.random.default_rng(args.seed)
    parties = [parties[i] for i in rng.choice(len(parties), min(args.parties, len(parties)),
                                              replace=False)]
    ids = [int(t) for t in args.trainers.split(",")]
    with open(args.out, "w") as f:
        for tid in ids:
            enemy = pack_team(trainer_mons(trainers[tid]))
            for p in parties:
                f.write(json.dumps({"run_id": "synth", "trainer_id": tid, "info": "full",
                                    "won": None, "player_team": p, "enemy_team": enemy}) + "\n")
    print(f"wrote {len(parties) * len(ids)} matchups ({len(parties)} parties x {len(ids)} trainers) -> {args.out}")


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
    procs = [ctx.Process(target=collect_worker_sim, args=(w, chunks[w], out_dir, args.teacher))
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["collect", "synth", "holdout"])
    p.add_argument("--teacher", choices=["heuristic", "calc"], default="heuristic",
                   help="collect: who labels the decisions (SimpleHeuristics or ai_util.calc_bot)")
    p.add_argument("--trainers", help="comma-separated trainer_ids to restrict collect to, "
                                      "or the enemy teams for synth")
    p.add_argument("--parties", type=int, default=400, help="synth: player parties sampled")
    p.add_argument("--min-order", type=int, default=120,
                   help="synth: parties come from fights at this gauntlet order or later")
    p.add_argument("--jsonl", default="data/teams/gauntlet_harvest/battles.jsonl")
    p.add_argument("--battles", type=int, default=20000)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="holdout json path")
    args = p.parse_args()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "synth":
        synth(args)
    else:
        holdout(args)


if __name__ == "__main__":
    main()
