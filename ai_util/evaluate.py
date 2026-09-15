"""Line evaluator + top-k rollout search over the BC prior (roadmap step 1).

evaluate(): P(win) and P(death) per party mon for one (party, enemy) matchup
under a brain, by Monte Carlo over fresh-seed battles in the in-process sim.
This is the interface the run layer queries.

Searcher: at each decision expand the prior's top-k actions, play each out
`rollouts` times from a reseeded fork (prior vs enemy AI), pick the best
risk-adjusted mean. The compute dial: k=1 is the prior, bigger is a boss fight.

CLI (no Showdown server needed):
  uv run python ai_util/evaluate.py --check                       # heuristic mirror (~68.5%)
  uv run python ai_util/evaluate.py --policy models/game_policy_best.pt          # prior (~70%)
  uv run python ai_util/evaluate.py --policy models/game_policy_best.pt --search  # go/no-go
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(_here.parent))

import argparse
import json
import time

import numpy as np
import torch

from ai_util.battler_obs import embed_battle
from ai_util.battler_policy import N_ACTIONS, PolicyPlayer, load_policy
from ai_util.sim import Brain, SimBattle, SimBridge, order_to_choice, player_brain
from poke_env.environment import SinglesEnv
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

GAME_FORMAT = "gen8customgamenogimmicks"
_BASELINES = {"heuristic": SimpleHeuristicsPlayer, "maxbp": MaxBasePowerPlayer, "random": RandomPlayer}


def heuristic_brain(kind: str = "heuristic", fmt: str = GAME_FORMAT) -> Brain:
    return player_brain(_BASELINES[kind](battle_format=fmt, start_listening=False, log_level=50))


def prior_brain(policy) -> Brain:
    return player_brain(PolicyPlayer(policy, deterministic=True, battle_format=GAME_FORMAT,
                                     start_listening=False, log_level=50))


def prior_probs(policy, mirror) -> tuple[np.ndarray, np.ndarray]:
    obs = embed_battle(mirror)
    mask = np.array(SinglesEnv.get_action_mask(mirror))
    if len(mask) < N_ACTIONS:
        mask = np.pad(mask, (0, N_ACTIONS - len(mask)))
    with torch.no_grad():
        od = {"observation": torch.as_tensor(obs, device=policy.device).unsqueeze(0),
              "action_mask": torch.as_tensor(mask, device=policy.device).unsqueeze(0)}
        policy._mask = od["action_mask"]
        probs = policy.get_distribution(od).distribution.probs[0].cpu().numpy()
    return probs, mask


class Searcher:
    """Top-k rollout search for one side; the enemy plays its fixed AI."""

    def __init__(self, policy, enemy: Brain, k: int = 3, rollouts: int = 8,
                 max_turns: int = 100, importance: list[float] | None = None,
                 death_weight: float = 0.5, confident: float | None = None):
        self.policy, self.enemy = policy, enemy
        self.prior = prior_brain(policy)
        self.k, self.rollouts, self.max_turns = k, rollouts, max_turns
        self.importance, self.death_weight, self.confident = importance, death_weight, confident
        self.n_decisions = self.n_searched = 0

    def value(self, sb: SimBattle, side: str) -> float:
        st = sb.state()
        opp = "p2" if side == "p1" else "p1"
        mine, theirs = st[side], st[opp]
        imp = self.importance or [1.0 / len(mine)] * len(mine)
        deaths = sum(w * m["fainted"] for w, m in zip(imp, mine))
        if not st["ended"]:  # truncated by max_turns: hp lead as a weak proxy
            hp = lambda team: np.mean([m["hp"] / max(m["maxhp"], 1) for m in team])
            return 0.5 * (hp(mine) - hp(theirs)) - self.death_weight * deaths
        return (1.0 if st["winner"] == side else -1.0) - self.death_weight * deaths

    def __call__(self, sb: SimBattle, side: str) -> str:
        m = sb.mirrors[side]
        if m.wait:
            return "default"
        self.n_decisions += 1
        probs, mask = prior_probs(self.policy, m)
        cands = [int(a) for a in np.argsort(-probs) if mask[a]][:self.k]
        if len(cands) <= 1 or (self.confident is not None and probs[cands[0]] >= self.confident):
            return order_to_choice(SinglesEnv.action_to_order(np.int64(cands[0]), m))
        self.n_searched += 1
        opp = "p2" if side == "p1" else "p1"
        best, best_score = None, -np.inf
        for a in cands:
            msg = order_to_choice(SinglesEnv.action_to_order(np.int64(a), m))
            vals = []
            for _ in range(self.rollouts):
                f = sb.fork()
                first = {side: msg}
                if f.pending[opp]:
                    first[opp] = self.enemy(f, opp)
                f.step(first)
                f.play({side: self.prior, opp: self.enemy}, self.max_turns)
                vals.append(self.value(f, side))
                f.close()
            score = float(np.mean(vals))
            if score > best_score:
                best, best_score = msg, score
        return best


def evaluate(bridge: SimBridge, party: str | None, enemy: str | None, brain: Brain,
             enemy_brain: Brain, n: int = 1, importance: list[float] | None = None,
             max_turns: int = 100, fmt: str = GAME_FORMAT, dedupe: bool = True) -> dict:
    """Run n fresh-seed battles of `party` (p1, driven by brain) vs `enemy`.
    Returns P(win), P(death) per party mon (packed order), importance-weighted risk."""
    wins, deaths, turns = 0, None, 0
    stats = {}
    for _ in range(n):
        sb = SimBattle(bridge, fmt, party, enemy, dedupe=dedupe)
        sb.play({"p1": brain, "p2": enemy_brain}, max_turns)
        won, dead = sb.outcome()
        turns += sb.turn
        for k, v in sb.stats.items():
            stats[k] = stats.get(k, 0) + v
        for k, v in sb.why.items():
            stats["why:" + k] = stats.get("why:" + k, 0) + v
        sb.close()
        wins += won
        deaths = np.array(dead, dtype=float) if deaths is None else deaths + np.array(dead)
    p_death = (deaths / n).tolist()
    imp = importance or [1.0 / len(p_death)] * len(p_death)
    return {"p_win": wins / n, "p_death": p_death,
            "risk": float(sum(w * p for w, p in zip(imp, p_death))),
            "turns": turns / n, "n": n, "stats": stats}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default="models/gauntlet_holdout.json")
    p.add_argument("--from-harvest", metavar="JSONL",
                   help="sample pairs from the harvest jsonl like collect_game baseline "
                        "(seed 0) instead of --pairs; --check here compares to the 68.5%% line")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="battles per pair")
    p.add_argument("--policy", help="state_dict of the prior; omit for --check")
    p.add_argument("--net-arch", default="256,256")
    p.add_argument("--check", action="store_true", help="SimpleHeuristics on both sides")
    p.add_argument("--check-vs", choices=["maxbp", "random"],
                   help="mirror reference: SimpleHeuristics vs this baseline on generated "
                        "gen9randombattle teams (server numbers: maxbp ~88%%, random ~97%%)")
    p.add_argument("--search", action="store_true")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--death-weight", type=float, default=0.5)
    p.add_argument("--confident", type=float,
                   help="skip search when the prior's top prob >= this (default: never skip; "
                        "the converged BC prior is >0.9 almost everywhere, so 0.9 disabled search)")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--raw-names", action="store_true",
                   help="keep duplicate-species nicknames (reproduces the server/poke-env "
                        "team collapse; only for apples-to-apples checks vs old numbers)")
    args = p.parse_args()

    fmt = GAME_FORMAT
    if args.check_vs:
        fmt = "gen9randombattle"
        pairs = [(None, None)] * args.n_pairs
    elif args.from_harvest:
        from ai_util.collect_game import sample_pairs
        pairs = sample_pairs(args.from_harvest, args.n_pairs, args.seed)
    else:
        pairs = json.loads(Path(args.pairs).read_text())[args.offset:args.offset + args.n_pairs]
    bridge = SimBridge()
    enemy = heuristic_brain(args.check_vs or "heuristic", fmt)
    searcher = None
    if args.check_vs:
        brain, tag = heuristic_brain("heuristic", fmt), f"heuristic vs {args.check_vs}"
    elif args.check:
        brain, tag = heuristic_brain(), "heuristic"
    else:
        policy = load_policy(args.policy, [int(a) for a in args.net_arch.split(",")], device="cpu")
        if args.search:
            searcher = Searcher(policy, enemy, k=args.k, rollouts=args.rollouts,
                                max_turns=args.max_turns, death_weight=args.death_weight,
                                confident=args.confident)
            brain, tag = searcher, f"search k={args.k} r={args.rollouts}"
        else:
            brain, tag = prior_brain(policy), "prior"

    t0, wins, deaths, mons, turns, stats = time.time(), 0.0, 0.0, 0, 0.0, {}
    for i, (party, enemy_team) in enumerate(pairs):
        r = evaluate(bridge, party, enemy_team, brain, enemy, n=args.n,
                     max_turns=args.max_turns, fmt=fmt, dedupe=not args.raw_names)
        wins += r["p_win"]
        deaths += sum(r["p_death"])
        mons += len(r["p_death"])
        turns += r["turns"]
        for k, v in r["stats"].items():
            stats[k] = stats.get(k, 0) + v
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            el = time.time() - t0
            print(f"{tag}: {i + 1}/{len(pairs)} pairs  win {wins / (i + 1):.1%}  "
                  f"deaths/mon {deaths / max(mons, 1):.2f}  turns {turns / (i + 1):.1f}  "
                  f"{el / (i + 1):.2f}s/pair", flush=True)
    n_choices = max(stats.get("choices", 0), 1)
    print(f"choices {n_choices}: rejected by sim {stats.get('rejected', 0) / n_choices:.2%}, "
          f"forced default {stats.get('forced_default', 0) / n_choices:.2%}, "
          f"brain default {stats.get('brain_default', 0) / n_choices:.2%}")
    why = sorted(((v, k[4:]) for k, v in stats.items() if k.startswith("why:")), reverse=True)
    for v, k in why[:8]:
        print(f"  {v:6d}  {k}")
    if searcher:
        print(f"searched {searcher.n_searched}/{searcher.n_decisions} decisions")
    bridge.close()


if __name__ == "__main__":
    main()
