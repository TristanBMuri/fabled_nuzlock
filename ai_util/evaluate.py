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
from ai_util.battler_policy import PolicyPlayer, action_mask, load_policy
from ai_util.sim import Brain, SimBattle, SimBridge, order_to_choice, player_brain
from poke_env.environment import SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer

GAME_FORMAT = "gen8customgamenogimmicks"


def heuristic_brain() -> Brain:
    """SimpleHeuristics as a sim Brain: the enemy AI, and the default player."""
    return player_brain(SimpleHeuristicsPlayer(battle_format=GAME_FORMAT, start_listening=False, log_level=50))


def prior_brain(policy) -> Brain:
    return player_brain(PolicyPlayer(policy, deterministic=True, battle_format=GAME_FORMAT,
                                     start_listening=False, log_level=50))


def _obs_dict(policy, mirror) -> dict:
    return {"observation": torch.as_tensor(embed_battle(mirror), device=policy.device).unsqueeze(0),
            "action_mask": torch.as_tensor(action_mask(mirror), device=policy.device).unsqueeze(0)}


def prior_probs(policy, mirror) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        od = _obs_dict(policy, mirror)
        policy._mask = od["action_mask"]
        probs = policy.get_distribution(od).distribution.probs[0].cpu().numpy()
    return probs, od["action_mask"][0].cpu().numpy()


def outcome_value(state: dict, side: str, death_weight: float = 0.5,
                  importance: list[float] | None = None) -> float:
    """win (+1/-1) minus death_weight * importance-weighted fainted fraction of
    `side`'s team; unresolved battles score an hp-lead proxy. The Searcher's
    objective and the value head's training target."""
    opp = "p2" if side == "p1" else "p1"
    mine, theirs = state[side], state[opp]
    imp = importance or [1.0 / len(mine)] * len(mine)
    deaths = sum(w * m["fainted"] for w, m in zip(imp, mine))
    if not state["ended"]:
        hp = lambda team: np.mean([m["hp"] / max(m["maxhp"], 1) for m in team])
        return 0.5 * (hp(mine) - hp(theirs)) - death_weight * deaths
    return (1.0 if state["winner"] == side else -1.0) - death_weight * deaths


class Searcher:
    """Top-k rollout search for one side; the enemy plays its fixed AI.
    policy=None: every legal action is a candidate (up to k) and `rollout`
    (a scripted brain such as the calc bot) plays the rollouts — no network.
    Rollouts play to the end and the outcome is the value: a learned leaf
    value (twice) and a second ply (depth 2) were null (NOTEBOOK.md)."""

    def __init__(self, policy, enemy: Brain, k: int = 3, rollouts: int = 8,
                 max_turns: int = 100, importance: list[float] | None = None,
                 death_weight: float = 0.5, crn: bool = True, seed: int | None = None,
                 rollout: Brain | None = None):
        self.policy, self.enemy = policy, enemy
        self.prior = rollout or prior_brain(policy)
        self.k, self.rollouts, self.max_turns = k, rollouts, max_turns
        self.importance, self.death_weight = importance, death_weight
        # Common random numbers: rollout j uses the same engine seed for every
        # candidate, so candidates differ only by the action, not by luck.
        self.crn, self.rng = crn, np.random.default_rng(seed)
        self.n_decisions = self.n_searched = 0
        self.last_action: int | None = None  # chosen action index
        self.searched_last = False

    def value(self, sb: SimBattle, side: str) -> float:
        return outcome_value(sb.state(), side, self.death_weight, self.importance)

    def _settle(self, f: SimBattle, side: str, opp: str) -> None:
        """If the leaf is the enemy's forced switch (we just KO'd), let it happen first."""
        while not f.ended and f.mirrors[side].wait and f.pending[opp]:
            f.step({opp: f.choice(opp, self.enemy)})

    def __call__(self, sb: SimBattle, side: str) -> str:
        m = sb.mirrors[side]
        self.last_action, self.searched_last = None, False
        if m.wait:
            return "default"
        self.n_decisions += 1
        if self.policy is None:
            mask = action_mask(m)
            probs = mask.astype(float) / max(mask.sum(), 1)
            cands = [int(a) for a in range(len(mask)) if mask[a]]
            # the rollout brain's own choice first: ties in the noisy rollout values
            # go to it, not to whichever switch happens to come first (which stalled)
            own = self.prior(sb, side)
            cands.sort(key=lambda a: order_to_choice(SinglesEnv.action_to_order(np.int64(a), m)) != own)
            cands = cands[:self.k]
        else:
            probs, mask = prior_probs(self.policy, m)
            cands = [int(a) for a in np.argsort(-probs) if mask[a]][:self.k]
        if not cands:
            return "default"
        self.last_action = cands[0]
        if len(cands) <= 1:
            return order_to_choice(SinglesEnv.action_to_order(np.int64(cands[0]), m))
        self.n_searched += 1
        self.searched_last = True
        opp = "p2" if side == "p1" else "p1"
        seeds = [self.rng.integers(1, 2 ** 16, 4).tolist() if self.crn else None
                 for _ in range(self.rollouts)]
        best, best_score = None, -np.inf
        for a in cands:
            msg = order_to_choice(SinglesEnv.action_to_order(np.int64(a), m))
            vals = []
            for seed in seeds:
                f = sb.fork(seed=seed)
                first = {side: msg}
                if f.pending[opp]:
                    first[opp] = self.enemy(f, opp)
                f.step(first)
                f.play({side: self.prior, opp: self.enemy}, self.max_turns)
                self._settle(f, side, opp)
                vals.append(self.value(f, side))
                f.close()
            score = float(np.mean(vals))
            if score > best_score:
                best, best_score, self.last_action = msg, score, a
        return best


def evaluate(bridge: SimBridge, party: str | None, enemy: str | None, brain: Brain,
             enemy_brain: Brain, n: int = 1, importance: list[float] | None = None,
             max_turns: int = 100, fmt: str = GAME_FORMAT, dedupe: bool = True,
             carry: list | None = None) -> dict:
    """Run n fresh-seed battles of `party` (p1, driven by brain) vs `enemy`.
    Returns P(win), P(death) per party mon (packed order), importance-weighted
    risk, and hp = mean fraction of the party's HP left at the end (margin:
    separates parties that all show P(death)=0 at small n)."""
    wins, deaths, turns, hp = 0, None, 0, 0.0
    stats = {}
    for _ in range(n):
        sb = SimBattle(bridge, fmt, party, enemy, dedupe=dedupe, carry=carry)
        sb.play({"p1": brain, "p2": enemy_brain}, max_turns)
        st = sb.state()
        won, dead = st["winner"] == "p1", [m["fainted"] for m in st["p1"]]
        hp += float(np.mean([m["hp"] / max(m["maxhp"], 1) for m in st["p1"]]))
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
            "hp": hp / n, "turns": turns / n, "n": n, "stats": stats}


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
    p.add_argument("--policy", help="state_dict of the prior, or 'calc' for the damage-calc bot; omit for --check")
    p.add_argument("--net-arch", default="256,256")
    p.add_argument("--check", action="store_true", help="SimpleHeuristics on both sides")
    p.add_argument("--search", action="store_true")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--death-weight", type=float, default=0.5)
    p.add_argument("--no-crn", action="store_true",
                   help="fresh RNG per rollout instead of common random numbers across candidates")
    p.add_argument("--max-turns", type=int, default=100)
    args = p.parse_args()

    if args.from_harvest:
        from ai_util.collect_game import sample_pairs
        pairs = sample_pairs(args.from_harvest, args.n_pairs, args.seed)
    else:
        pairs = json.loads(Path(args.pairs).read_text())[args.offset:args.offset + args.n_pairs]
    bridge = SimBridge()
    enemy = heuristic_brain()
    searcher = None
    if args.check:
        brain, tag = heuristic_brain(), "heuristic"
    else:
        if args.policy == "calc":
            from ai_util.calc_bot import calc_brain
            policy, fast, tag = None, calc_brain(), "calc"
        else:
            policy = load_policy(args.policy, [int(a) for a in args.net_arch.split(",")], device="cpu")
            fast, tag = prior_brain(policy), "prior"
        if args.search:
            searcher = Searcher(policy, enemy, k=args.k, rollouts=args.rollouts,
                                max_turns=args.max_turns, death_weight=args.death_weight,
                                crn=not args.no_crn, seed=args.seed,
                                rollout=fast if policy is None else None)
            tag = f"{tag}+search k={args.k} r={args.rollouts}{' nocrn' if args.no_crn else ''}"
            brain = searcher
        else:
            brain = fast

    t0, wins, deaths, mons, turns, stats = time.time(), 0.0, 0.0, 0, 0.0, {}
    for i, (party, enemy_team) in enumerate(pairs):
        r = evaluate(bridge, party, enemy_team, brain, enemy, n=args.n, max_turns=args.max_turns)
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
    print(f"totals: pairs {len(pairs)} n {args.n} wins {wins:g} deaths {deaths:g} mons {mons}")
    bridge.close()


if __name__ == "__main__":
    main()
