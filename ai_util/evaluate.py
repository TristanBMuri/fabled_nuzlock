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
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

GAME_FORMAT = "gen8customgamenogimmicks"
_BASELINES = {"heuristic": SimpleHeuristicsPlayer, "maxbp": MaxBasePowerPlayer, "random": RandomPlayer}


def heuristic_brain(kind: str = "heuristic", fmt: str = GAME_FORMAT) -> Brain:
    return player_brain(_BASELINES[kind](battle_format=fmt, start_listening=False, log_level=50))


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


def prior_value(policy, mirror) -> float:
    """The policy's value head: expected outcome_value from this state (only
    meaningful for checkpoints trained with pretrain_bc --value)."""
    with torch.no_grad():
        return float(policy.predict_values(_obs_dict(policy, mirror))[0, 0])


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
    (a scripted brain such as the calc bot) plays the rollouts — no network."""

    def __init__(self, policy, enemy: Brain, k: int = 3, rollouts: int = 8,
                 max_turns: int = 100, importance: list[float] | None = None,
                 death_weight: float = 0.5, confident: float | None = None,
                 crn: bool = True, seed: int | None = None, horizon: int | None = None,
                 depth: int = 1, r2: int = 4, rollout: Brain | None = None):
        self.policy, self.enemy = policy, enemy
        # depth=2: at our next decision inside each rollout expand the prior's
        # top-k again (r2 rollouts each, common seeds) and take the best, so the
        # value is a max over our reply too, not what the prior would have done.
        self.depth, self.r2 = depth, r2
        self.prior = rollout or prior_brain(policy)
        self.k, self.rollouts, self.max_turns = k, rollouts, max_turns
        self.importance, self.death_weight, self.confident = importance, death_weight, confident
        # Common random numbers: rollout j uses the same engine seed for every
        # candidate, so candidates differ only by the action, not by luck.
        self.crn, self.rng = crn, np.random.default_rng(seed)
        # horizon=None: play rollouts to the end. horizon=h: play h more turns,
        # then score the leaf with the policy's value head (0 = one step + value).
        self.horizon = horizon
        self.n_decisions = self.n_searched = 0
        self.last_action: int | None = None  # chosen action index (for expert-iteration labels)
        self.searched_last = False
        # set to a list to record (leaf obs, mask, -1, rollout outcome) per rollout:
        # on-policy training data for the value head (collect_game collect_search)
        self.leaf_record: list | None = None

    def value(self, sb: SimBattle, side: str) -> float:
        if self.horizon is not None and not sb.ended:
            return prior_value(self.policy, sb.mirrors[side])
        return outcome_value(sb.state(), side, self.death_weight, self.importance)

    def _settle(self, f: SimBattle, side: str, opp: str) -> None:
        """The value head is trained on our decision states: if the leaf is the
        enemy's forced switch (we just KO'd), let it happen first."""
        while not f.ended and f.mirrors[side].wait and f.pending[opp]:
            f.step({opp: f.choice(opp, self.enemy)})

    def _play_out(self, f: SimBattle, side: str, opp: str, depth: int) -> float:
        """Value of fork f: play out with the prior, or (depth > 1) max over
        our top-k at the next decision, each scored by r2 play-outs."""
        if depth > 1:
            self._settle(f, side, opp)
            m = f.mirrors[side]
            if not f.ended and f.pending[side] and not m.wait:
                probs, mask = prior_probs(self.policy, m)
                cands = [int(a) for a in np.argsort(-probs) if mask[a]][:self.k]
                if len(cands) > 1:
                    seeds = [self.rng.integers(1, 2 ** 16, 4).tolist() for _ in range(self.r2)]
                    best = -np.inf
                    for b in cands:
                        msg = order_to_choice(SinglesEnv.action_to_order(np.int64(b), m))
                        vals = []
                        for seed in seeds:
                            g = f.fork(seed=seed)
                            first = {side: msg}
                            if g.pending[opp]:
                                first[opp] = self.enemy(g, opp)
                            g.step(first)
                            vals.append(self._play_out(g, side, opp, depth - 1))
                            g.close()
                        best = max(best, float(np.mean(vals)))
                    return best
        f.play({side: self.prior, opp: self.enemy}, self.max_turns)
        self._settle(f, side, opp)
        return self.value(f, side)

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
        if len(cands) <= 1 or (self.confident is not None and probs[cands[0]] >= self.confident):
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
                if self.depth > 1 and self.horizon is None:
                    vals.append(self._play_out(f, side, opp, self.depth))
                    f.close()
                    continue
                leaf = None
                if self.leaf_record is not None and self.horizon is None:
                    self._settle(f, side, opp)
                    if not f.ended:
                        leaf = (embed_battle(f.mirrors[side]).astype(np.float16),
                                action_mask(f.mirrors[side]).astype(np.int8), np.int16(-1))
                until = None if self.horizon is None else f.turn + self.horizon
                f.play({side: self.prior, opp: self.enemy}, self.max_turns, until=until)
                self._settle(f, side, opp)
                vals.append(self.value(f, side))
                if leaf is not None:
                    self.leaf_record.append(leaf + (vals[-1],))
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
    p.add_argument("--no-crn", action="store_true",
                   help="fresh RNG per rollout instead of common random numbers across candidates")
    p.add_argument("--horizon", type=int,
                   help="truncate rollouts after this many turns and score the leaf with the "
                        "policy's value head (needs a --value-trained checkpoint); default: play out")
    p.add_argument("--depth", type=int, default=1, help="2 = max over our reply inside rollouts")
    p.add_argument("--r2", type=int, default=4, help="play-outs per second-ply candidate")
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
        if args.policy == "calc":
            from ai_util.calc_bot import calc_brain
            policy, fast, tag = None, calc_brain(), "calc"
        else:
            policy = load_policy(args.policy, [int(a) for a in args.net_arch.split(",")], device="cpu")
            fast, tag = prior_brain(policy), "prior"
        if args.search:
            searcher = Searcher(policy, enemy, k=args.k, rollouts=args.rollouts,
                                max_turns=args.max_turns, death_weight=args.death_weight,
                                confident=args.confident, crn=not args.no_crn, seed=args.seed,
                                horizon=args.horizon, depth=args.depth, r2=args.r2,
                                rollout=fast if policy is None else None)
            tag = (f"{tag}+search k={args.k} r={args.rollouts}{' nocrn' if args.no_crn else ''}"
                   f"{'' if args.depth == 1 else f' d={args.depth} r2={args.r2}'}"
                   f"{'' if args.horizon is None else f' h={args.horizon}'}")
            brain = searcher
        else:
            brain = fast

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
    print(f"totals: pairs {len(pairs)} n {args.n} wins {wins:g} deaths {deaths:g} mons {mons}")
    bridge.close()


if __name__ == "__main__":
    main()
