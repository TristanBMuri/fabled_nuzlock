"""Player search on the Rust engine, Showdown search as the fallback.

The battle itself always runs on Showdown (the RR fork, ~1 ms a turn); only the expensive
part, the search behind each decision, moves to poke-engine. A decision runs poke-engine's
MCTS (both sides search, full information: a pessimistic model of the scripted enemy AI)
when the position converts cleanly (rl/convert.py: every name known, no mega pending) and
no known gap is on either active mon (GAPS); otherwise the existing Showdown rollout search
(ai_util.evaluate.Searcher, the fallback) decides, exactly as before.

    from rl.engine_search import EngineSearcher
    brain = EngineSearcher(Searcher(enemy, k=9, rollouts=8, rollout=calc), iterations=20000)

`stats` counts engine / fallback decisions and why each fallback happened. With `log` set to
a path, every engine decision appends one JSON line (the decision view and phase 4's data):
`tag` (who set it: boss, team, battle, seed), turn, the engine state string, each option's
visits and mean value, the pick, and the RR AI model's predicted enemy options.
"""

import json
import os
from collections import Counter

from rl import convert
from rl.route import GAPS, turn_gaps

# what the RR build still gets wrong, on top of route.GAPS (rl/README.md -> Radical Red)
RR_GAPS = {}


class EngineSearcher:
    def __init__(self, fallback, iterations: int = 20000, rr: bool = True, death: float = 60.0,
                 opponent: str = "ai", p_model: float = 0.8):
        # opponent: "minimax" = both sides search (MCTS); "ai" = the enemy plays the RR AI in Rust
        # (poke-engine rr_ai.rs; agreement with ai_util/rr_ai.py 97% off-policy, 96% on-policy since 2026-11-24), so the
        # search is against the boss's script. Only when the searcher is p1 (side one).
        # death: evaluation points per fainted mon of ours (the engine reads POKE_ENGINE_DEATH once,
        # at its first evaluation). Misty, 10 boxes, 2026-11-23: 0 -> 53% / 0.73 d/mon, 60 -> 52% /
        # 0.66 (the Showdown search: 52% / 0.69), 200 -> nearly every box lost
        os.environ.setdefault("POKE_ENGINE_DEATH", str(death))
        if rr and not convert.RR:
            convert.use_rr()
        # p_model (opponent "ai"): per enemy decision in the tree, follow the AI model with this
        # probability, else UCB over every enemy option (phase 1's mixture; 1.0 = pure model).
        # Default ai / 0.8 since D4 (2026-11-24): +11 points, -0.13 deaths/mon vs minimax on fresh seeds
        self.fallback, self.iterations, self.opponent, self.p_model = fallback, iterations, opponent, p_model
        self.gaps = {**GAPS, **RR_GAPS}
        self.stats = Counter()
        self.last_scores: dict[str, float] = {}
        self.log, self.tag = None, {}

    # the probes seed and read the searcher's attributes (rng, counters): forward them
    def __getattr__(self, name):
        return getattr(self.fallback, name)

    def __setattr__(self, name, value):
        if name == "rng":
            setattr(self.fallback, name, value)
        else:
            object.__setattr__(self, name, value)

    def why_not(self, dump: dict) -> list[str]:
        _, issues = convert.to_engine(dump)
        gaps = [g for g in turn_gaps(dump) if g in self.gaps] + \
               [f for f in _active_features(dump) if f in RR_GAPS]
        return sorted(set(issues) | set(gaps))

    def __call__(self, sb, side: str) -> str:
        if sb.mirrors[side].wait:
            return "default"
        dump = sb.bridge.call(op="dump", id=sb.id)
        mine = dump[side]["choices"]
        if len(mine) <= 1:  # nothing to search
            self.stats["single"] += 1
            return mine[0][0] if mine else self.fallback(sb, side)
        why = self.why_not(dump)
        if why:
            self.stats["fallback"] += 1
            self.stats.update(f"why:{w}" for w in why)
            return self.fallback(sb, side)
        state, _ = convert.to_engine(dump)
        if side == "p2":  # poke-engine's side one is the searcher
            state = convert.pe.State(side_one=state.side_two, side_two=state.side_one, weather=state.weather,
                                     weather_turns_remaining=state.weather_turns_remaining, terrain=state.terrain,
                                     terrain_turns_remaining=state.terrain_turns_remaining,
                                     trick_room=state.trick_room,
                                     trick_room_turns_remaining=state.trick_room_turns_remaining)
        # seeded from the battle's searcher rng, so a seeded battle replays exactly
        seed = int(self.fallback.rng.integers(1, 2 ** 62)) if hasattr(self.fallback, "rng") else None
        if self.opponent == "ai" and side == "p1":
            res = convert.pe.monte_carlo_tree_search_vs_ai(state, 60_000, self.iterations, seed=seed,
                                                           p_model=self.p_model).side_one
        else:
            res = convert.pe.monte_carlo_tree_search(state, 60_000, self.iterations, 1, seed=seed).side_one
        by_id = {convert.engine_choice(dump, side, c): c[0] for c in mine}
        # a stone holder is offered only `<move>-mega` (the RR build megas on the first move)
        scored = [(r.visits, r.total_score / max(r.visits, 1), r.move_choice.removeprefix("switch ").removesuffix("-mega"))
                  for r in res]
        self.last_scores = {by_id[c]: v for _, v, c in scored if c in by_id}
        legal = [x for x in scored if x[2] in by_id]
        if not legal:  # the engine's options and Showdown's request disagree: trust Showdown
            self.stats["fallback"] += 1
            self.stats["why:options"] += 1
            return self.fallback(sb, side)
        self.stats["engine"] += 1
        pick = by_id[max(legal)[2]]
        if self.log:
            rec = {"kind": "decision", **self.tag, "turn": sb.turn, "side": side, "seed": seed, "state": state.to_string(),
                   "options": [[c, n, round(v, 4)] for n, v, c in scored], "pick": pick,
                   "ai_pred": convert.pe.rr_ai_choice(state, "side_two") if hasattr(convert.pe, "rr_ai_choice") else None}
            with open(self.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return pick


class RustEnemy:
    """The RR AI's Rust port as the real enemy (phase 1's exact-model oracle): a search that
    models the enemy with the same port then has an exact model, so "the search method is wrong"
    and "the port is wrong" separate. Samples the AI's weighted choices (uniform over ties on a
    build without weights) with the global `random`, which the probes seed per battle; the
    Python AI decides where the position does not convert. Enemy side (p2) only."""

    def __init__(self, fallback):
        self.fallback, self.stats = fallback, Counter()

    def __call__(self, sb, side: str) -> str:
        import random
        if sb.mirrors[side].wait:
            return "default"
        dump = sb.bridge.call(op="dump", id=sb.id)
        mine = dump[side]["choices"]
        state, issues = convert.to_engine(dump) if len(mine) > 1 else (None, ["single"])
        if issues:
            self.stats["python"] += 1
            return self.fallback(sb, side)
        pe = convert.pe
        opts = (pe.rr_ai_choice_weighted(state, "side_two") if hasattr(pe, "rr_ai_choice_weighted")
                else [(c, 1.0) for c in pe.rr_ai_choice(state, "side_two")])
        by_id = {convert.engine_choice(dump, side, c): c[0] for c in mine}
        opts = [(by_id[k], w) for c, w in opts if (k := c.removeprefix("switch ").removesuffix("-mega")) in by_id]
        if not opts:
            self.stats["python:options"] += 1
            return self.fallback(sb, side)
        self.stats["rust"] += 1
        return random.choices([c for c, _ in opts], weights=[w for _, w in opts])[0]


def _active_features(dump: dict) -> set[str]:
    out = set()
    for s in ("p1", "p2"):
        a = dump[s]["pokemon"][0]
        out |= {f"ability:{a['ability']}", f"item:{a['item']}"} | {f"move:{m}" for m, _ in a["moves"]}
    return out
