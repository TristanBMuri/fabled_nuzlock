"""Scratch: does the value head rank candidate leaves like full rollouts do?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from poke_env.environment import SinglesEnv

from ai_util.battler_policy import load_policy
from ai_util.evaluate import (GAME_FORMAT, Searcher, heuristic_brain, outcome_value, prior_probs,
                              prior_value)
from ai_util.sim import SimBattle, SimBridge, order_to_choice

policy = load_policy(sys.argv[1] if len(sys.argv) > 1 else "models/game_policy_cap_value.pt", device="cpu")
bridge, enemy = SimBridge(), heuristic_brain()
pairs = json.loads(Path("models/gauntlet_holdout.json").read_text())[:40]
s = Searcher(policy, enemy, k=3, rollouts=8)
agree, n, v_track = 0, 0, []
for party, foe in pairs:
    sb = SimBattle(bridge, GAME_FORMAT, party, foe)
    while not sb.ended and sb.turn < 100:
        m = sb.mirrors["p1"]
        if sb.pending["p1"] and not m.wait:
            v_track.append((sb.turn, prior_value(policy, m)))
            probs, mask = prior_probs(policy, m)
            cands = [int(a) for a in np.argsort(-probs) if mask[a]][:3]
            if len(cands) > 1:
                seeds = [s.rng.integers(1, 2 ** 16, 4).tolist() for _ in range(8)]
                mc, vh = [], []
                for a in cands:
                    msg = order_to_choice(SinglesEnv.action_to_order(np.int64(a), m))
                    mcs, vhs = [], []
                    for seed in seeds:
                        f = sb.fork(seed=seed)
                        first = {"p1": msg}
                        if f.pending["p2"]:
                            first["p2"] = enemy(f, "p2")
                        f.step(first)
                        while not f.ended and f.mirrors["p1"].wait and f.pending["p2"]:
                            f.step({"p2": f.choice("p2", enemy)})
                        leaf = outcome_value(f.state(), "p1") if f.ended else prior_value(policy, f.mirrors["p1"])
                        vhs.append(leaf)
                        f.play({"p1": s.prior, "p2": enemy}, 100)
                        mcs.append(outcome_value(f.state(), "p1"))
                        f.close()
                    mc.append(np.mean(mcs)); vh.append(np.mean(vhs))
                n += 1
                agree += int(np.argmax(mc) == np.argmax(vh))
                if n <= 12:
                    print(f"turn {sb.turn} cands {cands} mc {np.round(mc, 2)} vhead {np.round(vh, 2)}")
        choices = {}
        if sb.pending["p1"]:
            choices["p1"] = s.prior(sb, "p1")
        if sb.pending["p2"]:
            choices["p2"] = enemy(sb, "p2")
        sb.step(choices)
    final = outcome_value(sb.state(), "p1")
    sb.close()
print(f"argmax agreement value-head vs rollouts: {agree}/{n}")
vt = np.array(v_track)
print(f"value along battles: mean {vt[:, 1].mean():.2f}  turn-1 mean {vt[vt[:, 0] == 1, 1].mean():.2f}")
bridge.close()
