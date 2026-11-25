"""Phase 4 diagnostics: why does the fitted leaf evaluation not help the search?

  part A (data):   python diag_p4.py data
  part B (search): python diag_p4.py search <hand|fitted> <out.jsonl>     (env set by the caller)
  part C (moves):  python diag_p4.py moves <hand.jsonl> <fitted.jsonl> <out.json>
"""
import json
import os
import random
import sys
from collections import defaultdict

ROOT = "/home/tristan/CodingProjects/nuzlock_rl"
sys.path[:0] = [ROOT, ROOT + "/scripts", ROOT + "/rl"]
os.chdir(ROOT)
import numpy as np  # noqa: E402
from rl import convert  # noqa: E402

STATES = "logs/p4_data.states.jsonl"
W = [float(x) for x in open("data/rr/eval_weights_v1.txt").read().split()]


def sample_positions(n=200, seed=7):
    """n decision positions, stratified by boss, from battles that are neither trivial nor hopeless."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for line in open(STATES):
        r = json.loads(line)
        if r["kind"] == "decision" and rng.random() < 0.05:
            by[r["boss"]].append({"boss": r["boss"], "turn": r["turn"], "state": r["state"]})
    out = []
    for b, rows in sorted(by.items()):
        rng.shuffle(rows)
        out += rows[: n // len(by)]
    return out


def part_data():
    import fit_eval as fe
    rows = fe.load(STATES, 1)
    pe = convert.pe
    names = pe.evaluate_feature_names()
    x = fe.features(rows)
    y = np.array([u for *_, u in rows])
    logit = W[0] + x @ np.array(W[1:])
    p = 1 / (1 + np.exp(-logit))
    held = np.array([k[0] in fe.HELD_OUT for k, *_ in rows])
    boss = np.array([k[0] for k, *_ in rows])
    bat = [k for k, *_ in rows]
    ll = lambda pp, yy: float(-(yy * np.log(np.clip(pp, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1))).mean())
    res = {"n": len(rows)}
    # A1: how much is boss identity? per-boss constant (in-sample, train bosses) vs the linear fit
    bc = {b: y[(boss == b)].mean() for b in set(boss)}
    pb = np.array([bc[b] for b in boss])
    res["logloss"] = {"constant": ll(np.full_like(y[~held], y[~held].mean()), y[~held]), "per_boss_constant": ll(pb[~held], y[~held]),
                      "linear": ll(p[~held], y[~held]), "linear_held": ll(p[held], y[held]),
                      "constant_held": ll(np.full_like(y[held], y[~held].mean()), y[held])}
    # A2: calibration by decile of the prediction (train bosses and held-out bosses)
    cal = {}
    for tag, m in (("train_bosses", ~held), ("held_out", held)):
        q = np.quantile(p[m], np.linspace(0, 1, 11))
        cal[tag] = [[float(p[m][(p[m] >= a) & (p[m] <= b)].mean()), float(y[m][(p[m] >= a) & (p[m] <= b)].mean())]
                    for a, b in zip(q[:-1], q[1:])]
    res["calibration"] = cal
    # A3: variance of the fitted logit between battles vs within a battle (what the search compares)
    g = defaultdict(list)
    for k, v in zip(bat, logit):
        g[k].append(v)
    means = {k: np.mean(v) for k, v in g.items()}
    within = np.mean([np.var(v) for v in g.values() if len(v) > 1])
    between = np.var(list(means.values()))
    res["logit_variance"] = {"between_battles": float(between), "within_battle": float(within)}
    # A4: per-boss mean prediction vs mean outcome
    res["per_boss"] = {b: [float(p[boss == b].mean()), float(y[boss == b].mean()), int((boss == b).sum())] for b in sorted(set(boss))}
    # A5: one-turn change of the logit vs the turn's actual change in our HP / their HP (sensitivity)
    names_i = {n: i for i, n in enumerate(names)}
    res["weights"] = {n: W[1 + i] for n, i in names_i.items()}
    json.dump(res, open("logs/p4_diag_data.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("weights", "calibration")}, indent=1))


def part_search(tag, out):
    convert.use_rr()
    pe = convert.pe
    pos = sample_positions()
    with open(out, "w") as f:
        for i, r in enumerate(pos):
            st = pe.State.from_string(r["state"])
            res = pe.monte_carlo_tree_search_vs_ai(st, 60_000, 20000, seed=1, p_model=0.8).side_one
            opts = sorted(((x.visits, x.total_score / max(x.visits, 1), x.move_choice) for x in res), reverse=True)
            v = np.array([o[0] for o in opts], dtype=float)
            pv = v / v.sum()
            f.write(json.dumps({"i": i, "boss": r["boss"], "turn": r["turn"], "pick": opts[0][2],
                                "options": [[c, int(n), round(m, 4)] for n, m, c in opts],
                                "entropy": float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum()),
                                "value_spread": float(max(o[1] for o in opts if o[0] >= 100) - min(o[1] for o in opts if o[0] >= 100))
                                if sum(o[0] >= 100 for o in opts) > 1 else 0.0}) + "\n")
    print("done", tag, len(pos))


def _move_value(args):
    """Mean U of one first move from a state, then the hand policy (2k iterations) to the end."""
    import line as L
    state, move, n, seed = args
    convert.use_rr()
    pe = convert.pe
    ln = L.Line(iterations=2000)
    rng = random.Random(seed)
    us = []
    for _ in range(n):
        st = pe.State.from_string(state)
        opts = ln.enemy(st)
        e = rng.choices([c for c, _ in opts], weights=[w for _, w in opts])[0]
        outs = pe.generate_instructions(st, L._name(move), L._name(e))
        st = st.apply_instructions(rng.choices(outs, weights=[o.percentage for o in outs])[0])
        won, dead = ln.playout(st.to_string(), rng)
        us.append(won * (1 - 0.5 * sum(dead) / len(dead)))
    return float(np.mean(us))


def part_moves(hand, fitted, out, n=40):
    import multiprocessing as mp
    pos = sample_positions()
    h = {json.loads(x)["i"]: json.loads(x) for x in open(hand)}
    f = {json.loads(x)["i"]: json.loads(x) for x in open(fitted)}
    dis = [i for i in h if h[i]["pick"] != f[i]["pick"]][:60]
    jobs = [(pos[i]["state"], h[i]["pick"], n, 100 + i) for i in dis] + [(pos[i]["state"], f[i]["pick"], n, 100 + i) for i in dis]
    with mp.get_context("spawn").Pool(10) as pool:
        vals = pool.map(_move_value, jobs)
    k = len(dis)
    rows = [{"i": i, "boss": h[i]["boss"], "hand_pick": h[i]["pick"], "fitted_pick": f[i]["pick"],
             "hand_value": vals[j], "fitted_value": vals[k + j]} for j, i in enumerate(dis)]
    d = np.array([r["fitted_value"] - r["hand_value"] for r in rows])
    summary = {"positions": len(h), "agree": sum(h[i]["pick"] == f[i]["pick"] for i in h) / len(h), "disagreements_scored": k,
               "mean_U_diff_fitted_minus_hand": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(k)),
               "fitted_better": int((d > 0.02).sum()), "hand_better": int((d < -0.02).sum()),
               "entropy": {"hand": float(np.mean([x["entropy"] for x in h.values()])), "fitted": float(np.mean([x["entropy"] for x in f.values()]))},
               "value_spread": {"hand": float(np.mean([x["value_spread"] for x in h.values()])), "fitted": float(np.mean([x["value_spread"] for x in f.values()]))}}
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        convert.use_rr()
        part_data()
    elif cmd == "search":
        part_search(sys.argv[2], sys.argv[3])
    elif cmd == "moves":
        part_moves(sys.argv[2], sys.argv[3], sys.argv[4])
