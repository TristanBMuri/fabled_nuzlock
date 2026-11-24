"""Phase 4: fit the search's leaf evaluation to battle outcomes (Texel tuning).

MCTS stops looking ahead at a leaf and scores the position there. Today that score is a
hand-written material count (poke-engine evaluate.rs). Here it is *fitted*: every logged
position (rr_team_probe.py --engine --states) is labelled with how its battle ended, as the
user's utility U = win * (1 - 0.5 * deaths/6), and a logistic regression learns weights w with
sigmoid(bias + w . features(position)) ~ U. This is a *value function fitted by regression*
(Monte Carlo targets: each label is one sampled outcome, unbiased but noisy). Chess engines
tune their evaluations this way ("Texel tuning").

Splits: the user's held-out bosses (every 4th of the data run, D2) are never fitted on; the
rest is split by battle (positions of one battle are correlated) into train / validation.
Reported: cross-entropy (log-loss) of a constant (the mean U), the linear fit, and a probe with
one hidden layer of 32 units (D10: a small net only if it beats linear by > 1% on held-out bosses).

    uv run python rl/fit_eval.py logs/p4_data.states.jsonl --out data/rr/eval_weights.txt
    POKE_ENGINE_EVAL_WEIGHTS=data/rr/eval_weights.txt uv run python scripts/rr_team_probe.py ...
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl import convert  # noqa: E402

HELD_OUT = {"#18", "#34", "#77", "#95"}  # every 4th boss of the data run (Misty, Erika, Lt. Surge rematch, Giovanni)


def load(path: str, every: int, seed: int = 0) -> list[tuple]:
    """[(battle key, state string, U)] from a states log; every k-th decision of each battle."""
    rng = random.Random(seed)
    decisions, outcome = [], {}
    for line in open(path):
        r = json.loads(line)
        key = (r["boss"], r["battle"], tuple(r["team"]))
        if r["kind"] == "decision":
            if rng.random() < 1 / every:
                decisions.append((key, r["state"]))
        elif r["kind"] == "battle":
            dead = sum(r["fainted"]) / len(r["fainted"])
            outcome[key] = float(r["won"]) * (1 - 0.5 * dead)
    return [(k, s, outcome[k]) for k, s in decisions if k in outcome]


def features(rows: list[tuple]) -> np.ndarray:
    pe = convert.pe
    return np.array([pe.evaluate_features(pe.State.from_string(s)) for _, s, _ in rows], dtype=np.float32)


def logloss(p: torch.Tensor, y: torch.Tensor) -> float:
    p = p.clamp(1e-6, 1 - 1e-6)
    return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())


def fit(model, x, y, xv, yv, l2: float = 1e-4, epochs: int = 300):
    """Full-batch L-BFGS on the soft-label cross-entropy; keeps the best validation loss."""
    opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
    best, state = float("inf"), None
    for _ in range(epochs // 20):
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
            loss = loss + l2 * sum((p ** 2).sum() for n, p in model.named_parameters() if "weight" in n)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            v = logloss(torch.sigmoid(model(xv).squeeze(-1)), yv)
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", help="<out>.states.jsonl files")
    ap.add_argument("--every", type=int, default=1, help="keep each decision with probability 1/every")
    ap.add_argument("--out", help="write the weights file (bias, then one weight per feature, in raw feature units)")
    args = ap.parse_args()
    convert.use_rr()
    torch.manual_seed(0)
    rows = [r for p in args.states for r in load(p, args.every)]
    names = convert.pe.evaluate_feature_names()
    x_all, y_all = features(rows), np.array([u for *_, u in rows], dtype=np.float32)
    held = np.array([k[0] in HELD_OUT for k, *_ in rows])
    battles = sorted({k for k, *_ in rows if k[0] not in HELD_OUT})
    random.Random(1).shuffle(battles)
    val_b = set(battles[: len(battles) // 5])
    val = np.array([k in val_b for k, *_ in rows]) & ~held
    tr = ~held & ~val
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6
    t = lambda m: (torch.tensor((x_all[m] - mu) / sd), torch.tensor(y_all[m]))
    (x, y), (xv, yv), (xh, yh) = t(tr), t(val), t(held)
    print(f"{len(rows)} positions from {len({k for k, *_ in rows})} battles: train {tr.sum()}, validation {val.sum()}, "
          f"held-out bosses {held.sum()} ({sorted(HELD_OUT)}); {len(names)} features; mean U {y_all.mean():.3f}")
    const = lambda yy: logloss(torch.full_like(yy, float(y.mean())), yy)
    lin = fit(torch.nn.Linear(len(names), 1), x, y, xv, yv)
    mlp = fit(torch.nn.Sequential(torch.nn.Linear(len(names), 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)), x, y, xv, yv)
    with torch.no_grad():
        for label, m in (("linear", lin), ("1x32 probe", mlp)):
            ll = [logloss(torch.sigmoid(m(a).squeeze(-1)), b) for a, b in ((x, y), (xv, yv), (xh, yh))]
            print(f"  {label:<11} log-loss train {ll[0]:.4f}  validation {ll[1]:.4f}  held-out bosses {ll[2]:.4f}")
        print(f"  {'constant':<11} log-loss train {const(y):.4f}  validation {const(yv):.4f}  held-out bosses {const(yh):.4f}")
        ws = lin.weight.squeeze(0).numpy() * (sd > 1e-3)  # a feature constant in training gets no weight (it would explode in play)
        w = ws / sd
        b = float(lin.bias) - float((ws * mu / sd).sum())
    print("weights (raw units, sorted by |standardised weight|):")
    for i in np.argsort(-np.abs(ws)):
        print(f"  {names[i]:<32} {w[i]:+.4f}  (standardised {float(lin.weight[0, i].detach()):+.3f})")
    if args.out:
        Path(args.out).write_text("\n".join(f"{v:.6f}" for v in [b, *w]) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
