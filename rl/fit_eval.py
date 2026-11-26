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


def load(path: str, every: int, seed: int = 0, death_cost: float = 0.5) -> list[tuple]:
    """[(battle key, state string, U)] from a states log; every k-th decision of each battle."""
    rng = random.Random(seed)
    tag = Path(path).stem
    decisions, outcome = [], {}
    for line in open(path):
        r = json.loads(line)
        key = (r["boss"], (tag, r["battle"]), (tag, tuple(r["team"])))
        if r["kind"] == "decision":
            if rng.random() < 1 / every:
                decisions.append((key, r["state"]))
        elif r["kind"] == "battle":
            dead = sum(r["fainted"]) / len(r["fainted"])
            outcome[key] = float(r["won"]) * (1 - death_cost * dead)  # death_cost 1: a death = 1/6 of a win (D21)
    return [(k, s, outcome[k]) for k, s in decisions if k in outcome]


def within_team_logloss(scores: np.ndarray, y: np.ndarray, groups: list) -> tuple[float, float, float]:
    """Mean within-group (boss, team) log-loss with 1 scalar intercept per group, baseline, and correlation."""
    from collections import defaultdict
    by = defaultdict(list)
    for s, yi, g in zip(scores, y, groups):
        by[g].append((float(s), float(yi)))
    lls, base_lls, corrs = [], [], []
    for pts in by.values():
        ys = np.array([p[1] for p in pts])
        ss = np.array([p[0] for p in pts])
        if ys.std() < 1e-5:
            continue
        m = float(np.clip(ys.mean(), 1e-6, 1 - 1e-6))
        base_lls.append(float(-(ys * np.log(m) + (1 - ys) * np.log(1 - m)).mean()))
        a = float(np.log(m / (1 - m)))
        for _ in range(10):
            p = 1.0 / (1.0 + np.exp(-np.clip(a + ss, -20.0, 20.0)))
            a -= float((p - ys).mean()) / (float((p * (1 - p)).mean()) + 1e-4)
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(a + ss, -20.0, 20.0))), 1e-6, 1 - 1e-6)
        lls.append(float(-(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean()))
        if ss.std() > 1e-6:
            corrs.append(float(np.corrcoef(ss, ys)[0, 1]))
    return float(np.mean(lls)), float(np.mean(base_lls)), float(np.mean(corrs))


def fit_within_team(dx_tr: np.ndarray, y_tr: np.ndarray, g_tr_list: list,
                    dx_val: np.ndarray, y_val: np.ndarray, g_val_list: list,
                    l2_w: float = 1e-4, l2_a: float = 1e-4, epochs: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-effects logistic regression: logit = alpha[boss, team] + w . (f(s) - f(s0))."""
    uniq = sorted(set(g_tr_list))
    gmap = {g: i for i, g in enumerate(uniq)}
    g_tr = torch.tensor([gmap[g] for g in g_tr_list], dtype=torch.long)
    mu, sd = dx_tr.mean(0), dx_tr.std(0) + 1e-6
    x = torch.tensor((dx_tr - mu) / sd)
    y = torch.tensor(y_tr)
    alpha = torch.zeros(len(uniq), requires_grad=True)
    w = torch.zeros(dx_tr.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([alpha, w], lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
    best, best_w = float("inf"), None
    for _ in range(epochs // 20):
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(alpha[g_tr] + x @ w, y)
            loss = loss + l2_w * (w ** 2).sum() + l2_a * (alpha ** 2).mean()
            loss.backward()
            return loss
        opt.step(closure)
        ws = w.detach().numpy() * (sd > 1e-3)
        raw = ws / sd
        vll, _, _ = within_team_logloss(dx_val @ raw, y_val, g_val_list)
        if vll < best:
            best, best_w = vll, (ws.copy(), raw.copy())
    return best_w


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
    ap.add_argument("--death-cost", type=float, default=0.5, help="label U = win * (1 - death_cost * deaths/6): 0.5 = a death is 1/12 "
                    "of a win (D9 / D15); 1.0 = 1/6 (D21, the user)")
    ap.add_argument("--within-team", action="store_true",
                    help="Option B: fit w on f(s)-f(s0) with a per-(boss, team) fixed effect to remove boss/box difficulty")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply exported weights (and bias) by this factor")
    ap.add_argument("--out", help="write the weights file (bias, then one weight per feature, in raw feature units)")
    args = ap.parse_args()
    convert.use_rr()
    torch.manual_seed(0)
    rows = [r for p in args.states for r in load(p, args.every, death_cost=args.death_cost)]
    names = convert.pe.evaluate_feature_names()
    x_all, y_all = features(rows), np.array([u for *_, u in rows], dtype=np.float32)
    held = np.array([k[0] in HELD_OUT for k, *_ in rows])
    battles = sorted({k for k, *_ in rows if k[0] not in HELD_OUT})
    random.Random(1).shuffle(battles)
    val_b = set(battles[: len(battles) // 5])
    val = np.array([k in val_b for k, *_ in rows]) & ~held
    tr = ~held & ~val
    print(f"{len(rows)} positions from {len({k for k, *_ in rows})} battles: train {tr.sum()}, validation {val.sum()}, "
          f"held-out bosses {held.sum()} ({sorted(HELD_OUT)}); {len(names)} features; mean U {y_all.mean():.3f}")
    if args.within_team:
        first_idx = {}
        for i, (k, _, _) in enumerate(rows):
            if k not in first_idx:
                first_idx[k] = i
        dx = x_all - np.array([x_all[first_idx[k]] for k, *_ in rows], dtype=np.float32)
        groups = [(k[0], k[2]) for k, *_ in rows]
        g_tr = [groups[i] for i in np.where(tr)[0]]
        g_val = [groups[i] for i in np.where(val)[0]]
        g_held = [groups[i] for i in np.where(held)[0]]
        ws, w = fit_within_team(dx[tr], y_all[tr], g_tr, dx[val], y_all[val], g_val)
        b = 0.0
        for tag, m, g in (("train", tr, g_tr), ("validation", val, g_val), ("held-out bosses", held, g_held)):
            ll, bll, r = within_team_logloss(dx[m] @ w, y_all[m], g)
            print(f"  within-team {tag:<16} log-loss {ll:.4f} (group-const {bll:.4f}, delta {ll - bll:+.4f})  r={r:.3f}")
    else:
        mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6
        t = lambda m: (torch.tensor((x_all[m] - mu) / sd), torch.tensor(y_all[m]))
        (x, y), (xv, yv), (xh, yh) = t(tr), t(val), t(held)
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
    print(f"weights (raw units, scale={args.scale}, sorted by |standardised weight|):")
    for i in np.argsort(-np.abs(ws)):
        print(f"  {names[i]:<32} {w[i] * args.scale:+.4f}  (standardised {float(ws[i]):+.3f})")
    if args.out:
        Path(args.out).write_text("\n".join(f"{v * args.scale:.6f}" for v in [b, *w]) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
