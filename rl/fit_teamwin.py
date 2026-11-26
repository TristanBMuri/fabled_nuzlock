"""D18 A: a team-vs-boss win-probability model fitted on the box results we already logged.

The team builder needs a judge: P(win | our six with their sets, the boss's six). Every box event of
rr_team_probe.py (--out) logs the chosen team, its sets, its win rate over n_search battles and the
6x6 matchup matrix (ai_util.matchup cells: hits to KO each way, speed, damage dealt / taken, safe,
race). Features summarise that matrix per enemy mon (does any of ours answer it?) and over the team,
plus three set counts (pivots, priority attacks, status moves). A logistic regression and a 1x32
probe are fitted to the win rate (soft labels, cross-entropy), the same recipe as rl/fit_eval.py.

Splits (--split): d18 = train on Brock .. Blaine, hold out Lt. Surge .. Clair; d19 = train on every
non-boss trainer of the trainer order (rr_team_probe.py --bosses "#i,..."), hold out the bosses (gym
leaders, Giovanni, Elite Four, rematches) wherever they were logged. A fifth of the training boxes
(with their --alt variants) is validation. Labels (--label): win rate, or the D15 utility
win * (1 - 0.5 * deaths/mon) (route trainers are mostly won: their contrast is in deaths). Reported per split: log-loss vs the constant (train mean), and
the within-boss Spearman rank correlation between prediction and the measured win rate: the number
that matters for choosing teams. Labels are 6 battles each, so even a perfect model scores below 1.

Caveat: every team was picked by our own chooser (mostly 'cover', which reads the same matrix), so the
data only covers teams the chooser liked (off-policy coverage, D18).

    uv run python rl/fit_teamwin.py results/rr_*.events.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from poke_env.data import GenData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.fit_eval import fit, logloss  # noqa: E402

HELD_OUT = {"LT. SURGE", "GIOVANNI", "SABRINA", "PRYCE", "CLAIR"}  # d18
BOSSES = {"FALKNER", "BROCK", "BUGSY", "MISTY", "WHITNEY", "LT. SURGE", "MORTY", "ERIKA", "GIOVANNI", "CHUCK", "SABRINA",
          "KOGA", "PRYCE", "JASMINE", "BLAINE", "CLAIR", "LORELEI", "BRUNO", "AGATHA", "LANCE"}  # d19, rematches included
ORDER = json.load(open(Path(__file__).resolve().parents[1] / "data/rr/order.json"))["entries"]


def boss_name(b: str) -> str:
    """A probe's boss key ("MISTY" or "#76") -> the trainer-order name without "(REMATCH)"."""
    return (ORDER[int(b[1:])]["name"] if b.startswith("#") else b).replace("(REMATCH)", "").strip()
PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "shedtail"}
MOVES = GenData.from_gen(9).moves
NAMES = ["safe_mean", "safe_all", "race_mean", "race_all", "fast_ohko_mean", "best_ko_mean", "best_ko_worst",
         "best_hit_mean", "best_hit_worst", "least_take_mean", "least_take_worst", "faster", "hit", "take", "heals",
         "ohkod", "liability", "pivots", "priority", "status"]


def features(e: dict) -> list[float]:
    m = e["matrix"]
    ours = list(m)
    enemies = list(m[ours[0]])
    per = []
    for en in enemies:
        c = [m[o][en] for o in ours]
        per.append((any(x["safe"] for x in c), any(x["race"] for x in c), any(x["faster"] and x["ko_them"] == 1 for x in c),
                    min(min(x["ko_them"], 4) for x in c), max(min(x["hit"], 2) for x in c), min(min(x["take"], 2) for x in c)))
    p = np.array(per, dtype=float)
    cells = [m[o][en] for o in ours for en in enemies]
    mean = lambda k, cap=9: float(np.mean([min(float(x[k]), cap) for x in cells]))
    ohko = [sum(m[o][en]["ko_us"] == 1 for en in enemies) for o in ours]
    moves = [mv for s in e["sets"] for mv in s["moves"]]
    data = [MOVES.get(mv, {}) for mv in moves]
    return [p[:, 0].mean(), p[:, 0].min(), p[:, 1].mean(), p[:, 1].min(), p[:, 2].mean(), p[:, 3].mean(), p[:, 3].max(),
            p[:, 4].mean(), p[:, 4].min(), p[:, 5].mean(), p[:, 5].max(), mean("faster"), mean("hit", 2), mean("take", 2),
            mean("heals"), float(np.mean(ohko)) / len(enemies), max(ohko) / len(enemies),
            sum(mv in PIVOTS for mv in moves) / 6, sum(d.get("priority", 0) > 0 and d.get("category") != "Status" for d in data) / 6,
            sum(d.get("category") == "Status" for d in data) / 6]


def load(paths: list[str]) -> list[dict]:
    """Box events with a matrix and sets, from runs under the player's hardcore rules."""
    rows = []
    for path in paths:
        cfg = {}
        for line in open(path):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e["kind"] == "start":
                cfg = e.get("config", {})
            elif e["kind"] in ("box", "box_alt") and "matrix" in e and "sets" in e and cfg.get("legal") and not cfg.get("normal"):
                rows.append(e | {"n": cfg.get("n_search", 6), "cfg": [cfg.get(k) for k in ("engine", "enemy", "engine_opp", "items", "no_items")]})
    return rows


def ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for u in np.unique(v):  # ties share their mean rank
        r[v == u] = r[v == u].mean()
    return r


def spearman(pred: np.ndarray, y: np.ndarray, boss: np.ndarray) -> float:
    """Mean over bosses of the rank correlation between prediction and measured win rate."""
    out = []
    for b in np.unique(boss):
        k = boss == b
        if k.sum() > 5 and y[k].std() > 0:
            out.append(np.corrcoef(ranks(pred[k]), ranks(y[k]))[0, 1])
    return float(np.mean(out))


def pairs(rows: list[dict], pred: np.ndarray, held: np.ndarray, label: str, y: np.ndarray) -> None:
    """The team builder's question: the same box (boss, box contents) logged with two different teams or set
    choices (different runs, same simulator settings). Does the model's preference agree with the measured win-rate difference?"""
    groups = {}
    for i, e in enumerate(rows):
        groups.setdefault((e["boss"], json.dumps(e["box"]), json.dumps(e["cfg"])), []).append(i)  # same simulator settings only
    for tag, want in (("train bosses", False), ("held-out", True)):
        agree, n, d_pred, d_true = 0, 0, [], []
        for idx in groups.values():
            idx = [i for i in idx if held[i] == want]
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if rows[i]["sets"] == rows[j]["sets"] or y[i] == y[j]:
                        continue
                    n += 1
                    agree += (pred[i] - pred[j]) * (y[i] - y[j]) > 0
                    d_pred.append(pred[i] - pred[j])
                    d_true.append(y[i] - y[j])
        if n:
            print(f"  same box, different team/sets ({label}, {tag}): {n} pairs, model picks the better one {agree / n:.0%}, "
                  f"corr of differences {np.corrcoef(d_pred, d_true)[0, 1]:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+")
    ap.add_argument("--split", choices=("d18", "d19"), default="d18")
    ap.add_argument("--label", choices=("win", "utility"), default="win")
    args = ap.parse_args()
    torch.manual_seed(0)
    rows = load(args.events)
    x_all = np.array([features(e) for e in rows], dtype=np.float32)
    y_all = np.array([e["win"] * (1 - 0.5 * e["deaths_per_mon"]) if args.label == "utility" else e["win"] for e in rows], dtype=np.float32)
    boss = np.array([e["boss"] for e in rows])
    if args.split == "d18":
        held = np.isin(boss, list(HELD_OUT))
        tr = np.isin(boss, ["BROCK", "MISTY", "ERIKA", "KOGA", "BLAINE"])
    else:
        held = np.array([boss_name(b) in BOSSES for b in boss])
        tr = ~held
    rng = random.Random(1)
    val_box = {}
    val = np.array([val_box.setdefault((e["boss"], json.dumps(e["box"])), rng.random() < 0.2) for e in rows]) & tr
    tr = tr & ~val
    print(f"{len(rows)} boxes: train {tr.sum()}, validation {val.sum()}, held-out bosses {held.sum()}; {len(NAMES)} features")
    for b in sorted(set(boss)) if len(set(boss)) <= 20 else []:
        k = boss == b
        noise = float(np.mean(y_all[k] * (1 - y_all[k]) / 5))  # binomial variance of a 6-battle mean (unbiased, n-1)
        print(f"  {b:<10} {k.sum():>5} boxes  win {y_all[k].mean():.2f}  sd {y_all[k].std():.2f}  "
              f"share of variance that is battle noise {noise / max(y_all[k].var(), 1e-9):.0%}")
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6
    t = lambda m: (torch.tensor((x_all[m] - mu) / sd), torch.tensor(y_all[m]))
    (x, y), (xv, yv), (xh, yh) = t(tr), t(val), t(held)
    lin = fit(torch.nn.Linear(len(NAMES), 1), x, y, xv, yv)
    mlp = fit(torch.nn.Sequential(torch.nn.Linear(len(NAMES), 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)), x, y, xv, yv, l2=1e-3)
    print(f"{'':<14}{'log-loss train / val / held-out':<38}rank corr within boss: train / val / held-out")
    splits = ((x, y, tr), (xv, yv, val), (xh, yh, held))
    with torch.no_grad():
        for label, f in (("constant", lambda a: torch.full((len(a),), float(y.mean()))),
                         ("safe_mean", lambda a: torch.sigmoid(a[:, 0])),  # one hand feature: share of enemy mons we answer safely
                         ("linear", lambda a: torch.sigmoid(lin(a).squeeze(-1))),
                         ("1x32 probe", lambda a: torch.sigmoid(mlp(a).squeeze(-1)))):
            ll = [logloss(f(a), b) if label != "safe_mean" else float("nan") for a, b, _ in splits]
            rc = [spearman(f(a).numpy(), b.numpy(), boss[m]) if label != "constant" else 0.0 for a, b, m in splits]
            print(f"  {label:<12}{ll[0]:.4f} / {ll[1]:.4f} / {ll[2]:.4f}{'':<10}{rc[0]:+.3f} / {rc[1]:+.3f} / {rc[2]:+.3f}")
        print("  held-out per boss (linear / probe rank corr):",
              ", ".join(f"{b} {spearman(torch.sigmoid(lin(xh).squeeze(-1)).numpy()[boss[held] == b], yh.numpy()[boss[held] == b], boss[held][boss[held] == b]):+.2f}"
                        f" / {spearman(torch.sigmoid(mlp(xh).squeeze(-1)).numpy()[boss[held] == b], yh.numpy()[boss[held] == b], boss[held][boss[held] == b]):+.2f}"
                        for b in sorted(HELD_OUT)))
        pairs(rows, torch.sigmoid(lin(torch.tensor((x_all - mu) / sd)).squeeze(-1)).numpy(), held, "linear", y_all)
        pairs(rows, torch.sigmoid(mlp(torch.tensor((x_all - mu) / sd)).squeeze(-1)).numpy(), held, "1x32 probe", y_all)
    w = lin.weight.detach().squeeze(0).numpy()
    print("linear weights (standardised, by size):")
    for i in np.argsort(-np.abs(w)):
        print(f"  {NAMES[i]:<18} {w[i]:+.3f}")


if __name__ == "__main__":
    main()
