"""Paired sequential test (SPRT) for two runs of rr_team_probe.py on the same boxes and seeds.

Every gate in rl/RR_PLAN.md (plan v3) asks "is B better than A by at least delta win rate?".
The runs are paired: box i of a boss has the same species and battle seeds in both, so the
per-box difference x = win_B - win_A removes most of the box-to-box noise. The test is a
generalised SPRT on those differences (normal approximation, variance from the sample):

    LLR = (delta * sum(x) - n * delta^2 / 2) / var(x)
    accept H1 (B >= A + delta)  when LLR >= log((1 - beta) / alpha)
    accept H0 (no difference)   when LLR <= log(beta / (1 - alpha))
    otherwise: keep playing boxes

It is the same test chess engine testing uses (Fishtest's normalised SPRT), with boxes in place
of game pairs. Deaths per mon are reported beside it as a mean difference with a 95% interval.

    uv run python scripts/sprt.py logs/a.events.jsonl logs/b.events.jsonl --delta 0.08
"""

import argparse
import json
import math
from pathlib import Path


def boxes(path: str) -> dict:
    """{(boss, box index): (win, deaths per mon)} from a run's events file (kind "box")."""
    out = {}
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e.get("kind") == "box":
            out[e["boss"], e["i"]] = (e["win"], e["deaths_per_mon"])
    return out


def sprt(x: list[float], delta: float, alpha: float = 0.05, beta: float = 0.05) -> dict:
    n = len(x)
    lo, hi = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    if n < 2:
        return {"n": n, "llr": 0.0, "lower": lo, "upper": hi, "verdict": "continue"}
    mean = sum(x) / n
    var = max(sum((v - mean) ** 2 for v in x) / (n - 1), 1e-6)
    llr = (delta * sum(x) - n * delta ** 2 / 2) / var
    verdict = "H1: better by >= delta" if llr >= hi else "H0: no difference" if llr <= lo else "continue"
    return {"n": n, "mean": mean, "ci95": 1.96 * math.sqrt(var / n), "llr": llr, "lower": lo, "upper": hi, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="baseline run's <out>.events.jsonl")
    ap.add_argument("b", help="candidate run's <out>.events.jsonl")
    ap.add_argument("--delta", type=float, default=0.08, help="the win-rate gain H1 claims (default 8 points)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    args = ap.parse_args()
    a, b = boxes(args.a), boxes(args.b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no box is in both runs (same bosses, --boxes, --seed?)")
    r = sprt([b[k][0] - a[k][0] for k in keys], args.delta, args.alpha, args.beta)
    dd = [b[k][1] - a[k][1] for k in keys]
    dm = sum(dd) / len(dd)
    dci = 1.96 * math.sqrt(sum((v - dm) ** 2 for v in dd) / max(len(dd) - 1, 1) / len(dd))
    print(f"{len(keys)} paired boxes over {len({k[0] for k in keys})} bosses ({len(set(a) - set(b))} only in A, {len(set(b) - set(a))} only in B)")
    print(f"win B - A: {r.get('mean', 0):+.1%} ± {r.get('ci95', 0):.1%}   deaths/mon B - A: {dm:+.3f} ± {dci:.3f}")
    print(f"SPRT delta {args.delta:.0%}: LLR {r['llr']:.2f} in [{r['lower']:.2f}, {r['upper']:.2f}] -> {r['verdict']}")


if __name__ == "__main__":
    main()
