"""D23 gate: do cheap-search labels rank teams like the 20,000-iteration ones? Matches every box / box_alt of a
run against the D19 data run (same boss, box index, variant, team) and compares the win labels: Pearson r, the
r two equally noisy 6-battle labels of the same team would reach (binomial noise), and mean win per trainer.

    uv run python rl/label_gate.py logs/d23_gate_2k.events.jsonl [results/rr_d19_all.events.jsonl.xz]
"""
import json, lzma, sys
import numpy as np


def rows(path):
    out = {}
    for line in (lzma.open(path, "rt") if path.endswith(".xz") else open(path)):
        e = json.loads(line)
        if e["kind"] in ("box", "box_alt"):
            out[(e["boss"], e["i"], e.get("k", -1), tuple(e["team"]))] = e["win"]
    return out


cheap, ref = rows(sys.argv[1]), rows(sys.argv[2] if len(sys.argv) > 2 else "results/rr_d19_all.events.jsonl.xz")
keys = [k for k in cheap if k in ref]
a, b = np.array([cheap[k] for k in keys]), np.array([ref[k] for k in keys])
noise = lambda y: float(np.mean(y * (1 - y) / 5))  # per-label binomial variance, 6 battles
rel = lambda y: max(0.0, 1 - noise(y) / y.var())
print(f"{len(keys)} teams matched of {len(cheap)}; r = {np.corrcoef(a, b)[0, 1]:.2f}, "
      f"noise-only ceiling ~ {np.sqrt(rel(a) * rel(b)):.2f}")
for boss in sorted({k[0] for k in keys}, key=lambda s: int(s[1:])):
    m = np.array([k[0] == boss for k in keys])
    print(f"  {boss:<5} n {m.sum():>3}  win cheap {a[m].mean():.2f} / 20k {b[m].mean():.2f}  r {np.corrcoef(a[m], b[m])[0, 1]:+.2f}")
