"""Purposeful duping scan (2026-11-25, the user's idea): under the run's dupes clause (by evolution
line, reroll until new; one encounter per area, any method, delayable), owning the other lines of
an encounter table raises the odds of the one you want. For each target boss, list the tables
whose best mon for that boss is near the top of everything catchable, with the odds of rolling it
owning 0 / 1 / 2 / 3 blocker lines (the table's most common other lines first).

Value of a species for a boss = rr_route.box_boss_value({species}) (the duel-table matchup row
summed over the boss's team); "top" = its rank among every species in the boss table.

    uv run python scripts/rr_dupe_scan.py --bosses KOGA,BLAINE
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import rr_probe as rp  # noqa: E402
import rr_route as rt  # noqa: E402
from gauntlet import evo_line  # noqa: E402


def line_of(sp: str) -> str:
    return min(evo_line(sp) or {sp})


def odds(weights: dict[str, float], target: str, owned: set[str]) -> float:
    """P(rolling target's line) when owned lines are rerolled until new."""
    free = {ln: w for ln, w in weights.items() if ln not in owned}
    return free.get(target, 0.0) / sum(free.values()) if free else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bosses", default="KOGA,BLAINE")
    ap.add_argument("--top", type=int, default=15, help="targets ranked at most this high for the boss")
    a = ap.parse_args()
    rp.hardcore()
    enc = json.load(open(rp.RR / "tracker_encounters.json"))
    order = json.load(open(rp.RR / "order.json"))
    entries = order["entries"]
    locs = [x for x in enc["locations"] if not x["postgame"]]
    for boss in [b.strip() for b in a.bosses.split(",")]:
        tb = rt.build_boss_table(boss, extra_species=list(rp.STARTERS), seed=5)
        t = tb["progress"]
        val = {sp: rt.box_boss_value({sp}, tb) for sp in tb["rows"] if ">" not in sp}
        rank = {sp: i + 1 for i, sp in enumerate(sorted(val, key=val.get, reverse=True))}
        reach = entries[t]["reach"]
        print(f"\n=== {boss} (trainer #{t}, cap {tb['cap']}; {len(val)} species valued; best overall: "
              + ", ".join(f"{rt.label_sp(s, [tb])} {val[s]:.1f}" for s in sorted(val, key=val.get, reverse=True)[:5]) + ") ===")
        print(f"  {'location / method':<34} {'target':<26} {'rank':>4} {'value':>5}  P(0/1/2/3 blockers)   blockers (in order)")
        rows = []
        for li, loc in enumerate(locs[:reach + 1]):
            for meth, table in loc["methods"].items():
                if (0 if meth.startswith("grass") else order["gates"].get(meth, 999)) > t:
                    continue
                w = defaultdict(float)
                for x in table:
                    sp = rt.norm_sp(x["species"])
                    if int(x["levels"].split("-")[0]) <= tb["cap"] and sp in val:
                        w[line_of(sp)] += float(x["rarity"].rstrip("%")) or 1.0
                if len(w) < 2:
                    continue
                best = {ln: max((s for s in val if line_of(s) == ln), key=val.get) for ln in w}
                tgt_ln = max(w, key=lambda ln: val[best[ln]])
                tgt = best[tgt_ln]
                if rank[tgt] > a.top:
                    continue
                others = sorted((ln for ln in w if ln != tgt_ln), key=w.get, reverse=True)
                ps = [odds(w, tgt_ln, set(others[:k])) for k in range(4)]
                rows.append((rank[tgt], -ps[2], loc["name"], meth, tgt, val[tgt], ps, others[:3]))
        for r, _, ln, meth, tgt, v, ps, blk in sorted(rows):
            print(f"  {(ln + ' / ' + meth)[:34]:<34} {rt.label_sp(tgt, [tb])[:26]:<26} {r:>4} {v:5.1f}  "
                  + " ".join(f"{p:4.0%}" for p in ps) + "   " + ", ".join(blk))


if __name__ == "__main__":
    main()
