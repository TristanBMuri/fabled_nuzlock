"""On-policy agreement of the Rust RR AI with the enemy that actually played (phase 1 (c)).

rl/rr_ai_diff.py measures agreement on positions from random play (off-policy, 90%). A search
that follows the model steers into positions it chooses, where the model may be worse; this
reads a run's `<out>.states.jsonl` (rr_team_probe.py --engine --states) and compares, at every
turn the searcher decided, the Rust AI's predicted options (`ai_pred`, ties) with the enemy's
actual choice that turn. Split by the actual choice: a move, or a switch.

    uv run python rl/rr_ai_onpolicy.py logs/p1_ai.states.jsonl
"""

import json
import sys
from collections import Counter, defaultdict

from poke_env.data import to_id_str


def norm(choice: str) -> str:
    """'move toxic' / 'switch Lileep' (Showdown, as sent) or 'toxic' / 'switch lileep' (engine) as one
    key; Hidden Power without its type and power (Showdown sends 'hiddenpower', the engine names
    'hiddenpowerrock60')."""
    key = _norm(choice)
    return "hiddenpower" if key.startswith("hiddenpower") else key


def _norm(choice: str) -> str:
    kind, _, name = choice.partition(" ")
    if kind in ("move", "switch") and name:
        return ("switch " if kind == "switch" else "") + to_id_str(name.removesuffix(" mega"))
    return to_id_str(choice.removesuffix("-mega")) if not choice.startswith("switch") else "switch " + to_id_str(choice[7:])


def main(path: str) -> None:
    pred, actual = {}, defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        key = (r["boss"], r["battle"], tuple(r["team"]))
        if r["kind"] == "decision" and r.get("ai_pred"):
            pred[key + (r["turn"],)] = {norm(c) for c in r["ai_pred"]}
        elif r["kind"] == "battle":
            for turn, side, c in r["chosen"]:
                if side == "p2" and turn not in actual[key]:
                    actual[key][turn] = norm(c)
    n, hit = Counter(), Counter()
    for (*key, turn), p in pred.items():
        a = actual[tuple(key)].get(turn)
        if a is None:
            continue
        kind = "switch" if a.startswith("switch") else "move"
        for k in (kind, "all", key[0]):
            n[k] += 1
            hit[k] += a in p
    for k in sorted(n, key=lambda k: (k not in ("all", "move", "switch"), k)):
        print(f"{k:<10} {hit[k] / n[k]:6.1%} of {n[k]}")


if __name__ == "__main__":
    main(sys.argv[1])
