"""Scratch: reproduce the AssertionError on Collector Fernando (order 158, 3x Mr. Mime)."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gauntlet import BATTLE_FORMAT, Mon, load_trainers, pack_team, trainer_mons, encounter_moves
from ai_util.evaluate import heuristic_brain
from ai_util.sim import SimBattle, SimBridge

trainers, order = load_trainers()
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 158
enemy = pack_team(trainer_mons(trainers[order[idx]["trainer_id"]]))
party = pack_team([Mon(s, 22, encounter_moves(s, 22)) for s in ("grotle", "staravia", "luxio")])
print("enemy:", enemy)
bridge, h = SimBridge(), heuristic_brain()
fails = 0
for i in range(5):
    sb = SimBattle(bridge, BATTLE_FORMAT, party, enemy)
    try:
        sb.play({"p1": h, "p2": h}, 100)
        print(f"battle {i}: winner {sb.winner} turn {sb.turn}")
    except Exception:
        fails += 1
        print(f"battle {i}: FAILED at turn {sb.turn}")
        traceback.print_exc(limit=-6)
        for side in ("p1", "p2"):
            print(side, "last lines:", [l[:120] for l in sb.history[side][-4:]])
    sb.close()
print("failures", fails)
bridge.close()
