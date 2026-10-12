"""The env's battles, in-process through ai_util.sim (no Showdown server).

SimBattleRunner: player brain = SimpleHeuristics (default), the calc bot
(policy='calc') or a BC prior at a path; search=(k, rollouts) plays boss fights
with top-k rollout search (the compute dial); estimate() / estimate_block() are
the death-probability queries the planner ranks parties by; run() plays one
fight and returns the per-mon state the next no-heal fight starts from.
GAUNTLET_HARVEST=path records every (party, enemy, won) the env plays.
"""

import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_util.evaluate import Searcher, evaluate, heuristic_brain, prior_brain  # noqa: E402
from ai_util.sim import SimBattle, SimBridge  # noqa: E402

BATTLE_FORMAT = "gen8customgamenogimmicks"
# Unset = no recording: the battler's data must follow the progression rule the run
# layer plays under, so a harvest is always an explicit, fresh file.
HARVEST = Path(os.environ["GAUNTLET_HARVEST"]) if os.environ.get("GAUNTLET_HARVEST") else None
ENEMY = "heuristic"  # the enemy AI: poke-env's SimpleHeuristics, or "bdsp" = the Gen 4 trainer AI (ai_util/bdsp_ai.py)


def enemy_brain():
    if ENEMY == "bdsp":
        from ai_util.bdsp_ai import BDSPAI
        from ai_util.sim import player_brain
        return player_brain(BDSPAI(battle_format=BATTLE_FORMAT, start_listening=False, log_level=50))
    return heuristic_brain()


class SimBattleRunner:
    """Same contract as BattleRunner, but battles run in-process through
    ai_util.sim (no Showdown server, ~100 battles/s). Player brain is
    SimpleHeuristics (default) or the BC prior at `policy`, optionally with
    top-k rollout search `search=(k, rollouts)` for boss fights."""

    def __init__(self, policy: str | None = None, search: tuple[int, int] | None = None,
                 max_turns: int = 100, full_info: bool = False):
        self.max_turns, self.full_info = max_turns, full_info  # full_info: the player knows the enemy's sets
        self.bridge = SimBridge()
        self.enemy = enemy_brain()
        self.search = search  # default (k, rollouts) for boss fights; the compute dial
        self.policy, self.calc = None, policy == "calc"
        self._searchers: dict[tuple[int, int], object] = {}  # one Searcher per width
        if self.calc:  # the scripted damage-calc bot; search=(k, r) = all legal actions (up to k) as candidates
            from ai_util.calc_bot import calc_brain
            self.player = calc_brain()
        elif policy:
            import torch
            from ai_util.battler_policy import load_policy
            torch.set_num_threads(1)  # batch-1 MLP: the intra-op pool only adds overhead
            self.policy = load_policy(policy, device="cpu")
            self.player = prior_brain(self.policy)
        else:
            self.player = heuristic_brain()

    def searcher(self, width: tuple[int, int] | None):
        """Search brain at `width` = (k, rollouts); None or no prior = the fast player."""
        if width is None or (self.policy is None and not self.calc):
            return self.player
        if width not in self._searchers:
            self._searchers[width] = Searcher(self.policy, self.enemy, k=width[0], rollouts=width[1],
                                              max_turns=self.max_turns,
                                              rollout=self.player if self.calc else None)
        return self._searchers[width]

    def estimate(self, player_packed: str, enemy_packed: str, n: int = 8, carry: list | None = None):
        """Death-probability query: (P(win), [P(death_i)] in packed order, mean
        HP fraction left) under the fast player brain, n fresh-seed battles.
        None on failure."""
        try:
            r = evaluate(self.bridge, player_packed, enemy_packed, self.player, self.enemy,
                         n=n, max_turns=self.max_turns, carry=carry, full_info=self.full_info)
        except Exception as e:
            import traceback
            print(f"SimBattleRunner.estimate failed: {type(e).__name__}: {e}\n"
                  + "".join(traceback.format_exc().splitlines(True)[-4:]))
            return None
        return r["p_win"], r["p_death"], r["hp"]

    def estimate_block(self, player_packed: str, enemies_packed: list[str], n: int = 8):
        """estimate() over the League block: the same party through every fight,
        fully healed between fights (HP, status, PP), the dead stay dead.
        (P(all won), [P(death_i)], progress) — progress = fights won plus the
        lost fight's KOs, over the block (1.0 = cleared): the HP slot. When
        every party wipes (the fast brain at Cynthia) deaths are all 1 and
        how far a party gets is the only ordering left."""
        wins, deaths, progress = 0, None, 0.0
        try:
            for _ in range(n):
                carry, won, st, fights = None, True, None, 0.0
                for enemy in enemies_packed:
                    sb = SimBattle(self.bridge, BATTLE_FORMAT, player_packed, enemy, carry=carry,
                                   full_info=self.full_info)
                    sb.play({"p1": self.player, "p2": self.enemy}, self.max_turns)
                    st = sb.state()
                    sb.close()
                    # between fights: full heal (HP, status, PP), the dead stay dead
                    carry = [{"hp": 0 if m["fainted"] else None} for m in st["p1"]]
                    if st["winner"] != "p1":
                        won = False
                        fights += sum(m["fainted"] for m in st["p2"]) / len(st["p2"])
                        break
                    fights += 1
                wins += won
                dead = [m["fainted"] for m in st["p1"]]
                deaths = [d + float(x) for d, x in zip(deaths, dead)] if deaths else [float(x) for x in dead]
                progress += fights / len(enemies_packed)
        except Exception as e:
            print(f"SimBattleRunner.estimate_block failed: {type(e).__name__}: {e}")
            return None
        return wins / n, [d / n for d in deaths], progress / n

    def run(self, player_packed: str, enemy_packed: str, timeout: float = 0.0,
            boss: bool = False, search: tuple[int, int] | None = None, carry: list | None = None):
        """One battle. Returns (won, alive_mask, state) aligned to player_packed
        order — state = per-mon {hp, status, pp} after the fight, what the next
        no-heal fight starts from — or None on failure (bridge rebuilt).
        boss=True searches at the default width; search=(k, r) sets the width
        for this fight (deeper at hard ones); carry = state to start from."""
        if not player_packed or not enemy_packed:
            print("SimBattleRunner: empty packed team")
            return None
        brain = self.searcher(search or (self.search if boss else None))
        try:
            sb = SimBattle(self.bridge, BATTLE_FORMAT, player_packed, enemy_packed, carry=carry,
                           full_info=self.full_info)
            sb.play({"p1": brain, "p2": self.enemy}, self.max_turns)
            st = sb.state()
            won, dead = st["winner"] == "p1", [m["fainted"] for m in st["p1"]]
            state = [{"hp": m["hp"], "status": m.get("status", ""), "pp": m.get("pp")} for m in st["p1"]]
            sb.close()
        except Exception as e:
            import traceback
            print(f"SimBattleRunner: rebuild after {type(e).__name__}: {e}\n"
                  + "".join(traceback.format_exc().splitlines(True)[-4:]))
            try:
                self.bridge.close()
            except Exception:
                pass
            self.bridge = SimBridge()
            return None
        return won, [not d for d in dead], state


_HARVEST_LOCK = None


def record_battle(run_id: str, trainer_id: int | None, won: bool, player_packed: str, enemy_packed: str):
    global _HARVEST_LOCK
    if HARVEST is None:
        return
    if _HARVEST_LOCK is None:
        _HARVEST_LOCK = threading.Lock()
    HARVEST.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"run_id": run_id, "trainer_id": trainer_id, "info": "full", "won": won,
                       "player_team": player_packed, "enemy_team": enemy_packed}) + "\n"
    with _HARVEST_LOCK:
        with open(HARVEST, "a") as f:
            f.write(line)


