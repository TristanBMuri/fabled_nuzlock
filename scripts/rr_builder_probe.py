"""Set-builder experiments on rr_team_probe.py (NOTEBOOK.md 2026-11-23, "Phase 0").
Same arguments as rr_team_probe.py. Top-level patches run in every
spawn worker too. BUILDER_NOREST=1: Rest out of best4's status list. BUILDER_EVS=n: n EVs in every stat."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_team_probe as p  # noqa: E402
import gauntlet  # noqa: E402

if os.environ.get("BUILDER_NOREST"):
    while "rest" in gauntlet.STATUS_MOVES:
        gauntlet.STATUS_MOVES.remove("rest")
EVS = int(os.environ.get("BUILDER_EVS", "0"))
if EVS:
    _build = p.build_mon

    def build_mon(*a, **k):
        m = _build(*a, **k)
        if m is not None:
            m.evs = dict.fromkeys(p.STATS, EVS)
        return m
    p.build_mon = build_mon

_mon_of = p.mon_of
p.mon_of = lambda ctx, sp: None if sp in ("ditto", "smeargle") else _mon_of(ctx, sp)  # poke-env's mirror crashes on Transform / Sketch

if os.environ.get("BUILDER_STONES"):  # every stone / held-item evolution in hand (RR: Celadon shop, before Koga)
    gauntlet.first_index = lambda location: 0
    p.evolve = lambda sp, level, rng=None, caught_level=None, trainer_idx=None: gauntlet.evolve(sp, level, rng, caught_level, 10 ** 6)

if __name__ == "__main__":
    p.main()
