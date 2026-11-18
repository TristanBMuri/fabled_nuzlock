# CLAUDE.md

## Governing constraint

**Minimal code.** The project must stay small enough for one person to fully
comprehend. Two prior attempts ballooned (validation suites, curriculum systems,
Ray + MLflow + 8 servers). Do not add infrastructure, config systems, or
abstractions unless a concrete failure requires it. Prefer deleting code.

The prior attempt lives at `~/CodingProjects/DSPRO2_Pokemon/DSPRO2-FS26-Pokemon_RL`
— useful for reference (data tables, PPO lessons), not for copying architecture.

## Workflow

- Read `HANDOFF.md` first, then `project_overview.md` roadmap; `NOTEBOOK.md`
  is the full notebook, `results/` the logs behind every table.
- Everything runs in-process through `ai_util/sim*` — no Showdown server
  unless you are cross-checking the sim against it.
- The battler (BC prior + search) must return trustworthy win/death
  probabilities on held-out gauntlet matchups before the run layer is trained
  on top of it. Roadmap step 1 before step 2.
- Generality is a constraint: gauntlet-specific facts live in explicit
  memory, never weights. Always report train vs held-out gauntlets.
- "Beats SimpleHeuristics on gen9randombattle" is a regression check, not a
  milestone. The metric is trainers cleared without a death.
- Keep files short. One file per concern. No `src/` package tree until there are
  enough files to justify it.

## Commands

```bash
uv sync                                            # deps; the Showdown clone: README → Setup
uv run pytest -q                                   # 46 tests, ~2 s (node + built Showdown)
# the benchmark table (~35 min, 20 workers x ~1 GB; check free -g first)
uv run python -u scripts/plan_gauntlet.py --episodes 20 --seed 300 --slice 405 --policy calc --search 9,8 --search-all --deep 9,16 --workers 20
# League levers: paired replay from the saved openers (~30 min)
uv run python scripts/league_test.py --snapshots results/league_snapshots --replays 4 --slice 405 --policy calc --search 9,8 --search-all --deep 9,16 --workers 20 --out logs/league_x.jsonl --vs results/league_base.jsonl
# RL learning track (rl/README.md, off the planner's path; poke-engine = Rust sim)
uv run python rl/lesson1_qlearning.py              # ~10 s
uv run python rl/engine_diff.py --positions 1000   # poke-engine vs Showdown, ~3 min (97% agree)
uv sync --reinstall-package poke-engine            # after editing the fork in poke-engine/ (rl/README.md)
uv run python rl/route.py --harvest data/teams/gauntlet_harvest/battles_best4.jsonl --trainers 60  # ~4 min
# the run dashboard (UI_TODO.md): probes with --out write <out>.events.jsonl
uv run python ui/build.py --serve                  # http://localhost:8765
```

No Showdown server is needed for anything. **Ports 8000-8007 belong to the old
project** — never kill or bind them.

## Environment

- Everything runs in-process through `ai_util/sim*` (Node bridge per process).
  `poke-env` 0.15 is used for its Battle mirrors and SimpleHeuristics only.
- One heavy job at a time: a planner worker is ~1 GB + one core; kill a planner
  by its process group (the main PID alone orphans workers and node bridges).
- Runs are stochastic across invocations even with the same seed; compare
  20-episode tables, League levers on `league_test.py --vs`.

## Hard-won lessons from attempt #3 (do not relearn these)

Full list in `project_overview.md` → Hard-won lessons. The ones that change
decisions:

- From-scratch PPO capped ~14% vs SimpleHeuristics at 6M steps; BC reached
  40% in ~1h. Start from imitation, get beyond it with search, not PPO knobs.
- RL finetune from a BC init oscillates around the init; it does not exceed it.
- "The obs is missing X" was undertraining every time. Converge BC first.
- gamma 0.95 on 40-turn episodes makes the win reward negligible; the agent
  farms shaping instead.
- Progression rule > policy: the level rule decided the gauntlet ceiling.
  Any change to it means regenerating the harvest and retraining the prior.
- Search over the engine beats training the prior harder (search-all: 52%
  → 95% clears; expert iteration v1: nothing).
- poke-env pitfalls with same-species mons (team collapse, species-based
  mask/labels) and the server's 252-EV fill are documented in HANDOFF.md.
  Use `ai_util.evaluate.action_mask` (identity-based), never
  `SinglesEnv.get_action_mask` directly.

## Hard-won lessons from attempt #2 (do not relearn these)

PPO on poke-env battles — values that actually worked there:

- `grad_clip=5.0` (0.5 crushed transformer grads to ~1% — silent no-learning)
- `gamma=0.95` (battles are ~25 turns; 0.99's horizon is too long)
- `entropy_coeff=0.05` (lower collapses entropy and stops exploration)
- `vf_clip_param=5.0` (reward range is ±10-ish)
- PyTorch 2.10+ `TransformerEncoderLayer` with float additive `src_mask` can
  produce NaNs — if using attention, verify logits for NaN early.
- Fixed-team phases matter: the agent cannot handle fully random teams early on.
  Keep the same team until win rate improves, then diversify.
- Entropy management is the #1 issue training battlers from scratch.
