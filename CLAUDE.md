# CLAUDE.md

## Governing constraint

**Minimal code.** The project must stay small enough for one person to fully
comprehend. Two prior attempts ballooned (validation suites, curriculum systems,
Ray + MLflow + 8 servers). Do not add infrastructure, config systems, or
abstractions unless a concrete failure requires it. Prefer deleting code.

The prior attempt lives at `~/CodingProjects/DSPRO2_Pokemon/DSPRO2-FS26-Pokemon_RL`
— useful for reference (data tables, PPO lessons), not for copying architecture.

## Workflow

- Read `HANDOFF.md` first, then `project_overview.md` roadmap; `README.md`
  Status is the full notebook.
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
uv sync                                            # deps
uv run python scripts/smoke_test.py                # bot-vs-bot sanity check
node pokemon-showdown start --no-security --port 8020 &   # battle server
uv run pytest -q                                   # tests (when they exist)
```

Port 8020 is this project's server. **Ports 8000-8007 belong to the old project**
— never kill or bind them.

## Environment

- Showdown server must be running before any poke-env script.
- `poke-env` 0.15: `battle_against` is async (wrap in `asyncio.run`); players
  connect via `ServerConfiguration(websocket_url=..., authentication_url=...)`.
- `--no-security` server flag = no login needed for bots.

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
  Use `ai_util.battler_policy.action_mask/order_to_action`, never
  `SinglesEnv.get_action_mask/order_to_action` directly.

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
