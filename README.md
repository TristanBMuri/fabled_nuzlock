# nuzlock_rl

Nuzlocke gauntlet RL — minimal rebuild (attempt 3). Full plan: `project_overview.md`.

**Rule of this repo: as little code as possible.** Complexity is only added when a
concrete failure demands it. Two prior attempts died of scope bloat.

## Stack

- **Battles**: local Pokémon Showdown server + `poke-env` (Gymnasium-style Python client)
- **RL**: single-file CleanRL-style PPO (PyTorch), TensorBoard for logging
- **Python**: uv, Python 3.14

## Setup

```bash
uv sync                                   # python deps
cd pokemon-showdown && npm install --production   # once, showdown's node deps

# start the server (port 8020 — the old project uses 8000-8007, don't collide)
cd pokemon-showdown && node pokemon-showdown start --no-security --port 8020 &
```

`--no-security` = no accounts/login needed, bots just connect.

## Run

```bash
uv run python scripts/smoke_test.py       # 3 random-vs-random battles, prints state
uv run python scripts/train.py --random   # env sanity: 3 random-policy battles vs heuristic
uv run python scripts/train.py --updates 60   # PPO smoke run (~25 min, ~450 steps/s)
tensorboard --logdir runs
```

Checkpoint lands in `checkpoints/ppo_latest.pt` after every update.

## Layout

```
scripts/           # runnable entry points (smoke_test.py, later train.py)
pokemon-showdown/  # local server (cloned, one port)
project_overview.md  # the plan / roadmap
```

## Status

- [x] Showdown + poke-env working, battle state readable (smoke test)
- [x] Battle env + PPO + TensorBoard; smoke run shows learning signal
- [x] **Battler beats heuristic**: 500 updates / 512k steps / ~35 min single env →
      95% win rate (crossed 50% at update ~128 / 130k steps), reward -17.9→+15.4,
      entropy 1.59→0.54 (healthy), checkpoint `checkpoints/ppo_latest.pt`
- [x] Team pool (16 teams harvested from Showdown's generator) + attention policy
      (token obs, 1-layer transformer) + parallel envs (--envs N, ~2.9x at N=4)
- [x] **Pool of 4 solved by brute force**: 5000 updates x 2048 rollout / 10.2M
      steps / 4 envs (~3.9h, ~880 it/s) → **94% win rate** (all teams ≥85%).
      Diagnosis: sample starvation, not architecture. Curriculum unnecessary.
      Checkpoint `checkpoints/ppo_latest_attention.pt`
- [ ] **The real battler metric is performance under nuzlocke-level team
      entropy** (effectively-unique teams every battle: random encounters,
      deaths, level spread). Pool-of-4 mastery is necessary, not sufficient.
      Gauntlet env doubles as the source of realistic training teams.
- [ ] MEASURED (2026-08-19): the 94% pool-4 checkpoint scores **0/20 vs unseen
      Showdown random teams** (greedy policy). Generalization gap is total.
      Also found: harvested Therian formes (tornadusm etc.) mangle through
      parse_showdown_team→join_team — server rejects team → challenge hangs;
      eval_teams.py rebuilds env on timeout as workaround.
- [ ] Path forward options: (a) train battler on fresh random teams every
      battle (true entropy regime), (b) gauntlet env first to source realistic
      teams, (c) much larger fixed pool. Need decision.
- [ ] Nuzlocke gauntlet env (the actual goal)
