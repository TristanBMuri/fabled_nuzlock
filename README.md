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
- [ ] **Pool of 4 is hard**: attention 1500 updates / 1.5M steps → 28% avg win
      (peak 70%), still climbing slowly. MLP plateaued ~20% at 512k. Next levers:
      curriculum (grow pool on win-rate milestones), warm-start, longer runs
- [ ] Nuzlocke gauntlet env (the actual goal — only after the battler works)
