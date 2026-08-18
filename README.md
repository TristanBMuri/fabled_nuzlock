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
tensorboard --logdir runs                 # once training exists
```

## Layout

```
scripts/           # runnable entry points (smoke_test.py, later train.py)
pokemon-showdown/  # local server (cloned, one port)
project_overview.md  # the plan / roadmap
```

## Status

- [x] Showdown + poke-env working, battle state readable (smoke test)
- [ ] Battle env (Gymnasium wrapper around one battle, fixed team) + PPO
- [ ] Beat heuristic opponent on fixed team
- [ ] Nuzlocke gauntlet env (the actual goal — only after the battler works)
