# nuzlock_rl

A Nuzlocke solver for Pokémon Brilliant Diamond / Shining Pearl, played on
Pokémon Showdown's simulator in-process (no server, no ROM). Attempt 3; the
rule of this repo is **as little code as possible**.

**Start here: [`HANDOFF.md`](HANDOFF.md)** (state, findings, defects, next steps).
[`project_overview.md`](project_overview.md) is the plan and roadmap;
[`NOTEBOOK.md`](NOTEBOOK.md) the chronological lab notebook with every number;
[`results/`](results/) the logs each table cites; [`HUMAN_TODO.md`](HUMAN_TODO.md)
what only a human can do.

**What it is (2026-10-10).** The battler is a scripted damage-calc bot
(`ai_util/calc_bot.py`) with one-ply rollout search over the real engine
(`ai_util/evaluate.py`); the run layer is a scripted planner that fields the
party with the lowest simulated death probability (`scripts/plan_gauntlet.py`).
Nothing learned is on the path: PPO, behaviour cloning, RL finetuning, expert
iteration and a learned value leaf were each measured and retired (notebook).
Benchmark: the whole game, 405 trainers to Cynthia, hardcore rules
(level cap, dupes clause, best4 movesets over level-up moves + TMs, found
items, flat EVs). The Elite Four never loses; Cynthia costs ~3 deaths per run
and decides the clear rate (League test: 0.70 ± 0.07 block clears).

## Setup

```bash
uv sync                                                   # python deps
git clone https://github.com/smogon/pokemon-showdown.git  # the simulator (gitignored, vendored)
cd pokemon-showdown && git checkout 84d7ceb && npm install && node build && cd ..
./scripts/setup_custom_formats.sh                        # the gen8customgamenogimmicks format
uv run pytest -q                                          # 22 tests, ~2 s (node + built Showdown)
```

No Showdown server is needed for anything: `ai_util/sim_bridge.js` drives the
simulator over JSON lines. Ports 8000–8007 belong to another project.

## Run

```bash
# the benchmark table (~30 min on 10 workers, ~1 GB each; check free -g first)
uv run python -u scripts/plan_gauntlet.py --episodes 20 --seed 300 --slice 405 \
    --policy calc --search 9,8 --search-all --deep 9,16 --workers 10

# the League alone, replayed from 20 saved openers, paired against the baseline (~30 min)
uv run python scripts/league_test.py --snapshots results/league_snapshots --replays 4 \
    --slice 405 --policy calc --search 9,8 --search-all --deep 9,16 --workers 10 \
    --out logs/league_x.jsonl --vs results/league_base.jsonl

# one verbose episode through Gardenia (~30 s)
uv run python scripts/plan_gauntlet.py --episodes 1 --slice 59 --policy calc --search 3,8 --search-all --verbose
```

Runs are stochastic across invocations even with the same `--seed` (the sim
seeds battles fresh): compare 20-episode tables, and League levers on the
paired test, never single runs. Noise floor on a 10-episode table: ±25 points
of clear rate (notebook, 2026-10-09).

## Layout

```
ai_util/sim_bridge.js    Node: Showdown's simulator over JSON lines (new / choose / fork / state)
ai_util/sim.py           SimBridge, SimBattle (+ poke-env mirrors), dedupe_nicknames, player_brain
ai_util/evaluate.py      evaluate() = P(win) / P(death_i) for a matchup; Searcher = rollout search
ai_util/calc_bot.py      CalcPlayer: the damage-formula battler (the player brain)
ai_util/battler_*.py, collect_game.py, pretrain_bc.py   the BC prior pipeline (off the path; HUMAN_TODO 5)
scripts/gauntlet.py      the game folder, Mon / packed teams, movesets, items, evolution, abilities
scripts/runner.py        SimBattleRunner: run(), estimate(), estimate_block(); harvest recording
scripts/nuzlocke_env.py  NuzlockeEnv: the run as a state machine (offers, party, fight, League block); Rules
scripts/plan_gauntlet.py the scripted planner + the table
scripts/league_test.py   the League replayed from saved openers, paired by box
scripts/build_*.py       rebuild data/bdsp/ from Bulbapedia / Serebii / the Showdown dex
data/bdsp/               the game folder (trainers, order, encounters, learnsets, evolutions, species, moves, tms, items)
tests/                   env rules, the sim bridge, the calc bot's stat model
results/                 the logs behind every table + the League baseline and its 20 opener snapshots
models/, logs/, data/teams/gauntlet_harvest/   gitignored: checkpoints, run logs, battle harvests
```
