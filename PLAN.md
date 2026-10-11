# Nuzlocke Gauntlet RL — Agent-Executable Implementation Plan

Spec: `README.md` (do not edit its design sections). This file is the work
order. Written 2026-09-08 against verified versions: poke-env 0.16.1,
Pokémon Showdown commit `6b4bc34e44cc2541929cc4b8fff96e756ab3f268`
(package 0.11.11, needs Node ≥ 16), Node 24.20.0 LTS, Python 3.12,
stable-baselines3 / sb3-contrib 2.9.0 (accepts gymnasium 1.3, torch ≥ 2.8).

---

## 0. Protocol for the executing agent

1. Work tasks **in order** (P1.1, P1.2, …). Do not start a task until the
   previous task's **Verify** block passes.
2. Each task ends with one commit: `[P1.3] <task title>`; tick the task's
   checkbox in this file in the same commit. Commit from a clean tree.
3. If Verify fails three times, write what happened under
   **§8 Decision log**, commit, and stop with a report. Do not "work around"
   a failing verification by weakening it.
4. Source of truth for poke-env is the **installed package**, not blogs:
   `uv run python -c "import poke_env, os; print(os.path.dirname(poke_env.__file__))"`
   and read `environment/env.py`, `environment/singles_env.py`,
   `environment/single_agent_wrapper.py`, `player/player.py`,
   `teambuilder/teambuilder_pokemon.py`. Old class names
   (`Gen8EnvSinglePlayer`, `EnvPlayer`) do not exist any more.
5. Never commit `third_party/`, `.venv/`, `.nodeenv/`, `replays/`,
   `checkpoints/`, `runs/`, `*.log`.
6. Unit tests: `uv run pytest -m "not integration"`. Integration tests
   (start a Showdown server): `uv run pytest -m integration`. Both must be
   green at every commit from P1.6 on.
7. Keep functions small and typed. Every public function gets a docstring
   with its contract. No global mutable state except the poke-env loop.
8. Machine facts: Linux, 32 CPU cores, 60 GB RAM, RTX 5090 32 GB, `uv 0.12`,
   Python 3.12.3 system, Docker available, **Node not installed**, no sudo
   assumed. GitHub push works via `gh` auth.

---

## 1. Fixed decisions (do not re-open without a Decision-log entry)

| Topic | Decision |
|---|---|
| Generation / format | Gen 4, Showdown format id `gen4customgame` (no restrictions: any level, 1–6 mons, illegal movesets ok; verified in `config/formats.ts`). No team preview, no gimmicks → battle action space `Discrete(10)` = 6 switches + 4 moves (poke-env `SinglesEnv.get_action_space_size(4)`). |
| Battle engine | Showdown via poke-env websocket, behind the `Battler` and `BattleBackend` interfaces in §2. |
| Battler order | `HeuristicBattler` (poke-env `SimpleHeuristicsPlayer`) → `RLBattler` (MaskablePPO) → optional `SearchBattler` (poke-engine). Run-level work never waits for the RL battler. |
| Opponent AI | `TrainerAI` (ours, in-game style) is the default opponent; `SimpleHeuristicsPlayer` is the "hard" opponent. |
| Opponent information | Run env injects the known `TrainerSpec` into battler observations; `blind=True` disables that. No open team sheets. |
| RL library | `stable-baselines3==2.9.0` + `sb3-contrib==2.9.0` (`MaskablePPO`) + `torch>=2.8`. Fallback if the resolver refuses the poke-env pins: hand-written PPO in `models/ppo.py` (CleanRL style). |
| Node | `nodeenv` into `./.nodeenv` (user-space). Fallback: Docker `node:24` container running the same commands. |
| Showdown source | `third_party/pokemon-showdown` cloned by script at the commit in `SHOWDOWN_COMMIT`; git-ignored. |
| Package | `nuzlocke_gauntlet_rl` (README layout), `uv`-managed, Python 3.12. |
| Stats | v0: all mons IVs 31, EVs 0, nature Hardy, ability = first dex ability, no items. Trainer mons identical rules. Documented in `utils/specs.py`. |
| Config | YAML under `nuzlocke_gauntlet_rl/config/`; every species/move/ability validated against poke-env `GenData.from_gen(4)` at load. |

Non-goals for v0: doubles, items/shops, XP, ROM-hack mods, human-replay
pretraining, public ladder.

---

## 2. Contracts (implement exactly these signatures)

```python
# nuzlocke_gauntlet_rl/utils/specs.py
@dataclass(frozen=True)
class PokemonSpec:
    species: str                 # Showdown id, e.g. "starly"
    level: int
    moves: tuple[str, ...]       # Showdown ids, 1..4
    ability: str | None = None   # None -> first dex ability
    item: str | None = None

@dataclass
class MonInstance:
    uid: int
    spec: PokemonSpec
    level: int
    moves: list[str]
    alive: bool = True
    in_party: bool = False
    importance: float = 1.0      # set by run policy, read by battler reward

@dataclass(frozen=True)
class TrainerSpec:
    name: str
    team: tuple[PokemonSpec, ...]
    ai: str = "trainer"          # "trainer" | "hard"

@dataclass(frozen=True)
class RouteSpec:
    name: str
    encounters: tuple[tuple[PokemonSpec, float], ...]  # probabilities sum to 1
    unlock_after: int            # trainer index after which the route unlocks

@dataclass(frozen=True)
class MoveEventSpec:
    species: str; move: str; after_trainer: int; source: str  # "level_up"|"tm"

@dataclass(frozen=True)
class GauntletSpec:
    name: str
    starters: tuple[PokemonSpec, ...]
    trainers: tuple[TrainerSpec, ...]
    routes: tuple[RouteSpec, ...] = ()
    move_events: tuple[MoveEventSpec, ...] = ()
    max_roster: int = 12

def load_gauntlet(path: str | Path) -> GauntletSpec   # validates against Gen 4 dex, raises ValueError with the offending field
```

```python
# nuzlocke_gauntlet_rl/utils/teams.py
def spec_to_teambuilder_pokemon(spec: PokemonSpec, level: int | None = None, moves: Sequence[str] | None = None) -> TeambuilderPokemon
def mons_to_packed_team(mons: Sequence[MonInstance | PokemonSpec]) -> str   # uses Teambuilder.join_team
class QueueTeambuilder(Teambuilder):
    """yield_team() pops the next packed team pushed with push(team). Raises if empty."""
```

```python
# nuzlocke_gauntlet_rl/battle/result.py
@dataclass
class BattleResult:
    win: bool
    alive_mask: list[bool]       # same order as the party passed to simulate()
    hp_fraction: list[float]     # 0.0 for fainted
    turns: int
    battle_tag: str
def result_from_battle(battle: Battle, party_size: int) -> BattleResult
```

```python
# nuzlocke_gauntlet_rl/battle/battler.py
class Battler(ABC):
    def simulate(self, party: Sequence[MonInstance], enemy: TrainerSpec,
                 importance: Sequence[float] | None = None, blind: bool = False,
                 seed: int | None = None) -> BattleResult: ...
    def close(self) -> None: ...
```

```python
# nuzlocke_gauntlet_rl/utils/showdown.py
class ShowdownServer:
    """Context manager. Starts `node pokemon-showdown start --no-security PORT`
    from third_party/pokemon-showdown, waits until the websocket accepts a
    connection (timeout 120 s), kills the process on exit."""
    def __init__(self, port: int | None = None, log_path: str | None = None): ...
    port: int
    def server_configuration(self) -> ServerConfiguration   # ws://localhost:{port}/showdown/websocket
def free_port() -> int
```

Run-level env spaces (P3.3), fixed for v0:

* Observation `Dict`:
  `phase` `Discrete(4)` one-hot; `roster` `Box(max_roster × F_MON)`;
  `next_trainer` `Box(6 × F_MON)`; `trainer_idx` `Box(1)`;
  `pending_event` `Box(F_EVENT)`; `encounter` `Box(F_MON)`;
  `action_mask` `MultiBinary(A)`.
  `F_MON` = [species_id, type1_id, type2_id, level/100, hp_frac, alive,
  in_party, move_id×4, importance] (ids are integer indices into tables
  built from `GenData.from_gen(4)`; embed them in the policy).
* Action `Discrete(A)` with `A = max(max_roster + 2, 5)`, interpreted per phase:
  `PRE_BATTLE`: toggle mon *i* in/out of party, action `max_roster` = "confirm
  party" (≥1 alive mon required); `ENCOUNTER`: 0 skip, 1 add to box,
  2+i replace roster slot i; `MOVE_SELECTION`: 0 skip, 1–4 replace slot.
  The mask makes illegal choices impossible.
* Reward: `+1.0` per trainer beaten, `−death_penalty·importance` per death
  (default 0.5), `+clear_bonus` (default 5.0) on finishing, else 0.

---

## 3. Phase 1 — Toolchain and first battle

### P1.1 Project skeleton  [ ]
Create:
```
pyproject.toml   .gitignore   nuzlocke_gauntlet_rl/__init__.py
nuzlocke_gauntlet_rl/{utils,battle,envs,models,config}/__init__.py
tests/__init__.py   tests/conftest.py   scripts/   SHOWDOWN_COMMIT
```
`pyproject.toml` (uv, PEP 621): `requires-python = ">=3.12,<3.13"`,
dependencies `poke-env==0.16.1`, `numpy>=2.0`, `pyyaml>=6`;
dev group `pytest>=8`, `pytest-asyncio>=0.25`, `pytest-timeout>=2.3`,
`nodeenv>=1.10`, `ruff`. Configure `[tool.pytest.ini_options]`
`markers = ["integration: needs a local Showdown server"]`, `timeout = 300`.
`.gitignore`: `.venv/ .nodeenv/ third_party/ replays/ checkpoints/ runs/
__pycache__/ *.log .pytest_cache/ .ruff_cache/`.
`SHOWDOWN_COMMIT` contains `6b4bc34e44cc2541929cc4b8fff96e756ab3f268`.
**Verify:** `uv sync --all-groups && uv run python -c "import poke_env; print(poke_env.__version__)"` prints `0.16.1`.

### P1.2 Node + Showdown setup script  [ ]
`scripts/setup_showdown.sh` (bash, `set -euo pipefail`, idempotent):
1. `uv run nodeenv --node=24.20.0 --prebuilt .nodeenv` if `.nodeenv/bin/node` missing.
2. `git clone https://github.com/smogon/pokemon-showdown.git third_party/pokemon-showdown` if missing; `git -C third_party/pokemon-showdown checkout $(cat SHOWDOWN_COMMIT)`.
3. `cp -n config/config-example.js config/config.js` inside the clone.
4. `PATH="$PWD/.nodeenv/bin:$PATH" npm --prefix third_party/pokemon-showdown install`.
5. Print `node --version` and the Showdown commit.
`scripts/run_showdown.sh [PORT]`: `cd third_party/pokemon-showdown && exec ../../.nodeenv/bin/node pokemon-showdown start --no-security "${1:-8000}"`.
(Check `node pokemon-showdown --help` once: PORT is positional; `--no-security`
disables login and rate limiting. If the flag name differs at this commit,
fix the script and note it in §8.)
**Verify:** run setup, then `scripts/run_showdown.sh 8000 &`, wait, then
`curl -s http://localhost:8000/ | head -c 200` returns HTML; kill it.
Fallback if nodeenv fails: `docker run --rm -v $PWD/third_party/pokemon-showdown:/ps -w /ps -p 8000:8000 node:24 sh -c "npm install && node pokemon-showdown start --no-security 8000"` and document it.

### P1.3 `ShowdownServer` context manager  [ ]
Implement §2 `utils/showdown.py`. Readiness check: open a websocket to
`ws://localhost:{port}/showdown/websocket` with the `websockets` package
(already a poke-env dependency) and wait for the first message. Kill the
whole process group on exit. Add `tests/conftest.py` fixture
`showdown_server` (session scope, marker `integration`).
**Verify:** `uv run pytest -m integration tests/test_showdown_server.py`
starts a server on a free port, connects, stops it, and leaves no `node`
process behind (`pgrep -f pokemon-showdown` empty after the test).

### P1.4 Team packing  [ ]
Implement §2 `utils/teams.py` plus `utils/dex.py`:
`gen4() -> GenData`, `species_ids()`, `move_ids()`, `first_ability(species)`,
`to_id(name)` (use `poke_env.data.to_id_str`), and integer index tables
`SPECIES_INDEX`, `MOVE_INDEX`, `TYPE_INDEX` (sorted ids, 0 = padding).
Packed format (Showdown `sim/TEAMS.md`):
`NICK|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS]…`;
build it with `TeambuilderPokemon(...)` and `Teambuilder.join_team`, never by
hand. Set `level`, `ability`, `moves`, `ivs=[31]*6`, `evs=[0]*6`, `nature="Hardy"`.
**Verify (unit):** `tests/test_teams.py`: a 3-mon level-15 team packs to a
string with exactly 2 `]` separators, contains `|15|`, parses back through
`Teambuilder.parse_packed_team` with the same species, and
`load`-style validation rejects `"starlyy"` and move `"tackel"`.

### P1.5 Random-team generation utility  [ ]
`utils/showdown_cli.py`: `generate_random_team(format_id="gen4randombattle", seed=None) -> str`
running `.nodeenv/bin/node pokemon-showdown generate-team <format> [seed]`
in `third_party/pokemon-showdown` and returning the packed string;
`validate_team(packed, format_id) -> tuple[bool, str]` via `validate-team`.
**Verify (integration):** `generate_random_team()` returns 6 mons that
`validate_team(..., "gen4customgame")` accepts; a 3-mon level-15 team from
P1.4 is also accepted (this proves Custom Game allows short low-level teams).

### P1.6 First battles (smoke test)  [ ]
`tests/test_smoke.py` (integration): with `showdown_server`, create two
`RandomPlayer(battle_format="gen4customgame", team=QueueTeambuilder(...),
server_configuration=server.server_configuration(),
account_configuration=AccountConfiguration(<unique ≤18 chars>, None),
max_concurrent_battles=5, save_replays="replays/smoke")`, push 20 packed
3-mon teams to each, run `await p1.battle_against(p2, n_battles=20)`
(use `asyncio.run` in the test), assert `p1.n_finished_battles == 20`,
`p1.n_won_battles + p2.n_won_battles == 20`, and at least one `.html` in
`replays/smoke`. Print battles/sec.
**Verify:** test passes in < 120 s. Record battles/sec in §8.
Then tick the four "# 1" checkboxes in `README.md` in this commit.

---

## 4. Phase 2 — Battler interface and scripted battlers

### P2.1 `BattleResult` extraction  [ ]
Implement §2 `battle/result.py`. Map party order → `battle.team` order by
species+level (nickname-free packing keeps Showdown order = packed order;
assert it). `win = bool(battle.won)`, `turns = battle.turn`,
`alive_mask[i] = not mon.fainted`, `hp_fraction[i] = mon.current_hp_fraction`.
**Verify (integration):** after one battle, `len(alive_mask) == party_size`
and `win` agrees with `battle.won`; the loser has `all(not a for a in alive_mask)`.

### P2.2 Opponent AIs  [ ]
`battle/opponents.py`:
* `TrainerAI(Player)`: if `battle.available_moves` empty → best switch
  (highest level). Else score each move: `base_power × STAB(1.5) ×
  type_effectiveness(vs opponent_active_pokemon types, Gen 4 chart) ×
  accuracy`; status moves score 0 except on turn 1 with 20 % probability
  (mimics in-game "sometimes uses status"). Pick argmax; never voluntarily
  switch. Use `move.base_power`, `move.type.damage_multiplier(...)`.
* `HardTrainerAI = SimpleHeuristicsPlayer` (alias, imported from `poke_env.player`).
* `make_opponent(kind: str, **kw) -> Player`.
**Verify (integration):** `tests/test_opponents.py`, 100 battles each on a
fixed fair matchup: `RandomPlayer` vs `TrainerAI` win rate < 0.35;
`SimpleHeuristicsPlayer` vs `TrainerAI` win rate > 0.6. If the ordering
random < trainer < heuristic does not hold, fix the AI before continuing.

### P2.3 `HeuristicBattler`  [ ]
`battle/heuristic_battler.py`: implements §2 `Battler` using a persistent
pair (`SimpleHeuristicsPlayer` for us, `make_opponent(enemy.ai)` for them),
both with `QueueTeambuilder`s, one server configuration, unique usernames
per process. `simulate()` packs both teams, pushes them, runs
`battle_against(n_battles=1)` on the poke-env loop via
`asyncio.run_coroutine_threadsafe(..., POKE_LOOP).result()`, and returns
`result_from_battle(last_battle, len(party))`. `importance`/`blind`/`seed`
are accepted and ignored here (documented).
**Verify (integration):** `Battler.simulate` called 50 times in a loop
returns 50 results with no leaked battles (`len(player.battles)` bounded)
and no exceptions; average wall time per call logged.

### P2.4 Throughput benchmark  [ ]
`scripts/bench_throughput.py --servers K --concurrent C --battles N`:
starts K `ShowdownServer`s, runs `RandomPlayer` pairs round-robin, prints
battles/sec per configuration. Try K∈{1,4,8}, C∈{1,5,10}.
**Verify:** table written to `docs/benchmarks.md`. Decide from it how many
servers P4 will use. If best < 10 battles/sec, schedule P6.1 right after P3.

---

## 5. Phase 3 — Run-level environment with scripted battler

### P3.1 Specs, YAML loader, dex validation  [ ]
Implement §2 `utils/specs.py`. YAML schema example
(`config/gauntlets/toy_3.yaml`):
```yaml
name: toy_3
max_roster: 8
starters:
  - {species: turtwig, level: 10, moves: [tackle, withdraw, absorb]}
  - {species: starly,  level: 8,  moves: [tackle, growl, quickattack]}
trainers:
  - name: youngster_1
    ai: trainer
    team:
      - {species: bidoof, level: 7, moves: [tackle, growl]}
  - name: lass_1
    team:
      - {species: shinx,  level: 9, moves: [tackle, leer, charge]}
      - {species: budew,  level: 8, moves: [absorb, growth]}
  - name: rival_1
    team:
      - {species: chimchar, level: 12, moves: [scratch, ember, leer]}
      - {species: starly,   level: 10, moves: [tackle, quickattack]}
routes:
  - name: route_202
    unlock_after: 0
    encounters:
      - [{species: bidoof, level: 5, moves: [tackle]}, 0.6]
      - [{species: kricketot, level: 5, moves: [bide, growl]}, 0.4]
move_events:
  - {species: turtwig, move: razorleaf, after_trainer: 1, source: level_up}
```
Also create `config/gauntlets/sinnoh_early.yaml`: 8–10 trainers up to and
including the first gym (Roark: geodude 12 / onix 12 / cranidos 14), routes
201/202/203/204, starters = one of the three Sinnoh starters at level 10
plus a level-5 bidoof and a level-5 starly. Use public Platinum trainer data;
exact fidelity is not required, plausibility is.
**Verify (unit):** both YAML files load; a copy with `species: starlyy`
raises `ValueError` naming the field; encounter probabilities not summing to
1 (±1e-6) raise.

### P3.2 Observation builder  [ ]
`utils/observation_builder.py`: `mon_features(mon | PokemonSpec | None) -> np.ndarray(F_MON)`
(None → zeros) using `dex` index tables; `build_observation(state) -> dict`
matching §2 spaces; `F_MON`, `F_EVENT` exported constants.
**Verify (unit):** shapes match the declared spaces for a random state;
padding rows are all zero; species index is stable across calls.

### P3.3 `NuzlockeGauntletEnv`  [ ]
`envs/nuzlocke_env.py`, `gymnasium.Env`, constructor
`(gauntlets: Sequence[GauntletSpec], battler: Battler, cfg: RunConfig, seed=None)`.
State: `roster: list[MonInstance]`, `trainer_idx`, `phase`, `event_queue`,
`pending_encounter`. Phase machine exactly as README §5.2. `PRE_BATTLE`
uses toggle actions + confirm (§2). On confirm: party = alive mons with
`in_party`, call `battler.simulate(party, trainer, importance)`, apply
permadeath, reward, advance. `info` returns `{"trainer_idx", "deaths",
"alive", "cleared"}`. `reset()` samples a gauntlet uniformly with the env RNG.
**Verify (unit, with a `FakeBattler` that returns scripted results):**
`tests/test_nuzlocke_env.py`: (a) a mon reported dead never appears in a
later party (property test, 200 random episodes); (b) wiping ends the
episode; (c) clearing gives `clear_bonus`; (d) illegal actions are masked
(`action_mask` never allows confirming an empty party); (e) `gymnasium.utils.env_checker.check_env` passes.

### P3.4 Scripted run policies  [ ]
`models/run_policy/scripted.py`: `RunPolicy` protocol
`act(obs, info, env_state) -> int`; `RandomRunPolicy`; `GreedyRunPolicy`
(bring the 6 highest-level alive mons; catch if roster has room else skip;
learn a move if its base power exceeds the weakest current move's).
**Verify (unit):** with `FakeBattler` always winning with no deaths,
`GreedyRunPolicy` clears `toy_3` 100 %.

### P3.5 Evaluation script  [ ]
`scripts/run_evaluation.py --gauntlet <yaml>... --policy greedy|random
--battler heuristic --episodes N --servers K --seed S` → prints and writes
JSON to `runs/eval_<ts>.json`: success rate, mean trainer depth, expected
survivors, deaths/run, with 95 % bootstrap CIs. `utils/evaluation.py` holds
the metric code.
**Verify (integration):** `greedy + heuristic` on `toy_3`, 30 episodes:
success ≥ 0.9. On `sinnoh_early`, 50 episodes: success strictly between
0.15 and 0.7 — if not, **tune the YAML** (levels/team sizes) until it is,
and log the change in §8. Also `random` < `greedy` on `sinnoh_early`.

---

## 6. Phase 4 — RL battler with entropy-managed curriculum

### P4.1 Add RL dependencies  [ ]
`uv add stable-baselines3==2.9.0 sb3-contrib==2.9.0 torch tensorboard`.
**Verify:** `uv run python -c "import torch, sb3_contrib; print(torch.cuda.is_available())"` prints `True`.
If the resolver conflicts with poke-env's `gymnasium==1.3.0` pin, log it in
§8 and implement `models/ppo.py` (masked PPO, ~300 lines) instead.

### P4.2 `BattleEnv` (poke-env `SinglesEnv` subclass)  [ ]
`envs/battle_env.py`:
* Constructor takes `matchup_sampler: Callable[[], tuple[list[MonInstance], TrainerSpec]]`,
  `opponent_kind`, `blind`, `server_configuration`, `reward_cfg`.
  Uses `QueueTeambuilder`s for both agents; on every `reset()` draws a
  matchup, pushes both teams, stores the `TrainerSpec` for observation
  injection, then calls `super().reset()`.
* `embed_battle(battle) -> np.ndarray` (float32, fixed length):
  own 6 × [species_idx, type1, type2, level/100, hp_frac, fainted, active,
  status_idx, 7 boosts/6, move_idx×4, move_pp_frac×4, importance] +
  opponent 6 × [known-or-seen block same layout, `known` flag] +
  field [weather idx, own/opp hazards, turn/100] + `blind` flag.
  With `blind=False` the opponent block is filled from the `TrainerSpec`;
  with `blind=True` only from `battle.opponent_team`.
  Set `self.observation_spaces = {a: Box(low, high, shape) for a in self.possible_agents}` in `__init__` (read `env.py` to confirm the attribute the base class expects).
* `calc_reward(battle)`: use `self.reward_computing_helper(battle, fainted_value=…, hp_value=0.01, victory_value=1.0)` and add `Σ importance_i·alive_i / n` at battle end.
* `envs/masked_wrapper.py`: `MaskedSingleAgentEnv(gymnasium.Env)` wrapping
  `SingleAgentWrapper(env, opponent)`: observation = `obs["observation"]`,
  stores `obs["action_mask"]`, exposes `action_masks()` for `MaskablePPO`
  (`sb3_contrib.common.wrappers.ActionMasker` compatible).
**Verify (integration):** `check_env` (sb3) passes; 200 random masked steps
never raise; `action_masks().sum() >= 1` at every step; obs has no NaN.

### P4.3 Vectorised training script  [ ]
`scripts/train_battle_policy.py --level L0|L1|L2|L3 --gauntlet <yaml>
--servers K --envs N --steps S --opponent trainer|hard --blind --run-name`.
`SubprocVecEnv` with N envs; env *i* uses server `i % K` (started by the
script via `ShowdownServer`), unique usernames `b{pid}{i}` (≤ 18 chars).
`MaskablePPO("MlpPolicy", …)` with an embedding feature extractor
(`models/battle_policy/features.py`: `nn.Embedding` for species/moves/types,
concat with scalars). Log to TensorBoard under `runs/`, checkpoint to
`checkpoints/` every 100 k steps, eval callback = 100 battles vs the same
opponent every 200 k steps, `EvalCallback`-style JSON.
**Verify:** `--level L0 --steps 2000 --envs 2` runs end-to-end and writes a
checkpoint. Measure env-steps/sec with `--envs 16 --servers 4`; record in
`docs/benchmarks.md`. Target ≥ 50 steps/sec before L2.

### P4.4 Curriculum L0 → L3  [ ]
Matchup samplers in `envs/matchups.py`:
* **L0**: one fixed party vs one fixed trainer (`sinnoh_early` rival_1 vs
  the starters). **Gate:** win rate ≥ 0.9 vs `TrainerAI` within 1 M steps.
  If not reached, the env/obs/mask/reward has a bug — debug (compare with
  `HeuristicBattler`'s win rate on the same matchup) and do not continue.
* **L1**: fixed party, trainer sampled uniformly from the gauntlet. Gate: ≥ heuristic win rate − 5 pp.
* **L2**: party sampled from `config/rosters/pool20.yaml` (20 hand-built
  parties of 3–6 mons, levels 8–20), trainers sampled. Gate: ≥ heuristic.
* **L3**: party sampled from `runs/party_log.jsonl` written by P3.5
  (actual parties the run env produces). Gate: ≥ heuristic, and ≥ 0.6 vs
  `hard`.
Each level starts from the previous level's checkpoint. Keep `importance`
sampled uniformly in [0.5, 1] per mon from L1 on so the policy learns to
read it.
**Verify:** `docs/curriculum.md` with a table (level, steps, win rate vs
trainer, vs hard, heuristic baseline). All gates met.

### P4.5 Behaviour-cloning warm start (only if L2 or L3 stalls)  [ ]
`scripts/collect_bc.py` records `(obs, mask, action)` from
`SimpleHeuristicsPlayer` (or `SearchBattler` if P6.2 exists) over 50 k
steps; `scripts/bc_pretrain.py` trains the same policy network with masked
cross-entropy, saves weights loadable by `MaskablePPO.set_parameters`.
**Verify:** BC policy win rate ≥ 0.8 × teacher's; PPO from BC init beats
PPO from scratch at equal steps on the stalled level.

### P4.6 `RLBattler`  [ ]
`battle/rl_battler.py`: `Battler` backed by a `MaskablePPO` checkpoint
driving a poke-env `Player` subclass whose `choose_move` runs the policy
(`deterministic=True`) on `embed_battle` + mask. Same server/teambuilder
plumbing as `HeuristicBattler`.
**Verify (integration):** `run_evaluation.py --battler rl` on
`sinnoh_early` ≥ `--battler heuristic` on success rate (CIs reported).

---

## 7. Phase 5 — Run-level policy

### P5.1 Win-probability dataset + model  [ ]
`scripts/collect_winprob.py`: sample (party, trainer) pairs from the
gauntlets and roster pool, run `Battler.simulate` in parallel (K servers),
write `runs/winprob.parquet` with features + outcome + survivors.
`models/win_prob/model.py`: MLP over pooled mon embeddings; predicts P(win)
and expected deaths. Train/val split by gauntlet seed.
**Verify:** ≥ 20 k battles collected; val AUC ≥ 0.8; calibration plot saved.

### P5.2 `SearchRunPolicy`  [ ]
At `PRE_BATTLE` enumerate candidate parties (all subsets ≤ 6 when roster ≤ 10,
else greedy-add beam width 8), score = P(win) − λ·E[deaths·importance];
`ENCOUNTER`: catch if marginal best-party score improves or box has room;
`MOVE_SELECTION`: pick the move that maximises the estimator for the next
trainer. λ in config.
**Verify:** beats `GreedyRunPolicy` on `sinnoh_early` success rate with
non-overlapping 95 % CIs over 200 episodes.

### P5.3 `RLRunPolicy`  [ ]
`scripts/train_nuzlocke_policy.py`: `MaskablePPO` on `NuzlockeGauntletEnv`
with the frozen best battler; N envs each with its own battler/server; cache
`simulate` results keyed by `(party hash, trainer, seed)` for evaluation
reproducibility. Optionally initialise from `SearchRunPolicy` rollouts by BC.
**Verify:** ≥ `SearchRunPolicy` on held-out gauntlet seeds; ablations from
README §7 run by `scripts/run_ablations.py` and summarised in `docs/results.md`.

---

## 8. Phase 6 — Faster / searchable engine (optional, parallel after P3)

### P6.1 Stdio backend  [ ]
`battle/backends/stdio.py`: drive `node pokemon-showdown simulate-battle`
(protocol `sim/SIM-PROTOCOL.md`: write `>start {"formatid":"gen4customgame","seed":[…]}`,
`>player p1 {"name":"p1","team":"<packed>"}`, `>player p2 …`, then
`>p1 move 2` / `>p1 switch 3`; read `sideupdate`/`update`/`end` blocks) with
one subprocess per worker. Feed the `|request|` JSON into a poke-env
`Battle` object so existing players/embeddings work unchanged.
**Verify:** 1000 seeded battles, `HeuristicBattler` results identical between
websocket and stdio backends (same seed → same log); speed-up recorded.

### P6.2 `SearchBattler` on poke-engine  [ ]
`uv add poke-engine` (PyPI 0.0.48; build with the Gen 4 feature flag — read
its README for the exact `--config-settings`). Convert a poke-env `Battle`
to a poke-engine state, run expectiminimax depth 2–3 per turn.
**Verify:** ≥ `SimpleHeuristicsPlayer` win rate vs `hard` over 300 battles
on Showdown; disagreement rate between engines logged.

---

## 9. Pitfalls (read before each phase)

1. poke-env 0.16 RL classes: `SinglesEnv` (PettingZoo `ParallelEnv`,
   observations are `{"observation", "action_mask"}` dicts) and
   `SingleAgentWrapper`. `SingleAgentWrapper.reset` returns only the wrapped
   dict; you must unwrap for SB3.
2. poke-env pins `gymnasium==1.3.0`, `websockets==16.1.1`; SB3 2.9.0 accepts
   both. Do not add anything that pins gymnasium < 1.
3. `team=` is fixed at player construction; per-battle teams go through
   `Teambuilder.yield_team()` → `QueueTeambuilder`.
4. Usernames ≤ 18 chars, unique per server; `AccountConfiguration(name, None)`
   is enough with `--no-security`.
5. One Showdown process is single-threaded: scale with more servers on more
   ports, not more concurrency on one server.
6. poke-env runs its own loop thread (`poke_env.concurrency.POKE_LOOP`);
   call coroutines with `asyncio.run_coroutine_threadsafe(coro, POKE_LOOP)`.
   Never create players inside a different running loop.
7. Custom Game: level goes in the packed team (blank = 100); species/move
   ids must be lowercase Showdown ids; set an explicit ability.
8. Action masking is mandatory for PPO here; without it training "does
   nothing".
9. A gauntlet on which the greedy baseline scores 0 % or 100 % cannot
   discriminate policies; tune it first (P3.5).
10. Measure throughput (P2.4, P4.3) before any long run.
11. Never bump `SHOWDOWN_COMMIT` without green integration tests.

---

## 10. Decision log

Append entries as `YYYY-MM-DD — task — what changed and why`.

* 2026-09-08 — plan — initial version; versions verified against PyPI and
  GitHub on this date.
