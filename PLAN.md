# Nuzlocke Gauntlet RL — Implementation Plan

Status: proposal, 2026-09-08. Companion to the spec in `README.md`.
Everything in this document was checked against the *current* versions of the
tools (poke-env 0.16.1, Pokémon Showdown 0.11.11, Node ≥ 16) on 2026-09-08.
Previous attempts at this project died on stale APIs, unverified setup and
"build everything at once". This plan is built around small, testable steps
with explicit exit criteria and known-pitfall notes.

---

## 0. Decisions made up front (and why)

| Decision | Choice | Why |
|---|---|---|
| Generation | **Gen 4** (`gen4customgame` on Showdown) | No team preview (matches in-game), physical/special split, no mega/Z/dynamax/tera so the battle action space is only 10 (6 switches + 4 moves). It is the one gen supported by *all three* engines we may want: Showdown, `poke-engine` (Rust search engine, gens 4–8) and Metamon (human-replay datasets, gens 1–4 + 9). Renegade Platinum, the archetypal hard hack, is Gen 4. |
| Battle format | `gen4customgame` | Only Showdown format with no restrictions: any level, teams of 1–6 (rule `Max Team Size = 24`), illegal movesets allowed. This is what makes trainer-style teams (level 14 Starly, 3 mons) possible. Verified in `config/formats.ts`. |
| Battle engine | Showdown via poke-env, **behind a `Battler` interface** | Ground truth for mechanics. The interface lets us swap in a faster/searchable engine later without touching the run-level code. |
| First battler | Scripted heuristic, then search, then RL | The run-level environment must not block on the RL battler. A search-based battler also becomes the *teacher* for the RL battler (behaviour cloning warm start), which is the cheapest form of the "pretrain on good heuristics" idea from the README. |
| RL algorithm (battler) | PPO with invalid-action masking (`sb3-contrib` `MaskablePPO`) | Discrete masked actions, well tested, works with Gymnasium 1.x. No need for self-play/league: the opponents are *scripted trainers*, not humans. |
| Opponent AI | Our own `TrainerAI` player mimicking in-game AI, plus poke-env's `SimpleHeuristicsPlayer` as the "hard" AI | Nuzlocke difficulty comes from team composition and permadeath, not from clever opponents. In-game AI: highest expected damage with type awareness, rarely switches. |
| Information model | Env-side **known opponent team** injected into observations, with a `blind` toggle | The run env owns the `TrainerSpec`, so it can hand the battler the full enemy team (like a player reading a guide). No dependence on Showdown open-team-sheets, which Gen 4 lacks anyway. |
| Python / packaging | `uv`, Python 3.12, single package `nuzlocke_gauntlet_rl` | Matches README layout. Poke-env pins `gymnasium==1.3.0` and `websockets==16.1.1`; everything else must be chosen to be compatible with those pins. |
| Node.js | Installed **user-space via `nodeenv` into the uv venv** (fallback: Docker) | No Node on this machine and no sudo assumed. `nodeenv` gives one `uv run` toolchain for both Python and the Showdown server. |
| Showdown source | Cloned into `third_party/pokemon-showdown` at a **pinned commit** by a script; `third_party/` is git-ignored | Showdown moves fast and poke-env occasionally breaks on new commits. Pin, and bump deliberately. |

Non-goals for v0: doubles, items/shops/money, XP curves, ROM-hack mods, human
replay pretraining, playing on the public ladder.

---

## 1. Architecture

```
                      ┌──────────────────────────────┐
                      │   NuzlockeGauntletEnv (Gym)   │  run-level, phases:
                      │   PRE_BATTLE / ENCOUNTER /    │  party selection, catch/skip,
                      │   MOVE_SELECTION / DONE       │  move learning, permadeath
                      └──────────────┬───────────────┘
                                     │ simulate_battle(party, trainer, importance)
                      ┌──────────────▼───────────────┐
                      │   Battler (interface)         │  -> BattleResult(win, alive_mask,
                      │   HeuristicBattler            │     hp_left, turns, log)
                      │   SearchBattler (poke-engine) │
                      │   RLBattler (PPO policy)      │
                      └──────────────┬───────────────┘
                                     │ poke-env Player / SinglesEnv
                      ┌──────────────▼───────────────┐
                      │   BattleBackend (interface)   │
                      │   ShowdownWebsocketBackend    │  local server, --no-security
                      │   (later) ShowdownStdioBackend│  `simulate-battle` subprocess
                      └──────────────────────────────┘
```

Two independent tracks share the `Battler` interface:

* **Track A — battle policy.** Train `RLBattler` on `BattleEnv` (a thin
  Gymnasium env wrapping poke-env `SinglesEnv` + `SingleAgentWrapper`).
* **Track B — run-level policy.** Build `NuzlockeGauntletEnv`, first with
  `HeuristicBattler`, then with whatever battler is best.

Neither track can block the other. Both end in the same evaluation harness.

### Data model (`utils/specs.py`)

```python
@dataclass(frozen=True)
class PokemonSpec:      # static definition, also used for trainer mons
    species: str; level: int; moves: tuple[str, ...]
    ability: str | None = None; item: str | None = None
    nature: str = "Hardy"; ivs: int = 31; evs: int = 0   # simple, uniform for v0

@dataclass
class MonInstance:      # per-run instance
    uid: int; spec: PokemonSpec; level: int; moves: list[str]
    alive: bool = True; in_party: bool = False; importance: float = 1.0

@dataclass(frozen=True)
class TrainerSpec:      name: str; team: tuple[PokemonSpec, ...]; ai: str = "trainer"
@dataclass(frozen=True)
class RouteSpec:        name: str; encounters: tuple[tuple[PokemonSpec, float], ...]; unlock_after: int
@dataclass(frozen=True)
class GauntletSpec:     name: str; trainers: tuple[TrainerSpec, ...]; routes: tuple[RouteSpec, ...]; starters: tuple[PokemonSpec, ...]
```

Specs live in YAML under `config/gauntlets/` and `config/builds/`; a loader
validates every species/move/ability against Showdown's Gen 4 dex (poke-env
ships this data via `GenData`) so typos fail at load time, not mid-battle.

Conversion to Showdown: `specs_to_packed_team(list[MonInstance]) -> str`
(Showdown packed format; level and fewer-than-6 mons are legal in Custom Game).

### Battle result contract

```python
@dataclass
class BattleResult:
    win: bool
    alive_mask: list[bool]      # same order as the party passed in
    hp_fraction: list[float]    # remaining HP per party mon (0 for dead)
    turns: int
    log: str | None             # Showdown log for debugging/replays
```

---

## 2. Milestones

Each milestone has a **done-when** that is an executable check. Do not start
the next milestone until the current one's check passes in CI (`pytest -m
integration` spins up the local server).

### M1 — Toolchain and a battle you can watch (≈1–2 days)

1. `uv init`, `pyproject.toml` with `poke-env==0.16.1`, `numpy`, `pyyaml`,
   `pytest`, `pytest-asyncio`. Python 3.12.
2. `scripts/setup_showdown.sh`: installs Node via `nodeenv` into `.venv`,
   clones Showdown at the commit in `third_party/SHOWDOWN_COMMIT`, runs
   `npm install`, copies `config/config-example.js` → `config/config.js`.
3. `scripts/run_showdown.sh [port]`: `node pokemon-showdown start --no-security --port N`.
   `--no-security` = no login, no rate limiting (this is the "no account" README item).
4. `utils/showdown.py`: `ShowdownServer` context manager that starts/stops a
   server subprocess on a free port and waits until the websocket answers
   (needed by tests and by multi-server training later).
5. `utils/teams.py`: `PokemonSpec → packed team string`, plus
   `QueueTeambuilder(Teambuilder)` whose `yield_team()` pops the next team.
   (poke-env asks the teambuilder for a team at every battle start; this is
   how per-battle teams work without recreating players.)
6. Smoke test: `RandomPlayer` vs `RandomPlayer`, 20 battles, format
   `gen4customgame`, each side a 3-mon level-15 team, `save_replays=True`.

**Done when:** `pytest -m integration tests/test_smoke.py` passes: 20 battles
finish, no exceptions, replay HTML files are written, battles/sec is logged.
Also generate a random Gen 4 team via `node pokemon-showdown generate-team
gen4randombattle` from Python (last README checkbox of milestone 1).

### M2 — Battler interface, scripted battlers, opponent AI (≈2–3 days)

1. `battle/battler.py`: `Battler` ABC with `simulate(party, enemy, importance, blind) -> BattleResult`.
2. `battle/opponents.py`: `TrainerAI(Player)` — Gen 4-style in-game AI:
   score = expected damage (use poke-env's damage estimate or a simple
   `base_power × STAB × type_eff × accuracy`), prefer KO, never switch unless
   forced; `HardTrainerAI` = poke-env `SimpleHeuristicsPlayer`.
3. `battle/heuristic_battler.py`: `HeuristicBattler` = `SimpleHeuristicsPlayer`
   playing our side through `Player.battle_against`.
4. `battle/result.py`: build `BattleResult` from the finished poke-env `Battle`
   (`battle.won`, `battle.team[...].fainted`, `current_hp_fraction`, `turn`).
5. Throughput benchmark script: N concurrent battles per player
   (`max_concurrent_battles`), K server processes. Record battles/sec on this
   box (32 cores). Expect tens of battles/sec with several servers; a single
   Showdown process is single-threaded.

**Done when:** `HeuristicBattler` beats `TrainerAI` ≥ 80 % over 200 battles
on a hand-made "fair" matchup, and `RandomPlayer` beats `TrainerAI` < 30 %.
(Sanity ordering: random < trainer AI < heuristics. If this ordering fails,
the AI or the result extraction is wrong; fix before anything else.)

### M3 — Run-level environment with a scripted battler (≈3–5 days)

1. `utils/specs.py` + YAML loader + dex validation.
2. `config/gauntlets/toy_3.yaml`: 3 trainers, 2 routes, 3 starters, ~10 species.
   `config/gauntlets/sinnoh_early.yaml`: first two gyms of Platinum-like
   content (8–10 trainers), built from public trainer data.
3. `envs/nuzlocke_env.py`: `NuzlockeGauntletEnv(gym.Env)` exactly as in
   README §5, with:
   * **Fixed-size party action**: Stage 0 uses `MultiBinary(max_roster)` masked
     to alive mons, capped at 6 (invalid → pick top-6 by index, log a flag).
     Keeps the action space simple; upgrade later.
   * `ENCOUNTER` action: `Discrete(2 + max_roster)` = skip / add to box /
     replace slot i.
   * `MOVE_SELECTION` action: `Discrete(5)`.
   * Observation: `Dict` with per-mon feature block (species id, types, level,
     alive, in_party, 4 move ids, hp), trainer index, next-trainer team block,
     phase one-hot, pending event block. All ids are integer indices into
     Gen 4 tables so policies can use embeddings.
   * Reward: `+1` per trainer beaten, `−death_penalty × importance` per death,
     `+run_bonus` on clearing, `0` otherwise. All coefficients in config.
4. Scripted run policies for baselines: `RandomRunPolicy`,
   `GreedyRunPolicy` (bring the 6 highest-level alive mons, always catch if
   box has space, always learn the higher-power move).
5. `scripts/run_evaluation.py`: runs N episodes with a (run policy, battler)
   pair over a gauntlet distribution and prints README §7 metrics
   (success rate, mean depth, expected survivors, deaths/run).

**Done when:** `GreedyRunPolicy + HeuristicBattler` clears `toy_3` ≥ 90 % of
episodes and reaches a *non-trivial* success rate (say 20–60 %) on
`sinnoh_early`. Non-trivial matters: if it is 0 % or 100 % the gauntlet is
mis-tuned and no policy can be evaluated on it. Also: a `MonInstance` that
died never appears in a later party (property test).

### M4 — RL battler, entropy-managed curriculum (≈1–2 weeks)

This is the part that killed earlier attempts, so it is deliberately staged.

1. `envs/battle_env.py`: subclass poke-env `SinglesEnv`; implement
   `embed_battle`, `describe_embedding`, `calc_reward`. Wrap in
   `SingleAgentWrapper(env, opponent=TrainerAI(...))`. Action space is
   poke-env's `Discrete(10)` for Gen 4; always pass the action mask from
   `SinglesEnv.get_action_mask` to `MaskablePPO`.
   Observation = own 6 mons (features + move ids) + opponent known/seen mons +
   field state (weather, hazards, boosts) + `importance` vector + `blind`
   flag. `blind=False` fills the opponent block from the `TrainerSpec`.
2. Reward: `+1 win / −1 loss`, `+Σ importance_i × alive_i / n` at the end,
   small per-turn HP-difference shaping (`0.01`, ablatable). Nothing else.
3. Curriculum, each level must pass before the next:
   * **L0 fixed matchup**: one party vs one trainer, `TrainerAI`.
     Done when win rate ≥ 90 % (heuristic battler's rate is the bar).
     If PPO cannot learn L0 in ≤ 1 M steps, the env/obs/mask is broken —
     stop and debug, do not scale.
   * **L1 fixed party, sampled trainers** from one gauntlet.
   * **L2 party sampled from a pool of 20 hand-built rosters**, trainers sampled.
   * **L3 party sampled from the run env's actual roster distribution**
     (log parties from M3 episodes and replay them).
   The README's key insight is honoured: the agent keeps a team for a while
   (each level fixes more than the previous one), and the entropy is raised
   only when the previous level is solved.
4. Warm start (only if L2/L3 stalls): behaviour cloning on
   (observation, action) pairs generated by `HeuristicBattler` or the
   `SearchBattler` (M6), then continue with PPO. This is the "pretrain on
   heuristics" option; it needs no external data.
5. Throughput: `SubprocVecEnv` with N envs, each env its own poke-env players
   with unique usernames, K Showdown servers on distinct ports (round-robin).
   Target ≥ 50 env-steps/sec aggregate before starting L2.

**Done when:** `RLBattler` ≥ `HeuristicBattler` win rate on the L3
distribution vs `TrainerAI`, and ≥ 60 % vs `HardTrainerAI`. Then plug it into
M3's evaluation and compare run-level metrics to the heuristic battler.

### M5 — Run-level policy (≈1–2 weeks)

1. **Win-chance estimator** `WinProbModel(party_feats, enemy_feats) -> p`
   trained supervised on outcomes from thousands of `Battler.simulate` calls
   (data generation is cheap and embarrassingly parallel). This is the helper
   policy the README asks for and it makes party selection a search problem.
2. `SearchRunPolicy`: at `PRE_BATTLE`, enumerate candidate parties (all
   subsets of size ≤ 6 for small rosters, else greedy/beam), score with
   `WinProbModel` × survival estimate, pick best. Same for encounters
   (marginal value of the new mon) and move choices.
3. `RLRunPolicy`: PPO on `NuzlockeGauntletEnv` with the battler frozen.
   Episodes are short (10–30 decisions) so this is cheap; the expensive part
   is the inner battles, cached by `(party, trainer, seed)` where possible.
4. Ablations from README §7 (with/without move events, encounters, penalty scales).

**Done when:** `RLRunPolicy` or `SearchRunPolicy` beats `GreedyRunPolicy` on
success rate over a held-out set of gauntlet seeds, with confidence intervals.

### M6 — Faster / searchable battle engine (side goal, parallel, optional)

* `ShowdownStdioBackend`: drive `node pokemon-showdown simulate-battle` over
  stdin/stdout (protocol in `sim/SIM-PROTOCOL.md`: `>start`, `>player p1 …`,
  `>p1 move 2`). No websocket, no server, one process per worker. Likely 5–10×
  faster than the websocket path and trivially parallel across 32 cores.
* `SearchBattler` on `poke-engine` (Rust, `pip install poke-engine`, PyPI
  0.0.48; build with the Gen 4 feature flag — verify the exact
  `--config-settings` at install time). Expectiminimax/MCTS gives a strong,
  deterministic, explainable battler and a teacher for BC. Caveat: it is an
  approximation of Showdown, so always **evaluate** on Showdown.

**Done when:** results of `ShowdownStdioBackend` and the websocket backend
agree on 1000 seeded battles, and the speed-up is measured.

---

## 3. Repository layout (matches README, concrete)

```
fabled_nuzlock/
├── README.md, PLAN.md, pyproject.toml, uv.lock, .gitignore
├── nuzlocke_gauntlet_rl/
│   ├── config/gauntlets/{toy_3,sinnoh_early}.yaml
│   ├── config/builds/gen4_builds.yaml
│   ├── utils/{specs,teams,showdown,dex,observation_builder,evaluation}.py
│   ├── battle/{battler,heuristic_battler,rl_battler,search_battler,opponents,result}.py
│   ├── envs/{battle_env,nuzlocke_env}.py
│   ├── models/{battle_policy,run_policy,win_prob}/
│   └── showdown/{server_config/,mods/}
├── scripts/{setup_showdown.sh,run_showdown.sh,bench_throughput.py,
│            train_battle_policy.py,train_nuzlocke_policy.py,run_evaluation.py}
├── tests/ (unit: specs, packing, masks, permadeath; integration: server)
└── third_party/pokemon-showdown   (git-ignored, pinned by SHOWDOWN_COMMIT)
```

---

## 4. Known pitfalls (read before writing code)

1. **Stale poke-env API.** Old tutorials use `Gen8EnvSinglePlayer` /
   `EnvPlayer`. In 0.16.x the RL classes are
   `poke_env.environment.SinglesEnv` (a PettingZoo `ParallelEnv` with two
   agents) and `SingleAgentWrapper` (Gymnasium single-agent view with a
   scripted opponent). Do not copy old code.
2. **Version pins.** poke-env pins `gymnasium==1.3.0` and `websockets==16.1.1`.
   Check that the chosen `stable-baselines3` / `sb3-contrib` release supports
   Gymnasium 1.3 before adding it; otherwise use CleanRL-style PPO in plain
   PyTorch (≈300 lines, no dependency conflict). Decide this in M4, not M1.
3. **Node is not installed** on the dev machine. `nodeenv` (Python package)
   installs Node into the venv without sudo. Showdown needs Node ≥ 16.
4. **Server flags.** Always `--no-security` locally. Multiple servers need
   distinct `--port`. Each poke-env player needs a unique username per server.
5. **Per-battle teams.** `team=` is fixed at construction; use a
   `Teambuilder` subclass whose `yield_team()` returns the next team.
6. **Custom Game details.** Level goes in the packed team; default level is
   100 if omitted. Teams may have fewer than 6 mons. No team preview in Gen 4.
   Species/moves must be Gen 4 dex names in Showdown id form
   (lowercase, no spaces).
7. **Action masking is mandatory.** Unmasked PPO in poke-env wastes most of
   its updates on illegal actions and looks like "RL doesn't learn".
8. **asyncio.** poke-env runs its own event loop thread (`POKE_LOOP`). Do not
   call its async APIs from a different loop; use `SingleAgentWrapper` /
   `battle_against` and keep one player set per process.
9. **Determinism.** Showdown battles are seeded per battle; poke-env does not
   expose the seed on the websocket path. For reproducible evaluation use
   many battles and CIs, or the stdio backend (M6) which accepts a seed.
10. **Mis-tuned gauntlets** make every policy look identical (0 % or 100 %).
    Tune difficulty until the greedy baseline lands in the middle.
11. **Throughput before scale.** Measure battles/sec in M2 and env-steps/sec
    in M4 before launching long runs. Add servers/processes, not hope.

---

## 5. Risks and kill criteria

| Risk | Signal | Response |
|---|---|---|
| RL battler cannot learn even L0 | < 90 % win after 1 M steps | Bug hunt in obs/mask/reward; compare with BC from heuristics; do **not** add complexity. |
| Battler generalises poorly across parties (L2/L3) | L1 fine, L2 flat | BC warm start from `SearchBattler`; larger embedding tables; more parties per rollout batch. |
| Showdown websocket too slow | < 10 battles/sec on 4 servers | Move to `ShowdownStdioBackend` (M6) early. |
| `poke-engine` diverges from Showdown for Gen 4 mons/moves | disagreement > 5 % on seeded battles | Use it only as a teacher/search prior, never for evaluation. |
| poke-env breaks on new Showdown commit | integration tests fail after bump | Revert `SHOWDOWN_COMMIT`; bump only with green tests. |
| Run-level RL is too sample-hungry | flat after 10 k episodes | Prefer `SearchRunPolicy` with the win-prob model; RL only as fine-tuning. |

---

## 6. Compute plan

Dev box: 32 cores, 60 GB RAM, RTX 5090. Showdown is CPU-bound and
single-threaded per server, so run 8–16 servers and 16–32 env processes;
the GPU is more than enough for the policies (small MLPs/transformers over
per-mon embeddings). Log with Weights & Biases or TensorBoard; keep
checkpoints out of git.

---

## 7. Immediate next actions (M1, in order)

1. `uv init --package nuzlocke_gauntlet_rl`, add pins, `.gitignore` with
   `third_party/`, `.venv/`, `replays/`, `checkpoints/`.
2. `scripts/setup_showdown.sh` + `SHOWDOWN_COMMIT` (record the commit tested).
3. `utils/showdown.py` server context manager.
4. `utils/teams.py` packing + `QueueTeambuilder`.
5. `tests/test_smoke.py` (integration) and `tests/test_teams.py` (unit).
6. Tick the four M1 checkboxes in `README.md`.

References checked while writing this plan:
poke-env 0.16.1 source (`environment/singles_env.py`, `env.py`,
`single_agent_wrapper.py`, `player/player.py`, `pyproject.toml`),
Showdown `config/formats.ts`, `COMMANDLINE.md`, `sim/SIM-PROTOCOL.md`,
`package.json` (`engines.node >= 16`),
[UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon),
[pmariglia/poke-engine](https://github.com/pmariglia/poke-engine),
[pmariglia/foul-play](https://github.com/pmariglia/foul-play).
