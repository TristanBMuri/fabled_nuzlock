# HANDOFF — state of the project as of 2026-09-15

Read this first. `project_overview.md` is the plan, `README.md` → Status is the
full lab notebook (chronological, with every number). This file is the
compressed version: what exists, what was learned, what to do next.

## One paragraph

The project is a general Nuzlocke solver on Pokémon Showdown. The battler is
a behaviour-cloned prior plus rollout search over the real engine; the run
layer is currently a *scripted* planner that picks parties by simulated death
probability. Everything runs in-process (no Showdown servers). On the BDSP
slice through Gym Leader Roark (51 trainers), the current default config
clears **100% of runs with 2.40 deaths per run and zero lost battles**
(20 episodes, `logs/plan_cap_crn.log`; 2.65 before common random numbers).
The old ceiling of ~2 trainers was the environment's level rule, not the
policy. Next: evolution, then a learned run layer that must beat the
scripted one.

## Architecture (what each file is)

```
ai_util/sim_bridge.js   Node: Showdown's simulator over JSON lines. new / choose /
                        fork (+reseed or fixed seed) / forcelose / state / close.
                        Teams go through the TeamValidator like the server (252-EV fill).
ai_util/sim.py          SimBridge (process), SimBattle (one battle + two poke-env
                        Battle mirrors fed from the per-side protocol stream),
                        dedupe_nicknames, player_brain. Fork replays history;
                        fork(seed=) gives reproducible chance events; play(until=).
ai_util/evaluate.py     evaluate(): P(win), P(death_i), risk for one matchup.
                        Searcher: top-k rollout search over the prior, common
                        random numbers across candidates (default). outcome_value
                        is the objective. --horizon/--value paths = null result (README 2l).
ai_util/battler_policy.py  MaskedActorCriticPolicy, PolicyPlayer, load_policy,
                        action_mask / order_to_action (identity-based; poke-env's
                        are species-based and wrong with duplicate species).
ai_util/battler_obs.py  1014-dim observation encoder (unchanged).
ai_util/collect_game.py collect (sim, BC pairs from harvest matchups, + battle
                        outcome per decision as `value`), collect_search
                        (search-labelled pairs + search-leaf outcomes `leaves.npz`),
                        holdout, baseline.
ai_util/pretrain_bc.py  masked-CE pretrain; multiple --data; sim-based eval;
                        --init/--value-only/--value-coef train the critic on `value`.
ai_util/diff_sets.py    sim-vs-server team-setup diff (found the EV bug).
scripts/gauntlet.py     BDSP data, Mon/pack_team, evolve(), BattleRunner (server,
                        legacy), SimBattleRunner (default): run(), estimate(),
                        boss search. --build-moves writes moves + evolutions json.
scripts/nuzlocke_env.py NuzlockeEnv: slice_len trainers (51 default), encounters,
                        roster ≤12, party ≤6, permadeath, level_rule cap with
                        evolution, risk() query, risk features in obs, search_all.
scripts/plan_gauntlet.py Scripted planner + evaluation harness + death histogram;
                        --slice N.
scripts/train_nuzlocke.py PPO meta-policy trainer (sim backend; NOT retrained yet
                        on the new obs — old checkpoint is skipped on resume).
tests/test_sim.py       bridge: play, mirror, fork. `uv run pytest -q`.
```

## The default run config (what "a run" is now)

```bash
uv run python -u scripts/plan_gauntlet.py --episodes 20 --seed 300 \
    --policy models/game_policy_cap_best.pt --search 3,8 --search-all
```

Per trainer: catch encounter if room → candidate parties = top-6-by-level ∪
single swaps with the box → each scored by 8 sim battles (P(win) − mean
P(death)) → best party fields → battle runs with the prior proposing top-3
actions and 8 rollouts per candidate (same 8 engine seeds for every
candidate) picking one → deaths applied. Level rule: every mon at the next
boss's max level (14 = Roark). ~85 s/episode.
`--fast` = top-6-by-level, no search on routes, ~1 s/episode (harvesting).

## Findings (in order of importance)

1. **The level rule set the old ceiling.** "Match max enemy level" ⇒ parity
   with every route trainer ⇒ dead by trainer 2 regardless of battler (that
   was run 4's 2.1). `level_rule="cap"` (hardcore nuzlocke) ⇒ heuristic
   clears 75–95%. Margin sweep: 0 → 2.0 cleared, 2 → 18.6, 4 → 50, 6 → 50.
2. **Search everywhere is the battler.** Same prior: 52% clears / 8.6 deaths
   without search → 95% / 4.1 with search on every trainer (k=3, r=8).
   Search at bosses only did nothing because deaths happen on routes too.
3. **Risk-based party selection halves deaths again.** Heuristic battler,
   same seeds: 6.78 → 2.20 deaths/run; with prior+search: 4.10 → 2.65 and
   0 lost battles. Fielding by death probability is the run layer's lever.
4. **The battler must follow the run layer's distribution.** The prior
   trained on old-rule data was OOD under the cap (45% clears vs heuristic
   75%); regenerating the harvest under the cap and retraining took ~15 min
   server-free and closed most of it (52%). Any progression change ⇒ redo.
5. **Expert iteration v1 was null.** 14k search-labelled pairs (Roark, Hiker
   Daniel) + 104k teacher pairs → no change in the gauntlet. Search's value
   comes from the engine, not from what the prior can absorb at this size.
6. **Step-1 search verdict (matchup level):** search 70.3% / 0.33 deaths per
   mon vs prior 68.3% / 0.40 on holdout ×3. Win gain within noise, death
   reduction clear. r=16 = r=8 ⇒ the ceiling is the 1-ply rollout value.
7. **Where deaths happen:** Roark (#50) and Hiker Daniel (#32, 2× Geodude /
   Machop / Bidoof @11). Everything else is noise. `plan_gauntlet.py` prints
   the histogram.
8. **Value estimate (2l):** common random numbers across candidates (same
   engine seed for rollout j of every candidate) 67.2% / 0.35 vs 63.7% /
   0.37 without, prior 59.2% / 0.44, on the match-rule holdout ×3. Free;
   default. A learned value head as the rollout leaf was null twice
   (teacher-outcome data 22.7%, on-policy search-leaf data 41.2% — below
   the prior alone; 44% / 57% argmax agreement with full rollouts). Do not
   retry without a different idea (bigger critic, far more leaf data, or a
   value that conditions on the candidate). Rollouts stay the value.

## Defects found and fixed (do not re-find these)

- **Server fills 252 EVs per stat** (Custom Game has no EV limit); raw sim
  used 0 ⇒ every mon weaker ⇒ 62% vs 69%. Bridge validates teams now.
- **poke-env collapses same-species mons** (team keyed by nickname). Harvest
  parties often have two of a species ⇒ 16% rejected choices, 6% defaults,
  "switch to an active Pokémon". Fix: `dedupe_nicknames` (Bidoof, Bidoof2)
  + pre-populate own team before any protocol line (poke-env's lazy path
  renames same-species entries and then KeyErrors).
- **poke-env's action mask / labels are species-based** ⇒ marks the active
  duplicate as switchable, labels switches as the first slot. Fix:
  identity-based `action_mask` / `order_to_action`. Rejections now 0.00%.
  This also silently threw inside the runner and counted runs as lost.
- **Slice had no boss**: Roark is order_index 50; SLICE_LEN was 50.
- **Searcher confidence gate** (0.9) skipped every decision — the converged
  BC prior is >0.9 almost always. Off by default now.
- **Early stop on the route holdout is useless** (saturates at 99%); train
  full epochs or build a boss-only holdout.
- **Obs risk queries in the planner** cost +50%; planner runs with
  `risk_features=False` and queries candidates directly.
- **Species names in packed teams** (found on the 186 slice): `Mon.pack`
  used `capitalize()`, so 'Mr. mime' ≠ Showdown's 'Mr. Mime'; Showdown
  renames a nickname equal to the species, poke-env keys the team by our
  nickname, and three Mr. Mimes collapsed (AssertionError, order 158 —
  every run "ended" there). Source formes 'Shellos (East Sea)' packed
  literally became species 'eastsea' (KeyError, order 162). Fix:
  `gauntlet.species_name` (pokedex display name, forme resolution);
  `tests/test_sim.py::test_duplicate_multiword_species`. A runner failure
  ends the episode as a loss, so a crash looks like a wall in the death
  histogram: check the log for `rebuild after` before believing a wall.

## Artifacts

- `models/game_policy_cap_best.pt` — 51-slice prior (cap-rule data, 104k pairs).
- `models/game_policy_cap186.pt` — 186-slice prior (`models/bc_game_20260915_194709`,
  91k pairs from `data/teams/gauntlet_harvest/battles_cap186.jsonl`, 35k matchups).
- `models/game_policy_cap_value.pt` / `_value2.pt` — same actor + trained
  critic (teacher outcomes / search leaves). Null as a search leaf; reference only.
- `models/bc_game_20260915_164023` (match, 178k) / `_164025` (cap, 127k) —
  BC pairs with the `value` column. `models/bc_search_20260915_1709{13,15}/
  leaves.npz` — 233k search-leaf (obs, outcome) samples.
- `models/game_policy_ei.pt` — expert-iteration v1 (no better; keep for reference).
- `models/game_policy_best.pt` — old prior (match-rule data); OOD now.
- `models/gauntlet_holdout_cap.json` — 200 held-out cap-rule pairs (easy: route
  trainers; heuristic 99%). Not a boss benchmark.
- `data/teams/gauntlet_harvest/battles_cap_all.jsonl` (24k, 50-slice),
  `battles_cap51.jsonl` (200 eps through Roark), `battles_cap_ei.jsonl` (merge).
  `battles.jsonl` (483k) is the old match-rule harvest.
- `logs/plan_cap_crn.log` — the current default run (`plan_cap_full.log` =
  same without CRN). `logs/search_{prior,crn,nocrn,h0,h3,h0_k6r16,h0_v2}.log`
  — the 2l holdout table. `models/` and `logs/` are gitignored.

## Next steps (roadmap step 2m onward, in order)

1. **Value estimate (2l) is done as far as cheap ideas go**: common random
   numbers are in; the learned leaf was null (finding 8). If the battler
   still needs to improve later, the remaining levers are deeper search
   (2-ply over the enemy's likely replies) or a candidate-conditioned value.
2. **Evolution is in** (level-up only; `data/evolutions.json`, move table to
   L50) and **the slice extends**: `plan_gauntlet.py --slice 186` = through
   Gardenia (cap 22 after Roark). The OOD loop was run there (harvest
   `battles_cap186.jsonl` 35k → 91k BC pairs → `models/game_policy_cap186.pt`,
   val CE 0.102). Table, 10 eps, search-all + risk parties:
   **162.9 cleared / 63.5 no-death / 12.0 deaths / 0% clears** (heuristic
   fast: 116.7 / 34.6 / 11.0 / 0%). Every run is a roster wipe between #154
   and #170: Collector Edwin #139 (3x Sudowoodo@19), then the Eterna stretch
   #158–166 (Jupiter, rival with 4 mons at 19–21) grinds the L22 roster of
   route catches down. That stretch is the new frontier. Levers, in order:
   (a) roster quality — encounters stop at trainer 44 (12 hand-written
   routes; add Eterna Forest / Route 205 N / 211 tables with real levels),
   (b) the run layer's party choice when candidates all score +1.00 at n=8
   (raise RISK_BATTLES or score by expected HP loss), (c) battler at parity
   (search depth). Also still owed: the 51-slice default with evolution
   (5/20 episodes done: all clear, 2.2 deaths).
   Pipeline commands (one job at a time — RAM):
   ```bash
   GAUNTLET_HARVEST=data/teams/gauntlet_harvest/battles_cap186.jsonl \
     uv run python -u scripts/plan_gauntlet.py --episodes 300 --seed 1000 --fast --slice 186
   uv run python -u ai_util/collect_game.py collect --jsonl data/teams/gauntlet_harvest/battles_cap186.jsonl --battles 20000 --workers 8
   uv run python -u ai_util/pretrain_bc.py pretrain --data models/bc_game_<stamp>/data.npz --epochs 20 --out models/game_policy_cap186.pt
   uv run python -u scripts/plan_gauntlet.py --episodes 10 --seed 300 --policy models/game_policy_cap186.pt --search 3,8 --search-all --slice 186
   ```
3. **Learned run layer (2e)**: retrain `train_nuzlocke.py` on the sim backend
   with the risk features; it must beat the scripted table (100% / 2.65).
   If it can't, the scripted planner *is* the run layer.
4. **Generality (step 3)**: second source game/hack, segment slicing, held-out
   gauntlets, memorisation probe. Nothing has been done here yet.
5. Later: move events/builds, blind regime (belief over sets), hack mods.

## Environment gotchas

- `git` is not on PATH inside the container this assistant ran in; commit
  from your own terminal. Ports 8000–8007 belong to the old project.
- No Showdown server is needed for anything in the sim path. The server
  path (`--backend server`, `collect_game.py baseline`) still exists for
  cross-checks: `./scripts/start_showdown.sh 1 8021`.
- The Showdown clone must be built (`dist/`) and `config/custom-formats.ts`
  present (`scripts/setup_custom_formats.sh`) — the bridge requires the
  `gen8customgamenogimmicks` format.
- Runs are stochastic across invocations even with the same `--seed` (the
  sim seeds battles fresh); compare 40-episode tables, not single runs.
- **RAM: one heavy job at a time.** Three 186-slice gauntlet runs plus a
  harvest and collectors in parallel exhausted 60 GB and forced a reboot
  (2026-09-15). A single default-config run is ~1 GB; check `free -g`
  before and a few minutes after launching anything.
- Harvest recording is opt-in: set `GAUNTLET_HARVEST=<fresh file>`; unset
  means no writes. Evaluation runs used to append to `battles.jsonl`.
- `--slice 186` episodes take ~260 s with search-all (vs ~85 s for 51).
