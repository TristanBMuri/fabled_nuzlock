# HANDOFF — state of the project as of 2026-09-15

Read this first. `project_overview.md` is the plan, `README.md` → Status is the
full lab notebook (chronological, with every number). This file is the
compressed version: what exists, what was learned, what to do next.

## One paragraph

The project is a general Nuzlocke solver on Pokémon Showdown. The battler is
a behaviour-cloned prior plus rollout search over the real engine; the run
layer is currently a *scripted* planner that picks parties by simulated death
probability. Everything runs in-process (no Showdown servers). On the BDSP
slice through Gym Leader Roark (51 trainers), the default config cleared
**100% of runs with 2.40 deaths per run** (20 episodes, `logs/plan_cap_crn.log`;
stale since encounters v2). The working slice is now **186 trainers through
Gym Leader Gardenia**: evolution, roster graveyard, 22 encounter areas, and a
prior retrained on that distribution plus targeted data on the deadliest
fights give **168/186 cleared, 16 deaths, 10% clears** (10 episodes). The
old ceiling of ~2 trainers was the environment's level rule, not the policy.
Next: the rival fights at #164–166, then a learned run layer that must beat
the scripted one.

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
scripts/nuzlocke_env.py NuzlockeEnv: slice_len trainers (51 default), encounters
                        (data/encounters.json v2: 22 areas gated by level band
                        over the 186 order), roster ≤12 living + graveyard, party
                        ≤6, permadeath, level_rule cap with evolution, risk()
                        query, risk features in obs, search_all.
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
9. **Targeted teacher data beats everything else on a deadly fight** (186
   slice): the prior had 156/38k matchups vs Collector Edwin and played it
   at 71% vs the teacher's 89%; 3600 synthetic matchups (late parties x
   deadliest teams) → 44k pairs → 87%, +search 92%. Gauntlet 0% → 10%
   clears. Roster refills and HP-margin party scores did nothing.
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
- `models/game_policy_cap186v3.pt` — current 186-slice prior: `bc_game_20260915_220345`
  (96k, harvest v2 `battles_cap186v2.jsonl` 38k) + `bc_game_20260916_001825`
  (44k, `battles_hard_synth.jsonl`). `_cap186.pt` / `_cap186v2.pt` = earlier rounds.
- `models/edwin_pairs.json` — 30 late parties vs Collector Edwin: the matchup probe.
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

The roadmap in `project_overview.md` now also carries: **1a** (decide the
teacher ceiling: one new beyond-teacher attempt or accept it), **1b** (search
width per trainer), **2o** (sample-starved risk query and party choice),
**2p** (roster cap is an obs artefact), **2q** (owed measurements), a
**deletion pass (2½)** for the off-path and null-result code, and **3a**
(real BDSP data: the order file is level-sorted, encounters are hand-made,
non-level evolutions missing). Added 2026-09-16 after a project review.

1. **Value estimate (2l) is done as far as cheap ideas go**: common random
   numbers are in; the learned leaf was null (finding 8). If the battler
   still needs to improve later, the remaining levers are deeper search
   (2-ply over the enemy's likely replies) or a candidate-conditioned value.
2. **The 186 slice (through Gardenia, cap 22) is the working slice now.**
   Evolution, graveyard (dead mons free roster slots), encounters v2 (22
   areas gated by level band) and the OOD loop are in. Current prior:
   `models/game_policy_cap186v3.pt` = 96k base pairs (harvest v2) + 44k pairs
   on synthetic hard matchups (400 late-game parties x the 9 deadliest teams,
   `battles_hard_synth.jsonl`). Table, 10 eps, search-all + risk parties:
   **168.2 cleared / 44.8 no-death / 16.2 deaths / 10% clears** (v2 prior
   160.2 / 48.7 / 18.8 / 0%; heuristic fast 127.2 / 33.3 / 17.7 / 0%).
   What moved it: targeted teacher data on the deadly fights (Edwin alone:
   prior 71% → 87%, +search 92% vs teacher 89%). What did not: roster
   refills (more catches = more deaths), the HP-margin party score.
   Killers now: rival #164–166 (4 mons at 19–21 incl. an evolved starter;
   47 deaths / 10 runs), Edwin #139 (16), Roark (12), Craig #163, Gyarados
   #169. Levers left, in order: (a) boss-level search at the rival and
   breakpoints (k=4 r=16 was +2 pts on Edwin at 3x cost; `search=(k, r)`
   applies to breakpoints only when `search_all` is off — add the rival to
   the breakpoints or make search width per-trainer), (b) another
   targeted-data round on the rival (cheap, ~10 min), (c) a party layer
   that keeps type answers alive (Fighting/Grass for Sudowoodo, Electric
   for Gyarados) instead of top-6-by-level ∪ swaps. Still owed: 51-slice
   default with evolution + encounters v2 (20 eps) — the old 2.40 table is
   stale (10 pre-Roark encounters now, was 12).
   Pipeline commands (one job at a time — RAM):
   ```bash
   uv run python -u scripts/plan_gauntlet.py --episodes 300 --seed 1000 --fast --slice 186 \
     --harvest data/teams/gauntlet_harvest/battles_cap186v2.jsonl
   uv run python -u ai_util/collect_game.py collect --jsonl data/teams/gauntlet_harvest/battles_cap186v2.jsonl --battles 20000 --workers 8
   uv run python -u ai_util/collect_game.py collect --jsonl data/teams/gauntlet_harvest/battles_hard_synth.jsonl --battles 3600 --workers 8
   uv run python -u ai_util/pretrain_bc.py pretrain --data models/bc_game_<base>/data.npz models/bc_game_<hard>/data.npz --epochs 20 --out models/game_policy_cap186v3.pt
   uv run python -u scripts/plan_gauntlet.py --episodes 10 --seed 300 --policy models/game_policy_cap186v3.pt --search 3,8 --search-all --slice 186
   uv run python -u ai_util/evaluate.py --pairs models/edwin_pairs.json --n-pairs 30 --n 3 --seed 5 --policy models/game_policy_cap186v3.pt --search   # matchup probe
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
- Harvest recording is opt-in: `plan_gauntlet.py --harvest <fresh file>` (or
  `GAUNTLET_HARVEST=`); otherwise no writes. Evaluation runs used to append
  to `battles.jsonl`.
- `--slice 186` episodes take ~260 s with search-all (vs ~85 s for 51).
