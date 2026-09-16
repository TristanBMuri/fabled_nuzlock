# HANDOFF — state of the project as of 2026-09-16 (evening)

Read this first. `project_overview.md` is the plan, `README.md` → Status is the
full lab notebook (chronological, with every number). This file is the
compressed version: what exists, what was learned, what to do next.

## One paragraph

The project is a general Nuzlocke solver on Pokémon Showdown. The battler is
a behaviour-cloned prior plus rollout search over the real engine; the run
layer is a *scripted* planner that picks parties by simulated death
probability. Everything runs in-process (no Showdown servers). The working
slice is **186 trainers through Gym Leader Gardenia**; the default config
(`--search 3,8 --search-all --deep 4,16`, prior `game_policy_best4.pt`)
clears **185.95 trainers / 6.30 deaths / 95% of runs** with the box
unbounded, the dupes clause on and `best4` movesets with the prior cloned
under them (20 episodes, `logs/plan_cap186_best4prior.log`; before the env
upgrade the same recipe gave 178.6 / 13.4 / 40%, and 183.15 / 16.25 / 60%
with the box and clause alone; noise floor between identical runs: ±2
cleared, ±1.9 deaths, ±15 points of clear rate). On 2026-09-16 every
run-layer knob was measured null on that table (refine, climb, beam and
exhaustive candidate parties, lead order), the Gyarados wall was shown to
be the battler's at any search width, and the cause of the conserved
deaths was found in the **environment**: the 12-slot box drops 5–7 of the
21 encounters per run, movesets are level-up-only at the cap level. The
**environment upgrade (project_overview.md → 2¾)** is where the gains
came from: game folder (E1), known defects (E3), and above all `best4`
movesets (E4: deaths per run 16.25 → 6.30, clears 60% → 95%). Left: E5
trace, E2 rules object, E6 later.

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
                        synth (late harvest parties x given trainers' teams ->
                        harvest-shaped jsonl: the targeted-data step), holdout, baseline.
ai_util/pretrain_bc.py  masked-CE pretrain; multiple --data; sim-based eval;
                        --init/--value-only/--value-coef train the critic on `value`.
ai_util/diff_sets.py    sim-vs-server team-setup diff (found the EV bug).
data/bdsp/              the game folder: trainers.json, order.json, encounters.json,
                        learnsets.json, evolutions.json, species.json. A second game
                        is a second folder (gauntlet.GAME names the default).
scripts/gauntlet.py     load_game() (cached, the only reader of the folder), Mon/
                        pack_team, encounter_moves (last4 rule), evolve(), ability
                        slots, BattleRunner (server, legacy), SimBattleRunner
                        (default): run(boss=, search=(k,r)), estimate(), one Searcher
                        per width. --build-moves / --build-species regenerate the
                        folder's dex-derived files.
scripts/nuzlocke_env.py NuzlockeEnv: slice_len trainers (51 default), encounters
                        (data/bdsp/encounters.json v2: 22 areas gated by level band
                        over the 186 order), roster ≤12 living + graveyard, party
                        ≤6, permadeath, level_rule cap with evolution, risk(n=)
                        query, risk features in obs, search_all, deep=(k,r) width
                        at breakpoints and risk-flagged hard fights (DEEP_HP),
                        rules=Rules(level_rule, margin, box, dupes_clause, moveset),
                        pending offers queue, dropped list, tracker().
scripts/plan_gauntlet.py Scripted planner + evaluation harness + death histogram;
                        --slice N, --deep k,r, --parties swap|beam|all, --roster N,
                        --no-dupes, --workers N, --verbose (encounter trace).
tests/test_nuzlocke_env.py  env rules: queued offers, dupes clause, box cap.
scripts/train_nuzlocke.py PPO meta-policy trainer (sim backend; NOT retrained yet
                        on the new obs — old checkpoint is skipped on resume).
tests/test_sim.py       bridge: play, mirror, fork. `uv run pytest -q`.
```

## The default run config (what "a run" is now)

```bash
uv run python -u scripts/plan_gauntlet.py --episodes 20 --seed 300 --slice 186 \
    --policy models/game_policy_best4.pt --search 3,8 --search-all --deep 4,16 --workers 10
# defaults: --roster 30 (unbounded box), dupes clause on, --moveset best4
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

10. **Deaths are conserved across walls (2026-09-16).** Every party-side
    knob (climb, beam, exhaustive parties) halves Edwin's deaths and the
    saved mons die at the next wall (Gyarados #169), where no party in the
    box helps and no search width helps. What the box lacks is what the
    env dropped: 5–7 of 21 encounters per run at the 12-slot cap, and the
    level-up-only movesets. The env sets the ceiling, again (finding 1).

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
- **Two areas opening at one trainer lost an encounter** (the later offer
  overwrote the earlier; never fired on the current order file) and **a
  full box dropped offers silently**: offers are queued (`env.pending`),
  drops are counted (`env.dropped`). The hand-made ability table was a
  no-op (equal to the dex's first slot for all 41 species): deleted.
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
- `models/game_policy_best4.pt` — prior cloned under the best4 moveset rule
  (`bc_game_20260916_205924` base 20k matchups + `_205946` synth 3600 from
  `battles_best4.jsonl` / `battles_best4_synth.jsonl`); val CE 0.146, switch 69%.
  Only valid with `--moveset best4`; v3 only with last4 (the obs encodes moves).
- `models/edwin_pairs.json` — 30 late parties vs Collector Edwin: the matchup probe.
- `models/rival_pairs.json` — 10 hard-synth parties x each rival variant (351/352/353).
- `logs/plan_cap186v3_deep.log` — the current run config (deep 4,16), 10 eps.
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

## Next steps (2026-09-16 evening; the plan is project_overview.md → 2¾)

1. **Baseline is base2**: unbounded box + dupes clause, 183.15 / 60.95 /
   16.25 / 60% (README → Status). Measure every later env change against
   it. E3e (ability slots) is in the code but not in that table; it rides
   with E4's table.
2. **E1 game folder is in** (`data/bdsp/`, `gauntlet.load_game`; moveset
   rule checked equal on 3300 cells); acceptance table 184.0 / 49.0 / 15.5
   / 60% = base2 inside noise, so E1 preserved behaviour and E3e (ability
   slots) is neutral. E3 is complete.
3. **E4 `best4` movesets: done and default** (185.95 / 101.8 / 6.30 /
   95%). Any future moveset change repeats the loop: harvest 300 fast eps
   with `--harvest`, collect base + synth on the deadliest teams, pretrain,
   table. The harvest carries the rule (collectors read packed teams).
4. **E5 and E2 are in** (`env.tracker()` under `--verbose`;
   `nuzlocke_env.Rules`). The env plan is complete except E6.
5. Deletion pass (2½) before step 3; `--parties beam|all` goes with it
   unless the lifted box changes the picture.
6. Then 1a (teacher ceiling decision), 2e (learned run layer, needs 2p),
   step 3 (generality).

Not to retry (all measured, 2026-09-16, README → Status): `--refine`,
`--climb` (deleted), `--parties beam|all` (null), lead order, a
switch-out-after-a-hit wrapper, wider search at the Gyarados fight, a
learned value head as the rollout leaf (null twice, 2026-09-15), more
rival teacher data.

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
- `--slice 186` episodes take ~260 s with search-all (vs ~85 s for 51), ~440 s
  with `--deep 4,16`. One planner process uses one core (Python + one Node
  sim, sequential); `--workers N` runs N episodes at once (~1 GB each).
