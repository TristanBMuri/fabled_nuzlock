# HANDOFF — state of the project as of 2026-09-26

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
under them (20 episodes, `logs/plan_cap186_best4prior.log`; replicate on the
final code `logs/plan_cap186_final.log`: 184.6 / 7.05 / 85%; before the env
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

**2026-09-18:** the three walls (Edwin, the 19–21 rival, Kenneth's Gyarados)
were level-sort artefacts — all sit after Gardenia in the real game. The
order file is now **route-ordered** (`scripts/build_order.py` from
Bulbapedia; 405 trainers to Cynthia; Roark #20, Gardenia #58, Maylene #166,
Cynthia #404; `SLICE_LEN` 21). **Every table above is on the old order and
must be regenerated** (harvest → collect → pretrain → table). Also found:
trainer EVs/IVs/natures exist in the sheet and are dropped by `Mon.pack`
(the sim plays every enemy at 252 EVs / 31 IVs); fix before regenerating.
The teacher-ceiling round (1a) is decided: depth-2 search is null, a
scripted **damage-calc bot** beats the teacher and search on the probes;
the next prior is a clone of it. Details in README → Status (2026-09-18).
Later the same day: trainer EVs/IVs/natures are in (`Mon.pack` writes packed
strings directly), the encounter tables cover the whole game from
Bulbapedia (43 areas), evolutions carry every method behind progression
gates (friendship = catch level + 10, stones after Eterna, trade never), and
the first route-order tables exist: through Maylene the prior+search line is
167 / 157.7 / 1.25 deaths / 100% — solved. **The benchmark is the full game
(`--slice 405`) from here.** A fast episode over 405 wipes at the League rival.
Then: water / fishing / gift encounters as per-method alternatives the run
layer chooses between, the rival fought once per battle (was three times),
box 50. Full-game tables on this env (`logs/plan_route405b_*.log`):
heuristic + risk parties 403.05 / 139.2 / 31.95 deaths / 10% clears (20 eps);
prior + search 403.4 / 184.8 / 27.6 / 0% (10 eps). Cynthia wipes most runs;
the E4 costs ~5 deaths each. That table is the benchmark now.
2026-09-19: the calc bot is the battler (alone: 404 / 222.7 / 15.9 deaths / 15%
clears; its clone + search 385.5 / 196.1 / 17.9 / 30%). **The League is a
no-heal block now** (`no_heal` entries; party locked; HP / status / PP
carried through the sim bridge) — tables before this are on full heals.

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
ai_util/battler_obs.py  1098-dim observation encoder (2026-09-19: + level and real stats per mon).
ai_util/calc_bot.py     CalcPlayer: the damage-formula battler (KO check, switch out of a
                        lethal hit, set up when safe); collect --teacher calc, plan --policy calc.
ai_util/collect_game.py collect (sim, BC pairs from harvest matchups, + battle
                        outcome per decision as `value`), collect_search
                        (search-labelled pairs + search-leaf outcomes `leaves.npz`),
                        synth (late harvest parties x given trainers' teams ->
                        harvest-shaped jsonl: the targeted-data step), holdout, baseline.
ai_util/pretrain_bc.py  masked-CE pretrain; multiple --data; sim-based eval;
                        --init/--value-only/--value-coef train the critic on `value`.
ai_util/diff_sets.py    sim-vs-server team-setup diff (found the EV bug).
data/bdsp/              the game folder: trainers.json, order.json (route order,
                        location + mandatory per entry), encounters.json (gates =
                        first trainer of the location), learnsets (levelup + tm per
                        species) / evolutions / species / moves, tms.json (TM gates),
                        items.json (held items: gate, copies, effect).
scripts/build_items.py  rebuilds items.json from Serebii's BDSP item list (Bulbapedia is
                        behind a Cloudflare challenge since 2026-09-26): fixed pre-League
                        sources only (no Grand Underground / Battle Park / shops / wild
                        holders / consumables); effect model = the ITEMS table in the script.
scripts/build_tms.py    rebuilds tms.json from each TM's Bulbapedia page: move -> TM number,
                        before_trainer (earliest fixed source on the route order), source.
scripts/build_order.py  rebuilds order.json + encounters.json from Bulbapedia (trainer
                        tables and Generation VIII catch tables per story visit; level
                        bands for twice-visited locations, story battles by trainer_id).
                        Then `gauntlet.py --build-moves` and `--build-species`.
scripts/gauntlet.py     load_game() (cached, the only reader of the folder), Mon/
                        pack_team, encounter_moves (best4 / last4; k=None = ranked pool),
                        assign_items (E9: greedy by item_value at fielding), evolve(), ability
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
                        --no-dupes, --moves sim (E7d: the sim re-picks moves at hard
                        fights, Mon.chosen), --workers N, --verbose (encounter trace).
tests/test_nuzlocke_env.py  env rules: queued offers, dupes clause, box cap.
scripts/train_nuzlocke.py PPO meta-policy trainer (sim backend; NOT retrained yet
                        on the new obs — old checkpoint is skipped on resume).
tests/test_sim.py       bridge: play, mirror, fork. `uv run pytest -q`.
```

## The default run config (what "a run" is now)

```bash
uv run python -u scripts/plan_gauntlet.py --episodes 20 --seed 300 --slice 59 \
    --policy models/game_policy_best4.pt --search 3,8 --search-all --deep 4,16 --workers 10
# defaults: --roster 50 (unbounded box), dupes clause on, --moveset best4, TMs on, items on
# the full-game benchmark (the table that matters): --slice 405 --policy calc --search 9,8 --search-all --deep 9,16 --workers 10
# --slice 59 = through Gardenia on the route order (was 186 on the level order);
# the best4 prior was cloned on the old order's harvest — regenerate before trusting a table.
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
- **A bare nickname next to a suffixed duplicate** ('Staravia' + 'Staravia2'):
  Showdown resolved 'switch Staravia' to the active Staravia2 by species and
  rejected it. `dedupe_nicknames` now suffixes every member (2026-09-19).
- **Slice had no boss** (old order): Roark was order_index 50; SLICE_LEN was 50.
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
- **Dupes clause one step deep, offers rolled early, League party rebuilt by
  value** (2026-09-23): a Crobat left Zubat catchable (`_owned_families`
  now walks the whole line); two areas gated at one trainer both rolled
  Magikarp (offers roll one at a time, `env._queue`); two value-equal mons
  made a 7-mon locked party (`Invalid action 6 ... [6 switches]` rebuilds —
  a crash counted as a Cynthia loss). Boxes had 2–5 Crobats before this.

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

## Next steps (2026-09-18)

1. ~~Trainer EVs/IVs/natures~~ done. ~~Encounters to the League, evolution
   methods~~ done. Player-side EV fill is a flat 85 (`PLAYER_EVS`) — revisit
   with the level rule.
2. **Full-game tables** (`--slice 405`): harvest → collect → pretrain →
   heuristic and prior+search tables. Find the real walls (expect the League
   rival, E4, Cynthia; Maylene's Lucario costs one mon per run).
3. ~~Clone the damage-calc bot~~ done and measured: the clone fits badly
   (val CE 0.455) and search over the calc bot itself is the better line
   (probes: rival 0.03 deaths/mon; full game: route ≈ 1 death/run). The BC
   prior is no longer on the path; the battler is `--policy calc --search 9,8`.
4. **The League** (`no_heal` block: party locked, PP carried, full heal
   between fights, one cap 66): E4 costs 3 / 5 / 16 deaths per 10 runs,
   **Cynthia wipes 8 of 10** (2 of those 10 were crashes; box full of
   Crobats — see defects). **Benchmark on the fixed env (2026-09-23,
   `logs/plan_league4_swap.log`): 403.8 / 368.0 / 5.9 deaths / 20% clears;
   Cynthia 35 deaths, 6 wipes; E4 8 / 8 / 4.** E8 (`--league 10`: the
   League party from every 6-subset of a 10-mon shortlist, scored on the
   block by progress + deaths) is null: 404.0 / 325.6 / 6.4 / 10%. Default
   off. **E7 TM movesets (default on, `--no-tms`; `data/bdsp/tms.json`
   from `scripts/build_tms.py`; best4 v2: power x accuracy x STAB x
   category fit, `NO_SLOT_MOVES`) is the first lever that moved Cynthia:
   404.6 / 358.6 / 5.2 deaths / 70% clears (`logs/plan_best4v2_tms.log`),
   Cynthia 2 losses in 10, E4 12 deaths per 10 runs. That is the
   benchmark table now.** Cynthia still costs 3.4 deaths per run. Player
   EV training (`--evs trained`, 252 attack / 252 Spe) measured the same
   day: 404.5 / 375.9 / 5.8 / 50%, Cynthia 5 losses — null-to-negative,
   flat stays. **E7(d) moves chosen by the sim (2026-09-26, `--moves sim`:
   one-slot swaps over the ranked pool at hard fights, accepted at n=8 and
   n=16 by the risk query, persisting in `Mon.chosen`): 404.2 / 392.7 / 6.1
   / 20% with the League opener, 404.4 / 354.1 / 5.1 / 40% without it
   (`logs/plan_tms_moves{,_noblock}.log`); Cynthia 8 / 6 losses vs 2.
   Negative: the calc bot picks moves for the calc bot, search play needs
   the ones it drops (set-up). Off by default.** **E9 held items (2026-09-26,
   default on, `--no-items`; `data/bdsp/items.json` from
   `scripts/build_items.py` — Serebii, since Bulbapedia is behind Cloudflare;
   `gauntlet.assign_items` greedy by a multiplier model): 404.8 / 402.8 / 4.3
   / 80% clears (`logs/plan_tms_items.log`); one pre-League death in ten
   runs, E4 6 deaths per 10 runs (was 12), Cynthia 36 deaths / 2 losses
   (flat). That is the benchmark table now.** Every env-side pre-League lever
   is in; Cynthia's 3.6 deaths per run are the whole problem. Probe from full HP at parity: calc bot 2.7%
   wins; legal EV training +1–3 points; her Garchomp and Lucario kill 1.5
   mons each per fight. Next, on the roadmap only (2026-09-19, not started):
   E8 League party building from the whole box, then E7 TM movesets with
   **reusable TMs** (decided), then E9 held items.
5. **Deletion pass (2½)**, 2026-09-23: `evaluate.py --depth/--horizon`, the
   leaf recorder and `collect_game collect_search` (expert iteration) are
   gone, `order_bylevel.json` too. Left: `collect_game`'s BC pipeline and
   `pretrain_bc` if the prior stays off the path (the calc bot is the
   battler; the BC files are the fallback, ~700 lines).

Not to retry (measured, README → Status): `--refine`, `--climb`, `--parties
beam|all`, lead order, a switch-out-after-a-hit wrapper, wider search at a
wall, a learned value head as the rollout leaf, more rival teacher data,
**depth-2 search** (2026-09-18, null at 6x), **calc-bot rollouts under
prior candidates** (worse than either alone), **League party from the
whole box scored by the fast brain** (E8, 2026-09-23, null: the fast
brain's block ranking does not transfer to search play at Cynthia),
**252 attack / 252 Spe player EVs** (2026-09-23, 70% → 50% clears),
**moves chosen by the fast brain's risk query** (E7(d), 2026-09-26, 70% →
20% / 40%; anything scored by the fast brain at the League transfers badly
to search play — three times now: E8, EVs, moves).

## Environment gotchas

- Ports 8000–8007 belong to the old project.
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
