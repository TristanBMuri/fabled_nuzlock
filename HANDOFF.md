# HANDOFF — state of the project as of 2026-10-12

## Next task (2026-10-12): make the whole pipeline faster — progress

The goal and the two rules stand (profile first; every speed-up shown to change nothing on
paired seeds, or accepted as a change). Done on 2026-10-12 (NOTEBOOK.md → "Speed-up 1–3",
"Where a BDSP episode's time goes"):

- **RR: the unit of work is a pool job** (`scripts/rr_team_probe.py`: `fan` / `_POOL`, one
  boss context per worker). Boxes (`box_job`), a boss's stages A + B (`prelude_job`), every
  (team, battle) of stage C (`battle_job`) and the threat duels (`threats_job`) are each
  seeded on their own, so they spread over `--workers` (20 is the approved count, ~30 GB)
  with identical results: the box table 165 → 22 min, the pool table 151 → 25 min, both
  identical to the saved rows on every box / team / threat (`results/rr_cfru_*_par.*`).
  The early-boss table is the acceptance of the three-phase version (pending at hand-off
  time; see the notebook's last entry).
- **The damage formula is memoised** (`calc_bot.dmg_frac` → `_DMG`, exact by construction:
  the key is every input). -16% on a searched battle, same turns; applies to the calc bot,
  the RR AI and the Gen 4 AI alike.
- **Nothing to gain in the bridge**: Showdown's `Battle.choose` is 1.1 ms per turn in pure
  Node and the packaging 0.025 ms; a rollout turn costs the engine + ~0.4 ms of AIs + 0.2
  ms of mirror feeding. V8 GC flags change nothing.
- **BDSP, measured** (one timed Gen 4 AI episode, seed 300): 2129 s = party choice 1632 s
  (46,614 risk queries × 8 fast battles; grows with the box, ~6 × box candidates per
  trainer) + fights 496 s (395 fights, 14 deep). `--workers 20` (one episode each) bounds
  the benchmark by the slowest episode, ~36 min instead of 99.
- **`--swaps hard`** (`plan_gauntlet.py`, default `all` = unchanged behaviour): score the
  single swaps only when the cover six's own query is not clean (a loss, a death or HP <
  DEEP_HP: the deep-search trigger, already cached). Measured (NOTEBOOK.md, last entry,
  `results/bdsp_gen4ai_swapshard.*`): **17 min** for the 20-episode Gen 4 AI table on 20
  workers (99 on 7 with `all`; ~36 estimated on 20); League rows within the noise floor
  (404.55 / 390.10 / 4.50 / 55% vs 404.65 / 403.00 / 3.95 / 65%); the route party differs
  in 45% of fights; **4 route deaths at Candice in 3 of 20 episodes** where the saved run had 0.
  The flag's final form keeps the full swap scoring at every breakpoint and prunes only
  clean route trainers (`results/bdsp_gen4ai_swapshard2.*`): 17 min, 404.75 / 394.70 /
  3.80 / 75%, 3 route deaths (2 at Candice — the same process as `all` there — and 1 at a
  pruned route trainer). Recommendation: `all` stays the benchmark default, `hard` is the
  development setting (2x). **Decided 2026-10-12: the benchmark stays on `all`.**
- The paired route check for any BDSP run is the events file: per seed and trainer the
  party fielded, the result, the deaths (the compare was a scratch script: per seed, the
  first trainer where the fielded party differs, the per-episode means, deaths before
  the League).
- `league_test.py` now applies `--enemy` (it was parsed and ignored before; every League
  replay so far was against SimpleHeuristics) through `plan_gauntlet.apply(args)`.

User decisions 2026-10-12: the BC prior pipeline is deleted (HUMAN_TODO 5: `ai_util/battler_*`,
`collect_game.py`, `pretrain_bc.py`, the prior branches, `--harvest`, torch and stable-baselines3;
`action_mask` moved to `ai_util/evaluate.py`; `Searcher(enemy, ...)` lost its policy argument);
`no_AI_code_here/` stays untouched; HUMAN_TODO 7–12 (the
Radical Red run facts) are answered there and all match the sim's assumptions — the box
draw with the dupes re-roll is the real run, the planned pool an upper bound, singles only,
95 bosses (+ optional bosses for items), natures free, any ability after Cerulean.

**B1 matrix calibration done, B2 formula terms in** (`scripts/rr_matrix_calib.py`,
`results/rr_matrix_calib*.*`, NOTEBOOK.md 2026-10-12/13): the paper tiers scored against
duels (safe cells now win 95% of them, were 85%; 7 false safes, were 32) and the formula's
first hit scored against the engine's (within a damage roll in 95% / 91% of cells at Brock,
were 73%). Found and fixed on the way: the mirrors held vanilla data for the 40 / 250 / 134
entries the mod changed (`rr_probe.inject_mod_data`); `dmg_frac` never floored (12% high at
L13-16) and had no items or attack abilities (Eviolite, Life Orb, Choice, Technician, Sheer
Force, Pure Power, Knock Off, Air Balloon, Psyshock, charge moves, typed Hidden Power); the
matrix's healer rule and Air Balloon. **The RR tables moved down with the more accurate
model** (boxes Brock 75 -> 70%, Misty 51 -> 40%): the same teams replayed on the same seeds
drop the same way, and the split (`results/rr_formula_split.log`) puts it on the RR AI port,
which scores with `dmg_frac` too and now computes damage the way the game does. BDSP paired:
League rows within noise, 7 route deaths in 20 episodes where the saved run had 0 (the split
for the Gen 4 AI is in the notebook). Every table before 2026-10-13 is on the old formula.
**B3 run 2026-10-13, not adopted** (NOTEBOOK.md last entry, `results/rr_duel_boxes*.*`,
`results/rr_berry_duel_boxes.*`): `--chooser duel` judges every cover cell by playing the
one-on-one (`rr_team_probe.cover_team`, ~4 ms a duel; `matchup.safe / race / edge` read `win` /
`hp` when a cell has them). Paired on the 30 boxes: Brock +5–7 points of wins, Misty 0 to −6
with deaths up (+0.06–0.07 per mon, deathless 3% → 1–2%): Carnivine (a gift, in every box) wins
its Politoed duel and is dead weight against Mantine / Clodsire / Toxicroak, and the cover's
tiers have no term for a liability. The paper chooser stays the default; the flag stays.
**Found on the way: Misty's Mantine holds a Wacan Berry** — every paper Electric one-shot on it
landed at half. Resist berries are in the formula (`calc_bot.RESIST_BERRY`) and the matrix
spends them like the balloon (`matchup.hits_to_ko`); Misty's race precision 71 → 78%, our first
hit within a roll 87 → 90%. The enemy's heal berries as HP were tried and dropped (false nones
58 → 105: the tiers' cap on our hits calls a 3/6 cell "none"). Every table before 2026-10-13 has
the Wacan error in Misty's Electric cells.
**No PP Ups (2026-10-14, the user's catch):** Showdown gave every move three PP Ups on both sides;
`sim_bridge.js` now resets every slot to the base PP before the first request (`basePP`, the
carry applied on top). Paired: within the noise floor everywhere, the stallers kill less (NOTEBOOK.md
last entry). **Baselines from here: `results/rr_pp_boxes.*` (RR boxes, paper chooser) and
`results/bdsp_pp.*` (BDSP, Gen 4 AI: 404.60 / 391.60 / 4.45 / 65%)**; every earlier table has
the PP Ups. Next (TODO.md): C1 player items; the tiers' hit cap (a 3/6 cell is an answer) when
the matrix comes up again.

Still open: the BDSP swap scoring was the only big lever left and it is a behaviour change
(above, declined); within an episode nothing exact is large (the memo is in; the engine is the
floor); after it, an episode is the League opener's block query (193 candidates × 5 fights
× 8) plus the deep-searched League fights. For RR nothing structural is left: a table's
wall-clock is its battles / 20.

Operational: wait on a job by the *python main's pid* (`ps -eo pid,comm,args`, comm =
python3, the `--out` path in args) — `$!` after `setsid nohup … &` is the short-lived
wrapper, and a `pgrep -f`/`grep` pattern in a watcher matches the watcher's own command
line (both bit on 2026-10-12).

## What changed on 2026-10-11/12 (details in NOTEBOOK.md, newest last)

- **RR enemy AI = CFRU's** (`ai_util/rr_ai.py`, `data/rr/AI_DIFF.md`): switching from
  `ai_switching.c` (damage-calc switch-in score, also after faints) and the move-score
  tier ladder by fighting-style class. A no-switch enemy doubles Brock's deaths, so the
  AI's shape matters. RR tables rerun on it (`results/rr_cfru_*`): Misty pool 73% / 3%
  deathless (was 100 / 30), boxes 51%; Brock pool 100% / 63% deathless; the early bosses
  stay 30/30. Unverified against the ROM (A2.2 needs the user's emulator recordings).
- **BDSP enemy AI = Gen 4 (Platinum) trainer AI** (`ai_util/bdsp_ai.py`, `--enemy bdsp`;
  default is still SimpleHeuristics): Basic + EvalAttack + Expert, Gen 4 switching and
  the post-KO pick, from pret/pokeplatinum. All trainers get those three flags (our data
  has no per-trainer AI flags). The enemy sees only revealed player moves (no
  `--full-info`, as in the benchmark).
- **BDSP team building fixed** (`plan_gauntlet.cover_base`): the base party was
  top-6-by-level = the first six caught (the level rule puts the whole box at the cap)
  and single swaps only moved one slot — Bibarel / Kricketune at Volkner with 38 in the
  box. Now the base is the matchup cover (`ai_util/matchup.cover`) from the whole box
  against the next trainer (all of the League at its opener), then single swaps.
  Every BDSP table before 2026-10-12 is on the old planner.
- **League = `locked`** (was `no_heal` in `order.json`): party locked from Aaron to
  Cynthia, full heal (HP, status, PP) between fights, the dead stay dead (PP carried
  before).
- **Dashboard** (`ui/`, `uv run python ui/build.py --serve`, :8765): RR probe runs plus BDSP
  gauntlet runs (`plan_gauntlet.py --out logs/X.jsonl` → `X.events.jsonl`: per-episode
  progress strips with deaths, deadliest trainers, deaths by species). Every Champion
  fight is recorded as `logs/trace_X_s<seed>_404.json`, watchable in Showdown's replay
  viewer from the episode row or the Traces tab. Compare / Deaths / Traces / Matrices are
  otherwise fed by the RR scripts only.
- **BDSP benchmark on the new planner** (20 episodes, seed 300, slice 405,
  `results/bdsp_{gen4ai,heuristic}.*`): Gen 4 AI 404.65 cleared / 3.95 deaths / **65%**
  runs cleared; SimpleHeuristics 404.95 / 1.85 / 95% (was 403.8 / 5.9 / 20% on 2026-09-23).
  **No death before the League in 40 episodes**; Cynthia is the wall (63 deaths / 7 losses
  under the Gen 4 AI). The jump confounds the cover base and the healed League. The
  benchmark is now the League under `--enemy bdsp` (~95 min on 7 workers).

## Before 2026-10-11


Read this first. `project_overview.md` is the plan, `NOTEBOOK.md` is the
full lab notebook (chronological, with every number), `results/` holds the
logs every table cites. This file is the compressed version: what exists,
what was learned, what to do next.

**2026-10-10 clean-up** (tag `pre-cleanup` = the tree before it): the legacy
PPO / self-play / server code is deleted (7.2k → 3.9k lines), `gauntlet.py` is
split (game logic / `runner.py` / `build_dex.py`), `NuzlockeEnv` is a plain
state machine (no Gym, no obs, no reward; the level rule runs on entering
PRE_BATTLE so `risk()` is read-only), the measured-null planner options are
gone, `Mon` compares by identity. Behaviour-preserving: the paired League
replay against `results/league_base.jsonl` is the acceptance check
(`results/league_cleanup.log`: +0.00 ± 0.07 block clears, +0.00 ± 0.28 deaths, paired over 20 boxes, 0 errors). Open: `HUMAN_TODO.md`.

## One paragraph

The project is a general Nuzlocke solver on Pokémon Showdown. The battler is
a behaviour-cloned prior plus rollout search over the real engine; the run
layer is a *scripted* planner that picks parties by simulated death
probability. Everything runs in-process (no Showdown servers). The working
slice is **186 trainers through Gym Leader Gardenia**; the default config
(`--search 3,8 --search-all --deep 4,16`, prior `game_policy_best4.pt`)
clears **185.95 trainers / 6.30 deaths / 95% of runs** with the box
unbounded, the dupes clause on and `best4` movesets with the prior cloned
under them (20 episodes, `results/plan_cap186_best4prior.log`; replicate on the
final code `results/plan_cap186_final.log`: 184.6 / 7.05 / 85%; before the env
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
the next prior is a clone of it. Details in NOTEBOOK.md (2026-09-18).
Later the same day: trainer EVs/IVs/natures are in (`Mon.pack` writes packed
strings directly), the encounter tables cover the whole game from
Bulbapedia (43 areas), evolutions carry every method behind progression
gates (friendship = catch level + 10, stones after Eterna, trade never), and
the first route-order tables exist: through Maylene the prior+search line is
167 / 157.7 / 1.25 deaths / 100% — solved. **The benchmark is the full game
(`--slice 405`) from here.** A fast episode over 405 wipes at the League rival.
Then: water / fishing / gift encounters as per-method alternatives the run
layer chooses between, the rival fought once per battle (was three times),
box 50. Full-game tables on this env (`results/plan_route405b_*.log`):
heuristic + risk parties 403.05 / 139.2 / 31.95 deaths / 10% clears (20 eps);
prior + search 403.4 / 184.8 / 27.6 / 0% (10 eps). Cynthia wipes most runs;
the E4 costs ~5 deaths each. That table is the benchmark now.
2026-09-19: the calc bot is the battler (alone: 404 / 222.7 / 15.9 deaths / 15%
clears; its clone + search 385.5 / 196.1 / 17.9 / 30%). **The League is a
no-heal block now** (`no_heal` entries; party locked; HP / status / PP
carried through the sim bridge) — tables before this are on full heals.
(2026-10-12: renamed `locked`; HP / status / PP fully healed between fights, party still locked.)

## Architecture (what each file is)

```
ai_util/sim_bridge.js   Node: Showdown's simulator over JSON lines. new (+carry) / choose /
                        fork (+reseed or fixed seed) / forcelose / state / close.
                        Teams go through the TeamValidator like the server (252-EV fill).
ai_util/sim.py          SimBridge (process), SimBattle (one battle + two poke-env
                        Battle mirrors fed from the per-side protocol stream),
                        dedupe_nicknames, player_brain. Fork replays history;
                        fork(seed=) gives reproducible chance events; play(until=).
ai_util/evaluate.py     evaluate(): P(win), P(death_i), risk, HP left for one matchup.
                        Searcher: top-k rollout search (all legal actions up to k under
                        the calc bot; the prior's top-k with a BC policy), common random
                        numbers across candidates. outcome_value is the objective.
ai_util/calc_bot.py     CalcPlayer: the damage-formula battler (KO check, switch out of a
                        lethal hit, set up when safe). The player brain: plan --policy calc.
data/bdsp/              the game folder: trainers.json, order.json (route order,
                        location + mandatory + locked per entry), encounters.json (gates =
                        first trainer of the location; per-method tables), learnsets
                        (levelup + tm per species) / evolutions / species / moves,
                        tms.json (TM gates), items.json (held items: gate, copies, effect).
scripts/build_order.py  rebuilds order.json + encounters.json from Bulbapedia.
scripts/build_tms.py    rebuilds tms.json (move -> TM, before_trainer, source) from Bulbapedia.
scripts/build_items.py  rebuilds items.json from Serebii (fixed pre-League sources only).
scripts/build_dex.py    rebuilds learnsets / evolutions / moves / species from the Showdown
                        dex (--build-moves, --build-species); --check validates the folder.
scripts/gauntlet.py     load_game() (cached, the only reader of the folder), Mon (eq=False:
                        identity) / pack_team, encounter_moves (best4 / last4 over level-up
                        moves + TMs in hand), assign_items (greedy by item_value at
                        fielding), evolve() with progression gates, ability slots.
scripts/runner.py       SimBattleRunner: run(boss=, search=(k,r), carry=) -> (won, alive,
                        per-mon state), estimate() / estimate_block() = the risk queries,
                        one Searcher per width; the champion traces.
scripts/nuzlocke_env.py NuzlockeEnv: the run as a state machine. ENCOUNTER answers queued
                        offers (one per method), PRE_BATTLE builds the party and fights;
                        permadeath, graveyard, dupes clause, box cap, level rule applied on
                        entering PRE_BATTLE (evolution, moves, EVs), the League as a locked
                        block (party locked, full heal between fights), risk(n=) read-only and cached,
                        deep=(k,r) at breakpoints and risk-flagged fights, Rules, tracker().
scripts/plan_gauntlet.py The scripted planner + the table: top-6-by-level + single swaps
                        scored by risk(); --slice, --policy, --search, --deep, --workers,
                        --verbose, --save-league DIR (pickles the opener state).
scripts/league_test.py  The League alone: replays Aaron .. Cynthia from saved openers R
                        times each, same flags; --vs pairs two variants by snapshot.
tests/                  env rules (queue, dupes, box, League identity, level rule timing,
                        snapshot round trip), the bridge (play, mirror, fork, seeds), the
                        calc bot's stat model vs the engine. `uv run pytest -q`.
results/                the logs behind every table; league_base.{log,jsonl} and
                        league_snapshots/ (20 openers, seeds 300-319) = the paired baseline.
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
- `results/plan_cap186v3_deep.log` — the current run config (deep 4,16), 10 eps.
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
- `results/plan_cap_crn.log` — the current default run (`plan_cap_full.log` =
  same without CRN). `results/search_{prior,crn,nocrn,h0,h3,h0_k6r16,h0_v2}.log`
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
4. **The League** (`locked` block, was `no_heal`: party locked, full heal of HP / status /
   PP between fights since 2026-10-12 (PP carried before), one cap 66): E4 costs 3 / 5 / 16 deaths per 10 runs,
   **Cynthia wipes 8 of 10** (2 of those 10 were crashes; box full of
   Crobats — see defects). **Benchmark on the fixed env (2026-09-23,
   `results/plan_league4_swap.log`): 403.8 / 368.0 / 5.9 deaths / 20% clears;
   Cynthia 35 deaths, 6 wipes; E4 8 / 8 / 4.** E8 (`--league 10`: the
   League party from every 6-subset of a 10-mon shortlist, scored on the
   block by progress + deaths) is null: 404.0 / 325.6 / 6.4 / 10%. Default
   off. **E7 TM movesets (default on, `--no-tms`; `data/bdsp/tms.json`
   from `scripts/build_tms.py`; best4 v2: power x accuracy x STAB x
   category fit, `NO_SLOT_MOVES`) is the first lever that moved Cynthia:
   404.6 / 358.6 / 5.2 deaths / 70% clears (`results/plan_best4v2_tms.log`),
   Cynthia 2 losses in 10, E4 12 deaths per 10 runs. That is the
   benchmark table now.** Cynthia still costs 3.4 deaths per run. Player
   EV training (`--evs trained`, 252 attack / 252 Spe) measured the same
   day: 404.5 / 375.9 / 5.8 / 50%, Cynthia 5 losses — null-to-negative,
   flat stays. **E7(d) moves chosen by the sim (2026-09-26, `--moves sim`:
   one-slot swaps over the ranked pool at hard fights, accepted at n=8 and
   n=16 by the risk query, persisting in `Mon.chosen`): 404.2 / 392.7 / 6.1
   / 20% with the League opener, 404.4 / 354.1 / 5.1 / 40% without it
   (`results/plan_tms_moves{,_noblock}.log`); Cynthia 8 / 6 losses vs 2.
   Negative: the calc bot picks moves for the calc bot, search play needs
   the ones it drops (set-up). Off by default.** **E9 held items (2026-09-26,
   default on, `--no-items`; `data/bdsp/items.json` from
   `scripts/build_items.py` — Serebii, since Bulbapedia is behind Cloudflare;
   `gauntlet.assign_items` greedy by a multiplier model): 404.8 / 402.8 / 4.3
   / 80% clears (`results/plan_tms_items.log`); one pre-League death in ten
   runs, E4 6 deaths per 10 runs (was 12), Cynthia 36 deaths / 2 losses
   (flat). That is the benchmark table now.** **2026-10-09: a 20-episode replicate of
   that table gave 55% clears / 9 Cynthia losses (README → League test); the
   League test (`scripts/league_test.py`, baseline 0.70 ± 0.07 block clears,
   3.12 deaths at Cynthia) is where League levers are measured now.** Every env-side pre-League lever
   is in; Cynthia's 3.6 deaths per run are the whole problem. Probe from full HP at parity: calc bot 2.7%
   wins; legal EV training +1–3 points; her Garchomp and Lucario kill 1.5
   mons each per fight. Next, on the roadmap only (2026-09-19, not started):
   E8 League party building from the whole box, then E7 TM movesets with
   **reusable TMs** (decided), then E9 held items.
5. ~~**Deletion pass (2½)**~~ done 2026-10-10 (see the top of this file). Left:
   the BC pipeline, pending the decision in `HUMAN_TODO.md`.

**Measured 2026-10-10 (League test, paired):** full enemy information for the
battler (`--full-info`: sets, items, abilities, real stats known from turn one)
= +0.05 ± 0.08 block clears, -0.05 ± 0.34 deaths — null at 80 blocks, sign
right; default off, the flag stays (`results/league_fullinfo.log`). The
information is wasted on the calc bot's rules; a scorer / search that uses it
is the open question, not the information itself.

**Selector transfer diagnostic (2026-10-10, `scripts/selector_probe.py`,
`results/selector_probe.log`):** the fast party score orders League parties
only coarsely (within-box Spearman -0.31 ± 0.08 vs search-play deaths; the
bottom of the ranking costs +1.6 deaths) and cannot separate the top: ranks
0 / 1 / 2 are tied (3.65 / 3.65 / 3.45 deaths), rank ~24 within a death. The
planner's pick is a coin flip among the top few dozen — the mechanism behind
E8 / EVs / E7d. Next lever: shortlist by the fast score, decide among the top K
by search-played blocks (K=4, R=2 ≈ 15 min per opener; test on the League test).

Not to retry (measured, NOTEBOOK.md): `--refine`, `--climb`, `--parties
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

## Radical Red (2026-10-10/12)

**2026-10-12: the enemy AI now switches the CFRU way** (`data/rr/AI_DIFF.md`, NOTEBOOK.md
2026-10-12) and scores moves by CFRU's tier ladder. Rerun on it (`results/rr_cfru_*`):
Misty pool 73% / 3% deathless (was 100 / 30), boxes 51%; Brock pool 100% / 63% deathless,
boxes 75%; the early bosses stay 30/30. The numbers below are on the old port.

The question the project exists for. State (NOTEBOOK.md → Radical Red
entries, newest last): real pools and TMs, legal moves, arenas, no EVs /
items, a random starter. Boss #1 Brendan is solved (30 boxes: 100% wins,
87% deathless). Brock: planned team 100% with 0.13-0.21 deaths per mon,
boxes 70%; Misty: planned 57-80%, boxes 44% (starterless runs). **Why not
deathless** (2026-10-11): attrition — mons kept in below half HP because the
search prices a death at ≈0.08 of a win and its rollout brain never retreats;
and teams without a safe answer to one enemy mon (Lunatone, Mantine). The
answer under test: `ai_util/matchup.py` (hits-to-KO matrix on the real sets,
`safe` / `race` tiers, `cover()` chooser), `calc_bot.MATCHUP` switching,
`--death-weight 6`. Same team and seeds at Misty: 8/12 → 12/12 wins, 47 → 17
deaths. **Cover chooser results (2026-10-11 evening, n=30):** Misty planned
pool 30/30, 0.23 deaths per mon, 30% deathless (was 67–80%); Brock unchanged
(30/30, 0.21, 23%: the fast-stage six beats the cover six, the deaths are
Cacnea under sand). Boxes with a starter, rarity-drawn: Brock 66% mean / 80%
≥50% / 3% deathless, Misty 58 / 67 / 5% — a random box lacks the counters the
cover wants ("most missed" column = what routing must catch). Pre-Brock
fights #2, #4, #7, #15 are 30/30 deathless. Field-aware vs blind under the
matchup rule, same team: Erika 12 vs 24 deaths (field wins), Brock 37 vs 20
(field loses; it retreats into a bench with no Cacnea answer). Data fixes:
the ROM dump's 12-character names map in `rp.dump_id` (986dda0); the RR AI
takes the recharge turn unscored (31edf3f); #14 Lass Relli has no team data.
Every singles fight from #2 to #16 is 30/30 (Archer 73%, Grunt 80%, Rival 93% deathless, the rest 100%; `results/
rr_cover_early_*`), field-aware as default, doubles, route trainers, the
`UI_TODO.md` dashboard.
Pieces: `pokemon-showdown-rr/` (ignored; clone + `node build` +
`config/custom-formats.ts` as README → Setup for the main clone),
`scripts/build_rr.py` → `data/rr/` (`order.json` incl. gifts, `tms.json` are
the progression), `scripts/rr_probe.py` (`pools()`, `sample_box()`,
`arena()`, `where()`, `choose_nature/ability()`), `scripts/rr_team_probe.py`
(`--legal --arena --boxes N --repair --blind-field`), `scripts/rr_postmortem.py`,
`ai_util/rr_ai.py`, `gauntlet.BANNED_MOVES`, `SimBattle(arena=)`.
**Data for the full run is collected (2026-10-10 evening): `data/rr/README.md`**
— the three official 4.1 sheets (`data/rr/sheets/`), the hardcore rules as
JSON (no EVs for anyone, 31 IVs, 82 banned player moves, ability swaps, 18
caps), the ROM dump mapped to FireRed maps (`fr_slots.json`), and the item
timeline (care packages). It also lists what the probes got wrong (enemy EVs
252 → 0, Eviolite/Life Orb not available early, no EV training exists) and
the gaps in order: doubles (Sabrina, Lorelei, ~20 fights), permanent arena
effects, trainer/encounter order by segment, player legality.
`HUMAN_TODO.md` 7–12 are the run-specific questions only the user can answer.

## Environment gotchas

- Ports 8000–8007 belong to the old project.
- No Showdown server is needed for anything; the server path is deleted
  (2026-10-10). A server cross-check, if ever needed again, is
  `node pokemon-showdown start --no-security --port 8020` + poke-env.
- The Showdown clone must be built (`dist/`) and `config/custom-formats.ts`
  present (`scripts/setup_custom_formats.sh`) — the bridge requires the
  `gen8customgamenogimmicks` format.
- Runs are stochastic across invocations even with the same `--seed` (the
  sim seeds battles fresh); compare 20-episode tables, League levers on
  `league_test.py --vs`, never single runs (±25 points at 10 episodes).
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
