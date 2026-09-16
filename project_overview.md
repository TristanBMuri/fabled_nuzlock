# Nuzlocke Gauntlet RL

**Goal: a general Nuzlocke solver.** Given a gauntlet it has never seen
(ordered trainers, route encounter tables, permadeath, one catch per route),
clear it with as few deaths as possible. Battles run on a local Pokémon
Showdown + poke-env; no ROM, no emulator. One fixed mechanic set
(`gen8customgamenogimmicks`); ROM-hack mechanics are a later mod problem.

Reframed 2026-09-15 after the battler experiments (README → Status). What
changed: the battler is no longer "a policy that beats Showdown heuristics".
It is a learned prior plus search over the real engine, with a risk objective.
The run layer queries it for death probabilities instead of handing it a
party. Generality is a first-class constraint, not a stretch goal.

Side goal (unchanged): better Pokémon decision logic than Showdown's built-in
heuristics, both as opponent AI and as a teacher.

---

## 1. What a Nuzlocke is, as a problem

Normal Pokémon is resource-free: losses cost nothing. A Nuzlocke is a game of
**irreversible loss under uncertainty**, and that reshapes every layer.

Human view:
- Turn: not "best move" but "best move whose worst case I survive".
- Battle: who leads, who takes the hit, sacrifice ordering.
- Run: roster as a portfolio; box mons never risked, the ace protected for
  the Elite Four; encounters taken to hedge against death.
- Meta: knowing the game. Veterans win on prior knowledge, not reflexes.

ML view, same facts renamed:
- **Two timescales.** Hundreds of battles × tens of turns. A death at
  trainer 12 loses the run at trainer 40.
- **Risk-sensitive objective.** Maximise P(clear), not expected return. An
  expected-value agent takes 95% lines a hundred times and always dies.
- **Partial observability that is stable.** Enemy sets/AI are hidden per
  battle but fixed per gauntlet. That is exploitable memory, not noise.
- **Combinatorial state.** Species × sets × levels × roster. A policy that
  beat a pool of 4 teams scored 0/20 on unseen teams.
- **Sparse terminal signal.** "Cleared" is one bit at the end. Shaping
  produced degenerate optima twice (solo-starter, catch-and-bench, HP-farm).

---

## 2. Two information regimes

**Full docs (primary target).** Enemy sets, levels, items, and the AI's
scoring rules are known. The battle is a known model with stochastic
transitions (rolls, crits, misses, secondaries). A human "line" is a policy
tree found by hand search: trunk plus a few contingencies, the rest accepted
as risk. This regime is well-posed, cleanly evaluable, and what the hardest
hacks are designed for.

**Blind (extension).** Same planner, but a *belief* over enemy sets instead
of full info. Memory persists across attempts of the same gauntlet (attempt 4
knows the ace attempt 3 died to) and resets between gauntlets. The new
question is value of information: risk a mon now to scout a set for next
attempt?

The battler is not made redundant by lines:
1. Lines branch; something must execute the off-script tail, which is
   exactly where runs die.
2. Building a line is search, and search needs a prior (which 2–3 actions
   to expand) and a value (is this position winning). The prior makes
   search affordable; search makes the policy exceed its teacher.
3. Nobody builds lines for route trainers. One battler with a compute dial:
   shallow for routes, deep for bosses.

---

## 3. Knowledge split (the general-solver constraint)

Weights memorise anything stable across episodes. A fixed gauntlet is
entirely stable, so training repeatedly on one gauntlet bakes its docs into
the parameters even when no docs are given. The current meta-policy does
this with BDSP.

| Kind | Examples | Lives in |
|---|---|---|
| Game knowledge | type chart, damage, "L30 Gyarados probably has Waterfall" | weights (fine, desired) |
| Gauntlet knowledge | this trainer's sets, this route's table, gym 3 is Trick Room | explicit belief/memory, **never weights** |

Gradient descent does not respect this line on its own. Enforce it with:
- **Distribution training.** Every episode a different gauntlet, so
  memorising sets has no value and the weights learn the method.
- **Held-out real gauntlets.** Never trained on; the only place "general"
  is measured.
- **Memorisation probe.** Reroll sets + shuffle order on a training
  gauntlet, keep levels/species pools. If performance drops to the
  unseen level, the weights were carrying gauntlet facts. Build early.

Search-based battler helps: if the planner works from an explicit belief,
weights only carry the prior and search heuristics, not the answers.

---

## 4. Architecture: four layers

Only layers 2 and 3 get learning.

1. **Engine as model (not learned).** Showdown computes exact damage, speed,
   rolls. A battle is deterministic given seed + choice sequence, so a
   candidate line is evaluated by replaying under N seeds and counting
   deaths. Showdown's Node sim can also serialise/restore mid-battle state
   for search from a position. Feed the agent calcs (expected damage, KO
   prob, survival prob) as features rather than making it learn them.

2. **Battler = prior + search (learned prior, search on top).**
   - Prior: the BC policy (`models/bc_v5_256.pt` on randombattle,
     `models/game_policy_best.pt` on gauntlet matchups). It matches its
     teacher; RL finetune neither collapsed nor exceeded it (F1/F2).
   - Search: expand top-k actions per turn from the prior against the
     known/believed enemy AI, evaluate by seed replay. Depth = compute dial.
   - Objective: win while minimising importance-weighted deaths.
   - Output: `P(win)`, `P(death_i)` per party mon, and the line itself.
   - Beyond-teacher signal comes from search (expert iteration) if the
     prior needs improving; not from more PPO knobs.

3. **Run layer (learned, small).** Long horizon, tiny action space. For a
   boss, party and line are co-designed, so it does not hand the battler a
   party. It proposes candidate parties, gets back win/death probabilities,
   and chooses. Also: encounters, importance weights (which mon is
   sacrificable), later move events. Keep it small so it has little
   capacity to memorise.

4. **Route lookahead (not learned).** "Does a safe line exist for the next
   boss with my roster?" is answered by simulation, not a value net. Cheap
   given 24 Showdown servers.

**Interface between 2 and 3:**

```
evaluate(party, enemy, enemy_ai, importance) -> P(win), P(death) per mon, line
```

---

## 5. Benchmarks

Primary: **trainers cleared without a death** and **P(clear)** on the
gauntlet distribution, reported separately on train and held-out gauntlets.

Secondary: battler `P(win)` vs the enemy AI on harvested gauntlet matchups
(holdout pairs). Current: BC prototype 70.0% vs SimpleHeuristics baseline
68.5% (structural 68/32 first-side advantage).

Retired as a milestone: SimpleHeuristics on gen9randombattle. It is a
Showdown ladder benchmark, not this game. Keep as a regression check only.

---

## 6. Gauntlet distribution

For a general solver the distribution is the product, not a convenience.
Hack gauntlets are designed with intent (weather cores, Trick Room, bosses
built against the route pool), so pure random generation loses what makes
them hard. Build from real seeds:

- Sources: BDSP (have: 706 trainers, ordered, encounter tables), other
  mainline trainer lists (extractable per generation), hacks with published
  docs (Radical Red, Elite Redux, Emerald Kaizo, Unbound, Inclement Emerald).
- Slice into **boss segments** (boss + preceding routes). This is what
  `SLICE_LEN=50` already does with BDSP.
- Augment from real teams: swap species within level/role, reroll sets from
  learnsets, perturb levels, shuffle segment order.
- Hold out whole games/hacks. Ten sources × 8 gyms × a few dozen
  perturbations ≈ thousands of segments, the scale where meta-RL
  generalises.
- Boundary: one mechanic set. Hacks that change stats/abilities need
  Showdown mods; later.

---

## 7. Movesets, levels, encounters (unchanged, staged)

- **Stage 0 (current):** fixed movesets from learnsets at level, player level
  = highest enemy level seen, never decreases.
- **Stage 1:** a few predefined builds per species; env or agent picks.
- **Stage 2:** discrete `MoveEvent`s tied to gauntlet milestones;
  `MOVE_SELECTION` phase, action = skip / replace slot 1–4.
- Not modelled: TM inventory, shops, money, XP curves.

Env phases: `PRE_BATTLE` (party selection → battle → permadeath),
`ENCOUNTER` (skip / catch), `MOVE_SELECTION` (later), `DONE`.
Roster ≤ 12, party ≤ 6. See `scripts/nuzlocke_env.py`.

---

## Roadmap

Keep as simple as possible. Add complexity only on concrete failure.

### 0. Done (details in README → Status)
- [x] Showdown + poke-env, smoke test, single-file PPO, TensorBoard
- [x] BDSP gauntlet env with real trainer order + encounters; meta-policy
      trains (~2.1 trainers cleared, forced-full-party mask fixed alignment)
- [x] Throughput: one process + one Showdown port per env, 24 envs ~130 st/s
- [x] Battler from scratch: exhausted (PPO ceiling ~14% vs heuristics @ 6M)
- [x] BC from SimpleHeuristics: 40% vs teacher (randombattle), 70% on
      gauntlet matchups; RL finetune preserves but does not exceed init
- [x] Harvest: 483k real gauntlet matchups in
      `data/teams/gauntlet_harvest/battles.jsonl`
- [x] Radical Red / CFRU trainer AI port (`ai_util/heuristics_player/`)

### 1. Battler as prior + search
- [x] Line evaluator: `ai_util/evaluate.py::evaluate` → P(win), P(death_i),
      risk via the in-process sim (`ai_util/sim_bridge.js` + `ai_util/sim.py`,
      fork + reseed). Validated vs a live server on identical pairs
      (69.5% vs 69.0%) after two fixes: server-side EV fill, duplicate-
      species team collapse. ~0.01 s/battle.
- [x] Top-k rollout search over the BC prior vs SimpleHeuristics:
      `ai_util/evaluate.py::Searcher` (k, rollouts = the compute dial).
- [x] Risk objective: importance-weighted death penalty in `Searcher.value`.
- [x] Benchmark on holdout[:100]×3: search 70.3% / 0.33 deaths per mon vs
      prior 68.3% / 0.40. Consistent direction across all runs, win gain
      within noise, death reduction clear. Go (modest).
- [x] Run-layer integration (step 2), expert iteration v1 (null, 2i),
      common random numbers (2l), learned value leaf (null twice, 2l).
- [ ] **1a. The teacher ceiling — decide.** The prior is a clone of
      SimpleHeuristics; search recovers the teacher and adds a few points,
      but nothing learned has exceeded the teacher (RL finetune, expert
      iteration, value head all null). Two paths, pick one by measurement:
      (i) one more beyond-teacher attempt with a *new* idea — 2-ply search
      over the enemy AI's actual reply (it is deterministic given the state)
      as the label source, or search labels only on decisions where search
      and prior disagree, trained at higher weight; (ii) accept the ceiling:
      the battler is "teacher + search + targeted data", and all further
      effort goes to the run layer and data. Decision criterion: if (i)
      does not beat the search line on the Edwin/rival probes and the 186
      table within one round, take (ii) and delete the EI/value-head code.
- [x] **1b. Search width per trainer** (2026-09-16). `NuzlockeEnv(deep=(k, r))`
      / `plan_gauntlet --deep 4,16` plays breakpoints and every fight the
      fielded party's risk query flags (a loss, a death, or HP < `DEEP_HP` at
      n=8) at the deep width; routes stay at `--search`. The trigger is the
      sim, not a trainer list, so it carries no gauntlet facts. Rival probe:
      prior = teacher (0.37 vs 0.29 deaths/mon), search k3 r8 0.20, k4 r16
      0.16. 186 table, 20 eps: **178.6 cleared / 62.4 no-death / 13.4 deaths /
      40% clears** (10 eps without deep: 168.2 / 44.8 / 16.2 / 10%), 25 deep
      fights per run, +70% time. Edwin #139 is now a third of all deaths.

### 2. Run layer on the planner (plan written 2026-09-15)

Principle: build the *scripted* planner first and measure it. It is the
baseline any learned run policy must beat, it exercises the battler
interface end to end, and it needs no training run to produce a number.

- [x] **2a. Sim-backed BattleRunner.** `SimBattleRunner` in `gauntlet.py`,
      same contract; `--backend sim` default. Server path kept for now.
- [x] **2b. Death-probability query.** `SimBattleRunner.estimate` +
      `NuzlockeEnv.risk()`; 7 risk floats in the PRE_BATTLE obs.
- [x] **2c. Scripted planner** `scripts/plan_gauntlet.py`. Baseline table in
      README. **Result: the level rule, not the policy, set the old ceiling**
      (match rule: 2 trainers; cap rule: heuristic clears 95% of the slice).
- [x] **2d. Compute dial** (search on breakpoint trainers). No effect yet:
      deaths happen on routes, and the prior is OOD under the cap rule.
- [x] **2f. Progression model.** `level_rule="cap"` (player level = next
      boss's max level) is the default: hardcore-nuzlocke semantics, derived
      from the data, bosses at parity. `level_margin` kept as a knob.
- [x] **2g. OOD loop** done server-free in ~15 min: cap harvest (24k) →
      104k BC pairs → `models/game_policy_cap_best.pt`. Prior went from
      37/45% to 47.5/52% (heuristic 50.5/75%). Remaining gap is on routes
      (Hiker Daniel), not the boss. Slice extended to 51 to include Roark.
- [x] **2h. Risk-based party selection**: 2.20 vs 6.78 deaths/run, 98% vs
      85% slice clears, same battler and seeds (Finding 4 in README).
- [x] **2i. Expert iteration v1 — null.** 14k search labels + 104k teacher
      labels → no change in the gauntlet. Not pursued further for now.
- [x] **2i'. Search everywhere instead.** Cap prior + search on all 51
      trainers: 95% clears / 4.1 deaths vs 52% / 8.6 without search and
      75% / 7.0 for the heuristic. `--search-all` in plan_gauntlet,
      `search_all=True` in NuzlockeEnv. ~50 s/episode.
- [x] **2j. Planner cost**: planner runs with `risk_features=False`
      (it queries candidates directly); the obs re-queries were +50%.
- [x] **2k. Combined default measured** (20 eps): 100% clears, 2.65
      deaths/run, 0 lost battles. This is the baseline table step 2e's
      learned run layer must beat. Deaths now come from Roark (20) and
      Hiker Daniel (11) only.
- [x] **2l. Value estimate.** Common random numbers across candidates:
      67.2% vs 63.7% win, 0.35 vs 0.37 deaths/mon on the match-rule holdout
      (prior 59.2%); free, default now. Learned value head as rollout leaf:
      null twice (teacher-outcome data 22.7%, on-policy search-leaf data
      41.2% — both below the prior alone). Rollouts over the engine remain
      the value estimate. Gauntlet default with CRN: 100% / 2.40 deaths /
      36.25 no-death (was 2.65 / 33.8). Details in README → Status 2l.
- [x] **2m. Evolution.** Level-up evolutions from Showdown species data
      (`data/evolutions.json`, applied in `_apply_level_rule`); move table
      to L50. `--slice 186` reaches Gardenia. OOD loop rerun there
      (`game_policy_cap186.pt`). Table: 162.9 cleared / 63.5 no-death /
      12.0 deaths / 0% clears (heuristic fast 116.7 / 34.6 / 11.0). Runs end
      by roster wipe in the Eterna stretch (#139, #158–166, L19–21 vs our
      L22 route catches). Two packing defects fixed on the way (species
      display names, formes).
- [x] **2n. Roster + targeted data on the 186 slice.** Graveyard (dead mons
      free slots), encounters v2 (22 areas by level band): no gain by
      themselves (160.2 / 18.8 deaths). HP-margin party score: null.
      Targeted teacher data on the 9 deadliest teams (44k pairs) →
      `game_policy_cap186v3.pt`: **168.2 cleared / 16.2 deaths / 10% clears**;
      Edwin probe 71% → 92% with search. Next levers in HANDOFF next-steps 2.
- [ ] **2e. Learned run layer vs scripted.** Retrain `train_nuzlocke.py` on
      the sim backend with the 2b features. Success = beats the 2c table on
      trainers cleared without a death; otherwise the scripted planner *is*
      the run layer and effort goes to the battler. The 186 slice has the
      headroom (10% clears); the PPO obs/action layout must be rebuilt for
      it (see 2p) before training.
- [ ] **2o. Candidate parties.** The risk query sees 0 deaths in 8 battles
      for almost every candidate, but the estimate is not the limit: adaptive
      n (`--refine 32`) and one extra swap round (`--climb`) are null on the
      20-episode table (noise floor: 2 cleared / 1.9 deaths / 15 points of
      clear rate between identical runs). What is the limit at a wall is the
      candidate *set*: at Edwin #139 an exhaustive search over every 6-of-
      roster party cuts real deaths from 2.0 to 0.76 per fight (probe,
      2026-09-16, README → Status), with parties three changes from the base.
      `plan_gauntlet.py --parties beam|all` widens the set only when the best
      swap is still flagged hard (beam ~10 s, all ~2 min at 12 mons). Both
      are **null on the 20-episode table** (beam 178.15 / 13.75 / 35%, all
      178.95 / 13.05 / 30%, baseline 178.6 / 13.4 / 40%): a better party at
      Edwin saves mons that die one wall later, because the box holds no
      answer to the next wall. Deaths are conserved across walls until the
      box does — which is the env's box cap and encounter data (2¾), not
      the candidate search. Default stays `swap`; delete `beam|all` in the
      deletion pass unless the lifted box changes the picture. At Fisher Juan
      #169 (Gyarados) the roster is thin and every search agrees: that wall
      is the battler's (see 1a). Under the cap rule every roster mon sits at
      the same level, so "top-6-by-level" is catch order; the candidate
      search does the choosing. Lead order (packed order = party order) is a
      free variable the planner never sets; probe pending.
- [ ] **2p. Roster cap and obs layout.** MAX_ROSTER=12 exists only because
      the meta-policy obs is fixed-size; the graveyard hides it now but a
      real box is unbounded. When 2e rebuilds the obs, make the roster a
      set (per-mon features + attention or a pooled summary) and drop the
      12-slot action indexing.
- [x] **2q. Owed measurements** (2026-09-16). 51-slice default with evolution
      and encounters v2, 20 eps: 51.0 / 36.75 / 3.75 deaths / 100% (was 2.40;
      Roark 33 deaths in 20 runs, two fewer pre-Roark catches). 186 table at
      20 eps: `logs/plan_cap186v3_deep20.log` (README → Status).

Not in step 2: move events, builds, new gauntlets. Metric stays
"trainers cleared without a death"; report on the training slice only
until step 3 adds held-out gauntlets.

### 2½. Deletion pass (do before step 3; CLAUDE.md says prefer deleting code)
- [ ] Off-path code, about a third of the Python: `ai_util/battler_env.py`,
      `ai_util/self_play.py`, `ai_util/train_battler.py`, `ai_util/battler_eval.py`,
      `scripts/train.py`, `scripts/battle_env.py` (keep `SPECIES` if the
      vocab still needs it), `scripts/ppo.py` if 2e does not reuse it, the
      server-backed `BattleRunner` and `collect_worker` (server) paths,
      `collect_game.py baseline`, `diff_sets.py`. Each was a stage that is
      done; the notebook keeps the results.
- [ ] Null-result paths, per 1a: `Searcher.horizon`/`leaf_record`,
      `prior_value`, `pretrain_bc --value-only/--value-coef`, the `value`
      column, `temp/diag_value.py`. Delete unless 1a(i) reuses them.
- [ ] Housekeeping: cache `encounter_moves`/`evolve` tables (re-read from
      disk per call), move the README Status notebook out of a code fence
      into a `NOTEBOOK.md` with headings per finding, delete `temp/` or
      promote what docs reference, drop the `checkpoints/`, `runs/`,
      `test.py`, `no_AI_code_here` leftovers if unused.

### 2¾. Environment upgrade (plan written 2026-09-16; do before step 3)

The env is the one piece every approach reuses, so it gets the next effort.
Target shape: **one game folder + one rules object + one state machine**,
readable by someone who knows Pokémon, and no bigger than today's
`gauntlet.py` + `nuzlocke_env.py` (~750 lines). Budget: net ≤ +200 lines,
paid for by the deletion pass (2½) first. Not in scope: XP curves and
money (the cap rule stays), shops, TMs/tutors, held items, in-battle AI.

- [ ] **E1. Game folder.** `data/bdsp/` holds everything the env knows
      about the game: `trainers.json` (route order, parties with levels,
      moves, abilities, items), `encounters.json` (areas in route order:
      gate, species/level/weight table), `learnsets.json` (all level-up
      moves per species, not the last four), `evolutions.json` (species →
      [to, method, arg] for level / item / friendship / trade),
      `species.json` (types, ability slots, base stats; from the Showdown
      dex by one build script). `gauntlet.load_game("bdsp")` returns them
      and nothing else in code names BDSP. Acceptance: the 20-episode table
      reproduces, and a second folder loads with zero code changes (3a's
      real-source import lands here).
- [ ] **E2. Rules object.** A ~25-line dataclass a nuzlocker recognises:
      `level_rule` (cap | match), `level_margin`, `box` (None = unbounded),
      `dupes_clause`, `first_encounter_only`, `moveset` (last4 | best4),
      `wipe_ends_run`. Printed in every table header so each notebook entry
      states its rules. Falls out of E1–E4; do not write it first.
- [ ] **E3. Known defects, each a few lines and a test** (a, b, d done
      2026-09-16, `tests/test_nuzlocke_env.py`): (a) [x] hand-made ability
      table deleted — "first ability slot" is the rule, and the validator
      already applies it (41/41 equal); (b) [x] every area whose gate has
      arrived queues an offer (`env.pending`), answered one at a time,
      instead of the later overwriting the earlier; offers with no box room
      land in `env.dropped` and `--verbose` prints them; (c) [x] the box is
      unbounded by default (`--roster 30` ≥ starter + 21 areas; table
      2026-09-16: 180.15 / 58.0 / 16.35 / 45%, inside noise on cleared and
      clear rate; deaths per run is not comparable across box sizes because
      half the runs now wipe all 22 mons instead of 12);
      (d) [x] dupes clause on by default (`--no-dupes` to compare): an
      area's table is rerolled among families never owned, alive or dead.
      (e) [x] **ability slots** (raised and done 2026-09-16): a wild catch
      rolls slot 0 or 1 (Shinx: Rivalry / Intimidate; Geodude: Rock Head /
      Sturdy; Machop: Guts / No Guard), starters get slot 0, and evolution
      keeps the *slot*, not the name (Starly Keen Eye → Staravia
      Intimidate). `gauntlet.roll_ability` / `evolved_ability` over
      `data/species.json` (`gauntlet.py --build-species`: name, types,
      ability slots, base stats for 275 species — E1's species file, built
      early). Changes the catch distribution; not in the base2 table, so
      it is measured together with E4.
      Dupes-clause table (12-box): 176.25 / 61.5 / 14.0 / 25%, edge of noise
      in the wrong direction; kept as the standard rule. New baseline with
      both defaults: `logs/plan_cap186v3_deep20_base2.log`.
- [ ] **E4. Movesets: `best4`.** The largest expected gain at parity. Pool =
      every level-up move at or below the mon's level; keep the strongest
      STAB move, the strongest move of another type, the next strongest
      move, and one status move from a short list (recovery, sleep,
      paralysis, stat drops). Deterministic, ~30 lines, needs move type /
      power / category from the dex (`moves.json`, same build script).
      Meditite@22 then attacks. A moveset change is a progression change:
      regenerate the harvest and retrain the prior (~15 min + collect),
      then the table.
- [ ] **E5. Readable state.** `render()` as a tracker view (box with
      levels and moves, party, graveyard) and `plan_gauntlet.py --trace`
      printing encounter offers / catches / drops and the party per
      trainer — what today's scratch `replay_log.py` did, so it is deleted.
- [ ] **E6. Variable-roster obs/action (2p).** Only when 2e is attempted;
      the scripted planner never reads the obs. Do not build now.

Order: E3 (an afternoon, one table) → E1 (a day, behaviour-preserving) →
E4 (one OOD loop + table) → E5 alongside → E2 last. The `--roster 30`
table (2026-09-16, pending) is E3c's measurement.

### 3. Distribution and generality
- [ ] **3a. Real BDSP data first.** What exists is thin and hand-made:
      `bdsp_gauntlet_order.json` is sorted by trainer level, not by route,
      so "before trainer N" gates are level-band approximations; the
      encounter tables are recollection, version exclusives omitted;
      friendship/item/trade evolutions are not modelled (Budew, Buneary,
      Chingling, Roselia never evolve); formes only partly resolved. Find a
      real source (route order, encounter tables with rates and levels,
      evolution methods) and load it — a scripted import, not hand edits.
      Known forme gap past the 186 slice: the trainer file's 'heatrotom' /
      'mowrotom' / 'washrotom' are Showdown's 'rotomheat' etc.;
      `species_name` does not resolve them (`--build-species` reports 3
      unknown). Fix when the slice reaches them.
      Added 2026-09-16 from a roster dump at the Gyarados wall (README →
      Status): **(i) abilities:** the hand-made `abilities` table in
      `encounters.json` equals the dex's first ability slot for all 41
      species, and the validator fills evolved forms with their first slot
      too, so the rule is "first ability slot" everywhere — delete the
      table (E3a). **(ii) Movesets are the last four level-up moves
      at the cap level, nothing else** — Meditite@22 has no attack (Detect,
      Hidden Power, Mind Reader, Feint), Cherubi/Budew one weak Grass move;
      this is stage 0 of section 7 and is now a visible part of why fights
      at parity cost mons. Decide stage 1 (a few builds per species) before
      spending more on the battler. **(iii) The 12-slot box drops
      encounters silently** once full (seed 300 was 12/12 at Edwin); and
      when two routes unlock at one trainer the later offer overwrites the
      earlier (never fires on the current order file). Both go away with 2p.
- [ ] Second source game or hack; segment slicing; augmentation from seeds.
- [ ] Held-out eval; memorisation probe. Report train vs held-out always.

### 4. Blind regime
- [ ] Belief over enemy sets (prior from species × level, collapses on
      reveal); cross-attempt memory, reset per gauntlet.
- [ ] Value-of-information behaviour: does it scout?

### 5. Later
- [ ] Move events / builds (stage 1–2 above)
- [ ] Showdown mods approximating specific hacks
- [ ] Opponent AI tiers by trainer class (random < gym < E4/champion)

---

## Hard-won lessons (attempts 1–3, do not relearn)

- **Scope bloat kills.** Attempts 1–2 died of validation suites, curricula,
  Ray + MLflow. This repo: ~4.5k lines total.
- **Entropy management is the #1 from-scratch issue.** ent_coef 0.05 pinned
  entropy at 1.2; 0.01 let it sharpen. Neither reached the teacher.
- **Reward horizon.** gamma 0.95 × 40-turn episodes made the +30 win worth
  ~3.9 at turn 1 while HP shaping paid ±1/turn → agent farmed shaping.
- **"Obs deficit" was undertraining.** Three named-feature hypotheses
  falsified; 8→48 BC epochs took val CE 1.21→0.59. Converge before
  blaming the observation.
- **Imitation ceiling is real.** More data (91k→274k) and more capacity
  (256→512) did not move val CE. Beyond-teacher needs search/self-play.
- **RL finetune from BC oscillates around init.** Fresh value head knocks
  the policy off its init before the critic catches up.
- **Check metric polarity.** `_win_rate` returned losses; "MaxBP beats
  heuristics 88%" was inverted. SimpleHeuristics beats MaxBP 88–12.
- **Env facts.** 68/32 first-side advantage on gauntlet matchups; some
  low-level pairs cannot damage each other (need a turn cap); Team Preview
  hangs persistent-player challenges; therian formes mangle through
  poke-env team parsing.
- **Throughput.** Python-CPU-bound opponents need one process per env.
  Sync-per-step threads cap ~40 st/s.
- PPO values that worked: grad_clip 5.0, vf_clip 5.0, watch
  `TransformerEncoderLayer` float masks for NaN.
