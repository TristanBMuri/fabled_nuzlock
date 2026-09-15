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
- [ ] Next levers, in order: (a) run-layer integration first (step 2) —
      the death-probability interface is what it needs and it exists now;
      (b) expert iteration: collect search choices on gauntlet matchups,
      retrain the prior, re-search; (c) deeper search reusing forks.

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
- [ ] **2k. Measure the combined default**: risk-based parties (2h) +
      search-all (2i') on the 51-slice, 40 episodes. Expected: >95% clears,
      <3 deaths. Then make it the default run config and the baseline table
      that step 2e's learned run layer must beat.
- [ ] **2l. Value estimate is the next battler ceiling.** Rollouts under the
      prior are 1-ply and noisy (r=8). Options in order: reuse forks across
      candidates (common random numbers), r=16 at bosses only, a learned
      value head trained on rollout outcomes (cheap: the sim produces them).
- [ ] **2m. Evolution.** Not modelled; matters from gym 2 (starters evolve
      at 16–18). Showdown species data has evos + evoLevel; apply on level-up
      in `_apply_level_rule`. Do before extending the slice past Roark.
- [ ] **2e. Learned run layer vs scripted.** Retrain `train_nuzlocke.py` on
      the sim backend with the 2b features. Success = beats the 2c table on
      trainers cleared without a death; otherwise the scripted planner *is*
      the run layer and effort goes to the battler. Note the slice may need
      to extend past gym 1 for the heuristic baseline to leave headroom.

Not in step 2: move events, builds, new gauntlets. Metric stays
"trainers cleared without a death"; report on the training slice only
until step 3 adds held-out gauntlets.

Battler follow-up, interleaved once 2c has a baseline:
- [ ] **Expert iteration v1.** Play 20k gauntlet matchups with
      `Searcher(k=3, r=8)`, record (obs, mask, chosen action), retrain the
      prior with `pretrain_bc.py pretrain` on search labels, then re-run
      the step-1 table. Target: prior alone ≥ 70.3% (the search number),
      search on the new prior above that. If the second round is flat,
      the value estimate (rollouts under the prior) is the limit and a
      learned value head is next.

### 3. Distribution and generality
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
