# Round 3, Fable: last pass on plan v2 (2026-11-24)

Read-only. Checked against the code; two small measurements run (pure Python, < 10 s each, no
battles). No literature was read this round; the few method names below are recalled.

Three findings that change the plan, before the eight points:

- **Battles already run on Showdown.** Only the search behind each decision is on the Rust engine
  (`rl/engine_search.py:3`). So "re-run a bad battle on Showdown" cannot mean swapping the simulator.
  The useful check is whether the engine's *model* of each turn matched what Showdown did (point 3).
- **The search sees crits and KO rolls only at the root and depth 1** (`poke-engine/src/mcts.rs:175`).
  Deeper nodes use mean damage and no crit. Deaths in a nuzlocke come from exactly those events, and
  plan v2 does not mention it.
- **The MCTS is unseeded** (`mcts.rs:312`, `rng()`). Phase 0's gate "logging on/off gives identical
  paired results" cannot pass as written, and a battle can only be replayed from logged actions.

## 1. Movepool for sets

**Legal pool, as the code has it.** `encounter_moves` (`scripts/gauntlet.py:210–216`) already builds
the pool the user described, then throws it away with best4: level-up moves ≤ level, plus TM and
tutor moves in hand by trainer index (`data/rr/tms.json`, 165 entries, 17 marked unobtainable), minus
the 82 banned moves and the Toxic / Leech Seed whitelist. Egg moves are already excluded
(`scripts/build_rr.py:43–45` keeps only `9L`, `9M`, `9T` sources).

**Measured size** (every pool species, evolved at the cap, `--legal`):

| Boss (cap) | TM/tutor in hand | Level-up, median | TM/tutor, median | Legal pool: p25 / median / p75 / max | Attacks, median | Status, median |
|---|---|---|---|---|---|---|
| Brock (16) | 6 | 6 | 1 | 6 / 8 / 9 / 14 | 5 | 2 |
| Misty (28) | 26 | 9 | 4 | 12 / 14 / 16 / 23 | 11 | 3 |
| Erika (44) | 72 | 12 | 14 | 23 / 27 / 30 / 40 | 20 | 6 |
| Koga (68) | 111 | 15 | 20 | 30 / 35 / 39 / 55 | 27 | 7 |
| Blaine (76) | 141 | 15 | 23 | 33 / 37 / 43 / 60 | 29 | 7 |
| Lance (85) | 148 | 15 | 23 | 33 / 39 / 44 / 65 | 30 | 7 |

At Koga a median mon has C(35,4) = 52,360 four-move sets, so six mons cannot be enumerated.

**Prefilter, measured.** Keep an attack only if it is among the mon's two strongest against at least
one of the six boss mons. With a type-chart proxy (power × accuracy × STAB × category fit ×
effectiveness; the real filter should use `calc_bot.dmg_frac`, which also sees abilities, items and
the arena): median **3** attacks survive as "best against someone", **6** as "top two against
someone" (Koga and Blaine, max 10). With all legal status moves added that is a median of 715 sets a
mon; with status moves cut to ≤ 3 by class it is about C(9,4) = 126.

**Recommendation (decision for the user, D5 below).**
1. Split `legal_pool(species, level, progress)` out of `encounter_moves` (~10 lines moved, none added).
2. Shortlist per mon per boss: top-2 attacks per boss mon by `dmg_frac`, plus rule-kept utility
   classes whatever their damage: priority, pivot (U-turn / Volt Switch / Flip Turn), recovery,
   status spread, speed control, screens, Knock Off. About 8–10 moves.
3. Default set = greedy by marginal row gain (plan v2 phase 3 already says this). The damage part of
   move choice is solved by the calc: which move is best against each boss mon is deterministic.
4. Battles decide only the utility slots: in the wide per-boss budget, one-move mutations (swap a
   slot for a shortlist move), ordered by row gain, about 2 per mon = 12 neighbours a six. Not in
   the fast per-box budget.
5. The AI reacts to our moves, so the matrix cannot price a utility move; a line (point 4) compared
   for two sets can.

**Cost.** Battle search cost does not change (a mon still has four moves). The prefilter is about 160
`dmg_frac` calls a mon, built from `Move` objects on one dummy battle; unmeasured, expected well under
a second a box. Cells must be memoised by (set, boss), as the feature table already assumes.

**Feature spec.** Leaf features unchanged. Two additions for the chooser: a *pool cell* (the cell
computed with the mon's best legal move against that boss mon, an upper bound for any four-move set)
so the six can be drafted before sets are fixed, and *set slack* (pool-cell row minus actual-set row,
what the four-slot limit costs) as one more team aggregate.

**Data doubts.** (a) `build_rr.py:40` reads only the species' own learnset, so a move only the
pre-evolution learns is missing; whether RR's relearner offers those is a question for the user.
(b) `rr_probe.choose_nature` and `choose_ability` (`rr_probe.py:269`, `:310`) pick nature and ability
per boss, but in a real run both belong to the caught mon. This inflates every table and breaks the
advisor (point 5).

## 2. Visualising how the planner decides

What exists: `<out>.events.jsonl` (`boss_start` with enemy sets, `stage_a`, `box` with team, sets,
threats, counters, matrix), the dashboard tabs Runs / Compare / Deaths / Traces / Matrices
(`ui/build.py`, `ui/index.html`), `rr_trace.py --json` (per turn: actives, bench, every candidate's
value, the pick, events, Showdown replay). The trace uses the Showdown searcher only, not the engine.

| View | New logged fields | Where they come from | Cost |
|---|---|---|---|
| **Draft: answer table.** Per boss mon: our best and second-best answer with the cell numbers, unanswered threats | none; pivot of the existing `matrix` event | `matchup.rows_for` | cheap, UI only |
| **Draft: pick order.** Per pick: which boss mons it newly covers, its gain, the runner-up and its gain | `draft` event: `{pick, covers, gain, runner_up, runner_up_gain}` | the `gain` tuple `matchup.cover` already computes (`matchup.py:109–116`) | cheap, ~10 lines |
| **Draft: candidates that lost.** Each six: origin (cover / archetype / "swap A→B of #3"), score per halving round, round eliminated, margin | phase 3's archive rows plus `origin` and `round` | sequential halving | free once phase 3 exists |
| **Sets.** Per move: the boss mons it is the best move against; per item: its row gain; top-3 rejected moves | `sets` event extended | marginal row gain | free with point 1 |
| **Battle, per turn.** Visits and mean value of each of our options, the enemy model's predicted set, the enemy's actual move, match flag, value before / after | phase 0's `<out>.states.jsonl` plus both sides' chosen action strings and the battle seed | `engine_search.py:86` already builds `scored`, keeps only the argmax | cheap, in phase 0 |
| **Battle: where it went wrong.** Value curve with the largest one-turn drop marked | none | derived in `ui/build.py` | cheap |
| **Deaths expected, per mon.** | not available from the MCTS: the root value is one number mixing win and deaths | either per-mon death backups in Rust (~20 lines, every search pays) or the line tree, which gives them exactly | **expensive in MCTS, free from the line planner** |
| **"Why not X?" for a six nobody evaluated** | none | on-demand run, 48 battles | ~1 min a question |
| **What each component uses** | none | a static table in the dashboard from point 7 | no code |

Notes: feed the existing Traces tab from `states.jsonl` so any logged battle can be opened; then
`rr_trace.py` (118 lines) can be deleted. In wide budgets log one battle in k in full (record size
is unmeasured). `rl/cockpit.py` shows training curves: reuse it for the phase 4 fit, not here.

## 3. Showdown cross-check of bad battles

Since the battle outcome is already Showdown's, a bad battle has four possible causes. The check
names which one.

**Selection.** Needs a prediction to define "unexpected": (a) a loss or ≥ 2 deaths with a six whose
replay win rate is ≥ 70%; (b) any death on a turn whose root value was ≥ 0.8 the turn before;
(c) any one-turn value drop ≥ 0.4. Plus **a random 5% of ordinary battles as the control**: without
it we cannot say bad battles have more model errors than good ones.

**What is compared, per turn of a selected battle** (replayed from the logged actions and seed):

| Check | Method | Verdict it supports |
|---|---|---|
| Transition | `engine_diff.compare` at the logged position with the two actions actually chosen: the engine's outcome table against Showdown forked ~200 times (flags `missing`, `extra`, `tv`, `hp` exist) | engine mechanics bug |
| Enemy model | Rust `rr_ai_choice` set against the Python AI's actual move | AI port bug |
| Conversion | `EngineSearcher.why_not`, fallback count for that battle | converter gap |
| Luck | probability the engine gave the outcome that happened; < 15% and Showdown agrees | bad luck, no bug |
| Search | none of the above: re-search that position at 200k iterations; a different choice | search or evaluation error |

Output: one row a battle with the first flagged turn and its class, and a per-class count next to
the control's. New code: a `--from-states` mode in `rl/engine_diff.py`, ~30 lines. **Cost:**
`engine_diff` does 1,000 positions in ~3 min, so a 25-turn battle is ~5 s; 50 bad battles plus
controls under 10 min on one worker.

**Also, per team, on demand:** the same six on the same seeds under the Showdown searcher
(`ai_util.evaluate.Searcher`) against the engine searcher, n = 12. A large gap means the engine
misleads the search. Slower by an unmeasured factor (72 forks a decision); only for the worst sixes.

**Not covered:** Showdown and the Python AI against the real ROM (`data/rr/FIDELITY.md`). For an
advisor that is the fidelity that counts; the only check is the user noting where the real fight
left the line.

## 4. Line planner (`rl/line.py`, right after phase 1)

**Concept.** With the enemy a known script, the fight is a single-agent problem with chance events:
a Markov decision process. A line is a *policy tree*: at our nodes one chosen action, at chance
nodes one child per outcome. Computing win and death probabilities by expanding the tree instead of
sampling battles is *policy evaluation by tree expansion*; trying alternatives at a node and keeping
the best is *expectimax* (maximise over our actions, average over chance).

**Build, v0 (Python only, ~120 lines; every binding exists).**
- Our action at a node: the phase 1 search (`mcts_vs_ai`), argmax.
- Enemy options: `rr_ai_choice`. **Dependency on phase 1:** it must return weights (the 50% / 25%
  branches), not just the tied set (`lib.rs:1143` returns names only).
- Chance outcomes: `generate_instructions(state, ours, theirs)` gives each outcome with its
  probability, branching on miss, secondary effects, crit, and KO roll at every depth
  (`lib.rs:1085` passes `true`); `apply_instructions` gives the child state.
- Recurse while the reach probability ≥ ε. Below ε the node becomes a leaf valued by the search's
  root value, and its probability is added to an **unexplored mass** that is always reported.
- Back up exactly: P(win), expected deaths, P(each mon dies), P(deathless).

**Pruning and merging.** (1) Reach threshold ε (0.5% as a start). At most 1/ε nodes a depth level.
(2) Identical states merge by `state.to_string()`, so the tree is a graph. (3) For display only:
sibling outcomes after which our next action is the same and the risk differs by < 2 points are
folded into one row ("any roll"). This is what makes it read like a human line: only branches where
the plan or the risk changes are shown.

**v1, improvement at critical nodes.** Rank nodes by reach × death risk. At the top ~20, also expand
the second and third action by visits and keep the best by backed-up tree value. Nodes where an
alternative beats the search's pick are search errors: keep them as a regression set.

**Relation to the MCTS.** It reuses the MCTS to propose actions and to value pruned leaves. It
replaces *sampled battles* for judging a final team: 48 battles give ±14 points and cannot tell a 2%
death risk from 6%; the tree lists rare branches explicitly and gives a bound, P(death) between the
explored value and that plus the unexplored mass.

**Output.** Indented text (Markdown, collapsible in the dashboard) plus JSON nodes for a "Line" tab:

```
T1 Gyarados Waterfall vs Crobat          win 94%  deathless 81%  (explored 97.2%)
├─ 79% hit, no crit → T2 Waterfall       deathless 88%
├─ 12% Crobat flinches → T2 Waterfall    deathless 93%
├─  5% Crobat crits Brave Bird → switch Rhydon   deathless 41%   P(Gyarados dies) 22%
└─  4% miss → T2 stay                    deathless 60%
```

**Cost (unmeasured; ~0.1 s a 20k decision inferred from 2.7 worker-s a battle).** At ε = 0.5% and
~25 decision depths at most ~5,000 searches: ~8 min on one core, ~1 min on 10 workers, in practice
less because probability concentrates. v1 adds 2–3× at 20 nodes. Fine per boss, too slow inside team
search.

**Known approximation.** Non-KO damage is the mean roll, so a roll-dependent 2HKO is mispriced
(speed ties are branched 50 / 50, `generate_instructions.rs:4930`). Fix only if the check below
shows miscalibration.

**Verification, which is also the cheap form of point 3.** A line-follower brain plays the line on
Showdown against the Python AI, choosing by matching the observed outcome key (`convert.py`
`outcome_showdown` / `outcome_engine`). On-tree turns need no search, so 200 battles cost seconds.
Report: off-tree rate (a transition the tree did not contain = engine or AI-model disagreement),
and observed against predicted win and death rates. **Gate:** off-tree < 5% of battles before ε mass,
predicted P(deathless) inside the 95% interval of 200 Showdown battles.

**Later use by the bot.** Top-3 sixes of a search get a line and are chosen by bounded P(deathless);
a cached line is an opening book (no search on-tree); tree values are better fit targets for phase 4
than a single search's root value; critical nodes say where more iterations pay.

## 5. End use: advisor first

Changes:
- **Input is the user's real box**, a file with species, level, nature, ability, known moves.
  `sample_box` stays for evaluation tables. The builder must take nature and ability as given
  (unless the user confirms they can be changed in RR hardcore).
- **Budget shifts to wide per-boss.** One real box, minutes to an hour of compute a boss. The fast
  per-box budget becomes the regression harness only, and phase 5 (learned team filter) loses its
  reason until routing needs it.
- **Output is a plan sheet per boss:** six, lead, sets, the line, risk with bounds, the two worst
  branches, what to catch next.
- **Objective is the user's risk preference** (D1), reported as P(deathless) with bounds, not win rate.
- **Fidelity outranks strength.** Advice must hold on the ROM: the `FIDELITY.md` checklist, Koga's
  Tailwind, item and stone availability, and the unmodelled custom arena rules move up.
- **Interpretable parts are worth more:** a linear leaf evaluation with named terms, text lines.

Out of scope for now: emulator control or screen reading, doubles bosses (Sabrina, Lorelei, ~20
others: no advice at all, say so on the sheet), live turn-by-turn state entry (v0: follow the printed
line, re-run `line.py` from a typed state when the fight leaves it), autonomous routing.

## 6. ML ambition check

The generalisation unit is the boss, and there are ~78 singles bosses. That, not caution, caps what a
high-capacity model can prove on held-out data. Per method:

| Method | Reason given | Holds? | Switch when |
|---|---|---|---|
| **Battle value net** (beyond linear) | fit linear first; MLP only if a GBDT beats linear by > 2% log-loss | Mostly. Data is not the limit: phases 1–3 give ~0.5–1 M states, enough for a 32-unit layer, and its Rust inference is ~40 lines and < 1 µs. The offline GBDT probe costs 10 lines and decides. A net over raw per-mon tokens is a different size of job (days of Rust) | deep-search residual shows structured error no V1/V2 feature captures, after the MLP, with ≥ 2 M states from ≥ 50 bosses |
| **Policy prior (pUCT)** | "evaluation-bound, not width-bound" | **Unproven.** The ladder cannot separate the two: a gain from 200k is search-bound either way. If phase 1 passes, the enemy side is no longer searched and our branching is ≤ 9, which weakens the case. The training data (root visits) is logged free from phase 0 | after phase 4, 20k → 200k still ≥ +10 points **and** > 30% of 20k root visits go to moves that 200k gives < 5%. Cheap pre-test, no Rust: prune the root to the top 4 by a GBDT prior, ~30 lines |
| **Set transformer team scorer** | tree on 35 aggregates first | Yes, and more so for an advisor (compute a boss is plentiful) | filter needed for routing, T1 converged ≥ 0.1 Spearman under the noise ceiling, ≥ 30k distinct (six, boss) rows over ≥ 40 bosses |
| **GRPO pointer builder** | reward is a 2.7 worker-s battle; local search covers it | Yes. It only pays as amortisation | routing needs > 10⁵ box evaluations, and 2-swap beats the 1-swap optimum by ≥ 5 points in > 20% of boxes |
| **Offline RL** | the simulator is exact and cheap | Yes, fully | only if the ROM itself becomes the environment |
| **PPO** | 14% from scratch; BC + search won | Yes. A fixed scripted enemy is the friendly case for PPO, but search with the exact model gets the best response with no training | a no-search policy is needed for millions of rollouts; then distil the search (BC), not PPO |

Where plan v2 is too cautious:
1. **"Refit once" (phase 4).** Fit → search with the new evaluation → relabel → refit is *approximate
   policy iteration with a fitted value function*, the core RL loop of this project, and a round costs
   about an hour. Allow up to 3 rounds while the paired play test still improves.
2. **No per-mon death model.** A logistic model "this mon dies later in this battle" on leaf features
   is ~10 offline lines on the same states (KataGo-style auxiliary target, recalled). It need not
   enter the search to be useful: it is the per-turn risk display of point 2. Move it from phase 5 to 4.
3. **No trigger numbers for the prior.** Given above.
4. **The learning track has no next lesson.** CLAUDE.md allows it off the planner's path. Phase 4 is
   a natural lesson 3 (value function by regression); an AlphaZero-lite on 3v3 can follow as lesson 4
   without a production gate.

Nothing learned should move ahead of phases 0–2: the measured bottlenecks there are bugs, the
opponent model and fidelity, and no model fixes those.

## 7. Architecture (component list for the diagram)

| # | Component | Kind | Inputs → outputs | File | Status |
|---|---|---|---|---|---|
| 1 | Game data | data | sheets, dump, mod → learnsets, TMs, items, order, bosses | `data/rr/*`, `scripts/build_rr.py` | exists |
| 2 | Box source | rules / input | progress → box (sampled); user file → real box | `rr_probe.sample_box`; `box.json` | exists; file in phase 3 |
| 3 | Set builder | rules | mon, boss, items in hand → moves, item, nature, ability | `gauntlet.encounter_moves`, `assign_items`, `rr_probe.choose_*`, `rr_team_probe.build_mon` | exists; hygiene 0, pool + row gain 3 |
| 4 | Matchup matrix | calc | our sets, boss sets, arena → cells | `ai_util/matchup.py`, `calc_bot.dmg_frac` | exists |
| 5 | Candidate generator | rules (greedy cover) | cells → candidate sixes + leads | `matchup.cover`, `rr_team_probe.searched_cands` | exists; archetypes + mutations 3 |
| 6 | Team chooser | search (bandit) | candidates → best six, by paired battles | `rr_team_probe.searched_many` → halving | uniform screen exists; halving 3 |
| 7 | Simulator | ground truth | actions → next state | `ai_util/sim.py`, `sim_bridge.js` | exists |
| 8 | Enemy | rules | state → move | `ai_util/rr_ai.py` | exists |
| 9 | Battler | search | state → action, root visits | `rl/engine_search.py`, `rl/convert.py`, `poke-engine/src/mcts.rs` | exists |
| 9a | Enemy model in the tree | rules | state → enemy options + weights | `poke-engine/src/genx/rr_ai.rs` | weights + mixture 1 |
| 9b | Leaf evaluation | hand → fitted | state → value | `genx/evaluate.rs` | hand; fitted 4 |
| 9c | Fallback battler | search | state → action | `ai_util/evaluate.py` | exists |
| 10 | Line planner | search (policy tree) | six, boss → line, risk bounds | `rl/line.py` | **1.5** |
| 11 | Logs | data | — | `.jsonl`, `.events.jsonl`; `.states.jsonl`; archive; line JSON | exists; 0; 3; 1.5 |
| 12 | Cross-check | diagnostic | logged turns → bug class | `rl/engine_diff.py`, `rl/rr_ai_diff.py` | exists; `--from-states` 1.5 |
| 13 | Leaf fit | fitted model | states + outcomes → weights file | `rl/fit_eval.py` | 4 |
| 14 | Team filter | fitted model | archive → shortlist | — | 5, conditional |
| 15 | Statistics | test | paired results → decision | `scripts/sprt.py` | 0 |
| 16 | Views | UI | logs → pages | `ui/build.py`, `ui/index.html`, `rl/cockpit.py` | exists; Draft + Line 1.5 |
| 17 | Routing | arithmetic | scorer, encounters → what to catch | `scripts/rr_route.py` | 6 |

Flows: 1 → 2 → 3 ⇄ 4 → 5 → 6 ⇄ (9 ⇄ 7 + 8) → 11. 11 → 13 → 9b. 11 → 14 → 5. Best six from 6 → 10 →
plan sheet → user. Selected battles in 11 → 12 → bug list → 9a / engine. 15 gates every arrow back.

## 8. Decisions for the user

| # | When | Decision | Options | Evidence then |
|---|---|---|---|---|
| D1 | now | Risk preference (the objective) | U = win·(1 − λ·deaths/6) with λ; or deathless first, win second | none needed; Misty table under the floor as illustration after phase 0 |
| D2 | now | Held-out bosses | every 4th; or a block (e.g. all of Johto's leaders) | must be fixed before any data is collected |
| D3 | now | Advisor facts | natures / abilities fixed or changeable; pre-evolution moves; items, stones, Tailwind | the user's ROM knowledge |
| D4 | after 1a–c | Opponent in the tree | minimax; exact model; mixture p ∈ {0.5, 0.8, 0.95} | oracle result, on-policy agreement, paired SPRT |
| D5 | after 1 | Line planner settings | ε; objective; alternatives at critical nodes or not | first lines on Erika, their cost, off-tree rate |
| D6 | after 2 | What limits us, so what comes next | team search (3) or leaf fit (4) first; or fidelity work | mirrors, iteration ladder, reference teams |
| D7 | start of 3 | Set search scope | rule only; rule + utility-slot mutations; wider | pool sizes above, measured cell cost, halving budget |
| D8 | start of 3 | Score while wins < 20% | progress − w·deaths with w; or line-based P(deathless) | Koga / Blaine tables |
| D9 | start of 4 | Value target | battle outcome (Monte-Carlo target); deep-search value (bootstrapped target); line value | noise ceiling, state count |
| D10 | in 4 | Model class | linear; + one hidden layer | GBDT vs linear on the same features |
| D11 | end of 4 | Iterate the fit | stop; another round | paired play test per round |
| D12 | after 3 | Team filter | none; tree on aggregates | battles a boss needed, routing demand |
| D13 | after 4 | Policy prior / next lesson | not now; root-pruning pre-test; AlphaZero-lite as a lesson | ladder after the fit, visit statistics |

Each becomes a Step in the decision log (not written by me: read-only).

## Wrong, missing or mis-ordered in plan v2

1. **Crits and KO rolls beyond depth 1 are invisible to the search** (`mcts.rs:175`). Add to phase 1:
   branch at every depth against the current rule, paired, at equal iterations. One line of Rust.
2. **Phase 0 gate is unverifiable** while the MCTS is unseeded (`mcts.rs:312`). Pass a seed through
   the binding (~5 lines); it also tightens every paired comparison.
3. **Phase 0 logging lacks the two chosen action strings and the battle seed**; without them point 3
   cannot replay a battle.
4. **Natures and abilities are chosen per boss** (`rr_probe.py:269`, `:310`; `rr_team_probe.py:113`).
   Sampled boxes carry neither (`rr_probe.py:360`). Roll them at catch, or get the user's ruling (D3).
5. **Phase 2's reading "+15 from 20k to 200k → evaluation-bound"** should say search-bound; it does
   not separate evaluation from width (point 6).
6. **Phase 3 and 4 gates are in win rate.** For the stated goal add P(deathless) for the best six,
   from the line, with its bound.
7. **`rr_ai_choice` returns no weights** (`lib.rs:1143`); phase 1's "random branches as weighted
   options" must expose them, or the line planner cannot weight enemy branches.
8. **Effort** rises to about 11–13 days with points 1–4.

## Final phase list

| Phase | Content | Gate |
|---|---|---|
| **0** Hygiene and instruments (~1.5 d) | plan v2 phase 0, plus: seeded MCTS, actions + seed in `states.jsonl`, `draft` event, `legal_pool` split, D1–D3 | paired results identical with logging on / off (now testable); zero capped battles under minimax; Brock / Misty within ±3 |
| **1** Opponent in the tree (~1 d) | plan v2 phase 1, plus: weights from `rr_ai_choice`, damage branching at every depth as a paired test | paired SPRT ≥ +8 on Misty + Erika, deaths not worse, Brock ±3; on-policy agreement recorded |
| **1.5** Line planner, cross-check, views (~2.5 d) | `rl/line.py` v0, line-follower check on Showdown, `engine_diff --from-states` with controls, Draft and Line tabs, Traces fed from `states.jsonl` | off-tree < 5%; predicted P(deathless) inside the 200-battle interval on Brock, Misty, Erika; every bad battle gets a class |
| **2** Anchors and blame (~½ d) | plan v2 phase 2; lines for the reference teams | mirrors ≥ 75% and ≥ +20 over AI-vs-AI; reading decides D6 |
| **3** Team and set search (~2 d) | plan v2 phase 3, plus: real box file, shortlist + row-gain sets + utility-slot mutations, top-3 sixes judged by line | Erika 30-box +10 vs cover; Koga and Blaine ≥ 40% win and a stated P(deathless) bound, or a fidelity reading |
| **4** Fitted leaf evaluation (~2 d) | plan v2 phase 4, plus: per-mon death model for display, up to 3 fit rounds, critical-node regression set | held-out-boss log-loss −5%; paired SPRT +5 points or −0.05 deaths/mon; mirrors not worse; lines not worse |
| **5** Team filter | only if routing needs it | plan v2 gate |
| **6** Routing (~1 d) | "what to catch next" on the plan sheet | routed boxes +10 points |
