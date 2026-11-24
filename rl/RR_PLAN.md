# RL for Radical Red — plan (2026-11-23)

Drafted by the lead, reviewed independently by Fable and Opus against the code; this is the merged
version. Nothing here is built yet.

## The final plan v3 (2026-11-24): the build plan for the next chapter

v2 (below) plus a last pass by Fable and Gemini Neon on the user's additions
(`rl/research/FINAL_BRIEF.md`, `final_fable.md`, `final_neon.md`). **The phase list here replaces
v2's.** v2's findings, literature table, feature table and adequacy protocol still apply, with the
additions listed here. Every RL choice marked **D#** is the user's (CLAUDE.md, "Working with the
user on RL decisions"); the recommendation and the evidence available at that time are given.

### Decided by the user (2026-11-24)

- **End use: advisor first.** The system plans the six, lead, sets and a battle line for the
  user's own run on the real ROM; autonomy later. So the input is the user's real box, the
  budget per boss is wide (minutes to an hour), the output is a plan sheet, and fidelity to the
  ROM outranks strength. The fast per-box tables become the regression harness. For now the
  user has little time to play: the system runs on its own (sampled boxes, Showdown battles) and
  the user jumps in when they say so (2026-11-24).
- **Objective (D1): U = win · (1 − 0.5 · deaths/6)**, turn cap = loss.
- **Held-out bosses (D2): every 4th boss** in `order.json` order; never fitted on, always reported.
- **Movepool:** level-up moves ≤ level (relearner), reusable TMs in hand, tutors as reached; no
  egg moves; no pre-evolution-only moves. **Natures changeable** (mints) → chosen per boss.
  **Abilities** (revised the same day from the official sheets, `rr_probe.ability_gates`): rolled
  at catch until raid-den Ability Capsules are reachable (trainer #20, after Misty), the best slot
  0/1 per boss after, the hidden slot too from the Saffron Fanclub's Dream Patch (#44).
  **Stone evolutions** from the Celadon Dept. Store (#33, before Erika).
- **Line planner right after phase 1.**

### New facts from the last pass (checked by the lead in the code)

- **Battles already run on Showdown;** only the search runs on the Rust engine
  (`rl/engine_search.py:3`). A Rust bug hurts by misleading the search, so the Showdown check is
  "did the engine's model of each turn match what Showdown did", not a simulator swap.
- **The search sees crits and KO rolls only at the root and depth 1** (`mcts.rs:175`); deeper it
  uses mean damage, no crit. Nuzlocke deaths come from exactly those events.
- **The MCTS is unseeded** (`mcts.rs:312`), so a battle cannot be replayed and v2's phase 0 gate
  ("identical with logging on/off") was untestable.
- **`rr_ai_choice` returns names, no weights** (`lib.rs:1143`); the line planner needs the AI's
  25 / 50% branches as probabilities.
- **The U-turn / Volt Switch forced switch is not passed to the engine** (`rl/convert.py`; only
  `rl/rr_ai_diff.py` patches it).
- **Stone evolutions use the BDSP gate** (`gauntlet.py:294`, `STONE_GATE = "Eterna Gym"`).
- **Item data:** Air Balloon, Weakness Policy, Throat Spray, Light Clay, weather rocks are
  unobtainable in hardcore (`hardcore_rules.json:91`). Focus Sash, Assault Vest, Choice Specs and
  resist berries are absent because `build_rr.py:155` omits them, not because the game lacks them.
- **Legal pool size** (median / max moves a mon): Brock 8 / 14, Misty 14 / 23, Erika 27 / 40, Koga
  35 / 55, Blaine 37 / 60 (~52k four-move sets at Koga). After the hardcore bans only ~12–13
  distinct attack type × category and ~3 useful status moves remain; top-2 attacks against some
  boss mon leaves a median of 6.

### Status

- **Phase 0 (2026-11-24): done**, item audit included (`rl/research/rr_items_teams.md`; reference
  teams for Koga / Blaine are in the same file, for phase 2). Pool ceiling after it: Erika 100%,
  Koga 8%, Blaine 8% (n = 12): the wall stands.
- **Phase 1 (2026-11-24): gate passed; D4 = mixture p = 0.8, now the default** (`--engine-opp ai
  --engine-p 0.8`): +11.1 ± 4.8 points, −0.135 deaths/mon vs minimax on fresh seeds; the
  exact-model oracle +12.2. Then the Rust AI was fixed (user's choice): on-policy agreement
  75 → 98%; with it p = 0.8 gives **+30.2 ± 7.0 points, −0.28 deaths/mon** vs minimax on fresh
  seeds (Brock 83%, Misty 60%, Erika 69%), 0 capped; higher p is worse (Misty stalls). Branching
  crits at every depth: no gain, off. Pool ceiling: Koga 17%, Blaine 25% (n = 12).
- **Phase 1.5 (2026-11-24): done, D5 = accept and flag.** `rl/line.py` (policy tree + playouts),
  `engine_diff --from-states` (found 4 engine / converter bugs), `ui/` Draft / Trace / Line views.
  Line vs 200 Showdown battles at Erika: win 96 vs 97 ± 2%, deaths 1.0 vs 0.9; deathless 39 vs
  48 ± 7% (~9 points pessimistic: rolls are branched only on direct KOs), flagged in the output.
  Not done: v1 critical nodes, roll-sensitivity flag, the off-tree rate (replaced by the
  fixed-seed policy check).
- **Phase 2 (2026-11-24): done; D6 = phase 3 next.** Mirrors 100 / 77 / 98% (Erika / Koga /
  Blaine) vs AI-vs-AI 50 / 60 / 42%: the battler is sound. Reference teams
  (`data/rr/reference_teams.json`): Koga 0–38%, Blaine 46% and **100% deathless for a 2-mon
  Cloud Nine team** (a builder wall). Ladder 2k / 20k / 200k: K2 23 / 35 / 54%, B1 21 / 46 / 56%
  (search-bound). Deathless 0–10% even then. User: the advisor's final six is planned and played
  at 200k iterations; team search stays at 20k.
- **Phase 3 (2026-11-24): gates passed** (`scripts/rr_search.py`): Erika 30 boxes 78 → 89% (+11
  ± 8, SPRT H1); pool Koga **100% win, 88–90% deathless**, Blaine **92–96% win, 10–12% deathless**
  (independently replayed). Cloud Nine found at Blaine but cut by U. Next: D9 / D10 (phase 4) or
  the deathless question at Blaine.
- Phase 0 gate:
  logging on/off 60/60 boxes identical, zero capped battles, Brock 70 → 76%; Misty 51 → 24%, of
  which ~24 points are the abilities rolled at catch (Misty is before the capsules; the old 51%
  used a per-boss ability the run cannot have yet). NOTEBOOK.md 2026-11-24, `results/rr_p0_*`.

### Phases (gated; headline numbers are fresh-seed paired replays, n ≥ 48; held-out bosses reported apart)

**0. Hygiene and instruments (~1.5 d).** v2 phase 0 (engine fixes, won-terminal floor, timeout
logging, lead = best answer to the boss lead, chooser wired into `box_job`, set-builder rules,
Truant / duplicate filters, `scripts/sprt.py`) plus: a **seed through the MCTS binding** (~5 lines);
`<out>.states.jsonl` per turn with both chosen actions, the battle seed, root visits and mean
values (`engine_search.py:86` already has them), the AI model's prediction and the actual enemy
move; `force_switch` into `convert.to_engine`; RR stone gate; item audit of `build_rr.py`;
`legal_pool(species, level, progress)` split out of `encounter_moves` (no pre-evo moves);
abilities rolled at catch; one-mega check by stone, not by "any item" (`rr_team_probe.py:304`); a
`draft` event (pick, threats it newly covers, gain, runner-up). Gate: paired results identical with
logging on / off; zero capped battles under minimax; Brock / Misty within ±3 or the difference
explained by a named fix.

**1. The opponent in the tree (~1 d).** v2 phase 1 (re-run, exact-model oracle, on-policy
agreement, mixture p ∈ {0.5, 0.8, 0.95}) plus: `rr_ai_choice` returns weights; damage / crit
branching at every depth as a paired test at equal iterations. Gate: paired SPRT ≥ +8 points vs
minimax on Misty + Erika, deaths not worse, capped 0, Brock ±3. **D4** (opponent mode) on the
oracle, agreement and SPRT.

**1.5. Line planner, Showdown cross-check, views (~2.5 d) — the advisor's first usable form.**
- *Line planner* (`rl/line.py`, ~150–200 lines Python on existing bindings). With the enemy a
  known script the fight is a single-agent stochastic problem (an MDP). The line is a policy tree:
  our action at each decision from the phase-1 search, each chance outcome from
  `generate_instructions` (crit, miss, KO roll, secondary, speed tie) × the AI's weighted choices.
  Expand while reach probability ≥ ε (start 0.5%); below ε a leaf takes the search's value and its
  mass is reported as **unexplored**. States merge by `to_string()` (a graph). Back up exactly
  P(win), P(deathless), expected deaths, P(each mon dies); bounds = explored value + unexplored
  mass. Display folds sibling outcomes with the same next action and risk within 2 points. v1:
  expectimax over the search's top 2–3 actions at the ~20 riskiest nodes (reach × death risk);
  nodes where the search's pick loses become a regression set. Roll-sensitivity flag where the
  16 damage rolls (`calculate_damage`) straddle a KO threshold, since non-lethal rolls are averaged.
- *Line check on Showdown:* a line-follower plays 200 battles on Showdown vs the Python AI,
  following the tree by observed outcome (search off-tree). Off-tree transitions = engine or AI
  model disagreements.
- *Cross-check of bad battles* (`engine_diff --from-states`, ~30 lines): select losses or ≥ 2
  deaths with a ≥ 70% six, deaths after a root value ≥ 0.8, one-turn value drops ≥ 0.4, timeouts,
  **plus a random 5% control**. Per turn: transition (engine outcome table vs ~200 Showdown forks),
  enemy model vs actual move, converter fallbacks, luck (outcome probability < 15%), else re-search
  at 200k. One verdict per battle: engine bug / AI port bug / converter / luck / search error. On
  demand for the worst sixes: the same six under the pure-Showdown searcher, n = 12.
- *Views* (extend `ui/`): Draft (answer table per boss mon, pick order with runner-up, later the
  halving funnel), Trace fed from `states.jsonl` (visits and values per option, predicted vs actual
  enemy move, value curve with the largest drop), Line tab. Then `rr_trace.py` can go.
Gate: off-tree < 5% of battles; predicted P(deathless) inside the 95% interval of the 200
Showdown battles on Brock, Misty, Erika; every selected bad battle gets a verdict. **D5** (ε, risk
posture, v1 on / off) after the first Erika lines.

**2. Anchors and blame (~½ d).** v2 phase 2 plus lines for the reference teams. The 20k → 200k
ladder reads *search-bound*, not evaluation-bound (it cannot tell width from evaluation).
**D6**: team search (3) or leaf fit (4) first, or fidelity work.

**3. Team and set search, no learning (~2 d).** v2 phase 3 plus: `--my-box <file>` (species,
level, nature, ability, moves); per mon per boss a shortlist of ~8–10 moves (top-2 attacks per boss
mon by `dmg_frac`, plus priority, pivot, recovery, status, speed control, Knock Off when legal);
default sets by marginal row gain; battles decide only the utility slots through ~12 one-move
mutations per six, in the wide budget only (**D7**); the top-3 sixes judged by their lines'
P(deathless); a plan sheet per boss (six, lead, sets, line, risk with bounds, the two worst
branches). Gate: Erika 30-box +10 points vs cover; Koga and Blaine ≥ 40% win with a stated
P(deathless) bound, or a fidelity reading. **D8** (score while wins < 20%).

**4. Fitted leaf evaluation (~2 d).** v2 phase 4 (Texel: logistic regression of outcomes on
`evaluate_features`; V0 then V1) plus: up to **3 fit rounds** (fit → play → relabel → refit, which
is approximate policy iteration with a fitted value function) while the paired play test still
improves (**D11**); a per-mon death model for the risk display; the critical-node set from 1.5 as
a regression test; line values as a third target. **D9** (target: outcomes / deep-search values /
line values), **D10** (linear, or one hidden layer of 32 in Rust if a GBDT beats linear by > 1%
held-out log-loss; kept only if the play test passes). Gate: held-out-boss log-loss −5%; paired
SPRT +5 points or −0.05 deaths/mon; mirrors and lines not worse.

**5. Learned team filter (conditional).** Not needed to advise on one box (the wide budget
covers it); only if routing needs many box evaluations. v2 phase 5 order and gate. **D12**.

**6. Routing (~1 d):** "what to catch next" on the plan sheet. Gate: routed boxes +10 points.

**Learning track (off the planner's path, for the user to learn RL hands-on; no production
gate):** lesson 3 = phase 4's value fit as a notebook-style walkthrough; lesson 4 = AlphaZero-lite
(policy + value net with pUCT) on 3v3; a Set Transformer team scorer as an experiment once ≥ 10k
(six, boss) rows over ≥ 30 bosses exist (**D13**). Any of them moves onto the planner's path only
by beating the current component on held-out bosses.

### When the deferred ML comes back (both reviewers; trigger numbers)

| Method | Why not now | Switch when |
|---|---|---|
| Value net beyond linear / 1×32 | ~78 singles bosses limit what can be shown held out; the GBDT probe is cheap | deep-search residual shows structure no feature captures, ≥ 2 M states from ≥ 50 bosses |
| Policy prior (pUCT) | our branching is ≤ 9 once the enemy is the model | after phase 4, 20k → 200k still ≥ +10 points and > 30% of 20k root visits go to moves 200k gives < 5%; pre-test = root pruned by a GBDT prior, ~30 lines |
| Set Transformer scorer | ~786 rows today | ≥ 10k rows / ≥ 30 bosses as an experiment; production if it beats the tree on held-out bosses |
| GRPO / pointer builder | the reward is a 2.7 worker-s battle | routing needs > 10⁵ box evaluations; first expert iteration on ≥ 1–3k winning sixes |
| PPO / offline RL | search with the exact enemy model is the best response with no training | a search-free policy is needed for millions of rollouts: distil the search (BC), not PPO |

### Feature additions to v2's table

Chooser: *pool cell* (the cell with the mon's best legal move vs that boss mon, an upper bound to
draft before sets are fixed) and *set slack* (pool-cell row − actual row); `has_priority`,
`has_pivot`, `has_disruption`. Leaf: unchanged.

### Architecture

The diagram (inputs, outputs, what each decision uses, which phase adds each part):
https://claude.ai/artifact/Y4iDRRU9vDPKCvqRJEypy4. Components and files are listed in
`rl/research/final_fable.md` §7.

### Still needs the user

- Reference teams (players' sixes that beat Koga / Blaine in hardcore runs) and item / stone /
  mint availability: being researched online by Gemini Neon (2026-11-24); the user does not know.
- Koga's Tailwind **is permanent** on the ROM (user, 2026-11-24): the arena model stands.

### Effort

0: 1.5 d · 1: 1 d · 1.5: 2.5 d · 2: ½ d · 3: 2 d · 4: 2 d · 5: conditional · 6: 1 d.
About 11–13 working days if every gate passes; phases 0–1.5 (~5 d) give the first plan sheet.

## The plan v2 (2026-11-24): research round, Fable + Gemini Neon + lead, merged

**Its phase list is superseded by v3 above**; the findings, literature, features and adequacy
protocol still apply. v2 superseded the phase list and the battler plan below, which stay as history.
How it was made: Fable and Neon each researched (general RL / search / ML literature, emphasis
2022–2026) and wrote a plan without seeing the other's; the lead did a third research pass; then
each reviewed the other plan plus the lead's notes against the code. All seven documents are in
`rl/research/` (`BRIEF.md`, `fable.md`, `neon.md`, `lead.md`, `*_review.md`). Literature tags
there say whether a source was seen in a search summary or recalled; nobody read a paper in full.

### What the round established (checked in code / results by at least one reviewer)

- **The problem is single-agent.** The enemy is a fixed, known script, so the goal is a best
  response with a safety margin, not an equilibrium. Nash-seeking search, self-play and a learned
  enemy model are out; the team is just the first action of one decision problem.
- **The RR-AI search loss has two causes.** Misty (−33): ≥ 10 / 30 boxes have provable turn-cap
  losses (deaths/mon < 1 − win; 0 / 30 under minimax), so at least half the drop is the 100-turn
  force-loss (`ai_util/sim.py:225`) — NOTEBOOK.md's "no turn-limit losses" is wrong. Brock (−13):
  no capped battles, deaths rise, so it is ordinary losing, likely model error. The Rust port
  also omits the AI's 50% / random branches (`rr_ai.rs:1832` vs `ai_util/rr_ai.py:210`, `:277`,
  `:614`), so the tree treats coin flips as certain. Off-policy agreement is 90% (switches 78%);
  on-policy agreement is unmeasured.
- **`--chooser searched` is not wired into box mode** (`rr_team_probe.py:577` falls through to
  `greedy`): every 30-box table so far is cover or greedy. **Leads are alphabetical** for cover
  and greedy (`:277`, `:255`).
- **Set-builder faults, from the logs:** Rest always eligible (`gauntlet.py:188`, on a Choice Band
  holder at Blaine), +Spe on any mixed set (`rr_probe.py:269`, Jolly base-30 Coalossal at Koga),
  Truant Slaking and two Sawsbucks in pool "best sixes" (only `searched_cands` filters them).
  The Grass-into-Blaine six came from the calc-bot stages A/B, not from `cover`.
- **The leaf evaluation is material-only** (`evaluate.rs`): no speed order, damage margins,
  weather or bench matchups; a costly win can score below an open fight (`mcts.rs:226`).
- **Data on disk is thin:** no per-turn states are logged anywhere; the "12.5k duel cells" are
  6 reruns of the same 2,094 Brock + Misty cells; ~786 (six, boss) rows.
- **Scale:** boxes average 9 (Brock), 19 (Misty), 38 (Erika), 52 (Koga), 72 (Blaine), 81 (Lance);
  enumeration dies after ~30 mons, so late bosses need local search. ~78 named singles bosses.
- **Unmeasured cost assumptions** (MCTS-2k ≈ 0.3 worker-s a battle, `rows_for` ≈ 20 ms a box) —
  measure before planning on them.

### Literature that shaped it (details and links in `rl/research/`)

| Idea | Source(s) | Used where |
|---|---|---|
| Opponent model in the tree, mixed with UCB per node | Adversarial-policy MCTS (Wang et al. 2023), Restricted Nash / Data-Biased Response (Johanson 2007/09), CDBR (Milec 2024), BRExIt (2023), SES / OX-Search (2022/24) | Phase 1 |
| Opponent-model search steers into model errors | Carmel & Markovitch; Donkers (classic OM search) | Phase 1 on-policy agreement |
| Fit the leaf evaluation by logistic regression on outcomes | Texel tuning (Buro 1995, Österlund 2014); TDLeaf / TreeStrap for deep-search targets; NNUE as the upgrade path | Phase 4 |
| Best-arm identification under noisy, paired evaluation | Sequential halving (Karnin 2013), common random numbers, Fishtest SPRT, optimizer's curse | Phases 0, 3, all gates |
| Drafting / deck building = search over picks + win-rate predictor | JueWuDraft (2020), Art of Drafting, BPCoach, Q-DeckRec, LoCM deck+battle (2023), Reis et al. Pokémon team building (2023) | Phases 3, 5 |
| Quality-diversity archive of teams | MAP-Elites for Hearthstone decks (Fontaine 2019), MADRID (2024) | Phase 3 (10-line version) |
| Train / collect where regret is high | PLR (2021), ACCEL (2022) | Phases 4–5 sampling weights |
| Dense auxiliary labels | KataGo (per-mon death heads) | Phase 5 |
| Small-data tabular predictors | GBDT; TabPFN v2 / 2.5 (Nature 2025) as a probe only | Phase 5 adequacy check |

Considered and **not** adopted (each can return only through a failed gate that names it):
GRPO / RLOO pointer builder (reward is a 2.7 worker-s Bernoulli battle; local search through a
cheap filter covers the same ground), LLM-evolved heuristics à la AlphaEvolve (weights are a convex
fit; an evolution loop gets ~20 Rust-rebuild candidates an hour and overfits seeds — but any term an
LLM proposes may enter via the residual gate), Set Transformer / pointer net / ensemble / HL-Gauss /
Plackett-Luce (no noise ceiling yet; a tree on ~30 aggregates first), Gumbel pUCT and a battle net
(search is evaluation-bound, not width-bound), PQN / offline RL / world models (the simulator is
exact and cheap), Exp3 / regret matching, PPO.

### Phases (gated; every headline number is a fresh-seed replay, n ≥ 48, on seeds used for no decision)

**0. Hygiene and instruments (~1 day, ~130 lines, < 30 min machine).** The queued engine fixes
(mega holder, permanent Tailwind flat, spent Phoenix Down) and the won-terminal floor
`0.55 + 0.45·alive/6` as the interim objective fix; log `timeout` per battle; lead = best answer
to the boss lead in every chooser; wire the chooser into `box_job` (fail loudly on an unknown
name); Rest only with Sleep Talk, no Choice item with a status move, +Spe only where it buys
outspeeds; Truant and duplicate-evolved filters in every stage; per-turn state logging
(`<out>.states.jsonl`: state string, both sides' root visits, the Rust AI's predicted set, the
enemy's actual move, outcome, deaths per mon, battler settings); `scripts/sprt.py` (paired SPRT).
Gate: logging on/off gives identical paired results; zero capped battles under minimax;
Brock / Misty tables within ±3 of today (otherwise the hygiene changed something to explain).
Depth discount `0.995^d` only if capped battles survive the floor.

**1. The opponent in the tree (~1 day, ~45 lines Rust + Python, ~1 h machine).**
(a) Re-run `--engine-opp ai` after phase 0. (b) **Exact-model oracle:** the Rust AI
(`rr_ai_choice`, ties uniform) plays the real enemy, so the model in the tree is exact; ai vs
minimax against it separates "port wrong" from "search wrong". (c) On-policy agreement of the
Rust model with the Python AI, split move / voluntary / forced switch. (d) Per-node mixture: follow
the model with probability p ∈ {0.5, 0.8, 0.95}, else UCB. If the oracle says the method works:
the random branches as weighted enemy options, then the switch-logic diffs, to ≥ 97% on-policy.
Gate: paired SPRT ≥ +8 points vs minimax on Misty + Erika boxes, deaths/mon not worse, capped
battles 0, Brock ±3. Else minimax stays and on-policy agreement is recorded as the open number.

**2. Anchors and blame the walls (~½ day, ~20 lines, ~1 h machine).** Fair mirrors (Erika, Koga
without the arena, Blaine; n = 48) next to the AI-vs-AI baseline; a battler ladder MCTS 2k / 20k /
200k (+ the phase 1 mode) on the best Koga / Blaine sixes and the EV-85 sixes; reference human
teams (from the user); measured cost of a 2k battle; Spearman of six rankings across rungs.
Reading: mirrors ≥ 75% and ≥ +20 over AI-vs-AI → the battler is sound; ladder +15 from 20k to
200k → evaluation-bound, do phase 4 before 3; reference ≤ 10% and flat ladder → fidelity (ROM
Tailwind, item availability) before more compute. ρ ≥ 0.8 → 2k battles allowed in early halving
rounds. **A fidelity wall does not block the method phases**; it only stops Koga-specific work.

**3. Team and set search, no learning (~1.5–2 days, ~100 lines, ~2–3 h machine).** Sequential
halving on paired seeds over (six, lead, moves, items); score = `progress − w·deaths/mon` while
wins < 20%, wins after. Candidates: cover, `searched_cands`, the best six per archetype flag
(weather answer, speed control, pivot core, mega — the 10-line QD archive), 1-swap / lead / move
/ item mutations chosen by row gain. Two budgets: **fast per-box** (~120 battles a box, for 30-box
tables) and **wide per-boss** (3–10k battles, for ceilings and data). Writes the archive of every
evaluated (six, sets, boss, battler-version) row plus a random sample of unselected sixes
(halving survivors are selection-biased). Set-builder rule upgrade here: moves and items by
marginal row gain against this boss, with the compatibility penalties (rules, not a model).
Gate: Erika 30-box +10 points vs cover; Koga and Blaine ≥ 40% legal, or a fidelity reading from
phase 2.

**4. Fitted leaf evaluation, Texel (~2 days, ~100 lines Rust + ~80 Python `rl/fit_eval.py`; fit
in seconds, play tests ~1 h).** Rust `evaluate_features(state)`; `evaluate()` becomes a dot product
with weights from a file (env var, like `POKE_ENGINE_DEATH`); absolute sigmoid, not relative to the
root; terminals return the utility **U = win · (1 − λ·deaths/6)** (λ = 0.5, a stated preference;
turn cap = loss), replacing the death constant and the floor. V0: refit the existing terms only.
V1: add the root-precomputed 6×6×2 damage-table features (feature table below). Data: phase 1–3
per-turn states, weighted by p(1−p) of their (six, boss), battle-level splits, every 4th boss
held out; second target = MCTS-200k root value on a 5k-state subsample.
Gate: held-out-boss log-loss −5% vs the hand evaluation (scale refit); paired SPRT +5 points or
−0.05 deaths/mon on Brock / Misty / Erika and the Koga / Blaine bed; mirrors not worse. Fit
passes but play fails → keep only the terminal utility, stop. One hidden layer of 32 in Rust only
if a GBDT on the same features beats linear by > 2% log-loss. After adopting: re-screen phase 3's
top 20 per boss (versioned scores), refit once.

**5. Learned team filter (conditional: only if phase 3 is compute-bound, e.g. 30 boxes × 78
bosses or routing; ~2 days, ~180 lines).** Order, stop at the first that passes: (T1) a tree /
logistic model on the ~35 team aggregates → P(win), deaths/mon, per-mon death heads; (T2) the cell
calibrator only if `cover` itself needs probabilities (needs fresh duels beyond Brock / Misty);
(T3) a small set model over per-mon tokens only on a named failure of T1 (above the noise ceiling,
converged, tree not capacity-limited). Used inside 1-swap / lead-swap local search; top ~18 plus
~6 high-uncertainty or random picks go to halving; the model never picks the final six.
Gate (held-out bosses): true top-3 inside the top 24 in ≥ 80%; ≥ 90% of phase 3's win rate at
≤ 1/6 of its battles. Regret-weighted collection of new boxes (PLR / ACCEL, ~10 lines).

**6. Routing:** expected gain of an encounter over the bosses ahead, through whichever scorer
survived. Arithmetic, no RL. Gate: routed boxes +10 points clear rate vs random-method boxes.

The chicken-and-egg loop, as policy iteration: the battler is anchored without the builder
(mirrors, AI-vs-AI, reference teams); teams are scored by the battler that will play them; every
score carries the battler version; after a battler change only the top-K per boss are re-screened;
alternate battler change → anchors → re-screen → team search → refit, one heavy job at a time.

### Feature engineering (merged)

No species or boss identity anywhere (except inside the species-ID gap diagnostic).

| Use | Features | Source |
|---|---|---|
| Leaf V0 | HP sums, alive, our deaths, status counts, active boosts, hazards × grounded mons, screens / Tailwind (permanent flat) / Sub / Seed / item present | existing `evaluate.rs` terms, refit |
| Leaf V1 (from a 6×6×2 best-move damage table built once at the search root) | active vs active: our damage ÷ their HP and theirs on ours (capped 1); we move first (boosts, para, Tailwind, weather ability, TR, priority); we KO first; answers left; unanswered threats; sweep exposure (boost-aware); safest switch-in; temporary weather / terrain / TR turns; turn fraction | Rust, `damage_calc.rs` |
| Leaf V2 (only on a residual) | foe heals and is not 2HKO'd; sleep / toxic / Protect counters | state |
| Cell (ours i vs theirs j) | hit, take, ko_them, ko_us, faster, speed ratio, priority each way, heals, first-hit modifiers, our accuracy / recoil, safe / race / edge tiers; recomputed under the arena. The AI's opening move from the AI model only, never from the duel (label leak) | `matchup.pair`, `calc_bot` |
| Team aggregates (~35, the T1 input) | per boss mon best and **second-best** answer; weakest link (min best answer); # boss mons without a safe answer; lead cell + lead's worst incoming hit; best pivot per boss mon; # faster than ≥ 4 under the arena; # with priority; max # of ours one boss mon OHKOs; type-weakness overlap; mean stat margins (atk and def side); arena fit (row with − without arena); weather setter / negator, mega, recovery, status-spreader counts; `boss_ai_class` mix (`rr_ai._class`); cap − boss level | cells, `calc_bot.stat`, `rr_ai` |
| Set edits (rules in the builder) | Δ row from a move / item / ability / nature change; coverage count; accuracy, priority, recoil / self-KO, lock-in; Choice + status, Rest w/o Sleep Talk, booster w/o matching move as penalties; one mega per team | recomputed cells (memoised) |
| Per-mon tokens (T3 only) | level-scaled stats, 18 types, mega, a handful of item / ability classes the cell cannot see (weather set / negate, Intimidate, Sash-like, Choice lock), row summary, `is_lead` | — |

**Adequacy protocol, in order** (the lesson: "the obs is missing X" was undertraining every
time): (1) noise ceiling from 8× fresh-seed replays of 200 sixes / 500 states; (2) convergence at
¼ / ½ / 1× data; (3) tree vs linear on the same features (TabPFN as a no-training second
opinion), (3b) species-ID gap on held-out boxes of seen bosses; (4) a residual bucket that names
the feature (≥ 5 points, n ≥ 500); then group ablation (delete groups that do nothing). The paired
play test outranks every loss number. Only held-out *bosses* count as generalisation.

### Needs the user

- Reference human hardcore teams for Koga and Blaine (two each).
- Item availability in hardcore: are Focus Sash, Choice Specs, Assault Vest, Air Balloon and
  resist berries obtainable before Koga / Blaine (`data/rr/items.json` has none of them)? When are
  evolution stones usable (`build_mon` deliberately uses none)?
- ROM check: is Koga's Tailwind permanent in RR 4.1?

### Effort

0: ~1 d · 1: ~1 d · 2: ~½ d · 3: 1.5–2 d · 4: ~2 d · 5: ~2 d, conditional · 6: ~1 d.
About 8–10 working days if every gate passes; phases 0–2 (~2.5 days) decide most of the rest.

## The reframe

The first draft was: imitate the engine search into a value net, so the planner can score sixes
without searched battles. Both reviews took it apart on the code:

- **The RR planner plays no battles to pick a six.** `box_job` (`scripts/rr_team_probe.py:501`)
  takes one `cover_team` six (paper matrix, dummy battles) and plays only that six 6 times. A value
  net would replace the matrix, not searched battles, and the matrix costs almost nothing.
- **Searched battles are cheap now.** The 6-gym × 30-box table took 4.8 min on 10 workers
  (~2.7 worker-s per battle). Searching many sixes per box is affordable *without* a net, and
  nobody has measured it.
- **The Koga / Blaine wall is unexplained.** 0/180 battles each (`results/rr_engine_gyms30.*`).
  The older pool probe (before the sheet sets) found sixes that win there: Koga 50%, Blaine 100%
  (`results/rr_team_probe_arena_legal.log`). The box sets look poor (4 of 6 Blaine mons carry Rest,
  a Choice Band holder with Roost, Gyarados with Mystic Water + Thrash; no mega stones, no stone
  evolutions, no EVs). The wall may be the chooser or the set builder, not play.
- **pUCT gives no speed-up.** poke-engine's MCTS is decoupled UCB1 with a heuristic leaf
  (`poke-engine/src/mcts.rs:74`, `:247`): no priors, no value hook. A Python net per leaf is
  ≥100 µs vs ~5 µs an iteration; a net in Rust is 3–5 days for parity at best.
- **The proposed holdout (train Brock–Surge, test Erika–Blaine) is degenerate:** Koga / Blaine are
  all losses, so "always predict lose" calibrates perfectly, and levels 16–36 → 68–76 are a large
  shift.

So the question RL serves in RR is **"which six, with which lead, wins at this boss"**, and the
first answer needs no learning. Learning comes in only where search runs out of compute.

## Phase 0 result (2026-11-23): gate failed

Best pool six with today's builder: Erika 92%, **Koga 8%, Blaine 0%** (`results/rr_pool_ceiling.*`).
Rest out: no change. Stone evolutions: no change. 85 EVs in every stat (illegal in hardcore):
Koga 42%, Blaine 58%. The wall is our builds' stat / item margins, not team choice; Phases 1–5
wait until the item list (megas, Choice items, Life Orb, Sash, Vest) and the set builder are
fixed. NOTEBOOK.md 2026-11-23, "Phase 0".
**Megas + Choice Scarf added (same day): Koga 8%, Blaine 0%, unchanged.** The ceiling is only as
good as the calc-bot stages that propose the sixes (Cherubi / Deerling into Blaine's sun), so
Phase 1's searched chooser over a wider candidate set is the next diagnostic.

## Phases v1 (2026-11-23; superseded by the plan v2 above)

### 0. Diagnose the wall (½–1 day, little code)

- **0a. Pool ceiling on today's settings** (sheet sets, `--legal --arena`, `--engine 20000`) at
  Erika, Koga, Blaine. ~1 h of machine time. **Gate:** a pool six wins ≥ 50% at Koga. At ~0%,
  the problem is builds or fidelity: do 0b and stop here.
- **0b. Set-builder audit** (1 day): best-4 picking Rest / Roost on Choice items, item fit
  (Mystic Water + Thrash), mega stones never assigned, level evolutions only (no stones), no EVs.
- **0c. The lead.** Sixes are sorted tuples (`rr_team_probe.py:275`, `:253`), and there is no team
  preview, so the lead is the alphabetically first mon. That's a free lever; Phase 1 tries all 6.

### 1. Searched chooser, no net (1–2 days, ~40–80 lines: `--chooser searched`)

- **Candidates:** the cover six, the top-K greedy subsets and single-swap neighbours, × the 6 leads.
  Each gets a few cheap battles on the engine.
- **Validate the cheap scorer first** (MCTS-1k battles in the engine, against the Rust RR AI as
  the *actor*, not as the search's model) on the existing 180 (six, boss) pairs. **Gate:**
  Spearman ≥ 0.7 against the 6-battle Showdown win rate, and ≥ 70% top-quartile agreement.
- **Replay the winner on a fresh `--seed`.** Seeds are paired per boss (`:190`), so the best of K
  on the same seeds inflates its win rate.
- **Gate:** +10 points at Misty–Erika, or Koga / Blaine > 0, outside the paired ±10 interval, on
  the gyms30 command
  `rr_team_probe.py --boxes 30 --legal --arena --matchup --death-weight 6 --engine 20000 --workers 10`
  (baseline 66 / 47 / 44 / 22 / 0 / 0).
- **If it passes at < 10 min a box, skip Phases 2–3 and go to 4.** Its evaluated sixes are also
  the dataset if 3 is needed: thousands of (six, boss) outcomes, bad teams included.

### 2. Data + the free value baseline (1 day, ~30 lines)

- **Logging.** `EngineSearcher` (`rl/engine_search.py:80`) already has the root visits and means
  but keeps only the argmax. Log the state string (`from_string` exists), both sides' visit
  counts, the root value, and the deaths per mon in `one_battle`.
- **Labels** come from the Showdown outcome, so the engine's 3% disagreement is noise in the play,
  not in the label.
- **Bosses:** all ~75 singles bosses (`engine_diff.py --rr` walks them), not six gyms. Hold out
  every 4th boss and 20% of boxes.
- **Baseline:** the turn-0 MCTS root value (~0.1 s a six). **Gate:** within-box Spearman ≥ 0.5
  against the searched win rate.

### 3. Value net, only if Phase 1 is compute-bound (2–3 days, `rl/lesson3_value.py` ~250 lines)

- **Inputs:** no species identity (the generality rule); RR data from `data/rr/mod_data.json`.
  Stats, types, move features, ability / item embeddings, and the 6×6 `calc_bot.dmg_frac` matrix
  both ways. Lesson 2's code is Gen 8, 3v3 and species-keyed; only its `Net`, BC loss and cockpit
  logging carry over.
- **Heads:** P(win) and P(this mon dies), with the root value as an extra, lower-variance target.
- **Gates** (held-out bosses and boxes):
  - within-box ρ ≥ the root-value baseline + 0.1
  - Brier ≥ 15% better than the per-boss mean
  - death head within ±0.1 per decile
- **Use it as a filter, never the judge.** The argmax of a noisy net over many sixes selects its
  errors (the same failure as searching against the 90% AI). The net shortlists 20, search picks
  one. **Gate:** the shortlist holds a six within 5 points of Phase 1's best in ≥ 80% of boxes,
  for ≥ 70% of Phase 1's gain at ≤ 1/10 of its compute.

### 3b. A learned team builder (the user's question, 2026-11-23)

What a mon is worth depends on the other five and on the boss's six, so the builder is a set problem:
attention fits. The order of work:

- **Scorer:** f(our six, their six) → P(win), P(each of ours dies). Each mon is a token built from
  its stats, types, move features and its matchup row against the six boss mons (`dmg_frac` both
  ways, who is faster), with no species identity. Self-attention runs over our six, then
  cross-attention to theirs. This replaces Phase 3's plain MLP; same labels, same gates.
- **Builder:** pick one mon at a time, each pick attending to the picks so far and to the boss's
  team (a pointer network over the box), plus the lead. Train it on Phase 1's best sixes, then
  run expert iteration: the net proposes, search verifies, the verified sixes become the next
  targets. REINFORCE on raw battle outcomes only after that (attempt #3: RL from scratch is the
  slow road).
- **Order:** sets (moves, items, nature) stay rule-based; only *which six* is learned.
- **Limit:** it can never be better than the search that labels it. Phase 1 comes first because
  it is the teacher, and its evaluated sixes are the data.

### The team / battler loop (good teams need a good player and vice versa)

We break the circle with anchors that hold one side fixed, and by tuning the two in turns:

- **Battler anchors (no builder involved):**
  - *Mirror matches:* our battler plays the boss's own six against the RR AI playing the same
    six, so the team is equal by construction. Report it per boss.
  - *Reference teams:* known-good human teams for a boss (community runs, the user's own).
    Losing with those is the battler's fault.
  - *Conformance:* the engine search vs the Showdown search, as today.
- **Separability test:** rank the same sixes with weak and strong battlers (MCTS 1k / 20k, the
  Showdown search). If the ranking holds (Spearman ≥ 0.8), team quality can be scored apart
  from the battler. If not, teams must be scored with the battler that will play them.
- **Blame a loss:** replay it with a much stronger battler (more iterations, deep search).
  Win rate up a lot → the battler limited it. Unchanged → the team limited it (the EV
  experiment was this kind of test: same battler, stronger mons, 8% → 42%).
- **Alternate, don't co-learn:** fix the battler, search teams; fix the teams, improve the
  battler; measure the pair on held-out bosses after each round. Our objective is the pair
  against a *fixed* scripted enemy, so a team that only this battler can pilot well is a
  feature, not a bug.

### The battler plan after the mirrors (2026-11-23, Fable + Gemini Neon, merged)

Both reviews found bugs that invalidate part of the mirror and ceiling numbers:

- **Mega holders could only switch (Neon; verified).** Under `rr`, poke-engine offers a stone
  holder only `<move>-mega` choices (`poke-engine/src/genx/state.rs:556`). `EngineSearcher` maps
  choices by move id and drops the `-mega` ones (`rl/engine_search.py:84`), so only switches stay
  legal. It hits p1 only, and only since the player can hold stones (megas, same day): the mirrors
  (Toxtricity, Charizard-X), `rr_ceiling_mega` and every searched candidate with a stone. Neon's
  in-memory check, 12 battles each: Koga mirror without the arena 25% → 92%, Blaine 42% → 67–83%.
- **Tailwind 99 is scored per turn (both).** The dump sends a permanent Tailwind as 99 turns
  (`ai_util/sim_bridge.js:122`); the engine counts it down and the evaluation charges
  `turns × TAILWIND` (`evaluate.rs:270`, `:277`), so every ply looks +7 for us: a stall bias.
- **A costly win scores below an open fight (both).** Won terminals are `sigmoid(eval − root)`
  (`mcts.rs:226`) with −60 per death of ours, so the search avoids trades.
- **Phoenix Down revives forever in the engine (Neon; unverified).** Showdown keeps the ability
  after the revive and marks it in state the dump does not send (`abilities.ts:420`), so the
  converted Ho-Oh revives again on every search.
- **The Koga mirror with the arena is not a battler anchor (both):** Tailwind is on p2 only.

Order (each step's numbers decide the next):

1. **Fix the three bugs** (~1 h, ~15 lines): strip `-mega` in `engine_search.py`; send Phoenix
   Down as spent; treat a Tailwind ≥ 90 as permanent (no count-down, flat score).
2. **Re-measure the anchors at n = 48** (±14 points): mirrors at Erika, Koga (no arena), Blaine,
   each next to the **AI-vs-AI baseline** (`rr_mirror.py --p1 ai`), not 50%. Battler gate:
   ≥ 75% on the fair mirrors.
3. **Re-run the searched ceiling at Koga and Blaine.** Go if ≥ 40% at both: the battler bug was
   the wall; on to Phase 1 / 3b. Below 25%: steps 4–5.
4. **Search knobs on a sensitive test bed** (the EV-85 pool, or the losing pool sixes):
   death 0 / 30 / 60 and a won-terminal floor (`0.55 + 0.45 · alive/6`, `mcts.rs:226`);
   iterations 2k / 20k / 200k; the Showdown search at Blaine. Adopt only what holds on the
   Brock / Misty 30-box regression (±3 points, deaths per mon unchanged).
5. **Team archetypes, if the ceiling is still low:** speed control against Tailwind (Thunder
   Wave spread, priority), weather negation at Blaine (Cloud Nine / Air Lock), Rest only with
   Sleep Talk; plus human reference teams for Koga and Blaine (the user). A reference team ≥ 40%
   = builder wall; ≤ 10% = battler or fidelity.
6. **Opponent mixture** (the AI's reply with p = 0.5–0.7, else minimax) only if step 4 shows
   minimax pessimism on real pool sixes. Never the pure AI model (−13 to −33 points).

Gates for the learned team builder (3b): the battler gate above, and the searched chooser at
≥ 40% on Koga and Blaine. Until then expert iteration would learn from 0% labels. Also: check
permanent Tailwind on the ROM before Koga-specific work.

### 4. Encounter routing, G0 (`scripts/rr_route.py`)

Expected gain over location rarities, scored by whichever scorer won (Phase 1 or 3). It's arithmetic
on top of a scorer, not RL. Report it next to the settings ladder (E1).

### 5. Learning track only: policy head + pUCT, AlphaZero-lite

- **Rust work:** priors on `MoveNode`, a pUCT rule, and a leaf-value hook (3–5 days).
- **When:** only after 3, and only if it beats MCTS-20k on *strength* (speed is not the bottleneck).
- **How:** AZ-lite trains against a mix of the Rust AI and minimax, and is judged on Showdown
  against the Python RR AI.

## Not doing

- PPO from scratch
- direct RL for team choice or routing (too few episodes for the state)
- searching against the AI model (−13 to −33 points)
- a learned enemy-AI model (the 10% gap is a porting bug list, `results/rr_ai_diff.log`)
- species one-hots
- doubles (TODO.md F)

## Effort

About 7 working days to the end of Phase 4 if every gate passes. The Rust work (Phase 5) only
after that.
