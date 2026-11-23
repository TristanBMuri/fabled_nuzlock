# RL for Radical Red — plan (2026-11-23)

Drafted by the lead, reviewed independently by Fable and Opus against the code; this is the merged
version. Nothing here is built yet.

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

## Phases (gated; stop at any failed gate)

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
