# Fable's review of neon.md and lead.md (round 2, 2026-11-24)

Read-only; no battles run. Everything tagged "checked" was verified against the repo today
(file:line or a result file). Numbers I computed from `results/*.jsonl` are marked "computed".

## 1. Agreements (the merged plan's backbone)

Both plans, and lead.md where it speaks, share these:

1. **Diagnostics and bug fixes before any learning.** The three engine fixes, the won-terminal
   floor (`0.55 + 0.45 · alive/6` in both plans), then anchors (fair mirrors vs the AI-vs-AI
   baseline, reference human teams).
2. **The enemy model belongs in the tree as a mixture**: follow the Rust RR AI with probability
   p, else UCB, per node (Neon's "safe mixture", my RNR/DBR dial; the same ~15–25 Rust lines).
   Pure minimax is too pessimistic, the pure model is brittle. Nobody wants a learned enemy model.
3. **Sequential halving on paired seeds** replaces "24 sixes × 4 battles"; cheap low-iteration
   battles for early rounds only after a separability check (rank correlation across MCTS budgets).
   Headline numbers come from fresh seeds only.
4. **The battler is held fixed while teams are searched**, scores are tagged with the battler
   version, re-screen only the top-K after a battler change. No joint training.
5. **Any learned team model is a filter, never the judge**; search verifies. No species or boss
   identity features. Every 4th boss held out. Train vs held-out always reported.
6. **Team features = the existing matchup cell** (`matchup.pair`: hit, take, ko_them, ko_us,
   faster, heals) plus weakest-link, second-answer and lead aggregates, computed by `calc_bot`.
7. **A tree model on team aggregates before any network**; the network is deleted if it does not
   beat the tree. The first learned battler piece is a **small leaf value in Rust** fitted on
   outcomes over hand features that include damage and speed order (Neon's Tier 4a = my Phase E).
8. **Log every evaluated battle** (per-team rows and per-turn states) from now on.
9. **Not doing:** PPO / model-free battler, Nash-seeking search (Exp3 / RM), pUCT with a Python
   callback, argmax over a surrogate, routing as RL.

## 2. Where Neon's plan is better than mine (what I would adopt)

- **It found the turn-cap losses; I repeated the notebook's wrong sentence.** I wrote "long
   passive battles, no turn-limit losses" from NOTEBOOK.md:2669. Neon looked at the per-box rows
   and saw losses with almost nobody dead. Checked and confirmed (§4, D1). This turns my H2 from
   a hypothesis into the measured main cause at Misty, and it gives a free instrument:
   **log `timeout` per battle** in `one_battle`. Adopt both.
- **`--chooser searched` is silently ignored in box mode** (D2). Every 30-box table so far is the
  cover or greedy chooser. My Phase D gate ("the 30-box table improves by ≥ 10 points at Erika")
  assumed the wiring existed. Adopt Neon's "wire the new chooser into `box_job`" as an explicit item.
- **The alphabetical lead** in `cover_team` and `greedy` (D3). A two-line fix that precedes any
  chooser comparison; I missed it.
- **The set-builder hygiene list, from the logs**: Rest on four of six Blaine mons including a
  Choice Band holder, +Spe on a base-30-Speed Coalossal, Truant Slaking and two Sawsbucks in the
  "best six" (D4). I noticed "Grass into Desolate Land" and stopped there. These are cheaper and
  more certain than my "moves by marginal row gain" rewrite; do them first, then mine.
- **Box sizes** (9 at Brock → 73 at Blaine, C(73,6) ≈ 1.7·10⁸). My "enumerate the box through a
  millisecond scorer" only holds to ~30 mons. Neon's 1-swap neighbourhood (≈ 400 neighbours)
  through the scorer is the right form for late boxes; it is also what my Phase D mutations do
  through real battles.
- **Per-mon death labels as extra targets** for the team model (six labels per battle instead of
  one). Cheap, and per-mon death risk is what a nuzlocke needs. I had it only at T3.
- **Keeping a few high-uncertainty picks in the shortlist** (I had "6 random"; disagreement
  between two or three cheap fits is a better use of the same slots).
- **The species-ID gap test** as a feature probe: a tree with species IDs vs without, on held-out
  boxes of seen bosses. A direct measure of "do the mechanics features carry what species would".
  It joins my §4.4 list as check 3b.

## 3. Where I think Neon's plan is wrong or over-built

- **Code size.** Phases 2–4 add `rr_features.py`, `team_scorer.py`, `team_builder.py`,
  `rr_set_refine.py`, `fit_leaf_eval.py`, `battle_net.py`, `rr_route.py`: about 1,100 lines of
  Python plus ~210 of Rust, an edge-biased Set Transformer, a 3-seed ensemble, a pointer decoder,
  HL-Gauss, Plackett-Luce, PLR and Gumbel pUCT. CLAUDE.md's first rule is against exactly this
  shape. Each piece has a delete-gate, but the plan builds the net and the tree in the same phase;
  the tree should run alone first and the net should need a named failure to exist.
- **Data vs model.** "25–35k (six, boss) records" are sixes from ~860 boxes with 2–4 battles each
  on the dropped halves. A label from 2 battles is one of {0, ½, 1}. A ~100-float token × 12 mons
  with a 150–250k-parameter attention model on that will fit noise; the plan has no **noise
  ceiling**, so "Spearman ≥ 0.65" has no reference (two independent 4-battle estimates of the
  same six may not correlate at 0.65 with each other). My §4.4 step 1 must precede Gate 2a.
- **Labels from halving are selection-biased.** Survivors are the sixes that were lucky early,
  and they get the most weight (weighted by battle count). Either label survivors only from their
  later rounds or add an unbiased random sample; my "plus 6 random" covers this.
- **Unbiased reporting at n = 6.** "Top 2 × 6 fresh-seed battles → pick the winner and report on
  those seeds" still picks a max and has a standard error of ±20 points. Headline numbers need
  n ≥ 48 on seeds used for no decision.
- **The learned battler order.** Phase 4 is last and gated on "200k beats 20k by 8 points". A
  material-only evaluation can be the wall without more iterations helping (the search cannot see
  what the leaf does not score, at any budget). The gate should be the fitted-vs-hand paired play
  test, and the refit of existing terms (my V0, ~100 lines, seconds to fit) is cheaper than the
  Set Transformer, so it should not wait behind it.
- **"1–2 µs per leaf" for 42 features that include bench safe / race counts** needs ~290 damage
  calcs per leaf unless they are precomputed. It only holds with a per-search 6×6×2 damage table
  built once at the root (my §4.2). Without that the leaf is 10–100× slower.
- **The RR-AI step skips the discriminating experiment.** Neon goes straight to "fix three
  disagreement clusters to ≥ 97% and sweep ε". That cannot separate "the port is wrong" from "the
  search is wrong given a right model". The 15-line oracle (the *Rust* AI plays the enemy for
  real, so the model is exact) does, and the Brock result needs it (D1).
- **A `0.995^depth` discount** is a second fix for the same symptom as the terminal floor. Add
  the floor, log timeouts, re-run; add the discount only if capped battles remain.
- **"MCTS-2k ≈ 0.3 worker-s a battle" and "rows_for ≈ 20 ms a box"** are not measured anywhere I
  can find; the whole "15–20 s per box" budget rests on them. Measure before planning on it.
- **Functional ability / item one-hots (18 + 14 per mon).** Most of their effect is already
  inside the cell (`dmg_frac` applies abilities, items, weather). Keep only what the cell cannot
  see: weather setter / negation, Intimidate, Sash-like "survives one hit", Choice lock, mega.
- **Gate 0b (pool ceiling ≥ 40% at Koga and Blaine) blocks everything**, although Koga's
  permanent Tailwind is an open ROM-fidelity question. A fidelity wall should not gate the method.

Where lead.md overreaches is in §6.

## 4. Factual disputes (checked)

**D1. Cause of the RR-AI search loss. Neon's cause A is real; the notebook line is false.**
- `SimBattle.play` force-loses p1 past the cap (`ai_util/sim.py:225–226`, called with 100 at
  `scripts/rr_team_probe.py:151`). Neon says `winner = None`; in fact p1 is declared the loser,
  same effect on the table.
- A normal loss costs 6 deaths, so per box `1 − win − deaths/mon > 0` proves capped battles.
  Computed from `results/rr_engine_ai_boxes30.jsonl` vs `rr_engine_boxes30.jsonl`:

  | | minimax | AI model |
  |---|---|---|
  | Misty: boxes with provable capped losses | 0 / 30 | **10 / 30** |
  | Misty: lower bound on capped share of all battles | 0% | **≥ 16.7%** |
  | Brock: same | 0 / 30 | 0 / 30 |

  So at least half of Misty's −33 points is the turn cap, a lower bound (a capped battle with
  several deaths is not counted). Mean deaths are equal (0.681 vs 0.687) while wins fall 51 → 18.
- Paras: 7 of the low-death losing boxes hold Parasect (Dry Skin, permanent rain). Neon's
  "every one" is right for those, but boxes without Paras also fall, 40% → 14% (20 boxes,
  computed), and two Paras boxes go to 100% wins at 0.03 and 0.08 deaths. Paras is one stall
  engine, not the cause. Neon's Rust mechanism (a deathless stall outscoring a win with a death)
  is plausible and untested.
- Brock (−13) has no capped battles and deaths rise 0.49 → 0.58: ordinary extra losses. That fits
  model error (my H1 / H3 / H4, Neon's cause B), and the oracle test is what separates them.
- **Neon's "the RR AI is deterministic modulo ties" is false.** `ai_util/rr_ai.py:210` and `:277`
  are 50% branches, `:614` a 2-bit random; the port omits them (`rr_ai.rs:1832` comment). The
  tree treats coin flips as certain. This is part of the port gap and is not in the diff log's
  cluster list.
- `vs_ai` gives the enemy only the model's choices, uniformly among ties
  (`poke-engine/src/mcts.rs:95–125`): confirmed in both plans. `rr_ai_choice_mut` calls
  `root_get_all_options` at inner nodes (`rr_ai.rs:167`): true; it only differs under team
  preview (`state.rs:1263`), so harmless here.
- lead.md's "almost surely a bug or a model mismatch": half right. Misty is an objective bug
  that the passive model exposes; Brock is probably mismatch. "Agreement on logged positions"
  already exists (90.3%, `results/rr_ai_diff.log`) and is off-policy; the open number is
  on-policy agreement.

**D2. `--chooser searched` in `box_job`: not wired. True.** `scripts/rr_team_probe.py:577`:
`cover_team(...) if args.chooser in ("cover", "duel") else greedy(...)`. "searched" falls through
to `greedy` with no warning. The screen exists only in `run_pool` (`:535`).

**D3. Alphabetical lead: true for cover and greedy.** `cover_team` returns `tuple(sorted(...))`
(`:277`), `greedy` sorts too (`:255–259`); only `searched_cands.finish` puts the best answer to
the boss's lead first (`:310–311`).

**D4. Set-builder claims.** True: `rest` is in `STATUS_MOVES` (`scripts/gauntlet.py:188`) and one
status move is always taken (`:236`); a mixed set gets +Spe whatever its base Speed
(`scripts/rr_probe.py:269–276`); `results/rr_ceiling_mega.log:18,28` shows Truant Slaking at
Koga and two Sawsbucks plus Rest on a Choice Band holder at Blaine. Qualified: `build_mon`
passing `trainer_idx=None` is documented intent ("no stones", `rr_team_probe.py:103`), a fidelity
decision, not a bug. `data/rr/items.json` has 79 entries and no Focus Sash, Choice Specs, Assault
Vest or Air Balloon (computed); whether the hardcore care packages give them is a data question
for the sheet, not established by either plan.

**D5. Screen ranking.** The screen sorts by (win, −deaths) (`:543–545`), not by `progress`. At
Koga (0–8% wins) that is nearly a deaths-only ranking. My claim stands; Neon's halving rounds
"by (win, progress, −deaths)" has the same problem while wins are near zero.

**D6. Feature leaks.** None in Neon's spec that use outcomes; `root_val_t0` and `per_mon_died`
are labels / baselines only. One in mine: the cell feature "status / setup opener … *or the
duel*" would feed the calibrator its own label; it must come from the AI model on the opening
state only. One trap in both: "held-out boxes of seen bosses" share the boss six and most
candidate sixes with the training boxes, so only held-out *bosses* count as generalisation.

**D7. Dependencies.** `pyproject.toml` has torch and no tree library. Both plans' GBDT step adds
one dependency (scikit-learn's `HistGradientBoosting` is enough); TabPFN would add another.

## 5. Feature engineering: merged view

**Keep (both plans):** the cell from `matchup.pair` recomputed under the arena; `safe / race /
edge` as features; per-mon row summaries (answers, liabilities, worst incoming hit, outspeed-and-
OHKO count); team aggregates with min-over-boss-mons, **second-best** answer, lead cell, speed
control under the arena, sweep exposure; arena flags; level cap − boss level.

**Adopt from Neon:** `boss_ai_class` per boss mon (how the script pilots it; existing
`rr_ai._class`); `they_have_prio_ko` / `we_have_prio_ko`; type-weakness overlap (max number of
ours weak to one boss attacking type); set-coherence flags as *rules in the builder* rather than
model inputs; `turn_frac` and a permanent-vs-temporary split of weather / Tailwind in any battle
feature set; per-mon death labels.

**Keep from mine:** the **calibrated cell** `c_ij = P(i beats j)` fitted on the 12,564 duel rows
on disk (replaces hand tiers, usable inside `cover` on its own); stat margins (the 85-EV result
says margins decide Koga / Blaine; Neon has no margin feature); arena fit (row with vs without
the arena); pivot value against the AI's *predicted* move; the leaf features 8–15 from a
per-search damage table; one utility target `U = win · (1 − λ · deaths/6)` with wins and deaths
reported separately.

**Drop:** species / boss IDs (except inside the diagnostic gap test); the 18-way ability and
14-way item one-hots beyond the handful the cell cannot express; the 100-float tokens and the
6×6×14 edge tensor until a tree on ~30 aggregates is shown to be capacity-limited; the full
1,200-float battle observation (Neon §3.2C) unless a battle net is ever gated in; HL-Gauss and
Plackett-Luce until a plain logistic / tree fit is shown to be the limit.

**Adequacy protocol (merged, in order):** (1) noise ceiling from replays; (2) convergence curve
at ¼ / ½ / 1× data; (3) tree vs linear on the same features, (3b) species-ID gap; (4) a residual
bucket that names the feature (≥ 5 points, n ≥ 500); then group ablation; the paired play test
outranks every loss number. TabPFN can serve as a second opinion in step 3 (§6).

## 6. Proposed merged phase order

| # | Phase | Gate |
|---|---|---|
| 0 | **Hygiene** (½ day): engine fixes, won-terminal floor, timeout logging, lead fix, `searched` wired into `box_job`, Rest / Choice / +Spe rules, Truant + dupes filter in pool mode, per-turn + per-battle logging, SPRT helper. | Paired results identical with logging on; zero capped battles under minimax; Brock / Misty tables within ±3 of today. |
| 1 | **Opponent in the tree** (1 day): re-run `--engine-opp ai` after phase 0; exact-model oracle (Rust AI as the real enemy); on-policy agreement; mixture dial p. Then port fixes incl. the random branches as weighted options. | Paired SPRT: +8 points vs minimax on Misty + Erika, deaths not worse, capped battles 0, Brock ±3. Else minimax stays. |
| 2 | **Blame the walls** (½ day): battler ladder 2k / 20k / 200k, reference teams, mirrors; separability (Spearman across rungs); measured cost of a 2k battle. | A reading: battler / team / fidelity. ≥ 0.8 → cheap early rounds allowed. |
| 3 | **Team + set search, no learning** (1–2 days): sequential halving on paired seeds, continuous score while wins < 20%, 1-swap / lead / move / item mutations, best six per archetype flag; in both pool and box mode. Writes the archive. | Fresh seeds, n ≥ 48: Erika boxes +10 points; Koga / Blaine ≥ 40% *or* a fidelity reading from phase 2. |
| 4 | **Fitted leaf evaluation** (2 days): Texel refit of the existing terms, then the table features; Neon's tiny MLP only if a tree beats linear by > 2%. | Held-out-boss log-loss −5%; paired SPRT +5 points or −0.05 deaths/mon. Fit passes, play fails → keep the terminal utility only. |
| 5 | **Team filter** (2–3 days), only if phase 3 is compute-bound: cell calibrator → tree on aggregates (+ per-mon death heads) → set model only on a named failure. Used inside 1-swap local search; top 24 + a few uncertain picks go to halving. | Held-out bosses: true top-3 inside the top 24 in ≥ 80%; ≥ 90% of phase 3's win rate at ≤ 1/10 of its battles. |
| 6 | **Routing**: arithmetic over the surviving scorer. | Routed boxes +10 points clear rate vs random-method boxes. |

Order between 3 and 4 follows phase 2's reading (ladder steep → 4 first). Not in the plan unless
a gate names them: pointer builder, Set Transformer, ensemble, Gumbel pUCT, battle net.

**lead.md's ideas:**

- **Sequential halving: include** (phase 3; all three documents agree). At the battle root:
  no, the budget is 20k iterations over ≤ 9 actions.
- **BRExIt-style opponent search: include the search half, reject the rest.** The model in the
  tree with a mixture is phase 1. No learned opponent model, no expert-iteration loop around it.
  Their note that Q-values override the model at high budgets is an argument for the mixture.
- **TabPFN: include as a probe only.** A no-training second opinion in the adequacy check (if
  TabPFN on the aggregates beats the tree clearly, the tree is undertrained or badly tuned). Not
  the deployed filter: each prediction carries the whole training table in context, so scoring
  400 neighbours × several local-search steps × thousands of boxes is far from a millisecond
  scorer, it adds a dependency and a weights download, and the v2.5 licence should be read first.
- **MAP-Elites archive: include the 10-line version** (best six per archetype flag as seeds and
  survivors in phase 3). No QD library, no behaviour-descriptor grid.
- **GRPO pointer builder: reject for now.** Its reward is a searched battle (2.7 worker-s, a
  Bernoulli); K samples per (box, boss) over enough cases to train a policy is the budget of
  phase 3 many times over, and local search through a cheap filter covers the same ground. It
  becomes a candidate only if phase 5 passes and 1-swap search through the filter demonstrably
  misses sixes that wide search finds.
- **LLM-evolved heuristics: reject the loop, keep the idea.** Weights are a convex fit (phase 4),
  no evolution needed. New *terms* can be proposed by anyone, including an LLM, but each must
  pass the residual gate and the paired play test; a fitness call is a Rust rebuild plus ~500
  battles, so an evolutionary loop gets ~20 candidates an hour and overfits its seeds. An
  OpenEvolve-style harness is infrastructure the project has ruled out twice.
- **Regret-based case selection (PLR): include as sampling weights** (~10 lines) in phases 4–5.
- **PQN / TD value learning, offline RL: reject** (we regress on outcomes; the simulator is cheap).
