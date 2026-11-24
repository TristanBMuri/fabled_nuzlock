# Cross-Review of `fable.md` and `lead.md` (Gemini Neon)

Verified read-only against `/home/tristan/CodingProjects/nuzlock_rl` (code + `results/` logs).

---

## 1. Agreements (Backbone of the Merged Plan)

1. **Single-agent PvE framing vs a known script.** Because Radical Red's AI (`ai_util/rr_ai.py`) is a known script, Nash-seeking search (Exp3 / Regret Matching / CFR) is the wrong target; leave `poke-engine`'s decoupled UCB1 selection intact except for the opponent-model mixture.
2. **Do not learn an opponent model.** Fix the Rust port (`poke-engine/src/genx/rr_ai.rs`) and add a per-node safe mixture (`p` AI model, `1-p` UCB1) as insurance against port error.
3. **No from-scratch PPO, no species IDs, no Python-per-leaf MCTS.** Respect `CLAUDE.md`'s minimal-code rule and attempt #3's lessons. Every model is species-free and evaluated on held-out bosses (every 4th boss) and held-out boxes.
4. **Sequential Halving + paired seeds + fresh-seed verification.** Replace uniform `24 × 4` screening with adaptive Sequential Halving using paired seeds (CRN) and a continuous score (`progress` + deaths) on low-win bosses, always reporting headline metrics on disjoint fresh seeds to eliminate winner's curse.
5. **Break the team ↔ battler loop by alternating with fixed anchors.** Anchor the battler independently on fair mirrors (+ AI-vs-AI baseline) and reference teams; run a 2k / 20k / 200k iteration ladder to diagnose battler- vs team- vs fidelity-bound walls and check cross-fidelity rank correlation ($\rho \ge 0.8$) before cheap screening.
6. **Surrogate = filter, search = judge.** Any learned team scorer proposes/shortlists candidates for Sequential Halving; never deploy raw argmax on a learned team scorer.
7. **Reuse existing repo physics and log everything.** Reuse `calc_bot.dmg_frac`, `matchup.rows_for`, `progress()`, and `attribution()`; add per-turn state + outcome logging so every battle feeds downstream value and team-filter fits.
8. **Strict anti-undertraining protocol before adding features.** No new feature without checking convergence, capacity (linear vs GBDT), and residual/ablation evidence.
9. **Routing stays arithmetic.** Phase-final encounter routing is expected value over the validated scorer, zero RL.

---

## 2. Where `fable.md` Is Better Than `neon.md` (What I Would Adopt)

1. **Minimal-code discipline & linear-first progression.** My `neon.md` jumped too early to a PyTorch Edge-Biased Set-Transformer ensemble (~280 lines) in Phase 2 and a Pointer Builder (~220 lines) in Phase 3. Fable's **Texel-tuned linear leaf evaluator** (~40 lines Rust + ~60 Python, convex, zero GPU, human-readable weights) and **linear/GBDT-before-net team filter** fit `CLAUDE.md` far better and ship days sooner.
2. **Unified Nuzlocke utility `U = win * (1 - λ * our_deaths / 6)`** (`fable.md:239`). Cleaner than my ad-hoc won-terminal floor (`0.55 + 0.45 * alive/6`) plus separate `POKE_ENGINE_DEATH = 60`. Terminals return `U` directly, eliminating the hand constant (though we still need a small depth tie-breaker so equal-`U` lines finish; see §3.4).
3. **Root-precomputed 6×6×2 damage table in Rust** (`fable.md:215–217`). Computing full-HP damage fractions *once per search at the root* (~300 calcs, <1 ms) and indexing with active boost multipliers at leaves is strictly faster and cleaner than running damage calcs inside every leaf. This makes matchup-aware leaf evaluation genuinely O(1)/µs.
4. **Exact-Model Oracle + On-Policy Agreement diagnostic** (`fable.md:149–162`). Wrapping the existing `rr_ai_choice(state, side)` binding (`poke-engine/poke-engine-py/src/lib.rs:1143`) as a 15-line enemy brain cleanly isolates "search/eval bug" from "Rust-vs-Python port gap" in 15 minutes. Fable also rightly caught the **omitted 25/50% random branches** in `rr_ai.rs:1832` (matching `ai_util/rr_ai.py:117, 210, 277, 542, 614`) as weighted enemy options in the tree.
5. **Statistical gating & feature-adequacy tools.** Paired generalized SPRT (`scripts/sprt.py`, ~25 lines) beats ad-hoc fixed-`n` tables across every phase gate. Measuring the **empirical noise ceiling** (replaying 200 sixes / 500 states 8× on fresh seeds) and running a **deep-search residual probe** (`MCTS-200k − static eval`) are sharper diagnostics than my learning-curve gate alone.
6. **Ordering Texel leaf-eval ahead of a learned team filter.** Every team search battle benefits from a matchup-aware leaf evaluation; improving the battler's leaf with ~100 lines of Rust+Python before investing in neural team models is higher leverage.

---

## 3. Where `fable.md` Is Wrong or Over-Built

1. **Delaying verified set-builder & pool hygiene bugs past Phase C ("Blame the walls").** Fable defers set-builder fixes to Phase D (`fable.md:413`) and misses four concrete code bugs already poisoning Koga/Blaine:
   - `build_mon` passes `trainer_idx=None` (`scripts/rr_team_probe.py:108`), leaving stone evolutions unevolved (`nuzleaf` at Blaine in `results/rr_ceiling_mega.log`).
   - `choose_nature` (`scripts/rr_probe.py:269–277`) assigns `+Spe` to any mixed attacker regardless of Base Speed — putting **Jolly on 30-Spe Coalossal** under Koga's permanent Tailwind (`results/rr_ceiling_mega.log`).
   - `solo_stage`/`greedy`/`cover_team` do not filter Truant or duplicate evolved species (Blaine fielded two `sawsbuck`s, one with Choice Band + `rest`; Koga fielded Truant `slaking`). Only `searched_cands` (`rr_team_probe.py:292`) filters them.
   - `cover_team` (`rr_team_probe.py:277`) returns `tuple(sorted(...))` (alphabetical lead!), and `box_job` (`:577`) never calls `searched_cands`.
   - Running Phase C's battler ladder on teams with Choice Band + Rest, Jolly Coalossal, Truant Slaking, and alphabetical leads risks misdiagnosing build bugs as battler/fidelity walls. Fix these ~45 lines in **Phase 0**.
2. **Underestimating late-game box size (`C(30,6) = 594k` vs reality).** `fable.md:100` claims *"C(30,6) = 594k sixes can be enumerated through a millisecond scorer."* Verified via `rr_probe.sample_box`: mean box sizes are **9 (Brock), 19 (Misty), 38 (Erika), 52 (Koga: $\binom{52}{6} = 2.0\times 10^7$), 72 (Blaine: $1.6\times 10^8$), 81 (Lance: $3.2\times 10^8$)**. Past Erika, enumeration is impossible; surrogate-guided **local search (1-swap / lead-swap LNS)** is mandatory.
3. **Phase F's T1 duel-cell calibrator rests on miscounted data and adds an unnecessary indirection.** Fable's T1 assumes 12,564 duel cells on disk (actually 2,094 Brock+Misty cells duplicated 6×; see §4). Moreover, 1v1 duel win rate ignores switching, pivoting, and multi-mon synergy; fitting a team-level GBDT/linear model directly on `matchup.rows_for` aggregates is simpler and avoids maintaining a two-stage T1 → T2 pipeline (only build T1 if we specifically want calibrated probabilities inside `cover`).
4. **Texel `U` without depth tie-breaking or absolute sigmoid change.** Two concrete Rust details Fable leaves implicit:
   - `poke-engine/src/mcts.rs:228` currently evaluates `sigmoid(eval - root_eval)` (relative to root). Texel logistic regression requires switching to **absolute** `sigmoid(eval)` (and verifying `mcts.rs` backup signs stay consistent).
   - Even with `U = win * (1 - λ*deaths/6)`, an open state predicting `0.70` can still beat a costly win (`U = 0.67` at 4 deaths, `λ = 0.5`), risking stall loops against a deterministic AI unless non-terminals include a tiny depth discount (e.g. `0.995^depth`) or `U` puts a small bonus on winning (e.g. `0.2 + 0.8 * win * (1 - λ*d/6)` or `turn_frac`).
5. **Phase D budget conflates pool search with `boxes30` search.** `fable.md:412` budgets ~10k battles/boss (~45 min/boss on 10 workers). That fits 1 pool-ceiling run per boss, not `boxes30` (30 × 45 min = 22.5 h/boss). Separate the two budgets explicitly: **fast per-box SH** (~100–130 cheap+medium battles, ~15–20 s/box) vs **wide pool/archive evolution** (~3k–10k battles/boss).
6. **Dismissing OOD surrogate traps too lightly.** `fable.md:120` dismisses offline conservatism because the simulator is cheap. Even with a cheap simulator, hill-climbing a learned filter over 1,000+ unseen sixes exploits model error; keeping a 3-seed ensemble uncertainty bonus/penalty (or Fable's own `top 24 + 6 diverse/random` shortlist at `fable.md:473`) is ~10 lines and prevents blind spots.

---

## 4. Factual Disputes Checked Against the Repo

| Claim | Where | Verdict & Verified Evidence |
|---|---|---|
| "`--engine-opp ai` had long passive battles, **no turn-limit losses**" | `fable.md:138` (quoting `NOTEBOOK.md:2669–2670`) | **FALSE.** In Pokémon, every non-timeout loss has `deaths_per_mon = 1.0`, so across 6 battles `deaths_per_mon >= 1 - win` is an algebraic invariant. In `results/rr_engine_ai_boxes30.jsonl`, **10/30 Misty boxes** violate `d >= 1 - win` (vs **0/30** in `rr_engine_boxes30.jsonl`), including **5 boxes at `0% win` with `0.139–0.417` deaths/mon** (1-indexed boxes 10, 12, 15, 19, 29 — e.g. Box 12 has only 5 deaths in 6 losses!). 8 of the 10 carry `paras` (Dry Skin in Rain), proving turn-100 timeout stalling via `sb.play(..., 100)` (`scripts/rr_team_probe.py:151–153`) + `mcts.rs:226`. *(Self-correction on `neon.md:183–185`: exact count is **10/30** guaranteed timeout boxes and **5** at `0% win / 0.14–0.42 d/mon` [boxes 10, 12, 15, 19, 29], not "9/30 and 6/30, boxes 8,10,12,15,19".)* |
| "12,564 duel cells are already on disk" | `fable.md:23, 37, 464` | **MISLEADING / FALSE as distinct cells.** `results/rr_matrix_calib*.jsonl` has 6 files × 2,094 rows = 12,564 lines, but all 6 are successive bug-fix reruns (`calib`, `_berry`, `_formula`, `_mod`, `_pp`, `_resist`) of the **same 2,094 cells on Brock and Misty only**. Only the latest file (2,094 rows, 2 bosses) is valid; no held-out boss test is possible without new duels. |
| "At Blaine, the **matchup matrix** proposed Cherubi, Deerling, Seedot" | `fable.md:42–44` | **FALSE attribution.** In `results/rr_ceiling_mega.log`, `cover` (the matchup matrix) proposed `('carkol', 'elgyem', 'ponytagalar', 'rapidashgalar', 'screamtail', 'slakoth')` — zero Grass types! It was **Stages A/B (`solo_stage` / `greedy` under `calc_bot`)** that picked `cherubi`, `deerling`, `deerlingwinter`, `seedot` (evolved to Flower Gift Cherrim + Chlorophyll Sawsbuck ×2 + unevolved Nuzleaf) because they outsped Blaine's lead in sun. |
| "`--chooser searched` is unwired in `box_job`" | `neon.md:207` | **TRUE.** `scripts/rr_team_probe.py:577` hardcodes `cover_team(...) if args.chooser in ("cover", "duel") else greedy(...)`; `searched_cands` only runs in `run_pool`. |
| "Box sizes allow enumerating `C(30,6)=594k`" | `fable.md:100` | **FALSE for late bosses.** Verified `rp.sample_box` means: Brock 9, Misty 19, Erika 38, Koga **52**, Blaine **72**, Lance **81**. |
| "`rr_ai_choice(state, side)` exists" | `fable.md:38` | **TRUE** (minor path fix: `poke-engine/poke-engine-py/src/lib.rs:1143`, not top-level `poke-engine-py/`). Also confirmed `mcts.rs:175` branches damage rolls only at root & depth 1, `mcts.rs:418` toggles `use_last_used_move = true` only in `vs_ai`, `rr_ai.rs:167` calls `root_get_all_options()` at every node, and `rr_ai.rs:1832` omits CFRU's 25/50% random branches. |
| "91 non-double bosses in `rp.pools()`" | `neon.md:208` | **SELF-CORRECTION.** `rp.pools()` has **213 total**, **169 non-double**, **78 named non-double** bosses (`RR_PLAN.md:78` cites ~75). |
| "`items.json` has 28 non-mega items, missing Sash/Specs/Vest/resist berries" | `neon.md:206` | **TRUE.** Verified 79 total items: 51 mega stones + 28 non-mega items; Focus Sash, Choice Specs, Assault Vest, and resist berries are all absent. |

---

## 5. Merged Feature Engineering Spec (Keep / Drop / Add)

**Rule:** Zero species IDs everywhere. Reuse `calc_bot.dmg_frac`, `matchup.rows_for`, and `evaluate.rs`.

### A. Battle-State Features (Rust `evaluate.rs`, Texel Linear First)
- **Keep from Fable (V0 + V1, ~28–35 floats):** All existing V0 material/status/boost/hazard/screen/permanent-aware Tailwind terms, PLUS the **root-precomputed 6×6×2 damage table** features: active `hit_frac` / `take_frac` (capped at 1), `we_move_first` (effective speed + priority), `we_ko_first`, `answers_left`, `unanswered_threats`, `sweep_exposure` (enemy active boost-aware OHKO count), `safest_switch_in`, temporary weather/terrain/TR turns left.
- **Add from Neon:** `turn_frac = min(turn / 80, 1)` (or depth tie-breaker) to guard against turn-100 stalling; healer-unbreakable flag (`V2`: foe active heals and `ko_them > 2`).
- **Drop (unless residual probe demands it):** Neon's full 1,100-dim Phase 4b Python battle tensor (only needed if Texel linear + 1-hidden-layer Rust MLP both fail).

### B. Team / Set Scorer Features (Pre-Battle, Tabular First → Deep Sets / Edge-Biased Attention Only If Gated)
- **Keep (Stage T-Tab, ~35–40 floats, for Ridge / LightGBM / TabPFN probe):**
  - **Cover & backup (Fable + Neon):** Per boss mon best and **second-best** `edge` / `safe` / `race` (or calibrated cell prob); min-over-enemies best answer (weakest link); fraction with ≥1 `safe`, ≥2 `safe`, ≥1 `switch_in_safe`.
  - **Lead & pivot:** Lead-vs-lead `safe`, `race`, `edge`, `take`; mean and worst `min_i take_ij` across boss mons; pivot free switch-in count (immune / 4× resist to predicted AI move).
  - **Speed, sweep & margins:** Count faster than ≥4 foes under arena; priority count; max count of our mons OHKO'd by one boss mon (fixes Carnivine liability); mean attack/defence stat margin vs boss; `cap − boss_top_level`.
  - **Arena & set coherence:** Arena-fit delta (row score with arena minus without); weather-negator/setter, recovery, status-spreader, mega count; set-sanity flags (`choice_with_status`, `rest_no_sleeptalk`).
- **Keep if scaling to a Set Net (Stage T-Set, only if T-Tab stalls >2% below noise ceiling):** Shared per-mon tokens (~50 floats: level-scaled stats, 18 types, functional item/ability buckets — NOT 577/342 raw IDs — `is_lead`, `rr_ai._class` for boss mons, row summary) + `6×6×10` `matchup.rows_for` edge features (edge-biased cross-attention or Deep Sets over mon+row tokens). Train with **within-`(box, boss)` ranking loss + BCE(win) + 6 per-mon death heads + HL-Gauss(progress)**.
- **Drop:** Fable's separate T1 1v1 duel calibrator as a mandatory prerequisite (stale 2-boss data; optional later for `cover` only); Neon's 100-dim bloated mon token (halve it by keeping only high-signal move/ability/item buckets).

---

## 6. Proposed Merged Phase Order & Verdict on `lead.md`

Every gate uses paired seeds, `scripts/sprt.py` where applicable, and fresh-seed verification ($n \ge 48$) for headline numbers.

1. **Phase 0 — Hygiene, Instruments & Known Fixes (~1 d; ~130 lines; <30 min compute)**
   - Queued fixes: `-mega` strip (`rl/engine_search.py:84`), permanent Tailwind $\ge 90$ (`evaluate.rs`), spent Phoenix Down (`convert.py`/`sim_bridge.js`), unified terminal utility `U = win*(1 - λ*deaths/6)` + depth tie-breaker (`mcts.rs`).
   - Builder/probe hygiene: drop `rest` from `STATUS_MOVES` (or require Sleep Talk), block Choice+status, fix `choose_nature` `+Spe` threshold, filter Truant + duplicate evolved species in `run_pool`/`cover`, enable stone evos (`trainer_idx`), audit `data/rr/items.json` for Sash/Specs/Vest/resist berries, wire `--chooser searched`/`sh` into `box_job`, log timeouts & per-turn states (`<out>.states.jsonl`), add `scripts/sprt.py` (~25 lines).
   - **Gate:** Logging on vs off produces identical paired battles; Misty timeouts = 0%.
2. **Phase 1 — Opponent Model in the Tree (~1 d; ~45 lines Rust/Py; ~1 h compute)**
   - Run Fable's exact-model oracle (`rr_ai_choice` as enemy) + on-policy agreement log + post-fix `--engine-opp ai` rerun. Implement per-node RNR/DBR mixture (`p` RR-AI, `1-p` UCB1, ~15 lines in `mcts.rs`); if oracle wins ≥+10 pts, port CFRU's 25/50% random branches as weighted enemy options and fix top switch-logic diffs to ≥97% on-policy.
   - **Gate:** Paired SPRT ≥ +8 pts over minimax on Misty + Erika boxes, deaths/mon non-worse, Brock within ±3 pts. Else keep minimax.
3. **Phase 2 — Anchors & Wall Diagnosis (~0.5 d; ~20 lines; ~1 h compute)**
   - Fair mirrors (Erika, Koga no-arena, Blaine, $n=48$) vs AI-vs-AI baseline; 2k / 20k / 200k battler ladder on Koga/Blaine + user reference teams; cross-fidelity Spearman $\rho(\text{2k}, \text{20k})$.
   - **Gate:** Mirrors ≥ 75% (+20 pts over AI-vs-AI). Reading routes next step: if 200k − 20k ≥ +15 pts → swap Phases 3 and 4; if reference teams ≤ 10% and ladder flat → stop and check ROM/arena fidelity (`data/rr/FIDELITY.md`); else proceed to Phase 3. Also adopt 2k early screening iff $\rho \ge 0.75$.
4. **Phase 3 — Non-Learned Team & Set Search at Scale (~1.5 d; ~100 lines; ~2–3 h compute)**
   - Implement CRN Sequential Halving (`--chooser sh`) in both **fast per-box mode** (~120 battles, ~20 s/box for `boxes30`) and **wide archive mode** (~3k–6k battles/boss for Koga/Blaine ceiling + dataset). Include targeted 1-swap / lead-swap / marginal-row-gain set-edit mutations seeded from a lightweight **MAP-Elites archetype archive** (weather answer, speed control, pivot core, mega). Collect offline `(six, boss)` + per-turn state archive across 68/78 named singles bosses (every 4th boss held out).
   - **Gate:** Fresh-seed $n=48$: Koga & Blaine ceiling ≥ 40% legal; `gyms30` Misty–Erika ≥ +10 pts over `cover`. **Skip Phase 5** if fast per-box SH already hits target speed and headroom vs wide search is <3 pts.
5. **Phase 4 — Texel-Tuned Rust Leaf Evaluation (~1.5 d; ~100 lines Rust + ~70 Py; fit seconds, test ~1 h)**
   - Expose `evaluate_features(state)` in Rust; fit V0 (existing terms) then V1 (root-precomputed 6×6×2 table features) by L2 logistic regression on battle-split outcomes `U` + deep-search root-value check, weighted by `p(1-p)` (PLR-style).
   - **Gate:** Held-out-boss log-loss ≥ 5% below hand eval AND paired SPRT ≥ +5 pts or −0.05 d/mon on Brock/Misty/Erika + Koga/Blaine beds, mirrors non-worse. Escalate to a tiny 1-hidden-layer Rust MLP (32 units) *only* if GBDT beats linear by >2% log-loss on the same features.
6. **Phase 5 — Learned Team Filter + Surrogate-Guided LNS *(conditional)* (~2 d; ~180 lines; min GPU/CPU)**
   - Start with **TabPFN v2 + LightGBM / Ridge** on the ~35–40 tabular features (plus species-ID oracle gap & noise-ceiling check) on Phase 3's archive. Use the best filter to drive 1-swap/lead-swap LNS (top 18 + 6 diverse/uncertain → SH verification). Only build a small PyTorch Set-Transformer (ranking + BCE + 6 death heads + HL-Gauss) if tabular models sit >3% Spearman below the noise ceiling. Re-collect hard boxes via PLR regret.
   - **Gate (held-out bosses):** True top-3 in top-24 ≥ 80%; end-to-end ≥ 90% of wide Phase 3 win rate at ≤ 1/6 battles.
7. **Phase 6 — Encounter Routing (~1 d; ~100 lines; ~30 min verify)**
   - Expected-value arithmetic over whichever validated scorer survived (Phase 3 SH or Phase 5 filter).

### Verdict on `lead.md` Ideas

| `lead.md` Idea | Decision | Rationale |
|---|---|---|
| **Sequential Halving** | **Include (Phase 3)** | Unanimous win; ~25 lines, slashes noisy screening waste immediately. |
| **MAP-Elites / QD archive** | **Include, lightweight (Phase 3)** | ~30 lines; prevents candidate collapse (e.g. Grass into Blaine) and gives a clean regret metric for PLR sampling. |
| **TabPFN v2 / 2.5** | **Include as Phase 5 diagnostic probe only** | Near-zero-code signal check on 2k–10k tabular `(six, boss)` rows alongside LightGBM; too slow/heavy for inner-loop LNS/routing. |
| **BRExIt opponent search** | **Adopt core lesson only (Phase 1)** | Use exact `rr_ai.rs` + per-node RNR/DBR mixture (`mcts.rs`); no learned opponent model or PUCT needed since we own the script. |
| **GRPO / RLOO builder** | **Reject / defer past Phase 5** | Unnecessary RL complexity while surrogate-guided 1-swap/2-swap LNS + SH solves the 52–81 mon box cleanly; revisit only if LNS demonstrably misses multi-mon cores. |
| **LLM-evolved heuristics** | **Reject** | Violates `CLAUDE.md` minimal-code rule and risks seed overfitting; Texel logistic regression + deep-search residuals give convex, readable, deterministic improvements in seconds. |
