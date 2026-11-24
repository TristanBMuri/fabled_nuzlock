# Round 3 Review — Gemini Neon (independent)

## 1. Movepool for sets (measured from `data/rr`)

`data/rr/learnsets.json` + `data/rr/tms.json` already match the user's legality rules: `scripts/build_rr.py:45,249-279` collects `9L` (level-up), `9M` (TMs), and `9T` (hardcore move tutors, greyed-out tutors dropped), excluding egg moves (`9E`). Measured across the full encounter pool under `--legal` (`data/rr/hardcore_rules.json:95-178`, 82 banned moves):

| Boss | Cap | TMs+Tutors | Pool / mon (med / p90 / max) | Level-up + TM med | Atk / Status med |
|---|---:|---:|---|---:|---:|
| Brock | 16 | 6 | 8 / 11 / 14 | 6 + 1 | 5 / 2 |
| Misty | 28 | 26 | 14 / 18 / 23 | 9 + 4 | 11 / 3 |
| Erika | 44 | 72 | 27 / 33 / 40 | 12 + 14 | 20 / 6 |
| Koga | 68 | 111 | 35 / 44 / 55 | 15 + 20 | 28 / 7 |
| Blaine | 76 | 141 | 37 / 48 / 60 | 15 + 23 | 30 / 7 |
| Lance | 85 | 148 | 39 / 49 / 65 | 15 + 23 | 31 / 7 |

Pre-evo level-up moves add only 0.3–0.4 moves on average (max 8) — cheap to union along the evo chain so stone-evolved mons keep pre-evo moves. Crucially, because RR hardcore bans setup, hazards, weather/terrain moves, Protect, Sub, and Taunt, the *non-dominated* pool is small: median **12–13 distinct `(type, category)` attacks** and **3 useful status/utility moves** at Koga/Blaine.

**How set search should choose moves (2-stage):**
1. **Boss-aware prefilter (<1 ms/mon, pure Python, memoised per `(species, boss)`):** shrink ~37 legal moves to a **6–10 move shortlist** per mon: (a) highest `calc_bot.dmg_frac` move against each of the 6 boss mons (≤6, usually 2–4 unique); (b) best priority move (`fakeout`, `suckerpunch`, `iceshard`, etc., which `best4` currently drops because 40 BP < 90 BP STAB); (c) best pivot (`uturn`/`voltswitch`/`flipturn`/`partingshot`); (d) best recovery (no Rest without Sleep Talk); (e) top legal disruption (`encore`, `knockoff`, sleep/para/burn, `haze`/`clearsmog`, `counter`/`mirrorcoat` if Sash/Sturdy — lift those from `NO_SLOT_MOVES` in `scripts/gauntlet.py:181-185`).
2. **Boss-aware init + Phase 3 polish mutations:** initialize every mon with the best 4 from its shortlist covering the boss's 6 (plus item/nature compatibility). Do **not** search moves over every candidate six — only run 1-move / 1-item / 1-nature mutations from the 6–10 move shortlist on the **top 4–6 halving survivors** in Phase 3.

**Cost & feature impact:** prefilter adds <0.1 s/box; polishing top 4 sixes adds ~24–48 battles/box (fast) or ~150–300 (wide). Leaf eval needs no feature change (it reads the mon's 4 moves); add 3 cheap team aggregates to T1 (`has_fakeout`, `has_pivot`, `has_disruption`) that `best4` previously hid.

---

## 2. Visualising how the planner decides

Reuse `ui/` (`ui/build.py` + `ui/index.html`, port 8765) for all planner/battle inspection — it already has Runs, Compare, Deaths, Traces + Showdown `/replay`, and Matrices. Keep `rl/cockpit.py` (port 8766) for Phase 4/5 model-training curves.

**Minimum logged fields:**
- **Draft explanation** (extend `<out>.events.jsonl` in `scripts/rr_team_probe.py`; `screen` already exists for pool mode at `:548`, emit for box mode too):
  - `candidates`: list of `{id, source (cover|qd:weather|1-swap|move-mut), team, lead, sets, rounds: [(r, n, win, deaths, prog)], dropped_why}`
  - `winner_roles`: `{boss_mon: [{ours, tier, ko_them, ko_us, faster, best_move}]}` (which pick answers each enemy threat, plus backup)
  - `runners_up`: top 3 rejected sixes with `diff` (swapped mon/move/item) and `Δwin / Δdeaths`
  - `provenance`: one small dict per run naming what each component read (`set_builder`, `chooser`, `battler`, `eval_weights`)
- **Per-battle search transparency** (extend Phase 0 `<out>.states.jsonl` and `scripts/rr_trace.py`):
  - Per turn: `state_str`, `s1_root: [(choice, visits, mean_q)]` (already returned by `poke-engine` at `rl/engine_search.py:84`, currently discarded), `s2_pred` (Rust `rr_ai_choice` weighted set), `s2_actual`, `s2_agreed`, `expected_deaths`, `fallback_why`.

**Cheap vs expensive:**
- *Cheap (~0 ms overhead, ~5–15 KB/box):* `draft` events, `winner_roles`, `runners_up`, and per-turn root visits + `s2_pred` (already computed).
- *Moderate (~5–20 MB / 1k battles):* full `state_str` in `<out>.states.jsonl` — gate behind `--log-states`.
- *Expensive:* full Showdown protocol logs (`trace_*.json`) or internal MCTS subtrees. Emit protocol traces only on demand (`rr_trace.py`) or automatically for the **worst 3–5 bad battles** per run.

**UI additions (`ui/index.html`, ~150 lines vanilla JS):** (1) **Draft drawer** inside each box/boss row (Threat-Coverage table, halving funnel, runner-up diff); (2) **Trace upgrade** (MCTS visit/Q bars per candidate + `s2_pred` vs `s2_actual` with red badge on mismatch + Showdown `/replay` button); (3) **Line tab** rendering `logs/line_*.json` (collapsible tree + riskiest-branches table).

---

## 3. Showdown cross-check of bad battles

Key codebase fact: `EngineSearcher` (`rl/engine_search.py:3`) **already runs the real battle on Showdown (`SimBattle`)**; only the *search* runs in `poke-engine`. So a Rust bug hurts by making MCTS hallucinate wrong outcomes (e.g., `-mega` drop, Tailwind 99 stall, infinite Phoenix Down) or predict the wrong AI move.

**Concrete module (`scripts/rr_crosscheck.py`, ~90 lines):**
1. **Trigger (automatic triage):** pick up to K=5–10 bad battles from a run: (a) loss or ≥2 deaths for a team with ≥50% win rate, (b) any 100-turn `timeout`, (c) turn-0 MCTS root value ≥0.65 that lost, or single-turn root value drop ≥0.30.
2. **Step 1 — Per-turn transition & AI diff (~2 s/battle):** replay the logged `(c1, c2)` sequence on `SimBattle` with the saved seed; at each turn run `rl/engine_diff.py:36` (`compare(..., samples=100)`) and `rr_ai_choice` vs actual `c2` (`rl/rr_ai_diff.py:196`). Pinpoints the exact turn where `poke-engine`'s physics (`missing`/`extra`/`tv`/`dhp`) or AI model diverged from Showdown.
3. **Step 2 — Pure-Showdown policy replay (~20–30 s on 6 workers):** replay the same `(team, boss, arena)` on the bad seed + n=12 paired seeds using pure Showdown rollout search (`ai_util/evaluate.py:75` `Searcher(k=9, rollouts=8)` — zero Rust). Compare `(win, deaths/mon)` and find the first turn where `EngineSearcher` and Showdown `Searcher` disagree with Showdown-rollout gap ≥0.25.
4. **Diagnosis:** Step 1 flags → engine/AI bug to fix. Step 1 clean & Step 2 wins ≥+25 pts → search/eval bug (like Tailwind 99 or terminal floor). Step 1 clean & Step 2 also loses → genuine team/RNG wall.

**Cost:** ~30–60 s total for 5–10 flagged battles; run automatically at the end of milestone runs, never per battle.

---

## 4. Line planner (battle flowchart) — build right after Phase 1

With the exact RR AI model from Phase 1, the fight is a single-agent stochastic MDP (AND-OR / expectimax tree). `poke_engine_rr.generate_instructions(state, s1, s2)` (`poke-engine/poke-engine-py/src/lib.rs:1085`) **already sets `branch_on_damage = true`** and enumerates chance outcomes with probabilities: speed ties 50/50 (`generate_instructions.rs:4930`), crits & kill-rolls (`:2757-2785`), misses (`:1434`), secondaries (`:1229`), sleep/para/freeze/confusion (`:2052-2265`), and multi-hits (`:2802`).

**Two engine gaps to handle explicitly:**
1. *Non-lethal damage rolls:* `generate_instructions.rs:2752` collapses non-lethal 16 rolls to `avg_damage_dealt`. A high roll on turn 1 can put you in KO range on turn 2. Fix: query `calculate_damage` (`poke-engine-py/src/lib.rs:1092`, returns all 16 rolls) per edge, flag any turn where `[min..max]` crosses a 2HKO / berry / AI-switch threshold ("roll-sensitive"), and add a `--pessimistic` check (enemy max roll / our min roll on non-crits).
2. *AI random branches:* Phase 1 must expose weighted `(ai_choice, prob)` from `rr_ai_choice` (ties + CFRU 25/50% branches); each turn's chance layer is the Cartesian product of AI choices × `generate_instructions`.

**Algorithm (`rl/line_planner.py`, ~200 lines Python; AO\*-style hybrid, Nilsson 1971 / LAO\* 2001 recalled):**
- **Action selection (reuses Phase-1 MCTS, doesn't replace it blindly):** at high-mass nodes (`p_reach ≥ 0.05`), MCTS-20k nominates top `K=2–3` moves, and a 2-ply exact expectimax over true chance outcomes picks the move maximizing `E[U]` (prefers a 100% safe line over a 96% line MCTS noise might pick). At low-mass recovery nodes (`p_reach < 0.05`), take MCTS's top-1 move directly.
- **Pruning & merging:**
  - *Probability threshold:* stop expanding when `p_reach < ε` (default `ε = 0.01`, tunable to `0.002` for E4); score unexpanded tail leaves with MCTS-20k / 20 Rust rollouts and report `tail_mass` explicitly.
  - *State merging (DAG):* hash `(active_ids, fainted_mask, statuses, boosts, field, hp_exact)` (optional 5% HP bucketing below `p_reach = 0.05`) to merge equivalent states — cuts tree size 5–10×.
  - *Display collapsing:* merge sibling RNG branches that lead to the exact same optimal action and zero deaths ("hit or crit → same follow-up") while keeping exact probabilities underneath.
- **Outputs:** (a) CLI indented text flowchart; (b) `logs/line_<boss>.json` + `ui/` interactive tree + Mermaid export; (c) root & per-node exact `P(clean win)`, `P(win with deaths)`, `P(loss)`, `P(tail)`, per-mon `P(death)`, and a **"Riskiest Branches"** table sorted by `p_reach × P(bad)`; (d) **Showdown sanity check:** 100 `SimBattle` replays following the line's lookup table (MCTS fallback off-tree) to verify predicted vs Showdown clean-win %.
- **Cost:** a 12-turn fight at `ε = 0.01` expands ~15–60 decision nodes → **~5–20 s per boss** on 1 core.
- **Later value for autonomous bot:** (1) zero-cost opening book for known bosses; (2) low-variance team scoring (`P(clean)` without 6-battle Monte Carlo noise); (3) mines high-regret failure states for Phase 4/5.

---

## 5. End use = Advisor first, Autonomy later

Prioritising an advisor for the user's real playthrough reshapes the plan in four ways:
1. **Line planner + draft explanations move forward** (Phase 1.5 and Phase 3): a bare 6-mon list without *why* and *how to play around crits* is useless at the ROM.
2. **Add `--my-box <file>`** to `rr_team_probe.py` / `line_planner.py`: feed the user's real box + boss → get top 3 teams (with threat map & sets) + cooked line in 1–2 minutes.
3. **Compute budget relaxes for single-boss advising:** spending 60–120 s on one real box is fine, so **Phase 5 (learned team filter) is NOT needed for advising** — it only pays off for 30-box benchmarks and Phase 6 encounter routing.
4. **Fidelity outranks ML polish:** one wrong boss item/ability/arena rule breaks a real ROM line; finishing `data/rr/FIDELITY.md` checks + Showdown cross-check has higher ROI than neural models.
5. **Out of scope for now:** emulator/ROM I/O, fog-of-war/hidden-info play, <10 ms real-time inference, full 100-trainer autonomous loops.

---

## 6. ML ambition check (honest assessment)

| Method | Plan v2 stance | Holds? | Concrete evidence / data size to switch |
|---|---|---|---|
| **Texel linear leaf eval → small Rust MLP (Phase 4)** | Linear first; 1×32 MLP in Rust only if GBDT beats linear by >2% log-loss | **Mostly holds; slightly timid on MLP** | Linear is ~0.1 µs vs ~0.5–1 µs for a 35→32→1 Rust MLP (still 100× faster than Python). Pokémon has real interactions (`faster × we_KO`). Lower trigger to **>1% held-out-boss log-loss OR ≥+3 pts play** once we have **≥50k states across ≥25 bosses**. |
| **Policy prior + pUCT (AZ-lite in Rust)** | Deferred ("eval-bound, not width-bound") | **Holds for now** | Singles has only 4–9 legal actions; 20k UCB1 visits all of them. Switch if Phase 2 ladder shows **200k beats 2k by ≥+15 pts** AND we need strong play at ≤500 iters for fast team/route search, with **≥200k `(state, visits)` across ≥40 bosses**. |
| **Team scorer: GBDT (T1) vs Set Transformer (T3)** | GBDT first; Set Transformer only on named failure of T1 | **Holds at <5k rows; slightly conservative at ≥10–15k rows** | Tabular GBDT/TabPFN wins at ~800 rows (Hollmann 2025 seen; McElfresh 2023 recalled). But 35 scalar aggregates lose 2-body pivot/bait synergies. With an RTX 5090, run a small 6×6 cross-attention Set Transformer (with pairwise matchup matrix as attention bias + per-mon death heads) as an **open Phase 5 experiment once we hit ≥10k labeled sixes across ≥30 bosses** — adopt if it beats T1 on held-out bosses. |
| **GRPO / RLOO pointer builder** | Rejected (2.7 s Bernoulli reward) | **Holds vs raw battles; viable later via surrogate / ExIt** | Online GRPO on 2.7 s battles is too slow. Switch if Phase 3 local search exceeds ~60 s/box on 75-mon boxes or for 78-boss routing: first **Expert Iteration** (supervised pointer net on ≥1k–3k winning sixes), or **GRPO against a calibrated T1/T3 surrogate** verified by halving. |
| **Offline RL / PPO battler** | Rejected | **Fully holds** | Attempt #3 already showed PPO caps at ~14% vs SimpleHeuristics (`CLAUDE.md:82`). Exact sim + known AI script makes search + fitted leaf strictly stronger than model-free RL (Metamon RLC 2025 solves hidden-info ladder, not exact-script bosses). Only revisit if full-info sim is dropped. |

**Should anything learned come earlier?** Phase 4 (Texel + small Rust MLP) is already early — and moves *before* Phase 3 if Phase 2 shows evaluation-boundedness. Also, as soon as Phase 3 produces ~3k `(six, boss)` rows, a 30-min diagnostic GBDT/TabPFN fit gives the user an early ML milestone without blocking the advisor.

---

## 7. Architecture (components & data flows)

| Component | Kind | File | Inputs | Outputs | Status |
|---|---|---|---|---|---|
| Game & Legality DB | data/rules | `data/rr/*.json`, `scripts/build_rr.py`, `scripts/rr_probe.py` | ROM dump, official sheets, `gen9rr` Dex | boss teams, arenas, pools, legal movepools, items | exists (needs item/evo fixes P0) |
| Set Builder | rules + calc prefilter | `scripts/gauntlet.py`, `scripts/rr_team_probe.py` | box species, boss sets, arena, progress | 6–10 move shortlist + boss-aware sets + mutations | exists (`best4`); upgrade P0/P3 |
| Matchup Matrix | rules (`calc_bot`) | `ai_util/matchup.py`, `ai_util/calc_bot.py` | player sets, boss sets, arena | 6×6 cells (`hit`, `take`, `ko`, `faster`, tiers) | exists |
| Team Chooser | search (halving + QD + local search) | `scripts/rr_team_probe.py` (`--my-box`) | box/pool, matchup cells, optional filter | top teams + draft explanation + archive rows | partial (`cover`/`searched`); P3 |
| Team Filter (T1/T3) | fitted model (GBDT → Set Transformer) | `rl/team_filter.py` | candidate sixes + 6×6 cells + aggregates | `P(win)`, per-mon `P(death)` shortlist | P5 (conditional) |
| Battler | MCTS + exact RR AI + fitted leaf | `rl/engine_search.py`, `poke-engine/src/mcts.rs`, `rr_ai.rs`, `evaluate.rs` | Showdown `dump` → `pe.State` | chosen action + root visits/Q + `s2_pred` | exists; fixes P0, AI P1, eval P4 |
| Leaf Eval Fitter | fitted model (Texel → Rust MLP) | `rl/fit_eval.py` | `<out>.states.jsonl` (battle outcomes + 200k targets) | weights file read by `evaluate.rs` | P4 |
| Line Planner | expectimax + MCTS + DAG | `rl/line_planner.py` | team, boss, arena, Phase-1 battler + `rr_ai` | text/JSON flowchart, exact win/death %, risks | **P1.5 (new)** |
| Showdown Cross-Check | differential tester | `scripts/rr_crosscheck.py`, `rl/engine_diff.py`, `rl/rr_ai_diff.py` | flagged bad battles + `<out>.states.jsonl` | per-turn engine/AI diffs + pure-Showdown replay | **P1.5 (new)** |
| Dashboard & Cockpit | static UI / viewers | `ui/build.py`, `ui/index.html`, `rl/cockpit.py` | `.events.jsonl`, `trace_*`, `line_*`, `logs/rl/*` | Draft/Matrix/Trace/Line/Replay/Training views | exists; extend P0/P1.5/P3 |

```
[data/rr + --my-box] ──▶ [Set Builder (P0/P3)] ──▶ [Matchup Matrix]
                                                        │
                                                        ▼
   ┌────(P5: shortlist)──── [Team Filter (P5)] ◀── [Team Chooser (P3)]
   │                              ▲                     │ (candidate sixes)
   └──────────────────────────────┼─────────────────────▼
                                  │              [Battler (P0/P1/P4)] ◀──┐
                                  │               │      │               │
                        (archive) │   (states)    │      ▼               │
                                  └─── [Logs] ◀───┘  [Line Planner (P1.5)]
                                         │               │               │
                       ┌─────────────────┼───────────┐   │               │
                       ▼                 ▼           ▼   ▼               │
              [Leaf Fitter (P4)]  [Cross-Check]   [ui/ Dashboard]        │
                       └─────────────────────────────────────────────────┘
```

---

## 8. User RL decision points (per `CLAUDE.md`)

1. **End of Phase 0 — Nuzlocke utility objective (`U = win·(1 − λ·deaths/6)`):** choose `λ` (e.g. `0.5` vs `0.8` vs strict deathless) based on Phase 0 Brock/Misty win vs death-rate trade-offs. *(Concept: reward / utility shaping & terminal floors.)*
2. **End of Phase 1 — Opponent model in the tree:** choose minimax vs pure RR-AI vs per-node mixture `p ∈ {0.5, 0.8, 0.95}` using the exact-model oracle + on-policy agreement + SPRT results. *(Concept: best-response vs minimax & exploitation of model error.)*
3. **Phase 1.5 — Line risk posture:** choose branch pruning `ε` (`0.01` vs `0.002`) and mean-roll vs `--pessimistic` roll check after inspecting the first real Koga/Erika lines. *(Concept: expectimax / risk-sensitive planning.)*
4. **End of Phase 2 — Bottleneck branch:** choose Phase 3 (team/set search) vs Phase 4 (leaf eval) first, based on mirrors, MCTS 2k/20k/200k ladder, and reference human teams. *(Concept: separating search/evaluation error from action-space/team error.)*
5. **Phase 4 — Value function architecture & target:** choose outcome regression vs MCTS-200k distillation (TDLeaf-style), and linear vs 1-hidden-layer Rust MLP, based on held-out-boss log-loss + GBDT gap + play SPRT. *(Concept: Texel tuning, function approximation, off-policy generalization.)*
6. **Phase 5 — Learned team filter & builder:** choose stop-at-search vs T1 GBDT/TabPFN vs T3 Set Transformer (and whether to try ExIt/GRPO builder), based on noise ceiling, learning curve, and speed bottleneck. *(Concept: surrogate-guided search, set architectures, expert iteration.)*

---

## 9. Specific issues in Plan v2 & final gated phase list

**Found in code vs Plan v2:**
- **Stale user question (`rl/RR_PLAN.md:167`):** asks if Air Balloon is obtainable, but `data/rr/hardcore_rules.json:91` already confirms Air Balloon, Weakness Policy, Throat Spray, Light Clay, and Weather Rocks are unobtainable in hardcore. Meanwhile `data/rr/items.json` lacks Focus Sash, Assault Vest, Choice Specs, and resist berries (`scripts/build_rr.py:155` omitted them), and gates Choice Scarf/Band at #83/#85 (after Koga #80) and Life Orb at #91 (after Blaine #90) — audit `build_rr.py` item gating in Phase 0!
- **Broken stone evolutions (`scripts/rr_team_probe.py:108`, `scripts/gauntlet.py:294-297,318`):** `build_mon` passes `trainer_idx=None`, and `gauntlet.py:294` hardcodes BDSP's `STONE_GATE = "Eterna Gym"`. Fix RR stone-evolution gating in Phase 0.
- **Latent bug in `searched_cands` (`scripts/rr_team_probe.py:304`):** enforces "1 mega stone" by checking `any(mons[t].item ...)`. Any Phase 3 set builder that assigns non-mega items before selection will break it — check `is_mega_stone` instead.
- **Missing `force_switch` on alive pivots (`rl/convert.py:208` vs `rl/rr_ai_diff.py:54-102`):** `EngineSearcher` doesn't carry `dump[side]["forced"]` on U-turn/Volt Switch/Flip Turn into `pe.Side.force_switch`, while `rr_ai_diff.py` had to patch it manually. Move `_with_force_switch` into `convert.to_engine` in Phase 0!
- **Ordering:** build the **Line Planner + Showdown Cross-Check right after Phase 1** (user decision, advisor-first) so every later wall diagnosis can inspect exact lines and catch engine bugs immediately.

**Final phase list (short, gated):**
- **P0. Hygiene, engine/data fixes & logging (~1.5 d):** `-mega`, Tailwind ≥90 flat, spent Phoenix Down, `force_switch` in `convert.py`, RR stone-evo & `items.json` audit, won-terminal floor, `timeout` + per-turn state/visit logging, lead = best lead answer, wire `--chooser` in `box_job`, set hygiene (Rest/Choice/+Spe/Truant/dupes), `scripts/sprt.py`. *Gate:* logging invariance; 0 minimax timeouts; Brock/Misty ±3 (or explained by item/evo fix).
- **P1. Exact RR-AI in tree (~1 d):** re-run `ai`, exact-model oracle, on-policy agreement, per-node mixture `p ∈ {0.5, 0.8, 0.95}`, weighted random AI branches + switch diffs to ≥97% on-policy. *Gate:* SPRT ≥+8 pts vs minimax on Misty+Erika, deaths not worse, 0 timeouts.
- **P1.5. Line planner & Showdown cross-check (~1.5 d) [Advisor MVP]:** `rl/line_planner.py` (expectimax + MCTS + DAG merging + roll-sensitivity flags + UI view) + `scripts/rr_crosscheck.py` (auto-triage worst battles). *Gate:* line predicted clean-win % within ±5 pts of 100-battle Showdown replay on Brock/Misty/Erika; cross-check flags 0 unfixed engine bugs on top teams.
- **P2. Anchors & wall diagnosis (~½ d):** mirrors (n=48) vs AI-vs-AI, MCTS 2k/20k/200k ladder, user reference teams, line inspection of Koga/Blaine. *Gate:* decides P3-before-P4 vs P4-before-P3 vs fidelity fix; ρ ≥ 0.8 unlocks MCTS-2k early halving.
- **P3. Team & boss-aware set search + draft UI (~2 d):** sequential halving + 10-line QD archive + boss-aware 6–10 move prefilter + top-survivor move/item mutations + `--my-box` + draft explanation in `ui/`. *Gate:* Erika 30-box ≥+10 pts vs cover; Koga/Blaine ≥40% legal (or fidelity reading).
- **P4. Fitted leaf evaluation (~2 d):** `evaluate_features` + Texel linear weights (V0→V1 6×6×2 damage table) + terminal `U`; upgrade to 1×32 Rust MLP if GBDT beats linear >1% log-loss. *Gate:* held-out-boss log-loss −5%; SPRT +5 pts or −0.05 d/mon; mirrors not worse.
- **P5. Learned team filter (conditional, ~2 d):** T1 GBDT/TabPFN baseline; open T3 Set Transformer experiment at ≥10k rows / ≥30 bosses. *Gate:* true top-3 in top-24 ≥80% on held-out bosses; ≥90% of P3 win rate at ≤1/6 battles.
- **P6. Encounter routing (~1 d):** expected gain over future bosses via surviving scorer. *Gate:* routed boxes +10 pts clear rate vs random.
