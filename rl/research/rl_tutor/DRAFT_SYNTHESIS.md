# Draft Consensus Blueprint — `rl-tutor` (Post-Round 2 Synthesis)

Merged from Round 1 & Round 2 of **Neon** (`gemini-4-neon`), **Fable** (`claude-fable-5-2`), and **Opus** (`claude-opus-5-5`). Everything in Sections 1–5 reflects 3/3 convergence (with repo facts verified); Section 6 isolates the 5 genuine remaining tradeoffs for Round 3 votes and stress-testing.

---

## 1. Delivery, Cross-Provider Footprint & Schedule (3/3 Converged)

- **Core delivery:** An **on-demand Workspace Skill** (`rl-tutor`) — zero Python LMS/quiz code, zero scheduled lecture feed, zero XP/streaks/badges.
- **Minimal file footprint (0 lines of new Python):**
  ```text
  .agents/skills/rl-tutor/SKILL.md            ≤ 115 lines: protocol, anti-illusion rules, compute & tool rules
  .agents/skills/rl-tutor/references/map.md   ≤ 135 lines: GPI map, 14 regions (case files), "Why Not Here?" shelf
  .claude/skills/rl-tutor -> ../../.agents/skills/rl-tutor   (directory symlink; 5-line stub SKILL.md if either harness skips symlinks)
  rl/LEARNING.md                              ≤ 80 lines forever: single shared hand-editable learner state (Antigravity + Claude)
  no_AI_code_here/rl/                         user's own katas & growing bot; tutor NEVER edits here
  CLAUDE.md                                   +2–3 lines (exact wording in Open Tradeoff 1)
  ```
- **Invocation:** `/rl-tutor` (default 5–10 min check-in), `/rl-tutor lab [region]` (25–35 min deep lab), `/rl-tutor why <D#|commit|file>` (decision rehearsal / autopsy), `/rl-tutor map` (show map + calibration).
- **Draft YAML frontmatter (for progressive disclosure in Antigravity + Claude Code):**
  ```yaml
  ---
  name: rl-tutor
  description: >-
    Interactive RL tutor grounded in nuzlock_rl. Use when the user asks to learn RL,
    run an RL check-in or lab, quiz themselves, rehearse a D# decision, explore the RL
    map, or understand why an experiment, bug, or algorithm in this repo behaved as it did.
  ---
  ```
- **Scheduled task (`schedule_task`):**
  - **OFF at launch.** The user may ask to turn it on at any time (no gate blocks the user).
  - The tutor may **suggest it once** only if `rl/LEARNING.md` shows **2 gaps ≥ 10 days**.
  - When enabled: **at most 1 weekly mystery teaser / cold question** (2–4 lines, never a lecture) posted to **one bound thread**, self-silencing if the last teaser got no reply (mechanism in Open Tradeoff 5).
  - Also allowed on explicit user request: a **one-shot** `schedule_task` to settle an open bet on a multi-minute cockpit run.

---

## 2. Curriculum Architecture: Big Picture → Fine Detail (3/3 Converged)

- **The Unifying Map (1 caption + 4+1 columns):**
  - **Caption (Generalized Policy Iteration):** *Every RL method alternates evaluating a policy ↔ improving it.*
  - **Columns:** (1) *What do you know about the world?* · (2) *What do you store?* · (3) *Where do targets come from?* · (4) *Whose data, which opponent?* · **(5, ambient on every row)** *How do you know it worked?*
- **Session 1 Entry (Active Partial-Fill, Not Lecture, Not 32-Cell Exam):**
  - Show the user 5–6 eras they actually lived through (`Attempt #3`, `lesson1_qlearning`, `lesson2 --algo ppo`, `lesson2 --algo bc`, `engine_search` / Phase 1, `fit_eval` / Phase 4, `line.py`). Pre-fill 2–3 rows as examples and have the user fill **3–4 rows** in their own words (~5 min), correct wrong cells, reveal the GPI caption, then run one fast live bet (e.g. `lesson1_qlearning.py --gamma 0.9`, never pre-run!). The user's filled rows live in `rl/LEARNING.md` and grow as new rows are encountered.
- **Ordering (Dynamic, No Rigid Global Lock):**
  - Pick priority: **(1)** explicit user request > **(2)** pending real `D#` (e.g., D14) > **(3)** fresh commit / `NOTEBOOK.md` result > **(4)** due spaced-recall / active misconception > **(5)** next unseen region (breadth tie-breaker: after 3 sessions in one region, default to an unseen one).
- **14 Regions (Case Files) in `references/map.md`:**
  - *Symptom first → repo anchor by symbol/grep string (never brittle line numbers!) → bet (with reveal type) → twist → kata/derivation + primary source (marked read/recalled) → high-stakes non-Pokémon transfer question (ICU ventilator level).*
  1. **MDP, Value & Bellman / DP** (`lesson1_qlearning.py:train`, `rl/line.py` exact policy-tree backup) — *no unverified exact expectimax on Lesson 1*.
  2. **State Aliasing & Observability** (`git show 8240c1c:rl/lesson1_qlearning.py` `"brn"` vs `"BURN"`; `lesson2_ppo` obs ablations; POMDPs).
  3. **Horizon, Discounting & Reward Shaping** (Attempt #2 `γ=0.95` on 25t worked vs Attempt #3 `γ=0.95` on 40t failed; `no_AI_code_here/players/policy/policy_player.py` `calc_reward`; `POKE_ENGINE_DEATH` 0/60/200; Ng et al. 1999).
  4. **MC vs TD & GAE (Bias vs Variance)** (`lesson2_ppo.py` `advantages`/`lam`; `fit_eval.py` MC labels; `reinforce` vs `a2c`).
  5. **Policy Gradients → A2C → PPO** (`a2c` 2% vs `ppo` 74%; `grad_clip=0.5` silent death — note `cockpit.html` plots *pre-clip* `grad_norm`; clipped surrogate; cockpit sabotage labs).
  6. **Exploration & Value Scale** (`--ent 0.05`; Phase 4 root spread `0.189 → 0.096`, `rel_2x`; UCB1 / pUCT / bandits).
  7. **Generalization & Opponent Diversity** (`greedy` 100% vs `0–22%` MCTS; `--pool` + `--holdout 0.2`; self-play cycles).
  8. **Imitation, DAgger & Expert Iteration** (Attempt #3 BC 40% vs PPO 14%; `--algo bc` = DAgger; BC→PPO oscillation; Numbrion vs Jaxcalibur; Lesson 3 capstone).
  9. **Search & Simultaneous Moves** (2k/20k/200k ladder; decoupled UCT; crit branching depth ≤1; 4×4 matrix game / regret matching / Nash vs BR).
  10. **Opponent Models & Model Exploitation** (buggy Rust AI Misty `−33` → `p=0.8` → fixed 98% `+30.2`; restricted Nash / model-based exploitation).
  11. **Value Fitting, Confounding & Replay** (Phase 4: log-loss 0.44 vs play `−2.4`, 43→46 features, fixed effects `f(s)−f(s₀)` `+5.9`, `R2-only` `−5.2` vs `B2` / `+2.8` vs hand, `R1+R2` `+11.0 ± 4.1` on seed 4).
  12. **Run-Level Hierarchy** (D14 per-mon death cost `c_i` from `V(box, next bosses)`; `rr_route.py`; options / hierarchical RL; Lesson 4 capstone).
  13. **Statistics I — Noise Floor** (early case file: SE `√(p(1-p)/n)`, 200-battle eval CI vs training-seed bands, paired Common Random Numbers) + *ambient error bands on every reveal*.
  14. **Statistics II — Selection & Testing** (late case file: Optimizer's Curse / best-of-K then untouched fresh seed 4, `scripts/sprt.py`, held-out bosses).
- **"Why Not Here?" Expansion Shelf (General RL Beyond `nuzlock_rl`):**
  - 6 concise cards grounded in `RR_PLAN.md`'s rejection table: **Deadly Triad & DQN · Continuous Control / SAC · Hard Exploration (RND / Go-Explore) · Learned World Models (MuZero / Dreamer) · Offline RL (CQL / IQL, Metamon) · LLM Post-Training (RLHF KL, GRPO)**.
  - Offered via a ledger counter (`foreign ≥ 3` sessions since last shelf card) or on demand — never a rigid schedule.

---

## 3. Fun Engine, Bets, Widgets & Anti-Illusion Guardrails (3/3 Converged)

- **Main Bet + Calibration (Zero Theatre):**
  - One main bet per session with one-digit confidence (`50 / 70 / 90` or `5 / 7 / 9`). Cold recalls and twists take plain answers.
  - Scoreboard = O(1) in-place bucket counters (`50%: a/b · 70%: c/d · 90%: e/f`) + **confident misses by theme** (e.g., `"more information helps": 0/3`). No Brier decimal; no unbounded bet rows. Gate: after 15 bets, ask once if the user wants to keep calibration tracking.
  - **Reveal type declared upfront:** `live` (unrun command/flags/seed), `archive` (`NOTEBOOK.md` / `results/`), or `derivable` (math/code). **The tutor ("house") bets ONLY on `live` runs** — never on archive/arithmetic, and never peeks first.
- **Anti-Illusion Core (Hard Rules in `SKILL.md`):**
  1. Never reveal a result or explain a mechanism/fix before the user commits/attempts it.
  2. Multiple-choice options (A/B/C/D) are allowed *only* for direction; mechanisms, fixes, and twists are always free-form.
  3. Never ask *"does that make sense?"* — always follow a reveal with a **twist** (right bet + wrong twist = logged as illusion, not pass).
  4. User writes a ≤2-sentence takeaway in their own words; tutor sharpens until exact.
  5. Verbatim misconceptions return disguised in new contexts until 2 clean passes.
  6. Never show the size of an overdue SRS backlog after a break (serve 1 card silently).
  7. Every archive number quotes `source` **and its baseline**; every external claim is tagged *read* or *recalled*.
  8. A session with no committed prediction and nothing user-made is NOT logged as progress.
- **Widgets (`html_render`) Discipline:**
  - `html_render` is one-way (no return channel): **all guesses are typed in chat first**, never locked inside an iframe.
  - Widgets appear **after commitment**, with the user's chat guess **baked in as a ghost marker** beside real logged numbers (or visibly stamped `"toy numbers"`).
  - Budget: **0–1 widget per check-in** (only for a continuous knob worth dragging: `γ`, `λ`, UCB `c`, value scale, `p`), 1–2 per lab. Default visual is a Markdown table. Saved to `references/widgets/` only after the **user** asks for the same one twice (max 3).
- **Hand-Coding Arc (`no_AI_code_here/rl/`):**
  - Cleanly isolated from legacy `no_AI_code_here/players/`.
  - 10–15 line session katas (`td.py`, `gae.py`, `bandit.py`, `pg.py`, `fit_within.py`, `sprt.py`) checked strictly by **execution** against repo references via a scratchpad harness (tutor reports pass/fail + first mismatch, **never edits `no_AI_code_here/`**), building toward **Lessons 3–4 capstones** importing `rl/lesson2_ppo.py`'s verified helpers (`new_battle`, `observe`, `legal`, `step`).
  - Cockpit compatibility note: any custom agent logging to `logs/rl/<run>.jsonl` must emit `start` (`t`, `config: {}`), `eval` (`step`, `t`, `vs: {opp: {win, n}}`), and `end` records (`cockpit.html:185,197`).
- **Compute Guardrails:**
  - Check-in runs: single-core, ≤60 s (`uv run` only).
  - Lab runs: `lesson2_ppo.py --minutes ≤ 3` (minimal parallelism, one job at a time, check `MemAvailable`, ports 8000–8007 untouched) or queued in cockpit as an open bet.
  - Never run 30-min planner tables for teaching; reveal from `results/`.

---

## 4. State, Sync & Spaced Repetition (`rl/LEARNING.md`, Hard-Capped ≤80 Lines)

- **Cross-env git sync:** Run `git log <synced>..HEAD --oneline` (fallback `distrobox-host-exec git` if `git` is missing) and `grep -n '^## ' NOTEBOOK.md | tail -n 10`. Read ONLY cited sections of `NOTEBOOK.md` (never the whole 212 KB file).
- **Spaced repetition with zero code:** Intervals `2 → 4 → 8 → 16 → 32` days (miss resets to `1` day + logs verbatim misconception). Due dates computed via `date -d '<YYYY-MM-DD> +<N> days' +%F` and written out in full `YYYY-MM-DD` (zero LLM date math).
- **Strict line budget (≤80 lines forever):** Header + O(1) calibration counters (3 lines) · User's Map (≤10 lines) · Regions (≤14 rows) · Fresh Queue (≤5) · Open Bets (≤3) · Misconceptions (≤8) · Rolling Recent Log (≤12; older lines folded into Regions). If the file exceeds 80 lines, folding/pruning is the tutor's first action.

---

## 5. Pre-Build Verification Checklist (10 min before shipping the skill)
1. Test whether BOTH Antigravity and Claude Code discover `.claude/skills/rl-tutor` as a directory symlink (else use a 5-line stub `SKILL.md`).
2. Time `uv run python rl/lesson1_qlearning.py` once on this machine (~10 s per `CLAUDE.md` vs ~1 min per docstring) without running `--gamma 0.9` (save `--gamma 0.9` as the first honest live house bet!).
3. Verify a minimal 3-record JSONL (`start`/`eval`/`end`) renders in `rl/cockpit.py`.

---

## 6. Five Open Tradeoffs for Round 3 Votes & Stress-Testing

1. **Tradeoff 1 — `CLAUDE.md` Work-Session Hooks:**
   - **Option A (Opus + Fable):** Keep **Hook A** at real `D#` decisions (*"Your call first? choice + 5/7/9, or `rec`/`pass`"* before showing recommendation, with sticky `"rec all today"` escape).
   - **Option B (Neon):** Strictly **passive-only** in `CLAUDE.md` (append 1 line to `Fresh` queue; never add a pre-recommendation round-trip unless user opts in).
   - *Sub-question:* Keep or drop the passive 1-line `Fresh` append (Neon/Fable keep; Opus says `git log` sync makes it redundant)?

2. **Tradeoff 2 — Check-In Turn Cadence & Line Caps:**
   - **Option A (Fable + Neon):** **3 user round trips** (Turn 1 combines 1 cold recall + today's bet; Turn 2 reveal + twist; Turn 3 takeaway + close). Line cap: **4–6 lines** on question turns, **8–10 lines** on reveal turns (excluding code/tables/widgets).
   - **Option B (Opus):** **4 user round trips** (Turn 1 asks ONLY cold recall from `LEARNING.md` before running git sync so sync hides behind user thinking; Turn 2 bet; Turn 3 reveal + twist; Turn 4 takeaway). Line cap: **6 lines** pre-action, **10 lines** reveal.

3. **Tradeoff 3 — `html_preview` Before `html_render` in Check-Ins:**
   - **Option A (T3 Standard / Safe):** Always call `html_preview` then `html_render` (as T3 Code instructions require to get accurate `contentHeight` and catch broken layouts), and keep check-in latency low by rendering a widget only when a continuous slider/plot truly adds value (~1 in 2 check-ins).
   - **Option B (Opus Fast Path):** Skip `html_preview` in check-ins and pass a fixed `height` directly to `html_render` (1 tool call instead of 2).

4. **Tradeoff 4 — Mastery Evidence Schema & Local Prerequisite:**
   - **Option A (Fable):** **4 letters `P / T / B / D`** (`T` = non-Pokémon transfer, queued automatically as next session's opening recall); owned = 2 kinds on 2 days with at least one of `T` or `B`. No local gate before katas/derivations (just a 2-line map pin).
   - **Option B (Opus + Neon):** **3 letters `P / B / D`** (where `P` already requires twist/transfer, and `B` absorbs derivations); owned = 2 kinds on 2 days. Plus Opus's optional local gate: no kata/derivation in a region until its map row is filled and 1 `P` exists.

5. **Tradeoff 5 — Scheduled Teaser Self-Silencing & Git-Dirty Risk:**
   - **Option A (Neon + Fable R2):** Read the bound thread before posting; if the last teaser has no user reply, output nothing. **Zero writes to `rl/LEARNING.md`** from headless scheduled runs (prevents dirtying `git status` while the user works on Phase 6).
   - **Option B (Opus):** Set/clear a `card: open` flag in `rl/LEARNING.md` (works cleanly across providers without reading thread history, at the cost of a background file edit).
