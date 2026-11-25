# Final Consensus Blueprint: `rl-tutor` for `nuzlock_rl`

Outcome of a **3-round structured discussion** across **9 fresh instances** of **Neon** (`gemini-4-neon`), **Fable** (`claude-fable-5-2`), and **Opus** (`claude-opus-5-5`). All three models gave conditional sign-off in Round 3, unanimously locking the architecture, curriculum, fun engine, and anti-rot guardrails below, and isolating **3 crisp user decisions (`D1–D3`)** before building.

---

## 1. Executive Summary

- **Delivery verdict (3/3 unanimous):** Build an **on-demand cross-provider Workspace Skill (`rl-tutor`)** — **0 lines of new Python**, zero LMS/database/streaks, zero scheduled lecture feed. Optional weekly `schedule_task` mystery teaser ships **OFF by default** (opt-in anytime; tutor suggests it once only after 2 gaps ≥ 10 days, self-silencing by reading its bound thread with zero file writes).
- **The core loop:** Not a textbook with Pokémon examples — **your repo *is* the course.** Every check-in runs in **3 user round trips (5–10 min)** built around **commit-before-reveal prediction bets** (with `50 / 70 / 90` confidence on the main bet), **forensic bug autopsies & cockpit sabotage**, **historical & upcoming `D#` rehearsals**, **10–15 line execution-checked katas** in `no_AI_code_here/rl/` culminating in **Lessons 3–4**, and **high-stakes non-Pokémon transfer checks**.
- **Big picture → fine detail:** One unifying **Generalized Policy Iteration (`evaluate ↔ improve`) Map** with **4+1 columns** (*World knowledge · Stored state · Target source · Data/opponent · Ambient noise floor*), grown 1 row per session starting *after* a 60-second live Session-1 bet. Covers **14 repo-anchored regions** (including 2 explicit Statistics case files + ambient bands everywhere) plus a **6-card "Why Not Here?" Expansion Shelf** (DQN, SAC, RND/Go-Explore, MuZero/Dreamer, Offline RL, LLM RLHF/GRPO).
- **Minimal footprint:** 2 authored skill files (`SKILL.md`, `references/map.md`), 1 symlink (`.claude/skills/rl-tutor` → `.agents/skills/rl-tutor`, fallback 5-line stub), and 1 bounded plain-Markdown ledger (`rl/LEARNING.md`, ≤80 lines / ≤700 words forever).

---

## 2. How the 3-Round Discussion Evolved

| Model | Best Ideas Adopted by the Panel | Major Concessions Made Across Rounds 1 → 3 |
|---|---|---|
| **Neon** (`gemini-4-neon`) | • High-stakes non-Pokémon transfer checks (e.g., ICU ventilator confounding) & delayed transfer gate<br>• **"Why Not Here?" Expansion Shelf** grounded in `RR_PLAN.md` rejections<br>• 3-round-trip check-in pace & thread-read self-silencing (zero headless file writes)<br>• User's own `no_AI_code_here/players/policy/policy_player.py` `calc_reward` as shaping anchor<br>• Round 3 catches: **Bet-First Session 1**, merging Map + Regions into 1 table to stay ≤80 lines, worktree-shared gitignored ledger, `AGENTS.md` parity for Antigravity | Dropped 24-node/7-arc structure, clickable mastery dashboard, `"intuited"` exposure status, and "no confidence score" stance; adopted Fable's `50/70/90` counters and `P/T/B/D` evidence; dropped passive `Fresh` append; corrected own R1 `grad_norm` pre-clip and 43→46 feature citations. |
| **Fable** (`claude-fable-5-2`) | • **"Betting table on your own project"** + O(1) `50/70/90` calibration counters + confident-miss theme tally<br>• 4+1 map columns & **evidence-only mastery (`P / T / B / D`)** (owned at 2 kinds on 2 days, one `T` or `B`)<br>• Hidden SRS backlog count, execution-checked katas, cockpit sabotage labs, `NOTEBOOK.md` as held-out bank<br>• One-way `html_render` constraint (chat-first bets; bake guess as ghost marker)<br>• Round 3 killers: **`next:` pre-written opener line** in `LEARNING.md`, **betting on Q-value (+0.98) not 98.6% win rate** in Session 1 (SE 0.26 pts catch!), word caps, cross-model log audit quotes | Conceded "tutor bets too" on archive/arithmetic was theatre (restricted strictly to unrun live runs); conceded Statistics needs 2 case files beyond ambient bands; shrunk Session 1 worksheet from 32 blanks to 1 row post-bet; demoted mandatory Hook A to auto-off user decision; dropped `Fresh` append; corrected `git` on PATH and cockpit `start`/`eval`/`end` JSONL requirement. |
| **Opus** (`claude-opus-5-5`) | • **Generalized Policy Iteration (`evaluate ↔ improve`)** caption & lived-era rows<br>• **Decision rehearsals (`D4, D10, D11, D14`)**, primary-source reading pointers (marked read/recalled), explicit `YYYY-MM-DD` due dates (`date -d`), Lessons 3–4 as `no_AI_code_here/rl/` capstones<br>• Rule: widgets plot real logged numbers or stamp `"toy numbers"`<br>• Round 3 killers: **pre-registering real multi-minute project runs in `NOTEBOOK.md`** as an unlimited live-bet stream, **~1/3 twists confirm the naive answer** (kills surprise meta-gaming), worked check-in transcript + literal ledger skeleton in `SKILL.md`, `git merge-base` rebase fallback | Withdrew Brier decimal & unbounded bet table, 14×4 rung grid, 6-session global R1-first lock, end-of-work quiz card, client-side locked widgets, 4th check-in turn, `html_preview` skip, local gate before `B`, and `card: open` headless write; corrected R1 Attempt #2 vs #3 `γ=0.95` attribution and Lesson 1 worst-case expectiminimax assumption. |

---

## 3. Final Converged Design (Panel-Locked on All Core Mechanics)

### 3.1 Delivery, File Footprint & Exact Trigger Frontmatter
```text
.agents/skills/rl-tutor/
├── SKILL.md                  # ≤ 115 lines / ≤ 1,100 words: protocol, worked 15-line example, skeleton, compute rules
└── references/
    └── map.md                # ≤ 135 lines / ≤ 1,800 words: GPI caption, 4+1 cols, 14 regions (symbol anchors), 6-card shelf
.claude/skills/rl-tutor       # Directory symlink -> ../../.agents/skills/rl-tutor (5-line stub SKILL.md if discovery fails)
rl/LEARNING.md                # ≤ 80 lines / ≤ 700 words forever: single shared learner ledger (Antigravity + Claude)
no_AI_code_here/rl/           # User's katas & Lessons 3–4 capstones (tutor checks by execution, NEVER edits)
CLAUDE.md (+ AGENTS.md?)      # +1 safety-net pointer line (+ optional hook per User Decision D1)
```

**Tightened YAML frontmatter (prevents false-triggering during normal Phase 6 debugging & ensures Antigravity description-only matching):**
```yaml
---
name: rl-tutor
description: >-
  RL tutor for nuzlock_rl: short check-ins, deep labs, D# rehearsals, and the learner's RL map,
  with a prediction bet before every reveal. Use ONLY when the user types /rl-tutor or rl-tutor,
  or explicitly asks to learn RL, be quizzed/tested on RL, run an RL lab, or settle RL bets.
  Do NOT trigger for ordinary repo work: debugging, running or explaining experiments, code edits,
  or routine D# decisions follow CLAUDE.md directly.
---
```
Invocation modes parsed by `SKILL.md`: `/rl-tutor` (default 5–10 min check-in), `/rl-tutor lab [region]` (25–35 min lab), `/rl-tutor why <D#|commit|file>` (rehearsal/autopsy), `/rl-tutor map` (map + calibration).

### 3.2 Curriculum Backbone: Big Picture → Fine Detail
- **Unifying Frame (GPI Caption + 4+1 Columns):** *Every RL method alternates evaluating a policy ↔ improving it*, placed along **(1)** World knowledge · **(2)** Stored object · **(3)** Target source · **(4)** Data / opponent · **(5, ambient)** How we know it worked.
- **Dynamic Pick Priority:** `(1)` explicit user ask > `(2)` pending `D#` (e.g., D14) > `(3)` fresh commit / `NOTEBOOK.md` result > `(4)` due spaced-recall / misconception > `(5)` next unseen region (breadth nudge: after 3 sessions in one region, default to an unseen one).
- **14 Repo-Anchored Regions** (`references/map.md` stores **symbols/grep strings, never brittle line numbers or stale numbers**; numbers are grepped fresh at reveal time and quoted with source + baseline):

| # | Region | Repo Anchor / War Story (Verified) | Execution-Checked Kata / Capstone (`no_AI_code_here/rl/`) | Primary Source (*recalled*) |
|---|---|---|---|---|
| 1 | **MDP, Value & Bellman / DP** | `lesson1_qlearning.py` (`train`), `rl/line.py` exact policy-tree backup | Tabular TD + SARSA variant vs `lesson1_qlearning` | S&B ch. 3–4 |
| 2 | **State Aliasing & Observability** | `git show 8240c1c:rl/lesson1_qlearning.py` (`"brn"` vs `"BURN"`); `lesson2_ppo` obs ablations | Scratchpad bug hunt; break `key()` differently | S&B ch. 3; POMDPs |
| 3 | **Horizon, Discount & Shaping** | Attempt #2 `γ=0.95` (25t worked) vs Attempt #3 `γ=0.95` (40t failed); `policy_player.py` `calc_reward`; `POKE_ENGINE_DEATH` 0/60/200 | Live `--gamma 0.9` bet; potential-based shaping term | Ng et al. 1999 |
| 4 | **MC vs TD & GAE** | `lesson2_ppo.py` (`advantages`/`lam`); `fit_eval.py` MC labels; `reinforce` vs `a2c` | 12-line GAE matched against `lesson2_ppo.advantages` (scratchpad `buf` incl. `done=None` cut) | Schulman et al. 2016 §3 |
| 5 | **Policy Gradients → A2C → PPO** | `a2c` 2% vs `ppo` 74%; `grad_clip=0.5` silent death (note `cockpit.html` plots *pre-clip* `grad_norm`) | Clipped surrogate + cockpit sabotage diagnosis | S&B ch. 13.3; Spinning Up |
| 6 | **Exploration & Value Scale** | `--ent 0.05`; Phase 4 root spread `0.189 → 0.096`, `rel_2x` | 4-arm UCB1 bandit on Garchomp's moves | S&B ch. 2 |
| 7 | **Generalization & Opponents** | `greedy` 100% / `0–22%` vs MCTS; `--pool` + `--holdout 0.2` | Leak-free trio holdout split + `--opp` ablation | Cobbe; PLR |
| 8 | **Imitation, DAgger & ExIt** | Attempt #3 BC 40% vs PPO 14%; `--algo bc` = DAgger; BC→PPO oscillation; Numbrion vs Jaxcalibur | **Lesson 3 capstone** (importing `lesson2_ppo` `new_battle`/`observe`/`legal`/`step`) | Ross et al. 2011 |
| 9 | **Search & Simultaneous Moves** | 2k/20k/200k ladder; decoupled UCT; crit branching depth ≤1 | 4×4 real-turn payoff matrix solved by regret matching | Browne et al. 2012 |
| 10 | **Opponent Models & Exploitation** | Buggy Rust AI Misty `−33` → `p=0.8` → fixed 98% `+30.2` | 8-line per-node mixture + `p` ranking bet | Carmel & Markovitch |
| 11 | **Value Fit, Confounding & Replay** | Phase 4: log-loss 0.44 vs play `−2.4`, 43→46 feats, FE `f(s)−f(s₀)` `+5.9`, `R2-only` `−5.2` vs `B2` (`+2.8` vs hand), `R1+R2` `+11.0 ± 4.1` seed 4 | Within-group fit on `f(s)−f(s₀)` vs `fit_eval.fit_within_team` | Texel tuning; DQN replay |
| 12 | **Run-Level Hierarchy** | D14 per-mon death cost `c_i` from `V(box, next bosses)`; `rr_route.py` | Hand `c_i` calculation + D14 rehearsal + **Lesson 4 AlphaZero-lite capstone** | Options / HRL |
| 13 | **Statistics I — Noise Floor** | SE `√(p(1-p)/n)`, 200-battle eval CI vs training-seed bands, paired CRN *(+ ambient bands on every reveal)* | Seed-variance live bet (`lesson1 --seed`) | Henderson 2018; `rliable` |
| 14 | **Statistics II — Selection & Testing** | Optimizer's Curse (best-of-K → untouched seed 4), `scripts/sprt.py` (`R3` LLR −11.85), held-out bosses | 6-line SPRT LLR kata vs `scripts/sprt.py` | Fishtest SPRT |

- **6-Card "Why Not Here?" Expansion Shelf** (grounded in `RR_PLAN.md` rejections; offered when ledger counter `foreign ≥ 3` or on demand): **Deadly Triad & DQN · Continuous Control / SAC · Hard Exploration (RND / Go-Explore) · Learned World Models (MuZero / Dreamer) · Offline RL (CQL / IQL, Metamon) · LLM Post-Training (RLHF KL, GRPO)**.

### 3.3 The Fun Engine, Check-In / Lab Flows & Anti-Illusion Rules
- **3-Round-Trip Check-In (5–10 min):**
  - **Step 0 (instant):** Read `rl/LEARNING.md` (check `wc -l > 80` → fold Log first; verify `git log` with `git merge-base --is-ancestor` fallback to `--since=<last_date>`, using `git` then fallback `distrobox-host-exec git`). Print the pre-written `next:` line (unless a fresh commit/D# overrides the bet).
  - **Turn 1 (≤70 words / ≤6 lines):** 1 cold recall (last session's queued `T` transfer question or oldest due item) + today's main bet with declared reveal type (`live` / `archive` / `derivable`). User replies with recall answer + bet + confidence (`5 / 7 / 9` for 50/70/90%).
  - **Turn 2 (≤130 words / ≤10 lines + optional ≤10-line table/code):** 1-line recall grade; tutor posts own bet *only if `live`* (never peeks, never bets on archive/arithmetic); run ≤60 s single-core command or quote archive source+baseline+band; present a free-form **twist** (different reveal type than bet; **~1 in 3 twists confirm the naive answer** to prevent meta-gaming).
  - **Turn 3:** Twist result + optional post-commit HTML widget (default 0 in check-ins, ≤1 in 3 check-ins, only for continuous sliders, preview-then-render per host rules, user's chat guess baked in as ghost marker, real numbers or stamped `"toy numbers"`); user writes ≤2-sentence takeaway. Tutor does **at most 1 sharpening pass**, pins 1 map row, writes next session's `next:` line, updates ledger with single-line edits, and stops.
- **Deep Lab (25–35 min):** 1–2 cold recalls → symptom-first bug autopsy / cockpit sabotage OR primary-source derivation → 10–15 line execution-checked kata in `no_AI_code_here/rl/` (or 1 single-job `lesson2_ppo.py --minutes ≤ 3` cockpit run, checking `MemAvailable`, ports 8000–8007 untouched) + 1–2 post-commit visualizers → non-Pokémon transfer or `D#` rehearsal → ledger update.
- **Evidence-Only Mastery (`P / T / B / D`):**
  - `P` predicted unseen variation with right reason · `T` delayed non-Pokémon transfer (auto-queued into `next:`) · `B` built by hand & execution-checked (or derived from primary source) · `D` made independent call first at a real `D#`.
  - **Owned = 2 different kinds on 2 different days, one of them `T` or `B`.** No same-day ownership; no `"intuited"` exposure grade; no hard local gate before `B`.
  - **Calibration scoreboard:** O(1) bucket counters (`50% / 70% / 90%`, printing `"too few"` under `n = 15`, plus 15-bet keep/drop gate) + confident misses by theme.
  - **Anti-rot safeguards:** Rotate bet wrappers (never same twice running); plateau breaker (after 5 check-ins or ≥3–4 `P`-only regions with zero `B`, offer a 10-min micro-kata or 3–5 line typed-in-chat mini-`B` twist); hidden SRS backlog count; single-line ledger edits (no full LLM rewrites); Log records model + ≤8 quoted words that earned each letter; `SKILL.md` ships literal ledger skeleton + 15-line worked check-in example.

### 3.4 Bounded Single-Table Ledger Schema (`rl/LEARNING.md`, ≤80 Lines / ≤700 Words)
Merging the user's map and regions into one 14-row table keeps the file comfortably under 80 lines forever:
```markdown
synced: a83a7a0 (2026-11-25) · last: 2026-11-25 · gaps_10d: 0 · foreign: 0 · no_b_streak: 0
next: recall "" · bet 3-horizon, live, "lesson1 --gamma 0.9: what is Swords Dance's opening Q (+0.98 at γ=1)?"
cal: you 50% 0/0 · 70% 0/0 · 90% 0/0 (too few <15) | house(live) 70% 0/0 | themes: none

## Map & Regions (14 rows max · P/T/B/D · latest date/letter · owned = 2 kinds, 2 days, ≥1 T|B)
| # | Region | User's 4-col Map (world/store/target/data) | Evidence | Due | Note |

## Open Bets (max 3; fold into cal on settle)
## Misconceptions (max 8, verbatim; retire after 2 disguised hits)
## Log (max 12, newest first: date · model · format · ≤8-word quote)
```
Spaced intervals: `2 → 4 → 8 → 16 → 32` days (miss resets to `1` day); explicit `YYYY-MM-DD` dates computed via `date -d`.

### 3.5 Exact Opening UX on Session #1 (Bet First, Guaranteed Above-Noise Reveal)
*(Shipped as the initial `next:` line; bets on opening Q-value `+0.98` rather than 98.6% win rate whose SE is only ~0.26 pts!)*
> **RL tutor, session 1.** One rule: you bet before I show anything. ~8 min; `stop` ends it.
>
> **Live bet (unrun — house bets too before running).** `rl/lesson1_qlearning.py` learns Garchomp vs a random Rotom-Wash with `γ = 1.0`. Swords Dance ends at **Q = +0.98** at the start (98.6% win rate over 2,000 battles).
> I'm about to rerun it with `--gamma 0.9` on the same seed. What will Swords Dance's starting **Q-value** be? Give a number, a 1-line *why*, and a confidence (`5 / 7 / 9` for 50 / 70 / 90%). I post my bet after yours, then run it.

*(Turn 2: house bet → live run → both scored → twist: did Swords Dance stay the best opener and did win rate leave ±0.5 pts? Turn 3: user's 1-sentence takeaway + fill lesson 1's single map row + reveal GPI caption + write Session 2's `next:` line.)*

---

## 4. Pre-Build Verification Checklist (~10 Min Before Shipping)
1. **Skill discovery & negative trigger test (both harnesses):** Test directory symlink `.claude/skills/rl-tutor -> ../../.agents/skills/rl-tutor` (fall back to 5-line stub if either skips symlinks); verify `/rl-tutor map` activates the skill AND normal debugging (*"why did R2-only lose to B2?"*) does **not**.
2. **Time default `lesson1_qlearning.py`:** Measure default `γ = 1.0` runtime (~10 s vs ~1 min) **without** running `--gamma 0.9` (preserving Session 1's honest live house bet!).
3. **Cockpit 3-record smoke test:** Confirm a minimal `start` + `eval` + `end` JSONL draws in `rl/cockpit.py`.
4. **Antigravity rules check:** Confirm whether Antigravity reads `CLAUDE.md` or requires `AGENTS.md` (relevant only if User Decision D1 enables a work-session hook).

---

## 5. Decisions for Tristan Before Building (`D1–D3`)

Following `CLAUDE.md`, the panel resolved everything mechanical and leaves **3 workflow choices** for you:

| ID | Decision | Options | Panel Basis & Recommendation |
|---|---|---|---|
| **D1** | **Work-session hooks** (outside `/rl-tutor`) | **A)** Strictly on-demand only (zero prompts during coding; `git log` + `NOTEBOOK.md` sync feeds `/rl-tutor`)<br>**B)** `D#` Hook A: at real `D#`s ask *"Your call first? (choice + 5/7/9, or `rec`)"*, auto-off after 3 skips *(Fable)*<br>**C)** Pre-register live bets while multi-minute runs are executing (and at `D#`s) into `NOTEBOOK.md` as `user bet: ...`, settled later by `/rl-tutor` via `grep` *(Opus's 2× idea)* | Passive `Fresh` append is unanimously dropped (redundant with `git log`). **C** gives an unlimited stream of real, zero-cost live bets without blocking shipping turns; **A** is cleanest zero-friction. If B or C is chosen, confirm mirroring those 2 lines into `AGENTS.md` for Antigravity parity. |
| **D2** | **Git tracking for `rl/LEARNING.md`** | **A) Untracked + worktree-shared** *(Panel rec — 3/3)*: add to `.gitignore`, resolve via `--git-common-dir` so all worktrees share one file with zero `git status` noise or `git stash` loss.<br>**B) Tracked in git**: versioned learning history; only `/rl-tutor` writes it (you commit separately). | **Recommend A**: Phase 6 worktrees and `git stash`/`reset` make a tracked ledger hazardous; trade-off is no git backup (protected from everything except `git clean -fdx`). |
| **D3** | **Weekly self-silencing teaser (`schedule_task`)** | **A) OFF at launch** *(Panel rec — 3/3)*: pure on-demand skill; tutor suggests turning it on *once* only if 2 gaps ≥ 10 days occur.<br>**B) ON from Day 1**: 1 weekly 2–4 line mystery teaser in one bound thread; stays silent if last teaser got no reply (zero file writes). | **Recommend A**: you already open T3 Code frequently on this repo, and `next:` gives instant pull when you invoke `/rl-tutor`. |
