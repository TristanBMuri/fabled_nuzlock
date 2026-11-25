# Round 2 Cross-Critique & Merged Proposal — Neon (`gemini-4-neon`)

## 1. What I'm Stealing from the Other Two (and Owning My Own Misses)

Reading **Fable** (`r1_fable.md`) and **Opus** (`r1_opus.md`) side-by-side exposed genuine blind spots in my Round 1 proposal (`r1_neon.md`). Here is what I am adopting outright:

1. **50 / 70 / 90 confidence buckets — not raw Brier decimals (from Fable, adapting Opus).**
   I dismissed numerical confidence in Round 1 as bookkeeping friction. Fable and Opus convinced me I was wrong: typing `B, 70%` adds three characters, directly trains the exact skill `CLAUDE.md:19` demands (*"trustworthy win/death probabilities"*), and maximizes the hypercorrection effect when a 90% call busts. I take **Fable's** 3-bucket tally (`50%: 4/9 · 70%: 7/11 · 90%: 5/6`) over **Opus's** Brier score because it fits on one O(1) header line forever and requires zero mental math to read.
2. **One-way `html_render` constraint & disposable widgets (from Fable & Opus).**
   Fable caught a real platform limit I glossed over: `html_render` cannot send user input back to the agent, so commitments *must* happen in chat first. I also adopt Fable/Opus's **disposable-first** widget rule (save a widget to `references/widgets/` only after 2–3 reuses) and drop my plan to fire `html_preview` + `html_render` in every 5-minute check-in.
3. **Evidence-only mastery (`P / B / D`) + hidden backlog count (from Fable) + explicit written-out `due` dates (from Opus).**
   My 4-status ladder (`unseen → intuited → practiced → mastered`) was fuzzier than Fable's `P / B / D` (predicted an unseen variation / built by hand / used in a real `D#`, owned at 2 kinds on 2 different days). Combining that with **Opus's** explicit `YYYY-MM-DD` due dates (eliminating LLM date-arithmetic drift) and **Fable's** anti-guilt rule (*never show "14 overdue cards" after a break*) is strictly better.
4. **Cockpit sabotage labs, `NOTEBOOK.md` as a held-out bet bank, and primary-source assignments (from Fable & Opus).**
   Reinjecting historical bugs (`git show 8240c1c:rl/lesson1_qlearning.py`, `done` bootstrap leak, `grad_clip=0.5`) into scratchpad copies so the user diagnoses from `rl/cockpit.py` curves is sharper than static code reading. And **Opus's** rule to assign exact primary-source sections (Sutton & Barto, GAE paper) for deep derivations instead of paraphrasing them closes a major nod-along loophole.
5. **Ambient uncertainty bands on every reveal (from Fable).**
   Every revealed outcome carries its noise band; any difference inside the band is logged as *"a lead, not a result"*.
6. **Owning two repo-accuracy errors in `r1_neon.md`:**
   - I cited `grad_norm = 0.5` (`r1_neon.md:88`) as the cockpit symptom of `grad_clip=0.5`. Checking `rl/cockpit.html:298` shows `grad_norm` is plotted **before clipping** (`"Gradient norm (before clipping)"`), so `grad_clip=0.5` does *not* flatten the plotted `grad_norm` curve to `0.5` — the symptom is flat win-rate / slow policy drift despite normal pre-clip norms!
   - I cited 43 features for Phase 4 (`r1_neon.md:183`), which holds for raw/`B1` (`NOTEBOOK.md:3042`), whereas `B2` expanded to **46 features** (`NOTEBOOK.md:3126`). I also tighten my 24-node syllabus down to **12 core nodes + 4 expansion cards**.

---

## 2. Sharpest Critiques & Pushback

### Pushback on Fable (`r1_fable.md`)
- **"Tutor bets too" is kayfabe unless strictly limited to genuinely unrun live runs.** Look at Fable's own sample check-in (`r1_fable.md:272`): *"My bet was 0.13. It is 0.95⁴⁰ = 0.129."* Calling a deterministic arithmetic fact a "bet" — or "betting" on `NOTEBOOK.md` experiments the tutor literally *just read* to author the question (`r1_fable.md:156`) — is fake theater. An analytical engineer will clock that instantly and stop trusting the scoreboard. Restrict tutor bets strictly to **unrun live commands** (`lesson1 --gamma 0.9`, queued cockpit runs) and drop the permanent `tutor:` calibration row.
- **Hook A (`CLAUDE.md` mandatory pre-recommendation bet on every `D#`) is intrusive friction.** `CLAUDE.md:34-37` currently gives options + basis + recommendation in a single turn. Splitting every real `D#` decision into two mandatory round-trips slows down shipping Phase 6 at 10 p.m. Make pre-recommendation calls **opt-in** (*"want to call it first?"* or via `/rl-tutor why D#`), not a mandatory speedbump on every work session.
- **Session 1's empty 4×8 table (`r1_fable.md:83`) is a cold pop quiz, not onboarding.** Handing a learner 32 blank cells across 8 repo files on Day 1 feels like an exam and punishes the exact gap the tutor exists to bridge. Pre-fill 4–5 landmark rows and leave **3–4 targeted blanks** for the user to place.
- **The 6-line tutor turn cap (`r1_fable.md:168`) is unworkable.** Fable breaks his own cap in §5 (`r1_fable.md:275-290`) by packing dense multi-sentence lines. **10–12 lines of prose** (excluding code blocks, tables, and widgets) is honest and readable.
- **"Katas accumulate into your own playable bot" (`r1_fable.md:159-162`) doesn't compose — and its cockpit claim is technically wrong.** A Q-table, GAE estimator, 4-arm UCB bandit, and 40-line MCTS are four *different* algorithm families, not four layers of one agent. Worse, Fable claims writing `{"kind": "eval", ...}` lines to `logs/rl/<run>.jsonl` makes a custom agent appear in `rl/cockpit.py` (`r1_fable.md:161-162`). Verified against `rl/cockpit.html:185,197` and `rl/cockpit.py:38,51`: **false** — `cockpit.html` runs `selected.map(parse).filter(r => r.start)`, silently dropping any file without a `kind: "start"` header (and `cockpit.py:38` reads line 1 for `config` and `pid`).
- **Statistics as "ambient only, not a module" (`r1_fable.md:116`) leaves core methods untaught.** Ambient error bands on every reveal are great, but they don't teach *how* `scripts/sprt.py` computes LLR, *why* Optimizer's Curse inflates best-of-K selection (`p=0.8` out of 4), or *why* 200-battle eval CI ≠ training-seed variance. Statistics must be **both** ambient *and* an explicit node.

### Pushback on Opus (`r1_opus.md`)
- **"Every widget opens locked" (`r1_opus.md:217,308-310`) violates the one-way iframe constraint.** Opus's sample session has the user type `45%` inside an `html_render` widget, then has the tutor react as if it saw the guess (`r1_opus.md:309-313`). **`html_render` cannot transmit user input back to the LLM.** Capturing guesses inside the iframe makes them invisible to the tutor, breaking calibration tracking and misconception logging.
- **Strict "all 14 nodes hit R1 before any node hits R3" (`r1_opus.md:109`) kills intrinsic pull.** If the user just landed Phase 4 and wants to derive fixed-effects logistic regression or rehearse D14 today, forcing ~6 sessions of R1 summaries across unrelated nodes first clashes with his own *"fresh from the lab"* priority.
- **Unbounded `## Bets` table (`r1_opus.md:266`) breaks ledger compactness.** Storing every bet row forever so Brier score can be recomputed on demand will blow `rl/tutor_ledger.md` past 150+ lines within a month.
- **Three concrete repo-accuracy mistakes:**
  1. `r1_opus.md:117` blames **"attempt #2's `gamma=0.95` on 40-turn episodes"** — wrong! `CLAUDE.md:86` puts the 40-turn shaping trap in **Attempt #3**; `CLAUDE.md:102` explicitly lists `gamma=0.95` (~25-turn battles) as working in **Attempt #2**.
  2. `r1_opus.md:131-134` claims Lesson 1 *"prints expectiminimax next to Q"* and proposes solving Lesson 1 exactly via memoized expectimax over `generate_instructions`. Checking `rl/lesson1_qlearning.py:116-117`: it prints only one move string (`get_safest_move()`) against a **worst-case** opponent (not random, no Q value), and exact memoized expectimax over unhashable Rust `State` objects (183×125 HP pairs × 16 damage rolls × crits/misses/Protect counters) explodes before depth 5 without abstraction.
  3. `r1_opus.md:235` hardcodes `distrobox-host-exec git` as *"git is host-only in this box."* In Antigravity's environment (where Neon runs), `/usr/bin/git` is native (`git rev-parse --short HEAD` → `a83a7a0`); hardcoding `distrobox-host-exec` breaks cross-provider portability.

---

## 3. Position on All 6 Fault Lines

### 1. Confidence Bets, Calibration Scoreboard & "Tutor Bets Too"
**Verdict:** Adopt **user answer + `50 / 70 / 90` confidence** tracked in a **1-line O(1) calibration tally** (`50%: a/b · 70%: c/d · 90%: e/f`); drop raw Brier scores and the permanent tutor calibration row. The tutor commits a public bet **only** before genuinely unrun live experiments (never on `NOTEBOOK.md` history or arithmetic).

### 2. Curriculum Entry, Ordering & Statistics Placement
**Verdict:**
- **Session 1:** Guided **partial-fill** 4-Question GPI Map (4 landmarks filled, 3 blanks for the user to place) + one 10-second `--gamma 0.9` bet. Beats both passive Day-1 diagrams (Neon) and intimidating 32-cell blank exams (Fable).
- **Ordering:** Soft **Tourist → Mechanic → Researcher** spiral with **no rigid R1-first lock**. Priority at every check-in: *fresh commit / user curiosity > 1 due recall card > next unseen node*.
- **Statistics:** **Both** ambient (every reveal includes its uncertainty band) **and** dedicated Node 12 (SPRT, paired CRN, Optimizer's Curse, seed-vs-eval variance). Keep my **4 "Why Not Here?" expansion cards** (SAC, DQN/Offline RL, MuZero/Dreamer, LLM RLHF/GRPO) interleaved every ~4th session.

### 3. Habit Hooks in `CLAUDE.md` & the Scheduled-Task Gate
**Verdict:**
- **`CLAUDE.md` hook (+2 lines, non-intrusive):** On `D#` decisions or completed phases, append a 1-line "Fresh Case File" to `rl/tutor_progress.md` and mention it in one line at wrap-up; offer *"want to call it before seeing my recommendation?"* only if the user opted in, never forcing a mandatory extra round-trip during shipping work.
- **Scheduled task:** Keep **OFF at launch** until Fable/Opus's hard gate triggers (**2 gaps ≥ 10 days** in `rl/tutor_progress.md`) *or* the user explicitly asks to enable it. When enabled, max 1 weekly 4-line mystery teaser in a single bound thread that **skips itself while `card: open`**. Allow a user-requested one-shot `schedule_task` only to settle an open bet on a multi-minute cockpit run.

### 4. `html_render` Discipline & The One-Way Iframe Constraint
**Verdict:** **Chat-first bet always** — never put graded inputs inside an iframe. `html_render` is strictly a **post-commit visualizer**, disposable by default (saved to `references/widgets/` only after 2 reuses). Use it as a staple in **Deep Labs** (≤1 per lab) and **sparingly in 5–10 min Check-ins** (only when a visual directly unlocks the surprise, avoiding unnecessary 2-tool-call latency).

### 5. Mastery Gating, Hand-Coding (`no_AI_code_here/`) & Tutor Conciseness
**Verdict:**
- **Mastery:** **Fable's `P / B / D` evidence bar** (`P` = predicted unseen twist + non-Pokémon transfer, `B` = built by hand & execution-verified, `D` = used in a real `D#` decision; **owned at 2 kinds on 2 different days**, never same-day). Explicit `YYYY-MM-DD` due dates (Opus), hidden overdue backlog count (Fable), primary-source pointers for derivations (Opus).
- **Conciseness:** **10–12 lines of prose max** per tutor turn (excluding code/tables/widgets), and never >4 lines before the user's first commitment.
- **`no_AI_code_here/rl_labs/` coding arc:** Keep it cleanly separate from the legacy `poke-env` files in `no_AI_code_here/players/`. Run standalone 10–15 line katas checked by execution against repo references (`lesson1:82`, `lesson2_ppo.advantages`, `fit_eval.fit_within_team`, `scripts/sprt.py`), culminating in **Opus's Lesson 3–4 capstones** importing `rl/lesson2_ppo.py`'s verified exports (`new_battle`, `step`, `observe`, `legal` at lines `90, 197, 207, 221`).

### 6. Concrete Implementation & Repo-Accuracy Traps
**Verdict:** Lock these five technical guardrails into `SKILL.md` from day one:
1. **Cross-env git:** Run `git` first; fall back to `distrobox-host-exec git` if `git` is unavailable (works in both Antigravity and Claude containers).
2. **Skill mirroring:** Test directory symlink `.claude/skills/rl-tutor -> ../../.agents/skills/rl-tutor` in both harnesses during setup; if either globber skips symlinked directories, fall back immediately to Fable's 5-line stub `SKILL.md`.
3. **Cockpit JSONL compatibility:** Any custom `rl_labs/` agent logging to `logs/rl/<run>.jsonl` MUST emit a valid `{"kind": "start", "pid": ..., "config": {...}, "t": ...}` line first (`rl/cockpit.html:185,197`), then `eval` lines (`step`, `t`, `a`, `vs`), then `{"kind": "end"}`.
4. **No exact expectimax on Lesson 1:** Use `rl/line.py` (exact policy-tree DP) for Node 1's DP anchor and Lesson 1's 200-state bucketed `key()` for small-MDP tabular intuition.
5. **Strict ≤75-line ledger cap forever:** O(1) calibration counters + 12-row Map + max 5 Fresh Queue items + max 8 active Misconceptions + max 12 rolling Recent Log lines.

---

## 4. Proposed Merged Architecture

### Minimal File Footprint (0 lines of new Python)
```text
.agents/skills/rl-tutor/
├── SKILL.md                  # ~120 lines: session flows, anti-illusion rules, compute & portability rules
└── references/
    └── curriculum.md         # ~140 lines: 4-Question GPI Map, 12 core nodes + 4 expansion cards, primary sources
.claude/skills/rl-tutor       # Symlink -> ../../.agents/skills/rl-tutor (5-line stub fallback if discovery fails)
rl/tutor_progress.md          # Single shared plain-Markdown ledger, bounded ≤75 lines forever
no_AI_code_here/rl_labs/      # User katas + Lesson 3–4 capstones (tutor runs & critiques, NEVER edits)
CLAUDE.md                     # +2 non-intrusive lines under "Working with the user on RL decisions"
```

Exact 2-line `CLAUDE.md` addition (non-intrusive):
> - When a `D#` decision is made or a phase finishes, append a 1-line item to `## Fresh Queue` in `rl/tutor_progress.md` and mention it in one line at wrap-up (never interrupt shipping turns unless the user asks to call the decision first).
> - When the user asks to learn RL, quiz themselves, or rehearse a `D#` decision, load the `rl-tutor` skill.

### Curriculum Backbone: 4-Question GPI Map + 12 Core Nodes + 4 Expansion Cards
Every method sits on one GPI loop answering four questions (*World Model · Representation · Credit/Target · Data/Opponent*) plus the ambient *5th Question: Noise Floor*:

| # | Core Node | Repo Anchor / War Story | Kata / Lab (`no_AI_code_here/rl_labs/`) | Primary Source (*recalled*) |
|---|---|---|---|---|
| 1 | MDP, Bellman & DP | `lesson1:82`, `rl/line.py` exact policy-tree backup | Tabular TD + SARSA variant vs `lesson1:82` | S&B ch. 3–4 |
| 2 | State Aliasing & POMDPs | `"brn"`/`"BURN"` (`git show 8240c1c`: 97.4% < 98.0%) | Bug hunt on scratch copy; break `key()` differently | S&B ch. 3; POMDPs |
| 3 | Horizon & Shaping | Attempt #2 (25t) vs #3 (40t) `gamma=0.95`; `POKE_ENGINE_DEATH` | Live `--gamma 0.9` bet; potential-based shaping term | Ng et al. 1999 |
| 4 | MC vs TD & GAE | `lesson2_ppo.py:366` (`lam`); `fit_eval.py` MC labels | 12-line GAE matched against `lesson2_ppo.advantages` | Schulman et al. 2016 §3 |
| 5 | Policy Gradients → PPO | `a2c` 2% vs `ppo` 74%; `grad_clip=0.5` (pre-clip norm!) | Clipped surrogate + cockpit sabotage diagnosis | S&B ch. 13.3; Spinning Up |
| 6 | Exploration & Scale | `ent=0.05`; Phase 4 root spread `0.189→0.096`, `rel_2x` | 4-arm UCB1 bandit on Garchomp's 4 moves | S&B ch. 2 |
| 7 | Generalization & Opponents | `greedy` 100% / 0–22% vs MCTS; `--holdout 0.2` | Leak-free trio holdout split + `--opp` ablation | Cobbe / PLR |
| 8 | Imitation, DAgger & ExIt | BC 40% (1h) vs PPO 14% (6M); `--algo bc` = DAgger | Lesson 3 ExIt capstone (importing `lesson2_ppo`) | Ross et al. 2011 |
| 9 | Search & Simultaneous Moves | 2k/20k/200k ladder; decoupled UCT; crit depth ≤1 | 4×4 turn payoff matrix solved via regret matching | Browne et al. 2012 |
| 10 | Opponent Models & Exploitation | Misty `−33` → `p=0.8` → port 98% `+30.2` | 8-line per-node mixture pseudocode + `p` ranking bet | Carmel & Markovitch |
| 11 | Value Fit, Confounding & Replay | Raw `−2.4`, 43→46 feats, FE `+5.9`, `R2-only` vs `R1+R2 +11.0` | Within-group fit on `f(s)−f(s₀)` vs `fit_within_team` | Texel tuning; DQN replay |
| 12 | Stats Rigor & Run Hierarchy | `sprt.py` (`R3` LLR −11.85), seed 4, D14 `V(box, bosses)` | 6-line SPRT LLR kata + D14 rehearsal + Lesson 4 | Henderson 2018; `rliable` |

**4 "Why Not Here?" Expansion Cards** (every ~4th session or on demand): *(a)* Continuous Control / SAC, *(b)* Deadly Triad / DQN & Offline RL (CQL/IQL, Metamon), *(c)* Learned World Models (MuZero/Dreamer), *(d)* LLM Post-Training (RLHF KL, GRPO).

### Session Flows & Hard Rules
- **Step 0 (silent):** Read `rl/tutor_progress.md`; run `git log <synced>..HEAD --oneline` (fallback `distrobox-host-exec git`) and `grep -n '^## ' NOTEBOOK.md | tail -n 10`.
- **Check-in (`/rl-tutor`, 5–10 min):** 1 cold due card (1.5m) → 1 chat-first bet (`answer + 50/70/90`) on fresh anomaly or ≤15 s single-core probe (tutor bets *only* if unrun) (3m) → reveal with uncertainty band + twist variation (3m) → user's 1-sentence takeaway + map pin + silent ledger update (1.5m). `html_render` optional (skip unless high visual payoff).
- **Deep Lab (`/rl-tutor lab [node]`, 25–35 min):** 1–2 cold due cards (3m) → symptom-first bug autopsy / cockpit sabotage OR primary-source derivation (10m) → 10–15 line kata in `no_AI_code_here/rl_labs/` checked by execution against repo code OR single-job `lesson2_ppo.py --minutes 3` cockpit run + post-commit `html_render` widget (15m) → non-Pokémon transfer gate + ledger update (4m).
- **Anti-Illusion & Compute Guardrails:** Zero cold lectures (>4 lines before user commits); ≤12 lines prose per tutor turn; never *"does that make sense?"*; no same-day ownership (`P/B/D`, 2 kinds on 2 different days); non-Pokémon transfer required for `P`; verbatim misconceptions re-tested in disguise until 2 clean passes; check-ins strictly single-core ≤60 s, labs max 1 `--minutes ≤ 3` job at a time (never run 30-min planner tables for teaching; check `MemAvailable`; leave ports 8000–8007 untouched; `uv run` only).

### Bounded Ledger Schema (`rl/tutor_progress.md`, always ≤75 lines)
```markdown
synced: a83a7a0 · last: 2026-11-25 · gaps_10d: 0 · card: none
calibration: 50% 4/9 · 70% 7/11 · 90% 5/6

## Fresh Queue (from git / CLAUDE.md hook; max 5)
- [ ] Phase 4 (a83a7a0): raw -2.4 -> FE + R1+R2 +11.0 (Node 11)
- [ ] Upcoming D14: per-mon death cost c_i vs flat λ (Node 12)

## Map (12 rows max · P=predict+transfer · B=built · D=real D# · owned = 2 kinds, 2 days)
| # | Node | Evidence | Due | Note |
|---|---|---|---|---|
| 2 | State Aliasing | P 11-22, B 11-24 | 2026-12-08 | owned; nailed "brn" + sensor transfer |
| 11 | Value Fit & FE | P 11-25 | 2026-11-28 | needs B kata or D#; recheck R2-only |

## Open Bets (optional, for multi-minute cockpit runs; max 3)
- 2026-11-25 `bet_ent005` (--minutes 3): you ">+5 pts vs MCTS", 70% · tutor "inside band", 60%

## Misconceptions (verbatim; retire after 2 disguised passes; max 8)
- 2026-11-25: "lower log-loss always means stronger MCTS leaf" [passes: 0/2]

## Recent Log (newest first, max 12 lines; fold older into Map)
- 2026-11-25 check-in (8m): Node 11 FE bet (70% miss -> nailed twist + ICU transfer)
```
