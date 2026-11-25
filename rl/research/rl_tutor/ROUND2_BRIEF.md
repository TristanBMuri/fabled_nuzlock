# Round 2 Brief: Cross-Critique & Direct Debate

All three independent Round 1 proposals are written in `/home/tristan/CodingProjects/nuzlock_rl/rl/research/rl_tutor/`:
- `r1_neon.md` (Neon — `gemini-4-neon`)
- `r1_fable.md` (Fable — `claude-fable-5-2`)
- `r1_opus.md` (Opus — `claude-opus-5-5`)

Read **ALL THREE** Round 1 files completely before writing your Round 2 critique. Do NOT read any other `r2_*.md` file.

## Where Round 1 Already Converged (Don't Re-Litigate Unless You Disagree)
All three panelists independently converged on:
- **On-demand Workspace Skill** as the core engine (no recurring cron lecture feed, no XP/streaks/badges, no Python LMS/quiz engine, no CartPole-first track).
- **Cross-provider parity**: canonical skill in `.agents/skills/rl-tutor/` mirrored/symlinked to `.claude/skills/rl-tutor`, sharing one plain-Markdown learner state file inside `rl/` (0 lines of new Python).
- **Core pedagogy**: commit-before-reveal prediction bets grounded in real `nuzlock_rl` war stories (`"brn"` vs `"BURN"`, A2C 2% vs PPO 74%, Phase 1 RR-AI −33 → `p=0.8` +30, Phase 4 raw fit −2.4 → fixed-effects + `R1+R2` +11.0, Attempt #2 `grad_clip=0.5`, Attempt #3 `gamma=0.95`), verbatim misconception logging, and non-Pokémon transfer checks.

## The 6 Real Fault Lines to Debate Sharply
Engage directly with Neon, Fable, and Opus by name on each of these tensions:

1. **Confidence Bets, Calibration Scoreboard & "Tutor Bets Too" (Fun vs. Friction)**
   - **Opus** logs every bet with a confidence and keeps a **Brier score / calibration curve** as the sole scoreboard.
   - **Fable** goes further: **both user AND tutor commit bets (50/70/90) before every reveal**, keeping a calibration row for each so the tutor loses in public.
   - **Neon** uses simple prediction bets (A/B/C + 1-sentence why) with **no numerical confidence or calibration score**.
   - *Debate*: Is calibration scoring genuinely fun & educational, or tedious bookkeeping in a 5-min check-in? And if the tutor already read `NOTEBOOK.md` / knows the outcome, isn't "tutor bets too" fake unless strictly restricted to genuinely unrun live experiments?

2. **Curriculum Entry, Ordering & Statistics Placement**
   - **Opus**: Generalized Policy Iteration (GPI) + 3 questions map; strict rule that **all 14 nodes hit R1 (Story) first (~6 sessions) before any node hits R3 (Derive)**; Statistics is Node 12; Lessons 3–4 are the R4 Build capstones.
   - **Fable**: Session 1 hands the user an **empty 4-column table + repo file list** for *them* to fill; 3 passes (**Tourist → Mechanic → Researcher**) across 12 case files where **fresh commits always preempt order**; **Statistics is NOT a module**, it's the ambient scoring rule of every bet; every 4th session is "foreign territory".
   - **Neon**: Session 1 renders an interactive `html_render` **4-Question Grand Map**; **6-Arc Zoom-In/Zoom-Out Spiral**; Statistics is Arc 6; adds a **"Why Not Here?" Expansion Shelf** (SAC, RND/Go-Explore, MuZero/Dreamer, IQL/CQL/DPO).
   - *Debate*: Best Session 1 UX (guided map vs empty-table fill)? Should breadth strictly precede depth, or should fresh commits / curiosity jump straight to deep math/labs? Should Statistics be its own node/arc, ambient-only, or both?

3. **Habit Hooks in `CLAUDE.md` & the Scheduled-Task Gate**
   - **Fable (Hook A)**: modify `CLAUDE.md` so at every real `D#` decision during project work, the user states their call + confidence *before* seeing the agent's recommendation (plus optional one-shot `schedule_task` to settle long cockpit runs).
   - **Opus**: `CLAUDE.md` logs explained RL concepts to *met in the wild* and offers **one due card at the end of regular work sessions**.
   - **Neon**: `CLAUDE.md` only appends a 1-line "Fresh Case File" to the ledger on `D#`/phase completion (doesn't interrupt work sessions).
   - **Scheduled task**: Fable & Opus keep it off until a hard gate (**2 gaps ≥ 10 days**), then offer max 1 self-skipping weekly bet in a bound thread. Neon makes the self-silencing teaser strictly opt-in from day 1.
   - *Debate*: Do intrusive work-session hooks (Fable Hook A / Opus end-of-work card) build a real habit or become annoying friction when trying to ship Phase 6? Should the weekly teaser require a 10-day-gap gate or just be a clean user choice?

4. **`html_render` Discipline & The One-Way Iframe Constraint**
   - **Fable** flagged that `html_render` widgets **cannot send user input back to the agent**, so bets must happen in chat first and widgets are disposable post-commit visualizers (saved only after 3 regenerations).
   - **Opus** proposed *"every widget opens locked: the user types a guess before the sliders come alive"* (saved after 2 asks).
   - **Neon** leans heavily on 5 bespoke interactive `html_render` math sandboxes + a clickable mastery map.
   - *Debate*: How do we get maximum visual fun from `html_render` without slow 2-tool-call overhead in 5-min check-ins, and without hiding user guesses inside a client-side iframe the tutor can't see or grade?

5. **Mastery Gating, Hand-Coding (`no_AI_code_here/`), and Tutor Conciseness Rules**
   - **Fable**: Evidence-only mastery (`P` / `B` / `D`, owned at 2 kinds on 2 different days), strict **6-line tutor turn cap**, never show SRS backlog count, katas accumulate in `no_AI_code_here/rl/` into the user's own playable mini-bot checked by execution against repo references.
   - **Opus**: 4 rungs (`Story / Predict / Derive / Build`) × 14 nodes, **no rung passes same-day**, strict **12-line prose cap**, explicit **primary-source reading assignments** (S&B / GAE paper) for R3 instead of paraphrasing, Lessons 3–4 as R4 capstones, explicit written-out `due` dates to avoid LLM date arithmetic errors.
   - **Neon**: 4 statuses (`unseen → intuited → practiced → mastered`, delayed transfer gate), 2-sentence Feynman close, standalone 10-line katas in `no_AI_code_here/rl_katas/`.
   - *Debate*: Which mastery rubric is actually usable across 30 sessions without feeling like bureaucratic grading? Is 6 lines too terse or 12 lines too long? What's the best `no_AI_code_here/` coding arc?

6. **Concrete Implementation & Repo-Accuracy Traps**
   - Check any factual claims or assumptions across the three Round 1 docs if skeptical (e.g., `git` vs `distrobox-host-exec git` across Antigravity vs Claude environments; symlink vs stub for `.claude/skills/rl-tutor`; whether `rl/cockpit.py` can plot a hand-written agent's JSONL; whether exact expectimax on Lesson 1 is feasible; keeping the ledger ≤80 lines forever; what's already inside `no_AI_code_here/`).

## Required Structure for Your Round 2 Response (Target ~150–250 lines)
Write your response to your assigned file (`r2_neon.md`, `r2_fable.md`, or `r2_opus.md`) with these 4 sections:
1. **What I'm Stealing from the Other Two** — concrete concessions citing Neon, Fable, and Opus by name.
2. **Sharpest Critiques & Pushback** — where the other two are over-built, gimmicky, boring, pedagogically flawed, or technically mistaken (cite file:line if verifying repo facts).
3. **Position on All 6 Fault Lines** — crisp, argued verdict on Fault Lines 1–6.
4. **Proposed Merged Architecture** — your concrete best-of-all-worlds blueprint (file layout, curriculum backbone, session flows, fun/anti-illusion rules, ledger schema).

## Rules
- **Read-only on the repo** except writing your ONE assigned output file (`r2_<model>.md`) in `/home/tristan/CodingProjects/nuzlock_rl/rl/research/rl_tutor/`.
- Do NOT read any other `r2_*.md` file. Do NOT run training, battles, pytest, or heavy compute. Do NOT touch any other file or git state.
- End your final message with a crisp ≤10-line summary of your Round 2 positions and key concessions.
