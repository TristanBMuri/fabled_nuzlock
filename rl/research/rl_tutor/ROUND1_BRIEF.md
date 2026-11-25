# Round 1 Brief: Designing the Best Possible RL Tutor for `nuzlock_rl`

You are one of three independent panelists in a 3-round design discussion: **Neon** (`gemini-4-neon`), **Fable** (`claude-fable-5-2`), and **Opus** (`claude-opus-5-5`).
Write your Round 1 proposal **completely independently** — do NOT read any other `r1_*.md` file in this directory. In Round 2 you will critique each other's proposals directly.

## The User's Request
> How can we best teach the user reinforcement learning in the context of this project (`nuzlock_rl`), in the **most fun way**, and one that **actually teaches** things from **big picture to fine detail**? Goal is to make this a **scheduled task** the user can check in to learn stuff, or — likely better — an **on-demand skill** they can call whenever they want (or a hybrid if justified).

## Inspect the Repo First (Read-Only)
Ground every concrete proposal in the REAL project — do not give generic textbook advice. Skim what you need from:
- `CLAUDE.md` (governing **minimal-code** rule, "Working with the user on RL decisions", Attempt #2 & #3 hard-won lessons)
- `rl/README.md` (RL learning track: Lesson 1 tabular Q-learning, Lesson 2 PPO/A2C/REINFORCE/BC + live cockpit, Rust `poke-engine` at ~13 µs/turn, Lessons 3–4 roadmap, 2026 Showdown bot wave)
- `rl/RR_PLAN.md` (Radical Red advisor plan v3: Phases 0–4 completed, Phase 4 Texel-tuned fixed-effects leaf evaluation story, upcoming Phase 6)
- `rl/lesson1_qlearning.py`, `rl/lesson2_ppo.py`, `rl/cockpit.py`, `rl/fit_eval.py`, `rl/line.py`
- `no_AI_code_here/` (the user's own hand-written code area)
- `ui/` (the run dashboard)

## Real Environment & Capabilities Available in T3 Code
Design for the actual tools the user has in T3 Code (not a generic chat box):
1. **Workspace Skills (On-Demand)**:
   - Antigravity discovers `.agents/skills/<name>/SKILL.md` (with optional `references/`, `scripts/`, `resources/`; progressive disclosure via YAML frontmatter `name` + `description`).
   - Claude Code discovers `.claude/skills/<name>/SKILL.md` (or `.claude/commands/`).
   - The user uses BOTH Antigravity (Neon) and Claude (Fable/Opus) in T3 Code — consider cross-provider skill portability.
2. **Scheduled Tasks (`schedule_task`)**:
   - Can run on an interval, fixed time/weekdays, or one-shot with no turn active, posting to a bound thread or creating a fresh thread per run.
3. **Inline Interactive HTML (`html_preview` + `html_render`)**:
   - Can render self-contained interactive HTML/CSS/JS widgets (charts, interactive quizzes, Bellman/MCTS/value visualizers, parameter sliders, battle-state cards) inline inside the chat thread (728px desktop, themed via CSS vars, no server needed).
4. **Fast Local Compute & Existing Dashboards**:
   - Rust `poke-engine` runs turns in ~13 µs (full Q-learning or mini-experiments run in 5–15 seconds!).
   - `rl/cockpit.py` (`:8766`) visualizes live RL training, seed bands, ablations, and turn-by-turn MCTS diffs.
   - Python via `uv` (never `pip`). User global rule: except for benchmarking, all training uses minimal parallelism (only what's necessary to verify code).
   - User has limited free time per check-in — sessions must work well in short bursts (5–15 min) AND deeper dives (25–45 min).
   - `CLAUDE.md` **minimal-code rule**: no bloated LMS, databases, or framework infrastructure.

## What Your Round 1 Proposal Must Cover
Take a bold, opinionated stance (do not hedge everything). Cover all 6 sections concretely (target ~200–350 lines):

1. **Delivery Architecture: On-Demand Skill vs. Scheduled Task vs. Hybrid**
   - Give a clear verdict and argue *why* (addressing habit formation, friction, stale notifications, multi-turn interactivity, and cross-provider Antigravity + Claude skill discovery).
   - Specify the exact minimal file footprint.
2. **Curriculum Architecture: Big Picture → Fine Detail**
   - How do you give the user a crystal-clear mental map of all of RL first (so every detail slots into a coherent framework), then drill down to real mathematical and engineering mastery (Bellman optimality, TD vs MC, variance/bias, policy gradients/PPO/GAE, exploration/UCB/pUCT, value fitting & confounding, opponent/world-model exploitation, imitation vs search vs self-play, statistical rigor)?
   - Map concrete modules onto *actual files, bugs, and war stories* in this repo (e.g., Lesson 1 `"brn"` vs `"BURN"` aliasing, Lesson 2 A2C 2% vs PPO 74% & greedy overfitting 0–22% vs MCTS, Phase 1 RR-AI −33 pts -> mixture `p=0.8` +30 pts, Phase 4 raw fit −2.4 pts -> fixed-effects +11.0 pts, Attempt #2 `grad_clip=0.5` & `gamma=0.95` shaping hack, Lessons 3–4 Numbrion/AlphaZero).
   - Also explain how the curriculum handles RL concepts *beyond* what's already in the repo (so the user learns general RL, not just Pokémon tricks).
3. **The Fun Engine & Anti-Illusion-of-Understanding Design**
   - What makes this genuinely *fun* to come back to — without cringey or shallow gamification?
   - What concrete active-learning mechanics guarantee real retention and intuition (not just nodding along to well-written prose)? Be concrete about how you use `poke-engine` micro-experiments, `html_render` interactive widgets, `rl/cockpit.py`, `no_AI_code_here/`, prediction-before-run bets, bug hunts, and real repo decision rehearsals.
4. **Session Anatomy & State/Progress Tracking (Minimal-Code)**
   - Walk through the exact flow of a 5–10 min check-in vs. a 25–30 min lab session.
   - How does the tutor track mastery, misconceptions, spaced repetition, and stay synced with new git commits / `NOTEBOOK.md` entries using near-zero code?
5. **One Concrete Sample Check-In (Show, Don't Just Tell)**
   - Script out one vivid, realistic mini-session (opening prompt / interactive move / twist / takeaway) so the user can *feel* what using your system is actually like.
6. **What You Would Explicitly NOT Build**
   - Name 3–5 tempting bad ideas you reject and why.

## Rules
- **Read-only on the repo** except writing your ONE assigned output file (`r1_<model>.md`) in `/home/tristan/CodingProjects/nuzlock_rl/rl/research/rl_tutor/`.
- Do NOT run training, battles, pytest, or heavy compute. Do NOT touch any other file or git state.
- End your final message with a crisp ≤10-line summary of your core thesis.
