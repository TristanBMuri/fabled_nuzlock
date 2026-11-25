# Round 3 Brief: Convergence Stress-Test & Final Sign-Off

Read `/home/tristan/CodingProjects/nuzlock_rl/rl/research/rl_tutor/DRAFT_SYNTHESIS.md` and all three Round 2 files (`r2_neon.md`, `r2_fable.md`, `r2_opus.md`). Do NOT read any other `r3_*.md` file.

This is the **final round** before we present the converged design and open decisions to the user. Be brutally practical, concise (target **≤ 150 lines**), and focused on what actually survives contact with real daily usage.

## Required Sections for `r3_<model>.md`

1. **Sign-Off or Veto + Single 2× Improvement**
   - State clearly: **Sign-off**, **Conditional Sign-off**, or **Veto**.
   - Name the **one single change** (not five) that would make this design 2× more fun, more effective, or cleaner to run.

2. **Failure Modes Across Sessions #1, #5, and #20**
   - Stress-test what silently breaks over time and give a concrete 1–2 line fix for each:
     - **Session #1** (cold start, onboarding friction, first impression)
     - **Session #5** (repetition fatigue, prompt-rule drift across Antigravity vs Claude, widget latency, `rl/LEARNING.md` git noise while working on Phase 6!)
     - **Session #20** (ledger bloat, `SKILL.md` / `map.md` token budget, plateauing in easy check-ins vs hard labs, syncing with a fast-moving repo).

3. **Crisp Votes on All 5 Open Tradeoffs**
   - For each of **Tradeoffs 1–5** in `DRAFT_SYNTHESIS.md`, give your **vote (Option A / B / hybrid)** + a 1–2 sentence reason. Flag if any tradeoff should be left as an explicit user decision (`D#`) vs locked by panel consensus.

4. **`SKILL.md` Trigger & First-60-Seconds UX Check**
   - Critique the draft YAML `name`/`description` in `DRAFT_SYNTHESIS.md §1` (will both Antigravity and Claude Code reliably trigger it — and NOT falsely trigger it during normal coding?).
   - Write (or sharpen) the exact first 4–6 lines the user sees the **very first time** they type `/rl-tutor`.

## Rules
- **Read-only on the repo** except writing your ONE assigned output file (`r3_<model>.md`) in `/home/tristan/CodingProjects/nuzlock_rl/rl/research/rl_tutor/`.
- Do NOT read any other `r3_*.md` file. Do NOT run training, battles, pytest, or heavy compute.
- Keep it tight (≤ 150 lines). End your final message with a ≤ 8-line summary of your verdict and votes.
