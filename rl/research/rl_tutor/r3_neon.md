# Round 3 Convergence Stress-Test & Sign-Off — Neon (`gemini-4-neon`)

## 1. Verdict & Single 2× Improvement

**Verdict: Conditional Sign-off.**
The converged blueprint is lean, honest, and ready to build, conditional on three pre-ship fixes: (1) tightening the `SKILL.md` trigger description so ordinary debugging never fires the tutor, (2) keeping `rl/LEARNING.md` out of git tracking (`.gitignore` + worktree-shared path) so it never collides with Phase 6 commits/worktrees, and (3) completing the 10-minute pre-build verification checklist.

**Single 2× Improvement: Invert Session #1 — Bet First, 2-Row Map Second (grow 1 row/session).**
`DRAFT_SYNTHESIS.md §2` currently opens the user's *very first* `/rl-tutor` session by asking them to fill 3–4 rows × 4 columns (12–16 blank cells) before running a single bet. That feels like intake paperwork and kills Day-1 pull. Flip it: open Session #1 directly with the 10-second live `--gamma 0.9` house bet (surprise + payoff in 60 seconds), then have the user fill just **2 rows** (`lesson1_qlearning`, `lesson2 --algo ppo`) to reveal the GPI caption as the close. Every later session appends 1 row as that region is hit—zero Day-1 exam, zero empty-table bloat.

---

## 2. Failure Modes Across Sessions #1, #5, and #20

### Session #1 (Cold Start, Onboarding Friction, First Impression)
- **Intake-form bounce (12–16 blank cells before first bet):** Starting with a taxonomy worksheet burns 5–8 minutes before the user experiences what makes the skill fun.
  *Fix:* Run the live `--gamma 0.9` bet in Turn 1; place 2 rows on the map at the close (see §1).
- **Missing ledger & unverified symlink discovery:** `rl/LEARNING.md` does not exist on Session #1, and `.claude/skills/rl-tutor` as a directory symlink is untested in both harnesses.
  *Fix:* `SKILL.md` Step 0 auto-creates `rl/LEARNING.md` from a 10-line template if missing; verify symlink discovery in Antigravity and Claude Code before shipping (fall back to a 5-line stub `SKILL.md` if either skips symlinks).

### Session #5 (Repetition Fatigue, Cross-Provider Drift, Widget Latency, Git Noise)
- **`rl/LEARNING.md` git noise & worktree split-brain during Phase 6:** A git-tracked ledger dirties `git status` after every check-in/D#, leaks into `git commit -am`, and diverges across T3 git worktrees.
  *Fix:* Add `rl/LEARNING.md` to `.gitignore`, and instruct `SKILL.md` to resolve it via `$(git rev-parse --path-format=absolute --git-common-dir)/../rl/LEARNING.md` so every worktree shares one local file with zero git noise.
- **Cross-provider hook & rule drift (Antigravity vs Claude):** Antigravity loads `AGENTS.md`/`GEMINI.md` (`.agents/rules/`), **not** `CLAUDE.md` (`agy-customizations/SKILL.md:25`), so any `CLAUDE.md`-only hook silently vanishes in Antigravity sessions; meanwhile, 12 prose rules in `SKILL.md` decay into hint-leaking bets and verbose reveals by Turn 2.
  *Fix:* Mirror any chosen workspace hook into `AGENTS.md` (not just `CLAUDE.md`), and replace prose rules in `SKILL.md` with one 8-line "leaky vs neutral bet" contrast example plus a hard 3-slot turn skeleton.
- **Check-in repetition fatigue:** Five consecutive `NOTEBOOK.md` archive bets in the same rhythm turn mechanical.
  *Fix:* Rotate wrappers across 4 types—*(a)* ≤15 s live CLI probe, *(b)* historical `D#` rehearsal, *(c)* 3-line bug/sabotage snippet, *(d)* "Why Not Here?" shelf card—never repeating the same wrapper twice in a row.
- **Widget dead air (`html_preview` + `html_render` = 10–15 s wait):** Firing two tool calls plus 150 lines of HTML inside a 5-minute check-in breaks momentum.
  *Fix:* Default to a compact Markdown table; render a widget in at most ~1 in 3 check-ins, strictly for continuous-knob twists.

### Session #20 (Ledger Bloat, Token Budget, Comfort-Zone Plateau, Fast-Moving Repo)
- **Ledger silently crosses 80 lines:** 14 region rows + 10-line `My Map` + 8 misconceptions + 12 log lines + headers already equals ~78–84 lines before a single long note.
  *Fix:* Merge `My Map` and `Regions` into **one 14-row table** (4 short user phrases + evidence + due date) to reclaim ~14 lines, and make `wc -l > 80` a mandatory Step-0 fold trigger.
- **`SKILL.md` + `map.md` context bloat on short check-ins:** Reading all 135 lines of `references/map.md` every check-in crowds prompt attention and weakens adherence to anti-illusion rules.
  *Fix:* Read `rl/LEARNING.md` first and `grep` only the active region block from `references/map.md`; load full `map.md` only on `/rl-tutor map` or when selecting an unseen region.
- **Easy-check-in plateau (all `P`, zero `B`):** User accumulates `P` across 10 regions via fun 5-minute bets without ever hand-coding a kata or derivation.
  *Fix:* Add one ledger trigger: once ≥4 regions hold `P` with zero `B`, Turn 1 defaults to a 10-line micro-kata in `no_AI_code_here/rl/` instead of another archive bet.
- **Repo velocity breaks sync & anchors:** A 2-week Phase 6/7 sprint creates 40+ commits in `git log <synced>..HEAD` and renames symbols referenced in `map.md`.
  *Fix:* Filter sync to `git log <synced>..HEAD --oneline -n 15 -- rl/ NOTEBOOK.md CLAUDE.md RR_PLAN.md scripts/`, advance `synced` to `HEAD` every session, and repair any stale `map.md` symbol anchor in-place on first miss.

---

## 3. Crisp Votes on All 5 Open Tradeoffs

1. **Tradeoff 1 — `CLAUDE.md` Work-Session Hooks:** **Vote: Hybrid — DROP passive `Fresh` append; flag Hook A as an explicit User Decision (`D#`), defaulting to Option B (OFF) unless opted in.**
   *Why:* `git log` + `NOTEBOOK.md` headings already surface every new phase/D# without dirtying state during coding, so `Fresh` append is redundant. Whether to pause before recommendations at rare `D#`s is pure personal-workflow taste—ask the user directly (and if enabled, mirror the 2 lines into `AGENTS.md` so Antigravity honors it too).
2. **Tradeoff 2 — Check-In Turn Cadence & Line Caps:** **Vote: Option A (Lock by panel consensus) — 3 user round trips, ≤6 lines question turns / ≤10 lines reveal turns.**
   *Why:* Local `git log` + `grep '^## ' NOTEBOOK.md` runs in <100 ms—there is no slow sync to hide behind an extra turn. A 4th round trip turns a 5-minute habit into a 10-minute chore.
3. **Tradeoff 3 — `html_preview` Before `html_render` in Check-Ins:** **Vote: Option A (Lock by panel consensus).**
   *Why:* T3 Code explicitly requires `html_preview` before `html_render`, and one clipped or broken chart costs more time than a 3-second preview. Control latency by rendering fewer check-in widgets (~1 in 3), not by skipping verification (word `SKILL.md` tool-agnostically for Antigravity/Claude).
4. **Tradeoff 4 — Mastery Evidence Schema & Local Prerequisite:** **Vote: Option A (`P / T / B / D`, no hard local gate — Lock by panel consensus).**
   *Why:* Changing my R2 vote to Fable's `P / T / B / D`: elevating `T` lets Session N queue the non-Pokémon transfer as Session N+1's opening recall, yielding honest cross-day ownership without bloating check-ins. Skip Opus's local gate before `B`; a 2-line map pin gives context without blocking a user who wants to code a kata today.
5. **Tradeoff 5 — Scheduled Teaser Self-Silencing:** **Vote: Option A (Lock by panel consensus; keep schedule OFF at launch, ask user only if they want Day-1 opt-in).**
   *Why:* Reading the bound thread before posting prevents headless cron runs from ever mutating `rl/LEARNING.md` mid-session or causing race conditions; fall back to `card: open` only on hosts without thread-read tools.

*(Panel-locked: Tradeoffs 2, 3, 4, 5. Surface to user as explicit `D#`: Tradeoff 1 Hook A, plus confirming `.gitignore` for `rl/LEARNING.md`.)*

---

## 4. `SKILL.md` Trigger & First-60-Seconds UX Check

### Frontmatter Critique & Fix
The draft description in `DRAFT_SYNTHESIS.md §1` ends with: *"or understand why an experiment, bug, or algorithm in this repo behaved as it did."* **That is a severe false-trigger bug.** During normal Phase 6 work, asking *"Why did Misty drop 4% on this experiment?"* or *"Why did `engine_search` fail?"* will trip progressive disclosure in both Antigravity (`agy-customizations/docs/skills.md:51-54`) and Claude Code, hijacking routine debugging into a Socratic quiz.

**Sharpened YAML frontmatter:**
```yaml
---
name: rl-tutor
description: >-
  Interactive RL tutor grounded in nuzlock_rl (check-ins, deep labs, decision rehearsals,
  RL map). Use ONLY when the user invokes /rl-tutor or explicitly asks to learn RL, run
  a tutor check-in or lab, quiz themselves, rehearse a D# decision, or review the RL map.
  Do NOT trigger for routine debugging, experiment analysis, or normal coding questions.
---
```

### Exact First 5 Lines on the User's Very First `/rl-tutor`
*(Verified against `rl/lesson1_qlearning.py:27,54,102,104`; house commits its bet in Turn 2 right before executing `uv run` so the user never anchors on Turn 1.)*

> **Live bet (unrun — house bets too before running).**
> In `rl/lesson1_qlearning.py` (Garchomp vs random Rotom, sparse `+1/−1` at KO, 30k episodes), default `--gamma 1.0` hits ~98% win rate.
> If we rerun with `--gamma 0.9`, what happens to the **turn-1 Q-ranking** (Swords Dance vs Dragon Claw) and the **final win rate**?
> **A)** Ranking & win rate unchanged   **B)** Switches away from Swords Dance, win rate ~same   **C)** Win rate drops >5 pts
> Reply with letter + 1-line *why* + confidence (`50 / 70 / 90`). Then we run it live and pin `lesson1` onto your RL map.
