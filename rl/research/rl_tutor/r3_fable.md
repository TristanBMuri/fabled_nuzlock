# Round 3 — Fable: stress-test, votes, sign-off

Read: `ROUND3_BRIEF.md`, `DRAFT_SYNTHESIS.md`, `r2_neon.md`, `r2_fable.md`, `r2_opus.md`; no `r3_*` file.
Checked read-only this session, nothing run: `rl/lesson1_qlearning.py`, `rl/README.md:210-230`, `.gitignore`,
`git worktree list`, Antigravity's `skills.md` and `rules.md`. Claims about habit are judgment, not data.

## 1. Verdict: conditional sign-off

The design is sound and small enough. Two conditions, both cheap:
1. **Rewrite the skill description** (§4). As drafted it will load tutor mode during ordinary debugging.
2. **Decide where `rl/LEARNING.md` lives with respect to git** (§2, session 5). I recommend untracked.

**The one change: every session ends by writing the next session's opening message.** The close already
sets a transfer question. Extend it to one header line in `rl/LEARNING.md`:

    next: recall "<question>" · bet <region>, <live|archive|derivable>, "<question>"

`/rl-tutor` then reads one file and prints that message. What the one line buys:

- **A first message within seconds, every time.** No `map.md` read and no question design before turn 1.
- **Session 1 is no longer a special flow.** The shipped `LEARNING.md` carries a `next:` line (the bet in
  §4), so the onboarding branch leaves `SKILL.md`. The map fill moves behind the first reveal and shrinks
  to the one row just played; each later session adds its row. The 3–4-row worksheet goes.
- **Less provider drift.** The model that taught a session picks the follow-up while it still has the
  context. The next model, possibly another provider, only delivers it.
- **Tradeoff 2 settles itself**, and the user leaves knowing what is waiting ("next: a live bet on
  entropy"), which is the pull the weekly teaser was meant to supply.

Override: `git log <synced>..HEAD` runs in the same tool batch as the read. If it shows a new result or a
pending D#, the bet is swapped for that. The recall stays.

## 2. Failure modes

**Session 1**
- **The first reveal lands inside the noise.** The draft's first bet is the win rate under `--gamma 0.9`.
  The baseline is 98.6% over 2000 battles (`rl/README.md:215`), so one standard error is
  √(0.986 × 0.014 / 2000) ≈ 0.26 points, and README:230 already asks whether 98.6 vs 98.0 is noise. The
  likely reveal is "no visible change". *Fix:* bet on the best opening move's Q-value (Swords Dance, +0.98
  at γ = 1.0), which γ must shrink (my rough estimate, not run: 0.7–0.85). The twist asks whether the
  play changed too. Value scale against policy is a real first lesson, and it shows above the noise.
- **Lesson 1's runtime is unknown:** ~10 s (README:210) or ~1 min (docstring, line 7). *Fix:* time the
  default run (checklist item 2) before fixing the first bet, and leave `--gamma 0.9` unrun.
- **Five rules arrive at once** (confidence digit, reveal type, house bet, takeaway, map). *Fix:* the
  first message states one rule. Each of the others is introduced in one line when first used.
- **`/rl-tutor` is not a command in Antigravity.** Its `skills.md` documents activation by description
  only, with `name` and `description` as the only fields. `/rl-tutor lab` is plain text there. *Fix:* the
  literal string goes in the description (§4), and checklist item 1 tests the typed string.

**Session 5**
- **The ledger in git.** Three hazards. (a) `git stash`, `git checkout -- .` or `git reset --hard` during
  Phase 6 work silently removes uncommitted ledger edits. (b) A phase commit made with `git add -A`
  carries the file to GitHub, verbatim misconceptions included. (c) A session in a worktree reads a stale
  copy and writes to a branch (one worktree today). *Fix:* add `rl/LEARNING.md` to `.gitignore`, and have
  `SKILL.md` resolve it at the main checkout:
  `$(git rev-parse --path-format=absolute --git-common-dir)/../rl/LEARNING.md`. Cost: no history and no
  copy off this machine; only `git clean -x` removes it. This is the user's file, so it is their call.
- **CLAUDE.md hooks may not exist in Antigravity.** Its `rules.md` names `GEMINI.md` and `AGENTS.md` as
  rule files, and the repo has neither. Whether Antigravity sessions see Hook A is unverified. *Fix:* one
  checklist item. If they do not, either symlink `AGENTS.md -> CLAUDE.md` (the user's call: it changes
  every Antigravity session) or accept that Hook A is Claude-only.
- **Three models grade differently.** One awards `P` for an almost-right twist. *Fix:* each Log line names
  the model and quotes the ≤ 8 words of the user's that earned the letter, so the next model can audit it.
- **Repetition, and the takeaway loop.** Every check-in has the same four beats, and "sharpen until exact"
  becomes two or three extra round trips. *Fix:* one sharpening pass, then the tutor prints the exact
  sentence beside the user's and closes. Log the bet format (number, ranking, diagnose a curve, decision
  rehearsal) and never use the same one twice running.
- **Live bets run out.** A check-in run is capped at 60 s on one core, which today means lesson 1's flags.
  The house bet then disappears. *Fix:* an existing command with an unused `--seed` counts as live (lesson
  1 has `--seed`, line 106). It also teaches seed variance.

**Session 20**
- **Line caps do not cap size.** The draft's region entries run past 250 characters, and evidence cells
  grow sideways. *Fix:* budget in words: `SKILL.md` ≤ 1,100, `map.md` ≤ 1,800, `LEARNING.md` ≤ 700. A
  region row keeps only the latest date per evidence letter.
- **`map.md` goes stale.** It quotes results (`+11.0 ± 4.1`, `0.189 → 0.096`) and flags that the notebook
  and the code will supersede. *Fix:* it holds symbols and grep strings, not numbers. The tutor greps an
  anchor before using it and repairs a dead one in `map.md` in the same session. Numbers are read from the
  source at reveal time, which rule 7 already requires.
- **The ledger drifts when rewritten.** Twenty full rewrites of an 80-line file by an LLM will corrupt
  counters. *Fix:* single-line edits only; counters are incremented, never recomputed; folding touches the
  Log section alone.
- **Plateau.** Check-ins produce `P` and `T`; `B` needs a lab. Regions stall, or the user owns eight
  regions without writing code. *Fix:* after five check-ins in a row with ≥ 3 regions lacking `B`, the
  opener offers one 15-minute kata in place of the bet. Declining resets the counter.
- **`synced: <sha>` breaks after an amend or rebase**, and `git log <sha>..HEAD` errors. *Fix:* store the
  date beside it and fall back to `git log --since=<date>`.
- **The scoreboard is noise at this size.** About 7 bets per bucket gives a standard error near 0.17 at
  70%. *Fix:* each bucket prints "too few" under n = 15. The gate question after 15 bets says the same.

## 3. Votes

1. **A plus auto-off; drop the Fresh append. User decision (D#).** A `D` row is the only evidence the
   tutor did not stage, and a call made after reading the recommendation cannot be recovered as
   independent. But D4–D8 all sit in commits dated 2026-11-24: five in one day, not Opus's one a week.
   So: after three `rec` or `pass` in a row the hook writes `hook: off` and stays silent until the user
   turns it back on. Fresh append: I kept it in Round 2 and Opus is right. It makes every work session a
   ledger writer without the skill's schema loaded, and `git log` plus NOTEBOOK headings recover the same.
2. **A, three round trips. Lock.** Opus's fourth turn hides the sync behind the user's thinking. In T3,
   text written before a tool call may reach the user only as a one-line summary, so that needs a
   host-specific tool. The sync is two cheap calls, and `next:` removes the wait anyway. Caps in words,
   not lines: question turn ≤ 70, reveal turn ≤ 130 plus one table or code block.
3. **A, preview then render. Lock.** T3's instructions say so, and a skill that contradicts its host is
   a drift source. Antigravity renders differently, so `SKILL.md` names no tool: "render with the host's
   HTML tool, following its rules". The latency is the 100–200 lines of HTML, not the second call, so
   control it with the budget: a check-in has no widget by default.
4. **A, four letters, no local gate. Lock.** `T` is the only letter delayed by construction, and it is
   what `next:` stores. Opus's gate contradicts pick rule 1 (the user asked for a derivation today); the
   plateau fix in §2 covers its purpose. Letters stay visible: "owned: P, T" is not "owned: P, B".
5. **A, read the thread, write nothing. Lock.** A headless write can race a live session on the same
   file. If the thread cannot be read, stay silent. One line in `SKILL.md` until the user turns the
   teaser on: it is off at launch, and `schedule_task` is a T3 tool that Antigravity does not have.

One more decision for the user, outside the five: **`rl/LEARNING.md` tracked or untracked** (§2).

## 4. `SKILL.md` trigger and the first 60 seconds

**Critique of the draft frontmatter**
- *"...or understand why an experiment, bug, or algorithm in this repo behaved as it did"* describes most
  work in this repo. If the skill loads mid-debugging, rule 1 (no mechanism before the user's attempt)
  makes the agent withhold the answer the user asked for. That clause must go, and an exclusion must
  replace it.
- Antigravity matches on the description alone, so the literal `/rl-tutor` belongs in it. No extra keys
  (`disable-model-invocation` and `argument-hint` are Claude Code fields); the body parses the mode.
- Two tests for checklist item 1, in both harnesses: typing `/rl-tutor map` loads the skill; asking "why
  did R2-only lose to B2?" in the middle of work does not.

```yaml
---
name: rl-tutor
description: >-
  RL tutor for nuzlock_rl: a short check-in, a lab, a D# rehearsal or the learner's RL map, with a bet
  before every reveal. Use ONLY when the user types /rl-tutor or rl-tutor, or explicitly asks to be
  taught, quizzed or tested on RL. Do NOT use for ordinary work in this repo: debugging, running or
  explaining experiments and making D# decisions follow CLAUDE.md, where answers are given directly.
---
```

**The first message, the very first time** (shipped as the `next:` line; Q and win rate from README:215)

```text
RL tutor, session 1. One rule: you bet before I show anything. About 8 minutes; `stop` ends it.

Live bet (nobody has run this). `rl/lesson1_qlearning.py` learns Garchomp vs a random Rotom-Wash with
gamma = 1.0. Its best opening move ends at Q = +0.98, and it wins 98.6% of 2000 battles.
I will rerun it with `--gamma 0.9`. What will that Q be? Give a number and a confidence: 5, 7 or 9
(for 50 / 70 / 90%). I post my own bet after yours, then run it.
```

Turn 2: the tutor's bet, the run, both scored, then the twist (is Swords Dance still the best opener,
and did the win rate leave ±0.5 points?). Turn 3: the user's one-sentence takeaway and lesson 1's map row.
