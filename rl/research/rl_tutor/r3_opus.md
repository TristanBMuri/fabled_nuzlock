# Round 3 — Opus: stress-test and votes

Read: `ROUND3_BRIEF.md`, `DRAFT_SYNTHESIS.md`, `r2_neon.md`, `r2_fable.md`, `r2_opus.md`. Checked read-only
this session: `rl/lesson1_qlearning.py` (docstring says "~1 min", `--gamma` default 1.0 at :104, prints
"Win rate vs the random Rotom, 2000 battles each" at :119), `.gitignore` (`rl/LEARNING.md` is *not*
ignored; `logs/` is), `.agents/` still absent, NOTEBOOK.md records D#s inline (e.g. `:3064` "the user's
D11 choice"). Learning-science and harness claims are *recalled* unless a line is cited.

## 1. Verdict: **Conditional sign-off**

Conditions (all small): (a) fix the trigger description (§4), it will fire during normal debugging as
written; (b) no tutor writes to a tracked file outside `/rl-tutor` sessions (§2, #5); (c) SKILL.md ships
the exact `LEARNING.md` skeleton and one worked check-in transcript, not only rules (§2, #1 and #5).

**The single 2× change: pre-register the project's own experiments as the live bet stream.**
When the working agent (any provider) launches a real multi-minute run (planner table, league test,
phase fit) or reaches a D#, it asks once, *while the run is already going*: "Bet? (result + 5/7/9, or
`skip`)". The answer goes into that experiment's NOTEBOOK.md entry as `user bet: … 70%` (or
`user call: … 70%` at a D#). `/rl-tutor` settles them via `grep -n 'user \(bet\|call\):' NOTEBOOK.md`.
Why this doubles the value:
- **It fixes a scarcity the draft hides.** "House bets only on live runs" + "check-in runs ≤ 60 s" leaves
  roughly one honest live bet family: lesson 1 flag variants. By session #5 those are used up and every
  bet is archive (tutor-known) or arithmetic. Project runs are live, nobody knows the answer, and the house
  can bet honestly every time.
- **Zero compute and zero wall-clock**: the run happens anyway; the user is waiting on it anyway.
- **Highest stakes available**: it trains the exact skill CLAUDE.md demands ("trustworthy win/death
  probabilities") on the results the project actually depends on, and the settled bet is a real P or D.
- **Zero new files and no git noise**: NOTEBOOK.md is already committed with each phase; it also turns
  the notebook into a pre-registration record, which is good research practice independent of tutoring.
It subsumes Hook A and the passive Fresh append (see Tradeoff 1).

## 2. Failure modes at sessions #1, #5, #20

**Session #1 — cold start**
- *`LEARNING.md` does not exist; each provider invents its own schema.* Fix: SKILL.md carries the literal
  ~20-line skeleton; first action on a missing file is to copy it, nothing else.
- *The partial-fill table opens the course as a worksheet.* 3–4 rows × 4 abstract columns ("where do
  targets come from?") is 12–16 cells before the first surprise. Fix: **bet first, table second** — commit
  the γ bet, start the runs in the background, and fill **2 rows** while they run (latency hidden by work).
- *The bet has no baseline.* `--gamma 0.9` means nothing without the γ = 1.0 win rate on the same seed,
  and lesson 1 is "~1 min" per its docstring (10 s per CLAUDE.md). Fix: run both concurrently (2 cores,
  same `--seed`) and quote the 2000-battle band (±~2 pts at p≈0.5: √(.25/2000)≈1.1%, ×2) with the reveal.
- *Skill not discovered in one harness → user's first impression is "nothing happened".* Fix: already in
  §5 checklist; add one line to CLAUDE.md as the safety net: "`/rl-tutor` → read
  `.agents/skills/rl-tutor/SKILL.md` and follow it".

**Session #5 — fatigue and drift**
- *Template fatigue*: same three beats, same reveal type every time. Fix: the twist must use a different
  reveal type than the bet (live → derivable, archive → live), so each session has two textures.
- *Surprise meta-gaming*: if every twist reverses intuition, the user learns "pick the counterintuitive
  option" — a calibration illusion, not RL. Fix: SKILL.md rule: about 1 in 3 twists confirms the naive
  answer; the tutor never signals which.
- *Provider drift (Gemini wordier, Claude more lecture-prone; both drift from 11 abstract rules).* Fix: one
  ≤ 15-line worked transcript in SKILL.md (models copy examples more reliably than they follow rules,
  *recalled*), plus a pre-send check: "≤ N lines? ends in a user action? nothing explained before attempt?"
- *Widget latency*: the real cost is generating 100–200 lines of HTML (tens of seconds), not the second
  tool call. Fix: check-in widget default **0**; only on "show me" or in labs.
- *Git noise during Phase 6*: `rl/LEARNING.md` is tracked; any mid-work append makes `git status` dirty and
  gets swept into a Phase 6 commit. Fix: **only `/rl-tutor` sessions write it**; mid-work signals go into
  NOTEBOOK.md (already being committed) per the 2× change; the session close says "ledger changed —
  commit it separately when you commit next" (never commits itself).

**Session #20 — bloat, plateau, rot**
- *Ledger bloat*: LLMs append, rarely delete. Fix: first tool call is `wc -l rl/LEARNING.md`; > 80 → fold
  before anything else (draft has this; make it the literal first step, not a guideline).
- *Anchor rot in `map.md`*: grep strings break too (six phases in two days). Fix: grep every anchor before
  quoting it; on a miss, say so and fix that one line in `map.md` (the only tutor-owned file besides the
  ledger).
- *Broken sync*: `synced` hash disappears after an amend/rebase → `git log <synced>..HEAD` errors. Fix:
  `git merge-base --is-ancestor` fails → fall back to `git log --since=<last date>`.
- *Plateau: check-ins only, labs never.* P accrues, B never does, "owned" stalls everywhere. Fix: when a
  region has ≥ 2 P and no B, the check-in twist becomes a 3–5 line fill-in (one line of GAE, the SPRT
  update) typed in chat and checked by running it — a mini-B inside a check-in.
- *SKILL.md + map.md budget (~250 lines, a few k tokens)*: fine; the risk is growth by accretion. Fix: the
  line caps in §1 of the draft are hard; a new rule must replace an old one.

## 3. Votes on the five tradeoffs

**T1 — work-session hooks: hybrid, and leave it as an explicit user decision.**
Vote: replace both Hook A and the Fresh append with the pre-registration line (§1): one question, asked
while a run is going or at a D#, `skip` sticky for the day, answer stored in NOTEBOOK.md. I withdraw my
Hook A *as a separate pre-recommendation round trip* where no run is pending: Neon is right that it costs
a turn at 10 p.m.; asking during a run costs nothing. Drop the Fresh append (git log + NOTEBOOK headings
already find it, and it dirties the tree). **This changes how the user's work sessions feel → user D#.**

**T2 — cadence: Option A (3 round trips).** I concede. My 4th trip existed to hide sync latency, but sync
is two sub-second commands. Caps: ≤ 6 lines before an action, ≤ 10 on a reveal (excluding one table/code
block). The binding rule is "nothing explained before the attempt", not the count. Lock by consensus.

**T3 — html_preview: Option A.** I withdraw the fast path: T3's runtime instructions say `html_preview`
then `html_render`, and the call is not the bottleneck (see #5). Keep SKILL.md tool-agnostic (Antigravity
uses `<agent-embed>`). Lock by consensus.

**T4 — mastery: Fable's P / T / B / D, no local gate.** I switch. T is the only evidence that lands on a
later day automatically (the transfer question opens the next session), and "one of T or B" stops pure
prediction streaks from counting as ownership. My local gate collides with priority #1 (explicit user
request) and adds a check the tutor must remember; the 2-line map pin is enough. B still absorbs
derivations. Lock by consensus.

**T5 — teaser silencing: Option A (read the thread, zero writes).** I withdraw `card: open`. A headless run
writing a tracked file mid-Phase-6 is exactly the git-noise failure in #5, and can race the working agent's
edits. The schedule ships off anyway, so this is low-stakes. Lock by consensus.

Explicit user decisions to put in front of the user: **T1** (hook in work sessions), plus one the draft
missed: **`rl/LEARNING.md` tracked vs gitignored** (tracked = history of learning in git, needs separate
commits; ignored = zero noise, but `git clean -fdx` deletes it and no history).

## 4. Trigger and the first 60 seconds

**Critique of the draft description.**
- *False trigger (serious):* "understand why an experiment, bug, or algorithm in this repo behaved as it
  did" describes ordinary Phase 6 debugging. A model asked "why did the planner lose to Misty?" would load
  the tutor and start quizzing. "rehearse a D# decision" will also match routine D# discussion.
- *Missed trigger:* the literal invocation strings are not in the text; Antigravity matches on description
  only (Claude Code also matches `/rl-tutor` by name).
- If false triggers still appear in week 1, Claude Code's `disable-model-invocation: true` frontmatter
  (*recalled*; verify) makes it slash-only there; Antigravity would still need the description fix.

**Sharpened frontmatter:**
```yaml
---
name: rl-tutor
description: >-
  RL tutor for the nuzlock_rl repo: short check-ins and labs where the user bets on results, then the
  repo answers. Use ONLY when the user explicitly asks for /rl-tutor, an RL check-in, lab or quiz, to be
  taught an RL concept, or to settle their RL bets. Do NOT use for ordinary debugging, experiments, code
  changes or D# decisions during project work, even when they involve RL.
---
```

**First run, exactly what the user sees** (bet first; runs start after the commitment):
```
RL tutor, session 1. No lectures: you bet, the repo answers, I only correct. ~8 min; `stop` ends it.
Lesson 1's Q-learner (Garchomp vs a random Rotom-Wash, +1/−1 at the end) trains with γ = 1.0.
I'm about to train it again with γ = 0.9, same seed. Nobody has run that, me included.
Its win rate vs the random Rotom (2000 battles): higher, same (±2 pts), or lower?
Reply: letter + one sentence why + how sure (5/7/9). I'll bet after you, before the run starts.
```
Then: both runs start in the background, the user fills two map rows while they run, reveal with band,
twist free-form ("why might γ < 1 matter even with only ±1 at the end?"), takeaway, ledger created.
