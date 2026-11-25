# Round 2 — Fable: keep the bet, drop the theatre

Read: `ROUND2_BRIEF.md` and all three `r1_*.md`. Checked in the repo (read-only, nothing run):
`rl/lesson1_qlearning.py`, `rl/cockpit.py`, `rl/cockpit.html:123-300`, `rl/lesson2_ppo.py:366-424`, one
record of `logs/rl/first.jsonl`, NOTEBOOK.md:3058-3166, `project_overview.md:600-606`,
`poke-engine/src/mcts.rs`, `no_AI_code_here/`, the `html_render` tool schema, Antigravity's own skills doc.
Learning-science claims are **recalled, not read**.

**Position.** My Round 1 was right about the core (a typed bet before every reveal, evidence-only mastery,
one hand-editable state file) and wrong in four places: "the tutor bets too" was partly theatre, Hook A was
too heavy as a mandatory step, "statistics is not a module" left the user reading bands someone else
computed, and five round trips are too many for five minutes. Neon has the best session shape and transfer
questions. Opus has the best map sentence and the best end of the coding arc. Both still let a recognised
answer or a retold story count as understanding.

---

## 1. What I'm stealing from the other two

**From Opus**

- **The GPI sentence as the map's caption** (every method alternates *evaluate* ↔ *improve*). My four
  columns had no verb. With it, lessons 3–4 "closing the loop" means something.
- **The era table as session 1's rows.** Mine were repo files; Opus's are things the user lived through.
- **Decision rehearsals from history** (D4, D10, D11 with only that date's evidence): sharper than my
  "NOTEBOOK as a held-out set", because it rehearses the decision format CLAUDE.md gives the user.
- **Primary source instead of paraphrase** for derivations, and **retrieve before re-teaching**.
- **Lessons 3–4 as the capstones**, written by the user on the env from `rl/lesson2_ppo.py` (`new_battle`
  :90, `observe` :197, `legal` :207, `step` :221 all exist). Better than my "40-line MCTS".
- **"Met in the wild"** logging, **full written-out due dates** (my `12-05` breaks in January), and
  tool-agnostic widget wording: Antigravity renders through `<agent-embed>`, not `html_render`.

**From Neon**

- **Three round trips per check-in**: recall question and new bet in one opening message. Round trips, not
  lines, are what a 5-minute session runs out of.
- **Option bets (A / B / C / something else)** for *direction* questions: one keystroke.
- **The "Why Not Here?" shelf**, grounded in RR_PLAN's rejection table (`RR_PLAN.md:197`, `:276`
  verified). It replaces my "every 4th session is foreign territory", a schedule in disguise.
- **The delayed non-Pokémon transfer gate**: the best single mastery gate in the three proposals.
- **The 1-line "fresh case file" append** as the default work hook, and the **thread-read self-silencing
  rule** for the teaser (no state write from a scheduled run; simpler than my `card: open` flag).
- **`no_AI_code_here/players/policy/policy_player.py:157`** as the shaping anchor (verified: `calc_reward`
  → `reward_computing_helper`). The user's own reward function beats any war story I listed.

---

## 2. Sharpest critiques and pushback

**Neon**

1. **The sample session is a lecture with a multiple-choice letter in front.** Option B's text *is* the
   explanation. The user recognises it, then the tutor explains fixed effects, shows the fix's code and
   three results (~18 lines, against its own "never >4 lines before asking"). The user never predicted the
   fix, and the node moves to `practiced`. Options suit a direction; the *mechanism* and the *fix* must be
   produced free-form.
2. **`intuited` is a status awarded for looking.** A rubric with a rung for exposure fills with exposure.
3. **Five bespoke widgets plus a clickable mastery map is a widget library written in prose**: each is
   100–200 lines of generated HTML and two tool calls, and the "live dashboard" is a snapshot of a table.
   24 nodes with status, two dates and a note do not fit a "~60-line" ledger.
4. **Statistics as Arc 6 means five arcs of reveals judged by eye.**
5. **Numbers the tutor would quote wrongly.** The teaser says R2-only "lost 5.2 points vs keeping Round 1".
   NOTEBOOK.md:3147: −5.2 ± 3.9 is against **B2**; against hand it was +2.8 ± 3.6. (My Round 1 made the
   same mix.) `mcts.rs:226` is node-insert code; D14 is at `RR_PLAN.md:180`, not :173.

**Opus**

1. **"R1 on all 14 nodes before R3 anywhere" fights its own best feature.** In its sample the user has just
   linked the policy-gradient baseline to fixed effects and the tutor proposes R3 on node 6: illegal under
   the rule until all 14 nodes hold a later-day R1. And R1 ("explain with its war story") is the rung most
   passable by recitation. Six sessions of it is the lecture feed with a quiz.
2. **Brier is the wrong scoreboard.** "7 bets, Brier 0.21" carries no information at n = 7, hides the
   direction of the error, and needs every bet kept as a row and recomputed by an LLM, the arithmetic risk
   Opus avoids for dates. The useful line in that sample is not Brier: "overconfident on *more information
   helps*, 0/3". That theme tally is worth keeping.
3. **Locked widgets hide the guess from the grader.** `html_render` takes `html / path / title / height`,
   returns nothing, and renders "above your final reply". "Guesses 45%, unlocks, sees 29%" exists only in
   the iframe unless also typed in chat, and then the lock is redundant.
4. **"At the end of a work session, offer a due card" has no trigger.** Sessions end by the user leaving.
5. **Attribution errors that erase a lesson.** Opus gives attempt #2 "PPO from scratch, 6M steps, ~14%"
   and "γ = 0.95 on 40-turn episodes". CLAUDE.md files both under attempt #3 (`project_overview.md:207`,
   `:603`); attempt #2 is where the same γ **worked** on ~25-turn battles. That contrast is the lesson.
   `mcts.rs:175` is sampling code (the crit rule is the comment at :19).
6. **14 × 4 = 56 cells, some meaningless** (R4 "Build" for "RL in LLMs", or for statistics).

**Both** cite line numbers into files that moved six phases in two days: anchors in `map.md` must be a
symbol or a grep string.

**My own Round 1**, same standard: "my bet was 0.13" on 0.95⁴⁰ is arithmetic, not a bet. The old shaping
scale does not "live in the DSPRO2 repo": `project_overview.md:603-604` has it (+30 win worth ~3.9 at turn
1, HP shaping ±1 per turn). I hardcoded `distrobox-host-exec git`, guessed that `eval` lines suffice for the
cockpit, and proposed checking GAE "on a saved buffer" that does not exist.

---

## 3. Position on the six fault lines

**FL1. Confidence yes, Brier no, tutor bets only on live runs.**
- Confidence does work *inside* the session: a miss at 90 is a misconception (verbatim, due tomorrow), a
  miss at 50 is just unseen. Neon's A/B/C bet cannot tell these apart. Cost: one digit (`5 / 7 / 9`).
- Scoreboard = counters incremented in place (`70%: 7/11`), no per-bet rows, plus the theme tally of
  confident misses. Shown only on `/rl-tutor map`. Whether it is fun is my judgment, not data, so it gets
  a gate: after 15 bets, ask once whether to keep it.
- The brief's objection to "tutor bets too" is correct. Before each bet the tutor declares the reveal type.
  **live**: a run whose flags or seeds are not in `logs/`, `results/` or NOTEBOOK, and the tutor may not
  look the answer up first; both bet. **archive**: only the user bets, the tutor says "I have read this
  one". **derivable**: no tutor bet. The tutor's row counts live bets only, roughly one check-in in three.

**FL2. The user fills the table; breadth is a default, not a gate; statistics is ambient *and* two case files.**
- Session 1: Opus's era rows × the four questions, two rows pre-filled, the user fills four in ~8 minutes,
  the tutor corrects only wrong cells, then states the GPI sentence. A guided interactive map is something
  you look at. The filled table is stored in the user's words and is what `/rl-tutor map` prints.
- Pick order: (1) what the user asked for, (2) a pending D# (D14 today), (3) a fresh commit, (4) a due
  item, (5) the next unseen region. A derivation in week 1 is allowed, preceded by two lines placing it on
  the map. One tie-breaker for breadth: after three sessions in one region the default is an unseen one.
- Statistics: I concede half. Ambient from bet 1 (every reveal carries its band; inside the band is "a
  lead, not a result"), plus *the noise floor* within the first three sessions (SE = √(p(1−p)/n), pairing,
  eval luck vs seed luck) and *selection* later (best-of-K then a fresh seed, SPRT, held-out bosses).

**FL3. Passive hooks by default; "your call first" is one optional question; the schedule is the user's choice.**
- Adopt Neon's append and Opus's "met in the wild" as one line format. Drop the end-of-session card.
- Hook A, demoted. CLAUDE.md already orders options → basis → recommendation → user decides. The change is
  one line before the recommendation: "Your call first? (choice + 5/7/9, or `rec`)". `rec` costs three
  keystrokes and can be sticky ("rec all today") while shipping phase 6. It fires only on decisions tagged
  D#. I keep it because a D row is the only mastery evidence the tutor did not stage, and a judgment made
  after seeing the recommendation cannot be recovered as independent.
- Schedule: Neon is right that a gate against the user's explicit request is paternalism. Opt in any time
  by asking; the gate (two gaps ≥ 10 days) only governs when the tutor may *suggest* it. Either way: one
  weekly teaser, one bound thread, silent while the last teaser has no reply.

**FL4. Chat is the input device; at most one widget per check-in, after the commitment.**
- Every guess is typed in chat. No client-side lock. Opus's lock survives honestly: the tutor **bakes the
  typed guess into the page** as a marker beside the real points and their bands.
- Budget: check-in 0–1 widget, at the twist or the close, never in the opening turn; lab 1–2. The default
  visual is a Markdown table. The two-call cost (`html_preview` for the height, then `html_render`) is why
  the budget is small. The map is a Markdown table unless `/rl-tutor map` is asked.
- A widget uses only numbers produced or quoted in this session, source path in its title. Saved only after
  the user asks for the same one twice (Opus's rule), at most three, in `references/widgets/`.

**FL5. Evidence letters, a cap per turn type, one accumulating folder.**
- Rubric = logged events, not awarded grades. `P` predicted an unseen variation with the right reason,
  `T` answered a non-Pokémon transfer question, `B` built by hand and matched the repo's reference, `D` made
  the call first in a real D#. **Owned = two different kinds on two different days, one of them T or B.**
  "P 2026-11-27" needs an event; "node 9 → R2, box 2" needs a judge. Opus's rungs survive as activity names.
- T lands on a later day for free: each session's transfer question opens the *next* session.
- Cap: question turns ≤ 4 lines; reveal turns ≤ 8 lines plus one table or code block ≤ 10 lines; every turn
  ends in something the user must do. Six was too tight for a reveal with a band and a twist; twelve lets a
  mini-lecture through. The rule that matters more than the count: **no mechanism or fix is explained
  before the user has attempted it.**
- Coding arc: Neon's size (10–15 lines a session), my continuity (one folder), Opus's destination.
  `no_AI_code_here/rl/`: `td.py` → `gae.py` → `bandit.py` → `pg.py` → lessons 3–4. Checked by execution;
  the tutor reports pass / fail and the first mismatch and never edits there. Backlog count never shown.

**FL6. Repo facts, checked.**

| Claim | By | What the repo says | Consequence |
|---|---|---|---|
| git needs `distrobox-host-exec` | Fable, Opus | `/usr/bin/git` 2.56.1 is on PATH in this box; `git log` works | skill says `git`, fallback `distrobox-host-exec git` |
| `.agents/skills/` is Antigravity's root | all | `~/.gemini/antigravity/builtin/skills/agy-customizations/docs/skills.md:10` names it | confirmed; neither `.agents/` nor `.claude/skills/` exists in the repo yet |
| a symlinked skill is discovered | all | **not checked** (needs creating it) | directory symlink first, else a 5-line stub; Neon's file-level links add upkeep for no known gain |
| `html_render` is one-way | Fable | schema has no return value; "above your final reply" | FL4 |
| cockpit draws a hand-written agent from `eval` lines | Fable | `cockpit.html:185,197` drops a run with no `start` record; `cockpit.py:38-41` takes status from the first line's `pid` and an `end` record | three record kinds: `start` (`t`, `config: {}`), `eval` (`step`, `t`, `vs: {opp: {win, n}}`), `end`. Read, not tried in a browser. `logs/` is gitignored |
| exact expectimax on lesson 1 | Opus | `lesson1:116` prints one *move* from a 500 ms search against a best-playing foe: no value, wrong opponent. `key()` drops exact HP and the Protect counter; the engine has `DecrementPP` (`instruction.rs:87`) | not a plain memoised recursion: with PP in the state the keys multiply, without it Earthquake into Levitate plus a Hydro Pump miss is a self-loop. Backward induction with turns-left in the key (the lesson caps at 50). Reachable-set size unmeasured: measure before listing the lab |
| GAE slider on a real lesson-2 battle | Opus | `battle` records hold per-turn `value` and a final `result` | works, for eval battles of the argmax policy |
| GAE kata "on a saved buffer" | Fable | `logs/rl/` holds `.jsonl` and `.pt` only | the tutor builds a small `buf` dict (`env, done, rew, val, act`) in the scratchpad with one `done=None` cut |
| ledger ≤ 80 lines forever | Fable (Neon: 60) | the budget in §4 is ~72 | only with region-level rows, settled bets folded into counters, capped lists; over 80, folding is the tutor's first act |
| `no_AI_code_here/` contents | all | the user's SB3 / poke-env code (`policy_player.py` 229 lines, `self_play.py` 192), last commit 2026-09-03 | one new folder, `rl/` |
| lesson 1 takes ~10 s | all | `CLAUDE.md:55`, `README.md:210`: ~10 s; the file's docstring (line 7): ~1 min | time it once before fixing the check-in budget |

---

## 4. Proposed merged architecture

```
.agents/skills/rl-tutor/SKILL.md            ≤ 110 lines: the protocol below
.agents/skills/rl-tutor/references/map.md   ≤ 130 lines: caption, 4 + 1 questions, 14 case files, shelf
.claude/skills/rl-tutor -> ../../.agents/skills/rl-tutor     (directory symlink; fallback: 5-line stub)
rl/LEARNING.md                              ≤ 80 lines, hand-editable, the only state
no_AI_code_here/rl/                         the user's code only
CLAUDE.md                                   +3 lines
```

Zero Python. `/rl-tutor` (check-in), `/rl-tutor lab`, `/rl-tutor <topic | file | D#>`, `/rl-tutor map`.

**The three CLAUDE.md lines:** (1) on a new D#, a finished phase, or an RL concept explained during work,
append one line to `rl/LEARNING.md` → Fresh; (2) at a D#, after options and basis and before the
recommendation, ask once "Your call first? (choice + 5/7/9, or `rec`)" and log it as a D row; (3) to learn
or be quizzed, use the `rl-tutor` skill.

**Curriculum backbone (`map.md`)**

- Caption (Opus): evaluate ↔ improve. Columns (Fable / Neon): what do you know about the world · what do
  you store · where do targets come from · whose data, which opponent. Fifth question on every row: how do
  you know it worked.
- 14 case files = my twelve (MDP and Bellman · state aliasing · horizon and reward · TD vs MC · policy
  gradient to PPO · exploration · who you train against · imitation and expert iteration · search ·
  opponent models · value fitting and confounding · the run level) plus *the noise floor* and *selection*.
  Each row: symptom first, anchor as a symbol or grep string, one bet with its reveal type, one kata, one
  transfer question.
- Shelf (Neon, plus Opus's nodes 5 and 14): deadly triad / DQN · SAC · RND / Go-Explore · MuZero / Dreamer
  · offline RL · RL in LLMs. Each: the repo's own "why not here" sentence, one primary-source section, its
  place in the four columns, one question back.
- Depth is an activity, not a gate: tourist bet → kata or derivation → the D# call first.

**Check-in, 5–10 minutes, three round trips**

| Turn | Tutor | User |
|---|---|---|
| 1 | recall question (last session's transfer question, or the oldest due item) + today's bet with its reveal type; ≤ 8 lines | answers both; choice or number + `5/7/9` |
| 2 | recall graded in one line; reveal with band and `file:line` or the command (≤ 60 s, one core); the tutor's bet scored if live; **twist**, free-form | predicts the twist |
| 3 | twist result; optional widget with the user's guess baked in; "your takeaway, one sentence" | writes it |
| close | corrects the sentence until exactly true, names the map cell, sets the next transfer question, edits the ledger, stops | — |

**Lab, 25–30 minutes:** two recalls → briefing with symptoms only → the user orders evidence and names a
suspect with a confidence → kata in `no_AI_code_here/rl/`, or a derivation after reading the named section
→ rehearsal of a pending D# or a transfer question → ledger.

**Compute:** lesson-2 runs only in labs with `--minutes ≤ 3`, or queued in the cockpit as an open bet.
Planner tables are never run for teaching; they are revealed from `results/`. One job at a time,
`MemAvailable` checked, ports 8000–8007 untouched.

**Rules in `SKILL.md`**

1. No reveal before a typed commitment. No mechanism or fix explained before the user's attempt.
2. Options only for direction; mechanisms, fixes and twists are free-form.
3. Line caps by turn type; every turn ends in an action for the user. Never "does that make sense?".
4. Declare the reveal type; the tutor bets on live runs only and never looks the answer up first.
5. Every archive number is quoted with `file:line` **and its baseline** (the R2-only slip above).
6. Right on the first bet and wrong on the twist is logged as an illusion, not a pass.
7. The user writes the takeaway. Misconceptions are kept verbatim and return in a new setting.
8. Evidence P / T / B / D only; owned = two kinds, two days, one of them T or B.
9. Katas are checked by execution; the tutor never writes in `no_AI_code_here/`.
10. Derivations start from a named primary-source section, marked read or recalled.
11. Widgets per FL4. 12. A session with no commitment and nothing user-made is not logged as progress.

**Ledger (`rl/LEARNING.md`)**

```
synced: a83a7a0 · last: 2026-11-27 check-in · next recall: "ICU ventilator weight −1.8: what is α here?"
calibration (counters)  you 50%: 4/9 · 70%: 7/11 · 90%: 5/6     tutor, live only 70%: 2/4
confident misses by theme   "more information helps" 0/3

## My map (session 1, the user's own words; 8 rows)
| era / file | know about the world | store | targets from | whose data, which opponent |

## Regions (≤ 14 rows)
| region                      | evidence                     | due        | note                         |
| horizon and reward          | P✗ 2026-11-27                | 2026-11-28 | said 0.95^40 ≈ 0.4, at 50%   |
| value fitting / confounding | D 2026-11-25, T 2026-11-28   | 2026-12-05 | owned                        |

## Fresh (≤ 5; appended by any agent during work)
- 2026-11-25 a83a7a0 phase 4: why did the better predictor play worse?
## Open bets (≤ 5)
## Misconceptions (≤ 8, verbatim; retire after 2 clean hits in a new setting)
## Log (≤ 12 lines, newest first; older lines fold into Regions)
```

Budget: header 3, map 10, regions 16, fresh 5, bets 5, misconceptions 8, log 12, headings and blanks ~13:
about 72 lines. Intervals 2 / 4 / 8 / 16 / 32 days, reset to 1 on a miss, the due date written in full.

**Ten minutes of checks before building:** symlink discovery in both harnesses · git on PATH in an
Antigravity session · lesson 1's real runtime · a three-record log in the real cockpit page · the
reachable-state count of lesson 1 for the exact-DP lab.
