# Round 2 — Opus: cross-critique and merged proposal

Read: `ROUND2_BRIEF.md`, `r1_neon.md`, `r1_fable.md`, `r1_opus.md` (no other r2 file). Repo facts below
were checked this session with read-only commands (cited file:line). Literature claims stay *recalled*.

**Short version:** Fable's design is the best of the three, and the merge should start from it. I'm
giving up my Brier score, my 14×4 rung grid, my global "breadth before depth" rule, my end-of-work
card and my "locked widget". I'm keeping the GPI caption on the map, the primary-source assignments,
the decision rehearsals from history, and explicit due dates. Neon contributes the transfer questions
(the ventilator question is the best one in any of the three docs) and the "Why not here?" shelf as
material for the transfer questions, not as a separate track.

---

## 1. What I'm stealing from the other two

**From Fable:**
- **Fable's map columns replace my three GPI questions.** "Where do targets come from?" separates TD,
  MC, GAE, the teacher's move and visit counts more cleanly than my "how is it evaluated?". The fifth
  question, "how do you know it worked?", is the right home for statistics. I keep GPI only as the
  one-line caption over the table: every row is a way to alternate evaluate ↔ improve.
- **The user fills in the table in Session 1.** Writing an answer and having it corrected sticks
  better than reading a finished one (the generation effect, *recalled*). My atlas and Neon's
  rendered map are both things the user only looks at.
- **P / B / D evidence ("owned at two kinds on two different days")** replaces my 56-cell rung grid.
  It records what the user did, not a label someone assigned, and D gets filled during ordinary
  project work.
- **Hook A:** at a real D#, the user states a call and a confidence before seeing the
  recommendation. This is the single best idea in the panel. It costs about 30 s, happens only at
  D#s (rare by nature), and makes the existing CLAUDE.md rule ("let the user decide") concrete.
- **Confidence buckets of 50 / 70 / 90, kept as hit counts per bucket.** "Of your 70% bets, 7/11
  came true" can be read at a glance. A Brier score of 0.21 cannot. Fable's version is also
  stronger than mine: confidence goes on the one main bet of the session only, not on every question.
- **NOTEBOOK.md as a held-out set:** show the situation as of a date, hide what came after. That
  gives unlimited bets with no compute, all from the user's own history.
- **Sabotage, diagnosed from the cockpit curves alone.** This is debugging as it actually works in
  practice, which my "bug hunts" only approximated.
- **Fable's catch that my own Round 1 was wrong.** My node 1 says "attempt #2's `gamma=0.95` on
  40-turn episodes". CLAUDE.md:86 puts that failure under **attempt #3**, and CLAUDE.md:102 lists
  0.95 as a value that worked in attempt #2 (25-turn battles). Fable's §5 makes that contradiction
  into a twist question, which is better than any fix I'd make. The skill must ship the corrected fact.
- **The kata bot that keeps growing** (table → GAE → UCB → 40-line MCTS) and plays lesson 1's
  opponents. It is more motivating than separate katas, and it ends naturally in lessons 3–4, which
  were my R4 capstones.
- **The tutor never shows the size of the backlog.** After a three-week break the user gets one
  question, not "14 cards due".

**From Neon:**
- **Transfer questions with real stakes outside Pokémon:** the ICU ventilator weighted −1.8 turns
  "confounding by indication" into something the user won't forget. Every case file should end at
  that level, not with "where else do you subtract a baseline?" (my sample).
- **The "Why not here?" shelf** (SAC, RND / Go-Explore, Dreamer / MuZero, IQL / CQL / DPO), grounded
  in RR_PLAN's rejected methods. It is the right content for Fable's "foreign territory" sessions. I'd
  add no new structure for it.
- **The Feynman close, kept to 2 sentences.** It is the same move as "the user writes the
  takeaway", but with a length limit, which is what keeps it honest.
- **Pre-decision rehearsals for D14 and lesson 3.** Mine covered past D#s only. Neon points them
  forward, and forward is where the stakes are.

## 2. Sharpest critiques and pushback

**Neon: over-built, and the sample isn't a bet.**
- Count the structure: 7 arcs (0–6, although the text says "6"), 24 nodes, 4 zoom depths, 4 statuses,
  5 bespoke widgets, a clickable mastery map, a 4-item shelf and a teaser cron. That is an LMS written
  in Markdown. The **clickable mastery map is a progress dashboard**, which all three of us rejected
  in §6. It needs a renderer kept in sync with the ledger on every session, and the minimal-code rule
  exists to stop exactly that.
- **The sample bet gives its answer away.** Options A–D, where B is the only one that mentions bosses
  and is worded as the explanation. There's no confidence, nothing the user is likely to get wrong,
  and so no surprise. The tutor then explains confounding for about 20 lines with no twist. That is
  the nodding-along this design is meant to prevent, presented as a session.
- **The widget data is made up.** "Brock [85% base] and Koga [15% base]" is not a number from
  phase 4, and Koga is not in the phase 4 boss set as far as the docs show. A scatter that *looks*
  like repo evidence and isn't teaches the wrong thing. **Rule: a widget plots real logged numbers,
  or it carries the visible label "toy numbers".**
- **"intuited (engaged with visual/example)" is a status the user earns by having watched
  something.** That is a fluency label in a ledger meant to measure the opposite.
- **The teaser cron with no gate** is close to fine (it is self-silencing), but see Fault Line 3.

**Fable: mostly right. Three places to tighten.**
- **The tutor's bet is fake on any reveal from NOTEBOOK or the logs.** The model either has read the
  answer or can find it with one grep. A tutor that "loses in public" while it holds the answer key is
  theatre, and the user will notice. **Restrict the tutor's bet to runs nobody has made yet**
  (`lesson1 --gamma 0.9` in Fable's sample is a good example). Also the tutor is a different model
  instance, and sometimes a different provider, each session, so "the tutor's calibration" mixes Neon,
  Fable and Opus. Label each tutor bet with the model, or keep a single row and call it "the house".
- **The Session-1 table has 32 empty cells.** That is a worksheet, not a hook, for someone who has
  only run lesson 1 and lesson 2. The user should fill **three rows** (lesson 1, `--algo ppo`,
  `engine_search`), and each later session adds the row for what it covered. The table grows the
  way the user's knowledge grows.
- **6 lines is right for the prompt, too tight for a reveal.** A reveal has to carry the number,
  its error band and the one-line mechanism. Fable's own reveal turns run to about 8 lines. Make it
  6 lines before the user must act, and up to 10 on a reveal turn.
- Smaller: **"every 4th session is foreign territory" is a schedule.** Make it a counter: "if
  `foreign: 3` sessions since, offer it". That gives the same cadence and skips cleanly when a fresh
  commit should come first.

**Opus (myself): what I'm withdrawing, and why.**
- **The Brier score** (above). **The 14 × 4 rung grid** and **"R1 everywhere before R3 anywhere"**:
  six sessions of stories only for an engineer who has already shipped PPO in lesson 2. Fable's rule
  that "fresh commits come first" is more fun, and a local prerequisite (below) keeps the useful part.
- **"Every widget opens locked until the user types a guess."** The `html_render` tool shows a
  finished page in an iframe and has no channel back to the agent (see its tool schema, which only
  takes `html|path, title, height`). A guess typed into the widget is a guess the tutor never sees,
  grades or logs. Fable's assumption is right, and my mechanic fails. Its intent survives in a different
  form: **the user bets in chat, and the tutor draws that guess into the widget as a ghost marker**,
  so the slider shows how far off the guess was.
- **The end-of-work card.** The working agent can't tell when a session has ended, and a quiz offered
  while the user is pushing phase 6 is friction. "Met in the wild" logging duplicates what the git sync
  already finds. Both go.
- **The bets table in my ledger grew without limit.** Once a bet is settled it should collapse into
  the bucket counters.

## 3. Position on all six fault lines

**FL1. Confidence and calibration: keep them, but only on the single main bet.**
One confidence (50/70/90) per session, on the bet that has a reveal. The cold question and the twist
get a plain answer, so the bookkeeping costs one word per session. Calibration is shown as bucket hit
counts, never a Brier score. Neon's version with no confidence loses two things. The first is
hypercorrection: a miss stated at 90% is the one the user remembers. The second is that the skill
being trained is the project's own goal, "trustworthy win/death probabilities". The tutor also bets,
**only on reveals that haven't been run**, labelled with the model, and settles in the same message.

**FL2. Entry point, ordering and statistics.** Session 1 hands the user an empty map with **three
rows** and the file list, and the user fills them in (Fable's move, made smaller). It is not a
rendered map, which is Neon's version. Ordering: **fresh result > something due > the default
path** (Fable). One prerequisite, enforced locally: **no derivation or kata in a region until the
user has filled that region's map row and made one P there.** That keeps what my breadth rule was
for (a place for each detail before it arrives) without the global gate. Statistics: **woven into
every reveal, and also a few named case files**. Every reveal carries its band (the fifth
column). SPRT, the optimizer's curse (best of K, then a fresh seed 4) and paired common random
numbers each get a Researcher-pass case file, because "± band" alone never teaches why picking
p = 0.8 out of four values needs a fresh seed. Statistics gets no separate arc.

**FL3. Hooks and the schedule.** **Hook A only**, worded so it never blocks: "the user's call +
confidence, or `pass`". A D# happens perhaps once a week, so 30 s there is not friction, and it
produces D evidence with no tutor session at all. Neon's "fresh case file" append and my "met in
the wild" both duplicate the git/NOTEBOOK sync, so drop them. **Schedule: the user decides, and the
tutor brings it up.** It is off by default. If the user asks, it gets set up at once (no gate stands
between the user and their own tool). The tutor *brings it up* exactly once, when the ledger shows two
gaps ≥ 10 days. The format is the same whichever way it starts: weekly, one bound thread, one cold
question of 2 lines, skipped while `card: open` is set in the ledger. Neon's "read the thread; if no
reply, stay silent" and Fable's `card: open` are the same mechanism. The ledger flag is cheaper and
works across providers.

**FL4. `html_render` discipline.** The tool shows a page and returns nothing, so: **the bet is always
made in chat, and a widget is only ever a picture drawn after the bet.** Budget: **at most one widget
per check-in**, and only when there is a continuous knob worth dragging (γ, λ, UCB `c`, value scale,
the mixture p). Discrete results stay as text. Keep it to one tool call: `html_render` with an inline
`html` and a fixed `height` (charts take fixed pixel heights anyway), and skip `html_preview` in
check-ins. Every widget plots real logged numbers or is labelled "toy numbers", and has the user's
chat guess drawn in as a ghost marker. A widget is saved to `references/widgets/` only after the
**user** has asked for it twice. The tutor regenerating it doesn't count.

**FL5. Mastery, line cap, code.** **P / B / D, owned at two kinds on two different days.**
B covers anything built by hand: a kata, or a derivation typed in chat and checked line by line.
That absorbs my R3. A primary-source assignment (S&B section, GAE paper §3, *recalled* numbering)
comes with every B that is a derivation, so the tutor does not paraphrase. **Line cap: 6 before the
user acts, 10 on a reveal.** The Feynman close is 2 sentences. **Code:** one growing bot in
`no_AI_code_here/rl/`. It imports the environment from `rl/lesson1_qlearning.py` / `rl/lesson2_ppo.py`
(`new_battle`, `step`, `legal`), and the algorithm is the user's. It is checked by running it against the
repo's reference (`lesson1:82`, `lesson2_ppo.py:366` `advantages`, `:407` the clipped loss,
`fit_eval.py:77` `fit_within_team`). The checker is written in the scratchpad, never in the repo, and the
tutor reports pass/fail and the first mismatch, nothing more. Lessons 3–4 are where the bot ends up,
not a separate R4.

**FL6. Repo-accuracy traps, checked this session.**
- **git works inside this box now**: `git --version` → 2.56.1 at `/usr/bin/git`, and
  `git log -1` → `a83a7a0`. Fable, my Round 1 and the memory note all say `distrobox-host-exec git`;
  that is stale for this box. SKILL.md should say: "use `git`; if it's missing, `distrobox-host-exec
  git`". Antigravity's environment is unverified.
- **`.agents/` and `.claude/skills/` do not exist yet** (`.claude/` holds only settings files). It is
  unverified whether Claude Code follows a *directory* symlink when discovering skills. Ship the
  directory symlink, test one invocation in each harness, and if one fails, fall back to Fable's 5-line
  stub. Neon's per-file symlinks add nothing over that.
- **`8240c1c` exists and holds the bug**: `git show 8240c1c:rl/lesson1_qlearning.py` line 58 has
  `status.lower() == "brn"`. Fable's resurrection works as written.
- **Lesson 1's reference is worst-case, not expectimax against the random Rotom.**
  `lesson1_qlearning.py:116` prints `iterative_deepening_expectiminimax(...).get_safest_move()`
  ("opponent plays its best"). My "solve it exactly and compare with Q" lab has no reference
  function, so it becomes a B kata (memoized expectimax over the random foe) and its feasibility at
  full depth is still unverified. `--gamma` exists (`:104`, default 1.0), so Fable's sample run works.
- **The cockpit can plot a hand-written bot, with care.** `cockpit.html:186` defaults the config to
  `{}` and plots `kind == 'eval'` lines shaped `{"vs": {opp: {"win", "n"}}}` (`:134`). The x-axis
  goes through `X(e, start)`, so write a minimal `start` line too. `cockpit.py:44` marks a run
  "running" only if `lesson2_ppo.py` is in its cmdline, so a hand bot shows as done or stopped, which
  is fine.
- **NOTEBOOK.md is 212 KB / 3177 lines.** Read headings with `grep -n '^## '` and only the cited
  section, never the whole file.
- **Line refs checked:** `lesson1:60-64` `key()`, `:82` "the whole algorithm is this line",
  `lesson2_ppo.py:366/:407/:521`, `fit_eval.py:80/:94`, `CLAUDE.md:86` (attempt #3 γ) — all correct
  in Neon's and Fable's docs.
- **Due dates:** compute them with `date -d '2026-11-25 +8 days' +%F` at write time and store
  them written out. That is exact, costs zero code, and settles the date-arithmetic worry both of us
  raised.
- **≤ 80 ledger lines** holds only if every section has a limit: settled bets fold into counters,
  the log keeps 15 lines, misconceptions retire after 2 clean hits, and the map has one row per region
  (about 12), not one per concept.

## 4. Proposed merged architecture

**Files (3 authored, 1 symlink, 1 state file, +2 lines in CLAUDE.md, 0 lines of Python):**
```
.agents/skills/rl-tutor/SKILL.md            ~110 lines: protocol, rules, flows, compute limits
.agents/skills/rl-tutor/references/map.md   ~130 lines: 4+1-column map, 12 case files, shelf
.claude/skills/rl-tutor -> ../../.agents/skills/rl-tutor   (stub SKILL.md if discovery fails)
rl/LEARNING.md                              learner state, ≤ 80 lines, hand-editable
no_AI_code_here/rl/                         the user's growing bot; the tutor never writes here
CLAUDE.md "Working with the user on RL decisions"  +2 lines: Hook A
```
Invocation: `/rl-tutor` (check-in), `/rl-tutor lab [region]`, `/rl-tutor map`,
`/rl-tutor why <D#|commit>` (a decision rehearsal, past or upcoming).

**Curriculum backbone.** Fable's map with caption *"every row is generalized policy iteration:
evaluate ↔ improve"*: columns are **world knowledge · what's stored · where targets come from · whose
data / which opponent · how we know it worked**. Fable's **12 case files** are the regions. Each one
has the same arc: symptom from this repo → bet → twist → (lab) kata or derivation with a primary
source → a transfer question at Neon's level. Three passes (Tourist / Mechanic / Researcher) mark
depth inside each region. They are not phases of the course. Neon's shelf supplies the
foreign-territory content, offered when `foreign ≥ 3`.

**Check-in (5–10 min).**
1. A cold question from `LEARNING.md` alone, asked before any tool call. Git and NOTEBOOK sync runs
   while the user thinks.
2. **The bet** (fresh > due > default): the user commits answer + 50/70/90, and the tutor bets too
   only if the run is new.
3. Reveal: a ≤ 15 s single-core run, or a logged result with its path, always with its band.
4. Twist: one variation, predicted, then reasoned or run. Optional single widget.
5. A 2-sentence takeaway in the user's words, corrected until exactly true. The user names its map
   cell. Update 3–5 ledger rows. Stop.

**Lab (25–40 min).** Cold question → case briefing (symptoms only) → investigation (the user orders
evidence, names a suspect at a confidence) → B: kata in `no_AI_code_here/rl/` checked by running it, or
a derivation with the primary-source section assigned → transfer question or a forward D# rehearsal
→ file it. Lesson-2 runs go only here, `--minutes ≤ 3`, one job, check `MemAvailable`, or queue
in the cockpit as an open bet that settles at the next invocation. Planner tables are never run
for teaching; reveal them from `results/`. Ports 8000–8007 are off limits.

**Rules in SKILL.md (the anti-illusion core).** No explanation before a committed prediction ·
6 lines before the user acts, 10 on a reveal · never "does that make sense?", always a twist ·
the user writes the takeaway · misconceptions are kept verbatim and come back in a new setting until
hit cleanly twice · a session with no prediction and nothing the user made is not logged as progress
· the backlog count is never shown · every external claim is marked read or *recalled* · widgets
come after the bet, use real or labelled numbers, at most one per check-in · the tutor never edits
`no_AI_code_here/`, never fixes the kata, never runs heavy compute to make a point.

**Ledger schema (`rl/LEARNING.md`).**
```
synced: a83a7a0 · last: 2026-11-27 check-in 8 min · card: none · foreign: 2
calibration  you: 50% 4/9 · 70% 7/11 · 90% 5/6   house(opus/fable/neon): 70% 6/11 · 90% 8/9

## Map (one row per region; P predicted a variation · B built by hand · D used in a real D#)
| region                     | row filled | evidence             | due        | note                    |
| 11 value fit / confounding | yes        | P 11-25, D 11-25     | 2026-12-02 | B pending: fit_within   |
| 3 horizon & reward         | yes        | P✗ 11-27             | 2026-11-28 | said 0.95^40 ≈ 0.4      |

## Open bets (settle at next invocation, then fold into calibration)
- 11-27 cockpit `bet_ent005_s0..2`: you "0.05 > 0.02 by >5 pts vs MCTS" 70% · house "in band" 60%

## Misconceptions (verbatim · date · clean hits 0/2)
- 11-27 "a lower γ makes the agent more aggressive" 0/2

## Log (newest first, keep 15; older lines fold into the map)
```
Spacing: 2 / 4 / 8 / 16 / 32 days. A pass doubles the interval, a miss resets it to 1 day and
writes a misconception. Due dates come from `date -d`. At most 1 cold question per check-in and 2 per lab.
`LEARNING.md` records what the user understands; the decision-log doc keeps recording decisions.
A D row carries the Step number, and neither document copies the other.

**Hook A (the +2 CLAUDE.md lines):** "At a real D#, after options and basis: ask for the user's
call and confidence (50/70/90) or `pass`, *then* give the recommendation. Append one `D` evidence
line to `rl/LEARNING.md`."

**Schedule:** off by default. Set up immediately if the user asks. The tutor brings it up once,
when there are two gaps ≥ 10 days. It is weekly, posts to one bound thread, sends a 2-line cold
question, and skips itself while `card: open` is set.

**Verify before building (10 min):** the symlink is discovered in both harnesses · the git
command in Antigravity · a hand bot's `start` + `eval` lines render in the cockpit ·
whether `html_render` with a fixed height and no `html_preview` call sizes correctly in a thread.
(Do *not* pre-run `lesson1 --gamma 0.9`: it is the first honest house bet.)
