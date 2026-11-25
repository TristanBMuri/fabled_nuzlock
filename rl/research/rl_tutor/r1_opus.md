# Round 1 — Opus: "The lab is the textbook"

**Thesis in one line:** don't build a course *about* RL and attach Pokémon examples. This repo
already is an RL course: four attempts, ~15 named decisions (D1–D14), and a dozen experiments
where the obvious answer was wrong. The tutor is a thin **on-demand skill** that turns that
history (and each new commit) into *bets, bug hunts and decision rehearsals*, places every one on
**one map of RL**, and keeps a plain-text ledger of what the user can actually do without help.
No scheduled pushes, no app, no new code beyond a prompt and a markdown file.

Sources: everything below was checked in the repo (README, RR_PLAN v3, NOTEBOOK phase 4,
`lesson1_qlearning.py`, `lesson2_ppo.py`, `fit_eval.py`, `line.py`, `no_AI_code_here/`) unless
marked *recalled* (literature I know but did not open this session).

---

## 1. Delivery: on-demand skill. No schedule at launch, one with a trigger later

**Verdict: an on-demand skill, invoked as `/rl-tutor [quick|lab|map|why <thing>]`.** A scheduled
task stays off until the ledger shows the habit failing (trigger below).

Why not scheduled-first:

- **The habit already exists.** The user opens T3 Code on this project nearly every day to
  steer the planner work. The cheapest habit loop has no new cue: *after a work session, one card*.
  A scheduled post adds a second cue that competes with the first.
- **Scheduled content goes stale.** A 09:00 post is written against the repo as it was at 09:00.
  This project moves fast: four phases landed on 2026-11-24 alone. A skill reads `git log` and
  `NOTEBOOK.md` when it is invoked, so "fresh from the lab" really is fresh.
- **Teaching needs several turns.** The core move is *predict → run → be surprised → explain*.
  A posted lesson is one turn the user reads and nods at, which builds the illusion of
  understanding rather than understanding.
- **Unread notifications pile up as guilt.** A thread with 9 unanswered "Daily RL" posts is the
  usual way self-study tools die. A skill nobody invoked leaves nothing behind.

**Opt-in schedule, with a trigger** (in the same spirit as RR_PLAN's "switch when" table): if the
ledger shows **two gaps of more than 10 days**, the tutor offers *one* weekly `schedule_task` that posts
**a single bet** (a question with a hidden answer, under 60 words) into **one bound thread**, not a new
thread per run. The bet only invites the user in; answering it means invoking the skill. Until
that trigger fires, the skill gets the habit from the work sessions (see §4, "fed by the work").

**Cross-provider footprint (one copy of the content, two ways to find it):**

```
.agents/skills/rl-tutor/SKILL.md          ~150 lines: the protocol (session flows, rules, grading)
.agents/skills/rl-tutor/references/map.md ~150 lines: the RL map, 14 nodes x 4 rungs, repo anchors
.claude/skills/rl-tutor -> ../../.agents/skills/rl-tutor     (symlink; git stores it fine)
rl/tutor_ledger.md                        state: rungs, due dates, misconceptions, bets, sessions
no_AI_code_here/rl_labs/                  the user's own lab code (the tutor reviews it, never writes it)
CLAUDE.md                                 +2 lines (see §4)
```

That is three text files and a symlink. No Python. SKILL.md is written tool-agnostic ("if an
inline HTML render tool exists, render; otherwise write the widget to the scratchpad and give
the path"), so Neon in Antigravity and Fable/Opus in Claude Code run the same protocol against
the **same ledger**. The user can switch provider in the middle of the curriculum and the mastery
record still holds. The skill's `description` frontmatter is what both providers match on, so it
names the triggers: "teach me RL, quiz me, explain this decision, why did X work".

Widgets are **generated per session**, not stored. A widget gets saved
(`references/widgets/<name>.html`, at most ~5) only after the user asks for the same one twice.

---

## 2. Curriculum: one map, four rungs, breadth before depth

### 2.1 The map (the big picture the user carries everywhere)

RL is easy to learn as a zoo of acronyms (DQN, PPO, MCTS, GAE, BC …). I'd teach it instead as
**one loop and three questions**, rendered once as an interactive atlas (`html_render`) and shown
again at the start of every lab with the user's rungs filled in:

> **Generalized policy iteration** (*recalled*: Sutton & Barto ch. 4.6): every method alternates
> *evaluate the current policy* ↔ *improve it*. Methods differ only in:
> 1. **What is known?** Model known (plan: DP, expectimax, MCTS) / learned (Dyna, MuZero) /
>    unknown (model-free: Q-learning, PPO).
> 2. **How is a policy evaluated?** Exactly (DP, `line.py`) · Monte Carlo returns · TD
>    bootstrapping · a fitted function (table → linear → net).
> 3. **How is it improved?** Greedy argmax · a gradient step on the policy · **search** as the
>    improvement step (MCTS) · **imitating** an improved teacher (BC, DAgger, expert iteration).
>
> Plus two frame questions: **whose game is it?** (a single agent in an MDP vs a game against an
> adaptive opponent) and **what can you see?** (full state vs a POMDP).

The project's own history is a walk across this map, and that walk is the first story the tutor
tells:

| Era | Point on the map | Outcome |
|---|---|---|
| Attempt #2 | model-free · PPO · from scratch · 6M Showdown steps | ~14% vs SimpleHeuristics |
| Attempt #3 | imitation (BC) + search over a known engine | 40% in ~1 h; search-all 52 → 95% clears |
| Lesson 1 | tabular TD control (Q-learning), exact expectiminimax as reference | 98.6% |
| Lesson 2 | policy gradient (A2C / PPO / REINFORCE) vs DAgger-style BC | PPO 74%, A2C 2%, BC 64% vs MCTS |
| RR phase 1 | the enemy is a known script, so this is an **MDP**: planning against a model | +30 pts |
| RR phase 1.5 | exact policy evaluation by tree expansion (`line.py`) | P(deathless) with bounds |
| RR phase 4 | Monte Carlo value fitted by regression, as the search's leaf | −2.4, then +11.0 |
| Lessons 3–4 | expert iteration → AlphaZero (search ↔ net, closing the GPI loop) | open |

Once the user can see that table as *moves across the map*, every detail has a place to go.

### 2.2 Four rungs per node (how breadth turns into depth)

| Rung | Name | Passed when the user can… |
|---|---|---|
| R1 | **Story** | explain the concept with its repo war story, unprompted, on a *later* day |
| R2 | **Predict** | bet on the result of an experiment before it runs, and be well calibrated |
| R3 | **Derive** | write the maths by hand (typed in chat), e.g. the Bellman optimality equation, the score-function gradient |
| R4 | **Build** | implement it from scratch in `no_AI_code_here/rl_labs/` against poke-engine, and it reproduces a repo number |

**Ordering rule: R1 across the whole map before R3 anywhere.** The first ~6 sessions take every node
to R1. Then the user picks where to dig, which is where the fun comes from. R4 is rare and
earns its cost: lessons 3 and 4 *are* the R4 capstones, written by the user and reviewed by the
tutor. The user's code imports the environment from `rl/lesson2_ppo.py` (`new_battle`, `step`,
`observe`, `legal`), so the folder stays true to its name: the algorithm is theirs.

### 2.3 The 14 nodes (each one: concept → war story → micro-lab → maths → beyond the repo)

1. **MDP and reward = the goal.** *War story:* attempt #2's `gamma=0.95` on 40-turn episodes
   shrinks the win signal to ~0.13, so the agent farmed shaping; D1 `U = win·(1−0.5·deaths/6)`;
   D14 "a death costs that mon's future value". *Lab:* lesson 1 with `--gamma 0.9`. *Maths:* return,
   discount as a horizon (1/(1−γ)). *Beyond:* **potential-based shaping** (Ng et al. 1999,
   *recalled*), the shaping that provably leaves the optimal policy unchanged. Was attempt #2's
   shaping of that kind? (No, and that's why it got farmed.) Reward hacking in RLHF.
2. **State, the Markov property, observability.** *War story:* the `"brn"` vs `"BURN"` key bug.
   Burned and unburned states shared one table row, so the table was *aliased* and the process
   it saw was not Markov: 97.4% → 98.6%, which flipped the comparison with always-Dragon-Claw.
   Lesson 2's obs ablation ("information made no difference on one fixed matchup, it is
   redundant"). *Beyond:* POMDPs and belief states; Foul Play's sampled guesses about hidden
   sets (determinization); why the real ROM run is a POMDP and the planner's full-information
   search is an approximation of it.
3. **Bellman equations and dynamic programming.** *War story:* `line.py` is exact policy
   evaluation by expanding the tree; lesson 1 prints expectiminimax next to Q. *Lab:* solve lesson 1's
   matchup exactly by memoized expectimax over `generate_instructions` (the opponent is random,
   so it's an MDP) and compare with Q-learning's +0.98 for Swords Dance at the start. (Feasibility at
   full depth is unverified; depth-limit if needed.) *Maths:* V\*, Q\*, the optimality equation,
   why the backup is a contraction.
4. **Monte Carlo vs TD: bias against variance.** *War story:* `fit_eval.py` labels every position
   with the battle's final U ("unbiased but noisy", its docstring); phase 4's option B was a
   TD-style difference target. Lesson 1's one line `q[s][a] += alpha*(target - q[s][a])` and the
   SARSA exercise. *Maths:* n-step returns, λ-returns. *Beyond:* eligibility traces.
5. **Function approximation and the deadly triad.** *War story:* table → 156k-parameter MLP;
   linear leaf vs a 1×32 net (net better on seen bosses, *worse* held out: 0.575 vs 0.535).
   *Beyond:* DQN, target networks, the deadly triad (*recalled*: S&B 11.3).
6. **Policy gradients → A2C → PPO → GAE.** *War story:* A2C **2%** vs PPO **74%** vs MCTS,
   same net and budget, because A2C takes one gradient step per 4096 samples. Attempt #2's
   `grad_clip=0.5` crushed gradients to ~1% and nothing was learned, silently. Entropy at 0.05. *Code:*
   `advantages()` (lesson2_ppo.py:366) and the clipped loss (:407). *Maths:* the log-derivative
   trick, why a baseline adds no bias, GAE as a λ-weighted sum of TD errors, the clipped surrogate.
7. **Exploration: ε, entropy, UCB, pUCT.** *War story:* phase 4's fitted evaluation halved the
   spread of root values (0.189 → 0.096), so a UCB constant tuned for the hand evaluation's scale
   explored more and committed less (visit share on the pick 48% → 29%). The fix was rescaling ×2 (+1.4
   pts). *Beyond:* bandits, regret, Thompson sampling, optimism under uncertainty.
8. **Planning and search.** *War story:* search-all 52 → 95% clears; the 2k / 20k / 200k ladder
   reads "search-bound"; crits branched only at depth ≤ 1 (`mcts.rs:175`); a death penalty of 200
   made the search too afraid to trade. *Beyond:* decoupled UCT for simultaneous moves, MuZero's
   learned model.
9. **Value fitting and confounding.** *War story:* the fit predicts outcomes much better (log-loss
   0.44 vs 0.69) and **plays worse** (−2.4 pts): 70% of the gain is *boss identity*, which the
   search cannot change. Per-`(boss, team)` fixed effects on `f(s) − f(s₀)` gave +5.9, and +11.0
   after the bench features and an R2 replay buffer. This is the deepest lesson in the repo: **search
   ranks moves, so only differences inside a battle matter. That is an advantage, not a value.**
10. **Opponent models and exploitation.** *War story:* searching against the buggy Rust port of the RR AI:
    **−33 ± 12 pts** at Misty (the search exploited the model's errors). The mixture `p=0.8` gave +11.
    Fixing the port (agreement 75 → 98%) gave **+30.2**. Lesson 2's policy beat the greedy script
    100% and MCTS 0–22%. *Beyond:* best response vs Nash, safe or robust exploitation,
    fictitious play / PSRO, self-play cycles.
11. **Imitation vs search vs self-play.** *War story:* BC got 40% in 1 h where PPO got 14% in 6M
    steps; an RL finetune oscillates around its BC init; expert iteration v1 gave nothing;
    `--algo bc` is DAgger. Numbrion (0.8M params, 10 h on an M1) vs Jaxcalibur. *Beyond:* AlphaZero,
    offline RL (Metamon), distribution shift.
12. **Statistical rigor.** *War story:* "2000 battles at 98% have an SE of ~0.3 points"; seed bands;
    paired replays; SPRT; held-out bosses; phase 4's three fit rounds, then an *untouched* seed 4
    for confirmation (+11.0 ± 4.1). *Beyond:* forking paths, Henderson et al. "Deep RL that
    Matters" (*recalled*).
13. **Hierarchy and the run layer** (*beyond the repo, coming next*): routing, catching, D14's
    run-level V(box, next bosses) are decisions made over battles, the way options work over
    actions. *Beyond:* options / hierarchical RL, the value of information.
14. **RL in LLMs** (*fully beyond*, but the user will meet it everywhere): RLHF / PPO with a KL
    to a reference, GRPO (RR_PLAN already lists it). Each idea maps back to a node above (KL
    penalty ≈ PPO's trust region; reward model ≈ a fitted value with all of node 9's confounding traps).

**How general RL gets taught where the repo has no example:** a **transplant**. The concept
gets rebuilt on a poke-engine toy in under 15 s of compute: Thompson sampling chooses among
Garchomp's 4 moves as a 4-armed bandit against the random Rotom; potential-based shaping goes
onto lesson 1; Dyna goes onto lesson 1's table. A transplant may only use a gridworld when there
is no Pokémon form at all. Each "beyond" ends with a **transfer question** set outside Pokémon
(robotics, Go, LLMs) to check that the concept, not the anecdote, is what stuck.

---

## 3. The fun engine and the guard against fake understanding

Fun here is **being surprised by your own predictions, about a system you care about**. Not
points. Five mechanics, all forced into the open:

1. **Bets with confidence.** Before any number is revealed, the user commits to a guess and a
   confidence ("PPO beats A2C by >30 pts, 70%"). The ledger keeps every bet, and the user's
   **Brier score and calibration curve** are the only scoreboard. It's a fitting one: the project's
   goal is *calibrated* win and death probabilities, and the user gets trained to be a calibrated
   value function too. Being confidently wrong is the most valuable event: errors made with high
   confidence get corrected best (the hypercorrection effect, *recalled*: Butterfield & Metcalfe 2001).
2. **Decision rehearsals ("you're the lead").** A real D# decision, with only the evidence
   that existed at the time: D4 opponent mode, D10 linear vs a 1×32 net, D11 phase 4 options A / B
   / C / D. The user decides and justifies, then sees what was chosen and what happened. The later
   commits are the answer key. This also prepares the user for the *upcoming* decisions (D13,
   D14), which CLAUDE.md says are theirs to make.
3. **Bug hunts from real history.** The tutor reinjects a historical bug into a **scratchpad
   copy** (never the repo file): `"brn"`, `grad_clip=0.5`, A2C's single epoch, the active-vs-bench
   `answers` bug, `gamma=0.95` with shaping. It shows only the *symptoms* (the curves, the Q table)
   and the user diagnoses. This is attempt #2's pain, relived safely in 5 minutes.
4. **Micro-experiments under 15 s.** Lesson 1 runs in ~10 s, so each lesson-1 exercise listed in
   the README (`--episodes 2000`, `--gamma`, always Will-O-Wisp, SARSA, seeds) becomes a bet. Lesson
   2 runs (minutes) go into labs only, one at a time under the RAM rule, watched in `rl/cockpit.py`.
   The cockpit's note field already asks "what do you expect?", so the tutor makes the user fill it in.
5. **Widgets you push on** (`html_render`, built from *real* logs): a GAE λ slider over a real
   lesson 2 battle from `logs/rl/*.jsonl`, advantages per turn; a UCB visit simulator seeded
   with phase 4's real root spreads (0.189 / 0.096) and a `c` slider; a Bellman backup stepper on
   lesson 1's tree. **Every widget opens locked: the user types a guess before the sliders come alive.**

**Rules against fake understanding (in SKILL.md, not optional):**
- **Retrieve before re-teaching.** Every session opens with a due card answered from memory. The tutor
  never explains a concept the user has seen before without first asking them to produce it.
- **Rungs pass on a later day only.** Same-session success is fluency, not mastery.
- **No prose wall over ~12 lines.** After that, a question, a bet or a widget.
- **The user writes the takeaway.** One sentence in their own words goes into the ledger. If it's
  wrong, it goes into *misconceptions*, verbatim, with a date.
- **Transfer questions** outside Pokémon before any R2+.
- **Assign the primary source.** For R3 the tutor points at the exact section (S&B ch. 13.3, the
  GAE paper §3, *recalled*) instead of paraphrasing it. A summary that reads well gets nodded at.

---

## 4. Session anatomy and state

### Step 0 (silent, every invocation, ~15 s)
Read `rl/tutor_ledger.md` → `distrobox-host-exec git log --oneline <last_hash>..HEAD` (git is
host-only in this box) → new `## ` headings in `NOTEBOOK.md` since the last session's date →
new D# lines in `rl/RR_PLAN.md`. A new commit or notebook entry becomes a **"fresh from the lab"**
card, tagged with its node on the map.

### Quick check-in (5–10 min) — `/rl-tutor quick`
1. **Due card** (2 min): one retrieval question from the Leitner queue. Graded pass / partial / fail.
2. **One fresh-from-the-lab bet** (4 min) on today's commit, or a 10-s lesson 1 run. Commit, reveal,
   one "why".
3. **Takeaway** (1 min): the user writes one sentence; the ledger is updated (rung, due date, bet).

### Lab (25–45 min) — `/rl-tutor lab [node]`
1. **The map** (1 min): the atlas widget with rungs filled in, and "you are here".
2. **Due cards** (3 min): two retrievals.
3. **Concept chunk** (5 min): ≤12 lines of prose, then a widget with a locked guess.
4. **Bet and run** (10 min): a micro-experiment or a real-history bug hunt.
5. **Climb a rung** (10–20 min): R3 derivation in chat, or R4 code in `no_AI_code_here/rl_labs/`,
   which the tutor reviews (comments in chat, never edits there).
6. **Transfer question + takeaway + ledger** (3 min).

Other entries: `/rl-tutor why <commit|phase|D#>` explains a real decision as a rehearsal, and
`/rl-tutor map` shows the atlas.

### The ledger (`rl/tutor_ledger.md`, the only state, maintained by the LLM)

```markdown
last_session: 2026-11-25  last_hash: a83a7a0
## Map            (node | rung | box | due | met in the wild)
9 value-fit/confounding | R2 | 2 | 2026-11-28 | phase 4 (a83a7a0)
## Misconceptions (date | verbatim | status)
2026-11-25 | "a better predictor of who wins is a better leaf eval" | fixed 11-25, recheck 11-28
## Bets           (date | claim | p | outcome)
2026-11-25 | fitted eval beats hand by ≥5 pts | 0.70 | no (−2.4)
## Sessions       (date | kind | minutes | one line)
```

**Spaced repetition with zero code:** Leitner boxes with intervals of 1 / 3 / 7 / 21 days. A pass
moves the card up one box, a fail sends it back to box 1. **The due date is written out explicitly**
so the LLM never has to do date arithmetic it could get wrong at read time. Calibration
(Brier score) is recomputed from the bets table when the map is shown.

**Fed by the work (the +2 lines in CLAUDE.md):** "When an RL concept is explained during project
work, append it to *met in the wild* in `rl/tutor_ledger.md`. At the end of a work session, if a
ledger card is due, offer it in one line." That's how the work habit drives the spaced
repetition without anything being scheduled. The RL decision-log doc stays the record of
*decisions*; the ledger is the record of *understanding*. Each rehearsal links the decision-log Step it comes from.

---

## 5. One sample check-in (quick, ~8 min, 2026-11-25, the day after phase 4)

> **User:** `/rl-tutor quick`
>
> **Tutor:** Due card (box 2, last seen 11-23): *Before 2026-11-22 Q-learning lost to "always
> Dragon Claw". Why, in one sentence?*
>
> **User:** the burn check was broken so it couldn't tell burned from not, so it learned an average.
>
> **Tutor:** Pass. "Learned an average over two states it couldn't tell apart" is the exact name
> for it: **state aliasing**. Node 2 → box 3, due 12-02.
>
> Fresh from the lab, `a83a7a0` ("Phase 4 done"). Back on 11-24 the first fit predicted battle
> outcomes far better than a constant (log-loss **0.44 vs 0.69**). Then it was dropped into the MCTS
> leaf in place of the hand evaluation. **Bet: win-rate change vs hand, paired, 5 bosses. A number and a
> confidence.**
>
> **User:** +5 pts, 70% sure it's positive.
>
> **Tutor:** **−2.4 ± 3.7, and deaths got worse.** You were confidently wrong, which is the best
> case: these errors stick. Why would a *better predictor* play *worse*?
>
> *[html_render: two root moves, values from phase 4's 192-position diagnostic. A slider for the
> evaluation's scale (hand spread 0.189 / fitted 0.096), a slider for UCB's `c`, 20k simulated visits.
> Locked until the user types a guess for "visit share on the best move under the fitted eval".]*
>
> **User** guesses 45%, unlocks, sees **29%** vs 48% for hand. Drags scale ×2 and the commitment
> comes back.
>
> **Tutor:** That's option A, which was tried: +1.4 pts. Real, but small. Now the twist. **70% of the
> fit's gain was boss identity.** Suppose the evaluation were *only* a per-boss constant. How often
> would it change which move MCTS picks inside a battle?
>
> **User:** …never? Every leaf in the battle gets the same constant.
>
> **Tutor:** Almost. To first order it cancels. Through the sigmoid it doesn't fully: a big constant
> pushes every leaf toward 0 or 1 and *squashes the differences*. That's the spread compression you
> just dragged. So the fit spent most of its power on something the search **can't use**, and paid for
> it in resolution. The fix that worked: fixed effects per `(boss, team)` on `f(s) − f(s₀)`,
> i.e. learn only what changes *within* a battle: +5.9, then +11.0.
>
> **Transfer:** where else do you subtract something that doesn't depend on the action, to keep only
> what does?
>
> **User:** the baseline in policy gradients? A2C's value baseline.
>
> **Tutor:** Yes. Node 6 just got a new edge to node 9. Write the takeaway in your own words.
>
> **User:** "Search compares moves within a battle, so the leaf only needs to rank positions inside
> one battle. That's an advantage, not who's winning overall."
>
> **Tutor:** Logged. Node 9 → R2 (bet made, mechanism explained; R2 passes on recall 11-28).
> Misconception logged as fixed, recheck 11-28. Calibration: 7 bets, Brier 0.21 (you're
> overconfident on "more information helps", 0/3). Next lab suggestion: R3 on node 6, deriving
> why a baseline adds no bias, which you just used without proving.

What the user gets from this session: a surprise, a hands-on mechanism, a link between two
distant nodes, and a sentence of their own. Eight minutes, and nobody read a paragraph about UCB.

---

## 6. What I would explicitly NOT build

1. **A daily scheduled lesson feed.** It goes stale, it can't be interactive, it piles up
   unread, and it competes with the habit that already exists. Allowed only after the trigger in
   §1, and then only as a single weekly bet into one thread.
2. **A tutor app, a "learning mode" in `cockpit.py` or `ui/`, or an SRS/quiz engine in Python.**
   Every one is the attempt #2 trajectory: infrastructure instead of learning. The ledger is
   markdown, the LLM is the scheduler, and widgets are thrown away unless asked for twice.
3. **XP, badges, streaks.** A streak turns a missed day into a reason to quit. The calibration
   curve is the only score, because it measures the one skill the whole project is about.
4. **Generated lecture notes / a homemade textbook.** Fluent prose is exactly how the illusion of
   understanding forms. Point to the primary source section, then test.
5. **Generic toy environments (CartPole, gridworlds, a gym wrapper) and tutor-written code in
   `no_AI_code_here/`.** poke-engine is faster than any gym env this user will set up, and every
   number from it can be checked against a real war story. The user's folder holds only their code,
   or R4 means nothing.
