# Round 1 — Fable: the tutor is a betting table on your own project

Read for this: `CLAUDE.md`, `rl/README.md`, `rl/RR_PLAN.md`, `rl/lesson1_qlearning.py`, `rl/fit_eval.py`,
`rl/lesson2_ppo.py` (`collect` / `advantages` / `update`), `rl/cockpit.py` and `rl/line.py` (docstrings),
NOTEBOOK.md phases 1 and 4, `no_AI_code_here/`, `logs/rl/`, git history of lesson 1. Nothing was run.
Learning-science and RL-paper references below are **recalled, not read** in this session.

**Thesis.** The user does not need a course. They have a 3,177-line notebook of experiments with known
outcomes, a simulator that settles a question in 10 seconds, and real decisions (D14 is pending) that
change the project. The tutor's whole job is to make them **commit a prediction before every reveal**,
score it against the noise floor, and let the misses pick the next topic. Fun = an uncertainty you care
about × fast resolution × your call having consequences. All three are already in the repo.

---

## 1. Delivery: an on-demand skill plus event hooks. No clock.

**Verdict: on-demand skill. The "hybrid" half is triggered by project events, not by a schedule.**

| Concern | Recurring scheduled task | On-demand skill |
|---|---|---|
| Multi-turn | fires with no turn active: it can only post a monologue | every step waits for the user's move |
| Commitment before reveal | impossible: the post contains question and answer, or a question nobody answers | native |
| Stale posts | 3 missed days = 3 unread cards = guilt, then mute | none; state waits silently |
| Habit cue | a notification seen only when T3 Code is opened, which is when the user came to work anyway | the work itself (below) |
| Fits 5 min *and* 30 min | fixed size | the user says which |

A schedule adds no cue that opening the project does not already give. The cues that do exist:

- **Hook A, decisions (2 lines added to `CLAUDE.md` → "Working with the user on RL decisions").** Today:
  options → basis → recommendation → user decides. New order: options → basis → **the user's call and a
  confidence (50 / 70 / 90), or "pass"** → recommendation → decision. One line goes to `rl/LEARNING.md`.
  Every D# becomes an exam with real stakes, at a cost of 30 seconds, inside work that happens anyway.
- **Hook B, fresh results.** At the start of a tutor session the skill diffs git and NOTEBOOK headings
  against `synced: <hash>` in the state file. A new experiment the user has not predicted is the first
  candidate for the next bet, because its context is still warm.
- **One allowed use of `schedule_task`: a one-shot, user-requested bet settlement.** A bet on a run longer
  than the session (3 seeds × 6 min in the cockpit queue) may get a one-shot "your bet settled" post to the
  tutor thread. One-shot, so nothing can pile up. Default is cheaper still: open bets settle at the next
  invocation.
- **Recurring schedule: only through a failed gate that names it** (the repo's own rule for deferred
  methods). Gate: after 3 weeks, two gaps ≥ 10 days in `LEARNING.md`. Then one weekly card = one cold
  question, 2 lines, to a bound thread, and the task **skips itself while `card: open` is set** in the state
  file. At most one unanswered message can ever exist.

**Cross-provider.** Both harnesses read the same open format (`SKILL.md`, YAML `name` + `description`), so
there is one canonical copy and the state is a plain file in the repo. It must **not** live in Claude's
memory directory: Neon cannot read it, and the learner's state has to survive a provider switch mid-week.

```
.agents/skills/rl-tutor/SKILL.md           ~100 lines: the protocol (§3 rules, §4 flows, compute limits)
.agents/skills/rl-tutor/references/map.md  ~120 lines: the map + case-file table (§2), loaded on demand
.claude/skills/rl-tutor  ->  ../../.agents/skills/rl-tutor      (symlink; fallback: a 5-line stub SKILL.md
                                                                 that says "follow .agents/skills/rl-tutor")
rl/LEARNING.md                             state: one hand-editable file, ≤ 80 lines (§4)
no_AI_code_here/rl/                        the user's katas; the tutor never writes here
CLAUDE.md                                  +2 lines (Hook A)
```

Three authored files, one symlink, one state file. No `scripts/`, no widget library, no quiz bank.
Invocation: `/rl-tutor` (10 min, tutor picks), `/rl-tutor lab` (30 min), `/rl-tutor <topic or file>`,
`/rl-tutor map`. In Antigravity the description triggers on "teach me / quiz me / why did we …".

---

## 2. Curriculum: one map, three passes, twelve case files

### The map: four questions that place every RL method, plus one that judges them

| | 1. What do you know about the world? | 2. What do you store? | 3. Where do targets come from? | 4. Whose data, which opponent? |
|---|---|---|---|---|
| `lesson1_qlearning.py` | sampler only | table Q | TD, `max` (bootstrapped) | off-policy (ε-greedy collects, greedy is learned); random foe |
| `lesson2_ppo.py --algo ppo` | sampler only | net π + V | GAE: λ slides between TD and Monte Carlo | on-policy; `--opp` lever |
| `--algo bc` | sampler + a search teacher | net π | the teacher's move | the student's own states (DAgger) |
| `engine_search.py` | exact model | nothing | none: MCTS + UCB at play time | enemy = model with p = 0.8, else worst case |
| `fit_eval.py` | — | linear V, 46 weights | Monte Carlo outcome U | logged battles of older policies (R1 + R2 = a replay buffer) |
| `line.py` | exact model + enemy script | nothing | exact Bellman backup on a policy tree | none: policy evaluation, no sampling |
| `rr_route.py`, D14 | matchup scorer | V(box, bosses ahead) | — | the run level: the nuzlocke's real objective |
| lesson 4 (planned) | exact model | net π + V | the search's visit counts | self-play |

**5th question, asked of every row: how do you know it worked?** (noise floor, pairing, held-out, fresh seed.)

Session 1 hands the user the empty table and the file list; **they** fill it, the tutor corrects. From then
on every session ends with "where on the map was this", and every new method from outside the repo
(DQN, SAC, MuZero, CFR, GRPO) is taught by first placing it in the four columns.

### Three passes over the same map (a spiral, not a ladder)

1. **Tourist** (~8 check-ins). One war story and one bet per region. Exit: places any method on the map
   and predicts the *direction* of an effect.
2. **Mechanic** (labs). Derives and hand-writes the one equation of each region, ≤ 15 lines, checked
   against the repo's implementation on real data. The user's old code (`no_AI_code_here/players/policy/
   self_play.py`) wraps SB3's PPO: they have driven PPO, not opened it. This pass opens it.
3. **Researcher**. Designs the experiment, sets the gate before the run, makes the D# call first, reads one
   paper section and maps it onto the repo.

The order inside a pass is not fixed: fresh project results first, then what is due, then the default path.

### Case files (each: a symptom from this repo → bet → kata → transfer)

| # | Region | Repo anchor (the symptom shown first) | Bet (live ≤ 60 s, or from logs) | Kata, by hand | Beyond the repo |
|---|---|---|---|---|---|
| 1 | MDP, value, Bellman | `lesson1:82` "the whole algorithm is this line"; `key()` | `--episodes 2000` → 94%: what does the unconverged table get wrong? | the TD update; SARSA variant | DP when the model is known (`line.py` is exactly this) |
| 2 | State = what you can tell apart | `"brn"` vs `"BURN"` (buggy file: `git show 8240c1c:rl/lesson1_qlearning.py`): Q-learning 97.4% **below** always-Dragon-Claw 98.0% | which of 3 suspects; one test allowed | fix `key()`; then break it differently | POMDPs, hidden sets (`--obs foe_moves`), frame stacking |
| 3 | Horizon and reward | attempt #2/#3 γ = 0.95; `--shape`; D1's U; `POKE_ENGINE_DEATH` 0 / 60 / 200 ("too afraid to trade") | `--gamma 0.9` (§5) | discounted return; potential-based shaping term | Ng et al. 1999 shaping; CVaR for "deathless" |
| 4 | TD vs Monte Carlo, bias vs variance | `advantages()` `lam`; `reinforce` vs `a2c`; `fit_eval` docstring "unbiased but noisy"; D9 | λ = 0 / 0.95 / 1: order the explained-variance curves | GAE, 12 lines, vs `lesson2_ppo.advantages` on a saved buffer | n-step, eligibility traces, target networks |
| 5 | Policy gradient → PPO | a2c **2%** vs ppo **74%**, same net, same 12 min; `--grad-clip 0.5` | which cockpit curve shows a2c's problem first | the clipped surrogate; proof that a baseline adds no bias | TRPO, SAC, continuous actions |
| 6 | Exploration | ε decay; `--ent` (attempt #2's "#1 issue"); UCB in `mcts.rs`; phase 4: root spread 0.189 → 0.096, visits on the pick 48% → 29% | rescale logits 2×: does play improve? (+3.2 ± 3.2 vs H1) | UCB1 pick for a 4-arm bandit | regret, Thompson sampling, pUCT, best-arm identification (`rr_search` halving) |
| 7 | Who you train against | greedy-only: 100% vs greedy, **0–22% vs MCTS**; fixed teams ~75% → pool 16–23%; the 94% pool-4 checkpoint at 0 / 20 unseen | add MCTS-100 to the opponents: what vs MCTS-1000? (70–88%) | a held-out split that cannot leak | self-play cycles, league training, PLR / ACCEL |
| 8 | Imitation and expert iteration | BC 40% in 1 h vs PPO 14% at 6 M steps; bc 64% (peak 77%) on half the decisions; "finetune from BC oscillates"; search-all 52 → 95% | does a 1000-iteration teacher make a better student? | cross-entropy to a teacher; why DAgger relabels | DAgger's error bound, offline RL (Metamon), AlphaZero |
| 9 | Search | 2k / 20k / 200k ladder; crits branched only at root and depth 1; decoupled UCB for simultaneous moves | branch crits at every depth: gain? (+1.3 ± 3.6: none) | a real turn as a 4 × 4 matrix game, solved by regret matching | Nash vs best response, CFR, MuZero |
| 10 | Opponent models | following the port: Misty **−33**; oracle +12.2; off-policy agreement 90%, on-policy 70.5%; p = 0.8 → +11; port fixed to 98% → **+30.2** | rank p ∈ {0.5, 0.8, 0.95, 1.0} before seeing the table | per-node mixture in 8 lines of pseudocode | model exploitation in model-based RL, restricted Nash response |
| 11 | Value fitting and confounding | log-loss 0.69 → 0.44, play **−2.4**; a per-boss constant alone reaches 0.506; logit spread 4.5 between battles, 1.4 within; fixed effects → +5.9; R2-only −5.2 vs R1+R2 +11.0; 1×32 net better on seen, worse held-out | will the better predictor play better? will dropping round 1 help? | within-group logistic fit on `f(s) − f(s₀)` vs `fit_within_team` | advantage vs value, Simpson's paradox, fitted Q iteration, forgetting and replay |
| 12 | The run level | D14: a death costs that mon's value over the bosses ahead; lesson 4 | the user's own D14 call, before the recommendation | c_i for one box by hand from the archive | hierarchical RL, options, long-horizon credit |

**Statistics is not a module. It is the scoring rule of every bet** (the 5th question): the reveal always
comes with its band, a difference inside the band is logged as "a lead, not a result" (the README's own
phrase), and the repo supplies the cases: 98.6 vs 98.0 at SE 0.3; the cockpit's seed bands; `sprt.py`;
picking p = 0.8 out of four and re-running on fresh seeds; held-out bosses; NOTEBOOK's wrong "no
turn-limit losses".

### Beyond the repo (general RL, not Pokémon tricks)

- **Every case file ends with a transfer question that leaves Pokémon**: "CartPole pays +1 every step for
  500 steps at γ = 0.99. Does attempt #3's disease appear? Why not?"
- **Every 4th session is foreign territory**, taught on the smallest honest host:
  deadly triad / DQN → lesson 1 with a linear Q over `key()`'s four features (~25 lines, inline);
  Nash, exploitability, CFR → the 4 × 4 payoff matrix of one real turn from `generate_instructions`;
  offline RL → the 196k logged positions: why not just argmax a Q fitted on them (the R2-only story);
  learned world models (Dreamer, MuZero) → phase 1 as the analogue: search steers into model error;
  SAC / continuous control and GRPO / RLHF → no host in this repo, so: one section of a primary source,
  placed on the map, and one question back ("why can poke-engine's MCTS not act here?").
- **Spine for depth**: Sutton & Barto 2e chapter per region (2 bandits, 3–4 MDP / DP, 5 MC, 6 TD, 8
  planning and MCTS, 11 off-policy approximation, 13 policy gradient: recalled numbering), Spinning Up
  for PPO. The tutor says "read" or "recalled" for every external claim, as CLAUDE.md already demands.

---

## 3. The fun engine, and why nodding along cannot pass

**What makes it fun:** a reveal you have money on. Four mechanics, no points:

1. **Bets, and the tutor bets too.** The user commits (answer + 50 / 70 / 90). Then the tutor commits its
   own prediction in the same message, *before* running. Then the engine decides. This repo's results
   surprised its own authors repeatedly ("information made no difference", "branching crits: no gain",
   "a better predictor is not a better guide"), so the tutor will lose bets in public. The only score kept
   is a **calibration row for each of us**: of your 70% bets, how many came true. That is not decoration:
   "trustworthy win / death probabilities" is this project's stated goal for the battler, and calibrated
   prediction is the actual skill of an RL practitioner.
2. **Whodunits from real history.** Symptom first (numbers only), the user orders evidence, the tutor is the
   lab. Buggy code is resurrected from git into the scratchpad, never the repo
   (`git show 8240c1c:rl/lesson1_qlearning.py`). Later, **sabotage**: the tutor plants one bug in a scratch
   copy of `lesson2_ppo.py` (`done` not cutting the bootstrap, advantage normalised per minibatch, logp of
   the wrong action, `--grad-clip 0.5`) and the user diagnoses it **from the cockpit's curves alone**
   (entropy, KL, clip fraction, explained variance). That is how the skill is used in real life.
3. **NOTEBOOK.md as a held-out set.** Hundreds of experiments with outcomes the user has not memorised.
   The tutor presents the situation as of a date, hides what came after, takes the bet, reveals from the
   file with its path. Zero compute, unlimited supply, and each one is their own project's history.
4. **Your bot, by hand.** The katas accumulate in `no_AI_code_here/rl/` into the user's own agent: table →
   GAE → UCB bandit → a 40-line MCTS. Every few sessions it plays 2,000 battles against lesson 1's Q-table,
   greedy and MCTS-100. Watching your own 12 lines beat a script is the oldest fun in the field. If it
   writes `{"kind": "eval", ...}` lines to `logs/rl/<run>.jsonl` it shows up in the cockpit beside PPO.

**Anti-illusion rules (these are the core of `SKILL.md`):**

- **No explanation before a committed prediction.** Wrong, confident guesses are remembered best
  (pretesting / hypercorrection effect, recalled).
- **The 6-line rule.** The tutor never writes more than 6 lines before the user must act again.
- **Never "does that make sense?"** After each reveal comes a **twist**: a variation to predict. Getting the
  first right and the twist wrong is the signature of an illusion, and it is logged as one.
- **The user writes the takeaway**; the tutor edits it until it is exactly true.
- **Mastery is evidence of three kinds, never a feeling:** **P** predicted an unseen variation with the
  right reason; **B** built it by hand and it matched the repo's reference; **D** used it in a real D#
  with a stated basis. A concept is owned at two kinds on two different days.
- **Katas are checked by execution, not by reading.** The repo already holds the answer keys
  (`lesson1:82`, `lesson2_ppo.advantages`, `update`, `fit_eval.fit_within_team`): the tutor runs both on
  the same inputs and reports pass / fail and the first mismatch, nothing else. It does not fix the code.
- **Misconceptions are kept verbatim** with the wrong reason, and come back in different clothes until hit
  cleanly twice.
- **A session with no committed prediction and no user-made artifact is not logged as progress.**

**Widgets (`html_render`)** come *after* the commitment, generated on the spot with the numbers just
produced: the horizon slider (§5); a 4-arm UCB bandit where dragging "value spread" from 0.19 to 0.10
moves the visit share of the best arm (phase 4's indecision, live); "guess the curve", where the user's
typed guesses for p = 0.5 / 0.8 / 0.95 / 1.0 are drawn against the real points with their ± bands. The
bet itself is typed in chat, because I assume the widget cannot report back to the agent.
**The cockpit** is the lab bench for lesson-2 bets: its note field already says "write the prediction
first", its seed bands are the noise floor, "branch from this net" is the BC → PPO experiment.

---

## 4. Session anatomy and state

**Check-in, 5–10 min** (no coding):

| t | Step |
|---|---|
| 0:00 | **Cold question** from the due list, asked from `LEARNING.md` alone, before any other tool call. Git sync runs while the user thinks. |
| 1:30 | **The bet**: one situation (fresh result > due topic > default path), answer + confidence. Tutor commits its own. |
| 3:00 | **Reveal**: a ≤ 15 s single-core run, or a logged result with its file path, always with its band. |
| 5:00 | **Twist**: one variation, predicted, then run or reasoned. Optional widget. |
| 7:00 | **Takeaway** in the user's sentence, corrected; map location; 3–4 rows of `LEARNING.md` edited. Stop. One "one more?" at most. |

**Lab, 25–30 min:** cold question (2) → case briefing, symptoms only (3) → investigation, the user orders
evidence and names a suspect with a confidence (8) → kata in `no_AI_code_here/rl/`, checked by execution
(10) → the one equation, derived by the user line by line with the tutor only checking (3) → transfer
question or a rehearsal of a pending D# (4) → file it.

**Compute rules in the skill:** check-in runs are single-core and ≤ 60 s (`lesson1` is ~10 s). Lesson-2
runs only in labs with `--minutes ≤ 3`, or queued in the cockpit as an open bet. Planner tables (30 min,
10 workers) are **never** run for teaching: they are revealed from `results/`. One job at a time, check
`MemAvailable`, ports 8000–8007 untouched, `uv run` only.

**State: `rl/LEARNING.md`, edited row by row, no code reads it:**

```
synced: a83a7a0 · last: 2026-11-27 check-in 8 min · card: none
calibration  you: 50% 4/9 · 70% 7/11 · 90% 5/6      tutor: 70% 6/11 · 90% 8/9

## Map   (P predicted a variation · B built by hand · D used in a real decision)
| concept                     | evidence            | due   | note                                   |
| off-policy vs on-policy     | P 11-23, P 11-27    | 12-05 |                                        |
| discount = horizon          | P✗ 11-27            | 11-29 | said 0.95^40 ≈ 0.4                     |
| fixed effects / confounding | D 11-25 (chose B)   | 12-02 | D only: has not built it               |

## Open bets
- 11-27 `bet_ent005_s0..2` (cockpit queue, --minutes 6): you "0.05 beats 0.02 by > 5 pts vs MCTS", 70%;
  tutor "inside the seed band", 60%.

## Misconceptions (verbatim; retire after 2 clean hits in a new context)
- 11-27 "a lower γ makes the agent more aggressive"

## Log (newest first, one line each; keep 20, fold older lines into the Map)
```

- **Spaced repetition with no scheduler:** a right answer doubles the interval (2, 4, 8, 16, 32 days), a
  wrong one resets it to 1 day and writes a misconception. The model does the date arithmetic in the `due`
  column. At most 1 cold question per check-in and 2 per lab, oldest and weakest first. The backlog is
  never shown: after three weeks away the user gets one question, not a count.
- **Sync:** `git log <synced>..HEAD --stat -- NOTEBOOK.md rl/ scripts/` (in the Claude box:
  `distrobox-host-exec git`) and `grep -n '^## ' NOTEBOOK.md | tail`. Only the cited NOTEBOOK section is
  ever read (the file is 212 KB). New D# rows and experiments become bet candidates; the hash is updated.
- **Hook A** writes the same `D` evidence rows from ordinary project sessions, so the Map fills even in
  weeks with no tutor session at all.
- The decision-log doc stays the project's record. `LEARNING.md` is the learner's record. Neither
  duplicates the other: a D row holds the user's call, confidence and whether it matched, plus the Step
  number.

---

## 5. One check-in, as it would read

Numbers from the repo are real. `‹live›` marks output I have not seen, because nothing was run for this
proposal; the tutor's lines there depend on what the run prints.

> **you** `/rl-tutor`
>
> **tutor** *(0:00)* Cold question, from 23 Nov. Lesson 1's ε decays linearly to 0. If I froze ε at 0.3
> for all 30k episodes, what gets worse: the learned Q-table, or only the training win rate? One sentence.
>
> **you** only the training win rate. it learns the greedy policy whatever it explores with
>
> **tutor** Right, and the word is *off-policy*: the `max` on `lesson1:81` evaluates the greedy policy while
> ε-greedy collects the data. SARSA would have learned the value of the ε = 0.3 player. Due again in 8 days.
>
> *(1:30)* Today, a line you have read fifty times. CLAUDE.md: "gamma 0.95 on 40-turn episodes makes the
> win reward negligible; the agent farms shaping instead." No calculator: at γ = 0.95, what is a win 40
> turns away worth at turn 1? Number and confidence.
>
> **you** 0.4? 50%
>
> **tutor** My bet was 0.13. It is 0.95⁴⁰ = **0.129**. The rule you now own: γⁿ ≈ e^(−n(1−γ)), so the
> horizon is 1/(1−γ) = 20 turns, the win sits two horizons away, and e⁻² ≈ 0.135.
>
> *(3:00)* Now the run. Lesson 1 at γ = 1: Q(start) is Swords Dance +0.98, Dragon Claw +0.88, win rate
> 98.6%. Reward is ±1 at the end only, wins are ~5 turns away. I am about to run `--gamma 0.9`. Predict:
> (a) Q(start, Swords Dance): ~0.98, ~0.6, or below 0.4? (b) Does the opening flip to Dragon Claw?
> (c) Win rate: unchanged within ±0.6, or lower?
>
> **you** 0.6, no flip, unchanged. 70%
>
> **tutor** Mine, before I run it: (a) ~0.6. (b) Flip, 60%: a setup turn now costs 10% of everything after
> it, and Swords Dance's whole edge at γ = 1 was 0.10. (c) Lower if it flips, 55%. Running, 10 s.
> `‹live: Q at the start, win rate vs random Rotom›`
>
> *(4:30)* `‹scores all six predictions, mine included, against the ±0.6 band›` Whatever (b) did, look at
> the whole column: every Q shrank by about γ^(turns left). Discounting is not a mood called impatience. It
> changed the problem: a win in 6 turns is worth less than a win in 5. And the half people miss: a
> **loss** in 8 turns is now worth *more* than a loss in 5.
>
> *(5:30)* Twist. Your CLAUDE.md contradicts itself. Attempt #2 lists `gamma=0.95` under "values that
> actually worked" ("battles are ~25 turns"). Attempt #3 says the same 0.95 made the win negligible. Which
> sentence is wrong, or what differs?
>
> **you** episode length, 25 vs 40
>
> **tutor** Half of it. 0.95²⁵ = 0.28 against 0.13: a factor of two, not a different regime. The other
> half is what the win competes with. Attempt #3 paid a shaping reward *now*, every turn.
> `[widget: γ slider and a per-turn shaping slider; the curve γ^t with markers at 5, 25 and 40 turns; a
> bar for "paid this turn"; drag until the win outbids the shaping at turn 1]`
> The shaping scale is a slider because the old run's exact value lives in the DSPRO2 repo, not here.
>
> *(7:00)* Your takeaway, one sentence.
>
> **you** γ sets a horizon of 1/(1−γ) turns, and whatever is paid beyond about two horizons loses to
> whatever is paid now
>
> **tutor** Filed, with one addition: "and with −1 at the end, discounting also pays for losing slowly."
> Map: *horizon and reward*. Evidence: P✗ on the arithmetic (due in 1 day), `‹P or P✗›` on the run.
> For next time, no answer today: under p = 1.0 the planner's search lost 10% of Misty battles to the
> 100-turn cap, and 0% under minimax. Same disease as today, or the opposite one? 8 minutes.

(The answer is the opposite one: an undiscounted search with nothing charging for delay stalls for free,
which is why RR_PLAN.md holds a `0.995^d` depth discount in reserve. One concept, seen from both sides,
both in the user's own project, and the second side arrives as the next cold question.)

---

## 6. What I would not build

1. **A recurring lesson feed** (daily digest, weekly lecture). It cannot take a commitment before the
   reveal, so it produces exactly the fluent prose that feels like learning. It returns only through the
   gate in §1, as one self-skipping card.
2. **An LMS**: progress database, XP, streaks, badges, a quiz-bank JSON, an Anki-style scheduler script.
   Streaks punish the user for having a job. A quiz bank is stale the day after phase 6 lands. The bank is
   NOTEBOOK.md plus a simulator, and the scheduler is one `due` column.
3. **A pre-written course** (20 lesson files, a textbook rewrite, a notebook series). It rots against a repo
   that moved six phases in two days, and Sutton & Barto already exists. Only the map and the anchor
   table are authored; everything else is generated at session time from the repo as it is that day.
4. **Tutor code in `cockpit.html` / `ui/`, or a saved widget library.** `cockpit.html` is 39 KB already.
   Widgets are disposable and built from the numbers just produced; one gets saved only after it has been
   regenerated three times.
5. **Gridworld / CartPole / FrozenLake as the vehicle.** The user owns a 13 µs engine and a dozen real
   failures. Toys appear only as transfer questions, or where the engine cannot host the concept.
6. **A tutor that types in `no_AI_code_here/`, repairs the user's kata, or runs a 30-minute planner table
   to make a point.** The first two delete the learning; the third breaks the one-heavy-job rule for a
   number that is already in `results/`.

---

## Unverified, to settle in ten minutes before building

- That both harnesses discover a **symlinked** skill directory (else the 5-line stub).
- That an `html_render` widget cannot send the user's input back to the agent (designed as if it cannot).
- That a hand-written agent's `eval` lines are enough for the cockpit to draw it (the `start` line's
  `config` may be required too).
- The result of `lesson1_qlearning.py --gamma 0.9`: I predicted it in §5 and did not run it.
