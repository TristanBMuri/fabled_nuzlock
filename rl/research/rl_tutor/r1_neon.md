# Round 1 Proposal — Neon (`gemini-4-neon`)

**Core Thesis:** Build an **on-demand, cross-provider Workspace Skill (`rl-tutor`)** backed by a single plain-Markdown ledger (`rl/tutor_progress.md`) — zero new Python frameworks, zero LMS bloat. Teach RL through a **"Zoom-In / Zoom-Out Spiral"**: anchor every concept on one unifying 4-question RL map, drill down via **commit-before-reveal prediction bets, forensic autopsies of this repo's real failures, playable `html_render` math widgets, and 10-second single-core `poke-engine` probes**, then zoom back out to general RL via non-Pokémon transfer gates.

---

## 1. Delivery Architecture: On-Demand Skill vs. Scheduled Task vs. Hybrid

### Verdict: **On-Demand Workspace Skill as the Sole Engine** (+ Strictly Optional, Self-Silencing Teaser Ping)

Do **not** build a scheduled-first drip course. Make an **on-demand skill** the entire core product, with an *opt-in* single-slot scheduled "Case File Ping" only if the user later decides they want a gentle reminder.

Why this verdict wins on every constraint:

1. **Multi-turn interactivity is non-negotiable for real learning.** A scheduled task fires with no turn active; it can only post a one-way monologue. Passive reading creates the *illusion of understanding* — the user nods along to clean prose about GAE or Texel tuning and forgets it by Friday. Teaching RL requires live back-and-forth: forcing a prediction *before* running code, adapting immediately when a misconception surfaces, and calibrating depth to whether the user has 6 minutes or 30 minutes right now.
2. **Stale notification debt kills fun.** The user has limited free time. A cron that posts lessons on Mon/Wed/Fri turns a fun hobby into an unread pile of homework guilt. Once two lessons pile up unread, people mute the whole system.
3. **Repo velocity invalidates canned schedules.** Phases 0–4 of `rl/RR_PLAN.md` landed in three days (Nov 23–25). An on-demand skill inspects `git log` and `NOTEBOOK.md` live at invocation time, turning yesterday's actual experiment into today's lesson.
4. **Habit formation without nag-bots.** Replace push-nagging with two pull triggers:
   - **Decision-triggered hooks** (fitting `CLAUDE.md`'s *"Working with the user on RL decisions"*): whenever any agent reaches a real `D#` decision (e.g., upcoming D14 per-mon death costs) or completes a phase, it appends a 1-line "Fresh Case File" to `rl/tutor_progress.md` and mentions it in one line.
   - **Optional self-silencing ping (off by default):** If the user opts into `schedule_task` (e.g., Tue/Sat in one bound "RL Gym" thread, `freshSession: false`), the prompt enforces one hard rule: *read the thread first; if the last ping got no user reply, emit nothing and stop.* When it does post, it NEVER posts a lesson — only a 4-line mystery teaser (*"Raw Phase 4 logistic regression gave `our_boost_off = -0.45` and lost −2.4 pts. Why would boosting your stats look bad? Reply with a 1-sentence guess when you have 5 min."*).

### Cross-Provider Portability (Antigravity + Claude Code) & Exact Minimal File Footprint

Because the user switches between Antigravity (Neon) and Claude (Fable/Opus) in T3 Code, the skill must work identically in both with **zero duplicated prose** and **zero Python infrastructure**:

```text
.agents/skills/rl-tutor/
├── SKILL.md                  # ~120 lines: tutor rules, session flows, html_render recipes, state contract
└── references/
    └── curriculum.md         # ~150 lines: the 6-Arc Map, 24 concept nodes, repo anchors, transfer bridges

.claude/skills/rl-tutor/      # Symlinks -> ../../.agents/skills/rl-tutor/{SKILL.md,references}
                              # (file-level symlinks so tools that glob SKILL.md directly discover it)

rl/tutor_progress.md          # ~60 lines: plain-Markdown Mastery Ledger, Misconception Log, Spaced-Recall Queue
```

Plus **one 2-line pointer** in `CLAUDE.md` under *"Working with the user on RL decisions"* telling agents to load `rl-tutor` whenever the user asks to learn RL or asks for a pre-decision briefing. Total code added: **0 lines of Python**.

---

## 2. Curriculum Architecture: Big Picture → Fine Detail

Project-based learning usually fails in one direction: you memorize isolated fixes (*"use `grad_clip=5.0`"*, *"add fixed effects"*) without ever seeing the coherent architecture of RL. Textbooks fail in the opposite direction: 200 pages of Bellman contraction proofs before you care.

### The Organizing Principle: The 4-Question Grand Map + Zoom-In/Zoom-Out Spiral

**Session 1 (10 minutes)** establishes the **Grand Map of RL** (rendered interactively via `html_render`). Every method in RL — tabular Q-learning, PPO, AlphaZero, Texel-tuned MCTS, SPRT, Offline RL, and Nuzlocke routing — is just a set of engineering choices answering four questions about an MDP $(S, A, P, R, \gamma)$:

| Pillar | The Fundamental Question | Core Axes & Tradeoffs |
|---|---|---|
| **I. Problem & State** | *What is the game and what does the agent actually see?* | MDP vs POMDP, state aliasing, reward vs utility, horizon $\gamma$, single-agent vs game-theoretic |
| **II. Value Estimation** | *How do we score a state or action from noisy experience?* | MC vs TD bootstrapping, Bellman equations, bias-variance, linear/NN approx, confounding & distribution shift |
| **III. Control & Search** | *How do we turn estimates (or a simulator) into better actions?* | Value-based vs Policy Gradients vs Tree Search (Expectimax/MCTS/pUCT) vs Imitation/DAgger; Generalized Policy Iteration |
| **IV. Exploration & Truth** | *How do we find unseen lines — and prove we didn't fool ourselves?* | $\varepsilon$-greedy / entropy / UCB, generalization vs memorization, paired testing, CIs, SPRT, optimizer's curse |

Every future session pins a **"You Are Here"** marker on this map and spirals through 4 zoom depths: **L1 Intuition → L2 Equations & Code → L3 Failure Modes → L4 General-RL Transfer**.

### The 6-Arc Curriculum Mapped to Real `nuzlock_rl` Artifacts

| Arc | Core RL Theory & Math | Real `nuzlock_rl` Anchors & War Stories | General-RL Transfer Bridge |
|---|---|---|---|
| **Arc 0: The Unifying Compass** | MDP tuple, 4-axis taxonomy (model-free/based, value/policy, on/off-policy, single/multi-agent) | Why RR bosses (`rl/RR_PLAN.md`) are a single-agent stochastic MDP (best response, no self-play needed), whereas Showdown ladder (`Jaxcalibur`, `rl/README.md`) is an imperfect-info 2-player game. Why this repo legitimately uses *both* PPO (`lesson2_ppo.py`) and MCTS (`engine_search.py`). | Where MuJoCo, Atari, Stockfish/AlphaZero, Bandits, and LLM RLHF sit on the same map. |
| **Arc 1: MDPs, Aliasing & Bellman Foundations** | Markov property, state aliasing, Bellman expectation/optimality, TD(0) vs MC, Q-learning vs SARSA, discounting $\gamma$, potential-based shaping | • **The `"brn"` vs `"BURN"` bug** (`rl/lesson1_qlearning.py:64`): accidental aliasing blinded Garchomp to Burn (97.4% < always-Dragon-Claw 98.0%; fixed → 98.6%).<br>• **Attempt #3 `gamma=0.95` shaping trap** (`CLAUDE.md:86`): $0.95^{40} = 0.13$ erased the win reward, causing shaping farming (contrast Attempt #2's 25-turn horizon & Lesson 1's $\gamma=1.0$).<br>• **User's own `no_AI_code_here/players/policy/policy_player.py:157`** (`reward_computing_helper` + 12 features): real entry point into non-potential shaping.<br>• **Costly win < open fight** (`mcts.rs:226`) & **Tailwind 99 stall bias** (`RR_PLAN.md:537`). | POMDPs & belief states; Ng et al. potential-based shaping; reward hacking in robotics & reasoning models. |
| **Arc 2: Policy Gradients, Credit Assignment & Deep RL Failure Modes** | Score-function trick (REINFORCE), variance & baselines (A2C), GAE ($\lambda$ bias-variance dial), importance ratios & PPO clipping, entropy collapse, action masking, grad clipping | • **A2C 2% vs PPO 74%** (`rl/lesson2_ppo.py:391-421`, `rl/README.md:319`): 1 step/4096 batch vs 4 epochs × 512 minibatches.<br>• **REINFORCE vs A2C** (`lesson2_ppo.py:366-384`): 3 lines apart (`use_value=False, lam=1.0`).<br>• **Attempt #2 `grad_clip=0.5` silent death & `ent_coef=0.05`** (`CLAUDE.md:101-109`, `lesson2_ppo.py:521`).<br>• **Greedy overfitting**: 100% vs greedy, **0–22% vs MCTS** (`rl/README.md:258`), fixed solely by `--opp greedy,mcts` (70–88%).<br>• **Same-species action mask bug** (`CLAUDE.md:92`, `policy_player.py:88`). | TRPO/KL constraints; why LLM **GRPO** drops the critic (and why `RR_PLAN.md:276` rejected GRPO for team building); continuous-control actor-critics (SAC/TD3). |
| **Arc 3: Search, Planning & Opponent Modeling** | Exact Bellman backup on policy trees, Minimax vs Expectimax vs MCTS, UCB1 / decoupled UCT, opponent-model exploitation (Carmel & Markovitch) | • **Phase 1 RR-AI disaster → breakthrough** (`RR_PLAN.md:59-64, 238-244`): searching against the 90%-accurate Rust AI dropped Misty **−33 pts** (exploiting model errors + 100-turn caps). Fixing on-policy agreement (75%→98%) + UCB mixture **`p=0.8`** jumped **+30.2 ± 7.0 pts**.<br>• **Phase 1.5 `rl/line.py`**: exact policy-tree backup (`reach ≥ ε`) + MC tail playouts (`estimate_tail`), and why mean non-KO damage causes the ~9-pt deathless pessimism gap.<br>• **Search > prior training** (`CLAUDE.md:90`: 52%→95% vs ExIt v1: 0%). | Dyna-Q; model exploitation in learned world models (Dreamer/MBPO); CFR & Nash vs Best Response in poker. |
| **Arc 4: Value Function Approximation, Confounding & Policy Iteration** | Texel/MC regression vs TDLeaf, observational confounding, fixed effects (`f(s) − f(s₀)`), scale/temperature vs UCB, policy-iteration forgetting & replay buffers, adequacy protocol | • **Phase 4's 5-act detective story** (`rl/fit_eval.py`, `NOTEBOOK.md` 2026-11-25): Raw fit **−2.4 pts** (confounded weights: `our_items −0.33`, `our_boost_off −0.45`, `their_par −0.96`) → `rel_2x` scale fix (`+1.4 pts`, curing UCB indecision) → Fixed-effects `α[boss,team] + w·(f(s)−f(s₀))` (`B1`: `+5.9 pts`) → fixing `answers` active-vs-bench bug (`B2`: `+8.0 pts`) → `R2-only` forgetting (`+2.8 pts`) vs `R1+R2` replay buffer (**+11.0 ± 4.1 pts** on untouched seed 4, zero train/held-out gap).<br>• **Attempt #3 *"obs missing X" = undertraining*** (`CLAUDE.md:85`) & `lesson2_ppo.py` input-group ablation (`--obs core,moves` vs `all` on fixed vs `--pool`). | The **Deadly Triad**; why DQN needed an experience replay buffer (same bug as `R2-only`!); confounded offline RL (CQL/IQL, Metamon); Stockfish NNUE/Texel. |
| **Arc 5: Imitation, Self-Play & Hierarchical Value (Closing the Loop)** | Behavioral cloning & covariate shift, DAgger, Expert Iteration, AlphaZero (pUCT + policy/value net), BC→RL oscillation, hierarchical/run-level value | • **Attempt #3**: PPO 14% at 6M steps vs BC 40% in 1h; BC→PPO oscillated (`CLAUDE.md:82-84`).<br>• **`lesson2_ppo.py --algo bc`** (lines 313–334): secretly **DAgger** (student acts, MCTS labels), reaching 64% (peak 77%) on half PPO's decisions.<br>• **Lessons 3–4 roadmap & 2026 bot wave** (`rl/README.md:357-422`): Numbrion (search teacher → 0.8M net + play-time search) vs Jaxcalibur (AlphaZero pUCT self-play) vs Metamon (offline RL).<br>• **Upcoming Phase 6 / D14 (`RR_PLAN.md:173`)**: run-level value `V(box, next bosses)` replacing flat death penalty $\lambda$ with marginal future-value cost $c_i$. | SFT → RLHF/DPO oscillation in LLMs; KataGo auxiliary heads; hierarchical RL / options framework. |
| **Arc 6: Statistical & Experimental Rigor in RL** | Multi-seed variance, Common Random Numbers (paired eval), SPRT, Best-Arm ID (Sequential Halving), Optimizer's Curse, held-out distribution splits | • **`scripts/sprt.py`**: Fishtest SPRT on paired boxes (cleanly rejecting Phase 4 `R3`: `LLR -11.85 → H0`).<br>• **Phase 3 Sequential Halving & Optimizer's Curse** (`RR_PLAN.md:317, 451`): best-of-K on shared seeds inflates win rate → untouched fresh-seed confirmation (seed 4).<br>• **`rl/cockpit.py` seed bands vs 95% eval CIs** (`rl/README.md:293-304`): 200 battles = ±7 pts *eval* luck, NOT *training-seed* luck; `--holdout 0.2` trio split. | *Deep RL That Matters* (Henderson 2018); `rliable` (IQM); production bandit A/B testing. |

### Handling RL Beyond `nuzlock_rl` (No "Pokémon-Only Tricks" Trap)

Two mechanisms guarantee general RL fluency:
1. **Mandatory Zoom-Out Transfer**: Every concept node in `references/curriculum.md` carries a non-Pokémon counterpart (robotics, LLM post-training, recommendation bandits, medical/finance confounding, board games). You cannot graduate a node without solving a transfer prompt outside Pokémon.
2. **The "Why Not Here?" Expansion Shelf**: Four optional 10-minute side-quests covering major RL subfields deliberately *excluded* from `nuzlock_rl` (grounded in `RR_PLAN.md:275-282`): (a) Continuous Control & Max-Ent RL (SAC/TD3), (b) Intrinsic Exploration in Hard Sparse Worlds (RND, Go-Explore), (c) Learned World Models (MuZero/Dreamer), and (d) Offline RL & Preference Learning (IQL/CQL/DPO). Knowing *when NOT to use an algorithm* is the hallmark of an RL engineer.

---

## 3. The Fun Engine & Anti-Illusion-of-Understanding Design

Adult engineers bounce off Duolingo-style XP bars and confetti because fake rewards insult their intelligence. What *is* genuinely addictive to a builder? **Curiosity gaps, concrete wagers, fast empirical surprises, interactive toys, and owning real project decisions.**

### Five Active-Learning Mechanics

1. **Commit-Before-Reveal Prediction Bets (Every Session)**
   Never explain a result before the user bets on it. Before running a command or revealing a repo outcome, the tutor forces a concrete call (*"If Rotom ALWAYS clicks Will-O-Wisp in `lesson1_qlearning.py`, does Swords Dance still win turn 1? Bet A/B/C + 1-sentence why."*). Being right feels sharp; being wrong triggers a clean prediction-error signal that locks the lesson into memory permanently.
2. **Forensic Case Files (Playable Bug Autopsies)**
   Present real repo failures as 5-minute mysteries with the commit message hidden. Show only the crime scene (e.g., Misty −33 pts with `deaths/mon < 1 − win`, or `our_boost_off = -0.45`, or `clipfrac = 0.0` with `grad_norm = 0.5`). The user interrogates the evidence ("check switch rate", "check turn cap", "check within-boss correlation") until they crack it — then sees the real `NOTEBOOK.md` resolution.
3. **Playable Math Widgets via `html_render` (Not Walls of LaTeX)**
   Use T3 Code's inline `html_preview` + `html_render` (728px, themed with `--background`/`--primary`/`--chart-*`) to render **interactive, manipulable intuition machines** right inside the thread:
   - *Discounting & Shaping Explorer*: drag $\gamma \in [0.80, 1.00]$ and shaping weight on a 40-turn battle trace; watch terminal win value collapse to $0.95^{40} = 0.13$ while HP-shaping overtakes it.
   - *PPO Clipping Sandbox*: drag probability ratio $r(\theta)$ and flip $A > 0$ / $A < 0$ to feel the exact zero-gradient cliff.
   - *Confounding / Fixed-Effects Flipper*: scatter of `(our_boosts, outcome)` across Brock vs Koga; click **"Subtract `α[boss, team]`"** and watch the negative regression line flip positive live.
   - *UCB Spread & Temperature Widget*: slide Phase 4's logit scale (`0.5× → 2×`) and watch MCTS root visit entropy tighten from indecisive uniform to sharp commitment.
   - *Clickable 6-Arc Mastery Map*: live visual dashboard of the user's progress across all 24 nodes.
4. **10-Second `poke-engine` Micro-Experiments & `rl/cockpit.py` Labs**
   Exploit the 13 µs/turn Rust engine! All tutor probes run **single-threaded with minimal compute** (strictly honoring the user's minimal-parallelism rule):
   - *Quick probes (5–15 s)*: `uv run python rl/lesson1_qlearning.py --episodes 2000 --gamma 0.5`, or tiny inline probes comparing Q-learning vs SARSA, or evaluating `fit_eval.py` feature correlations on existing logs (zero new battles needed!).
   - *Lab runs (2–4 min, user-approved)*: controlled `lesson2_ppo.py --minutes 3 --envs 16` ablations (e.g., `--algo reinforce` vs `a2c`, `--grad-clip 0.5`, `--shape 1.0`) inspected live in `rl/cockpit.py` (`:8766`) alongside turn-by-turn MCTS diffs.
5. **Surgical `no_AI_code_here/` Katas & Live `D#` Decision Rehearsals**
   - **10-Line Code Katas**: In Deep Labs, the user writes the core 5–15 lines themselves in `no_AI_code_here/rl_katas/` (e.g., implement GAE without peeking at `lesson2_ppo.py:366`, or write the 6-line SPRT LLR formula, or convert Lesson 1 to SARSA). The tutor checks their implementation against the repo's reference output and probes edge cases (`done=None` cut trajectories!).
   - **Pre-Decision Rehearsals**: Before upcoming `D14` (per-mon death costs) or Lesson 3/4, the tutor runs a 5-minute "Chief RL Architect" rehearsal where the user designs the formulation first, then compares with `RR_PLAN.md`.

### Anti-Illusion-of-Understanding Guardrails (Hard Rules in `SKILL.md`)

1. **Zero Cold Lectures**: Never open with >4 lines of explanation before asking the user to predict, diagnose, or recall.
2. **No Free Mastery**: `"Makes sense!"` never advances a node. Status progression is strictly gated:
   - `unseen` → `intuited` (engaged with visual/example)
   - `intuited` → `practiced` (passed a prediction bet, bug autopsy, or 10-line kata)
   - `practiced` → `mastered` (passed a **delayed** spaced-recall **non-Pokémon transfer** question on a *later* day — proving retention, not short-term echo).
3. **2-Sentence Feynman Rule**: To close any concept, the user explains *why* it works (or breaks) in 2–3 sentences of their own words.
4. **Misconception Harvesting**: Every wrong bet or subtle conflation (e.g., *"eval 95% CI covers training luck"*) gets logged verbatim in `rl/tutor_progress.md` and re-tested in disguise 3–7 days later.

---

## 4. Session Anatomy & State/Progress Tracking (Minimal-Code)

### Mode A: 5–10 Minute Quick Check-In (Default)

| Min | Phase | Concrete Action |
|---|---|---|
| `0–1` | **Spaced Recall** | 1 crisp question targeting the top due node or logged misconception in `rl/tutor_progress.md`. |
| `1–2` | **Map Pin + Hook** | One-line map coordinate (`Arc 4 · Value Approx`) + a real repo anomaly/puzzle → **user places a Prediction Bet**. |
| `2–7` | **Reveal & Toy** | Run a 5–10 s `poke-engine` check OR render one `html_render` interactive widget OR inspect 15 lines of real code. Connect surprise → exact math. |
| `7–9` | **Transfer + Log** | 1 non-Pokémon transfer question (or 2-sentence Feynman check). Silently update `rl/tutor_progress.md` and drop a 1-line teaser for next time. |

### Mode B: 25–30 Minute Deep Lab (On Request: *"rl tutor lab"*)

| Min | Phase | Concrete Action |
|---|---|---|
| `0–2` | **Warmup + Map** | Quick recall on prerequisite node + zoom out on the 6-Arc Map. |
| `2–7` | **Hypothesis Spec** | Frame a live RL question (e.g., *Why does `a2c` get 2% while `ppo` gets 74%?* or *Implement GAE*). User states explicit hypothesis + what metric would falsify it. |
| `7–20` | **Hands-On Lab** | Either (a) user writes a 10–15 line kata in `no_AI_code_here/rl_katas/` and verifies it against repo code, OR (b) run a short minimal-parallelism `lesson2_ppo.py --minutes 3` / `lesson1_qlearning.py` ablation and inspect `rl/cockpit.py` live curves & MCTS diffs. |
| `20–26` | **Math & Edge-Case Autopsy** | Lock in the exact equation line-by-line against code + one "break it on purpose" counterfactual. |
| `26–30` | **Zoom-Out & Ledger** | General-RL transfer bridge + update `rl/tutor_progress.md` + link to next repo milestone (`D14` / Lesson 3). |

### Minimal-Code State & Live Repo Sync (`rl/tutor_progress.md`)

No SQLite, no JSON schema migrations, no Python state engine. One ~60-line Markdown file the agent reads at turn 0 and edits at session end:

```markdown
# RL Tutor Progress
last_synced_commit: a83a7a0

## Fresh Repo Case Files (auto-harvested from git log / NOTEBOOK.md)
- [ ] Phase 4 (a83a7a0): Why raw Texel fit lost -2.4 pts & how fixed effects + R1+R2 hit +11.0 pts
- [ ] Upcoming D14 (RR_PLAN.md): Flat death penalty λ vs run-level V(box, next bosses)

## Mastery Ledger (Leitner intervals: 1d → 3d → 7d → 21d → 60d)
| Node | Arc | Status | Last | Next Review | Note |
|---|---|---|---|---|---|
| 1.2 State Aliasing | Arc 1 | mastered | 2026-11-22 | 2026-12-13 | Nailed "brn"/"BURN" + robot sensor transfer |
| 2.3 PPO Clipping | Arc 2 | practiced | 2026-11-24 | 2026-11-27 | Needs delayed non-Pokémon transfer check |
| 4.2 Confounding & FE | Arc 4 | unseen | — | — | Queued via Phase 4 Case File |

## Active Misconceptions (re-test in disguise; remove after 2 passes)
- [2026-11-23] Assumed 200-battle 95% CI (±7 pts) captures training-seed variance (actually eval-only).
```

**Automatic repo sync (0.1 s, zero maintenance):** At session start, the skill runs `git log <last_synced_commit>..HEAD --oneline`. If new commits landed (like `a83a7a0`), it skims the commit subjects and tail of `NOTEBOOK.md` to add 1–2 fresh Case Files and bumps `last_synced_commit`.

---

## 5. One Concrete Sample Check-In (8-Minute Quick Drill)

> **User:** `/rl-tutor 8 min`

*(Tutor silently reads `rl/tutor_progress.md`, runs `git log 7cb5c03..HEAD --oneline`, spots `a83a7a0` Phase 4 completion, and sees Node `2.3` is due for spaced recall.)*

> **Tutor:**
> **Warmup (30 s — Spaced Recall from Arc 2):** Before we open today's fresh case file: in `lesson2_ppo.py`, `a2c` and `ppo` run on the exact same 4096-step batches, yet `a2c` got **2% vs MCTS** and `ppo` got **74%**. In one sentence, what did PPO change *and* why couldn't A2C just do the same thing without the clip?
>
> ---
> **You Are Here:** `Arc 4 · Value Approximation → Node 4.2 Confounding & Fixed Effects` (triggered by your new commit `a83a7a0`!).
>
> **Case File #04 — The Backwards Value Function:**
> Yesterday's first Phase 4 fit (`rl/fit_eval.py`) trained logistic regression on ~130k real battle states to predict utility $U = \text{win}\cdot(1 - 0.5\cdot\text{deaths}/6)$. Held-out log-loss looked fine — yet in play it **lost −2.4 pts** against the hand-crafted eval. Look at three weights the regression learned:
> - `our_items`: **−0.33** *(having items left looks BAD)*
> - `our_boost_off`: **−0.45** *(having +Attack/+SpA boosts looks BAD)*
> - `their_par`: **−0.96** *(paralyzing the boss looks TERRIBLE)*
>
> **Your Bet (1 sentence each):** Why would an unbiased Monte Carlo regression on real battle outcomes assign *strongly negative* weights to things that obviously help you win?
> - **A)** Overfitting — too many features (43) for 130k states
> - **B)** Confounding — those features appear mostly in *hard* bosses/weak boxes where you lose anyway
> - **C)** Label noise — MC outcomes have too much variance so signs flipped randomly
> - **D)** Something else?

> **User:** Warmup: PPO reuses the batch for 4 epochs of minibatch steps, and the clip stops the policy moving too far from the old policy that collected the data so importance ratios don't explode.
> For the bet: **B** — against Brock you just click attacks and win easily, whereas against Koga/Blaine you're desperate, clicking Thunder Wave and setup moves and still losing most games!

> **Tutor:**
> Warmup nailed — exact intuition on off-policy drift (`2.3` → next review `+7d`).
>
> And **dead-on with B!** You just diagnosed **observational confounding (Simpson's paradox)** in value fitting. Against easy Brock boxes you win fast without paralysis or setup; against brutal Koga/Blaine boxes you throw out Thunder Wave and Swords Dance trying to survive — and lose 80% of the time anyway. A naive global regression $\sigma(b + w \cdot f(s))$ uses `their_par` as a *detector for "I am fighting Koga"* rather than *"is paralyzing this turn good?"*.
>
> *(Tutor calls `html_preview` + `html_render` to render a crisp 320px interactive widget: a 2-boss scatter plot of `boosts` vs `win probability` for Brock [85% base] and Koga [15% base] with a toggle button **"Switch: Raw Fit ↔ Within-(Boss,Team) Fixed Effects (`α[g] + w·(f(s)−f(s₀))`)"**. Clicking the toggle shifts each group by its intercept `α[boss,team]` and flips the red negative trend line into a clean green positive slope!)*
>
> Look at how `rl/fit_eval.py:80-95` actually fixed this in 10 lines:
> ```python
> # logit = alpha[boss, team] + w . (f(s) - f(s0))
> loss = F.binary_cross_entropy_with_logits(alpha[g_tr] + x @ w, y)
> ```
> Every `(boss, team)` gets its own baseline difficulty intercept `α[g]` (thrown away at test time!), so `w` can ONLY explain **what changed during the battle** ($f(s) - f(s_0)$) to swing *this* matchup above or below its starting odds.
> - Raw fit: **−2.4 pts** vs hand
> - Fixed effects (`B1`): **+5.9 pts** immediately
> - Fixed effects + bench-answer fix + `R1+R2` replay buffer: **+11.0 ± 4.1 pts** on untouched seed 4!
>
> **Final Transfer Gate (20 s — Outside Pokémon):** Suppose a hospital fits an offline RL value function on ICU patient logs and finds `put_on_ventilator` has weight **−1.8** on survival. What happens if a planner maximizes that value function, and what's the exact analog of `α[boss, team]` here?

> **User:** The planner would actively avoid putting dying patients on ventilators! The analog of `α[boss, team]` is controlling for each patient's baseline illness severity at admission before estimating the treatment effect.

> **Tutor:** Spot on — in econometrics/RL that's **confounding by indication** (controlling for baseline severity / propensity). Logged `4.2 Confounding & FE → practiced` (delayed check in `+3d`).
>
> **Teaser for next time:** Once Phase 4 worked, you ran a second round of on-policy data collection — and training on *only* Round 2 (`R2-only`) **lost 5.2 points** vs keeping Round 1 (`R1+R2`). Same exact bug that forced DeepMind to invent the DQN replay buffer. See you next check-in!

---

## 6. What I Would Explicitly NOT Build

1. **No Custom LMS Web App, Database, or Python Quiz Engine.** `CLAUDE.md`'s #1 rule is **minimal code** — two earlier attempts ballooned into infrastructure monsters. T3 Code's chat + `html_render` + existing `rl/cockpit.py` (`:8766`) and `ui/build.py` (`:8765`) already give us interactive graphics, live training plots, and turn-by-turn diffs. A 60-line `rl/tutor_progress.md` beats a 1,500-line SQLite quiz server every single day.
2. **No Scheduled-First Daily Lesson Bot.** Unread cron lectures breed guilt, rot against fast-moving repo commits, and strip away the multi-turn Socratic prediction loop that actually creates understanding.
3. **No Streaks, XP, Badges, or Shallow Gamification.** Artificial points incentivize rushing through questions instead of sitting with a hard idea. Prediction bets, forensic repo mysteries, playable math widgets, and real `D#` ownership generate genuine intrinsic pull.
4. **No Parallel Toy-Environment Track (CartPole / FrozenLake / Gridworld).** Starting over on CartPole throws away this project's superpower: a 13 µs Rust simulator, a live cockpit, and documented battle scars. General RL breadth comes from **Zoom-Out Transfer gates**, not abandoning the codebase.
5. **No Heavy Compute or Production-Code Edits Inside Tutor Sessions.** Tutor runs stay strictly single-core and fast (5–15 s probes, or explicit 2–3 min low-parallelism `cockpit.py` runs) to respect the machine's memory limits and minimal-parallelism rule. User coding exercises stay strictly sandboxed in `no_AI_code_here/rl_katas/` so the planner and `rl/` track stay spotless.
