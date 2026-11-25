# `rl-tutor` Curriculum Map (`references/map.md`)

## The Unifying Map (Generalized Policy Iteration)
> **Caption:** Every RL method alternates **evaluating** a policy ↔ **improving** it.

**4+1 Columns (filled row-by-row in the user's own words, 1 row per session):**
1. **World:** What do you know about the world? (exact model / learned model / sampler only)
2. **Store:** What do you store? (nothing / table Q / linear V / net π+V)
3. **Targets:** Where do targets come from? (exact DP / MC outcome / TD bootstrap / GAE / teacher move / search visits)
4. **Data & Foe:** Whose data and which opponent? (on/off-policy, DAgger, replay buffer, random / scripted / minimax / self-play)
5. **Ambient 5th Question (every reveal):** *How do you know it worked?* (error band, pairing, held-out, fresh seed)

**Pick order:** `(1)` user ask > `(2)` pending `D#` > `(3)` unsettled `user bet:` or fresh commit > `(4)` due recall/misconception > `(5)` next unseen region (after 3 sessions in one region, default to an unseen one).
**Anchor rule:** Grep every symbol before quoting numbers; quote source + baseline; if a symbol moved, repair that line here in place.

---

## 14 Core Regions (Symptom First → Bet → Twist → Kata/Derivation → Transfer)

### 1. MDP, Value & Bellman / DP
- **Anchors:** `rl/lesson1_qlearning.py` (`def train`, `def step`); `rl/line.py` (exact policy-tree backup). *(Note: `lesson1`'s `iterative_deepening_expectiminimax` plays a worst-case foe, not random Rotom; do not assign unverified full-depth exact expectimax on Lesson 1.)*
- **Kata (`no_AI_code_here/rl/td.py`):** Tabular Q-update + SARSA variant checked against `lesson1_qlearning.train`.
- **Source (*recalled*):** Sutton & Barto 2e ch. 3–4.
- **Transfer:** Inventory restocking with known demand probabilities—when does exact DP beat model-free TD?

### 2. State Aliasing & Observability (POMDPs)
- **Anchors:** `git show 8240c1c:rl/lesson1_qlearning.py` (`"brn"` vs `"BURN"` bug); `rl/lesson2_ppo.py` (`OBS`, `--obs` ablations in `rl/README.md` grep `"What it looks at"`).
- **Kata (`no_AI_code_here/rl/`):** Resurrect `8240c1c` into scratchpad, diagnose from output alone, then deliberately alias another feature in `key()`.
- **Source (*recalled*):** S&B ch. 3; POMDP belief states & determinization (`rl/README.md` Foul Play).
- **Transfer:** A warehouse robot sees only current camera frame (no velocity) and oscillates near walls—why isn't the state Markov?

### 3. Horizon, Discounting & Reward Shaping
- **Anchors:** `CLAUDE.md` (grep `gamma 0.95` — Attempt #3 40-turn shaping failure vs Attempt #2 25-turn working `gamma=0.95`); `project_overview.md` (grep `shaping`); `no_AI_code_here/players/policy/policy_player.py` (`calc_reward`); `rl/engine_search.py` (`POKE_ENGINE_DEATH`).
- **Kata:** Live `--gamma 0.9` bet on `lesson1_qlearning.py`; implement a potential-based shaping term $\gamma\Phi(s')-\Phi(s)$.
- **Source (*recalled*):** Ng, Harada & Russell 1999 (potential-based shaping).
- **Transfer:** CartPole pays `+1` per step up to 500 steps at `γ=0.99`—does Attempt #3's shaping-farming bug occur? Why or why not?

### 4. MC vs TD & GAE (Bias vs Variance)
- **Anchors:** `rl/lesson2_ppo.py` (`def advantages`, `lam`, `reinforce` vs `a2c`); `rl/fit_eval.py` docstring (Monte Carlo outcome labels).
- **Kata (`no_AI_code_here/rl/gae.py`):** 12-line GAE matched against `lesson2_ppo.advantages` on a scratchpad `buf` dict containing a `done=None` truncated trajectory.
- **Source (*recalled*):** S&B ch. 5–6; Schulman et al. 2016 GAE §3.
- **Transfer:** Predicting delivery time of a 20-stop truck route: compare waiting for final arrival (MC) vs updating after each stop (TD).

### 5. Policy Gradients → A2C → PPO
- **Anchors:** `rl/lesson2_ppo.py` (`def update`, clipped surrogate, `grad_clip`); `rl/README.md` (grep `Comparison 1` — `a2c` vs `ppo`); `rl/cockpit.html` (grep `before clipping` — plotted `grad_norm` is pre-clip!).
- **Kata (`no_AI_code_here/rl/pg.py`):** Clipped surrogate + scratchpad cockpit sabotage (`done` leak or `--grad-clip 0.5`), diagnosed from curves alone.
- **Source (*recalled*):** S&B ch. 13.3 (baseline zero-bias proof); OpenAI Spinning Up PPO.
- **Transfer:** Why does reusing one batch for 4 gradient epochs work in PPO but destabilize vanilla A2C?

### 6. Exploration, Entropy & Value Scale
- **Anchors:** `CLAUDE.md` (`entropy_coeff=0.05`); `poke-engine/src/mcts.rs` (UCB1); `NOTEBOOK.md` (grep `spread` — Phase 4 root value spread compression & `rel_2x` rescale).
- **Kata (`no_AI_code_here/rl/bandit.py`):** 4-arm UCB1 bandit over Garchomp's 4 moves vs random Rotom.
- **Source (*recalled*):** S&B ch. 2 (bandits, UCB, Thompson sampling).
- **Transfer:** A recommender's click-rate model outputs scores in `[0.48, 0.52]` instead of `[0, 1]` under fixed UCB `c=1.0`—what happens to exploration?

### 7. Generalization & Opponent Diversity
- **Anchors:** `rl/lesson2_ppo.py` (`split_teams`, `--pool`, `--holdout`, `--opp`); `rl/README.md` (grep `First result` — greedy-only vs MCTS; grep `Comparison 2`).
- **Kata:** Write a leak-free team-combo holdout split and run a ≤3 min `--opp` cockpit comparison.
- **Source (*recalled*):** Cobbe et al. (Procgen generalization); PLR / league training.
- **Transfer:** An autonomous car trained only in sunny California achieves 99.9% on validation streets there—why is random frame splitting a trap?

### 8. Imitation, DAgger & Expert Iteration
- **Anchors:** `CLAUDE.md` (BC vs PPO from scratch; BC→RL oscillation); `rl/lesson2_ppo.py` (`--algo bc` = DAgger); `rl/README.md` (grep `Numbrion`, `Jaxcalibur`).
- **Kata (`no_AI_code_here/rl/lesson3_exit.py`):** **Lesson 3 capstone** (search teacher → net + shallow search) importing `new_battle`, `observe`, `legal`, `step` from `rl/lesson2_ppo.py`.
- **Source (*recalled*):** Ross, Gordon & Bagnell 2011 (DAgger); Anthony et al. 2017 (ExIt).
- **Transfer:** Why does behavioral cloning on expert human driving drift off the road after one tiny steering mistake, and how does DAgger fix it?

### 9. Planning, Search & Simultaneous Moves
- **Anchors:** `rl/RR_PLAN.md` (grep `Ladder 2k`, `crits`); `poke-engine/src/mcts.rs` (decoupled UCT, root/depth-1 crit comment); `CLAUDE.md` (search-all vs prior training).
- **Kata (`no_AI_code_here/rl/matrix.py`):** Extract a 4×4 turn payoff matrix via `generate_instructions` and solve by regret matching.
- **Source (*recalled*):** Browne et al. 2012 (MCTS survey); Nash vs best response, CFR.
- **Transfer:** Rock-Paper-Scissors with asymmetric payoffs: why does decoupled UCT / regret matching find mixed strategies where deterministic minimax fails?

### 10. Opponent Models & Model Exploitation
- **Anchors:** `rl/README.md` (grep `The RR AI in Rust`); `rl/RR_PLAN.md` (grep `Phase 1`, `p = 0.8`); `rl/rr_ai_diff.py` & `rl/rr_ai_onpolicy.py`.
- **Kata:** 8-line per-node mixture pseudocode (`p` model + `1-p` minimax) + ranking bet over `p ∈ {0.5, 0.8, 0.95, 1.0}`.
- **Source (*recalled*):** Carmel & Markovitch (opponent model exploitation); model-based RL exploitation.
- **Transfer:** A trading optimizer plans against a 90%-accurate simulator of market slippage and blows up live—what happened?

### 11. Value Fitting, Confounding & Replay
- **Anchors:** `rl/fit_eval.py` (`fit_within_team`, `within_team_logloss`); `NOTEBOOK.md` (grep `Phase 4` — raw fit vs `B1`/`B2` 43→46 features, `R2-only` vs `B2` and vs hand, `R1+R2` seed 4).
- **Kata (`no_AI_code_here/rl/fit_within.py`):** Within-group fixed-effects fit on `f(s)−f(s₀)` verified against `fit_eval.fit_within_team`.
- **Source (*recalled*):** Chess Texel tuning; Mnih et al. 2015 (DQN replay buffer & catastrophic forgetting).
- **Transfer:** Hospital ICU logs give `put_on_ventilator` weight `-1.8` on survival—what happens if a planner maximizes that model, and what is `α[boss, team]` here?

### 12. Hierarchy & Run-Level Value
- **Anchors:** `rl/RR_PLAN.md` (grep `D14`, `Phase 6`); `scripts/rr_route.py` (encounter routing & marginal future box value).
- **Kata (`no_AI_code_here/rl/lesson4_az.py`):** Hand-compute marginal death cost `c_i` on one box; **Lesson 4 AlphaZero-lite capstone** closing the GPI loop.
- **Source (*recalled*):** Sutton, Precup & Singh 1999 (Options / Hierarchical RL).
- **Transfer:** In an emergency room or cloud cluster, why is a flat cost per unit of resource spent wrong compared to marginal future-capacity cost?

### 13. Statistics I — Noise Floor & Pairing
- **Anchors:** `rl/README.md` (grep `Seeds and the queue`, `standard error`); `README.md` (grep `Noise floor`); `scripts/league_test.py` (paired openers).
- **Kata:** Live `--seed` variance bet on `lesson1_qlearning.py`; verify binomial SE `√(p(1-p)/n)` vs training-seed spread.
- **Source (*recalled*):** Henderson et al. 2018 (*Deep RL That Matters*); Agarwal et al. 2021 (`rliable`).
- **Transfer:** Two RL runs differ by `+4%` over 200 eval episodes (1 seed each). What two distinct sources of variance could explain the entire gap?

### 14. Statistics II — Selection, SPRT & Holdouts
- **Anchors:** `scripts/sprt.py`; `rl/RR_PLAN.md` (grep `SPRT`, `seed 4`, `Held-out bosses`); `scripts/rr_search.py` (sequential halving).
- **Kata (`no_AI_code_here/rl/sprt_kata.py`):** 6-line paired SPRT LLR update checked against `scripts/sprt.py`.
- **Source (*recalled*):** Wald SPRT / Fishtest; Optimizer's Curse (Smith & Winkler).
- **Transfer:** You tune 10 hyperparameters on one validation set, pick the winner (+6%), and ship it—why does production systematically underperform +6%?

---

## Cockpit Compatibility Note (Hand-Written Agents)
To plot a `no_AI_code_here/rl/` bot in `rl/cockpit.py`, write `logs/rl/<run>.jsonl` with all three record kinds (`rl/cockpit.html:185,197,267`):
1. `{"kind": "start", "t": <epoch>, "pid": 0, "params": 0, "n_obs": 0, "config": {"algo": "hand", "opp": "mcts", "steps": 0, "eval_n": 200}}`
2. `{"kind": "eval", "step": <n>, "t": <epoch>, "a": <sec>, "vs": {"random": {"win": <0..1>, "n": 200}}}`
3. `{"kind": "end", "t": <epoch>}`

---

## 6 "Why Not Here?" Expansion Shelf Cards (Offer When `foreign ≥ 3`)
Grounded in `rl/RR_PLAN.md`'s rejection table; each card places the method on the 4-col map, names one primary source (*recalled*), explains why `nuzlock_rl` rejected it, and asks 1 question back:
1. **Deadly Triad & DQN:** Off-policy + function approx + bootstrapping (`S&B 11.3`). Why rejected here: exact simulator + MCTS beat value-only deep Q; *Q: how did Phase 4's `R2-only` mirror DQN without replay?*
2. **Continuous Control (SAC / TD3):** Soft actor-critic & max-entropy RL (`Haarnoja 2018`). Why rejected here: 7–9 discrete actions; *Q: why can't poke-engine's UCB MCTS act directly in 12-D robot torque space?*
3. **Hard Sparse Exploration (RND / Go-Explore):** Intrinsic novelty bonuses (`Burda 2018`). Why rejected here: 10–40 turn battles, dense simulator forks; *Q: when does curiosity reward hack itself (noisy-TV trap)?*
4. **Learned World Models (MuZero / Dreamer):** Latent dynamics + planning (`Schrittwieser 2020`). Why rejected here: exact Rust simulator exists; *Q: how is Phase 1's buggy RR-AI exploit the exact failure mode of learned world models?*
5. **Offline RL (CQL / IQL, Metamon):** Conservative Q-learning on fixed datasets (`Kumar 2020`; `rl/README.md` Metamon). Why rejected here: we can query the sim live; *Q: why does naive `argmax Q` fail on logged trajectories?*
6. **LLM Post-Training (RLHF KL & GRPO):** Critic-free group-relative PPO + KL to reference (`Shao 2024`; `RR_PLAN.md` GRPO rejection). *Q: map KL-to-ref and reward-model hacking onto Regions 5 and 11.*
