# Radical Red Hardcore Nuzlocke: Independent Research & Learning Plan

*Prepared independently from repository inspection and web literature search (read-only on repo; no edits, no battles run).*

---

## Part 1 — Literature Research Notes (Emphasis 2022–2026)

Every entry states **what it is**, **what it concretely gives this project**, **its implementation/compute cost**, and **provenance** (`[web summary]` = web search summary retrieved this session, full PDF not read; `[recalled]` = recalled from prior training knowledge, not verified against full PDF this session; `[repo]` = directly verified in repository code/logs). No numbers are invented.

### 1.1 Planning Against a Known Fixed Opponent Policy & Model-Error Robustness

1. **Safe Opponent-Exploitation Subgame Refinement (SES)** — Liu et al., NeurIPS 2022 `[web summary]`
   - *What it is:* Real-time search that interpolates between a safe (Nash/minimax) backbone and best-response exploitation of a bounded/suboptimal opponent model, bounding exploitability when the opponent deviates from the model.
   - *Concrete gain for RR:* Directly explains and solves our `--engine-opp ai` failure (−13 to −33 pts). Rather than pure best-response to a 90.3%-accurate Rust RR AI (zero support on unmodeled replies) or pure pessimistic minimax, we search against an **$\epsilon$-robust opponent mixture**: with probability $1-\epsilon$ the opponent follows `rr_ai`, and with probability $\epsilon$ (e.g. $0.15\text{–}0.30$) it uses adversarial UCB1 (minimax).
   - *Cost:* ~25 lines in `poke-engine/src/mcts.rs`; zero neural overhead; ~15 min to sweep $\epsilon$ on 30 boxes.

2. **Safe and Robust Subgame Exploitation in Imperfect Information Games (OX-Search)** — Ge et al., ICML 2024 `[web summary]` (and **Safe Opponent Exploitation for $\epsilon$-Equilibrium Strategies**, Jeary & Turrini, arXiv:2307.12338, 2023 `[web summary]`)
   - *What it is:* Formalizes *adaptation safety* under inaccurate opponent models: exploiting an opponent model must not drop worst-case value below a safety floor if the opponent model is misspecified.
   - *Concrete gain for RR:* Gives a concrete root-backup rule: even when exploiting `rr_ai`, reject any candidate move whose *worst-case* (minimax or 1-ply adversarial) reply causes an avoidable lethal death when a safer positive-EV line exists. Especially valuable in Nuzlocke, where one unexpected non-switch by the boss permanently kills a mon.
   - *Cost:* ~15 lines in `mcts.rs` or root selection in `rl/engine_search.py`.

3. **MCTS–Minimax Hybrids & Neighborhood-Constrained Minimax** — Baier & Winands `[recalled]` / JAIR & recent robust-MCTS literature `[web summary]`
   - *What it is:* Replacing average-case rollouts against an imperfect opponent model with shallow minimax checks (or worst-case selection inside a divergence ball $\mathcal{P} = \{P : D(P, \hat{P}) \le \epsilon\}$ around the modeled policy $\hat{P}$).
   - *Concrete gain for RR:* Confirms that tapering opponent-model reliance with depth (use `rr_ai` + small $\epsilon$ at depth 0–2 where state is accurate, widen $\epsilon$ deeper in the tree where model drift compounds) prevents deep switch-loop hallucinations.
   - *Cost:* ~10 lines in `mcts.rs` (`epsilon_at_depth`).

4. **Consistent Opponent Modeling in Imperfect-Information Games** — Ganzfried, arXiv:2508.17671, 2025 `[web summary]`
   - *What it is:* Analyzes when naive opponent-modeling procedures fail to converge to a static opponent's true policy.
   - *Concrete gain for RR:* Mostly a negative/cautionary takeaway: since the RR AI is a *known deterministic-modulo-ties script* (`ai_util/rr_ai.py`), we should **not** learn an opponent model from data; fixing the remaining 9.7% Rust porting bugs (`results/rr_ai_diff.log`) is faster, exact, and avoids statistical estimation error.
   - *Cost:* N/A (justifies avoiding learned opponent models).

---

### 1.2 Simultaneous-Move Search

5. **Simultaneous-Move MCTS: Decoupled UCT (DUCT) vs Regret Matching (RM) / Exp3** — Lanctot, Lisý, Winands, Shafiei et al. `[recalled + web summary]`
   - *What it is:* Analyzes simultaneous-move tree search. DUCT (what `poke-engine/src/mcts.rs` runs) runs independent UCB1 bandits for each player; it is fast and empirically strong in spatial/tactical games, but has **no Nash convergence guarantee** and can cycle or mis-weight correlated simultaneous actions. Exp3 and Regret Matching are Hannan-consistent ($\epsilon$-NE convergence) at the cost of higher variance and slower empirical convergence in deep trees.
   - *Concrete gain for RR:* Explains why `poke-engine` DUCT works well as a *pessimistic* baseline against a scripted AI (our enemy is NOT an adaptive Nash adversary, so DUCT's lack of Nash convergence against a rational opponent is rarely fatal), **while** warning us that switching to RM/Exp3 is *not* a priority unless we observe cycling pathologies in mirrors. Our actual search failures (mega bug, Tailwind 99, won-terminal score, turn-100 stall loops under `vs_ai`) are evaluation/objective bugs, not DUCT-vs-RM issues.
   - *Cost:* Zero (avoid premature rewrite of DUCT to RM/Exp3).

---

### 1.3 Search + Learned Value / Prior at Small Budgets

6. **Policy Improvement by Planning with Gumbel (Gumbel AlphaZero / MuZero)** — Danihelka, Guez, Schrittwieser, Silver et al., ICLR 2022 `[web summary + recalled]`
   - *What it is:* Replaces PUCT + Dirichlet noise at the root with **Gumbel-Top-$k$ sampling without replacement + Sequential Halving**, paired with a principled completed-$Q$ policy improvement target. Proven monotonic policy improvement even with **very small simulation budgets** (e.g. 2–16 simulations). Extended to combinatorial optimization (*Policy-Based Self-Competition*, Pirnay et al., ICLR 2023 `[web summary]`) and large combinatorial action spaces (*Multiagent Gumbel MuZero*, Hao et al., AAAI 2024 `[web summary]`).
   - *Concrete gain for RR:* Two direct uses:
     1. **At the team-selection layer (immediate, Phase 1):** Sequential Halving at the root over candidate sixes/sets is mathematically identical to Gumbel root allocation without needing a policy net first; once a team-builder policy exists (Phase 3), Gumbel-Top-$k$ + Sequential Halving gives a drop-in policy improvement operator over candidate sixes.
     2. **At the battle layer (deferred, Phase 4):** If we ever put a neural value/prior into `poke-engine`, Gumbel Sequential Halving lets us use **32–256 simulations** instead of 20,000, making batched GPU/ONNX inference viable without a 100× slowdown.
   - *Cost:* ~40 lines in Python for team-level Sequential Halving (Phase 1); ~150 lines Rust + 250 lines Python if used in-battle (Phase 4).

7. **Stop Regressing: Training Value Functions via Classification for Scalable Deep RL (HL-Gauss)** — Farebrother, Orbay, Vuong, Taïga, Chebotar, Xiao, Irpan, Levine, Castro, Faust, Kumar, Agarwal, ICML 2024 (arXiv:2403.03950) `[web summary + recalled]`
   - *What it is:* Replaces MSE regression for continuous value targets with cross-entropy over a categorical histogram smoothed with a Gaussian kernel (HL-Gauss). Dramatically improves calibration, feature learning, and stability under noisy, non-stationary, or bimodal targets (extended to RLVR critics in 2026 `[web summary]`).
   - *Concrete gain for RR:* Battle outcomes and team scores (`progress ∈ [0, 2]`, win/death margins) are strongly **bimodal** (a team either snowballs to a 6–0 win or gets swept 0–6 after losing its check). MSE regresses to the uninformative mean (~0.8) and learns flat representations. Training our team scorer's `progress` head and any future battle-value head with **HL-Gauss (e.g., 21 bins on `[0, 2]`)** gives sharper ranking and better representations at virtually zero code cost.
   - *Cost:* ~15 lines of PyTorch.

8. **KataGo Auxiliary Training Targets** — Wu, 2019–2020 `[recalled + web summary]`
   - *What it is:* Accelerates AlphaZero-style value/policy training by orders of magnitude via dense auxiliary supervision (ownership maps, score distribution, opponent move prediction) alongside the sparse win/loss label.
   - *Concrete gain for RR:* A single binary `won ∈ {0, 1}` label per battle throws away massive information. Every battle already yields: (a) **6 per-mon death indicators** `died_i ∈ {0, 1}`, (b) **6 enemy KO/HP-left fractions**, (c) continuous `progress`, and (d) turn-0 MCTS root value. Supervising all of these jointly on the shared Set-Transformer encoder gives ~13× denser gradient per battle—crucial when we only have $10^4\text{–}10^5$ battles, not $10^7$.
   - *Cost:* ~20 lines in the training loss.

9. **Amortized Planning with Large-Scale Transformers ("Searchless Chess")** — Ruoss et al., DeepMind, arXiv:2402.04494, 2024 `[web summary]`
   - *What it is:* Trains a transformer by supervised learning on Stockfish-annotated action-values (categorical value bins) to play strong chess in a single forward pass without test-time search.
   - *Concrete gain for RR:* Reinforces two repo lessons: (a) amortizing a strong search teacher into a value/action-value predictor *works* when supervised on rich search values (not just sparse binary wins), **but** (b) requires huge data to replace search outright—so in RR (small budget, fast Rust engine), a small amortized scorer should **guide/shortlist** search rather than replace it.
   - *Cost:* Concept only (informs Phase 2/4 targets).

10. **2025–2026 Competitive Pokémon Bots: Metamon, Foul Play, PokéAgent Challenge, Jaxcalibur, Numbrion** `[repo + web summary]`
    - *What it is:* Metamon (UT Austin, arXiv:2504.04395) uses offline RL on human replays; Foul Play uses DUCT MCTS on `poke-engine`; PokéAgent Challenge (NeurIPS 2025, arXiv:2603.15563) showed search (Foul Play) + RL dominate over LLMs; Numbrion/Jaxcalibur (2026 forum reports) showed tiny nets (~0.8M params) + search over a fast Rust/JAX simulator reach ladder #1.
    - *Concrete gain for RR:* Validates our exact substrate (`poke-engine` Rust sim + small ~0.1M–0.5M param net + search) while highlighting our key structural difference: RR is **full-information PvE against a fixed scripted AI with permanent death**, so team & set construction dominate far more than in random-battles PvP.

---

### 1.4 Learned Combinatorial Selection (Picking a Six from a Box)

11. **Set Transformer** — Lee et al., ICML 2019 `[recalled + web summary]` + **Pairwise Edge-Biased Attention (Graphormer-style)** `[recalled]`
    - *What it is:* Permutation-equivariant self-attention over unordered sets; adding a learned projection of pairwise edge features $E_{ij}$ directly to attention logits $\text{Attn}(Q_i, K_j) + \phi(E_{ij})$ encodes bipartite relationships (our 6 mons × enemy 6 mons) without losing permutation invariance.
    - *Concrete gain for RR:* A 6-mon team is an unordered set (except the lead slot, marked by a binary `is_lead` flag). Injecting our existing exact `calc_bot` / `matchup` 6×6 pairwise features (`hit`, `take`, `ko_them`, `ko_us`, `faster`, `safe`, `race`) directly as **cross-attention edge bias** between our 6 mon tokens and the boss's 6 mon tokens gives the network exact damage-calc physics from step 0 while letting it learn multi-mon synergies (pivoting, sac-and-revenge, shared checks, defensive cores) that the greedy cover matrix misses.
    - *Cost:* ~80 lines of PyTorch; ~150k–250k params; trains in <5 min on an RTX 5090.

12. **Pointer Networks** — Vinyals et al., NeurIPS 2015 `[recalled + web summary]` + **POMO (Policy Optimization with Multiple Optima)** — Kwon et al., NeurIPS 2020 & 2024–2025 Best-of-$K$ extensions `[web summary + recalled]`
    - *What it is:* Autoregressively points to indices in a variable-size input set (box of size $N \in [9, 81]$) with legality masking (dupes clause, ≤1 mega stone, no Truant). POMO exploits symmetry (multiple starting picks / rollouts) to stabilize subset construction and avoid local optima.
    - *Concrete gain for RR:* In Phase 3 (only if surrogate + local search saturates on 50–80 mon boxes), a lightweight pointer decoder conditioned on the boss + picks-so-far proposes 16–32 diverse candidate sixes (by sampling different first picks / leads, POMO-style), which search then verifies.
    - *Cost:* ~120 lines of PyTorch on top of the Phase 2 encoder.

13. **Listwise / Plackett-Luce Surrogate Ranking Losses for Expensive Combinatorial Optimization** — ListNet/ListMLE & recent 2022–2025 offline MBO / ambiguous subset-selection papers `[web summary]`
    - *What it is:* Training a surrogate model to minimize **within-instance listwise ranking loss (Plackett-Luce / ListMLE / pairwise Bradley-Terry)** rather than pure MSE, because combinatorial selection only cares about preserving the top-K order within each box/boss instance.
    - *Concrete gain for RR:* Directly addresses the repo's calibration/shift problem across bosses (Brock 70% vs Koga 8%): raw MSE wastes capacity fitting boss-level difficulty offsets, whereas **within-`(box, boss)` pairwise/Plackett-Luce loss** focuses gradients on *which six beats which other six in the same box*. Combine with calibrated BCE/HL-Gauss so probabilities stay meaningful for Phase 5 encounter routing.
    - *Cost:* ~15 lines of PyTorch.

14. **Neural / Surrogate-Assisted Large Neighborhood Search (NLNS / Surrogate-Guided Local Search)** — 2022–2025 combinatorial optimization literature `[web summary]`
    - *What it is:* Instead of asking a neural net to generate a globally optimal combinatorial object in one shot, use a cheap learned surrogate to screen hundreds of **local destroy-and-repair moves** (1-swaps, 2-swaps, lead changes, item swaps) around strong seeds and invoke the expensive true simulator only on the top candidates.
    - *Concrete gain for RR:* **Highest ROI paradigm for this project.** For a 73-mon box at Blaine, there are $\binom{73}{6} \approx 1.7\times 10^8$ sixes, but the 1-swap neighborhood of a seed six has only $6 \times 67 = 402$ neighbors, and 6 lead swaps. Scoring 1,000 candidates with a tiny Set-Transformer surrogate takes **<30 ms on an RTX 5090**, letting us do 3–5 rounds of surrogate-guided local search before spending a single MCTS battle!
    - *Cost:* ~60 lines of Python; zero RL instability; strictly monotonic improvement over seed sixes once verified by search.

15. **Automated Pokémon Team Building & Metagame Balance** — Reis, Novais, Reis & Lau, *IEEE Trans. on Games*, 2023 `[web summary]`
    - *What it is:* Combines evolutionary search, linear optimization, and a neural battle-outcome predictor to evolve competitive Pokémon teams against a metagame.
    - *Concrete gain for RR:* Confirms the **surrogate-assisted evolutionary/local search** template works specifically on Pokémon team composition, and warns that pure argmax on the neural predictor exploits surrogate errors unless periodically grounded in real simulated battles (matching `rl/RR_PLAN.md` Phase 3's "filter, never judge" rule).
    - *Cost:* Design validation.

---

### 1.5 Set / Item Selection

16. **Coordinate / Structured Local Search & Surrogate-Guided Discrete Optimization** — standard discrete optimization / SAEA literature `[web summary + recalled]`
    - *What it is:* Factoring a huge joint discrete space (`species × moves × items × natures × abilities`) into hierarchical coordinate updates: fix the 6 mons, then optimize discrete set knobs per mon (nature, item, 1–2 flex move slots) via surrogate screening + paired simulation.
    - *Concrete gain for RR:* Trying to learn sets jointly with 6-of-73 team selection explodes the action space ($>10^{15}$). Fixing deterministic set-builder bugs first (Phase 0c), then doing a **constrained coordinate sweep** over high-leverage levers only on the final shortlisted teams (Phase 3: +Spe vs +Atk/SpA nature, top-3 legal items like Sash/Scarf/Leftovers/Resist Berry, 1 utility move like Protect/Priority/T-Wave/Weather) captures ~90% of optimal set customization in ~50 lines of code.
    - *Cost:* ~50 lines of Python; ~20 extra fast battles per box (only on top-2 teams).

---

### 1.6 Curricula & Choosing Which `(box, boss)` Cases to Train On

17. **Prioritized Level Replay (PLR / $\text{PLR}^\perp$)** — Jiang et al., ICML 2021 / arXiv:2110.02439 `[web summary + recalled]` & **ACCEL** — Parker-Holder, Jiang, Dennis et al., ICML 2022 (arXiv:2203.01302) `[web summary]`
    - *What it is:* Samples training instances (levels) not uniformly, but prioritized by estimated **learning potential / regret** (prediction error / TD-error / variance between current policy and best seen), with ACCEL mutating high-regret instances to track the agent's frontier.
    - *Concrete gain for RR:* Across 91 singles bosses, early bosses are ~100% wins and broken/under-built fights are 0% wins—both carry near-zero ranking signal! Prioritizing `(box, boss)` training generation by:
      1. **Non-degenerate win-rate band** (boxes where candidate teams span $0.10 \le \text{win} \le 0.90$, or progress variance $> 0.15$), and
      2. **Surrogate regret / disagreement** (boxes where the surrogate's top-K disagrees most with search verification, or Brier residual is highest)
      prevents wasting compute on 0% or 100% bosses while naturally focusing data collection on the hard frontier (Surge → Erika → Koga → Blaine → E4).
    - *Cost:* ~25 lines of sampling weight logic in the data collector.

---

### 1.7 Sample-Efficient Evaluation of Candidates Under Noisy Win Rates

18. **Sequential Halving (SH)** — Karnin, Koren & Somekh, ICML 2013 `[recalled + web summary]` & **Best-Arm Identification / Top-Two Thompson Sampling (TTTS)** — Russo, 2016/2020 & 2022–2025 bounded/small-sample extensions `[web summary]`
    - *What it is:* Fixed-budget pure-exploration bandit algorithms that allocate evaluation samples adaptively instead of uniformly ($K \times n$). Sequential Halving runs $\lceil \log_2 K \rceil$ rounds, doubling battles per survivor and dropping the bottom half each round, with zero hyperparameters and trivial parallelization across CPU workers. TTTS splits samples between the current leader and strongest challenger.
    - *Concrete gain for RR:* Currently `--chooser searched` spends $24 \times 4 = 96$ battles uniformly across 24 candidates, then $3 \times 6 = 18$ verification battles (114 battles total), giving noisy 4-battle estimates (SE = $\pm 25\%$ at $p=0.5$!). With **Sequential Halving** on the *same* ~112-battle budget over 32 candidates:
      - Round 1: 32 teams × 2 battles (at MCTS-2k, ~0.3s each) = 64 cheap battles (~19 worker-s)
      - Round 2: 16 teams × 2 more battles (4 total, MCTS-2k) = 32 cheap battles (~10 worker-s)
      - Round 3: 8 teams × 4 battles (MCTS-5k) = 32 battles (~25 worker-s)
      - Round 4: 4 teams × 6 battles (MCTS-20k, fresh seeds) = 24 battles (~65 worker-s)
      - Final verification: top-1/top-2 × 6 fresh-seed MCTS-20k battles (unbiased reporting!)
      Total wall-clock on 10 workers: **~15–20 seconds per box** instead of minutes, with much lower winner's curse!
    - *Cost:* ~35 lines of Python; massive immediate win with zero ML.

19. **Common Random Numbers (CRN) in Ranking & Selection / OCBA** — Chen et al. & 2022–2025 CRN-aware R&S literature `[web summary]`
    - *What it is:* Evaluating competing candidates on identical RNG streams positively correlates their noise, tightening confidence intervals on *differences* $Y_a - Y_b$ even when marginal win rates are noisy. Note: CRN creates **winner's curse** if you report the winning arm's score on the same seeds used to pick it.
    - *Concrete gain for RR:* `rr_team_probe.py` already pairs seeds per boss! Combine CRN within each Sequential Halving round (so team differences are sharp even at $n=2\text{–}4$), and **always** evaluate the selected winner on **disjoint fresh seeds** for unbiased win-rate reporting and dataset labeling (already noted in `RR_PLAN.md` Phase 1).
    - *Cost:* ~5 lines (seed offset per round + disjoint eval seeds).

---

### 1.8 Co-Optimising Two Coupled Components (Team/Set Builder ↔ Battler)

20. **Bilevel & Alternating Optimization for Co-Design (Hardware/Team + Control Policy)** — 2022–2025 arXiv literature (e.g., *Generalized Alternating Method for Bilevel Learning*, 2023; *Gradient-Based Co-Design with Differentiable Simulators*, 2024; *Learning to Solve Constrained Bilevel Control Co-Design*, 2025) `[web summary]`
    - *What it is:* Formalizes coupled design ($x$, here team + sets) and policy ($\pi$, here battler) as $\max_x J(x, \pi^*(x))$, showing that simultaneous noisy updates destabilize both, whereas **alternating optimization** (hold $\pi$ fixed to optimize $x$, hold $x$ fixed to improve $\pi$) converges reliably when the inner problem is near-stationary.
    - *Concrete gain for RR:* Directly validates and operationalizes the chicken-and-egg strategy:
      1. Fix the battler first using **team-independent anchors** (fair mirrors + reference teams).
      2. Run the **Separability Test**: measure rank correlation $\rho(\text{MCTS-2k}, \text{MCTS-20k})$ and $\rho(\text{MCTS-20k}, \text{Showdown search})$ over 100 `(six, boss)` pairs. If $\rho \ge 0.80$, fast-battler screening transfers cleanly to strong-battler play.
      3. Alternate outer rounds: *(Round 1)* fix Phase-0 MCTS battler → build Phase-1/2/3 team/set chooser; *(Round 2)* only if mirror/iteration diagnostics show battler limits on those good teams, upgrade the battler on the *discovered* winning-regime teams, then re-rank/re-train the scorer. Never jointly RL-train both from scratch.
    - *Cost:* Methodology + ~30-line separability script.

---

### 1.9 Offline Data Reuse

21. **MuZero Reanalyse / MuZero Unplugged** — Schrittwieser et al., NeurIPS 2021 `[web summary + recalled]`
    - *What it is:* Re-labels historical states/trajectories in an offline buffer with a stronger/updated search or value function rather than discarding old data whenever the policy improves.
    - *Concrete gain for RR:* Every battle played during Phase 1 screening, Phase 2 validation, and Phase 3 local search produces `(box, six, sets, boss)` outcomes **and** turn-level battle states. If we log them to a unified append-only archive (`logs/rr_archive.jsonl`), we never throw away compute: (a) bad teams rejected in Round 1 of Sequential Halving are *gold* negative examples for ranking losses; (b) if the battler upgrades later, we only re-evaluate a small calibration subset (or weight recent battler labels higher) rather than regenerating everything from scratch.
    - *Cost:* ~25 lines of logging in `rr_team_probe.py` / `engine_search.py`.

22. **Implicit Q-Learning (IQL)** — Kostrikov et al., ICLR 2022 `[web summary + recalled]` & **Calibrated Q-Learning (Cal-QL)** — Nakamoto et al., NeurIPS 2023 `[web summary]`
    - *What it is:* Offline value learning methods that prevent out-of-distribution (OOD) overestimation when optimizing over a static dataset—IQL via expectile regression without querying OOD actions, Cal-QL via calibrating conservative Q-values against a behavior policy baseline so online fine-tuning doesn't collapse.
    - *Concrete gain for RR:* Explains the exact pathology `RR_PLAN.md` warns about ("the argmax of a noisy net over many sixes selects its errors"): when a surrogate team scorer is queried on unseen `(six, boss)` combinations far from its training distribution, unconstrained optimization overestimates OOD teams. Two concrete defenses:
      1. **Ensemble uncertainty penalty (LCB / conservative screening):** Train a 3-seed or 5-head ensemble of the tiny Set-Transformer (~150k params each, trivial cost on RTX 5090) and rank candidates by $\mu - \lambda \sigma$ (for safe picks) or keep both top-$\mu$ and high-$\sigma$ candidates in the shortlist fed to Sequential Halving (active exploration!).
      2. **Never skip search verification:** The surrogate proposes/shortlists; real engine battles remain the judge.
    - *Cost:* ~15 lines (train 3 seeds / multi-head ensemble).

---

## Part 2 — Codebase Verification & Cheap Pre-Learning Diagnostics

Reading the actual repo files and result logs revealed **concrete bugs and mechanisms** that directly explain both open mysteries in the brief and must be fixed *before* any learning:

### 2.1 Diagnosis of the `--engine-opp ai` Drop (−13 pts Brock, −33 pts Misty)

Inspecting `results/rr_engine_ai_boxes30.jsonl` vs `results/rr_engine_boxes30.jsonl`, `results/rr_ai_diff.log`, `poke-engine/src/mcts.rs`, and `poke-engine/src/genx/rr_ai.rs` shows **two distinct, compounding causes**:

1. **Cause A — Turn-100 Stalling/Timeout Pathology (explains Misty's −33 pts):**
   - Verified directly in `results/rr_engine_ai_boxes30.jsonl`:
     - Minimax (`rr_engine_boxes30.jsonl`): Misty `mean_win = 51.1%`, `mean_d = 0.681`, `clean = 3.9%`, **0/30** boxes with low-death losses.
     - AI-model (`rr_engine_ai_boxes30.jsonl`): Misty `mean_win = 18.3%`, **`mean_d = 0.687` (identical deaths!)**, **`clean = 7.2%` (nearly 2× higher deathless wins!)**, and **9/30 anomalous low-death loss boxes** — including **6/30 boxes with `0%` wins and only `0.14–0.42` deaths/mon** (less than 1–2 fainted mons out of 6!).
   - In Pokémon you cannot lose with 5/6 mons alive unless `sb.play(..., 100)` (`scripts/rr_team_probe.py:151`) hits the **100-turn cap**, returning `winner = None`, which `one_battle` (`:153`) scores as `won = False`.
   - Every single one of those 0%-win/low-death Misty boxes (boxes 8, 10, 12, 15, 19) contains **`paras` (Dry Skin in Rain)**!
   - **Exact Rust mechanism (`poke-engine/src/mcts.rs:226`):** Under `rr`, a won terminal scores `sigmoid(eval - root_eval)` with `-60` per fainted mon of ours and **no turn/depth discount** (`gamma = 1.0`). Against a *deterministic* `rr_ai` (`s2_options.len() == 1` in `mcts.rs:99`), MCTS discovers that switching/stalling with Dry Skin Paras (or baiting `rr_ai`'s voluntary absorb/dodge-KO switches at `rr_ai.rs:1745–1855`) keeps `eval - root_eval > 0` with zero deaths—scoring *higher* than winning with 1 death—so it loops harmlessly until turn 100!

2. **Cause B — Zero-Support Exploitation of the 9.7% Model Error (explains Brock's −13 pts):**
   - On Brock there are **0/30** timeout boxes, yet wins still dropped `70.0% → 56.7%` (while deathless nudged up `16.1% → 17.2%`).
   - `results/rr_ai_diff.log` shows 90.3% overall agreement between Rust `rr_ai.rs` and Python `ai_util/rr_ai.py`, but only **78.0% on voluntary switches** (14 switch-vs-move flips) and **89.6% on forced switches**.
   - In `mcts.rs:95–125`, `vs_ai` puts **100% probability mass** on `rr_ai_choice_mut` and **0%** on every other legal enemy move. Whenever Rust predicts a switch that Python does not take (or vice versa), MCTS selects a maximally brittle move (staying in at low HP, greedy double-switch) that gets OHKO'd on Showdown.
   - *(Minor secondary code check in `rr_ai.rs:167` & `mcts.rs:418`: `rr_ai_choice_mut` calls `root_get_all_options()` even at non-root tree nodes, and `perform_mcts_vs_ai` forces `state.use_last_used_move = true` which subtly shifts move-history tracking inside the tree.)*

**Implication:** Do **not** permanently discard opponent-aware search before testing: (i) won-terminal floor + per-ply discount `0.995^depth`, (ii) fixing the top Rust-vs-Python RR AI porting diffs (58/600 cases, mostly damage-calc & switch threshold parity), and (iii) the **Safe Mixture Opponent** `(1-ε) rr_ai + ε UCB1`.

### 2.2 Verification of the Koga / Blaine Wall in Logs & Code

Inspecting `results/rr_ceiling_mega.log`, `results/rr_ceiling_norest_ev85.log`, `scripts/rr_team_probe.py`, `scripts/rr_probe.py`, `scripts/gauntlet.py`, and `data/rr/items.json` confirms that the 8% Koga / 0% Blaine ceiling is heavily confounded by **five non-ML bugs/omissions**:

1. **Mega-holder switch-only bug** (`rl/engine_search.py:84` vs `poke-engine/src/genx/state.rs:556`): Already queued for fixing; invalidated `rr_ceiling_mega` (Mega Kangaskhan at Blaine could only switch!).
2. **Duplicate evolved species & Truant in pool mode**: `results/rr_ceiling_mega.log` fielded **two Sawsbucks** (`deerling` + `deerlingwinter`) at Blaine, and `rr_ceiling_norest_ev85.log` fielded **Truant Slaking** at Koga. Only `searched_cands` (`rr_team_probe.py:292`) filters Truant and duplicate evolved species—`solo_stage`/`greedy`/`cover_team` do not.
3. **Broken set builder on late bosses**:
   - `gauntlet.STATUS_MOVES` puts **`rest` without Sleep Talk** on 4/6 Blaine mons (including a **Choice Band Sawsbuck carrying Rest**!).
   - `rr_probe.choose_nature` (`:269`) gives `+Spe` to any mixed attacker regardless of Base Speed—putting **Jolly (+Spe) on 30-Spe Coalossal** under Koga's permanent Tailwind!
   - `build_mon` (`rr_team_probe.py:108`) passes `trainer_idx=None` to `evolve()`, leaving stone evolutions un-evolved (e.g., `nuzleaf` instead of `shiftry` at Blaine).
   - `data/rr/items.json` has only **28 non-mega items** (mostly 1.2× type boosters, 1 Eviolite, 1 Leftovers, 1 Scarf at #83, 1 Band at #85, Life Orb at #91): **missing Focus Sash, Choice Specs, Assault Vest, and resist berries** (all major hardcore boss tools).
4. **Alphabetical lead & `--chooser searched` unwired for boxes**: `cover_team` (`:277`) returns `tuple(sorted(...))`, making the alphabetically first mon the lead! And `box_job` (`:577`) never calls `searched_cands` (`searched` only runs in `run_pool`).
5. **Real search-space scale**: Counting `rp.pools()` gives **91 non-double bosses**; sampled boxes grow from **9 mons at Brock** ($\binom{9}{6}=84$) → **18 at Misty** → **37 at Erika** → **52 at Koga** ($\approx 2\times 10^7$) → **73 at Blaine** ($\approx 1.7\times 10^8$) → **81 at Lance** ($\approx 3.2\times 10^8$). This explains why exhaustive/greedy screening suffices early (Brock–Misty) while surrogate-assisted search becomes genuinely necessary from Erika/Koga onward.

---

## Part 3 — Feature Engineering Specification

**Governing principle (from `CLAUDE.md`):** *Zero species identity one-hots* in any learned model (generality across held-out bosses and unseen mons). Every entity is described by its **mechanics, stats, types, moves, abilities/items, and exact physics-computed matchup interactions** from `data/rr/mod_data.json` + `ai_util/calc_bot.py` + `ai_util/matchup.py`.

### 3.1 Existing Hand Features in the Repo to Reuse (Do Not Rebuild)

| Existing Function / Data | Location | What It Computes | Where to Reuse |
|---|---|---|---|
| `calc_bot.dmg_frac` (memoized) | `ai_util/calc_bot.py:154` | Exact floored damage fraction (items, abilities, weather, terrain, burn, multi-hit, charge, resist berries) | Team scorer 6×6 edge matrix; set/item scorer; battle net active matchup |
| `calc_bot.stat` / `max_hp` | `ai_util/calc_bot.py:46,68` | Effective stats incl. boosts, para, weather/terrain speed, Tailwind | Speed-tier features (`faster`, `speed_ratio`, `outspeeds_count`) |
| `matchup.pair`, `ours`, `safe`, `race`, `edge` | `ai_util/matchup.py:30–100` | `ko_them`, `ko_us`, `faster`, `heals`, `hit`, `take`, balloon/berry spend, 3-tier cover flags | Direct 6×6 edge tensor + mon-level summary features for scorer & builder |
| `matchup.rows_for` | `ai_util/matchup.py:130` | Full box × boss matchup matrix in one dummy-battle pass per 6 mons | Precompute once per `(box, boss)` (~20 ms) for all 1,000+ candidate sixes! |
| `rr_ai._class` / `predict_class` | `ai_util/rr_ai.py:146`, `rr_ai.rs:706` | CFRU fighting-style class (`kill`, `setup_stats`, `stall`, `hazards`, `screens`, etc.) | Boss mon role token (tells the scorer/battler *how* the RR AI pilots each boss mon!) |
| `rr_ai.switch_in_score` | `poke-engine/src/genx/rr_ai.rs:1540` | Exact CFRU switch-in score & flags (`revenge`, `walls`, `resists`) | Optional boss-behavior feature in battle value net (Phase 4) |
| `gauntlet.item_value` / `mod_data.json` | `scripts/gauntlet.py:242`, `data/rr/mod_data.json` | Move flags (`priority`, `heal`, `drain`, `pivot`, `secondary`, `charge`, `recoil`), item tags, ability tags | Structured categorical/binary tags for moves, items, abilities |
| `lesson2_ppo.OBS` | `rl/lesson2_ppo.py:159` | Clean group-based observation dict (`core`, `status`, `types`, `stats`, `boosts`, `field`, `moves`, `foe_moves`, `matchup`) | Template for modular ablation-ready feature extraction |

### 3.2 Complete Feature Spec Table

#### A. Team Scorer, Surrogate & Pointer Builder Features (Pre-Battle: Box/Six vs Boss Six)

Because every `(box, boss)` problem evaluates a set of mons against 6 boss mons under a fixed arena, we represent the problem as a **bipartite graph**: $N$ player mon tokens (6 for Scorer; $N \le 81$ for Builder), 6 boss mon tokens, a $N \times 6$ **pairwise matchup edge tensor**, and 1 **global arena/boss context token**.

| Scope | Feature Group | Exact Features (Dimensions) | Scaling / Encoding | Reused From / Notes |
|---|---|---|---|---|
| **Per-Mon Token** (ours & theirs, shared schema) | `stats_base` (10) | Level `/100`, actual `hp/atk/def/spa/spd/spe` at cap `/500`, `bst/720`, `phys_spec_ratio = atk/(atk+spa)`, `bulk = (hp*(def+spd))/2e5`, `is_nfe` | Continuous `[0, ~1.5]` | From `Mon` / `mod_data.json`; no species ID |
| | `types` (18) | Multi-hot 18 Pokémon types | Binary `{0,1}` | Reuse `lesson2_ppo.TYPES` |
| | `role_flags` (10) | `is_lead`, `is_player_side`, `can_mega`, `is_already_picked` (builder only), `has_recovery`, `has_priority`, `has_pivot`, `has_status_move`, `has_hazard_or_removal`, `boss_ai_class` (1-hot or bucket for enemy; 0 for ours) | Binary / 1-hot (10+10 for AI class on enemy) | Reuse `rr_ai._class` for boss mons! |
| | `moves_agg` (16) | Top-2 STAB effective powers `/200`, coverage type count `/4`, physical/special/status move counts `/4`, max priority `/5`, min accuracy of attacks, has recoil/charge/drain/spread/setup/protect/weather-set flags | Continuous & binary | Derived from `mod_data.json["moves"]` |
| | `item_tags` (14) | Functional item buckets (no sparse 577-item ID!): `has_item`, `mega_stone`, `choice_band`, `choice_specs`, `choice_scarf`, `life_orb`, `focus_sash`, `eviolite`, `leftovers_or_heal_berry`, `assault_vest`, `type_booster`, `resist_berry` (+ type match flag), `air_balloon`, `heavy_duty_boots` | Binary `{0,1}` | Generalizes to any newly unlocked item without retraining embeddings |
| | `ability_tags` (18) | Functional ability buckets: `type_immune_absorb` (Water/Elec/Fire/Ground/Grass), `intimidate`, `weather_speed` (active in arena!), `weather_setter`, `weather_negate` (Cloud Nine/Air Lock), `sturdy_multiscale`, `damage_mult` (Huge/Pure Power, Tech, Sheer Force, Tough Claws, Adaptability), `magic_guard_unaware`, `prankster`, `regenerator`, `status_immune`, `guts_quickfeet`, `defensive_mult` (Filter/Fur Coat/Ice Scales/Thick Fat), `revive_phoenix` | Binary `{0,1}` | Covers >90% of battle-deciding RR abilities cleanly without overfitting 342 IDs |
| | `matchup_summary` (14) | Aggregated from the mon's row vs the 6 enemies: `n_safe/6`, `n_race/6`, `n_walls/6` (`ko_us>=4`), `n_outspeeds/6`, `n_ohko_them/6`, `n_ohko_us/6`, `mean_edge`, `max_edge`, `min_edge` (liability score!), `lead_safe`, `lead_race`, `lead_edge`, `best_hit_mean`, `worst_take_max` | Continuous | Directly fixes the B3 "Carnivine liability" blind spot (`min_edge`, `n_ohko_us`) |
| **Pairwise Edge Tensor** ($N \times 6$, our mon $i$ vs boss mon $j$) | `pair_physics` (12) | `hit` (best dmg frac, clipped `[0,2]`), `take` (best enemy dmg frac, clipped `[0,2]`), `1/ko_them`, `1/ko_us`, `faster` (incl. Tailwind/weather/Scarf!), `speed_ratio = log2(spe_i/spe_j)` clipped `[-2,2]`, `safe`, `race`, `edge/10`, `switch_in_safe = (race & take < 0.5)`, `we_have_prio_ko`, `they_have_prio_ko`, `opp_heals`, `we_immune_to_best` | Continuous & binary; injected as **cross-attention edge bias** | Directly from `matchup.rows_for` + `calc_bot` (~20 ms per box!) |
| **Team-Level Aggregate** (for tabular GBT probe & global residual) | `team_cover` (16) | Fraction of boss mons with ≥1 `safe` answer, ≥2 `safe` answers, ≥1 `race` answer, ≥1 `switch_in_safe` answer; **min-over-enemies** best `edge` (weakest link!); **max-over-enemies** count of our mons OHKO'd; lead-vs-lead `safe/race/edge`; team type-weakness overlap (max # of our mons weak to one boss attack type); # unique megas (`0/1`); # mons with recovery/priority/speed-control | Continuous | Explicitly encodes Nuzlocke set-cover + defensive backbone + liability |
| **Arena / Context Token** (1 per boss) | `arena` (12) | Permanent weather (7: none/sun/harshsun/rain/heavyrain/sand/snow), permanent terrain (5: none/elec/grassy/misty/psychic), `p2_tailwind`, `trick_room`, `cap/100` | 1-hot + scalar | From `rr_probe.arena` |

*Total per-mon token dim:* ~100 floats. *Pairwise edge dim:* 12–14 floats. *Zero species IDs.*

#### B. Set / Item Refinement Features (Per-Candidate Set Delta, Phase 3)

For a fixed 6-mon team, evaluating a candidate set edit on mon $i$ (nature swap, item swap, 1 move swap) requires **zero new model architecture**: re-run `matchup.pair` for mon $i$ alone (6 pairs, **<0.2 ms** via memoized `calc_bot.dmg_frac`), update mon $i$'s token and row $i$ of the $6 \times 6$ edge tensor, and score with the exact same Phase 2 Set-Transformer surrogate! Add 4 explicit set-coherence sanity flags to the token: `choice_with_status_move`, `rest_without_sleeptalk`, `speed_nature_outspeeds_extra_foes` (delta in `n_outspeeds` vs +Atk/SpA nature), and `item_hits_new_ko_threshold` (delta in `n_safe + n_race`).

#### C. Battle-State Features (For Phase 4 Value / Prior Net or Rust Leaf Evaluator)

Extends `rl/lesson2_ppo.py`'s `OBS` groups to full 6v6 Radical Red with exact `poke-engine` state fields (`rl/convert.py`):

| Group | Features | Dim | Notes |
|---|---|---|---|
| `core` | Per mon (12 mons): `hp_frac`, `alive`, `is_active`, `just_switched_in`, `mega_evolved`, `can_mega`; plus `force_switch` (2), `turn_frac = min(turn/60, 1)` | 75 | `turn_frac` prevents turn-100 stall blindness |
| `status_vol` | Per mon (12): 7 statuses (`none/brn/par/psn/tox/slp/frz`), `toxic_count/6`, `sleep_or_rest_turns/3`; active volatiles (2×10): `sub_hp_frac`, `leech_seed`, `taunt`, `encore`, `confusion`, `trapped`, `yawn`, `protect_streak`, `charging`, `choice_locked` | 128 | Directly maps from `convert.py` |
| `stats_boosts` | Per mon (12): effective `hp/atk/def/spa/spd/spe / 500`, 18 types; active boosts (2×7): `atk/def/spa/spd/spe/acc/eva / 6` | 302 | Includes effective stats under weather/Tailwind |
| `field` | Weather (8) + `weather_perm`, Terrain (5) + `terrain_perm`, `trick_room`, per-side hazards & screens (2×10: SR, spikes/3, tspikes/2, web, reflect, lightscreen, auroraveil, tailwind + `tw_perm`, wish, future_sight) | 36 | Distinguishes permanent arena Tailwind/weather from temporary! |
| `active_moves` | Our 4 moves + foe's 4 moves (full info!): type (18), BP/150, cat (3), acc, priority/5, **exact `hit_frac` on active foe/us**, `is_ko` (`hit >= opp_hp`), STAB, usable/pp>0, status/heal/boost/pivot flags | 2×120 = 240 | Computing exact `hit_frac` removes the network's burden of memorizing the Gen 9 damage formula |
| `bench_matchup` | For each of our 6 mons vs foe active, AND each of foe's 6 mons vs our active: `(best_hit_frac, worst_take_frac, faster, switch_in_survives, safe, race)` | 12×6 = 72 | Gives instant switch-target visibility (replaces memorizing 18×18 type chart × stats) |
| `item_ability` | Functional item tags (14) + ability tags (18) per mon (12 mons) | 384 (or per-token in attention) | Zero species IDs |
| `rust_leaf_lite` *(if Rust-native leaf evaluator)* | Compact ~42-float hand-selected subset: alive counts, our dead count, HP sums, active-vs-active `hit_frac`/`take_frac`/`faster`/`ko_in_1`, bench max `safe`/`race` counts, boosts, status, hazards, screens, permanent-aware Tailwind | 42 | Evaluable in **~1–2 µs** purely in Rust inside `evaluate.rs`! |

### 3.3 How to Test Feature Adequacy (Anti-Undertraining Protocol)

Directly addressing the hard-won lesson from attempt #3 (`CLAUDE.md:69`: *"The obs is missing X" was undertraining every time*):

1. **Gate A — Tabular Oracle Probe First (5 min):** Before debugging any neural architecture, train a fast **LightGBM / GBDT** on the flat `team_cover + lead + aggregated matchup` features and compare against a **memorization upper bound** (GBDT *with* species IDs on *training* bosses only, and on held-out boxes of seen bosses).
   - If species-free GBDT matches species-ID GBDT within 0.03 Spearman on held-out boxes of seen bosses → **features carry the species signal** (no missing mechanics!).
   - If species-ID GBDT beats species-free GBDT by >0.08 Spearman → inspect the top error cases (`|y - \hat{y}|` largest) to identify the exact missing ability/item/move tag (e.g., an unbucketed RR custom ability) and add that mechanical tag, NOT species IDs.
2. **Gate B — Capacity & Training-Data Learning Curve (diagnoses undertraining vs missing features):**
   - Always plot validation Spearman/Brier at **25%, 50%, 100%** of training data and **2× training epochs**.
   - Rule: **Never add a feature** while train loss is still dropping or while the 50%→100% data curve gains >0.03 Spearman. Fix training/data first.
3. **Gate C — Leave-One-Group-Out (LOGO) Ablation & Permutation Importance:**
   - Reuse `lesson2_ppo.py`'s zero-out importance harness (`importance()`), plus retrain ablations dropping one group at a time (`-pair_physics`, `-item_tags`, `-ability_tags`, `-role_flags`, `-arena`). Keep only groups that improve held-out-boss Spearman by $\ge 0.02$ or Brier by $\ge 2\%$ across 3 seeds. Delete dead groups (minimal-code rule).
4. **Gate D — Teacher Agreement Sanity Check (for Battle Net):**
   - On a fixed 10k-position dataset from `EngineSearcher`, the policy head must converge to ≥85% top-1 agreement on training positions before touching held-out bosses or search integration. If it cannot overfit 10k positions, the bug is optimization/masking, not features.

---

## Part 4 — Complete Phased Plan

### Overview & Addressing the Chicken-and-Egg Loop

We break the `battler ↔ team builder ↔ set builder` circle with **strict decoupling + alternating optimization**:
1. **Fix the battler independently first** using team-independent anchors: *fair mirrors* (same boss six vs RR AI, no asymmetric arena like Koga's p2-only Tailwind) next to the `AI-vs-AI` baseline, plus user/community reference teams if needed.
2. **Fix deterministic set-builder bugs** before searching or learning (garbage sets poison every downstream label).
3. **Measure Separability:** Verify on 100 `(six, boss)` pairs that cheap search (`MCTS-2k`) preserves the ranking of full search (`MCTS-20k` and Showdown search) with Spearman $\rho \ge 0.80$.
4. **Hold the battler fixed while building the team/set chooser** (Phase 1 non-learned Sequential Halving → Phase 2 Set-Transformer Surrogate + LNS → Phase 3 Pointer Builder + Set Refiner), storing every evaluated battle in a single reusable offline archive.
5. **Only return to battler learning (Phase 4)** if mirror anchors or a `20k vs 200k` iteration probe on the *newly discovered strong teams* proves play strength is still leaving ≥10 pts on the table.

---

### Phase 0 — Cheap Diagnostics, Engine Fixes & RR-AI-Search Resolution

- **Goal:** Remove known engine/set-builder bugs, resolve the `--engine-opp ai` mystery cleanly, and establish trustworthy battler anchors and legal Koga/Blaine ceilings *before* writing any ML code.
- **Method:**
  1. **Queued engine fixes** (~20 lines): strip `-mega` in `rl/engine_search.py:84`; treat Tailwind $\ge 90$ as permanent/flat in `poke-engine/src/genx/evaluate.rs`; mark spent Phoenix Down in `convert.py` / `sim_bridge.js`.
  2. **Fix Nuzlocke terminal & stalling objective** (~15 lines in `poke-engine/src/mcts.rs:222–236` + `rr_team_probe.py:152`):
     - Floor won terminals at `0.55 + 0.45 * (alive_p1 / 6.0)` so a costly win *always* beats an unfinished fight, while preserving preference for fewer deaths.
     - Add a tiny depth discount (`0.995^depth`) on non-terminal leaves to break infinite heal/switch loops.
     - Explicitly log `timeouts` (`sb.turn >= 100 and st["winner"] is None`) in `one_battle`.
  3. **RR-AI-model search diagnostic** (~40 lines Rust + 1 small benchmark):
     - Fix the top 3 disagreement clusters in `results/rr_ai_diff.log` (voluntary switch thresholds + forced-switch damage parity) to raise `rr_ai.rs` agreement from 90.3% → ≥97%.
     - Implement **Safe Mixture Opponent** in `mcts.rs`: at each tree node, with probability $1-\epsilon$ sample `rr_ai_choice_mut`, with probability $\epsilon$ pick `s2`'s UCB1 move. Sweep $\epsilon \in \{0.0, 0.15, 0.30, 1.0\}$ (where $1.0 =$ pure minimax) on Brock/Misty 30-box + Erika/Koga(no-arena)/Blaine fair mirrors ($n=48$) next to the `AI-vs-AI` baseline.
  4. **Set-builder & pool hygiene fixes** (~45 lines in `scripts/gauntlet.py`, `scripts/rr_probe.py`, `scripts/rr_team_probe.py`):
     - Remove `rest` from `STATUS_MOVES` (or allow only with Sleep Talk); prevent Choice items on mons with >0 status moves unless all 4 moves are attacks (or drop status move when given Choice).
     - Fix `choose_nature`: only pick `+Spe` on mixed sets if base Speed ≥ 70 (and never under permanent Trick Room); otherwise boost the stronger attacking stat or bulk.
     - Filter Truant and duplicate evolved species in `run_pool` too; enable RR stone evolutions at Celadon (`trainer_idx` gate); audit `data/rr/items.json` for missing care-package items (Focus Sash, Choice Specs, Assault Vest, resist berries).
- **Data:** Existing Brock/Misty 30-box seeds + Erika/Koga/Blaine fair mirrors ($n=48$) + Koga/Blaine pool ceiling.
- **Features:** None (pure engine/rules).
- **Rough code size & files:** ~120 lines total across `poke-engine/src/mcts.rs`, `poke-engine/src/genx/evaluate.rs`, `poke-engine/src/genx/rr_ai.rs`, `rl/engine_search.py`, `scripts/gauntlet.py`, `scripts/rr_probe.py`, `scripts/rr_team_probe.py`.
- **Machine-time cost:** ~30–45 min wall-clock on 10 workers (2–3 min per 30-box table, ~2 min per $n=48$ mirror).
- **Measurable gates:**
  - **Gate 0a (Battler Anchor):** Fair mirrors (Erika, Koga no-arena, Blaine) $\ge 75\%$ win rate ($n=48$, SE $\approx \pm 6\%$) AND beats `AI-vs-AI` baseline by $\ge +20$ pts; Misty timeouts = 0%. Adopt best $\epsilon^*$ only if it beats pure minimax by $\ge +5$ pts on Brock/Misty without increasing deaths/mon.
  - **Gate 0b (Pool Ceiling):** With `--chooser searched` + cleaned sets, pool ceiling $\ge 40\%$ at both Koga and Blaine (on fresh verification seeds).
- **If a gate fails:**
  - If **Gate 0a fails**: run a 2k vs 20k vs 200k iteration sweep + Showdown-search cross-check on the failing mirror; inspect 5 lost battle traces to distinguish engine conformance bug vs `evaluate.rs` blind spot. Do **not** proceed to team-builder ML until mirrors pass ≥75%.
  - If **Gate 0b fails** (while Gate 0a passes): test 2–3 human/community reference teams for Koga & Blaine. If reference teams win $\ge 40\%$, the gap is purely specific counter-archetypes (weather negation like Cloud Nine/Air Lock, priority/Thunder Wave vs Tailwind, specific TMs/items) → fix those set rules before Phase 1. If reference teams also lose ($\le 15\%$), inspect battle traces for ROM/arena fidelity bugs (e.g. confirm whether Koga's Tailwind is truly permanent in RR 4.1).

---

### Phase 1 — Sample-Efficient Non-Learned Search Chooser (CRN Sequential Halving + Local Search) & Offline Data Archive

- **Goal:** Build the strongest possible *zero-ML* team+lead chooser using adaptive bandit allocation (Sequential Halving with CRN) + 1-swap local search; prove separability of cheap vs full search; collect the offline dataset across all 91 singles bosses.
- **Method:**
  1. **Separability & Turn-0 Baseline Diagnostic** (~40 lines): Evaluate 100 diverse `(six, boss)` pairs across Brock–Blaine with `MCTS-1k`, `MCTS-2k`, `MCTS-20k`, and turn-0 root value `V_0`. Compute within-box Spearman $\rho$.
  2. **CRN Sequential Halving Chooser** (`--chooser sh`, wired into both `run_pool` and `box_job`):
     - Generate 32 diverse initial sixes from `box`: cover six, greedy six, 20 matchup-weighted random draws (one answer per enemy mon, respecting dupes + ≤1 mega), and 10 lead/1-swap perturbations.
     - **Round 1:** 32 teams × 2 CRN battles at cheap search (`MCTS-2k`, ~0.3 worker-s/battle) → keep top 16 (by `(win, progress, -deaths)`).
     - **Round 2:** 16 teams × 2 more CRN battles at `MCTS-2k` → keep top 8.
     - **Round 3:** Expand top 4 with 4 targeted 1-swap/lead neighbors (8 total) × 4 CRN battles at `MCTS-5k` → keep top 2.
     - **Round 4 (Unbiased Verification):** Top 2 × 6 **fresh-seed** `MCTS-20k` battles → pick winner and report on its disjoint seeds (eliminates winner's curse!).
  3. **Unified Offline Logger:** Append every battle's `(boss, cap, arena, box_species, team_sets, lead, mcts_iters, won, deathless, deaths_per_mon, per_mon_died[6], progress, turns, timeout, root_val_t0)` to `logs/rr_archive.jsonl`, plus optionally sample 1 in 5 battle trajectories' turn states (`state.to_string()`, root visit distribution, outcome) for Phase 4.
  4. **Boss Split (Strict Generality):** Split the 91 singles bosses systematically: **every 4th boss by progression = 23 held-out bosses** (never trained on), remaining 68 training bosses; within training bosses, hold out 20% of box seeds.
- **Data:** Run `--chooser sh` on `gyms30` (6 gyms × 30 boxes = 180 boxes) + a broad sweep of 10 boxes × 68 training bosses = 680 boxes → yields **~25,000–35,000 evaluated `(six, boss)` records** (~100k fast battles total) automatically as a byproduct of running the chooser!
- **Features:** Existing `matchup.rows_for` for candidate generation; full Phase 2 feature tensors computed on-the-fly from logged sets.
- **Rough code size & files:** ~120 lines in `scripts/rr_team_probe.py` (or a dedicated ~150-line helper `scripts/rr_select.py` imported by `rr_team_probe.py`) + ~25 lines logging in `rl/engine_search.py`.
- **Machine-time cost:**
  - Separability probe: ~3 min on 10 workers.
  - `gyms30` (180 boxes × ~120 worker-s/box ÷ 10 workers): **~35 min**.
  - 68-boss × 10-box archive sweep: **~2.2 hours** on 10 workers (one clean overnight/afternoon job).
- **Measurable gates:**
  - **Gate 1a (Separability):** Within-box Spearman $\rho(\text{MCTS-2k}, \text{MCTS-20k}) \ge 0.75$ and top-quartile recall $\ge 75\%$.
  - **Gate 1b (Chooser Lift):** On `gyms30`, `--chooser sh` beats `--chooser cover` by $\ge +10$ pts on Misty–Erika AND reaches $\ge 25\%$ mean win on Koga/Blaine boxes (outside paired 95% CI).
  - **Gate 1c (Compute Decision):** If `--chooser sh` already clears late bosses satisfactorily at $< 30\text{ s/box}$ on 10 workers and wider search (128→SH) gains $< 3$ pts, **skip Phases 2–3 and jump directly to Phase 5 (encounter routing)**! Proceed to Phase 2 only if larger candidate pools (500–2,000 teams on 50–80 mon boxes) show meaningful headroom that raw search cannot afford to screen.
- **If a gate fails:**
  - If **Gate 1a fails** (`MCTS-2k` noisy vs `20k`): bump Round 1–2 budget to `MCTS-5k` (still ~3× faster than 20k) or test `calc_bot + matchup` 1-ply rollout as Round 1 filter.
  - If **Gate 1b fails** on Koga/Blaine boxes while pool ceiling passed: diagnose whether *no winning six exists in random boxes* (box-coverage wall → jump to Phase 5 encounter routing to see what catches are needed) vs *winning six exists in box but 32 random candidates missed it* (candidate-generation wall → proceed to Phase 2 surrogate + LNS).

---

### Phase 2 — Feature Adequacy Probes & Edge-Biased Set-Transformer Surrogate (Filter, Never Judge)

- **Goal:** Train a fast, calibrated, species-free surrogate scorer $f(\text{six}, \text{boss}, \text{arena}) \to (P(\text{win}), P(\text{death}_{1..6}), P(\text{progress bin}))$ on the Phase 1 offline archive so we can screen **1,000+ candidate sixes (via Surrogate-Assisted LNS) in <50 ms** and feed the top 16–32 into Phase 1's Sequential Halving.
- **Method:**
  1. **Step 2a — Tabular Feature-Adequacy Probe (~80 lines, ~2 min CPU):**
     - Train LightGBM/Ridge on `team_cover + aggregated matchup` features from `logs/rr_archive.jsonl` (68 train bosses → 23 held-out bosses).
     - Run the **Species-ID Oracle Gap Test** (Section 3.3 Gate A) and **25%/50%/100% learning curve** (Gate B). Fix any missing mechanical tag before building PyTorch models.
  2. **Step 2b — Edge-Biased Set-Transformer Surrogate (~200 lines PyTorch, `rl/team_scorer.py`):**
     - Architecture (~150k–250k params): Linear projections of 6 player mon tokens, 6 boss mon tokens, and 1 arena token → 2 layers of self-attention over our 6 mons + 2 layers of **edge-biased cross-attention** to the 6 boss mons (using the $6 \times 6 \times 14$ `matchup.rows_for` edge tensor directly as attention bias + value gating).
     - **Multi-task losses (weighted by battle count $n_j$ per candidate):**
       - $\mathcal{L}_{\text{rank}}$: within-`(box, boss)` **Plackett-Luce / pairwise Bradley-Terry ranking loss** on realized `(win, progress)` (primary shortlist objective!).
       - $\mathcal{L}_{\text{win}}$: Binary Cross-Entropy on `win_rate`.
       - $\mathcal{L}_{\text{death}}$: 6-way per-mon BCE on `per_mon_died` (KataGo-style dense auxiliary signal, also directly usable for Nuzlocke risk).
       - $\mathcal{L}_{\text{prog}}$: **HL-Gauss** cross-entropy over 21 bins on `progress ∈ [0, 2]` (dense signal even on 0%-win fights!).
     - Train a **3-seed ensemble** (~15 min total on RTX 5090) to obtain both mean $\mu$ and epistemic uncertainty $\sigma$.
  3. **Step 2c — Surrogate-Assisted Local Search (LNS) + Sequential Halving (`--chooser surrogate`):**
     - Precompute `matchup.rows_for` for the box once (~20 ms).
     - Seed with cover + 20 matchup draws; run **4 iterations of 1-swap & lead-swap local search** scoring ~400–800 neighbors per step in one batched GPU forward pass (~10 ms/step!).
     - Take **top-12 by conservative score ($\mu - 0.5\sigma$)** + **4 high-uncertainty/diverse picks** (to avoid OOD blind spots) → run Phase 1's CRN Sequential Halving (16 → 8 → 4 → 2 → verify).
- **Data:** Phase 1 offline archive (`~25k–35k` `(six, boss)` records across 68 train bosses; evaluated strictly on the 23 held-out bosses + 20% held-out boxes).
- **Features:** Section 3.2A (zero species IDs; ~100-dim mon token + 14-dim pairwise edge tensor + 12-dim arena token).
- **Rough code size & files:** ~280 lines total: `rl/rr_features.py` (~120 lines, shared feature extractor + tabular probe) + `rl/team_scorer.py` (~160 lines, Set-Transformer + ensemble training + LNS shortlister).
- **Machine-time cost:**
  - Feature extraction + LightGBM probe + ablations: ~10 min CPU.
  - 3-seed Set-Transformer training (30 epochs on ~30k examples, RTX 5090): **~5–10 min GPU**.
  - Held-out evaluation on 23 held-out bosses + `gyms30`: **~35 min** on 10 CPU workers.
- **Measurable gates (all on held-out bosses & held-out boxes):**
  - **Gate 2a (Ranking):** Within-box Spearman $\rho \ge \max(0.65, \rho_{\text{root-val}} + 0.10, \rho_{\text{GBT}} + 0.03)$.
  - **Gate 2b (Calibration):** Win Brier score $\ge 15\%$ better than per-boss base rate; per-mon death calibration error $\le 0.08$ per decile.
  - **Gate 2c (End-to-End Shortlist):** On held-out bosses (and `gyms30`), `--chooser surrogate` (16-team SH) achieves $\ge +5$ pts over Phase 1's 32-team SH at $\le 0.6\times$ battle budget, OR matches an expensive 128-team SH oracle within 3 pts at $\le 1/6$ compute.
- **If a gate fails:**
  - If **Gate 2a beats GBT by <0.03**: **delete the Set-Transformer** and keep LightGBM as the surrogate (minimal-code rule!).
  - If **both GBT and Set-Transformer fail Gate 2a**: run the Section 3.3 diagnostic (learning curve + species-ID oracle gap). If under-data, collect 1 more PLR-prioritized round on mid-difficulty bosses; if missing mechanics, add the top missing ability/item tag from residual inspection.
  - If **Gate 2c shows surrogate picks OOD traps** (high surrogate $\mu$, low search win): increase ensemble penalty $\lambda$ on $\sigma$, add the newly discovered failure teams to `logs/rr_archive.jsonl`, and retrain 1 round (hard-negative mining / DAgger for surrogates).

---

### Phase 3 — PLR-Curated Expert Iteration, Pointer Builder & Targeted Set/Item Coordinate Refiner

- **Goal:** Only if Phase 2's 1-swap LNS still misses synergistic multi-mon cores on huge late-game boxes (52–81 mons, Koga–Lance) OR if set/item customization (Speed vs Attack nature, Focus Sash / Choice Scarf / Resist Berry / 1 utility move) adds substantial wins.
- **Method:**
  1. **PLR-Style `(box, boss)` Curriculum Collector (~30 lines):** Sample next training boxes with probability proportional to `(surrogate_vs_search_regret + 0.1) * I(0.05 < box_best_win < 0.95)`, skipping trivial 100% and hopeless 0% boxes.
  2. **Autoregressive Pointer Builder (~130 lines, `rl/team_builder.py`):**
     - Reuses the Phase 2 encoder over all $N$ box mons + 6 boss mons; a tiny 1-layer pointer decoder picks 6 mons sequentially (pick 1 = lead, picks 2–6 = team) with hard legality masks (no dupes, ≤1 mega stone, no Truant).
     - **Training (no from-scratch RL):** Behavioral cloning / weighted cross-entropy on the top-1/top-2 search-verified sixes per `(box, boss)` (POMO-style permutation augmentation over slots 2–6!), followed by 2 rounds of **search-verified Expert Iteration** (builder samples 16 diverse teams via Gumbel-Top-$k$ → merged with Phase 2 LNS → Sequential Halving verifies → archive grows → retrain both Scorer and Builder).
  3. **Targeted Set / Item Coordinate Refiner (~60 lines, `scripts/rr_set_refine.py`):**
     - Run **only on the top-2 teams** after team selection (keeps compute tiny).
     - For each of the 6 mons, generate $\le 8$ legal single-slot edits: alternative nature (`+Spe` vs `+Atk/SpA`), top-3 high-leverage available items (Focus Sash, Choice Scarf/Band/Specs, Leftovers, boss-type Resist Berry), and 1 flex move swap (Protect, strongest priority, Thunder Wave/Will-O-Wisp, weather-setter/negator if legal).
     - Score all ~48 single-set edits in <5 ms with the Phase 2 surrogate → verify top-4 edits + baseline via 4-battle CRN Sequential Halving.
- **Data:** Accumulated offline archive (~50k–80k `(six, boss)` records after 2 EI rounds).
- **Features:** Identical to Section 3.2A + 3.2B (4 set-coherence delta flags).
- **Rough code size & files:** ~220 lines total (`rl/team_builder.py` ~140 lines + `scripts/rr_set_refine.py` ~80 lines).
- **Machine-time cost:** ~2–3 h per EI round (10 CPU workers for verification + 10 min RTX 5090 training); 2 rounds = ~5 h total.
- **Measurable gates (held-out bosses, especially late-game cap $\ge 50$):**
  - **Gate 3a (Pointer Builder):** Adds $\ge +4$ pts win rate or $-0.04$ deaths/mon over Phase 2 LNS-only at equal wall-clock budget.
  - **Gate 3b (Set Refiner):** Paired lift $\ge +5$ pts or $-0.05$ deaths/mon on Koga–Lance held-out boxes (adds <15 s/box).
- **If a gate fails:** Keep whichever component passed (Set Refiner and Pointer Builder are independent) and **delete the failing one**. Do not add REINFORCE/PPO on team building if search-verified EI stalls (respects attempt #3's lesson).

---

### Phase 4 — Battler Learning (Strictly Conditional: Only If Battler Is Now the Bottleneck)

- **Trigger / Entry Gate (cheap diagnostic first, ~15 min):** Take 30 hard battles using our best Phase 2/3 teams (Erika–Lance) and compare `MCTS-20k` vs `MCTS-200k` (10× budget) AND check fair mirrors. **Only enter Phase 4** if either (a) fair mirrors remain $< 85\%$, OR (b) `MCTS-200k` beats `MCTS-20k` by $\ge +8$ pts (proving search quality, not team/set quality, is the remaining wall).
- **Goal:** Improve play strength (especially attrition/death avoidance and multi-turn tactical setups) without breaking the ~2.7 worker-s/battle budget.
- **Method (Two-Tier, Cheapest First):**
  - **Tier 4a — Rust-Native Learned Leaf Evaluator (Minimal-Code Sweet Spot, ~150 lines Python + ~60 lines Rust):**
    - Train a tiny **2-layer MLP (`42 → 32 → 1`, ~1,400 weights)** or ridge/logistic model in PyTorch on logged battle states from the Phase 1–3 archive (`~200k–500k` positions), predicting Nuzlocke Showdown outcome (`win` + `deaths` via HL-Gauss + death heads) from the 42 `rust_leaf_lite` features (Section 3.2C — which **already include `hit_frac` and `effective_speed`** computed natively in `rr_ai.rs`!).
    - Export weights as a tiny flat `f32` file loaded once at startup in `poke-engine/src/genx/evaluate.rs` (~40 lines pure Rust, zero FFI, zero Torch/ONNX dependency!).
    - Runs in **~1–2 µs per leaf** inside the existing 20k-iteration MCTS! Directly fixes `evaluate.rs`'s biggest flaw (it currently ignores damage/speed matchups completely).
  - **Tier 4b — Only if Tier 4a saturates: Gumbel Small-Budget pUCT (`rl/battle_net.py`, ~250 lines Python + ~150 lines Rust):**
    - Train a species-free ~300k-param battle policy+value net (Section 3.2C features, HL-Gauss value + 6 death heads + MCTS visit-share policy target) on logged engine-search battles (Numbrion/Searchless-Chess style supervised pretraining first, then 1–2 rounds of **Gumbel Sequential Halving** self-improvement against the Safe Mixture Opponent).
    - Because Gumbel root planning works at **64–256 simulations**, batched evaluation (or Rust `ort`/Candle) becomes fast enough to beat 20k heuristic iterations.
- **Data:** Turn-level trajectories already logged in Phase 1–3 (`logs/rr_archive.jsonl`), augmented with 200k positions from 68 training bosses (never the 23 held-out bosses).
- **Features:** `rust_leaf_lite` (42 dims) for Tier 4a; full Section 3.2C for Tier 4b.
- **Rough code size & files:** Tier 4a: ~150 lines (`rl/fit_leaf_eval.py`) + ~60 lines Rust (`evaluate.rs`). Tier 4b: ~250 lines (`rl/battle_net.py`) + ~150 lines Rust (`mcts.rs`).
- **Machine-time cost:** Tier 4a: ~10 min GPU training + ~30 min 30-box & mirror verification. Tier 4b: ~1–2 days compute (GPU training + CPU evaluation).
- **Measurable gates:**
  - **Offline Gate (before any Rust edits):** Held-out boss battle-outcome Brier improves by $\ge 15\%$ over current `evaluate.rs` sigmoid, and 1-ply + learned value agrees more with `MCTS-200k` than `evaluate.rs` does.
  - **Online Gate:** $\ge +5$ pts win rate OR $-0.05$ deaths/mon on held-out bosses AND fair mirrors, at $\le 1.2\times$ wall-clock per battle, with zero regression on Brock/Misty 30-box ($\pm 3$ pts).
- **If a gate fails:** Keep `evaluate.rs` with Phase 0 fixes; do not merge Rust pUCT complexity. (Remember: attempt #3's learned leaf value failed twice when data/features were weak — the offline gate protects us from repeating that.)

---

### Phase 5 — Encounter Routing (Arithmetic Over the Validated Scorer)

- **Goal:** Decide which encounter method (day/night grass, rod, Surf, gift) to use per route, and exploit dupes-clause ordering to maximize run survival across upcoming bosses.
- **Method:** Pure decision-theoretic expected value (`scripts/rr_route.py`, ~100 lines) on top of the Phase 2/3 surrogate scorer (zero RL): for each open location/method, compute $\mathbb{E}_{m \sim P(\text{encounter} \mid \text{dupes})}[\max_{S \subseteq B \cup \{m\}} \text{Score}(S, \text{upcoming bosses})]$.
- **Machine-time cost:** <1 min CPU/GPU per full run simulation; verify 30 routed runs vs 30 random-box runs (~30 min on 10 workers).
- **Gate:** Routed boxes beat random-method boxes by $\ge +10$ pts clear rate on Misty–Blaine (or cut deaths/mon by $\ge 25\%$).

---

## Part 5 — What We Will NOT Do (and Why)

1. **No PPO / from-scratch RL for battling, team choice, or routing.** Attempt #3 proved from-scratch PPO caps at ~14%, BC→PPO finetuning oscillates without beating BC, and lesson 2 showed PPO vs a fixed opponent overfits that script (`0–22%` vs MCTS). Search + supervised/ranking learning + verified Expert Iteration dominates in sample efficiency and stability.
2. **No learned enemy-AI model.** The RR AI is a known ~670-line script (`ai_util/rr_ai.py`) already 90.3% ported to Rust (`rr_ai.rs`). Fixing the remaining ~58/600 porting bugs takes half a day and gives 98%+ exact fidelity; learning an approximate copy adds code, noise, and distribution shift.
3. **No species ID embeddings or boss ID embeddings.** Violates `CLAUDE.md`'s generality constraint and memorizes training bosses/trios instead of transferring to the 23 held-out bosses.
4. **No Python-per-leaf neural MCTS callback.** At ≥100 µs per Python/PyTorch call vs ~5 µs per Rust MCTS iteration, naive Python pUCT is 20× slower before proving any strength gain. Only touch Rust MCTS via either a ~1–2 µs Rust-native tiny leaf evaluator (Tier 4a) or Gumbel small-budget batched search (Tier 4b) *after* passing offline gates.
5. **No Regret Matching / Exp3 rewrite of `poke-engine` DUCT.** Our enemy is a fixed non-adaptive script, not a Nash adversary; DUCT's theoretical non-convergence to NE is not our bottleneck.
6. **No unconstrained argmax over a learned surrogate.** Every surrogate (GBT, Set-Transformer, Pointer Builder) is strictly a **proposal / shortlisting filter**; CRN Sequential Halving over real engine battles is always the judge.
7. **No joint end-to-end co-learning of team + sets + battler.** Violates minimal-code and bilevel stability; alternate with fixed anchors instead.
8. **No doubles, distributed orchestration (Ray/MLflow), or config frameworks.** Keep one file per concern and delete any model/feature group that fails its ablation gate.

---

## Summary Timeline & Compute Budget

| Phase | Core Deliverable | Est. New Code | Effort | Heavy Compute | Stop / Skip Condition |
|---|---|---|---|---|---|
| **0** | Engine fixes, RR-AI stall/mixture fix, set-builder hygiene | ~120 lines (edits) | 1–1.5 d | ~45 min (10 CPU) | Stop if mirrors <75% or ceiling <40% (fix fidelity/rules first) |
| **1** | CRN Sequential Halving chooser + offline archive (91 bosses) | ~150 lines | 1.5–2 d | ~2.5 h (10 CPU) | **Skip 2–3** if SH is fast enough & beats wide-search oracle |
| **2** | Feature probes (LGBM) + Edge-Biased Set-Transformer + LNS | ~280 lines | 2–3 d | ~15 min GPU + 35 min CPU | Keep LGBM & delete net if net < GBT + 0.03 $\rho$ |
| **3** | PLR Expert Iteration + Pointer Builder + Set Refiner | ~220 lines | 2–3 d | ~5 h (10 CPU + GPU) | Drop either sub-component that fails its paired gate |
| **4** | *Conditional:* Rust-native tiny leaf eval (4a) / Gumbel pUCT (4b) | ~210 lines (4a) | 1.5 d (4a) | ~1 h (4a) | **Skip entirely** if mirrors ≥85% & 200k vs 20k < +8 pts |
| **5** | Decision-theoretic encounter routing (`rr_route.py`) | ~100 lines | 1 d | ~30 min (10 CPU) | Arithmetic over best validated scorer |
