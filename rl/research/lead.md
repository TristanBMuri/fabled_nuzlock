# Lead research (general RL, newer literature), 2026-11-24

Done in parallel with Fable / Neon, without reading theirs. "Seen" = abstract / search summary read
today; "recalled" = from memory, not re-checked. Nothing here is verified against full texts.

## Ranked by fit to RR (battler vs a known scripted AI, six-from-box chooser, set builder)

1. **Best response to a known opponent inside MCTS.** BRExIt (Hernandez et al., IJCAI 2023, seen):
   expert iteration with an opponent model biasing the search; notes MCTS Q-values override the model
   with enough compute (drifts to a rational-opponent response). Milec/Kubíček/Lisý CDBR (AAMAS 2024,
   seen): depth-limited best response to a fixed model, exploitable if the model is wrong. GenBR
   (IJCAI 2025, seen). → Our case is the easy end: the RR AI is a deterministic-ish script we have
   in Rust. Expectimax over the script's move distribution is the right search; a 13–33 pt loss
   from using the model is almost surely a bug or a model mismatch. Cheapest high-value test:
   agreement rate of the Rust RR-AI model vs the real (Showdown-side) AI on logged positions.
2. **Drafting = our six-picker.** JueWuDraft (Chen et al. 2020, Honor of Kings, seen): MCTS over picks
   with a supervised win-rate predictor of the final lineup as the leaf reward. "The Art of Drafting"
   (2018) and BPCoach (2023: a random forest was the best win-rate predictor, seen). → A cheap win-rate
   predictor over (our six, their six) features + search over picks is a well-trodden design; a tree
   model may beat a net at our data sizes.
3. **Deck building + battle co-training.** Legends of Code and Magic end-to-end deck + battle
   (ByteDance, arXiv 2303.04096, seen); Q-DeckRec (2018, deck building as learned search, seen);
   Vieira et al. 2020 arena-mode drafting. → Direct precedent for the team ↔ battler chicken-and-egg:
   the usual answer is to train the builder against a *fixed* battler snapshot, then re-snapshot.
4. **Small-data predictors: TabPFN v2 / 2.5** (Hollmann et al., Nature 2025; v2.5 Nov 2025; seen).
   In-context tabular model, strong up to ~10k rows / 500 features with no training. → Our
   (six, boss) → win-rate data will be thousands of rows: TabPFN on hand features is a near-zero-code
   baseline for the team scorer and a test of whether the features carry the signal before any net.
5. **Critic-free group baselines for the builder.** GRPO / RLOO; "Baseline-Free Policy Optimization
   for NCO" (Sepúlveda & Ruz, arXiv 2606.10321, June 2026, seen: GRPO within 2% of POMO on TSP-100,
   plain REINFORCE collapsed once its rollout baseline switched on). POMO (Kwon 2020) and the attention
   model (Kool 2019) recalled. A 2025 study (arXiv 2511.03527, seen) says critics still matter on long
   horizons — fine for us: team picking is a 6-step episode. → Builder = pick-one-mon-at-a-time
   attention policy, K samples per (box, boss), advantage vs the group mean of searched win rates.
6. **Few-simulation policy improvement.** Gumbel AlphaZero/MuZero (Danihelka 2022, recalled; MiniZero
   2023 seen: works at 2 simulations; progressive simulation budget helps). Sequential halving at the
   root. → Use sequential halving now, outside nets: for the chooser's candidate screening (best-arm
   identification, Karnin 2013) and possibly at the battle root.
7. **LLM-evolved heuristics.** FunSearch (2023) → AlphaEvolve (2025), open versions OpenEvolve /
   ShinkaEvolve (seen); "evolving scoring functions inside fixed algorithms" (Ye 2024, Yao 2025).
   → Our leaf evaluation (`evaluate.rs`) and the set builder's rules are exactly "a scoring function
   inside a fixed algorithm" with a cheap programmatic fitness (mirror / boss win rate on paired
   seeds). This is feature engineering by program search, and it keeps the result readable code,
   which suits the minimal-code rule. Risk: overfitting the fitness seeds; needs held-out bosses.
8. **Diversity of teams.** MAP-Elites / QD (MADRID, AAMAS 2024: regret as the fitness; GAME /
   tournament-informed QD, Jan 2026; seen). → An archive of sixes per boss binned by descriptors
   (speed control, weather answer, #megas) is the archetype step in RR_PLAN, done systematically.
   Also a curriculum signal: regret = best known six's win rate minus the builder's.
9. **Simpler value learning if we ever do TD.** PQN (Gallici et al., ICLR 2025, seen): LayerNorm + L2,
   no target net / replay, many parallel envs. SimbaV2 / BRO (2025, seen; continuous control). →
   Only relevant if a learned battle value is trained by TD; we would rather regress on outcomes.
10. **Offline RL with transformers** (Metamon, RLC 2025) and **UED curricula** (PLR 2021, ACCEL 2022,
   recalled): data reuse of every searched battle; train on (box, boss) cases with the largest regret.

## What this changes vs RR_PLAN.md (my position before the merge)

- Before learning anything: test the RR-AI model's fidelity and use it in an expectimax search (1).
- Chooser: sequential halving over candidates (6) and an explicit drafting-style win-rate predictor
  on hand features, first with TabPFN / a GBM (2, 4), before any attention net.
- Learned builder (if gated): autoregressive pick policy + group-relative advantage (5), trained
  against frozen battler snapshots (3), boss/box cases chosen by regret (8, 10).
- Feature engineering track: evolve `evaluate.rs` weights/terms and set-builder rules with an
  LLM-in-the-loop program search on mirror fitness (7), held-out bosses as the check.
