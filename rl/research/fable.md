# Learning in the Radical Red nuzlocke project: research notes and plan (Fable, 2026-11-24)

Independent round. Read-only on the repo; no battles were run. Every number below that is not
tagged as a guess comes from the repo's notebook, logs or code.

## 0. The short version

1. Against a **fixed, known, scripted** enemy the whole problem (six, lead, sets, play) is one
   single-agent stochastic decision problem. Game theory (minimax, Nash, self-play) is a
   robustness fallback here, not the target. The team is the first action; the value of the
   start state under our battler *is* the team's score.
2. The biggest untested lever is **the search's model of the enemy**. Minimax is too pessimistic
   against a script; the AI model lost 13–33 points, but that run had three confounds (below) and
   an exact-model test costs ~15 lines. Do this before any learning.
3. The first learned component should be the **leaf evaluation, fitted by logistic regression on
   battle outcomes** (Texel tuning). The evaluation is already linear-into-a-sigmoid, it is
   material-only (blind to speed order, damage margins, weather), and the search is ~3–4 plies
   deep. This is ~100 lines, no network, no GPU.
4. Team choice needs **wider search with proper noisy-evaluation statistics** (sequential
   halving, paired seeds, a continuous score) before it needs a learned builder: 10 workers play
   ~13k battles an hour, and today's screen uses 96.
5. A learned team scorer comes last, as a **filter distilled from the search**, starting with a
   calibrated matchup cell (12.5k duel rows are already on disk) and a small model on team
   aggregates. Attention / pointer networks only if that plateaus above the noise ceiling.

## 1. What I verified in the code

| Claim in the brief | Checked | Finding |
|---|---|---|
| No learned component in the RR path | `rl/engine_search.py`, `scripts/rr_team_probe.py`, `ai_util/matchup.py` | True. Chooser = `matchup.cover` / `searched_cands` (24 sixes × 4 battles), builder = rules, battler = poke-engine MCTS. |
| MCTS is decoupled UCB1, hand leaf | `poke-engine/src/mcts.rs:74`, `:247`, `:226` | True. Constant c = √2, leaf = `sigmoid(0.0125 · (eval − root_eval))`, damage rolls branch only at the root and depth 1 (`:176`). |
| Leaf evaluation | `poke-engine/src/genx/evaluate.rs` | Material only: HP fraction, alive, status, the *active* mon's boosts, hazards, screens, Tailwind, item present, minus `NUZLOCKE_DEATH` per fainted mon of ours. Nothing on speed order, damage either way, weather, bench matchups, enemy boosts as a sweep threat. |
| Death penalty | `engine_search.py:37`, `evaluate.rs:223` | 60 points per death = a 0.18 shift at the sigmoid's centre. A won terminal goes through the same sigmoid (`mcts.rs:226`), so a win with 3 deaths can score below an even open position. |
| Search vs the AI model | `mcts.rs:95–125`, `:411` | In `vs_ai` the enemy's options at every node are the Rust AI's tied top choices, sampled uniformly. `use_last_used_move` is switched on only in this mode. |
| The AI model is deterministic, the real AI is not | `ai_util/rr_ai.py:117`, `:210`, `:277`, `:542`, `:614`; `rr_ai.rs:1832` | The Python AI has 50% branches (recover, repeat Protect), a 2-bit random in "save the sweeper", random tie-breaks. The Rust port returns ties and **omits the random branches** ("CFRU's random 25/50% branch omitted"). |
| 90% agreement | `results/rr_ai_diff.log` | 542 / 600, measured on positions from battles the *minimax* searcher did not steer. Voluntary switches agree 78%, forced switches 90%. |
| Per-state data exists | `results/*.jsonl` | **No.** Logs hold per-team aggregates only: 786 (six, boss, win, deaths) rows over 62 RR files, and 12,564 duel cells (`ko_them, ko_us, faster, heals, hit, take, wins, hp`). No battle states are logged anywhere. |
| A binding for the Rust AI as an actor | `poke-engine-py/src/lib.rs:1143` | `rr_ai_choice(state, side)` exists. |

Two things I noticed that the brief does not list:

- At Blaine the "best" six is Cherubi, Deerling, Deerling-Winter, Seedot, Kangaskhan, Vikavolt into
  Desolate Land (`results/rr_ceiling_mega.log`). That is the matchup matrix proposing Grass types
  into a Fire gym. Whatever the battler does, the candidate generator is the first suspect at
  Blaine.
- At Koga the best six dies almost completely every battle (0.97 deaths / mon). With wins at
  0–8%, win rate carries no ranking signal between candidates; `progress` (already computed in
  `rr_team_probe.progress`) does.

## 2. Research notes

Honesty key: **[S]** = I saw a search-result summary of it in this session (not the full text);
**[R]** = recalled from training, not re-read; treat details as approximate. I read no paper in
full during this round. No numbers are quoted from memory unless marked.

### 2.1 Planning against a known, fixed opponent policy

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Wang, Gleave et al., "Adversarial Policies Beat Superhuman Go AIs" (ICML 2023) **[S]** | The attacker's tree search queries the *victim's own policy* at the opponent's nodes (A-MCTS). Two variants, a fast approximate one and a slow accurate one. | The template for our case: put the real script in the tree. Their adversary wins by exploiting specific blunders, not by playing better Go, which is exactly how human nuzlockers beat RR bosses with 0 EVs (baiting, free switches, AI move prediction). | The victim model must be *exact*; theirs was (the same network). Ours is a 90% port. |
| Johanson, Zinkevich, Bowling, "Restricted Nash Response" (NeurIPS 2007); Johanson & Bowling, "Data-Biased Response" (AISTATS 2009) **[S]** | A best response to a model is brittle when the model is wrong. RNR: the opponent plays the model with probability p, otherwise freely. DBR applies it per decision point and is more robust to inaccurate models. | The principled form of the plan's "opponent mixture": at each enemy node, follow the AI model with probability p, else UCB. p is a dial between exploit and safety. Per-node mixing is the DBR flavour. | ~15 lines in `mcts.rs` selection. |
| Milec, Kubíček, Lisý, "Continual depth-limited responses" (AAMAS 2024) **[S]** | RNR adapted to depth-limited search; reports it beats mixing a best response with a Nash strategy linearly. | Supports mixing *inside* the tree rather than averaging two searches' outputs. | — |
| Opponent-model search (Carmel & Markovitch 1996; Donkers et al., probabilistic opponent-model search, ~2001) **[R]** | Classic result: OM search gains with a correct model and can lose badly with a slightly wrong one, because the searcher steers into the states where the model errs. | Explains the −13 to −33 without a bug: 90% agreement measured off-policy says little about agreement on the lines the search *selects*. Gives the diagnostic: measure agreement on-policy. | — |
| Timbers et al., "Approximate exploitability: learning a best response" (IJCAI 2022) **[R]** | A learned best response against a fixed agent via search + RL. | Confirms "fixed opponent ⇒ single-agent problem" as the standard framing. Nothing to code. | — |

### 2.2 Simultaneous-move search

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Lisý, Kovařík, Lanctot, Bošanský, "Convergence of MCTS in simultaneous move games" (NeurIPS 2013); Kovařík & Lisý (MLJ 2019) **[S]** | UCB-style selection does not converge to a Nash equilibrium even in a one-shot matrix game; Exp3 / regret matching do. | We should *not* spend effort here. Nash play is the wrong target against a script. Decoupled UCB as a pessimistic model is fine as the safety half of a mixture. | 0. |
| Lanctot et al., Tron / Goofspiel studies (2013–14) **[S]** | Decoupled UCT often wins in practice despite the theory. | Same conclusion: leave the selection rule alone. | 0. |

### 2.3 Search plus a learned value at small budgets

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Texel's tuning method (Österlund 2014; Buro 1995 for Othello), chessprogramming.org **[S]** | Fit the weights of a linear evaluation by logistic regression of `sigmoid(K · eval)` on game results over many positions. Hundreds of parameters tune in reasonable time. Österlund's caveat: unclear how much of his gain came from the method vs a weak starting evaluation. | **The best fit in this whole survey.** Our leaf is already `sigmoid(0.0125 · eval)` with ~25 hand constants taken from random battles, plus a hand death weight that was set by three probes (0 / 60 / 200). Fitting it to RR boss outcomes is supervised, convex, runs in seconds, and stays readable (a table of weights). | ~40 lines Rust (features out, weights in), ~60 Python. No GPU. |
| NNUE (Nasu 2018; Stockfish 2020) **[R]** | A small network evaluation designed for CPU search: cheap, incrementally updated features. | The upgrade path if the linear fit plateaus: a one-hidden-layer net over the same features, in Rust, is still a few µs. Also the idea of precomputing expensive features once per search (our 6×6 damage table). | 1–2 days. Only on a failed gate. |
| Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022) **[S]** | Sequential halving at the root instead of UCB; guarantees policy improvement and helps most at very few simulations. | Relevant only if we cut to ≤ 1k iterations for cheap screening battles. At 20k iterations and ≤ 9 root actions it buys nothing. | ~40 lines Rust; parked. |
| Anthony et al., Expert Iteration (2017); Silver et al., AlphaZero **[R]** | Distil search into a prior / value, search with them, repeat. | The repo's own lesson: one round of ExIt on the *policy prior* gained nothing, and pUCT in poke-engine is 3–5 days. A prior narrows a search; our search is not width-bound, it is evaluation-bound. So distil into the *value*, not the policy. | — |
| Baxter et al., TDLeaf (1998); Veness et al., TreeStrap (2009) **[R]** | Train an evaluation toward the deeper search's value rather than the final result. | A second, low-noise target: the 200k-iteration root value at a logged state. Outcome labels are unbiased but Bernoulli; deep-search values are biased but nearly noise-free. Fit on a mix; check each against the other. | A few lines once states are logged. |

### 2.4 Pokémon-specific (2024–2026)

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Karten et al., PokéChamp (ICML 2025) **[S]** | Minimax search with an LLM for move proposal, opponent modelling and leaf value. | Evidence that depth-limited search with a damage-calc tool is strong; nothing to import (LLM in the loop violates minimal code and our speed). | — |
| Grigsby et al., Metamon, "Human-level competitive Pokémon via scalable offline RL with transformers" (RLC 2025) **[S]** | Offline RL on millions of human replays. Its own framing: the best prior bots used heuristic search in custom simulators. | Not for us: we have no human RR data, a perfect simulator and full information. Confirms search-in-simulator as the strong baseline class. | — |
| Angliss et al., VGC-Bench (2025; AAMAS 2026) **[S]** | Benchmark for generalising across teams. Single-team agents get strong; multi-team generalisation degrades sharply. **Team building is explicitly left open**; they suggest using a strong battler to score candidate teams. | Two things: (i) do not expect a learned *battler* to generalise across boxes cheaply, which supports search for play; (ii) the proposed recipe for team building is the one in this plan (battler as the reward for a search over teams). | — |
| J. Wang, "Winning at Pokémon Random Battles Using RL" (MIT M.Eng. thesis, 2024) **[S: record only; R: content]** | From memory: PPO self-play plus MCTS at test time for gen4 random battles. I could not confirm the method from the search result. | Weak evidence only; not relied on. | — |
| pmariglia, poke-engine / Foul Play **[R + repo]** | The engine in this repo. Its evaluation constants were tuned for competitive random battles. | The constants are a prior, not a fit to RR bosses: another argument for §2.3 row 1. | — |
| Smogon forum post, teambuilding as optimisation with an interaction (synergy / counter) model **[S]** | Not peer reviewed. Pairwise-interaction model of team value beat a no-interaction baseline. | Supports pairwise matchup aggregates as the first team feature set. | — |

### 2.5 Combinatorial selection: a six from a box, given their six

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Chen et al., JueWuDraft, "Which heroes to pick?" (2020); Ye et al., "Towards playing full MOBA games" (NeurIPS 2020) **[S]** | Drafting = tree search over picks with a learned win-rate predictor as the reward; the predictor is a 3-layer MLP. Trained on ~30M matches. | The architecture we want, at 1/10,000 of the data: a small win predictor as the *reward for a search over sixes*. Their predictor is species one-hots, which we cannot use (generality rule, and we have hundreds of rows per boss, not millions). | — |
| "The Art of Drafting" (2018); BPCoach (2023) **[S]** | Same idea; reported predictor accuracies near chance (~0.51–0.53). | A warning: team-composition predictors are weak when the outcome is dominated by play. Ours should be far better (fixed enemy, deterministic script, matchup features computed by a damage calculator), but it must be gated against a noise ceiling, and it must stay a filter. | — |
| Zaheer et al., Deep Sets (2017); Lee et al., Set Transformer (2019); Kool et al., "Attention, learn to solve routing problems" (2019) **[R]** | Permutation-invariant set encoders; pointer-style constructive decoders trained with REINFORCE. | The eventual form of the plan's 3b. With < 10k labelled sixes, sum-pooled per-mon tokens (Deep Sets) is the most we can fit; a pointer decoder is not needed while a search over the scorer is cheap (C(30,6) = 594k sixes can be *enumerated* through a millisecond scorer). | 150 lines; parked behind a gate. |
| Fontaine et al., MAP-Elites for Hearthstone decks (2019); García-Sánchez et al., evolutionary deck building (2016) **[R]** | Evolutionary / quality-diversity search over decks, scored by simulated games. | Mutation operators (swap a mon, the lead, a move, an item) with noisy simulated fitness is the standard no-learning baseline. Keeping the best six *per archetype* (QD) guards against the candidate generator collapsing to one idea (Grass into Blaine). | ~60 lines on top of `searched_cands`. |

### 2.6 Choosing among candidates under noisy win rates

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Karnin, Koren, Somekh, Sequential Halving (ICML 2013); Audibert & Bubeck, Successive Rejects (2010) **[S]** | Fixed-budget best-arm identification: play all, drop the worse half, repeat. | Replaces "24 sixes × 4 battles, keep 3". Same budget, more candidates, more battles where it matters. | ~15 lines. |
| Russo, Top-Two Thompson Sampling (2016) **[S]** | Bayesian best-arm identification; most theory is for fixed confidence. | Not needed; halving is simpler and matches the pool's batch shape. | — |
| Li et al., Hyperband (2018) **[R]** | Successive halving over a *fidelity* axis. | Cheap battles (MCTS 2k) for early rounds, 20k for the final. Only valid if rankings hold across fidelities (the plan's separability test). | A flag. |
| Smith & Winkler, "The optimizer's curse" (2006) **[R]** | The max of noisy estimates is biased up; more so with more candidates. | Already known here (replay on fresh seeds). With 10× more candidates it gets worse: the reported number must always be the fresh-seed replay at n ≥ 48. | 0. |
| Fishtest mathematics (generalised SPRT, pentanomial pairs) **[S]** | Sequential test for "is the patch better", run on paired games; pairing saves a lot. Resources scale inversely with the square of sensitivity. | A stopping rule for every accept / reject gate in this plan instead of fixed-n tables: paired by seed, outcomes in {−1, 0, +1} per pair, stop when the log-likelihood ratio crosses a bound. | ~25 lines. |
| Burch et al., AIVAT (AAAI 2018) **[S]** | Control variates from a value function and known chance / known-strategy probabilities; 44× fewer games in their poker match. | The idea, not the estimator: use a continuous, lower-variance outcome. Ours is already there: `progress` and deaths per mon. Full AIVAT needs per-node chance probabilities from the simulator; too much code. | 0 for the simple form. |

### 2.7 Curricula, coupled components, offline data

| Source | What it is | What it gives us | Cost |
|---|---|---|---|
| Jiang et al., Prioritized Level Replay (2021); Parker-Holder et al., ACCEL (2022) **[S]** | Sample training levels by estimated regret (value-loss), plus staleness; ACCEL mutates high-regret levels. | For the value fit: sample (box, boss) cases where outcome is uncertain (p(1−p) large) or where the evaluation's prediction error is large. All-loss cases (Koga today) and all-win cases teach almost nothing. ACCEL's edit idea = mutate boxes near the frontier. | ~10 lines of sampling weights. |
| Policy iteration / alternating optimisation **[R]** | Evaluate under the current policy, improve, repeat. | The chicken-and-egg is policy iteration on one decision problem. Rule: a team's score is always tagged with the battler version that produced it; when the battler changes, re-screen only the top-K. | 0. |
| Offline RL (CQL, IQL, Decision Transformer) **[R]** | Learning from fixed logs without a simulator. | **Not applicable**: our simulator costs 2.7 worker-seconds a battle. Old logs are for pre-fitting and for held-out checks; fresh on-policy data is cheap. | — |
| Schrittwieser et al., MuZero; Ye et al., EfficientZero (2021) / V2 (2024) **[R]** | Learned models for planning. | Not applicable: the model is known and exact to 97%. | — |

## 3. The two questions the brief asks to be addressed

### 3.1 Why did searching against the AI model lose 13–33 points?

With an exact model, a best response cannot be worse than a minimax response *given the same
evaluation and depth*: the tree is narrower (9 × 1 instead of 9 × 9 joint actions), so it is also
deeper. A loss of this size therefore needs a cause. Candidates, most likely first:

- **H1. The model is wrong where the search goes.** 90% was measured off-policy. Voluntary
  switches agree only 78%, and "the AI stays in and takes chip damage" vs "the AI switches" is
  exactly the error that produces long passive battles (70–90 turns, as observed). A best
  response steers into model errors (opponent-model search, §2.1).
- **H2. The stall bias had no adversary.** The run predates the fixes. "A costly win scores below
  an open fight" plus a −60 death term makes not-finishing attractive. Under minimax the modelled
  enemy punishes waiting; under a passive AI model nothing does. Symptom match: long passive
  battles, no turn-limit losses.
- **H3. Omitted randomness.** The port drops the AI's 50% branches (recover, repeated Protect, the
  sweeper save). The search then plans exact lines through coin flips.
- **H4. Mean damage below depth 1.** Damage rolls branch only at the root and depth 1. With a
  deterministic enemy, deep lines that rely on average rolls and exact KO thresholds look certain.
- **H5. `use_last_used_move` is on only in this mode**, so our own option sets and the state
  semantics differ between the two modes (Choice lock, Encore, first-turn moves); and the
  converted root state's `last_used_move` decides the AI's "first turn" logic (`is_first_turn`).

Tests, cheapest first (all ≤ 15 min of machine time each, Misty 10 boxes × 6, paired seeds):

1. **Exact-model oracle.** Let the *Rust* AI act as the enemy in the real battle (a 15-line brain
   that converts the dump and calls `rr_ai_choice`, picking uniformly among ties). Now the model
   in the tree is exact up to tie-breaks. Compare `--engine-opp ai` with `minimax`, both against
   this enemy.
   - `ai` wins clearly → the method is right; the port gap (H1 / H3) is the whole problem.
   - `ai` still loses → the cause is in the search (H2, H4, H5), independent of the port.
2. **On-policy agreement.** In `--engine-opp ai` battles against the Python AI, log at each turn
   whether the Python AI's actual choice is in the Rust model's predicted set. Compare with 90%.
   Split by move / voluntary switch / forced switch.
3. **Re-run after the three evaluation fixes**, with the won-terminal floor
   (`0.55 + 0.45 · alive / 6`). Isolates H2.
4. **Mixture dial** p ∈ {0.5, 0.8, 0.95}: at each enemy node follow the model with probability p,
   else UCB over all options (§2.1). If the curve over p is monotone, the model is good enough to
   trust; a peak inside means "exploit, with insurance".

What follows from the outcomes:

| Oracle (1) | On-policy (2) | Action |
|---|---|---|
| ai > minimax by ≥ 10 pts | well below 90% | Fix the port: the random branches as weighted enemy options (the tree already samples enemy choices; give them weights), then the switch logic, to ≥ 97% on-policy. Use the mixture meanwhile. |
| ai > minimax | ≈ 90% | H3 / tie handling; same fix, smaller. |
| ai ≤ minimax | any | Search bug (H2 / H4 / H5). Fix before anything else; every later number depends on it. |

I expect this to matter more than any learned component: a script that can be read is the reason
0-EV hardcore runs are possible for humans at all. This expectation is mine and untested.

### 3.2 The chicken-and-egg loop

It is one decision problem, so it has an order of attack rather than a circle:

- **Score teams with the battler that will play them**, always. A team only this battler can
  pilot is fine (the existing plan says this too).
- **Anchor the battler without the builder**: mirrors against the AI-vs-AI baseline, and
  reference human teams. Both already planned; keep them as the first gate after every battler
  change.
- **Blame each wall once, with a ladder of battlers**: the same six under MCTS 2k / 20k / 200k and
  under the exact-model search. Win rate rising along the ladder → battler-bound; flat → team- or
  fidelity-bound. This also yields the separability check (rank correlation of sixes across
  rungs), which decides whether cheap battles may be used for screening.
- **Version the scores.** Each stored (six, boss, outcome) row carries the battler's settings.
  When the battler changes, re-screen the top 10–20 sixes per boss, not everything. Learned team
  models train on the newest version, pre-fit on older ones.
- **Alternate in a fixed rhythm**: battler change → anchors → re-screen → (if gates pass) team
  search → new data → evaluation refit. One heavy job at a time fits this naturally.

## 4. Feature engineering

### 4.1 What already exists and must be reused

| Existing piece | Where | Reuse as |
|---|---|---|
| `dmg_frac(att, dfn, move, …)` with field, items, abilities, accuracy; memoised | `ai_util/calc_bot.py:154` | Every damage feature in Python (team / set features). |
| `stat(mon, key, own, battle)` incl. field and boosts | `calc_bot.py:46` | Speed order, stat-margin features. |
| `threat(opp, mon)` | `calc_bot.py:227` | Worst incoming hit. |
| `pair()` → `ko_them, ko_us, faster, heals, hit, take` | `ai_util/matchup.py:33` | The matchup cell: the per-(ours, theirs) feature vector. |
| `safe / race / edge` tiers | `matchup.py:76–104` | Hand labels to be *replaced by a calibrated probability* (§4.3), kept as features. |
| `duel()` win / HP per cell | `rr_team_probe.py:377` | The label for the cell calibrator; 12,564 rows on disk. |
| `progress(st)` | `rr_team_probe.py:92` | Continuous outcome for screening at walls. |
| `attribution()` deaths by enemy mon | `rr_team_probe.one_battle` | Per-mon death labels. |
| Engine `evaluate()` terms | `evaluate.rs` | The first Texel feature vector, unchanged. |
| Engine damage calc | `genx/damage_calc.rs`, binding `calculate_damage` | Matchup features inside the leaf (precomputed table). |
| Lesson 2's observation groups, `--obs` and column blanking | `rl/lesson2_ppo.py:159`, `:277` | The ablation mechanism (group on / off; blank at test time), not the Gen 8 features. |

### 4.2 Battle-state features (leaf value; also any later opponent model)

All from side one's (our) view, computed in Rust from `State`, so the fit and the search see the
same numbers. Linear first. "Table" = a 6 × 6 × 2 array of best-move damage fractions at full HP,
no boosts, current field, **computed once per search at the root** (≈ 300 damage calcs, < 1 ms)
and indexed at leaves; this keeps the leaf at a few µs.

| # | Feature (ours and theirs unless noted) | Source | Stage |
|---|---|---|---|
| 1 | Sum of HP fractions of living mons | exists | V0 |
| 2 | Living mons | exists | V0 |
| 3 | Our fainted count (the nuzlocke term) | exists | V0 |
| 4 | Status counts: burn (weighted by physical moves, as now), sleep, paralysis, toxic, poison, freeze | exists | V0 |
| 5 | Active's boosts through the existing multiplier table, per stat | exists | V0 |
| 6 | Hazards on each side × living grounded / non-Boots mons | exists | V0 |
| 7 | Screens, Aurora Veil, Safeguard, Tailwind (flat when permanent), Substitute, Leech Seed, Confusion, item present | exists | V0 |
| 8 | Active vs active: our best damage ÷ their remaining HP, capped at 1; and theirs on ours | table × boost multipliers × current HP | V1 |
| 9 | Active vs active: we move first (speed with boosts, paralysis, Tailwind, weather ability, Trick Room; priority if the best move has it) | state | V1 |
| 10 | "We KO first" indicator: hits we need ≤ hits they need, tie broken by #9 | from 8, 9 | V1 |
| 11 | Answers left: number of their living mons for which some living mon of ours wins the race (table, current HP) | table | V1 |
| 12 | Unanswered threats: number of their living mons that win the race against *every* living mon of ours | table | V1 |
| 13 | Sweep exposure: number of our living mons their active outspeeds and KOs in one hit at its current boosts (boost-aware threat, NOTEBOOK D2) | table × boosts | V1 |
| 14 | Safest switch-in: min over our bench of their active's best damage ÷ that mon's HP | table | V1 |
| 15 | Weather / terrain / Trick Room turns left (0 for permanent arena effects; their effect is already inside the table) | state | V1 |
| 16 | Their active has a recovery move and is not KO'd in ≤ 2 of our hits (`matchup.ours`' healer rule) | table + moves | V2 |
| 17 | Sleep / freeze turns, Toxic counter, Protect counter of each active | state | V2 |

Targets: **U = win × (1 − λ · our deaths / 6)** with λ = 0.5 as the single utility the search
maximises (it replaces both the death constant and the won-terminal hack; terminals return U
exactly). Fit a second logistic head, P(win), on the same features for reporting. λ is a stated
preference, not a fitted number; sweep 0.3 / 0.5 / 0.8 once on deaths per mon vs wins.

For a later opponent model (only if the port cannot reach 97%): the AI's own inputs, which are
features 8–10, 13, 14 from *its* side plus HP buckets, `first turn` and `last_used_move`. I do not
plan to build one; the script is known, so fix the port.

### 4.3 Matchup cell, per-mon, per-set and team features (builder, set builder)

No species identity anywhere (generality rule). Stats are at the mon's level with its nature /
IVs / item; types as 18 multi-hot bits only inside per-mon tokens (stage T3).

**Cell (ours i vs theirs j)**: one row per pair, the input to the calibrator
`c_ij = P(i beats j one-on-one)` and to `h_ij = expected HP fraction left`.

| Feature | Source |
|---|---|
| `hit`, `take` (best damage fractions each way), `ko_them`, `ko_us` (capped at 6) | `matchup.pair` |
| `faster`; speed ratio; we have a priority attack; they have one | `pair`, `calc_bot.stat`, move data |
| `heals` (theirs); ours has recovery | `pair`, `HEALS` |
| First-hit modifiers already handled by `hits_to_ko` (balloon, resist berry) as a "first hit differs" flag | `matchup.py:47` |
| Their best move is a status / setup opener (the AI's free first turn, B3 in the notebook) | AI model on the opening state, or the duel |
| Our best move's accuracy; recoil / self-KO flag | move data |
| Under the arena: same features recomputed (Tailwind flips `faster`, weather changes `hit`) | `battle` argument |
| Hand tiers `safe`, `race`, `edge` | `matchup.py` |

**Per mon (token for T3; aggregates for T2)**

| Feature | Source |
|---|---|
| Row summaries over the boss six: Σ c_ij, max c_ij, count c_ij > 0.9 (safe answers), count c_ij < 0.2 (liabilities) | cells |
| Worst incoming hit over the boss six (max `take`); count of boss mons that outspeed and OHKO it | cells |
| Stat margin: its attack-stat ÷ mean boss defence in its better category; boss mean attack ÷ its defences; its level − boss mean level | `calc_bot.stat` |
| Level-normalised stats (6), types (T3 only), is mega, item class (choice / life orb / sash / berry / none), ability class flags (weather setter, weather negation, immunity granted, Intimidate) | set data |
| Pivot value: number of boss mons whose *predicted AI move* it is immune to or resists 4× (free switch-ins) | AI model + type chart |

**Per set (for choosing moves / item / ability, boss-conditional)**: score a candidate change by
what it does to the mon's row.

| Feature | Source |
|---|---|
| Δ row when the move is added to the set: Δ Σ c_ij, Δ count of safe answers, Δ max `hit` per boss mon | recompute cells (`dmg_frac` is memoised) |
| Coverage: number of boss mons this move is the best move against | cells |
| Accuracy; priority; recoil / self-faint (Explosion is a death in a nuzlocke); charge turn; lock-in (Thrash / Outrage) | move data |
| Item compatibility: Choice item with a status move in the set; Rest without Sleep Talk; item type-boost with no move of that type | rules, as penalties |
| Δ row from the item (cells recomputed with it): damage items change `ko_them`, Sash / Vest / berries change `ko_us` | `dmg_frac`, `hits_to_ko` |
| Δ row from the ability slot (existing `choose_ability` logic) | `rr_probe` |
| One-per-team constraints: one mega stone, item uniqueness | builder |

**Team vs team (order-invariant aggregates; the T2 model's whole input, ~30 numbers)**

| Feature | Why |
|---|---|
| For each boss mon: best c_ij, **second-best** c_ij | A nuzlocke needs a backup answer; cover only counts one. |
| min over boss mons of best c_ij; count of boss mons with best c_ij < 0.5 | The weakest link decides a no-death run. |
| Σ of best c_ij; Σ of all c_ij | Bulk quality. |
| Lead cell: c and h of our lead vs their lead; our lead's worst incoming hit | The opening is deterministic against a script. |
| For each boss mon: min over our six of `take` (best pivot); mean over boss mons | Safe switching. |
| Count of ours faster than ≥ 4 of theirs under the arena; count with priority | Speed control (Koga's Tailwind). |
| Count of ours that the boss's best sweeper outspeeds and OHKOs | Sweep exposure. |
| Mean stat margin, attack and defence sides | The 85-EV result says margins decide Koga / Blaine. |
| Arena fit: count of ours whose row improves / worsens under the arena vs without it | Weather abuse vs victims (Grass into Desolate Land). |
| Has weather negation / own weather setter; has mega; count of status spreaders; count of recovery users | Archetype flags. |
| Level cap − boss top level | Progression. |

### 4.4 Is the feature set adequate? (The lesson: "the obs is missing X" was undertraining every time.)

A feature may be added only when all four of these hold, in this order:

1. **Noise ceiling known.** Take 500 logged states (or 200 sixes); replay each 8 times with fresh
   seeds. The spread gives the irreducible Brier / log-loss of an outcome label. A model at the
   ceiling is missing nothing, whatever intuition says.
2. **Converged.** Held-out loss vs data size (¼, ½, 1×) and vs training time is flat: doubling
   the data improves held-out log-loss by < 1%. For the linear fit this is automatic (convex);
   for any network it is a plotted curve in the cockpit before anyone says "missing input".
3. **Capacity ruled out.** A gradient-boosted tree model on the *same* features does not beat the
   linear / small model by more than 2% log-loss on held-out bosses. If it does, the features
   are fine and the model class is the limit (add interactions or the small net, not inputs).
4. **A residual names the feature.** Bucket held-out residuals by the candidate (speed order,
   weather on, enemy boosted, …). Require a calibration gap ≥ 5 points in a bucket with n ≥ 500.
   Then add it, and it must (a) improve held-out loss and (b) pass the play test (paired SPRT).

Supporting probes:

- **Deep-search residual probe.** At logged states, regress (MCTS-200k root value − static
  evaluation) on candidate features. This is the evaluation's error with almost no label noise,
  so 5k states suffice where outcomes would need 50k+.
- **Group ablation**, two ways (lesson 2 already has both mechanisms): refit without a group;
  blank a group at test time. A group whose removal changes nothing gets deleted (minimal code).
- **Sign and scale sanity.** In the linear fit every weight has an obvious expected sign (more
  HP good, our deaths bad). A wrong sign means collinearity or a data bug, not a discovery.
- **Splits.** By battle (states of one battle never straddle the split), by box, and every 4th
  boss fully held out. Always report train vs held-out bosses.
- **The play test outranks the loss.** Lower log-loss with no gain in paired battles means the
  evaluation was not the bottleneck; stop and say so.

## 5. The plan

Order: cheap diagnostics, then the search's opponent model, then team search at scale (it also
produces the data), then the learned evaluation, then a learned team filter. Each gate is a
paired comparison on seeds, read with a sequential test where it says SPRT; every headline
number is a fresh-seed replay at n ≥ 48.

Machine-time arithmetic used below: 2.7 worker-seconds a battle, 10 workers → ~3.7 battles/s,
~13,000 battles an hour.

### Phase A. Instruments (½ day; ~70 lines; < 30 min machine)

- **Goal:** every later phase can be measured and leaves reusable data.
- **Method / code:**
  - Per-turn logging in `EngineSearcher.__call__` behind a flag (~25 lines,
    `rl/engine_search.py`): engine state string, root visits and means for both sides, chosen
    move, the Rust AI's predicted set, and (next turn) the enemy's actual move. `one_battle`
    appends outcome, deaths per mon, turns, seeds, battler settings to a `<out>.states.jsonl`.
  - `scripts/sprt.py` (~25 lines): paired generalised SPRT over per-seed outcome differences.
  - Sequential halving in the screen (~15 lines in `run_pool`), scoring by
    `progress − death_weight · deaths/mon` rather than wins while a boss is below ~20%.
- **Data:** none needed.
- **Gate:** logging on vs off gives identical paired results on Misty 5 boxes; the SPRT helper
  reproduces a known verdict (the engine-vs-Showdown tie on `rr_engine_boxes30`).
- **If it fails:** logging perturbs RNG streams; move it after the decision, fix, repeat.

### Phase B. The opponent in the tree (1 day; ~15 lines Python, ~15–40 Rust; ~1 h machine)

After the three known bug fixes land (not part of this plan).

- **Goal:** decide how the search models the enemy; explain the −13 to −33.
- **Method:** the four tests of §3.1 in order (oracle, on-policy agreement, re-run after fixes,
  mixture dial). Rust: per-node mixture in `Node::selection` (~15 lines); if the oracle says the
  method works, weighted random branches in `rr_ai.rs` (~25 lines) and the switch-logic bug list.
- **Data:** Misty and Erika, 10 boxes × 6, paired; mirrors at n = 48.
- **Gate (adopt a mode):** paired SPRT vs minimax accepts +8 points or better on Misty + Erika
  boxes, deaths per mon not worse, and the Brock regression within ±3.
- **If it fails:** minimax stays. Record on-policy agreement as the open port-quality number.
  Do not build a learned AI model.
- **Why here:** it changes the battler, so it must precede team search and any value labels.

### Phase C. Blame the walls (½ day; ~10 lines; ~1 h machine)

- **Goal:** for Koga and Blaine, know whether the battler, the candidate sixes or fidelity is
  the limit, before spending on either side.
- **Method:**
  - Battler ladder on the current best sixes and the EV-85 sixes: MCTS 2k / 20k / 200k, plus the
    Phase B winner. n = 24 each, paired.
  - Reference human teams for Koga and Blaine (needs the user: two documented hardcore teams
    each), played by our battler at n = 48.
  - Mirrors against the AI-vs-AI baseline (already planned).
- **Gate / reading:**
  - Reference team ≥ 40% → the battler can win there; the builder / candidate generator is the
    wall → Phase D is the priority.
  - Reference ≤ 10% and the ladder is flat → fidelity or rules (permanent Tailwind on the ROM,
    item availability, sets); stop and check `data/rr/FIDELITY.md` rows before more compute.
  - Ladder rises steeply (≥ 15 points from 20k to 200k) → evaluation / depth-bound → Phase E
    moves ahead of Phase D.
- **Also yields:** Spearman of six rankings across rungs. ≥ 0.8 → screen early halving rounds at
  2k iterations (≈ 5–10× cheaper, my guess; measure).

### Phase D. Team and set search at scale, no learning (1–2 days; ~90 lines; ~45 min a boss)

- **Goal:** find out what search alone can reach at Koga / Blaine, and produce thousands of
  labelled (six, sets, boss) rows and per-turn states at bosses that are not 0% or 100%.
- **Method:** local search over (six, lead, moves, items) in `rr_team_probe.py`
  (`--chooser evolve`, ~60 lines):
  - Seeds: the cover six, `searched_cands`, the best archived six per archetype flag (weather
    negation, speed control, bulky pivot core; a tiny quality-diversity archive so one idea
    cannot fill the population).
  - Mutations: swap one mon for a box mon drawn in proportion to its row quality vs the boss
    mons the team answers worst; change the lead; swap one move by the largest row gain (§4.3
    per-set features); swap an item by row gain; respect the legality rules already in
    `rr_probe.hardcore`.
  - Evaluation: sequential halving per generation on paired seeds, continuous score;
    4 → 8 → 16 → 32 battles for survivors; ~10k battles a boss.
  - Set builder fix (~30 lines, `gauntlet` / `rr_probe`): pick moves and items by marginal row
    gain against *this* boss, with the compatibility penalties. This is a rule from features,
    not a model.
- **Data out:** every evaluated candidate with its feature vector and outcomes; per-turn states.
- **Gate:** fresh-seed n = 48: Koga and Blaine ≥ 40% legal; no regression at Misty–Erika. The
  30-box table improves by ≥ 10 points at Erika (22% today).
- **If it fails:** Phase C's reading decides. Battler-bound → Phase E. Team-bound with a winning
  reference team → inspect what the reference has that no candidate had (a missing mutation, a
  missing item, a missing move source) and add that, not a model. Neither → fidelity.
- **Winner's curse:** with ~500 candidates the selected six's screen score is inflated by
  construction. Only the fresh-seed number is reported or compared.

### Phase E. Learned leaf evaluation (2 days; ~40 lines Rust + ~80 Python `rl/fit_eval.py`; fits in seconds, play tests ~1 h)

- **Goal:** a leaf value fitted to RR boss fights that knows speed order and damage margins, and
  a nuzlocke objective without hand constants.
- **Method (Texel):**
  - Rust: `evaluate_features(state) -> Vec<f32>` exposed to Python; `evaluate()` becomes a dot
    product with weights loaded once from a file named in an env var (the pattern
    `POKE_ENGINE_DEATH` already uses). Terminals return U directly.
  - V0: refit the existing terms only (features 1–7) on U. Zero new features; it tests the method.
  - V1: add the table features 8–15 (~80 more Rust lines: the per-search damage table).
  - Fit: logistic regression with L2, by battle-level splits; second target = deep-search root
    value on a 5k-state subsample; report both.
- **Data:** per-turn states from Phases B–D (a 10k-battle Phase D run ≈ 300k states), sampled
  with weight p(1−p) of the (six, boss) pair so decided fights do not dominate; all singles
  bosses available, every 4th boss held out. Top up with ~1 h of battles on boxes at Brock–Erika
  where win rates are 20–70%.
- **Features:** §4.2.
- **Gates:**
  - Fit: held-out-boss log-loss at least 5% below the hand evaluation's (with its scale K
    refit, so the comparison is fair), and within 10% of the noise ceiling's distance for V1.
  - Play (the one that counts): paired SPRT, fitted vs hand evaluation, same iterations:
    +5 points or −0.05 deaths per mon on Brock / Misty / Erika boxes, mirrors not worse, and a
    gain on the sensitive bed (EV-85 Koga / Blaine sixes, or Phase D's best).
- **If the fit gate passes and the play gate fails:** the evaluation was not the bottleneck.
  Keep the terminal utility if it alone helps; drop the rest. Do not escalate to a network.
- **If the fit gate fails:** run §4.4 in order (ceiling, convergence, GBDT, residuals). Only a
  GBDT that beats the linear model by > 2% justifies the small net (one hidden layer of 32 over
  the same features, ~60 Rust lines), and only a named residual justifies a new feature.
- **Second round:** after adopting, replay Phase D's top 20 per boss (versioned scores), refit
  once on the new data. Stop after two rounds unless the second still gains ≥ half the first.

### Phase F. Learned team filter (2–3 days; `rl/team_score.py` ~150 lines; training minutes, CPU)

Only if Phase D passes and is compute-bound for what comes next (30 boxes × ~75 bosses, or
routing, where thousands of hypothetical boxes must be scored).

- **Goal:** rank sixes for a (box, boss) well enough that search verifies 24 instead of 500.
- **Method, in three rungs; stop at the first that passes:**
  - **T1. Cell calibrator.** Logistic regression (then GBDT if needed) from the cell features to
    duel win and HP left. 12,564 rows exist today; Phase D adds more. Replaces the hand tiers
    inside `cover` with probabilities. Gate: held-out-boss Brier 15% below the tier baseline
    (safe = 0.9, race = 0.6, else 0.2, constants fitted).
  - **T2. Team model.** GBDT or logistic regression on the ~30 aggregates of §4.3 → P(win) and
    expected deaths per mon. Labels: Phase D rows (tagged with the battler version), 786 older
    rows for pre-fitting.
  - **T3. Deep Sets** over per-mon tokens (token = per-mon features + its row of six cells, the
    boss mons in the boss's own order), sum-pooled, small MLP head; per-mon death head. Only if
    T2 is above the noise ceiling, converged, and beaten by nothing simpler (§4.4).
- **Use:** enumerate or locally search the box through the model, keep the top 24 plus 6 random
  for calibration data, verify by search with halving. The model never picks the final six.
- **Gates (held-out bosses and held-out boxes):** the true top-3 by search is inside the model's
  top 24 in ≥ 80% of cases; within-box Spearman ≥ 0.5 and above the turn-0 MCTS root-value
  baseline by 0.1; end to end ≥ 90% of Phase D's win rate at ≤ 1/10 of its battles.
- **If it fails:** keep Phase D as the chooser where it is affordable; use T1 alone inside
  `cover` (it is independent of T2 / T3). For routing, fall back to cell-level expected gain.
- **Curriculum:** label boxes where the model's top candidates disagree most with search
  (largest |predicted − observed|) and boxes mutated from those (the PLR / ACCEL idea, in 10
  lines of sampling), instead of uniformly random boxes.

### Phase G. Routing (unchanged from the existing plan)

Expected gain of an encounter = change in the best scored six over the bosses ahead, through
whichever scorer survived (Phase D search for a handful of options, Phase F's filter for many).
Arithmetic on a scorer; no RL.

### Summary table

| Phase | Code | Machine | Gate |
|---|---|---|---|
| A instruments | ~70 lines | < 30 min | identical paired results with logging on |
| B opponent in tree | ~30–55 lines | ~1 h | SPRT: +8 pts vs minimax on Misty + Erika, Brock ±3 |
| C blame the walls | ~10 lines | ~1 h | a reading: battler / team / fidelity |
| D team + set search | ~90 lines | ~45 min / boss | Koga and Blaine ≥ 40%, fresh seeds, n = 48 |
| E learned evaluation | ~120–200 lines | fit: seconds; tests ~1 h | held-out loss −5%; SPRT +5 pts or −0.05 d/mon |
| F team filter | ~150 lines | minutes + verification | top-3 in top-24 ≥ 80% on held-out bosses |
| G routing | existing plan | — | — |

About 8–10 working days if every gate passes; B and C alone (2 days) decide most of the rest.

## 6. What I would not do, and why

- **A policy prior / pUCT / AlphaZero-lite in Rust.** The search is evaluation-bound, not
  width-bound (≤ 9 actions a side); the repo's one round of policy distillation gained nothing;
  the Rust work is 3–5 days. Revisit only if Phase E's fitted value passes and the ladder still
  rises steeply with iterations.
- **PPO or any model-free battler.** 14% from scratch, oscillation around a BC init, and
  VGC-Bench's finding that learned battlers generalise poorly across teams.
- **A learned model of the enemy AI.** The script is known. A learned copy would be a second
  imperfect model to exploit. Fix the port; use the mixture as insurance.
- **Nash-seeking search (Exp3 / regret matching, CFR).** The right target against a fixed script
  is a best response with a safety margin, not an equilibrium.
- **A pointer-network builder trained with REINFORCE, or direct RL over teams.** Too few
  labels, and unnecessary while a millisecond scorer can enumerate or hill-climb the box.
- **Species embeddings / one-hots.** The generality rule; and with hundreds of rows per boss
  they would memorise.
- **Offline RL, learned world models, an LLM in the loop.** The simulator is exact and cheap;
  none of these fit "one person understands all of it".
- **Bayesian optimisation with a GP over sixes.** A GBDT filter plus halving does the same job
  with less code.
- **Full AIVAT.** The continuous score and paired seeds get most of the variance reduction.
- **More Koga-specific work before the ROM check on permanent Tailwind.**

## 7. Risks and open points

- My ranking of Phase B as the largest lever is a judgement from the code and the literature,
  not a measurement.
- Texel-style fits learn the value of the *current* battler's play. If Phase B changes the
  battler a lot, refit; the plan orders B before E for this reason.
- States within a battle are correlated; effective sample size is closer to the number of
  battles than of states. Splits and confidence intervals are by battle.
- The table features assume damage at full HP and no boosts for bench mons; that is an
  approximation chosen for speed. If the residual probe shows boosts on the enemy active matter
  beyond feature 13, extend the active-vs-active terms only.
- λ in the utility is a preference. Report wins and deaths per mon separately at every gate so
  it cannot hide a trade.
- Reference human teams need the user.

## Sources seen in search results this session

- [Texel's Tuning Method, Chess Programming Wiki](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
- [Adversarial Policies Beat Superhuman Go AIs (arXiv 2211.00241)](https://arxiv.org/abs/2211.00241v4)
- [Restricted Nash Response (NeurIPS 2007)](https://www.cs.ualberta.ca/~games/poker/publications/NIPS07-rnash.pdf)
- [Data Biased Robust Counter Strategies (AISTATS 2009)](https://webdocs.cs.ualberta.ca/%7Ebowling/papers/09aistats.pdf)
- [Continual Depth-limited Responses (arXiv 2112.12594)](https://arxiv.org/pdf/2112.12594)
- [Convergence of MCTS in Simultaneous Move Games (arXiv 1310.8613)](https://arxiv.org/pdf/1310.8613v1)
- [Analysis of Hannan consistent selection for MCTS in simultaneous move games](https://link.springer.com/content/pdf/10.1007/s10994-019-05832-z.pdf)
- [Policy improvement by planning with Gumbel (ICLR 2022)](https://iclr.cc/virtual/2022/poster/6418)
- [PokéChamp (arXiv 2503.04094)](https://arxiv.org/abs/2503.04094)
- [Metamon: Human-Level Competitive Pokémon via Scalable Offline RL with Transformers (arXiv 2504.04395)](https://arxiv.org/pdf/2504.04395)
- [VGC-Bench (arXiv 2506.10326)](https://arxiv.org/pdf/2506.10326)
- [Winning at Pokémon Random Battles Using Reinforcement Learning (MIT DSpace record)](https://dspace.mit.edu/handle/1721.1/153888)
- [Smogon post: teambuilding as an optimisation problem](https://www.smogon.com/forums/posts/10676712)
- [JueWuDraft: Which Heroes to Pick? (arXiv 2012.10171)](https://arxiv.org/pdf/2012.10171)
- [Towards Playing Full MOBA Games with Deep RL (arXiv 2011.12692)](https://arxiv.org/pdf/2011.12692)
- [The Art of Drafting (arXiv 1806.10130)](https://arxiv.org/pdf/1806.10130)
- [BPCoach (arXiv 2311.05912)](https://arxiv.org/pdf/2311.05912)
- [Best Arm Identification with Fixed Budget: A Large Deviation Perspective (arXiv 2312.12137)](https://arxiv.org/pdf/2312.12137)
- [Fishtest Mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
- [AIVAT (arXiv 1612.06915)](https://www.arxiv.org/pdf/1612.06915)
- [Evolving Curricula with Regret-Based Environment Design, ACCEL (arXiv 2203.01302)](https://arxiv.org/pdf/2203.01302.pdf)
