# Notebook

The lab notebook of attempt 3, chronological, with every number. Was the
`README.md → Status` section until 2026-10-10. `HANDOFF.md` is the compressed
state; `project_overview.md` the plan; `results/` holds the logs each table cites.
Entries are in the order they were written; later entries supersede earlier ones.

## Milestones (2026-08-18 .. 08-19)

- Showdown + poke-env working, battle state readable (smoke test)
- Battle env + PPO + TensorBoard; smoke run shows learning signal

## Battler beats heuristic: 500 updates / 512k steps / ~35 min single env

95% win rate (crossed 50% at update ~128 / 130k steps), reward -17.9→+15.4,
entropy 1.59→0.54 (healthy), checkpoint `checkpoints/ppo_latest.pt`

## Team pool (16 teams harvested from Showdown's generator) + attention policy

(token obs, 1-layer transformer) + parallel envs (--envs N, ~2.9x at N=4)

## Pool of 4 solved by brute force: 5000 updates x 2048 rollout / 10.2M

steps / 4 envs (~3.9h, ~880 it/s) → **94% win rate** (all teams ≥85%).
Diagnosis: sample starvation, not architecture. Curriculum unnecessary.
Checkpoint `checkpoints/ppo_latest_attention.pt`

## MEASURED (2026-08-19): the 94% pool-4 checkpoint scores 0/20 vs unseen

Showdown random teams** (greedy policy). Generalization gap is total.
Therian formes in harvested random teams mangle through
parse_showdown_team→join_team → server reject / hang; eval_teams.py
rebuilds env on timeout as workaround.

## Path decided (2026-08-19): gauntlet env first (heuristic battler),

harvest realistic teams, retrain battler on that distribution.
Player level rule: match highest enemy level seen, never decrease.

## Nuzlocke gauntlet env + healthy meta-policy (run 4, 2026-08-20)

500 updates / 4 envs / ~2h → trainers cleared ~2.1 (was flat 1.0),
battle win ~66%, harvest ~60% two-mon parties, explained variance ~0.60.
Greedy probe: catches encounters, fields both mons, ep 6 cleared 7
trainers. Checkpoint `checkpoints/nuzlocke_latest.pt` (commit `84dce2d`).
Lesson: reward shaping alone hit degenerate optima (solo-starter, then
catch-and-bench); the forced-full-party action mask fixed alignment.
Harvest: `data/teams/gauntlet_harvest/battles.jsonl` (grew past 240k
during run 5; BDSP species, levels ~3–14) — battler retrain distribution.

## (open, superseded) Next: battler retrain on gauntlet_harvest — small JSONL loader into

existing pool-mode training, then `eval_teams` on held-out gauntlet
battles (closes the OOD loop). Meta-policy headroom remains (dies ~trainer
3–5; entropy ~0.02 at heuristic-vs-heuristic ceiling) — push deeper via
longer training, level-rule revisit, or a better battler.

## (open, superseded) TODO (later, exact approach TBD): player-side battler needs

Radical-Red-level competency — SimpleHeuristics is Showdown-oriented,
weak, and wrong-shaped for nuzlocke decision-making (HP conservation,
death-risk respect, sacrifice ordering). Opponent AI can stay dumber by
trainer class (random trainers < gym leaders < E4/champion); old project
reserved `radical_red`/`elite_redux` opponent keys the same way.

## Throughput (2026-08-21): sync-per-step ThreadPool capped ~40 st/s

(one battling env stalled all siblings). Fix: one process per env
(GIL bypass; SimpleHeuristics is Python-CPU-bound) + independent
rollouts + 60s battle `wait_for` + checkpoint resume. Measured:
threads ~80 st/s → 12 proc ~110 → **24 proc ~130 st/s** (~580MB/worker).
Meta-step ≠ battler turn-step (each confirm is a full Showdown match).
RAM: 24 envs needs ~22GB train+sd; OOM'd at 59/60 when desktop+swap
were already hot — fine after reboot with ~45GB free.

## (open, superseded) Meta-policy run 5: 24 envs / ports 8020–8043, `ROLLOUT=1024`, seed 1,

log `logs/nuzlocke_train5.log` (also `/tmp/nuzlocke_train5.log` from
earlier). Crashed at update 330 on empty-party confirm (fixed in
nuzlocke_env mask/step + worker try/except).

## (open, superseded) Battler v2 vs SimpleHeuristics (`ai_util/train_battler.py`; baseline

self_play.py 33% n=100). Two changes vs baseline: full-info obs (12 mon
slots × 58: hp/status/boosts/types/moves/base-stats + field block = 714
dims; opponent bench encoded once revealed) and fixed-opponent mix
(SimpleHeuristics 50 / MaxBP 25 / Random 25) instead of pure self-play.
PPO: gamma 0.95, ent_coef 0.05, max_grad_norm 5.0 (CLAUDE.md lessons).
- Stage A obs dump (2026-09-02): sane — no NaN, ranges as expected,
  active flags track switches, battles complete vs SimpleHeuristics.
- Stage B smoke 51k steps / 2 envs: 665–719 fps (old run: 636), entropy
  ~1.6 healthy, value_loss 43→17, no NaN, checkpoints written. Win rates
  at this scale are noise (Random 74%, MaxBP 18%, heuristics 5%).
- Stage C 500k steps / 12 envs (~1150 fps): **win rates flat** (heuristics
  0–10%, MaxBP 22–24%) BUT reward −33→−17.5, explained_variance →0.6,
  entropy 1.64→1.31 (healthy). Reads as "learns HP/status shaping, hasn't
  converted to wins yet" — same ordering as the old run's final numbers,
  which took 20M steps. Proceeding to Stage D 4M before touching
  ent_coef/reward weights.
- Stage D crash found + fixed: heuristic opponents can pick a move that
  poke-env can't map back to an action (`available_moves` vs known-moves
  desync, ~1/250k steps) — killed a worker. Fix: SafeSingleAgentWrapper
  degrades that opponent order to a default order (RL side stays strict).
- **Stage D 4M verdict: reward-structure bug, not slow learning.** Win
  rates flat the whole run (heuristics 0–8%, MaxBP 6–30%, final n=200:
  Random 72 / MaxBP 22 / heuristics 4). ep_rew_mean plateaued ~−20 from
  25% on, entropy pinned 1.28, EV ~0.6: converged to a "farm HP shaping"
  local optimum. Cause: 40-turn episodes × gamma 0.95 → the +30 win is
  worth ~3.9 at turn 1 while HP shaping pays ±1/turn. **Fix for D2:
  gamma 0.99 + shaping down (hp 1.0→0.25, status 0.5→0.1)** — win is now
  worth ~20 at turn 1 vs ~±15 total shaping swing.
- **Stage D2 verdict: reward rebalance necessary but not sufficient.**
  Flat again (heuristics 0–10%, final n=200: Random 74 / MaxBP 26 /
  heuristics 4). Behavior stats on the final checkpoint (`--eval`):
  **switch rate 25%** (vs opponents that never switch — every switch is
  a free hit) and max-effectiveness move only 68% of the time; entropy
  pinned ~1.2 nats both runs (ent_coef 0.05). **Fix for D3: ent_coef
  0.01** (matches the 33% baseline run, which sharpened to entropy
  0.31 / 72% vs MaxBP), keep D2 reward, switch_rate now logged to TB.
- **Env-fairness check + corrected baseline (2026-09-02)**: drove the RL
  side of the training env with MaxBP's brain — 3/30, 0/20 vs
  SimpleHeuristics, matching the direct head-to-head. Env is fair; the
  earlier "MaxBP 88% vs heuristics" note was this metric inverted —
  `_win_rate` returns the subject's LOSSES; **SimpleHeuristics beats
  MaxBP 88–12**. So heuristics is genuinely strong vs naive play and the
  plateau is under-training, not sabotage.
- **Stage D3 (ent_coef 0.01) verdict: right trajectory, needs steps.**
  Entropy 1.67→0.88 (was pinned at 1.2 with 0.05), switch_rate
  0.50→0.30, EV ~0.7, evals still climbing at the 4M cutoff (last eval
  MaxBP 44% / heuristics 10%; final n=200: Random 80 / MaxBP 32 /
  heuristics 8). D4 = same config, 12M steps.
- **Stage D4 verdict (12M): more steps don't move it.** Heuristics
  pinned 2–8% the whole run, MaxBP noise 20–42%, switch_rate stuck
  ~0.33, entropy healthy ~0.7. Knob-tuning within PPO+this obs is
  exhausted; **budget capped at 5–6M steps for this model size**.
  Trainer split into two tiers (`train_battler.py`): `--single` (1 env,
  main process, as simple as possible) and default distributed
  (`--envs N --servers M`: round-robin over ports 8020..8020+M-1).
- **Stage E1 final verdict (6M, update-noise fix)**: approx_kl ~0.007
  (vs 0.03–0.06 before) — fix confirmed. Final n=200: Random 89 /
  MaxBP 45 / **heuristics 14%** — roughly 2x every D-series run, but
  the curve oscillated 2–16% with no trend after 500k: that's the
  from-scratch ceiling within the 5–6M budget. Mainline is now
  BC-init + RL finetune (F1).
- **E2 pre-registration (written BEFORE results)**: behavior cloning
  from SimpleHeuristics (`ai_util/pretrain_bc.py`: collect → masked-CE
  pretrain → `--init-from` RL finetune).
  *What the model receives* (714 dims): per mon slot (12 × 58): hp_frac,
  fainted, is_active, revealed; status one-hot 6; boosts 6 (÷6); types
  20; 4 moves × [bp/100, dmg_mult vs opposing active, is_status,
  is_priority]; base stats 6 (÷255). Field (18): weather 5, terrain 4,
  turn/20, hazards 4+4. **Not in obs** (expected impact order): move
  category physical/special (boosts unattributable to atk vs spa),
  accuracy, PP, species identity beyond the types+base-stats proxy, own
  items/abilities, tera state, volatile statuses (confusion, leech
  seed), sleep/tox counters. *Structurally hidden on randombattle*
  (irreducible for anyone): opponent bench until revealed, opponent
  items/abilities until shown, opponent EVs/nature → their true speed.
  *BC metrics*: floor = uniform-over-valid CE (measured on the data);
  **val CE stuck > 1.5 ≈ teacher uses info the obs lacks** → feature
  ablation next, not more depth. Switch acc ~0 with move acc high =
  missing switch-relevant info. CE low + win rate poor = memorization →
  more data, not capacity. Train AND val CE high = underfit (info or
  depth); train low / val high = too big for the data (505k params vs
  ~100k pairs → unlikely; val split decides). *BC eval expectations*:
  ≥95% vs Random, ≥80% vs MaxBP, 40–60% vs heuristics mirror; much
  lower = compounding error (open-loop BC limit), not capacity. *RL
  finetune expectations*: step-0 eval must ≈ BC mirror number (else
  init/mask bug); no collapse below half of init in the first 500k
  (else catastrophic forgetting → lower lr); switch_rate starts at the
  teacher's (~2–5%) and stays <15%; approx_kl <0.02; EV starts negative
  (fresh value head) and must climb >0.5 by 500k. *Headline: cross 33%
  within the 6M budget; target >50%.*
- **E2 results (v1→v3)**: 714-dim obs: val CE 1.237, BC policy 37%
  MaxBP / 6% mirror — compounding-error clause fired. Patch 1 (move
  category, accuracy, speed flag, canonical opp slots → 870 dims):
  CE 1.212, no change. **Read the teacher's source**: score = bp × STAB
  × stat-ratio × accuracy × expected-hits × type-mult; we encoded
  neither STAB nor expected-hits nor self-boost. Patch 2 (→1014 dims):
  CE 1.208, no change (Random 84 / MaxBP 42 / mirror 7%). Three
  named-feature hypotheses falsified → remaining candidates per
  pre-registration: **undertraining** (train CE still falling at epoch
  7, never converged) vs **architecture** (flat MLP can't cheaply
  express argmax-over-computed-scores). Test in flight: 48-epoch
  pretrain; if val CE floors >1.0, the architecture lever (set/attention
  head over candidate actions) is next.
- **E2 breakthrough: it was undertraining, not information.** 48-epoch
  pretrain: val CE 1.208 → **0.588**, top-1 fidelity 47% → **80%** (8
  epochs had never converged — the "obs deficit" was an artifact). The
  converged BC policy alone: Random 91 / **MaxBP 72 / heuristics 26%**
  (≈ the 20M-step baseline, from ~55 min of collect+pretrain),
  switch_rate 16% (teacher 18%), max-effectiveness move 85%
  (teacher 94%).
- **Stage F1 (running)**: 6M RL finetune from the BC init
  (`--init-from`, E1 hyperparams, port 8022). Step-0 eval MaxBP 82 /
  heuristics 32% — init transferred cleanly. Pre-registered: cross 33%
  within 6M, no early collapse, approx_kl <0.02.
- **Lunch experiments (2026-09-03, unattended)**:
  *Data scaling*: v3 (91k) + v4 (183k) = 274k pairs, 60-epoch pretrain →
  val CE 0.611 (vs 0.588 on 91k) — **data scale doesn't move the
  imitation ceiling**. *Capacity*: 512×512 net → val CE 0.626, train CE
  0.128 (big overfit gap) — **capacity doesn't either**. CE ~0.6 /
  ~82% fidelity is intrinsic: the teacher's near-tie argmaxes + the
  hidden-info minority are irreducible labels. *But play strength
  still improved with the cleaner run*: BC v5 256×256 = **Random 98 /
  MaxBP 78 / heuristics 40%** (vs 26–32% before) — above the 33%
  baseline with zero RL. 512×512 plays 36% (no gain; keep 256×256).
  *Stage F2 (running)*: 6M finetune from the 40% init on 8021; step-0
  eval 54% (n=50) / 40% (n=100). Only difference vs F1 is init quality,
  so F2-vs-F1 isolates what a stronger init buys.
- **F1 final verdict (6M finetune)**: n=200: Random 94 / MaxBP 78 /
  **heuristics 29%**. The whole curve is a random walk around the init
  (20–38%, no trend) — **RL as configured preserves but does not exceed
  the BC init.** Shared early-dip pattern in F1+F2 (init → −30–50% at
  ~500k → partial recovery): the freshly-random value head produces
  garbage advantages that knock the policy off its init before the
  critic catches up (F1 end: approx_kl 0.027, EV 0.67). F2 at 4M: 40%
  vs heuristics — same oscillation, higher level.
- **F2 (terminated externally at 4.45M, verdict already clear)**:
  identical shape to F1 — oscillates 22–44% around its ~40% init, no
  trend. **F1+F2 together: RL finetuning as configured neither collapses
  nor exceeds the BC init.** Current best artifact is the BC policy
  itself: `models/bc_v5_256.pt` = Random 98 / MaxBP 78 / **heuristics
  40%** (n=100) — above the 33% baseline with zero RL. Next levers,
  in order: stabilized finetune (freeze policy trunk ~300k while the
  value head learns, then low-lr unfreeze), DAgger-style relabeling of
  BC-visited states, self-play league for beyond-teacher signal.
- **Prototype battler on game teams (2026-09-04)**: `collect_game.py`
  samples real gauntlet matchups from `data/teams/gauntlet_harvest/
  battles.jsonl` (483k BDSP player-party vs trainer-team pairs) and BC's
  SimpleHeuristics in `gen8customgamenogimmicks`. Reference baseline:
  SimpleHeuristics-vs-SimpleHeuristics on holdout pairs = **68.5%**
  (n=200; matches the harvest's 68.4% over 483k — there is a structural
  ~68/32 first-side advantage). Collection: 240k pairs / 20k matchups /
  ~15 min (needs a turn>100 forfeit cap: some low-level pairs cannot
  damage each other and would battle forever; masks are gen8-22-dim,
  padded to the 26-dim head). Imitation here is near-perfect: val CE
  0.002, 100% top-1 by epoch 1 (vs 0.61/82% on gen9randombattle).
  Training with proper checkpointing (`--eval-pairs/--eval-every/
  --patience`), clean run: epoch-2 eval 64%, **best 70.0% @ epoch 5**,
  then 65/66/70 (never strictly above best) → **early stop @ epoch 14,
  patience 3**. The prototype converges in ~5 epochs (≈2 min GPU) and
  plateaus at teacher strength — `models/game_policy_best.pt` = Random
  n/a / **70% vs enemy heuristic on holdout game matchups** (baseline
  68.5%). Matches the gen9 finetune finding: pure imitation tops out at
  the teacher; exceeding it needs RL/self-play/DAgger on top.

## Step 1: in-process sim + line evaluator + search (2026-09-15)

`ai_util/sim_bridge.js` drives Showdown's simulator over JSON lines (new /
choose / fork+reseed / forcelose / state), no websocket server, no logins.
`ai_util/sim.py` mirrors each side's protocol stream into poke-env Battle
objects, so PolicyPlayer / SimpleHeuristics / embed_battle run unchanged;
fork rebuilds mirrors by replaying history. `ai_util/evaluate.py`:
evaluate() = Monte Carlo P(win), P(death_i), risk for one matchup (the
run-layer interface); Searcher = top-k rollout search over the BC prior
(fork, play each candidate out `rollouts` times vs the enemy AI, pick best
win − death_weight·deaths). Checks, in order:
  uv run pytest -q tests/test_sim.py
  uv run python ai_util/evaluate.py --check --n-pairs 200      # expect ~68.5%
  uv run python ai_util/evaluate.py --policy models/game_policy_best.pt --n-pairs 100   # expect ~70%
  uv run python ai_util/evaluate.py --policy models/game_policy_best.pt --search --n-pairs 30 --k 3 --rollouts 8
Go/no-go: search beats the prior-alone number on the same pairs.
- Sim runs (2026-09-15): `--check` on holdout **61.5%** (n=200), `--check
  --from-harvest` (same seed-0 sample as the old baseline) **62.0%** (n=200,
  7.5 turns avg, ~0.01 s/battle). Same sample on a live server today:
  **69.0%** (`collect_game.py baseline`). **Cause found** (`diff_sets.py`):
  the server's TeamValidator fills **252 EVs in every stat** when the
  format has no EV limit (Custom Game); the raw sim used 0 EVs, so every
  mon was weaker (L5 Turtwig 22 vs 25 HP). Fix: the bridge now unpacks +
  validates teams like the server. Confirmed: `diff_sets.py` 0 diffs,
  `--check --from-harvest` **69.5%** (server 69.0%).
- **Second defect, shared with the server path**: 16% of choices rejected
  ("can't switch to an active Pokémon"), 6% brain defaults — harvest
  parties often carry two of a species, and poke-env keys teams by
  nickname so duplicates collapse into one entry. Fix: `sim.py`
  dedupes nicknames (Bidoof, Bidoof2) and pre-populates our own team
  before any protocol line (poke-env's lazy path renames same-species
  entries). After: **0.00%** rejected/default, heuristic mirror 69.0%.
  `--raw-names` reproduces the old collapse for apples-to-apples checks.
- **Prior alone in the sim** (`models/game_policy_best.pt`, deduped teams):
  holdout[:100] **68.0%**, holdout[:30] 66.7% (server number was 70%
  with collapsed teams; within noise). 0 rejected choices.
- Search pilot v1 (k=3, r=8): expanded **0/269** decisions — the
  `confident=0.9` gate skipped everything because the converged BC prior
  is >0.9 almost always. Gate removed (opt-in `--confident`).
- **Step-1 go/no-go (holdout[:100], 3 battles/pair, ~4 s/pair)**:
  | brain | win | deaths/mon |
  |---|---|---|
  | prior (greedy BC) | 68.3% | 0.40 |
  | search k=3 r=8, death_weight 0.5 | **70.3%** | **0.33** |
  | search k=3 r=8, death_weight 0 | 71.3% | 0.38 |
  | search k=3 r=16, death_weight 0.5 | 71.0% | 0.32 |
  Every search run (30/100/300-battle) landed above every prior run
  (66.7–68.3), so the direction is consistent; the win gain (+2–3 pts) is
  inside n=300 noise (SE ~2.6), the death reduction (−18%) is the clearer
  effect and is what the run layer needs. Verdict: **go**, modestly. The
  ceiling is the rollout policy: values are "how the prior would finish
  from here", 1-ply. Levers next: deeper/PUCT search reusing forks,
  expert iteration (retrain prior on search choices), stronger enemy
  model (RR AI port) for boss fights.

## Step 2a–2d: sim-backed gauntlet + scripted planner (2026-09-15)

`SimBattleRunner` in gauntlet.py (same contract; no servers; `--backend sim`
default in train_nuzlocke), `estimate()` death-probability query,
7 risk features appended to the PRE_BATTLE obs (OBS_DIM +7; old
nuzlocke_latest.pt not resumed), boss fights use search when
`search=(k,r)`. `scripts/plan_gauntlet.py` = scripted planner: candidate
parties = top-6-by-level ∪ single swaps, scored P(win) − λ·mean P(death).

### Finding 1 — level rule dominates
20 episodes, heuristic battler,
old "match max enemy level" rule + margin:
  margin 0: 2.0 cleared / 1.0 no-death / 0% clear   (= run-4 ceiling)
  margin 2: 18.6 / 3.2 / 15%   margin 4: 50 / 21.4 / 100%   margin 6: 50 / 39 / 100%
The meta-policy's ~2.1 was the environment, not the policy. New default
`level_rule="cap"` (hardcore nuzlocke): player level = next boss's max
level; route trainers below you, boss at parity.

### Finding 2 — the prior is OOD under the cap rule
Cap rule, 20 eps:
  heuristic:            49.8 cleared / 32.9 no-death / 3.05 deaths / 95% clear
  BC prior:             34.2 / 16.4 / 5.60 / 25%
  prior + boss search:  36.3 / 16.0 / 6.55 / 30%
The harvest (483k, match rule, L3–14) doesn't cover cap-rule levels/
movesets; SimpleHeuristics is level-agnostic. Boss search can't help
when deaths happen on routes. Next: regenerate harvest under cap
(`GAUNTLET_HARVEST=... plan_gauntlet.py --fast`), recollect BC, retrain,
re-measure — the OOD loop, now minutes instead of hours.
Pipeline is now server-free end to end: `collect_game.py collect` runs
in the sim (200 matchups → 1047 pairs in seconds, 0 failures),
`collect_game.py holdout` writes eval pairs, `pretrain_bc.py`'s
win-rate eval uses the sim. `plan_gauntlet.py --fast` also disables the
obs risk queries (they made harvesting 10x slower: 5 s/episode).

### Third defect (poke-env mask)

`get_action_mask` marks a switch slot
legal by *species*, so with two of a species the active one is marked
too → rejected order → runner exception → run counted lost; and
`order_to_action` labels a switch as the first slot of that species (bad
BC labels). Fix: `battler_policy.action_mask` / `order_to_action` by
identity. Old prior in the gauntlet after the fix: 37.1 / 16.3 / 45%.

### OOD loop result

(cap harvest 24k battles → 104k BC pairs, 20 epochs,
val CE 0.084 / acc 97%, holdout win 100%): `models/game_policy_cap_best.pt`
  cap prior:            48.2 cleared / 29.1 no-death / 3.45 deaths / 90% clear
  cap prior + search:   47.3 / 29.0 / 4.45 / 80%
  (heuristic reference: 49.8 / 32.9 / 3.05 / 95%)
Prior ≈ heuristic now (20-episode noise); boss search still no effect →
find where deaths happen (per-trainer histogram) before tuning search.
Cap-rule holdout itself is easy (heuristic 99%, old prior 95%): the
route trainers are below the player; only bosses are at parity.

### Finding 3 — the slice had no boss
Roark is order_index 50; the
50-trainer slice ended at 49, so "boss search" never fired and the cap
fell back to L13. `SLICE_LEN=51` now (through Roark). Per-trainer death
histogram (`plan_gauntlet.py` prints the deadliest 8): Roark first,
Hiker Daniel (#32, 2×Geodude/Machop/Bidoof @11) second, everything else
small. 40 episodes, seed 300, `--fast` (top-6-by-level parties):
  heuristic:               50.5 cleared / 32.4 no-death / 6.97 deaths / 75% clear
  cap prior:               47.5 / 31.8 / 8.62 / 52%
  cap prior + Roark search 46.5 / 31.7 / 7.50 / 57%  (Roark losses 11→7)
Search at the boss again buys deaths, not much else; the prior still
trails the heuristic on the routes (Hiker Daniel).

### Finding 4 — risk-based party selection works
Same seeds, 50-slice,
heuristic battler, 40 episodes: planner parties (P(win) − λ·mean
P(death), n=8 per candidate) → **2.20 deaths/run, 34.7 no-death, 98%
clear**, vs 6.78 deaths / 26.9 no-death / 85% clear with top-6-by-level
(`--fast`). Same battler, same fights, 3× fewer deaths, 0 lost battles at
Hiker Daniel (27 deaths vs 65). Cost ~25 s/episode vs ~1 s. This is the
run layer's first real lever.

### Expert iteration v1 — null result
`collect_game.py collect_search`
(Searcher's chosen action as label) on ~890 Roark/Hiker Daniel
matchups → 14k pairs (95% searched; search switches 12% vs teacher
3.5%). Trained with the 104k teacher pairs (val CE 0.186; early stop on
the route holdout is useless — it saturates at 99% — so full 20 epochs).
`models/game_policy_ei.pt` on the 51-slice, 40 eps, fast:
  ei prior:              47.0 / 31.4 / 8.62 / 45%   (cap prior: 47.5 / 31.8 / 8.62 / 52%)
  ei prior + Roark search 45.5 / 32.2 / 7.38 / 48%; Roark deaths 98→55, losses 12→6
14k search labels diluted in 104k teacher labels changed nothing; search
itself keeps paying at the boss. Hiker Daniel (not a breakpoint) is then
the top killer → `--search-all`.

### Finding 5 — search at every trainer is the battler
Cap prior +
Searcher(k=3, r=8) on all 51 trainers, 20 eps, fast parties:
  **50.95 cleared / 33.8 no-death / 4.10 deaths / 95% clear**
  (same prior, no search: 47.5 / 31.8 / 8.62 / 52%; heuristic: 50.5 / 32.4 / 6.97 / 75%)
Roark: 14 deaths, 1 loss (was 98 / 12). Hiker Daniel 30 deaths, 0 losses
(was 68 / 5). Cost ~50 s/episode. This is the step-1 thesis holding at
run level: prior for candidates, engine for the value. The remaining
battler gap is not the prior — it's the value estimate (rollouts under
the prior, 1-ply).

### Combined default (risk-based parties + search-all)

, 20 episodes
seed 300: **51.0 cleared / 33.8 no-death / 2.65 deaths / 100% clear**,
0 lost battles anywhere (Roark 20 deaths, Hiker Daniel 11), ~90
s/episode (`results/plan_cap_full.log`). This is the run config to beat.

### 2l. Value estimate (2026-09-15, evening)
Benchmark: the match-rule
holdout (`models/gauntlet_holdout.json`, 200 pairs x3, seed 7) — the cap
prior is OOD there (59.2% win / 0.44 deaths per mon), which leaves room
for search to matter; the cap holdout saturates at 99%.
  prior:                                59.2% / 0.44
  search k=3 r=8, fresh RNG per rollout: 63.7% / 0.37   (`results/search_nocrn.log`)
  search k=3 r=8, common random numbers: 67.2% / 0.35   (`results/search_crn.log`)

### Common random numbers

(same engine seed for rollout j of every
candidate; `SimBattle.fork(seed=)`, Searcher default) is free and worth
+3.5 pts / −0.02 deaths per mon; 600 battles so ~2 pts is noise. Default now.

### Learned value head — null, twice
Critic trained by MSE on the battle
outcome (win ±1 − 0.5·fainted fraction), actor frozen (`pretrain_bc.py
--init --value-only`), used as the leaf of truncated rollouts (`--horizon`):
  v1 teacher data (305k decisions from both harvests; val MSE 0.72 → 0.30):
    h=0: 22.7% / 0.81   h=3: 42.5% / 0.62   h=0 k=6 r=16: 22.8% / 0.81
  v2 on-policy data (233k search leaves, outcome under the prior;
    val MSE 1.11 → 0.44):  h=0: 41.2% / 0.67
Both are below the prior alone. Diagnostic (`temp/diag_value.py`): argmax
agreement with full rollouts 44% (v1) / 57% (v2) over 3 candidates, and
v1 scores every switch candidate low — the teacher only switches when
already losing, so "just switched" states are labelled as losses
(off-policy value). Also: the value head is OOD on wait states (enemy's
forced switch after our KO); the Searcher settles those before scoring
(`_settle`). Verdict: rollouts over the engine stay the value estimate;
a 658k-param MLP critic does not resolve candidate differences that the
rollouts see. Kept the code paths (off by default) for one more attempt
with more leaf data / a bigger critic; delete them if that is also flat.
Gauntlet default re-measured with CRN (20 eps, seed 300,
`results/plan_cap_crn.log`): **51.0 cleared / 36.25 no-death / 2.40 deaths /
100% clear**, 0 lost battles (was 33.8 / 2.65; Roark 20 deaths again,
Hiker Daniel 13). Same direction as the holdout, within 20-episode noise.

### 2m. Evolution + slice parameter (2026-09-15, night)
Level-up
evolutions from Showdown's species data (`gauntlet.py --build-moves
--max-level 50` now also writes `data/evolutions.json`; move table goes to
L50 and covers the evolved forms). `_apply_level_rule` evolves a mon when
the rule raises its level (Wurmple's branch picked with the env rng;
item/friendship/trade evolutions not modelled). On the 51-slice the
starters and Starly evolve at 14 (Monferno, Staravia), Kricketot at 10.
Default config with evolution, 5 of 20 episodes before the machine was
rebooted (parallel runs exhausted RAM — see HANDOFF gotchas): all 5
cleared, deaths 1/3/1/2/4. `--slice N` (NuzlockeEnv slice_len) extends
the episode; 186 = through Gym Leader Gardenia (cap 22 after Roark).
186-slice, 20 eps, heuristic `--fast`: 115.2 cleared / 33.9 no-death /
11.3 deaths / 0% clears (Roark 44 deaths 3 losses, Hiker Daniel 35,
Collector Edwin #139 3x Sudowoodo@19 19 deaths 4 losses). Default config,
1 episode (260 s, ~1 GB): 146 cleared, 12 deaths, lost at #146 — one
death per trainer against route trainers 3–7 levels below the party, i.e.
the cap-14 prior is OOD at L22 with evolved species. Next: harvest at
slice 186 (`GAUNTLET_HARVEST=... --fast --slice 186`), collect, retrain,
re-measure. Harvest recording is now opt-in (env var unset = no writes);
earlier evaluation runs had been appending to `battles.jsonl` silently
(483k → 515k lines).

### OOD loop on the 186 slice

(one job at a time): harvest 300 fast
episodes (`battles_cap186.jsonl`, 35k matchups; heuristic 116.7 cleared /
34.6 no-death / 11.0 deaths / 0% clears), collect 91k BC pairs, 20 epochs
(val CE 0.102, acc 96%, switch 48%) → `models/game_policy_cap186.pt`.
With search-all + risk parties, 10 eps: **154.0 cleared / 60.7 no-death /
6.10 deaths / 0% clears** — but every episode ended at #158 regardless of
deaths (3 of 10 had ≤2 deaths). **Two defects, not a wall:** #158 is
Collector Fernando's 3x Mr. Mime — `Mon.pack` capitalized 'mr. mime' to
'Mr. mime', Showdown renames a nickname equal to the species to 'Mr. Mime',
poke-env keys by our nickname, three species matches → AssertionError;
#162 Tuber Jared's 'Shellos (East Sea)' packed literally → species
'eastsea' → KeyError. A runner exception ends the run as a loss, so both
looked like impossible trainers. Fix: `gauntlet.species_name` (pokedex
display name + forme resolution) + regression test.

### 186-slice after the fixes

(cap186 prior + search-all + risk parties,
10 eps, seed 300, ~300 s/ep, `results/plan_cap186_search2.log`):
  **162.9 cleared / 63.5 no-death / 12.0 deaths / 0% clears**
Every run ends by roster wipe (12 = MAX_ROSTER) between #154 and #170.
Deaths: Collector Edwin #139 (3x Sudowoodo@19) 22, Jogger Craig #163 12,
Commander Jupiter #159 11, rival Cedric #165 10 (Starly/Buizel/Roselia/
Monferno@21), Fernando #158 10, Roark 6. The first half (≤ #138) is
nearly free (3 of 10 runs had no death before #116); the Eterna stretch
at L19–21 vs our L22 roster of route catches is the new frontier — parity
again, now with a weak roster instead of a level rule. Encounters stop at
trainer 44 (12 hand-written routes), so no new catches arrive there.

### Roster levers (2026-09-15, late)

(1) dead mons now leave the roster
(`env.graveyard`), so box slots free up; (2) `data/encounters.json` v2: 22
BDSP areas gated by level band over the 186 order (the order is
level-sorted, not route-sorted), 10 encounters before Roark (was 12) and
12 after (was 0); new species Onix, Gastly, Meditite, Shellos, Bonsly,
Kadabra, Machoke. Harvest v2 (38k; heuristic fast 127.2 cleared / 33.3 /
17.7 deaths) → 96k pairs → `game_policy_cap186v2.pt` (val CE 0.137).
186-slice, 10 eps: **160.2 cleared / 48.7 no-death / 18.8 deaths / 0%**
(v1 roster: 162.9 / 63.5 / 12.0). Refills add catches and deaths, not
progress: runs still die at Edwin #139 (26), rival #164–166 (44), Gyarados
#169 (9 deaths, 3 losses); Roark up to 18 (fewer pre-Roark catches).
(3) Party score: the planner's candidates read +1.00 almost everywhere at
n=8 (`logs/mem_probe_186.log`), so ~5% death risks are invisible and add
up over 186 trainers. Added `hp` (mean party HP left) to `evaluate()` /
`risk()` and `+0.2·hp` to the score (`--hp-weight`, 0 = old score).
Result, 10 eps: 165.9 / 48.3 / 18.3 — same as without (160.2 / 48.7 /
18.8), within noise. Kept (harmless), not a lever. The frontier is a few
specific fights: Collector Edwin #139 (3x Sudowoodo@19, Rock Head +
Stone Edge / Hammer Arm / Wood Hammer: 28 deaths in 10 runs, ~3 per
fight against a L22 party), then Jupiter #159, Jared #162, Craig #163,
the rival #164–166 (Monferno/Prinplup/Grotle@21 + 3 mons), Gyarados #169.

### Edwin in isolation

(`models/edwin_pairs.json`: 30 harvest parties vs
his team, x3, seed 5):
  heuristic teacher          88.9% / 0.39 deaths per mon
  prior v2                   71.1% / 0.61
  prior v2 + search k3 r8    86.7% / 0.44
  prior v2 + search k4 r16   88.9% / 0.41   (3x cost)
The prior is far below its teacher here: 156 of 38k harvest matchups face
Edwin, and the late fights have 7–110 each. **Targeted data**: 400
late-game harvest parties x each of the 9 deadliest teams
(`battles_hard_synth.jsonl`, 3600 matchups; `collect --trainers`) → 44k
teacher pairs, trained with the 96k base → `game_policy_cap186v3.pt`
(val CE 0.149, switch acc 61%):
  prior v3                   86.7% / 0.51
  prior v3 + search k3 r8    **92.2% / 0.39**  (above the teacher)
Same lesson as attempt #3: "the prior can't do X" was undertraining on X.

### 186-slice with v3

(10 eps, seed 300, `results/plan_cap186v3_search.log`):
  **168.2 cleared / 44.8 no-death / 16.2 deaths / 10% clears** — first
  run through Gardenia. (v2: 160.2 / 48.7 / 18.8 / 0%.) Edwin 28 → 16
  deaths; the rival trio #164–166 is now the top killer (47 deaths, 3
  losses), then Roark 12, Craig #163 11, Gyarados #169 9 (2 losses).
Next: the same targeted-data step on the rival fights is cheap, but the
rival has 4 mons at 19–21 including an evolved starter — a real boss
fight. Boss-level search (k=4, r=16 = +2 pts on Edwin) at breakpoints and
the rival, and a roster/party layer that keeps a Fighting/Grass answer
alive for Sudowoodo/Gyarados, are the levers left before the run layer.

### Rival probe (2026-09-16)
The rival trio (trainer_ids 351/352/353, order
#164–166) was already in `battles_hard_synth.jsonl` (400 parties each), so a
second data round was pointless. `models/rival_pairs.json` = 10 hard-synth
parties x each variant, x3, seed 5 (`logs/rival_*.log`):
  heuristic teacher          96.7% / 0.29 deaths per mon
  prior v3                   96.7% / 0.37
  prior v3 + search k3 r8    98.9% / 0.20   (11 s/pair)
  prior v3 + search k4 r16   98.9% / 0.16   (34 s/pair)
The rival is not a win problem: the teacher itself loses ~1.7 mons per fight
(Starly's Endeavor + Quick Attack, 4 mons at 19–21 vs L22 route catches).
Prior = teacher; search halves the deaths and the wider width buys a bit more.
So the lever is the compute dial per trainer (roadmap 1b): `--deep k,r`
plays breakpoints and every fight the fielded party's n=8 risk query flags
(a loss, a death, or HP < 0.5) at the deep width; routes stay at k3 r8.
Also seen: 4/2908 choices rejected at k4 r16 ("can't switch to an active
Pokémon") — a stale-mask corner in the search, 0.14%, harmless (sim default).

### 1b. Compute dial per trainer (2026-09-16)
v3 prior, search-all k3 r8,
`--deep 4,16`, 10 eps, seed 300 (`results/plan_cap186v3_deep.log`):
  **175.4 cleared / 63.7 no-death / 15.7 deaths / 20% clears**, 21 deep
  fights per run, 440 s/ep (v3 without deep: 168.2 / 44.8 / 16.2 / 10%, 260 s/ep)
The rival trio dropped out of the killers (47 deaths → Cedric #166 10, the
other two below 7). Wall now: Edwin #139 24, Fisher Juan #169 (Gyarados) 18,
Craig #163 13, Roark 9, Jupiter 8, Fisher Kenneth #184 (Gyarados@22) 8 with 2
losses. Deaths per run barely moved (16.2 → 15.7): deep search converts
deaths at the rival into progress, and the runs then die later, at the
Gyarados fights. Same seeds, 10 episodes: treat the +7 cleared as a
direction, confirm on 20+. `plan_gauntlet.py --workers N` (2026-09-16) runs
episodes in parallel processes for that: one env + one sim each, ~1.1 GB
and one core per worker (Python and the Node sim alternate).

### 2q. Owed 51-slice table

(cap_best prior, search-all k3 r8, evolution +
encounters v2, 20 eps, seed 300, 5 workers, 7 min; `results/plan_cap51_evo_v2.log`):
  **51.0 cleared / 36.75 no-death / 3.75 deaths / 100% clears**
(stale table: 51.0 / 36.25 / 2.40 / 100%). Deaths up by ~1.3 per run:
Roark 33 (was 20), Hiker Daniel 16 (13), Picnicker Siena #36 14 (new).
Two fewer pre-Roark encounters (10 vs 12) means fewer swap candidates at
Roark; evolution at 14 does not compensate. Still 0 lost battles.

### Deep search confirmed at 20 episodes

(v3, search-all k3 r8, deep 4,16,
seeds 300–319, 5 workers, 47 min; `results/plan_cap186v3_deep20.log`):
  **178.6 cleared / 62.4 no-death / 13.4 deaths / 40% clears**, 24.9 deep
  fights per run (10-ep table: 175.4 / 63.7 / 15.7 / 20%; v3 without deep,
  10 eps: 168.2 / 44.8 / 16.2 / 10%)
8 of 20 runs cleared Gardenia. Deaths per run down 16.2 → 13.4. The wall
is now one fight: Collector Edwin #139 (3x Sudowoodo@19) with 42 deaths and
1 loss in 20 runs — a third of all deaths — then Fisher Juan's Gyarados
#169 (18, 1 loss), the rival trio (16/13/15, 1 loss), Craig 15, Fisher
Kenneth #184 (14, 1 loss). Edwin is where the party layer matters: three
Rock/Rock-Head Sudowoodo at 19 vs a party that has no Fighting/Grass/Water
answer kept alive. Next lever: type-aware candidate parties (2o), then the
Gyarados fights.

### `--refine 32` is null, and the 20-episode noise floor is measured
Same
seeds, deep 4,16 + top-3 parties re-queried at n=32, run twice by accident
(10 workers each, in parallel, 38 min; `results/plan_cap186v3_deep20_refine.log`
holds replicate B, replicate A's summary is in the notebook only):
  A: 175.85 cleared / 57.9 no-death / 15.8 deaths / 15% clears
  B: 177.85 / 55.4 / 13.95 / 30%          (no refine: 178.6 / 62.4 / 13.4 / 40%)
Two identical configs on identical seeds differ by 2 trainers, 2.5 no-death,
1.9 deaths and 15 points of clear rate: **that is the error bar of a
20-episode table** (the sim reseeds every battle). Anything inside it is
noise; the deep-search gain (+10 cleared, −3 deaths vs no-deep) is outside.
Refine sits inside it, so re-querying the finalists at n=32 does not help:
the candidate set, not the estimate, is the limit.
**`--climb` (one extra round of single swaps from the best party when it is
still hard) is null on the headline** (same seeds, 10 workers, 28 min;
`results/plan_cap186v3_deep20_climb.log`):
  178.55 cleared / 70.7 no-death / 13.4 deaths / 40% clears (no climb: 178.6 / 62.4 / 13.4 / 40%)
Edwin's deaths halved (42 → 24) and reappeared one wall later: Fisher Juan's
Gyarados #169 went 18 → 45 deaths with 4 losses. Deaths move, they do not
vanish, until every wall is handled.

### Is the candidate set the limit? Yes, at Edwin (probe, 2026-09-16)
Four
runs played to #139 with the default config, then at Edwin every C(alive, 6)
party scored with the planner's n=8 score against the swap set's pick; both
played out 16 times with the deep (4,16) search brain (scratch script, not
kept). Deaths per battle, planner → exhaustive: 0.94 → 0.62, 2.69 → 1.31,
1.81 → 0.50, 2.56 → 0.62 (mean 2.0 → 0.76). The exhaustive parties drop the
Flying/Bug/Electric mons and keep two or three Grass/Fighting/Water answers
— three changes from the base, out of the swap set's reach. Cost: 210–924
candidates, 30–135 s per fight. Second probe, four cheaper searches on
fresh rosters at Edwin (mean deaths per battle under the deep brain):
  swap set 1.60 | hill-climb until no gain 1.30 (4–9 s) | beam 3x3 1.25 (9–11 s) | exhaustive 1.08 (56–135 s)
Hill-climb and beam get stuck where the swap set's best is a local optimum
(seed 301: all three identical, exhaustive 1.69 → 1.19). At Fisher Juan the
roster is 2–10 mons by then, every search picks the same party, and the
Gyarados deaths (0.5–1.4 per battle) are the battler's: Gyarados@21
(Intimidate; Twister, Ice Fang, Scary Face, Waterfall) vs a party at 22
costs a mon unless an Electric lead one-shots it. Battler probe on that
roster's party (Luxio lead, n=16 battles each, deaths per battle): teacher
1.00, prior 1.12, search 3,8 1.00, search 4,16 0.75, search 6,32 0.94; a
scripted "switch out after taking a hit" wrapper is worse (1.75 on the
prior, 1.12 on search 4,16: the switch-in just eats a Waterfall). Width
does not buy a second mon here: this wall is one death per run at parity
unless the roster carries a specific answer, which the run layer would
have to plan for from the catch phase, not at the fight.

### Roster dump at the walls (seed 300, asked for on 2026-09-16)
Everything
sits at L22 (the cap); at Edwin the box was 12/12. Party at Gyarados:
Luxio (Spark is the only super-effective move in the box), Golbat,
Beautifly, Kricketune, Cherubi, Budew; bench Staravia, Shellos, Noctowl,
Bonsly. Two data defects surfaced: movesets are the last four
level-up moves only (Meditite@22 has no attacking move), and a full box
drops encounters silently. (A third, "evolved forms lose their ability",
was a false alarm: the hand-made ability table equals the dex's first
ability slot for all 41 species, and the validator gives evolved forms
that same first slot — Luxio's Rivalry is the rule, not a loss. The table
can go.) 20 heuristic replays: 10 of 351 offers dropped
(those runs die too often to fill the box); default-config replays with
encounter logging (default config, 2 seeds): **7 of 21 and 5 of 21 offers
dropped.** The box is full from Eterna Forest (#59, seed 300) or Route 206
(#81, seed 301) until deaths free slots, and what it drops is the
mid-game: Bronzor, Machop (twice), Gastly, Buizel, Duskull, Meditite,
Geodude — the Rock resist and the Fighting/Ghost answers to Edwin, the
Mr. Mimes and the rival — while three Kricketune and two Budew keep their
slots. Under the cap rule a catch at #120 is at the same level as one at
#4, so the 12-slot box (an artefact of the old fixed-size PPO obs, roadmap
2p) is a real ceiling on the run layer. Seed 300 finished Gardenia with
two mons left; seed 301 wiped at Gyarados with a 2-mon party. Next
measurement: the same table with the cap lifted (`--roster`).
Roadmap 3a carries the data fixes.

### Env upgrade E3 (2026-09-16, roadmap 2¾)
Encounter offers are queued
per area and answered one at a time (no more overwriting when two areas
open at one trainer); a dupes clause (default, `--no-dupes` to compare)
rerolls an area's table among families never owned; the hand-made
ability table is gone (first ability slot everywhere, which is what the
validator did anyway); `--roster N` lifts the box; `--verbose` traces
offers, catches and drops. `tests/test_nuzlocke_env.py`.

### Lifted box

(`--roster 30` ≥ starter + 21 areas = unbounded on this
slice; 12-box code otherwise identical, no dupes clause; 5 workers, 59 min;
`results/plan_cap186v3_deep20_roster30.log`):
  180.15 cleared / 58.0 no-death / 16.35 deaths / 45% clears (12-box: 178.6 / 62.4 / 13.4 / 40%)
Cleared, no-death and clear rate are inside the noise floor. Deaths per
run is **not comparable across box sizes**: 10 of the 20 runs end with
exactly 22 deaths, i.e. the whole box wiped at the Eterna walls, where a
12-box run would have wiped earlier with fewer bodies. The lever the
dropped encounters were supposed to be (a Bronzor or Machop for Edwin)
does not show: every catch is at the cap with level-up-only moves, so
the extra mons are more of the same. So E3c is decided by principle, not
gain: the box is unbounded by default from here (`--roster 30`), and
every table from now on says so in its header. Expect E4 (movesets) to
be where the extra catches start to matter.

### Dupes clause

(12-box, otherwise the baseline config; 10 workers, 28 min;
`results/plan_cap186v3_deep20_dupes.log`):
  176.25 cleared / 61.5 no-death / 14.0 deaths / 25% clears (no clause: 178.6 / 62.4 / 13.4 / 40%)
At the edge of the noise floor in the wrong direction (−2.3 cleared, −15
points, +0.6 deaths). Kept on anyway: it is the standard rule, and with
the box unbounded it is purely a variety rule (a dupe is no longer a
free body that costs a slot). The first table with both new defaults
(unbounded box + dupes clause, `results/plan_cap186v3_deep20_base2.log`) is
the new baseline every later env change is measured against.

### New baseline, base2

(unbounded box + dupes clause, no ability slots
yet; otherwise the default config; 10 workers, 33 min;
`results/plan_cap186v3_deep20_base2.log`):
  **183.15 cleared / 60.95 no-death / 16.25 deaths / 60% clears**, 21.6 deep fights
(12-box baseline 178.6 / 62.4 / 13.4 / 40%; lifted box alone 180.15 / 45%;
dupes clause alone on the 12-box 176.25 / 25%). Cleared +4.5 and clear
rate +20 points are outside the noise floor (±2, ±15); together the two
rules do what neither did alone, which fits the mechanism (the clause
only pays when the freed slot is not a fifth Zubat). Deaths per run
(16.25) is not comparable with 12-box tables: 8 of 20 runs wipe all 22
mons. 12 of 20 runs clear Gardenia. Wall order now: Fisher Kenneth #184
(Gyarados@22) 33, Edwin 26, Fisher Juan 26, the rival 22/19, Cameraman
Darryl #179 19, Roark 16. **This is the table every later env change is
measured against.**

### E1 game folder (2026-09-16)
Everything the env knows about BDSP is
`data/bdsp/`: `trainers.json`, `order.json`, `encounters.json`,
`learnsets.json` (every level-up move with its level, from the dex),
`evolutions.json`, `species.json`; one cached `gauntlet.load_game()` and
nothing else in code names the game. The last-four moveset rule is now
derived at load (`encounter_moves`) and checked equal to the old
per-level table on all 3300 (species, level) cells. A second game is a
second folder. Acceptance table (game folder + E3e ability slots,
otherwise the base2 config; 10 workers, 34 min; `results/plan_cap186v3_deep20_e3e.log`):
  184.0 cleared / 49.0 no-death / 15.5 deaths / 60% clears (base2: 183.15 / 60.95 / 16.25 / 60%)
Inside the noise floor on every headline: E1 preserved behaviour, and the
ability-slot roll is neutral at this slice (no-death 49 vs 61 is the
heavy-tailed first-death metric again). Fisher Kenneth's Gyarados@22
(#184) is now the top killer with 47 deaths in 20 runs.

### E4 `best4` movesets implemented (2026-09-16, not yet measured)
`gauntlet.encounter_moves(species, level, rule)`; `plan_gauntlet --moveset
best4`. From every level-up move learned so far: strongest STAB attack,
strongest attack of another type, next strongest, one status move from a
short preference list, slots filled with the next attacks. Variable-
damage attacks count as 50. Data: `data/bdsp/moves.json` (type, power,
category, accuracy; 246 moves, from `--build-moves`). At 22: Meditite
Confusion / Hidden Power / Bide / Feint (was Detect / Hidden Power / Mind
Reader / Feint), Luxio Spark / Bite / Tackle / Charge, Machop Revenge /
Seismic Toss / Karate Chop / Low Kick, Bronzor Extrasensory / Tackle /
Confusion / Hypnosis. It is a progression change: the plan is a table with
the current prior first (search adapts partly), then the OOD loop
(harvest → collect → pretrain) and the table again. Harvest done: 300
fast episodes under best4 + the new defaults, 40.6k matchups
(`battles_best4.jsonl`, `logs/harvest_best4.log`); the heuristic-fast
table there is 134.5 cleared / 33.8 no-death / 19.2 deaths / 0% (last4
harvest, 12-box: 127.2 / 33.3 / 17.7 / 0%) — confounded by the box and
the clause, so only a sanity number.

### best4 on the current (last4-trained) prior

(10 workers, 39 min;
`results/plan_cap186v3_deep20_best4.log`):
  180.2 cleared / 60.8 no-death / 19.2 deaths / 25% clears (base2: 183.15 / 60.95 / 16.25 / 60%)
Worse, outside the floor on clear rate and deaths. Two causes, one fixed:
(1) the rule handed Geodude Selfdestruct as its strongest attack (200
power) — a guaranteed death; self-KO moves are now excluded
(`SELF_KO_MOVES`), harvest and synth regenerated (40.9k + 3600
matchups); (2) the prior is out of distribution (its obs encodes the
moves it was cloned on). The rival trio went 22/19/15 → 37/29/23 deaths.
Next: the OOD loop under best4 (collect → pretrain →
`models/game_policy_best4.pt`) and the table with that prior; that is
the number E4 is judged on. Loop done (one job at a time, ~25 min):
collect 20k base matchups → 97k pairs + 3600 synth → 56k pairs
(`bc_game_20260916_205924`, `_205946`), 20 epochs → `models/game_policy_best4.pt`, val CE 0.146, acc
95%, switch acc 69% (v3: 0.149 / 61%).

### E4 verdict: best4 movesets + the best4 prior

(10 workers, 29 min;
`results/plan_cap186_best4prior.log`):
  **185.95 cleared / 101.8 no-death / 6.30 deaths / 95% clears**, 20.9 deep fights
  (base2, last4 + v3 prior: 183.15 / 60.95 / 16.25 / 60%)
Deaths per run 16.25 → 6.30, clear rate 60% → 95% (19 of 20 runs), no
lost battles, first death pushed from ~61 to ~102. Ten times the noise
floor. The whole gain is the environment: the same battler recipe, the
same planner, the same search; the mons just have attacking moves. Wall
order now: Fisher Kenneth's Gyarados@22 (#184) 24, Edwin 16, then the
rival at 8 and Roark 7. Runs are faster too (29 min vs 34: fewer deaths,
fewer deep fights). `--moveset best4` and `models/game_policy_best4.pt`
are the defaults from here; the v3 prior only works under last4.

### E2 + E5 (2026-09-16)
`nuzlocke_env.Rules` (level rule, margin, box,
dupes clause, moveset) is the one place the ruleset lives: the env takes
`rules=`, the planner builds it from its flags and prints it in every
table header. `env.tracker()` prints party / box / graveyard with level,
ability and moves at the end of a `--verbose` episode. The env plan (2¾)
is complete except E6 (variable-roster obs, only for a learned run layer).
Acceptance table on the finished code, all defaults (10 workers, 29 min;
`results/plan_cap186_final.log`):
  184.6 cleared / 97.95 no-death / 7.05 deaths / 85% clears (E4 table: 185.95 / 101.8 / 6.30 / 95%)
Inside the floor: the refactor preserved behaviour. Treat the pair as the
reference (≈185 cleared, ≈6.7 deaths, ≈90% clears). Walls: Fisher
Kenneth #184 27, Edwin 16, Fisher Juan 11, Tuber Jared #162 11, Roark 8.

### E3e ability slots (2026-09-16, after base2 was launched)
Catches roll
ability slot 0 or 1 at even odds (a Shinx is Rivalry or Intimidate),
starters get slot 0, evolution keeps the slot (Starly Keen Eye → Staravia
Intimidate; Shinx Intimidate → Luxio Intimidate). Data: `data/species.json`
from the dex (`gauntlet.py --build-species`; 275 species, also E1's
species file). Not in base2; measured with E4.

### Teacher-ceiling round (2026-09-18, roadmap 1a)
Edwin and rival probes
(30 pairs x3, v3 prior, last4, `logs` in the session scratchpad):
  Edwin  teacher 90.0% / 0.41   search k3r8 88.9% / 0.46   depth-2 search (6x compute) 90.0% / 0.38
         calc bot alone 91.1% / 0.33   prior candidates + calc-bot rollouts 84.4% / 0.45
  rival  teacher 97.8% / 0.31   search k3r8 96.7% / 0.20   depth-2 100% / 0.18
         calc bot alone 98.9% / 0.08   prior candidates + calc-bot rollouts 97.8% / 0.26
Depth-2 search (`evaluate.py --depth 2`: max over our top-k again at the
next decision, r2 play-outs each) is null: the searcher's limit is not
depth. The **calc bot** — a 90-line scripted policy that runs the real
damage formula (own stats from the request, enemy stats at the 252-EV /
31-IV fill, boosts, STAB, effectiveness, 0.925 mean roll), takes a KO when
it has one, switches out when the next hit kills and a bench mon
survives, otherwise the max expected-damage move — beats every search
line on the rival and matches the best on Edwin at zero search cost. Its
edge is one rule the teacher lacks: the teacher loses Luxio in 26/34 Edwin
slots and 20/27 rival slots by staying in. Mixing it into rollouts under
prior-chosen candidates is worse (the value assumes a policy that never
acts). Verdict: the ceiling was SimpleHeuristics' rules, not imitation;
the next prior is a clone of the calc bot (encoder needs level + stats).

### The walls were level-sort artefacts
Bulbapedia: Collector Edwin is in
the Cafe Cabin on Route 210 (after Solaceon, optional), Fisher Kenneth on
Route 213 (after Pastoria), the 19-21 rival is the Hearthome City fight —
all after Gardenia in route order. The order file was the trainer sheet
sorted by max level. Also found: the sheet has real EVs/IVs/natures per
trainer mon (rival Starly IVs 5) and `Mon.pack` drops them, so the sim
plays every trainer mon at 252 EVs / 31 IVs (Sudowoodo@19: HP 73 Def 66
SpD 47 vs ~57 / 50 / 31 real). Nothing catchable before Edwin one-shots
the sim's Sudowoodo at 22 (Roselia Magical Leaf 86% max roll; after one
Growth it does, outspeeding 54 vs 34); under real stats four species do.

### Route-ordered gauntlet (3a, 2026-09-18)
`scripts/build_order.py`
fetches Bulbapedia's BDSP trainer tables (raw wikitext, `{{Trainerentry}}`
with `game=BDSP`, rematches skipped) for the story locations in visit
order, splits twice-visited locations by level band (Route 204/205/208/
210/211/213, Oreburgh Gate, the lakes), matches each entry to
`trainers.json` by class+name+team (grunts by team, double battles as two
singles, wiki/sheet level typos by species), inserts the story battles by
id, and writes `data/bdsp/order.json` (405 trainers to Cynthia, every
optional included; `location`, `mandatory` per entry — mandatory only
for story battles and Galactic plot locations) plus the encounter gates
(`before_trainer` = first trainer of the location where the area opens,
`opens_at` for humans). Roark #20, Gardenia #58, Maylene #166, Wake #204
(now a breakpoint), Fantina #216, Byron #250, Candice #317, Volkner #355,
E4 #400-403, Cynthia #404. 3 wiki entries unmatched (missing from the
sheet), 301 sheet entries unplaced (rematches, tag partners, postgame,
Battle Tower). `SLICE_LEN` 51 -> 21; slices: 21 Roark / 59 Gardenia / 167
Maylene. Encounter tables still end at Route 215 (#152). Old order kept
as `order_bylevel.json` until the harvest is regenerated; every table
before this entry is on the old order and not comparable. One fast
episode through Gardenia: 59 cleared, 3 deaths (Roark 2, Gardenia 1).

### Trainer EVs/IVs/natures (2026-09-18)
`Mon.pack` writes the packed
string directly (poke-env's text roundtrip drops an all-zero EV spread and
the validator then fills 252s): trainers carry the sheet's nature / EVs /
IVs (1167 of 1563 trainer mons have 0 IVs and 0 EVs; bosses 31 / 510),
player mons get 31 IVs and a flat 85 EVs per stat (`PLAYER_EVS`). Edwin's
Sudowoodo@19: HP 73 -> 55; Roark's Cranidos 42 HP. **Route-order tables
through Maylene (slice 167; harvest 300 fast eps 2 min; collect 48k
pairs 1 min; pretrain val CE 0.061 -> `models/game_policy_route.pt`):**
  fast heuristic harvest (300)         167 / 73.7  / 3.80 / 100%
  heuristic + risk parties (20, 12 min) 167 / 143.8 / 2.20 / 100%
  prior + search 3,8 all, deep 4,16 (20, 26 min) 167 / 157.7 / 1.25 / 100%   (`results/plan_route167_*.log`)
Maylene's Lucario takes exactly one mon per run under every brain (20/20);
the rest is noise. The slice through Maylene is solved; the benchmark is
the full game now.
**Env finished for the full game (2026-09-18): encounters + evolution
methods.** `build_order.py` also parses Bulbapedia's Generation VIII
`{{Catch/entrybdsp}}` tables (walking encounters only: Grass / Cave /
Building, BD column, max over time-of-day columns; surf, fishing, radar,
swarm, honey, Underground excluded) into `encounters.json`: 43 areas, 225
rows, one area per story visit (twice-visited pages split by sub-area
keyword or level band; Mt. Coronet = South / Celestic side / Snowpoint
side), species as Showdown ids. Replaces the hand-written v2 tables (which
stopped at Route 215 and were recollection). Evolutions: `evolutions.json`
carries every method from the dex (gen 4 dex, no regional formes);
`gauntlet.evolve(species, level, rng, caught_level, trainer_idx)`: level
always; friendship at catch level + 10 (`FRIENDSHIP_LEVELS`; Mon.caught_level);
stones / held items / places behind location gates (`STONE_GATE` = Eterna
Gym, the Grand Underground; Moss Rock = Eterna Forest, Ice Rock = Route
217, magnetic field = Route 208, Razor Claw/Fang = Victory Road); Mimic
evolutions at the learn level; trade never. Branches (Eevee, Wurmple) pick
with the env rng. 81 species have learnsets, 274 species.json. One fast
episode over all 405: 397 cleared, 32 deaths, wiped by the League rival —
the full game is a real benchmark.

### Water, gifts, rival, box (2026-09-18, later)
Encounter rows carry a
`method`: walk (Grass/Cave/Building), old / good (Old Rod from the
Jubilife gate house = Route 203, Good Rod = Route 209; Super Rod is
postgame), surf (HM after Celestic = Route 218), gift (gift / egg rows:
Happiny egg in Hearthome, Riolu egg on Iron Island; Mew / Jirachi /
starters / postgame Eevee excluded). An area opens at its first visit if
any method is available then, else when the first water method arrives
(Twinleaf Town opens with the Old Rod); the pool is the methods available
at that point. The env rolls **one pick per method** and the run layer
takes at most one (`pending` is a list of alternatives; action k = take
alternative k); the scripted planner picks by the base stat total of the
species' final evolution (`final_bst`; Magikarp -> Gyarados wins the rod
every time). Dupes clause per method: a rod whose only family is owned is
not offered, and an area with no new family gives no catch (was: a dupe).
One encounter per route: the second visits of Routes 204 / 205 / 210 /
211 and the Surf revisits are not areas. 47 areas, 301 rows. **Rival:**
the table has one record per starter choice and the order carried all
three, so every rival battle was fought three times (the level-sorted
order did the same); `NuzlockeEnv._skip_other_rivals` now fights only the
variant whose starter counters ours (Turtwig -> the Chimchar rival). The
skipped records still count as cleared in the summary (+2 per rival
battle). **Box** default 30 -> 50 (47 areas). Tests cover both. Fast
episode over 405, seed 11: 400 cleared, 29 deaths, Riolu and one
Gyarados in the box. The first full-game tables (harvest
`battles_route405.jsonl`, prior `game_policy_route405.pt`, val CE 0.232)
ran on the env *before* these fixes: heuristic table
`results/plan_route405_heuristic.log`, the search table was stopped at 8/20
episodes (superseded). Regeneration on the fixed env: `*_route405b*`.
**First full-game table (old env: three rivals, walking encounters, box
30; heuristic + risk parties, 20 eps, 5 workers, 90 min,
`results/plan_route405_heuristic.log`):**
  402.5 cleared / 149.9 no-death / 32.45 deaths / 5% clears
Deaths sit where a Nuzlocke's do: Cynthia 55 deaths + 11 wipes / 20 runs,
the League rival 52-53 per variant, Flint 48, Aaron 36, Lucian 36,
Bertha 31, Volkner 28. Everything before Sunyshore is noise. The full
game is the benchmark and the League is the wall.
**Fixed env (one rival, water / gifts, box 50), harvest 200 fast eps
(394.6 / 76.7 / 36.0), collect 95.6k pairs, `game_policy_route405b.pt`
val CE 0.219. Prior + search 3,8 all, deep 4,16 (10 eps, 10 workers, 67
min, `results/plan_route405b_search.log`):**
  403.4 cleared / 184.8 no-death / 27.6 deaths / 0% clears, 20 deep fights per run
Cynthia 48 deaths and 8 wipes in 10 runs, Flint 38, Bertha 22, Lucian
21, Aaron 19, the League rival 18, Volkner 13, Maylene 11. Search moves
the first death from ~150 to ~185 and trims the E4, but Cynthia at parity
(Garchomp@66 vs our 66s) wipes 8 of 10 runs: the prior is a heuristic
clone and the rollouts are heuristic-quality, and the five League fights
are still played on full heals. This is where the calc-bot teacher, the
League-healing rule and the BDSP mod get measured.
Heuristic + risk parties on the same env (20 eps, 5 workers,
`results/plan_route405b_heuristic.log`): **403.05 / 139.2 / 31.95 / 10%**
(2 of 20 runs beat Cynthia). Search buys ~45 trainers of first death and
~4 deaths per run; neither line beats the League reliably.

### Calc-bot teacher (2026-09-19, roadmap 1a)
`ai_util/calc_bot.py`
(the probe promoted; enemy stats estimated at 31 IV / 85 EV, boss-safe,
since route trainers are 0/0 and bosses 31/510). `collect_game.py
--teacher calc` labels with it (switch 12% of decisions vs 1-3% for the
heuristic); `plan_gauntlet.py --policy calc` plays it without search. The
encoder gains level + real stats per mon (own from the request, theirs
estimated the same way; N_FEATURES 1014 -> 1098), so every earlier prior
is incompatible and `game_policy_heur.pt` is re-cloned from the heuristic
on the same matchups as the control. Chain: collect 30k battles x 2
teachers -> `game_policy_calc.pt` / `game_policy_heur.pt` -> tables on the
full game: calc bot alone (20 eps), calc prior + search, heur prior +
search (10 eps each).
Pretrain: calc clone val CE 0.455 / acc 84% / switch 18% (the damage
arithmetic does not clone into a 256x256 MLP from 94k pairs); heuristic
clone on the new encoder 0.213 / 93% / 37% (= yesterday's 0.219: the new
features are neutral for it). **Calc prior + search 3,8 all, deep 4,16
(10 eps, 10 workers, 62 min, `results/plan_route405b_calcprior_search.log`):**
  385.5 cleared / 196.1 no-death / 17.9 deaths / **30% clears** (3 of 10 beat Cynthia)
vs the heuristic prior + search 403.4 / 184.8 / 27.6 / 0%. Cynthia 47
deaths + 6 wipes (was 48 + 8), Flint 18 (38), Lucian 16 (21), Aaron 14
(19), Bertha 11 (22), the League rival 10 (18): the E4 cost halves. The
clone's weakness shows once: a Fantina wipe at #216 (12 deaths there),
which the heuristic prior never had. First full-game clears with search.

### Search over the calc bot, no network (2026-09-19)
`Searcher(policy=None,
rollout=calc_brain())`: every legal action is a candidate (up to k) and the
calc bot plays the rollouts; `evaluate.py --policy calc --search --k 9`,
`plan_gauntlet.py --policy calc --search 9,8`. Found a **latent defect**
with it: `dedupe_nicknames` left the first of two same-species mons with
its bare name, and Showdown resolves 'switch Staravia' to the *active*
Staravia2 by species -> "can't switch to an active Pokémon" (2.4% of
choices under all-legal candidates; the prior seldom proposed it, the
notebook's 0.14% rejections at k4 r16 were this). Every duplicate now
gets a suffix (Staravia1, Staravia2); rejections 0.00%.
**Calc bot alone, full game (20 eps, 5 workers, 109 min,
`results/plan_route405b_calcbot.log`):** 404 / 222.65 / 15.9 deaths / 15%
clears. Every run reaches the League; Cynthia 95 deaths + 15 wipes / 20,
Lucian 38, Bertha 37, Flint 29, Candice 16, Aaron 13, Byron 10, Volkner
9. Fewer deaths than any prior line (heur+search 27.6, calc prior+search
17.9) at zero search cost; the clone + search wins on clears (30% vs 15%)
by trading route deaths for League deaths.

### Encoder control

the heuristic clone on the new encoder + search, same
matchups (10 eps, `results/plan_route405b_heurprior_search.log`): 403.3 /
176.8 / 28.3 / 0% — yesterday's 27.6 on the old encoder. Level + real
stats change nothing for the heuristic clone; the teacher is the lever.

### Search over the calc bot, candidate order
With no prior, switch
actions came first among candidates and won every tie in the noisy
rollout values: battles stalled (42 turns vs 13) and Edwin went to 0.45
deaths/mon (calc alone 0.33). The rollout brain's own choice is now the
first candidate; 6-pair smoke back to 0.31 / 23 turns. Probes (30 x3):
  Edwin  calc+search k9 r8  86.7% / 0.35     (calc alone 91.1% / 0.33, teacher 90.0% / 0.41, prior+search 88.9% / 0.46)
  rival  calc+search k9 r8  97.8% / 0.03     (calc alone 98.9% / 0.08, teacher 97.8% / 0.31, prior+search 96.7% / 0.20)
Search over the calc bot is the best line on the rival and neutral on
Edwin, where no play fixes a party without a Rock answer. ~2 min/pair
under load; a full-game episode is hours, so the League-rule table runs
10 episodes on 10 workers.

### The League block (2026-09-19)
Bertha, Flint, Lucian and Cynthia
carry `no_heal` in `order.json` (Aaron is entered fresh). Inside the
block the party is locked (no swaps, no box) and every fight starts from
the last one's HP / status / PP: `sim_bridge.js new` takes `carry` per
p1 mon (the battle is built with p1 only, edited, then p2 is added, so
the first request already shows it), `SimBattleRunner.run` returns the
per-mon state, `NuzlockeEnv.carry` holds it, the risk query and the
fight use it, the planner steps through with `env.party_locked`. A fast
calc-bot episode: Aaron costs 2, the four survivors enter Bertha at
58 / 147 / 119 / 204 HP and wipe — every earlier League number was on
full heals. Tables under the rule: `results/plan_league_*.log`.

### Party for the whole block
At the fight that opens a no-heal block
(Aaron), `env.risk()` scores a candidate party by
`SimBattleRunner.estimate_block`: n samples of the five fights in a row
under the fast brain, state carried, P(all won) / deaths over the block /
HP left at the end. The planner's candidate loop is unchanged, so the
party that enters the League is the one that survives the League in
simulation, not the one that beats Aaron. ~0.4 s per candidate at n=8.
The single-fight query still drives the deep-search trigger.

### League with no items (2026-09-19, calc + search 9,8, 10 eps, 47 min)

402.0 / 345.3 / 6.7 deaths / 0% clears; ~1 death per run before the League,
then Aaron 5, Bertha 25, Flint 27 (8 wipes), Lucian 3 — the party enters
Flint on what Bertha left. The game is not this: a run enters each E4 fight
on Full Restores. Rule from here: **party locked, PP carried, HP / status
restored between fights** (items and money out of scope, so healing
between fights is free); the no-items line stays as the lower bound.
(Found on the way: killing a planner by its main PID orphaned its workers
and their sim bridges; 15 of them ran for 90 min. Kill the tree.)
**League with items between fights, calc + search 9,8 (10 eps, 36 min,
`results/plan_league2_calcsearch.log`):** 402.9 / 342.2 / 6.7 deaths / 10%
clears. Route ~1 death per run; the block: Bertha 23 deaths (1 wipe),
Cynthia 12 (3 wipes), Flint 10 (4 wipes), Lucian 10 (1), Aaron 2. Runs
end at 401 / 402x4 / 403 / 404x3, one clear. Full heals moved the wipes
from Flint to Cynthia and Flint but did not change the count: the locked
six chosen at Aaron (top-6-by-level + single swaps, scored by the block
estimate at n=8) do not carry a Bertha answer and a Garchomp answer at
once. Next: widen the candidate set at the block (`--parties beam`).
`--parties beam` on the same line (10 eps, 47 min): 402.3 / 380.4 / 6.8 /
0% — null again; the candidate set is not the limit. Found meanwhile: the
level rule kept stepping inside the block (57 at Aaron, 59 at Bertha, ...
66 at Cynthia), which is leveling where there are no wild encounters.
Rule now: **one cap per no-heal block = its highest level (66), reached
before Aaron** (the standard hardcore convention for the E4). The
summary also prints deaths by species from here.
**Block cap 66 + items between fights, calc + search 9,8 (10 eps, 38 min,
`results/plan_league3_calcsearch.log`):** 403.7 / 371.8 / 5.9 deaths / 0%
clears. Bertha 3, Flint 5, Lucian 16, **Cynthia 30 deaths + 8 wipes / 10**;
route ~1. Deaths by species: Bibarel 13, Gyarados 9, Staraptor 7, Luxray
5, Infernape 5. The E4 is handled at 66; Cynthia is the wall, and the
party that reaches her carries filler (Bibarel is top-6-by-level tie-break
material). Next: a Cynthia matchup probe from the harvest's League parties.
Calc bot alone, same rules (10 eps, 49 min, `results/plan_league3_calcbot.log`):
402.6 / 259.9 / 10.0 / 0% — dies at Flint (5) and Lucian (4); search halves
the League cost (5.9) and gets 8 of 10 runs to Cynthia.

### Cynthia probe (`models/cynthia66_pairs.json`)

150 six-mon parties the
fast heuristic fielded at trainers 380-399, re-leveled to 66 with best4
movesets, 85 EVs, vs Cynthia from full HP, n=2: **calc bot 2.7% wins,
0.99 deaths/mon** — a near-certain wipe at level parity against a team
with 31 IVs / 510 EVs and real held items. The League table's 8 wipes are
this, not the planner. Levers, all player-side realism: EV training
(`PLAYER_EVS` is a flat 85), TM movesets (E7), held items, in-battle
items. Species reaching the League: Weavile, Lapras, Abomasnow, Crobat,
Chimecho, Mantine, Chatot, Skuntank, Golduck, Luxray, Bronzong.
Player-stat sweep on the same probe (calc bot, n=2): 85 flat 2.7% wins;
legal EV training 252 attack + 252 HP 3.7%, 252 attack + 252 Spe 6.0%;
the old 252-in-every-stat fill 41%. Faints by her active mon per 100
battles: Garchomp 152, Lucario 144, Roserade 119, Gastrodon 75, Milotic
53, Spiritomb 52; we KO Spiritomb 100, Lucario 47, Gastrodon 39,
Roserade 33, Garchomp 24, Milotic 16. Our dead: Unown 46, Crobat 40,
Octillery 40, Girafarig 40, Kadabra 34, Blissey 32 — the level-sorted
party carries filler into the fight. Verdict: at level parity, under
hardcore rules (cap, no items in battle) with level-up movesets and
route-catch stats, Cynthia is a coin flip at best for the *best* party
and a wipe for a typical one. The legal levers are TM movesets (E7,
single-use TMs in BDSP), held items, and party building for the League
from the whole box (both need the run layer to plan for #404 from far out).
Calc + search 9,8 on the first 50 of those parties (n=2,
`results/probe_cynthia66_calcsearch.log`): **18.0% wins / 0.94 deaths per
mon** (calc alone on the same 150: 2.7% / 0.99). Search lifts the win
rate sixfold but a win still costs ~5 of 6 mons: no deathless line exists
for these parties. The box and the movesets are the lever, not the play. Roadmap: E8 League
party building, E7 TM movesets (reusable TMs, decided 2026-09-19), E9 items.
**Three env defects behind the League table (2026-09-23), each with a test
in `tests/test_nuzlocke_env.py`.** Six fast full-game runs on the old env
ended with 2–5 Crobats and two Gyarados / Bibarels each, under the dupes
clause: (a) `_owned_families` claimed one pre-evolution step only (a
Crobat left Zubat catchable) — now the whole line; (b) every area gated
at one trainer was rolled at once, so two water areas both rolled
Magikarp before either was answered — offers now roll one at a time
(`env._queue`), after the previous answer; (c) inside the League the
locked party was rebuilt with dataclass equality (`m in party`), so a
value-equal twin in the box (same species / level / moves / ability at
the cap) joined as a 7th member — the `Invalid action 6 ... not in valid
orders [6 switches]` rebuilds in `plan_league3_calcsearch.log`: two of
those ten runs ended in a crash, not at Cynthia. Now by identity. Six
fast runs on the fixed env: no duplicate species. **Every League table
before this is on a box full of Crobats.**
**E8, the League party from the whole box (2026-09-23,
`plan_gauntlet.py --league K`, default 10).** Every mon is at the cap at
the League, so top-6-by-level is roster order and the swap set is one
change away from the first six catches. At the block opener the planner
now shortlists the K mons by the best swap they appear in, scores every
6-subset of the shortlist on the block (n=8, ~0.5 s each), and re-scores
the top 8 at n=32 (winner's curse among hundreds of noisy scores). First
probe: every candidate wiped under the fast brain, all scores ≈ −0.9, and
the stable sort returned the base party — deaths carry no signal when all
six are certain. `estimate_block` therefore returns **progress** (fights
won + the lost fight's KOs, over the block; 1.0 = cleared) in the HP slot.
Same box after that: 475 candidates spread −0.94…−0.68; the E8 party's
mean P(death) at n=32 0.76 vs 0.92 for the best swap (progress 0.63 vs
0.69). ~2.5 min per episode at the opener. Table: swap vs E8 on the fixed
env, calc + search 9,8 / deep 9,16, seeds 300–309, 10 workers each
(`results/plan_league4_{swap,e8}.log`, 38 / 40 min, no rebuilds in either):
  swap (`--league 0`): 403.80 cleared / 368.0 no-death / 5.90 deaths / **20% clears**
    Cynthia 35 deaths, 6 losses; Flint 8, Bertha 8, Lucian 4; routes ~1
  E8 (`--league 10`):  404.00 / 325.6 / 6.40 / 10%
    Cynthia 34 deaths, 8 losses; Lucian 12, Bertha 6, Flint 6

### E8 is null

(noise floor ±1.9 deaths, ±15 points of clear rate). The
E8 parties are different (Crobat 7, Graveler 5, Lapras 4 among the dead
vs Staraptor / Luxray / Gyarados under swap) and no better: the fast
brain's block score does not order parties by what search-play does at
Cynthia (calc alone 2.7% there, search 18%), and no party in the box
has a deathless line anyway (the probe). Default back to `--league 0`;
the flag stays for a scorer that plays the block with search (8
candidates × n=4 × 5 fights × ~30 s ≈ 80 min per episode: not now).
**The benchmark table is the swap line on the fixed env: 403.8 / 368.0 /
5.9 / 20%.** The two clears beat Cynthia with 4 deaths each; the fix of
the three defects took the clear rate from 0% (two of ten runs crashed)
to 20%. Cynthia wipes 6 of 10. Next lever: E7 TM movesets.

### E7, TM movesets (2026-09-23, `--tms`, `Rules.tms`)
Data:
`scripts/build_tms.py` reads each TM's Bulbapedia page (TM001–TM100, the
BDSP row of its acquisition table) and writes `data/bdsp/tms.json` (move
→ TM number, the trainer index it is in hand before, the source). A TM
counts from its earliest *fixed* source: a pickup, an NPC gift (gym
leaders after their fight), the Veilstone Department Store (all 22 of
its TMs on arrival, #160, incl. Ice Beam / Thunderbolt / Flamethrower /
Swords Dance), the Route 212 shard collector. Not counted: Grand
Underground merchants (random stock; they would put 83 TMs at Eterna),
the Battle Tower and post-game areas — so Calm Mind, Bulk Up, Will-O-Wisp,
Thunder Wave, Energy Ball, Overheat, Facade, Explosion never arrive.
92 TMs gated: Earthquake at Wayward Cave (#78), Stone Edge / Dragon
Pulse / Dark Pulse at Victory Road (#381), Surf #235, Waterfall #347.
TMs are reusable (decided 2026-09-19). Compatibility from Showdown's
BDSP mod (`8M` flags, `--build-moves`; learnsets.json is now `{levelup,
tm}` per species, moves.json 344 moves). `encounter_moves(...,
progress=trainer_idx)` widens the best4 pool with the TMs in hand, and
the env recomputes moves at every fight (a TM lands the fight after).

### best4 v2
Over the wide pool the raw-power rule chose Hyper Beam /
Giga Impact / Focus Punch for nearly every species and Hydro Pump over
Waterfall on Gyarados (attack 125 / sp. attack 60); the calc bot cannot
value recharge or conditional moves, so those parties would have been
worse with TMs than without. The rule now scores an attack by power x
accuracy x STAB x category fit (atk or spa over the larger of the two),
the third attack must bring a third type, and `NO_SLOT_MOVES` (self-KO,
recharge / charge, conditional) never get a slot. Level-up-only picks
change too (Gyarados Aqua Tail / Thrash / Ice Fang / Dragon Dance instead
of Hydro Pump / Hyper Beam / Thrash), so the baseline moves with it.
With TMs at #404: Gyarados + Earthquake, Lapras Surf / Ice Beam /
Thunderbolt, Empoleon Surf / Flash Cannon / Ice Beam, Abomasnow Wood
Hammer / Earthquake / Ice Beam / Swords Dance, Golduck Surf / Psychic /
Ice Beam. The status slot is often Rest (the calc bot never uses it).
Tables (calc + search 9,8 / deep 9,16, seeds 300–309, 10 workers each,
29 min each, no rebuilds):
  best4 v2, level-up only (`results/plan_best4v2.log`):
    404.20 cleared / 337.4 no-death / 6.20 deaths / 20% clears
    Cynthia 34 deaths, 8 losses; Lucian 11, Flint 4, Bertha 4
  best4 v2 + TMs (`results/plan_best4v2_tms.log`):
    **404.60 / 358.6 / 5.20 / 70% clears**
    Cynthia 34 deaths, **2 losses**; Lucian 7, Bertha 2, Flint 2, Aaron 1

### TMs are the first lever that moves Cynthia

the rule change alone
(v2 vs v1: 20% vs 20%, 6.2 vs 5.9 deaths) is noise; the TMs take the
clear rate from 20% to 70% (noise floor ±15 points) and the E4 cost
from 19 to 12 deaths per 10 runs. Cynthia still costs 3.4 deaths per
run — the 8 clears beat her with 2–7 dead — so the League remains the
problem, now as a death count rather than a wall. `Rules.tms` is True by
default (`--no-tms` to compare). Deaths by species with TMs: Gyarados 8,
Bibarel 7, Luxray 6, Staraptor 6. Next levers, in order of cost: player
EV training (probe: flat 85 → 252/252 doubles the calc bot's Cynthia
wins), E7(d) the four moves chosen by the sim, E9 held items.

### Player EV training (2026-09-23, `--evs trained`, `Rules.evs`)

252
in the better attacking stat (base atk vs spa), 252 Spe, 4 HP — the
probe's best legal spread — on top of TMs, same seeds and config
(`results/plan_tms_evs.log`, 31 min): 404.50 / 375.9 / 5.80 / **50%
clears**; Cynthia 40 deaths, 5 losses; Flint 8, Bertha 2, Aaron 2,
Lucian 1. Against the TM line (70%, Cynthia 34 / 2) that is −20 points
at a ±15 noise floor: null at best, likely negative — the probe's
calc-bot-only gain (2.7% → 6.0% from full HP) does not transfer to
search play through the block, and a spread with nothing in HP / Def
dies to hits it used to survive. Flat stays the default. Untested: the
probe's other spread (252 attack + 252 HP, 3.7%), and natures.
**E7(d), the four moves chosen by the sim (2026-09-26, `plan_gauntlet.py
--moves sim`).** Rule: at every fight the planner flags hard (the deep
trigger: a loss, a death or HP < DEEP_HP at n=8), each party mon's set is
varied one slot at a time over the next three moves of its ranked pool
(`encounter_moves(..., k=None)`: level-up + TMs in hand); a variant that
beats the current set at n=8 and again at n=16 replaces it, greedily over
the party. The choice persists (`Mon.chosen`) while the pool holds it —
relearning is free — and the risk cache keys on the moves. Motivation: the
calc bot only ever uses set-up moves among status moves (`CalcPlayer`:
boosts, target self), so best4's status slot (Rest on Luxray, Lapras,
Golduck at #404) is a wasted slot the sim can see. A route risk query is
0.02 s, a Cynthia query 0.3 s, a block estimate 0.6 s: ~70 variants per
hard fight cost nothing on routes and ~40 s at the League opener. A test
found the first version scoring the unchanged set: `risk()` runs the
level rule, which rewrites `m.moves` from the rule — variants go through
`m.chosen`. Table, same seeds and config as the TM line
(`results/plan_tms_moves.log`, 35 min, 4.0 sets changed per run):
  404.20 / 392.7 / 6.10 / **20% clears**; Cynthia 42 deaths, **8 losses**;
  Lucian 7, Cedric 4, Flint 3 (TM line 404.6 / 358.6 / 5.2 / 70%, Cynthia 34 / 2).
Two effects pull apart: before the League the run is safer (no-death 358.6
→ 392.7, E4 12 → 13 deaths), at Cynthia it is much worse (2 → 8 losses),
−50 points at a ±15 floor. The League opener's choices are scored by the
fast brain's block progress — the instrument E8 already showed does not
transfer to search play at Cynthia (every party wipes under the fast
brain; the ordering is how far it gets through Aaron–Lucian, and a move
that helps there is bought at Garchomp). Second table: the same rule with
the League opener excluded (`env._block()`), so only route and gym
choices are made and carried in (`results/plan_tms_moves_noblock.log`, 36 min,
2.8 sets changed per run):
  404.40 / 354.1 / 5.10 / **40% clears**; Cynthia 36 deaths, **6 losses**; Lucian 7.
Still −30 points and 6 vs 2 Cynthia losses on the same seeds, with only
route and gym choices made: what the fast brain prefers for its own play
at a route fight (a fourth attack over Dragon Dance / Swords Dance, Leaf
Storm over a repeatable hit) is carried into the League, where search
play needed the dropped move. Two tables, both negative, one mechanism:

### the calc bot values moves for the calc bot, not for search over it

,
the same transfer failure as E8 and the EV probe. `--moves sim` stays off
(kept, ~40 lines, as the measured option); best4 v2 + TMs remains the
benchmark (70%). What would be different, not tried: scoring variants by
the search brain itself (deep 9,16 through the block: ~40 min per opener
at 70 variants — the compute dial, not a rule), or re-picking per fight
without persistence (loses the League slot-for-slot anyway). The lever
left for Cynthia is E9 held items.

### E9, held items (2026-09-26, `Rules.items`, `--no-items`)
Data:
Bulbapedia sits behind a Cloudflare challenge since today (`build_order.fetch`
cached the challenge page; deleted), so `scripts/build_items.py` reads
Serebii's BDSP item list instead — one row per item with its fixed locations,
then Pickup / shop / wild-holder lines, which are dropped. An item counts
from its earliest fixed pre-League source, one copy per source; not counted:
the Grand Underground, Battle Park prizes (all Choice items but Specs, Life
Orb, Muscle Band, Scope Lens), shops, wild holders, post-game areas, and
consumables (Focus Sash, berries, herbs — a battle rebuilt from packed teams
would hand them back every fight). 40 items in `data/bdsp/items.json`:
Quick Claw #21, Sea Incense #23 (Water: the starter's first item), Miracle
Seed #29, plates and type boosters along the route, Shell Bell #99, Choice
Specs / Wise Glasses / Black Glasses at Celestic #235, Expert Belt #271,
Leftovers #381 (Victory Road), Sky Plate #397. Rule (`gauntlet.assign_items`,
at every fielding, deterministic in party x progress so the risk cache needs
no item key): item_value = the multiplier on the mon's attacks of the boosted
type (1.2 STAB, 1.1 off-type) or category (Choice x0.85 for the lock), or a
flat value (Leftovers 1.25 so it is always held, Expert Belt 1.1, Metronome
1.1, Shell Bell 1.08, Quick Claw / Wide Lens / Razor Claw 1.05); greedy over
(value, level), one item per mon, `copies` holders per item. At #404 a
Gyarados / Luxray / Staraptor / Torterra / Lapras / Golduck party holds
Splash Plate / Zap Plate / Sky Plate / Soft Sand / Wave Incense / Choice
Specs (+ Leftovers on the first by sort). Table, same seeds and config as
the TM line (`results/plan_tms_items.log`, 31 min):
  **404.80 / 402.8 / 4.30 / 80% clears**; Cynthia 36 deaths, 2 losses; Lucian 5,
  Bertha 1, Cedric 1 (TM line 404.6 / 358.6 / 5.2 / 70%; Cynthia 34 / 2; E4 12).
One death before the League in ten runs (was ~10), the E4 halved (12 → 6),
Cynthia flat (34 → 36 deaths, 2 losses both). Clear rate +10 is inside the
±15 floor; deaths 5.2 → 4.3 and no-death 358.6 → 402.8 are the gain, and the
rule is honest (found items only, one copy each). Items are on by default;

### this is the benchmark table now
Cynthia costs 3.6 deaths per run with
every pre-League lever on the env side in (levels, movesets, TMs, items, the
box); what is left for her is play (search width at the block, the enemy
model) or the run layer bringing a different box.

### The League test (2026-10-09, `scripts/league_test.py`)
A 20-episode
replicate of the benchmark on unchanged code (seeds 300-319, `--save-league
results/league_snapshots`, `results/plan_league_snapshots.log`, 63 min):
  404.55 / 382.1 / 3.85 / **55% clears**; Cynthia 59 deaths, **9 losses**; Lucian 7.
The 10-episode table said 80% / 2 losses. 25 points between identical configs
is past the ±15 floor quoted above: every League verdict since 2026-09-19
that rests on clear rate alone (E8, EVs, E7(d), E9's +10) is inside noise.
`plan_gauntlet.py --save-league DIR` pickles the run state at Aaron (before
the League party is chosen); `league_test.py` replays the block from every
opener R times with the planner's flags and pairs variants by box (`--vs`).
~4 min per block vs ~28 min per full episode. Baseline, 20 boxes x 4 replays
(`results/league_base.{log,jsonl}`, 32 min):
  **block clear 0.70 ± 0.07 / 3.69 ± 0.25 deaths per block / 3.12 ± 0.20 at
  Cynthia**; Aaron, Bertha, Flint, Lucian 80/80 won, Cynthia 56/80.
(± = standard error over boxes.) Per box: 7 boxes clear 4/4, 6 clear 3/4,
4 clear 2/4, 2 clear 1/4, 1 clears 0/4 — most of the spread is the dice, not
the box. The E4 never loses; the whole League problem is Cynthia's 3.1 deaths.
A lever needs about +0.15 block clears or -0.5 deaths to show at this size;
smaller ones need more replays. Every League lever is tested here first now.

### Lead order is not a lever (probe, 2 seeds at Edwin)
The packed order
is the party's roster order, so the lead is whichever mon was caught
first. Rotating each of the six into the lead: under the fast prior the
n=32 estimate swings from 0.6 to 4.4 deaths per battle (Beautifly lead vs
Grotle lead), but the deep search brain plays every lead to 1.1–1.9
deaths, inside the 16-play-out noise, because it switches on turn one
when it has to. Consequence: nothing to gain in the fight; but the party
*score* the planner ranks by is the fast prior's, and it depends on the
accidental lead. A cheap cleanup for later: score a candidate with its
best-by-prior lead, or make the query lead-invariant. Note the mismatch: the risk query is the fast
prior, the fight is played by search; n=32 estimates and deep play-outs
disagree by up to 0.5 deaths. `plan_gauntlet.py --parties beam|all` (replaces
`--climb`/`--refine`, both null) widens the candidate set only when the best
swap is still flagged hard. Tables:
  beam (5 workers, 68 min, `results/plan_cap186v3_deep20_beam.log`):
    178.15 cleared / 58.9 no-death / 13.75 deaths / 35% clears, 17.1 deep fights
    (baseline 178.6 / 62.4 / 13.4 / 40%; climb 178.55 / 70.7 / 13.4 / 40%)
  all (10 workers, 106 min, `results/plan_cap186v3_deep20_all.log`):
    178.95 cleared / 79.8 no-death / 13.05 deaths / 30% clears, 16.4 deep fights
Both are null on cleared, deaths and clear rate. `all`'s no-death mean is
+17, but that metric is the first-death trainer and jumps between a few
values (32 Hiker Daniel, 50 Roark, 84, 134, 139), so three runs surviving
Daniel move it by 15 on their own; not evidence at 20 episodes. Beam is
null, and both fail the same way climb did: Edwin 42 → 22 deaths,
Fisher Juan 18 → 36 with 6 losses. A better party at Edwin keeps mons
alive that then die to the Gyarados, where no party in the box helps
(probe above). Deaths are conserved across the walls until the box holds
an answer to each — which the 12-slot box drops (next entry).

## Self-play PPO (no_AI, 2026-08-27): gen9randombattle, 20M steps / 12

vec envs / ~8.7h / 636 fps. Entropy 1.77→0.31 (collapsed), explained
variance -0.00→0.53, approx_kl ~0.006. Eval 100 battles: Random 97%,
MaxBasePower 72%, SimpleHeuristics 33%. TB
`logs/tensorboard/self_play_20260827_103605`. Weights not saved —
`self_play.py` never calls `ppo.save()`; only in-memory eval after learn().

## Full enemy information for the battler (2026-10-10, `--full-info`)

The player's mirror knows every enemy set from turn one (moves, item,
ability, level, real stats from the sheet's EVs / IVs / nature;
`SimBattle(full_info=True)`, forks keep it, the enemy side stays as it was),
and the calc bot uses a real stat and real max HP whenever it has one instead
of the 31 IV / 85 EV guess and the 60-power STAB guess for unrevealed moves.
This is the full-docs regime of `project_overview.md` → 2, which the battler
had never used. Default off; flag on the planner and the League test.

Cynthia probe (`models/cynthia66_pairs.json`, the old level-up-moveset
parties, n=2): calc bot alone 2.5% / 0.99 deaths per mon blind and 2.5% /
0.98 with full info (40 parties); calc + search k9 r8 20.0% / 0.91 both ways
(20 parties). Nothing moves on parties that have no line.

League test, 20 boxes x 4 replays, calc + search 9,8 / deep 9,16, paired
against `results/league_base.jsonl` (`results/league_fullinfo.{log,jsonl}`,
42 min):
  block clear 0.75 ± 0.05 / 3.64 ± 0.22 deaths per block / 3.09 ± 0.20 at Cynthia
  Aaron, Bertha, Flint 80/80, Lucian 79/80, Cynthia 60/79
  vs baseline: +0.05 ± 0.08 block clears, -0.05 ± 0.34 deaths per block (paired)
**Null at this size** (a lever needs about +0.15 clears here; the sign is
right, the error bar covers zero). Reading: the calc bot's rules spend the
information on the KO check and the threat estimate only, and at Cynthia the
problem is stat parity (her 31 / 252 spreads against flat 85s), not what the
player knows — the 2026-09-19 probe already said no deathless line exists for
a typical party. The cleaned-tree replay the same day
(`results/league_cleanup.log`, +0.00 ± 0.07) is the control for the harness.
What the information would still buy is a scorer that uses it: the party
selector still ranks by the fast bot's blind estimate. Not retried: more
replays to resolve +0.05 (160 blocks for ± 0.04; ~85 min).

## Selector transfer diagnostic (2026-10-10, `scripts/selector_probe.py`)

Question: does the fast-brain party score (the block estimate under the calc
bot, n=8: what the planner ranks League candidates by) order parties the way
search play does? Method: for each of the 20 saved openers, rank the planner's
whole candidate set (base + single swaps, ~200 parties) by its own score, then
play the candidates at ranks 0, 1, 2 and at 1/8, 1/4, 1/2, 3/4 and the end of
the ranking through the whole block with the benchmark search settings
(calc + search 9,8 / deep 9,16), once each: 160 blocks
(`results/selector_probe.{log,jsonl}`; the first run crashed on a probe bug
after 101 plays — the next candidate's estimate was read from the played-out
roster — and the 12 affected boxes were rerun with the fix).

  within-box Spearman(fast score, search-play deaths): -0.31 ± 0.08 over 20 boxes (pooled -0.29)
  slot 0 (rank 0, what the planner fields):  3.65 ± 0.47 deaths, 70% clears
  slot 1 (rank 1):                           3.65 ± 0.42, 75%
  slot 2 (rank 2):                           3.45 ± 0.44, 75%
  ranks ~24:                                 3.95 ± 0.47, 60%
  ranks ~48:                                 4.50 ± 0.39, 50%
  ranks ~96:                                 4.60 ± 0.43, 45%
  ranks ~144:                                4.75 ± 0.37, 45%
  ranks ~190 (the end):                      5.25 ± 0.31, 30%
  best sampled party: median rank 2; the fast winner was (tied) best in 4/20 boxes;
  min-of-8 "regret" 2.25 ± 0.41 deaths (inflated by selecting the luckiest of 8 single plays)

Reading: the fast score is a usable coarse filter (the bottom of the ranking
costs 1.6 deaths more than the top, and the correlation has the right sign)
but it cannot separate the top of the ranking — ranks 0, 1 and 2 are tied
within noise and rank ~24 is within a death. The planner's pick is a coin flip
among the top few dozen candidates. This is the mechanism behind the three
transfer failures (E8, EVs, E7d): every one of them optimised the fast score
deeper into a region where the fast score has no resolution. Consequence for
the selector: keep the fast score as the shortlist (it is cheap and it does
prune), and decide the final pick among the top K with the search brain
itself (K x R block plays at the opener: K=4, R=2 ≈ 8 blocks ≈ 15 min per
episode at the League — affordable on the League test, 2 h for 80 blocks,
not for a full-game table). Not built yet. A fairer regret would replay the
top-3 more times; the slot table is the unbiased view.

## Radical Red boss probe (2026-10-10, `scripts/rr_probe.py`, `results/rr_probe.{log,jsonl}`)

The question behind the project: can the solver beat a Radical Red hardcore
Nuzlocke? Step one: can the battler win the boss fights at the level cap?
Stack: the RadicalRedShowdown fork (`pokemon-showdown-rr/`, commit 5495579,
mod `gen9rr`, custom format `gen9rrcustomgamenogimmicks` = any team, no
Dynamax / Tera, Megas and Z-moves allowed) through the bridge (`SHOWDOWN_DIR`;
the channel now goes into the Battle constructor, since this older build
emits the opening requests only at start); `data/rr/` from the tracker's
docs export (encounters per location and method, hardcore trainer order with
caps, boss effects) and the ROM dump (sets with canonical names; EVs assumed
252/252/4 from the nature — the hardcore docs carry none); the mod's own
moves / species injected into poke-env's gen 9 tables; the CFRU AI port
(`ai_util/rr_ai.py`, back from the pre-cleanup tag) as the enemy, with megas.
Found on the way: a forme next to its base form collapsed in the mirror
(`dedupe_nicknames` by base species now); gen 9 species packed as raw ids
(`species_name` on the gen 9 dex, nickname = base species).

Method per boss: 30 boxes of 10 species drawn uniformly from the encounter
pool available before the boss (124 species by Brock, 219 by Misty, 272 by
Lt. Surge, 360 by Erika, 449 by Koga, 545 by Blaine), evolved by level at the
cap, the six with the highest base-stat total fielded at the cap with best4
movesets (level-up only / every TM and tutor move = upper bound), flat 85 EVs,
no items, ability slot 0 or 1, full enemy information; n=2 battles per box;
calc bot alone and calc + search k9 r8. Not modelled, all in the player's
favour: the gyms' permanent field effects (Brock's sandstorm and -33% on
super-effective hits, Misty's rain, Surge's terrain, Erika's terrain and
grass-move rules, Koga's tailwind, Blaine's Desolate Land), the real EV
spreads. In the boss's favour: no held items, no EV training, no natures, and
a random box instead of a planned one (Radical Red lets the player pick
encounters with DexNav).

  boss       cap  calc alone (lv-up / +TMs)   calc+search (lv-up / +TMs)   deaths/mon (search, +TMs)
  Brock       16      40% / 73%                   87% / 93%                 0.29
  Misty       28       2% /  7%                   17% / 38%                 0.85
  Lt. Surge   36       7% /  5%                   27% / 18%                 0.93
  Erika       44       3% /  2%                   17% / 18%                 0.92
  Koga        68       0% /  0%                    2% /  2%                 0.99
  Blaine      76       5% /  3%                    8% /  8%                 0.97
  (70 min on 12 workers; the RR AI costs 8 ms per decision, 10x SimpleHeuristics)

Verdict for step one: the battler clears Brock and nothing after it. From
Misty on a random box at the cap loses 60-98% of fights even with search and
every TM, and loses nearly the whole party when it loses. Search is worth 3-5x
over the calc bot alone at the early gyms and nothing at Koga / Blaine, where
the fights are decided before a decision matters. The criterion set before the
run (under 50% per boss = no until the battler or the team changes) is met
everywhere past Brock. What this does not settle: how much a planned box
(DexNav picks, items, EV training) recovers — a "best party in the pool" probe
(search over which six to bring) is the step-two measurement, and the field
effects would push the other way.

## Radical Red best-six probe (2026-10-10, `scripts/rr_team_probe.py`, `results/rr_team_probe_{v1,all,legal}.{log,jsonl}`)

Step two of the Radical Red question: does *any* team from the encounters
available before a hardcore boss win the fight reliably under search play?
Per boss: (A) every species in the pool (the tracker's encounter tables for
every non-postgame location up to the boss's, all methods; 124 species by
Brock … 545 by Blaine), evolved by level at the cap, best4 moves over every
TM, ability slot 0, fights the boss solo under the calc bot (n=3) — progress
(enemy mons fainted + damage on the rest, +survivors on a win) is its score;
(B) the 24 best are combined, greedy plus 150 random six-subsets, each team
scored by the calc bot (n=4); (C) the six best teams are played with calc +
search k9 r8, n=6, and the best is reported. Enemy = the RR AI port with full
information both ways, ability immunities in both bots (fixes from the
afternoon: Dry Skin Jynx had swept Misty).

Three runs. **v1** (61 min, 6 workers) gave the enemy the 252/252/4 EV guess
and the player four variants (Eviolite/Life Orb or none × flat 85 EVs or
252/252). Then the official hardcore sheet (`data/rr/README.md`) showed that
hardcore is Minimal Grinding Mode — **nobody has EVs**, items are single late
copies, and the player loses 82 moves — so **all** (21 min) and **legal**
(25 min) rerun it with EVs 0 on both sides and no items, without and with the
82-move ban in every stage.

  boss       cap  v1 no items / flat 85   all moves (no EVs)   legal moves (no EVs)   top-6 spread (legal)
  Brock       16  100%  0.00              100%  0.00           100%  0.03             100 100 100 100 100 83
  Misty       28  100%  0.36              100%  0.03           100%  0.17             100 100 100 100 83 83
  Lt. Surge   36  100%  0.53              100%  0.28           100%  0.25             100 100 100 100 100 67
  Erika       44  100%  0.28              100%  0.36           100%  0.19             100 100 100 100 100 100
  Koga        68  100%  0.61              100%  0.28           100%  0.14             100 100 100 100 100 100
  Blaine      76  100%  0.53               83%  0.67           100%  0.47             100 100 83 67 33 17
  (win rate of the best team under search, deaths per mon; v1's other three
  variants are in the log: all 100% except Blaine with items 83%)

Verdict: **a planned team beats every probed gym under search**, Blaine
least convincingly. With the real EVs (0) Koga goes from "one team at 100%,
the rest 33-83%" to all six at 100% with 0.14 deaths per mon; the move ban
costs nothing measurable (legal ≥ all at every boss but Brock's noise), because
the calc bot never set up anyway — the ban bites the enemy's sweepers' model,
not this player. Blaine's spread (100 → 17% across six teams, both runs) is
the one place the result is a winner's-curse pick at n=6; everywhere else the
top six agree. Deaths per mon of 0.1-0.5 in a won fight are still far from
Nuzlocke-clean.

What this still leaves out, all against the player: the gyms' permanent
arena effects (sandstorm, rain, terrains, Koga's Tailwind, Blaine's Desolate
Land), the player's ability swaps (Drizzle → Adaptability etc.), natures, and
the pool being the whole reachable dex rather than a run's box (no rarity, no
one-per-route, no HM gates, no starter). In the player's favour: nothing left
now — items and EVs are gone. So the open question moved: not "can the battler
win a boss fight" (yes, given the team) but "can the run layer assemble such
a team from real encounters, and survive the arena effects and the doubles
fights". `data/rr/README.md` lists the data and gaps in order.

## Radical Red: arenas, real pools, real TMs (2026-10-11, `results/rr_team_probe_{arena_all,arena_legal,real,real_noarena}.*`, `results/rr_postmortem_misty.log`)

Two corrections to the best-six probe, each measured on its own.

**Arena effects** (`sim_bridge.js` arena: the gym's permanent weather / terrain /
Tailwind set once the leads are in, no duration, re-asserted every turn; p2's
own weather and terrain frozen too — the hardcore rule). Same inflated pool
and every TM as the evening runs:

  boss       no arena (all / legal)   arena (all / legal)   effect
  Brock        100 / 100                100 / 100            sandstorm
  Misty        100 / 100                100 /  83            rain          (legal: 0.72 deaths/mon)
  Lt. Surge    100 / 100                100 / 100            Electric Terrain (spread widens to 50)
  Erika        100 / 100                100 / 100            Grassy Terrain
  Koga         100 / 100                 17 /  50            Tailwind      (five other teams 0%)
  Blaine        83 / 100                100 / 100            Desolate Land (spread 100 → 0 / 17)

Koga's Tailwind alone turns six teams at 100% into one coin flip; everything
else the pool absorbs by picking around it (Blaine's teams become Fire-immune
or -resistant: Salazzle, Galarian Ponyta, Diglett).

**Real pools and real TMs.** Printing the Brock and Misty teams showed the
probe's pool held Super Rod and Surf encounters at level 53-56 (Clawitzer,
Luvdisc, Tentacruel under Pallet Town) and its movesets held Surf, Ice Beam,
Earthquake, Sludge Wave (post-game) and Victory Road TMs at cap 16. Now
`build_rr.progression()` writes the hardcore trainer order with the last
encounter location open at each fight and the rods / Surf as trainer indices
(Old Rod #0, Good Rod #21 Vermilion, Super Rod #74, Surf #81), and every TM
and hardcore tutor move with the trainer index at which it is in hand (boss
rewards after the boss, places when open, HM-locked spots with the HM, the
grey tutors the sheet marks gone in hardcore skipped; money not modelled).
`rr_probe.pools()` keeps a species only if some open location offers it by a
method in hand at a level at or under the cap. Brock 124 → 105 species and 6
TM/tutor moves in hand, Misty 219 → 191 and 22, Surge 272 → 251 and 35, Erika
360 → 338 and 72. Scraggy at Brock now carries Thief / Low Sweep / Headbutt /
Feint Attack; Arbok at Misty Poison Jab / Crunch / Dig / Sucker Punch.
Early game only (the user's call: no point past the blender), legal moves,
no EVs, no items, n=12 per team under search:

  boss       cap  pool  no arena            arena               top-6 spread (arena)
  Brock       16   105   100%  0.07           100%  0.32          100 100 92 83 83 83
  Misty       28   191   100%  0.36           58%  0.67          58 50 42 42 33 25
  Lt. Surge   36   251   100%  0.51           75%  0.69          75 75 75 58 50 42
  Erika       44   338   100%  0.10           92%  0.36          (one team)
  (43 / 64 min on 4 workers; the no-arena run has one team at 100% at Misty and
  Surge, the rest 67-92 / 75-83; Brock without sandstorm: four teams at 100%)

**Verdict.** With what a run actually has, the bot beats Brock, loses to Misty
more often than not (58% best, 0.67 deaths per mon) and gets through Surge
three times in four. The blender is Misty. Rain Misty is the wall, and it is a
wall of two mons: the post-mortem (`scripts/rr_postmortem.py`, 12 replays of
the best team) attributes 22 of 55 deaths to Mantine (Swift Swim, Scald /
Hurricane / Roost) and 14 to Toxicroak (Dry Skin, Swords Dance); Politoed and
Mantine are the ones left standing in 5 of 12 fights. Our Electric types
(Boltund, Electabuzz) die 18 times for 13 KOs — Thunder Punch Electabuzz is
not a Mantine answer at level 28 under rain.

**What the bot needs to advance** (the question this run was for):
1. A threat → counter step on the box: name the enemy mons that cause the
   deaths and search the box for what walls or outspeeds them (here: a bulky
   Electric or Grass special attacker for Mantine, a Psychic or Flying hit
   that OHKOs Toxicroak before Swords Dance). The post-mortem is that step's
   input; the planner's `risk()` over swaps is its output. Encounter routing
   then follows: which route adds the counter.
2. Set-up moves. The ban cost nothing measurable because the calc bot clicks
   a boosting move only under a narrow safety rule (`calc_bot.py`: enemy threat
   under 20% and own HP over 60%), which the RR AI rarely allows, and because
   search can see a two-turn plan only if the rollout brain plays it. The
   enemy's set-up turns (Toxicroak's Swords Dance) are not read as a threat
   either. A boost-aware value in the rollout brain (expected damage over the
   remaining fight, not this turn) is what lets search use and punish set-up.
   This is the battler's ceiling, not the planner's.
3. Starter and gifts (HUMAN_TODO 8): the pool has no starter; a Bulbasaur
   line changes Misty.
Doubles, natures and the player's ability swaps are still out.

## Radical Red: the chooser, the field, and the first milestone number (2026-10-11, `results/rr_bench_{field,blind}.*`, `results/rr_c1_report.*`, `results/rr_boxes30.*`)

The plan (`~/.claude/plans/flickering-painting-valley.md`, approved 2026-10-11)
in the order it was run. Everything below: legal moves, arenas, no EVs, no
items, real pools and real TMs, the RR AI with full information both ways.

**Step 0, measurement.** `rr_team_probe.play()` now seeds every battle
(one engine seed per battle index, shared by every team and variant) and the
searcher's rollouts. Found afterwards: the RR AI port breaks ties and picks
switches with Python's global `random`, so the runs below were *not* fully
paired across processes (two "identical" stage-A passes ranked species
differently). Fixed after these runs (the global random is seeded per battle;
two passes now agree exactly); the field-vs-blind table is therefore two
independent samples, not a paired difference.

**C0, the calc bot sees the field** (`calc_bot.field_mod`, `stat(..., battle)`:
rain / sun on Water / Fire, the primal weathers, sure-hit Hurricane / Thunder /
Blizzard, sandstorm SpD, snow Def, the three terrains x1.3 for grounded users,
Grassy halving Earthquake, Misty halving Dragon, the weather speed abilities,
Surge Surfer, Tailwind). n=30 per team, best of six teams, 125 / 136 min on
4 workers:

  boss       cap  field-aware calc bot   blind calc bot      (win, deaths/mon)
  Brock       16   100%  0.39            100%  0.38
  Misty       28    80%  0.60             43%  0.88
  Lt. Surge   36    67%  0.69             90%  0.66
  Erika       44   100%  0.23            100%  0.21

Misty moves 37 points — the bot finally knows Scald is x1.5 and Mantine is
fast. Surge moves the other way by 23, with different teams chosen in the two
runs (the chooser ranks with the same calc bot, so a changed bot changes the
box it builds); at n=30 one team is ±9 points and the best-of-six adds a
winner's curse. The honest reading: a large gain where the arena is weather,
noise where it is terrain, and the comparison must be rerun under the exact
pairing that now exists.

**C1, threats → counters → where.** `rr_postmortem.attribution()` names the
enemy mon on the field at each of our deaths; the probe reports the best
team's two top killers and, for each, the pool species that beat it with
`rp.where()` (first open location / method / rarity). At n=30 (field-aware):
Misty 63% / 0.72 with the chooser picking Bramblin (Wind Rider: immune to
Hurricane) on its own — Mantine stops being the killer and **Ludicolo** takes
74 of the deaths; Surge 87% / 0.60, killers **Electrode-Hisui** (30) and
**Pawmot** (22). The first counter criterion (KOs minus deaths during a solo
fight against the whole team) found "none in the pool" for every mid-team
threat; counters are now ranked by **one-on-one duels** against the threat's
real set under the arena (`duel()`, `--n-duel`): Pawmot → Dedenne / Stufful /
Phanpy, Electrode-Hisui → Wimpod / Tarountula / Nincada, each with its route.
Forcing the top counter into the greedy build was a mistake (Misty 63% → 27%
with Rattata forced in); the repair is now unforced — counters join the
candidates and must earn a slot.

**C2, choices.** A random starter across every generation per run; gifts and
in-game trades in the pool with their trainer index (`order.json["gifts"]`:
Route 4 Magikarp, daycare Eevee, the Cerulean Snom → Carnivine trade, …);
natures by the set (mints), abilities by the boss (capsules / patch:
immunities to the boss's attack types, the gym weather's speed ability,
Intimidate). What the chooser then fields against rain Misty reads right:
Parasect (Dry Skin), Carnivine (Levitate), Drednaw (Swift Swim in her rain),
Gyarados / Arbok / Granbull / Mightyena (Intimidate), mostly Adamant.

**C3, the milestone number** (`--boxes 30`, `rp.sample_box`: one encounter per
open location drawn by rarity, a random starter, the gifts; the chooser picks
six per box; n=6 under search per box; 111 min on 3 workers):

  boss       cap  mean win  boxes with a >=50% team  deaths/mon  most-missed counter (how many boxes)
  Brock       16    70%            83%                 0.67     Sentret vs Lunatone (10), Seedot vs Archen (9)
  Misty       28    44%            53%                 0.81     Bramblin vs Mantine (24), Rattata vs Toxicroak (10)
  Lt. Surge   36    49%            53%                 0.80     Ekans vs Raichu-Alola (16), Yanma vs Pawmot (14)

Boxes hold 9 species by Brock and 18-19 by Misty. Brock's box results:
13 boxes at 83%, 5 at 100%, 1 at 0%. Misty's 16 winning boxes (≥50%) share a
core — Drednaw in 12, Parasect and Carnivine in 7, Arbok 6, Gyarados 5 — and
still lose half a party in a win. (These box counters predate the duel change.)

**P1, fork.** `SimBattle.fork()` pickle-copies the mirrors instead of
replaying the history: 1.6 → 5.8 ms (turn 5 → 30) became ~2 ms flat, read-
identical. A search decision (k9 r8) is ~1.8-2.2 s under load and is now
rollout steps, not forks.

**Deathless.** No team won without a death across its battles; the closest
was Brock without the sandstorm at 0.07 deaths per mon. The probe now reports
the deathless-win rate per team and box ("clean").

**Where this leaves the question.** From a real, rarity-sampled box the
chooser beats Brock most of the time, and Misty and Surge about half the
time, with heavy losses; from the best planned team, Misty 80% and Surge
67-90% (bot-dependent). The next levers, in order: the paired field-vs-blind
rerun; duel-based counters in the box report (what to route for); the
boost-aware threat (C4); doubles (C6) before anything past Erika.

## Why not deathless: deaths, matchups, the cover chooser (2026-10-11/12, `results/rr_deaths_*`, `results/rr_sameteam_*`, `results/rr_bench2_*`, `results/rr_boss1_*`)

The user's question: a deathless Brock from real boxes with perfect
information should be possible — if the approach cannot, rethink it.

**Boss #1 Brendan is solved**: planned team 30/30 deathless; 30 sampled boxes
100% wins, 87% deathless, 0.03 deaths per mon (`rr_boss1_*`). Every entry up
to Misty now resolves to a dump set (`rr_probe.boss_team` by name or `#index`;
`parse_dump` keeps "Max Level − n" mons; four of the first nineteen are
doubles, not playable yet).

**Deaths classified** (`scripts/rr_deaths.py`, 12 seeded fights of the best
planned team, search k9 r8, arena, legal): Brock 11/12 wins, 0 deathless, 34
deaths — all 34 below half HP, Lunatone 12 / Cacnea 10 / Varoom's Toxic 4;
Misty 8/12, 1 deathless, 47 deaths — 37 below half, Ludicolo 17 / Toxicroak
12 / Mantine 11, 31 after turn 20. Attrition, not surprise: mons stay in
until low and die, because the search values a death at 0.5 × 1/6 ≈ 0.08
against a win at 1 (death weight 2.0: identical 34 deaths — every rollout is
finished by a calc bot that never retreats).

**The trace** (`scripts/rr_trace.py`, every candidate's rollout value per
turn, Brock seed 0): a switch dance into Varoom's Protect / Toxic (turns 1-7),
Gumshoos at 27% poisoned kept in for Rock Tomb (+0.82 vs switch −0.36) and
dying to poison, two mons switched into a +3 Cacnea, twenty turns of Metal
Claw into Stockpile + Recover with every candidate at +0.75.

**Matchups with perfect information** (`ai_util/matchup.py`, `scripts/
rr_matchups.py`): hits to KO each way from the calc bot's model on the real
sets under the field. Brock's pool has 15-57 safe answers per mon and four
species (Axew, Bibarel, Cubchoo, Kadabra) cover all six; the probe's best
team had one answer to Cacnea and one to Lileep. The worst Brock box (0%)
held Chimchar — its only answer to Cacnea and Varoom — and the greedy chooser
left it in the box; the best box left Raboot, Spidops and Alolan Grimer out.
The first "safe" rule (KO in ≤2, survive ≥3) was wrong twice: it ignored speed
order and recovery (Gyarados "2/3" vs Roost Mantine was no answer) and it
threw away one-shots that absorb one hit (Zangoose, Boltund, Luxio vs
Mantine). Now: `safe` charges the hits absorbed before our last one with a
hit to spare, healers must fall before they heal; `race` is the looser tier
(we land the KO first); `cover()` fills safe answers first, race-winners
second. Misty's pool under the strict rule: Mantine 1 answer (Pachirisu),
Starmie 1 (Parasect), Ludicolo 2 (Parasect, Leavanny) — the routing answer in
its sharpest form; under `race`, Zangoose / Boltund / Luxio / Hariyama also
answer Mantine.

**Levers on the same teams and seeds** (12 fights):

  Misty                       wins  deathless  deaths      Brock                 wins  deathless  deaths
  baseline (dw 0.5)            8       1        47         baseline              11      0        34
  preserve rule                9       5        28         death weight 2.0      11      0        34
  matchup switch + dw 6       12       5        17         preserve              11      1        32
                                                           matchup switch        11      1        40
                                                           matchup + dw 6        11      1        30

Retreating low / poisoned mons and pricing a death like a loss turned Misty
from 8/12 into 12/12 with a third of the deaths; at Brock the team has no
Lunatone answer, so no policy fixes it — the chooser must. Matchup switching
without a target made Brock worse (40): switching is only good into a
race-winner, which is what the rule now requires.

**The field model** (`calc_bot.FIELD_AWARE`), exactly paired at n=30 but with
each bot choosing its own team: Brock 100/100 (clean 23 vs 43%), Misty 67 vs
57, Surge 77 vs 93, Erika 93 vs 100. Same team, same seeds, 12 fights:
field-aware 22 vs blind 7 deaths at Brock, 25 vs 13 at Erika, 37 vs 44 at
Misty, 58 vs 54 at Surge. The numbers the model changes are all correct
(Sand Rush Cacnea 20 → 40 speed, −33% special on Rock in sand, +30% Grass in
Grassy Terrain); the damage comes from the old switch logic turning a more
pessimistic picture into switch-in deaths. A precedence bug meant the rain
accuracy override never applied (fixed). Under the matchup rule (same team,
same seeds, death weight 6; `results/rr_sameteam_*_mu.log`) the picture
splits: Erika field-aware 12/12, 12 deaths vs blind 10/12, 24 deaths (3
deathless each); Brock field-aware 11/12, 37 deaths, 0 deathless vs blind
12/12, 20 deaths, 2 deathless. At Erika the model now pays off (Grassy
Terrain Grass damage read right, and switching only into race-winners); at
Brock the field model still doubles the deaths, all of them below half HP and
11 of 37 the turn the mon switched in, with Cacnea (19) and the three Rock
mons it under-rates us against under sand (Lunatone, Lileep, Archen: 12)
doing the killing. Brock's sand is the case where a correct, more pessimistic
model makes the search retreat into a bench that has no answer; the verdict
on field-aware as the default waits for the cover-chooser runs, which are the
first with a bench built to have one.

**Also**: 23 of 27 starters were missing from the mod tables (the extraction
took encounter and boss species only) — the box runs fielded a starter in 2
of 90 boxes; fixed, every run before c1a2e57 is starterless. The RR AI port
breaks ties with the global `random`: seeded per battle since 7a6675b. The
cover + matchup + death-weight-6 runs on Brock and Misty (pool n=30, 30
boxes) are the next numbers; `scripts/rr_status.py` and `UI_TODO.md` are the
run overview and its web successor.

**Cover chooser, Misty, planned pool** (n=30, cover + matchup + death weight
6, `results/rr_cover_pool.log`): **30/30 wins, 0.23 deaths per mon, 30%
deathless** with Grimer / Gyarados / Hariyama / Parasect / Toucannon / Crobat
(the cover of the matrix: Parasect for Starmie and Ludicolo, Gyarados and
Crobat race Mantine, Hariyama the race-winner tier) against 60 / 53 / 50 / 43
/ 20% for the greedy and fast-stage teams on the same seeds. The old planned
Misty was 67–80% with 0.4–0.5 deaths per mon. The deaths that remain are
Starmie (11) and Clodsire (8), both with duel-winning counters in the pool
(Ekans, Paras, Morelull; Kricketot, Mudbray, Smoochum) that the cover did not
take because something else already answered them on paper. **Brock** under
the same run: the best six is again the fast-stage team (Dewott / Watchog /
Alolan Raticate / Nuzleaf / Leavanny / Linoone), 30/30, 0.21 deaths per mon,
23% deathless — the same numbers as before the chooser, with the other five
candidates at 100 / 97 / 97 / 90 / 57%; the cover six (Chingling, Patrat,
Poochyena, Alolan Rattata, Tarountula, Yungoos) is one of those five (the log
does not say which: per-team lines are now printed, UI_TODO item 1). Cacnea
(16) and Archen (6) are the killers; their duel-winning counters (Galarian
Ponyta, Kricketot, Cubchoo; Shinx, Kricketot, Poochyena) are in the pool. So
the cover chooser decides Misty and leaves Brock where it was: Brock's
problem is the play against Cacnea under sand, not the six.

**Cover chooser, sampled boxes** (30 boxes per boss, a random starter plus one
encounter per open location drawn by rarity, n=6 per box; same play config;
`results/rr_cover_boxes.log`):

  boss    cap  mean win  boxes >=50%  deaths/mon  deathless  most missed counter
  Brock    16     66%        80%        0.63         3%      Galarian Ponyta vs Cacnea x16, Alolan Sandshrew vs Lileep x15
  Misty    28     58%        67%        0.68         5%      Bramblin vs Mantine x20, Ekans vs Starmie x14

Against the starterless greedy boxes (Brock 70 / 83 / 0.67, Misty 44 / 53 /
0.81) Misty's box number moved up 14 points and Brock's did not move; the
deathless rate per box stays near zero on both, because a rarity-drawn box
mostly lacks the counters the cover wants (the "most missed" column is what
routing would have to go and catch: at Misty a Mantine answer was absent from
20 of 30 boxes). That is the gap between the planned pool (30/30) and a run.

**The other early bosses** (planned pool, n=30, same config;
`results/rr_cover_early*.log`):

  #   trainer            cap  win   deaths/mon  deathless  killers (counters in the pool)
  2   Lass Anne           16  30/30    0.00       100%
  4   Catcher Sammy       16  30/30    0.00       100%
  5   Falkner             16  30/30    0.00       100%
  7   Lass Sally          23  30/30    0.00       100%
  9   Archer (Mt. Moon)   23  30/30    0.06        73%     Seviper 5, Mightyena 2 (Wattrel, Spearow; Wurmple, Tyrogue)
  10  Rival (Cerulean)    28  30/30    0.01        93%     Clefable 1, Arbok 1 (Ponyta, Tentacool; Riolu, Rockruff)
  15  Camper Ethan        28  30/30    0.00       100%
  16  Grunt (Nugget Br.)  28  30/30    0.04        80%     Druddigon 3, Arcanine 3 (Nacli, Lunatone; Mienfoo, Tentacool)

Every candidate six won every fight at all eight; the singles fights between
Brendan and Misty are not where runs die. Brock (23% deathless) and Misty
(30%) remain the two blenders, and both now have the shape "100% wins, a
handful of deaths to one or two named mons whose duel-winning counters sit
in the pool". Two data fixes on the way (986dda0, 31edf3f): the ROM dump's
12-character names ("Drain Kiss", "None" items) only resolved for the twelve
hand-fixed bosses, and the enemy AI crashed on a Metronome-called Hyper
Beam's recharge turn. #14 Lass Relli has no team in either source; #8 and
#11–13 are doubles.

## Full circle: the matrix was going to be a model (2026-10-11, evening, reflection)

The user's original idea for the piece that reads an enemy line-up and finds
the answers in the box was a learned one — an AI, a model trained on fights.
What ended up carrying the evening's result is `ai_util/matchup.py`: a
hits-to-KO table read off the calc bot's damage model, two hand-written tiers
(safe, race), a greedy set cover. A matrix again.

How it got there is the project's own history repeating itself. From-scratch
PPO capped at 14% against SimpleHeuristics; behaviour cloning reached 40% and
no more data or capacity moved it; search over the engine was the one thing
that jumped (52 → 95% clears). Every learned component oscillated around its
init and could not say why it chose; every explicit model could be read,
argued with, and fixed in an afternoon. The Gyarados-versus-Mantine "2 of 3"
cell was caught because a human looked at it and said it made no sense (no
Roost, no speed order). A learned chooser would have made the same mistake
silently, and we would have measured a win rate instead of finding a bug.

So the structure had to be explicit first. Not because learning is wrong here
but because without a readable baseline there is no way to tell a better
model from a different one. The matrix is now the interface: the cover
chooser, the switch rule, the threat report and the coming routing pass all
read cells. What fills a cell can change.

Where a learned piece earns its place is exactly the next step (TODO.md B1):
the duel calibration produces labelled pairs — features from the real sets
and the field, outcome from the engine — and a confusion table per tier. If
the hand-written tiers calibrate, the table stays. If the false cells pile up
and every fix is another special case, that is the signal to fit a small
model on the duel data, with the table as the baseline it has to beat and
the cells as the thing it fills. The idea was not wrong; it was early, and
the way to make it right is to let the table tell us when.

## 2026-10-12 — Is the result robust to the enemy AI port? (TODO.md A2.1, A2.3)

**A2.3, sensitivity.** The cover-pool teams (`results/rr_cover_pool.jsonl`) replayed
under search (9,8, matchup, death weight 6), n=30 on paired seeds, with the *real*
enemy perturbed and the search's model of it left at the base port
(`rr_deaths.py --ai`; logs and the dashboard JSON in `results/rr_aisens/`):

  variant    Brock wins  deathless  deaths   Misty wins  deathless  deaths
  base         30/30        7         37       30/30        9         41
  tiebreak     30/30       10         30       29/30        7         58   (AI tie-break RNG reseeded = noise floor)
  ko+50        30/30        7         37       30/30        9         41   (identical: KO +30 already dominates)
  ko-50        30/30        6         46       30/30        8         49
  noswitch     27/30        3         85       29/30        9         51

The noise floor is wide (Misty 41 vs 58 deaths from the tie-break alone), so the KO
weight is inside it. Switching is not: an enemy that never switches voluntarily, played
by a search that expects it to, more than doubles Brock's deaths (37 → 85, Cacnea 16 →
34, Lileep 5 → 18) and loses 3 fights; Misty moves within the noise. So the result
depends on the AI's switching shape, and the search's enemy model being wrong costs
more than any play lever measured so far.

**A2.1, read-through.** `data/rr/AI_DIFF.md`: the port against CFRU's
`src/Battle_AI/` (fetched from GitHub; RR 4.1's own changes and its trainers' AI flags
are unverified). The port is not the CFRU AI in the places that matter: CFRU scores in
small tiers (KO first +9, recover / protect +7 when a damage-calc check passes, Toxic
or safe setup +5, strongest attack +2, other attacks 0), so status and recovery beat
every non-KO attack, where ours lets a big attack beat Toxic and Slack Off; CFRU has
**no type-matchup switch** (it switches to absorb a predicted move, to dodge a predicted
KO when slower, on a Toxic counter > 6) and picks the switch-in by damage calc (+31 KOs,
+14 outspeeds, −39 dies), which is also who comes in after every faint; Power-Up Punch
and Stockpile are invisible to our setup check; hardcore bosses use no items.

Together: the switching rule is both the part the result is sensitive to and the part
the port gets most wrong. A2.2 (decision replay against the ROM) is now urgent, and the
first port fixes are the switch triggers and the switch-in choice (AI_DIFF.md §3 items
3–4), then the tier ladder (item 1), each accepted by a paired rerun of this table.

## 2026-10-12 — CFRU switching in the AI port (AI_DIFF.md §3 items 3–4)

`ai_util/rr_ai.py` switches the CFRU way now (`ai_switching.c`, singles, smart AI): the
switch-in is scored by damage calc (`CalcMostSuitableMonToSwitchInto`: outspeeds +14, KOs
+31, priority revenge kill +8, 2HKO +2, resists all +17, walls +2, faints −39 / −15 if it
outspeeds), which is also who comes in after every faint; the triggers are absorbing the
predicted move, only bad moves left, Toxic past six turns, avoiding a predicted KO when
slower, −3 offence, saving the sweeper; the type-matchup switch is gone, and a mon with
only dumped moves clicks the least bad one instead of switching. Damage is the calc bot's
model at its mean roll against the player's real moves; the player's predicted move is
its strongest hit. Not ported: Wish / Perish / locked moves / Yawn / sleep / hazard
removers, the fighting-style classes (damager = 3+ attacks), the absorb-switch memory.

Same teams, seeds and config as the A2.3 table; the search's enemy model is the new port
too, except in the noswitch row (`results/rr_cfrusw/`):

  variant    Brock wins  deathless  deaths   Misty wins  deathless  deaths   enemy switches/fight (B / M)
  base         29/30        2         76       17/30        1        126     2.6 / 2.4
  tiebreak     29/30        3         70       22/30        1        111     2.7 / 2.0
  noswitch     13/30        0        139       10/30        0        148     0.0 / 0.8 (Flip Turn)
  old port     30/30        7         37       30/30        9         41     (results/rr_aisens/ base)

**The CFRU-switching enemy is far harder**: Misty's cover team falls from 30/30 to 17–22/30
and 1.4 → 3.7–4.2 deaths per fight (Mantine 7 → 44–53 kills: it comes in on whatever it
outspeeds in rain and Hurricanes it), Brock doubles its deaths (Cacnea 16 → 28). A traced
fight (Misty, seed 1) reads like the game: Politoed Flip Turns into Mantine, which
one-shots Hariyama; Clodsire comes in on Crobat; every revenge kill is the bench mon that
outspeeds and KOs. The noswitch row (the post-faint choice is still CFRU's) shows most of
the damage is the switch-in choice, and that a search modelling a switcher plays worse
against one that does not. So every cover / box / early-boss number before this commit
was against an enemy that sent in the wrong mon; they have to be rerun before anything
else is tuned. Next on the AI: the tier ladder (AI_DIFF.md §3 item 1); A2.2 (ROM replay)
stays the real check of both.

## 2026-10-12 — CFRU move scoring in the AI port (AI_DIFF.md §3 items 1, 2, 5)

`ai_util/rr_ai.py` scores moves the CFRU way now: every move starts at 100, negatives are
soft (−10, −15 immune, −20 absorbed, recovery −10 at full / −9 at ≥90%, a second Protect
−6 half the time), positives are the tier ladder keyed by `PredictFightingStyle`
(kill / setup_stats / setup_status / stall / support classes): KO first +9, slow KO +8
(+6 for setup_stats, none with Protect in the set), desperation priority +9, strongest
attack +2, other attacks 0; Toxic / paralysis / burn / sleep via `IncreaseStatusViability`
(+5 for a setup_status mon), recovery by `ShouldRecover` (damage calc + speed order, the
50% roll), setup by `ShouldTryToSetUpStat` (+2 stage cap; Power-Up Punch and Stockpile
recognised), Protect by `ShouldProtect`, pivots by `ShouldPivot` (+9 / −9). The old
continuous scores, the type-matchup helpers and the weather table are deleted. Not ported:
predicted-switch exceptions, Destiny Bond, the long tail of rare effects (score 100).

Same paired check as the switching entry (`results/rr_cfrutier/`):

  variant    Brock wins  deathless  deaths   Misty wins  deathless  deaths
  base         26/30        2         80       19/30        0        127
  tiebreak     30/30        6         55       25/30        2        105
  noswitch     16/30        0        136        6/30        0        171
  switching only (base / tiebreak): Brock 76 / 70, Misty 126 / 111

The move choices changed as CFRU says they should (a Brock trace: Varoom Toxics every
new mon and Protects, Hippopotas Toxics, Lileep Stockpiles then drains, Cacnea opens with
Power-Up Punch), but the totals sit inside the tie-break noise: the switching was the part
that mattered. The read-through port is complete enough to rerun the RR tables on it;
what is left is the ROM check (A2.2).

## 2026-10-12 — The RR tables on the CFRU port (`results/rr_cfru_{pool,boxes,early}.*`)

Same flags as the cover runs of 2026-10-11 (legal, arena, cover chooser, matchup play,
death weight 6, search 9,8; pool n=30, boxes 30 × n=6), enemy = the port with CFRU
switching and scoring (commits 1cd9bdd, c8f94e1, 091d70a). Old = `results/rr_cover_*`.

  planned pool   win          deaths/mon     deathless    best six (new)
  Brock          100 → 100%   0.21 → 0.08    23 → 63%     Bonsly, Budew, Kricketot, Patrat, Zigzagoon, Galarian Zigzagoon
  Misty          100 →  73%   0.23 → 0.57    30 →  3%     Ekans, Ledyba, Smoochum, Spearow, Teddiursa, Zubat

  boxes          mean win     ≥50% boxes     deaths/mon   deathless   most missed counter
  Brock          66 → 75%     80 → 93%       0.63 → 0.54   3 → 11%    Sewaddle vs Lunatone ×19
  Misty          58 → 51%     67 → 53%       0.68 → 0.72   5 →  4%    Bramblin vs Mantine ×17

  early bosses (#2 #4 #5 #7 #9 #10 #15 #16): all still 30/30; deathless 100/100/93/100/
  67/63/100/90% (was 100/100/100/100/73/93/100/80%).

**Misty is the wall again.** Last night's 30/30 cover team (Grimer, Magikarp, Makuhita,
Paras, Pikipek, Zubat) is 63% / 0.71 / 0% against the CFRU enemy; the best of the six
candidates is 73% with 3% deathless. Toxicroak (27 deaths) and Mantine (25) are the
killers, both brought in as revenge killers by the switch-in score; their duel counters
in the pool (Pidove, Wattrel, Ekans; Bramblin 4%) are what routing would have to catch.
**Brock got easier**, not harder: the old cover six is 93% / 3% deathless, but a different
six (the "top" candidate) is 100% / 0.08 / 63% — under CFRU scoring the boss spends turns
on Toxic / Stockpile / Protect that the old port spent attacking, which a fast six punishes.
The best-of-six is picked on the same 30 seeds it is reported on (optimistic, as before).

The early singles bosses stay solved. The project's open question narrows to Misty under a
correct AI: chooser and play against Mantine-in-rain and a Swords Dance Toxicroak, and
whether the box can hold their counters (routing, TODO G). All of it rests on the port
matching the ROM (A2.2).

## 2026-10-12 — BDSP: Gen 4 enemy AI, team building by matchup cover, League locked + healed

Three changes to what a BDSP run is (HANDOFF.md → "What changed on 2026-10-11/12"):
the Gen 4 (Platinum) trainer AI as an optional enemy (`ai_util/bdsp_ai.py`, `--enemy
bdsp`); the planner's base party by matchup cover over the whole box (`cover_base`;
the old top-6-by-level base was catch order once the level rule puts the box at the
cap: Bibarel / Kricketune / Staraptor fielded from Roark to Volkner with 38 in the box,
one slot rotating); the League `locked` (was `no_heal`): party locked, full heal of HP /
status / PP between fights (PP was carried).

Full game, 20 episodes, seed 300, `--policy calc --search 9,8 --search-all --deep 9,16`,
7 workers each (`results/bdsp_{heuristic,gen4ai}.{log,jsonl,events.jsonl}`; every
Cynthia fight in `results/bdsp_traces/`, watchable in ui/):

  enemy                cleared   no-death   deaths/run   runs cleared   min
  SimpleHeuristics      404.95     403.30      1.85          95%          88
  Gen 4 trainer AI      404.65     403.00      3.95          65%          99
  (2026-09-23, old planner, PP carried: 403.8 / 368.0 / 5.9 / 20%, results/plan_league4_swap.log)

**Not one death before the League in 40 episodes**: the route, every gym and the rival
are solved under either enemy. All deaths are E4 + Cynthia; Cynthia alone is 25 deaths /
1 loss (SimpleHeuristics) and 63 deaths / 7 losses (Gen 4 AI). The Gen 4 AI is the
harder enemy where it matters: clears 95 → 65%, deaths at the League doubled (Staraptor,
Gyarados, Roserade, Mantine, Lapras the most killed). The jump from the 2026-09-23 table
confounds two changes (the cover base and the healed League); not separated — a paired
League replay (`league_test.py --vs`) with and without the heal would split them.
The benchmark is now "the League under the Gen 4 AI"; the route layer is a regression check.

## 2026-10-12 — Speed-up 1: the RR box probe spread over workers (`results/rr_cfru_boxes_par.*`)

The task from HANDOFF.md: the same results in a fraction of the wall-clock, every step
shown to change nothing on paired seeds. First the structure, then the profile.

**Where the RR box run's time went.** `rr_team_probe.py --boxes` pooled over *bosses*, so
the 2026-10-12 table (Brock + Misty, 30 boxes each) ran on 2 cores for 165 min with 30
boxes per boss one after another (Misty ~230 s, Brock ~310 s per box). A box is
seeded on its own (the sample, the engine, the RR AI's tie-breaks, the searcher's
rollouts), so a box is now a pool job (`box_job`; a worker builds the boss context once
and keeps it, `_CTX`); the per-boss banner / solo stage run first in the main process
(`boxes_start`), the row is folded from the jobs (`boxes_row`). The dashboard sorts a
boss's boxes by index (the workers finish them out of order).

**Acceptance (exact).** Two serial runs of Misty boxes 1–2 agree with each other and with
the 2-worker run on team, win, deaths, deathless and threats; the full table on 20
workers (`results/rr_cfru_boxes_par.{log,jsonl}`) matches `results/rr_cfru_boxes.jsonl`
on all 60 boxes (team, win, deaths/mon, deathless, threats; compare script in the
notebook's spirit: per box, per field):

  run                       workers   min   Brock mean win / ≥50% / d/mon / deathless   Misty
  2026-10-12 serial            2     165      75% / 93% / 0.54 / 11%                   51% / 53% / 0.72 / 4%
  boxes as jobs               20      22      identical                                identical

7.5x; 60 jobs on 20 workers is 3 rounds with Brock's longer boxes in the tail (30 workers
would be 2 rounds, ~30 GB). Per-box time under 20-way contention equals the serial time
(first finishers 233 s), so the machine is not the limit yet.

**Profile of one Misty box** (wall-clock split, no profiler, n_search=2, measured while the
20 workers ran): bridge `choose` 44% (1.0 ms per turn: the Node engine + the protocol
round trip), the RR AI 24% (0.67 ms per decision), the calc bot 13% (0.33 ms), forks 3%,
the rest (mirror feeding, attribution) 13%. Under cProfile the damage formula
(`calc_bot.dmg_frac`, 13.4M calls per box, both AIs) is the largest Python item and a
third of it is poke-env's `Move.entry` property chain. The searched play is 97% of a
box; the threat duels most of the rest.

**Speed-up 2: the pool probe's stage C as battle jobs** (`results/rr_cfru_pool_par.*`). The
planned-pool form pooled over bosses too, and its time is stage C: up to seven teams × 30
searched battles, one after another per boss (151 min for Brock + Misty on 2 cores). A
searched battle is seeded on its own (engine seed, the AI's tie-breaks, the rollouts, all
by battle index), so `one_battle` is the pool job and `searched_many` fans the teams' battles
out at once (`fan` / `_POOL`; a worker keeps its boss context, `worker_ctx`). Stages A / B
(~1 min per boss) and the threat duels stay in the main process. Acceptance: identical to
`results/rr_cfru_pool.jsonl` on every team's win / deaths / deathless, the best team, the
threats and the shortlist, for both bosses.

  run                       workers   min   Brock best win / d/mon / deathless   Misty
  2026-10-12 serial            2     151      100% / 0.08 / 63%                  73% / 0.57 / 3%
  stage C as battle jobs      20      25      identical                          identical

**The Node side is the engine.** One searched Misty battle: the `choose` round trip is 0.98
ms per turn (6 KB reply), JSON parse 0.03 ms, feeding both mirrors 0.18 ms. In pure Node
(random choices on the same matchup, the RR fork) `Battle.choose` is 1.09 ms per turn and
sendUpdates + channel extraction + stringify 0.025 ms; a fork (toJSON → fromJSON) 1.5 ms.
V8 young-generation flags (`--max-semi-space-size`) change nothing. Nothing to shave in the
bridge; the engine's turn is the floor of a rollout turn.

**Speed-up 3: the damage formula memoised** (`calc_bot.dmg_frac` → `_DMG`). Both AIs (the
calc bot, the RR AI, the Gen 4 AI) ask the formula the same question many times per turn
and across rollouts: 13.4M calls per searched Misty box, the largest Python item under
cProfile (a third of it poke-env's `Move.entry` property chain). The memo key is every
input the formula reads — move id and power override, each side's species, level, stats,
base stats, boosts, status, types, ability and item, the weather and terrain, `att_own`,
`with_acc`, `FIELD_AWARE` — so it is exact by construction (HP is not an input: the
formula returns a fraction of max HP). Checks: the same searched Misty battle plays the
same 25,048 turns in 46.5 s instead of 55.3 s (-16%, both under the 20-worker load);
Misty boxes 1–2 identical to the serial baseline (team, win, deaths, threats).

## 2026-10-12 — Where a BDSP episode's time goes (seed 300, Gen 4 AI, the benchmark flags)

One timed full-game episode (wall-clock, no profiler, wrappers on the planner's party
choice, the risk queries and the runner's fights; scratch script, numbers below):

  total 2129 s = party choice 1632 s (77%) + fights 496 s (395 fights, 14 deep) + rest 1 s
  party choice = cover base 5 s + risk queries 1614 s: 46,614 estimate() calls (8 fast
  battles each) + 193 block calls at the League opener

  trainers     fights   choose    fight   queries   box (alive)
    0- 59        57       38 s     34 s     1,395     4-13
   59-167       106      257 s     66 s     8,662    13-24
  167-300       129      581 s    116 s    18,147    25-36
  300-398        97      538 s    109 s    18,199    36-39
  398-405         2      216 s     77 s       392    38-39   (the League opener's block query)

The swap scoring grows with the box (≈ 6 × box candidates per trainer, each 8 battles) and
is three quarters of the episode; search is a quarter. With one episode per worker (20
workers, approved today) the benchmark is bounded by the slowest episode, ~36 min instead
of 99; inside an episode the exact levers are small (the damage memo above applies to the
calc bot and the Gen 4 AI; the Node turn is the floor), so the lever is the number of
queries — a decision change, measured below.

**Acceptance of 1–3 together: the early-boss table** (`results/rr_cfru_early_par.*`, the 8
singles bosses #2–#16, 6 teams × 30 searched battles each, bosses one after another with
their battles fanned out, the memo in): identical to `results/rr_cfru_early.jsonl` on every
team's win / deaths / deathless, best team, threats and shortlist for all 8 bosses; 119 →
40 min on 20 workers. Found on the way: a worker kept one context (a ~200 MB Node bridge)
per boss it had seen — 121 bridges / 41 GB at 5 of 8 bosses — fixed: one context per
worker, the previous bridge closed on a boss switch (`worker_ctx`). The per-boss serial
chain (stages A + B, then the fan-out, then the duels, per boss) is what the three-phase
version (`run_pool`: preludes, all battles, all duels, each one fan-out over every boss)
removes; its acceptance is the same table again (`results/rr_cfru_early_par2.*`, below).

**Three-phase pool mode accepted** (`results/rr_cfru_early_par2.*`): the early-boss table
again, now with all 8 bosses' preludes, then all 6 × 30 × 8 battles, then all duels as one
fan-out each — identical to `results/rr_cfru_early.jsonl` on every team, best team, threat
and shortlist; 35 min (40 with the per-boss chain, 119 serial), 20 workers + 8 main-process
bridges at ~24 GB. The RR tables now cost their battles / 20.

## 2026-10-12 — `--swaps hard`: the swap scoring only at fights the cover six does not clear cleanly (`results/bdsp_gen4ai_swapshard.*`)

The lever the timed episode points at: score the single swaps only when the cover six's own
n=8 query shows a loss, a death or HP left < DEEP_HP (the deep-search trigger, cached, so it
is free); otherwise field the six. Full game, 20 episodes, seed 300, Gen 4 AI, the benchmark
flags, 20 workers:

  swaps   workers   min    cleared   no-death   deaths/run   runs cleared   deaths before the League
  all        7       99    404.65     403.00       3.95          65%              0 (40 episodes incl. the heuristic run)
  hard      20       17    404.55     390.10       4.50          55%              4 (Candice #317, in 3 of 20 episodes: seeds 300, 301, 316)

Paired by seed and trainer (events files): the fielded route party is the same in 55% of
the 7,900 fights, differs in 45% (first at trainers 14–25: a swap with 0.2 × HP margin over
the clean six), and the fight result is the same in 7,888 of 7,900. The League rows are
within the noise floor (±2 cleared, ±1.9 deaths, ±15 points); the opener's block query is
always "hard", so the League party choice is unchanged in kind. The signal is the route:
four deaths at Candice in three of the 20 episodes, where the saved runs had none in 40 —
sixes whose 8-battle query under the fast brain was clean lost mons in the real
(deep-searched) hail fight. Three of 20 against 0 of 20 under the same enemy is suggestive, not decisive
(a 5–8% per-episode event fits both). **Not accepted** (the flag stays `all`). Refinement
run next: `hard` scores the swaps at every breakpoint (gym leaders, the rival, the League
opener — where deep search already runs) and prunes only route trainers whose query is
clean; the clean route fights are the bulk of the queries, so most of the saving should stay. Wall-clock for `all` on 20 workers is not measured; the timed episode (2129 s) puts it
at ~36 min.

**`hard` = breakpoints always + clean route trainers pruned** (`results/bdsp_gen4ai_swapshard2.*`,
the flag's meaning from here on; same 20 seeds, 20 workers):

  swaps                 min    cleared   no-death   deaths/run   runs cleared   route deaths
  all (7 workers)        99    404.65     403.00       3.95          65%        0
  hard, route-only       17    404.55     390.10       4.50          55%        4 (Candice ×4, 3 episodes)
  hard, breakpoints      17    404.75     394.70       3.80          75%        3 (Candice ×2, #397 ×1)

Same wall-clock: the clean route fights are the queries. The League rows are the best of
the three and within the noise floor. At a breakpoint this mode scores exactly what `all`
scores (same box, same cover base, same candidates), so its two Candice deaths are the
same process as the saved run's zero — a noise-level difference at 20 episodes. The one
death the pruning can own is #397 (seed 306, Crobat): a route trainer whose six was clean
at n=8 and lost a mon in the fight; under `all` a swap with more HP margin may have been
fielded. The headline metric (trainers cleared without a death) is what a route death
hurts, so: **`all` stays the benchmark's default; `--swaps hard` is the development
setting** (2x faster on 20 workers, 17 vs ~36 min, ~0.05 route deaths per run). The
user decides whether the benchmark table itself moves to `hard`.

## 2026-10-12 — Matrix calibration: are the safe cells safe? (TODO.md B1, `results/rr_matrix_calib.*`)

The user's decisions first (HUMAN_TODO 7–12, all recorded there): RR 4.1 hardcore with the
dupes clause, random starter across every generation, singles only, the 95 bosses (optional
bosses for items later), natures free, any ability after Cerulean, first encounter only —
every one is what the sim already assumes. The BDSP benchmark stays on `--swaps all`.

**The question.** The cover chooser and the calc bot's switching trust `matchup.pair()`: a
one-turn damage calc per (candidate, enemy mon) labelled safe / race / none. Nothing had
scored those labels against play. `scripts/rr_matrix_calib.py` plays every cell: each
stage-A candidate (pool + starters, built as the probes build them) against each enemy mon
alone under the arena, the calc bot vs the RR AI, 20 paired seeds (the probe's), and
tabulates the label against the duel. 4 ms a duel; Brock + Misty in 2 min serially.

| boss | tier | cells | duel win | HP left (won) | win ≥ 90% | win ≤ 10% |
|---|---|---|---|---|---|---|
| Brock (132 cands) | safe | 182 | 86% | 65% | 76% | 7% |
| | race | 138 | 58% | 40% | 38% | 20% |
| | none | 460 | 16% | 43% | 7% | 73% |
| Misty (219 cands) | safe | 76 | 83% | 57% | 75% | 9% |
| | race | 150 | 54% | 47% | 37% | 28% |
| | none | 1088 | 7% | 41% | 2% | 88% |

The tiers order the cells correctly (safe 85%, race 56%, none 9% over both bosses) and
the safe margin behaves (ko_us 3 → 83%, 4 → 89%, 5 → 90%). But one safe cell in six
loses the duel outright, and 56 none cells win it ≥ 90% of the time: answers the chooser
never sees. The false cells by name, traced turn by turn (seed 0):

- **False safes, 32** (safe, duel < 50%). Brock: Varoom 10, Hippopotas 7; Misty:
  Clodsire 9. Causes, in order of count:
  1. **The enemy's item and conditional moves are not in the formula.** Varoom's Air
     Balloon makes every Ground answer immune (Mudkip, Turtwig, Swinub, Sandygast,
     Toedscool, Greavard: "1–2 hits" on paper, 0% in play); Chimecho's Synchronoise
     reads as a one-hit KO and does nothing to a Steel/Poison. TODO.md C2.
  2. **Toxic + Protect / Recover / Slack Off stall.** The healer rule passes a "2 hits,
     faster" cell; Protect doubles the turns, Toxic + sandstorm chip finishes a mon that
     cannot two-shot through it (Hippopotas / Berry Juice, Clodsire / Black Sludge, Varoom).
  3. **Two-turn moves count as one hit.** Sudowoodo's Dig is "2/3 faster" on paper and
     takes four turns while Clodsire's Toxic + Bulldoze (double damage on a digger) lands.
  4. The 10–45% cells are rolls, crits and 95%-accuracy misses on thin margins (Weedle vs
     Hippopotas wins seed 0 and 10% overall).
- **False nones, 56** (none, duel ≥ 90%). Brock: Lileep 14 (healer rule), Archen 5;
  Misty: Politoed 13, Clodsire 6. Causes:
  1. **The mirrors carry vanilla data for species and moves the mod changed.**
     `rr_probe.inject_mod_data` adds only entries poke-env lacks; for the rest poke-env's
     gen 9 tables stand while the engine plays the mod's: **40 species with changed
     types** (Arbok Poison→Poison/Dark: immune to Starmie's Psyshock, "2/2" → wins 100%;
     Ledian Bug/Flying→Bug/Fighting: Ancient Power "113%" on paper, 37% in play), 250
     with changed base stats, 446 with changed ability sets, and **134 moves with changed
     power / type / accuracy** (Attack Order 90→120, Blaze Kick 90→100% accurate, ...).
     Hidden Power's typed variants read as Normal 60. The formula's inputs are wrong for
     those cells, both ways. Cheapest fix on the list: override the existing entries too.
  2. **The enemy AI wastes turns.** Politoed opens with Toxic, Lileep with Stockpile /
     Recover, so a "3 hits" answer gets its three hits; the healer rule discards Lileep's
     answers wholesale (18 of Brock's 30 false nones carry the healer flag).
  3. The formula undercounts some of our hits (Meditite's Force Palm 43% on paper, 61%
     in play; its paralysis then costs Lileep a turn). Not pinned down; part of B2.

**Verdict.** The matrix is a usable ranking (the chooser's safe tier wins 85% of its
duels) with a ~15% lie rate at the top and a pool of discarded winners underneath. The
first B2 term is not a tactic but data: the mod's own types / stats / moves in the mirrors
(A1's "the mod's changed types", found from the inside). Then the enemy's item (Air
Balloon, Sash, berries), then Toxic-stall as a "disables" flag, then two-turn moves. Each
is accepted by this table improving and the cover runs not getting worse (paired).

Caveat: the duel has no bench and no switching, so it scores the cell, not the fight;
D1 (sand switch-ins) and D3 (boxes) stay the judges of play.

## 2026-10-12 — The mod's data in the mirrors; the BC pipeline deleted (`results/rr_matrix_calib_mod.*`, `results/rr_moddata_pool*.*`)

**Mod data.** `rr_probe.inject_mod_data` now replaces poke-env's gen 9 entries with the
fork's merged gen9rr Dex (it only added the entries poke-env lacked before): 40 species
with other types, 250 with other base stats, 446 with other ability sets, 134 moves with
other power / type / accuracy. The calibration rerun on the same seeds: safe 85 → 86% of
duels, race 56 → 60%, false safes 32 → 31, false nones 56 → 46 (Arbok vs Starmie is a
safe cell now). Only the cells with a changed entry move, as expected.

The paired cover pool run (`rr_team_probe --bosses BROCK,MISTY --arena --legal --chooser
cover --matchup --death-weight 6 --n-search 30`, 20 workers, 26 min) against
`results/rr_cfru_pool_par.*`:

| boss | saved best | with the mod data |
|---|---|---|
| Brock | 100% / 0.08 deaths per mon / 63% deathless | 90% / 0.34 / 20% |
| Misty | 73% / 0.57 / 3% | 90% / 0.49 / 0% |

The calc bot plays stage A with the corrected data, so the shortlist, the greedy six and
the cover six all change; no Brock team is in both runs. Misty's cover six is the same
team both times and goes 63 → 90% under the corrected data. Brock's drop is a chooser
shuffle to judge on the box milestone (30 boxes average over many sixes), not a known
regression of the data; the saved best six is being replayed under the new data to
settle it (next entry). The data is the engine's, so the fix stands either way.

**The BC pipeline is deleted** (HUMAN_TODO 5, the user's decision): `ai_util/battler_obs.py`,
`battler_policy.py`, `collect_game.py`, `pretrain_bc.py`, the prior branches in
`evaluate.py` / `runner.py`, `--harvest` and the harvest recorder, torch and
stable-baselines3 (711 lines, 1.5 GB of checkpoints and harvests left on disk, under
`.gitignore`). `action_mask` (identity-based switch slots) moved to `ai_util/evaluate.py`;
`Searcher(enemy, ...)` lost its policy argument. Exact: the pool run repeated on the
tree after the deletion is identical to the one before it on every team, threat and
shortlist (`results/rr_moddata_pool2.*` vs `rr_moddata_pool.*`); 38 tests pass; the
recipe stays in this notebook, the code under the `pre-cleanup` tag.

## 2026-10-13 — The formula's floors, items and abilities; two matrix rules (B2) (`results/rr_matrix_calib_formula.*`, `results/rr_formula_*`, `results/rr_matrix_boxes.*`, `results/bdsp_formula.*`)

**The check.** `rr_matrix_calib.py` now also scores the formula itself: the first hit each
side lands in a duel (its move and the fraction it took, averaged over the seeds that opened
with the same move) against `dmg_frac` for that move, grouped by attacker ability, defender
item and move. One seed alone reads ~8% low across the board (the seeds share their first
roll), hence the mean. On the committed formula at Brock: median engine/paper 0.88, 73% of
cells within a damage roll (±20%); Eviolite 0.62, Pure Power 1.57, a typed Hidden Power 1.93,
Dig 0.63, Synchronoise 0. The 0.88 is the floors: the engine floors the level term, the base,
the roll, STAB and the type multiplier, and at L13–16 `floor(2·16/5 + 2)` alone is 8 against
8.4 (Axew's Breaking Swipe: 50% on paper, 43 / 43 / 47% in play).

**The terms** (`calc_bot._dmg_frac`, `effectiveness`; `sim.py`), each named by a group above
the band: the floors in the engine's order; Technician, Sheer Force, Pure / Huge Power; Life
Orb, Choice Band / Specs, Eviolite (NFE by the dex's `evos`), Knock Off's item bonus; Air
Balloon voids Ground; Psyshock hits Defence (`overrideDefensiveStat`); a charge move at half
damage per turn; a typed Hidden Power from a packed set keeps its type in the full-info
mirror (poke-env built `Move(retrieve_id)` and dropped it; the mod's own `hiddenpower` entry
reads Water, which is wrong too). After them: median 1.00 / 0.99, 95% of our hits and 91% of
theirs within a roll at Brock, 86% / 75% over both bosses; what is left is Synchronoise (type-
conditional, 0), Bullet Seed (each hit floors), Drain Punch 1.27 (unexplained, n=104) and Dig
2.0 (the check scores the full hit against the halved per-turn value: not an error).

**Two matrix rules** (`matchup.pair` / `ours`): a healer costs one extra hit unless it falls
before it heals (it disqualified every answer slower than two hits before); Air Balloon pops
on the first hit, so a Ground move lands from the second (Toedscool "15 hits" beat Varoom in
four turns: Mega Drain popped it, Mud-Slap finished). Calibration gain: +4 safe cells, false
nones 64 → 58, false safes unchanged.

**Calibration, both bosses, 20 seeds:**

| | safe | race | none | false safes | false nones |
|---|---|---|---|---|---|
| committed (mod data) | 258 cells, 86% | 288, 60% | 1548, 9% | 31 | 46 |
| + the formula | 195, 95% | 256, 72% | 1643, 11% | 7 | 64 |
| + the matrix rules | 199, 95% | 260, 72% | 1635, 11% | 7 | 58 |

Of the 7 false safes, 3 are Chimecho's Synchronoise. Of the 58 false nones, most are 2/2
"slower" coin flips won on rolls (right to discard for a safe cover) and 3-hit answers to
Lileep / Politoed / Clodsire that win because the enemy AI opens with Stockpile / Toxic /
Recover: a free turn the paper does not model. Modelling the AI's first move is a decision
for later (ask the port itself, or a duel-based cover: TODO.md B3).

**The tables moved, down** (paired on the saved seeds, boxes drawn the same):

| | Brock boxes: win / ≥50% / deaths per mon / deathless | Misty boxes |
|---|---|---|
| saved (`rr_cfru_boxes_par`) | 75% / 93% / 0.54 / 11% | 51% / 53% / 0.72 / 4% |
| the formula | 71% / 87% / 0.57 / 5% | 39% / 43% / 0.79 / 1% |
| + the matrix rules | 70% / 87% / 0.57 / 7% | 40% / 47% / 0.76 / 3% |

The pool table's best six: Brock 100% / 0.22 / 27%, Misty 63% / 0.70 / 0% (the best six
differs in every run; the number is the shortlist shuffle plus n=30 noise, which is why the
boxes judge).

**Why: the enemy got the formula too.** The saved run's 60 teams replayed unchanged on the
same seeds (`results/rr_formula_split.log`, 180 battles per boss, SE ≈ 3.5 points):

| who is on the new formula | Brock | Misty |
|---|---|---|
| nobody (the saved run) | 75% | 51% |
| the calc bot and the matrix only | 77% | 45% |
| the RR AI only | 74% | 33% |
| both (the tree) | 70% | 39% |

The chooser is not the cause (the same teams drop the same way). The RR AI port scores
moves and switches with `dmg_frac`, so it now computes damage the way the game does; at Misty
that is worth 12–18 points against us. The calc bot's own change is within the noise (−6 /
+6). CFRU's AI calculates with the real formula, so this is the game being harder than the
old tables said, not a regression of the method; the tables are regenerated from here.

**BDSP** (the Gen 4 AI and the calc bot share the formula; `results/bdsp_formula.*`, 38 min on
20 workers, paired by events against `results/bdsp_gen4ai.*`): 404.60 cleared / 389.30
no-death / 4.10 deaths / 60% runs cleared vs 404.65 / 403.00 / 3.95 / 65%. Cleared, deaths
and the clear rate are within the noise floor; the route is not: **7 deaths before the League
in 20 episodes where the saved run had 0** (Volkner 3, Candice 2, Cedric 2), the party fielded
differs from trainer 16 on in every seed (73.5% of fights). Cynthia 65 deaths / 8 losses (63 /
7). The same split as above runs for BDSP (the Gen 4 AI on the old formula, the calc bot on
the new): see the line below.

BDSP split (`results/bdsp_formula_enemyold.*`: the Gen 4 AI on the old formula, the calc bot
and the searcher on the new, 38 min): 404.40 / 393.45 / 4.80 / 45%, **2 route deaths**
(Fantina 1, Cedric 1), Cynthia 77 / 10. So of the 7 route deaths under the full change, the
Gen 4 AI's better calc accounts for most (7 → 2 when it loses it) and the calc bot's own
change for the rest (2 vs 0); every League row is within the noise floor (±2 cleared, ±1.9
deaths, ±15 points) in all three configurations, Cynthia 63 / 65 / 77 deaths. Same reading
as Radical Red: the enemy AI with the game's damage numbers is the harder opponent, and the
benchmark from here is `results/bdsp_formula.*` (404.60 / 389.30 / 4.10 / 60%).

## 2026-10-13 — The duel as the chooser's judge (B3); Mantine's Wacan Berry (`results/rr_duel_boxes*.*`, `results/rr_matrix_calib_{berry,resist}.*`, `results/rr_berry_boxes.*`, `results/rr_berry_duel_boxes.*`)

**B3, as run.** `--chooser duel` (`rr_team_probe.cover_team`): the cover's rows are still read
off the dummy battles, but every cell is then *judged by the duel* — `duel()` plays the
one-on-one on `--n-duel` paired seeds (8 here; ~4 ms each, a whole box in seconds) and the
cell's tiers come from it (`matchup.safe / race / edge` on `win` / `hp`: safe = wins ≥ 90% with
half its HP left, race = wins more often than not, edge = win·(1 + HP left) − losses). The
paper keeps its job in play (`--matchup` switching); only the chooser changes. The reason was
the calibration's false nones: 3-hit answers to Lileep / Politoed / Clodsire that win because
the enemy AI opens with Stockpile / Toxic / Recover, a free turn the paper cannot see and the
duel plays.

**Paired on the 30 boxes of `rr_matrix_boxes` (same seed, same boxes; the six differs in 27 / 29
of them):**

| | Brock: win / ≥50% / deaths per mon / deathless | Misty |
|---|---|---|
| paper chooser (`rr_matrix_boxes`) | 70% / 87% / 0.57 / 7% | 40% / 47% / 0.76 / 3% |
| duel chooser (`rr_duel_boxes`) | 77% / 93% / 0.55 / 9% | 40% / 43% / 0.83 / 1% |
| + the loss term in the edge (`rr_duel_boxes2`) | 75% / 90% / 0.55 / 9% | 40% / 43% / 0.83 / 1% |

Paired differences (duel − paper, SE over the 30 boxes): Brock win +7.2 ± 4.4, deaths −0.02 ±
0.03; Misty win 0.0 ± 4.7, deaths **+0.064 ± 0.035**, deathless −2.8 ± 1.4. Brock gains (the
duel brings Binacle / Tentacool / Shinx / Houndour in for Starly / Tarountula; Lileep's and
Cacnea's kills fall, Lunatone's rise); Misty does not: Carnivine (in every box: a gift) is the
duel's safe answer to Politoed (100%, 60% HP) and a race against Ludicolo / Starmie, and loses
0% to Mantine, Clodsire and Toxicroak — the chooser takes it in 19 of 29 differing boxes,
mostly for Chewtle (the paper's and the duel's race answer to Mantine), and Mantine's kills go
164 → 253. The loss term in the edge changed nothing at Misty (the same 30 sixes): Carnivine
enters through the safe / race tiers, not the tie-break. **Verdict: not the default.** The
chooser's problem at Misty is the team context the duel cannot see any more than the paper
can — a mon that wins its one duel and is dead weight in front of the other three — and the
cover's tiers have no term for a liability. The flag stays as the cheap way to ask the duel.

**Found on the way: the paper's Electric one-shots on Mantine land at half.** The calibration's
"race 1/2 slower" cells against Mantine (Thunder Punch / Spark / Thunder Fang, paper 1.01–1.58
of its HP) lose the duel (5–50%), and the engine's first hit reads 0.44–0.68: Misty's Mantine
holds a **Wacan Berry** (the first super-effective Electric hit halved, then eaten). The formula
had no resist berries. Added (`calc_bot.RESIST_BERRY`, 18 berries, in `_dmg_frac`; the mirrors
clear the item on `-enditem`, so the second hit is full) and the matrix spends it like the
balloon (`matchup.hits_to_ko`: the first hit under the item, the rest without). Calibration,
same 20 seeds (`rr_matrix_calib_resist`): Misty race cells 132 → 112 at 71 → 78% duel wins,
false nones 26 → 27, Brock unchanged (no Wacan there); our first hit within a roll 87 → 90%
over both bosses (Misty 81 → 86%, engine < 0.8× 8 → 4%).

**Tried and dropped: the enemy's heal berry as HP.** Sitrus / Oran / Berry Juice / Iapapa as a
heal eaten once at the threshold (`rr_matrix_calib_berry`) turned Brock's safe cells 156 → 123
and the false nones 32 → 70: Hippopotas' Berry Juice makes a 2/6 cell 3/6, which the tiers
call "none" (ours > 2) though it wins 100% of duels; Politoed's Iapapa is right half the time
(7 demoted cells still win ≥ 90%, 6 lose). The mechanic is real; the tiers' cap on *our* hits
is what breaks — a 3/6 is an answer — and that is a matrix question for later, not a berry one.

**The tables on the new formula** (paired on the same boxes; every RR table before this one
has the paper's Wacan error, which touches Misty's Electric cells only):

| | Brock: win / ≥50% / deaths per mon / deathless | Misty |
|---|---|---|
| paper chooser, old formula (`rr_matrix_boxes`) | 70% / 87% / 0.57 / 7% | 40% / 47% / 0.76 / 3% |
| paper chooser, Wacan in (`rr_berry_boxes`) — **the baseline from here** | 70% / 87% / 0.57 / 7% (identical: no berry at Brock) | 44% / 43% / 0.75 / 3% |
| duel chooser, Wacan in (`rr_berry_duel_boxes`) | 75% / 90% / 0.55 / 9% | 38% / 33% / 0.82 / 2% |

Paper, old → new at Misty: +3.9 ± 3.4 points of wins on 8 changed sixes — Chewtle in for Sentret
five times: the paper no longer sells Normal types with Thunder Punch as Mantine answers.
Duel − paper on the new formula: Brock +5.0 ± 4.8 wins, deaths −0.02 ± 0.03; Misty **−6.1 ± 6.0
wins, deaths +0.070 ± 0.042**, Carnivine in 18 of 29 differing sixes for Chewtle (12), Mantine's
kills 175 → 227. Same picture as on the old formula; the paper chooser stays the default.

Also fixed on the way: `hits_to_ko`'s first draft computed the no-item case as 1 + ceil((hp −
hit) / hit), and (1 − 1/6) / (1/6) is 5.000000000000001 in floating point — Psyduck's sixth hit
on Lileep became a seventh and one Brock six changed. The no-item path is the old ceil(hp / hit)
again, the item path carries an epsilon, and the Brock table is bit-identical across the formula
change, as it must be (no berry at Brock).

Operational: pair box tables by index on the `.jsonl` rows (`boxes[i]`, the fan-out keeps job
order); the events file is in worker-finish order and carries the index as `i`.

## 2026-10-14 — No PP Ups (`results/rr_matrix_calib_pp.*`, `results/rr_pp_boxes.*`, `results/bdsp_pp.*`)

**The question** (the user's): are all moves PP-maxed? They were. Showdown builds every move slot
at `pp × 8/5` — three PP Ups — unless the move forbids them (`sim/pokemon.ts`), in both forks, for
both sides, and nothing of ours overrode it. The games' trainers and a hardcore player have the
base PP: Recover / Roost / Slack Off / Sucker Punch 5 (the sim gave 8), Toxic / Protect / Hurricane
/ Drain Punch 10 (16). Within one fight that is the enemy's stall budget — Lileep's and
Clodsire's Recover, Mantine's Roost, Hippopotas' Slack Off, the Toxic users — and our own
five-PP moves. Across fights nothing carries any more (BDSP's League is `locked` with a full
heal), so the error was confined to the fight.

**The fix** (`ai_util/sim_bridge.js`): the battle's `start()` is wrapped so that, once both
sides' pokemon exist and before the first request goes out, every move slot is set to the
dex's base PP (`basePP`), then a carry is applied on top (`applyCarry`, capped at the new max).
No engine patch, no flag: the old results files are the paired baseline. Roost is 10 PP in gen 8
and 5 from gen 9, so BDSP and Radical Red get their own generation's numbers. `tests/test_sim.py`
checks the state's PP and a carry. poke-env's `Move.max_pp` still reads × 8/5 from its own dex;
nothing of ours reads it.

**Paired, same seeds and boxes.** The one-on-ones do not notice (2–4 turns, nobody runs dry):
the calibration is unchanged to the cell except one Hippopotas duel; the five Misty cells that
changed *paper* tier are the `hits_to_ko` rounding fix from the day before (a hit of exactly 1/3
was a fourth hit), since the previous calibration predates it. The 6v6 fights do:

| | Brock boxes: win / ≥50% / deaths per mon / deathless | Misty boxes |
|---|---|---|
| PP Ups (`rr_berry_boxes`) | 70% / 87% / 0.57 / 7% | 44% / 43% / 0.75 / 3% |
| base PP (`rr_pp_boxes`) — **the baseline from here** | 74% / 87% / 0.57 / 8% | 41% / 40% / 0.78 / 3% |

The same 60 sixes (the chooser reads nothing of PP); paired: Brock win +4.4 ± 3.7, Misty −2.8 ±
3.8, Misty deaths +0.03 ± 0.02. Kills by enemy over the 180 battles: the stallers fall —
Hippopotas 35 → 22, Lileep 146 → 128, Clodsire 79 → 61 — and Misty's attackers rise (Mantine
175 → 197, Starmie 157 → 185), which reads as the fights running longer into the attackers once
the stall is spent, within ~2 SE of a count.

| BDSP, Gen 4 AI, 20 episodes | cleared | no-death | deaths | runs cleared | route deaths | Cynthia deaths / losses |
|---|---|---|---|---|---|---|
| PP Ups (`bdsp_formula`) | 404.60 | 389.30 | 4.10 | 60% | 8 | 65 / 8 |
| base PP (`bdsp_pp`) — **the baseline from here** | 404.60 | 391.60 | 4.45 | 65% | 5 | 62 / 6 |

Paired: cleared +0.00 ± 0.21, no-death +2.3 ± 9.9, deaths +0.35 ± 0.74; the party fielded is the
same in 70% of fights (the risk queries' fast battles move with any engine change). Accepted as
a correctness change: nothing moves beyond the noise floor, and the mechanism that does move
(the stall budget) moves the right way. Every table before this one has three PP Ups on both
sides.

## 2026-10-14 — Player held items from the care packages (C1) (`results/rr_items_boxes.*`)

**The data.** `build_rr.progression` now also writes `data/rr/items.json` (the BDSP format
`gauntlet.assign_items` reads): the care-package tab of the official item sheet, stage by stage
in play order, mapped to the trainer index the stage is in hand at — "ENTERING ROUTE 2" is the
first trainer that place is open at (Brendan, 1), "EXITING PEWTER CITY" the next place (Lass
Sally, 7), "… AFTER DEFEATING GRUNT" the first grunt from the previous stage on (the grunts
repeat; stages never go back), "DEFEATING LT. SURGE" his index plus one — with the red (not in
hardcore) and blue (not in MGM) fills dropped and ×10 read as six holders; plus the overworld's
single held items by the existing `when()` (Eviolite from Whitney at 25, Expert Belt from Chuck
at 51, Leftovers at 53, Choice Band at 85; nothing EV-gated: hardcore has no EVs). 27 items;
stages at trainers 1, 5, 8, 17, 20, 33, 91. Priced the way the BDSP items are (`build_rr.HELD`):
type boosters ×1.2, Oran / Sitrus / Berry Juice / Shell Bell / Leftovers / Expert Belt / Life Orb
flat, Eviolite ×1.5 for a mon that still evolves (`item_value`'s new `nfe` flag), Choice items a
locked ×1.5. Not priced: the resist berries (the matrix knows them, the item model does not),
Choice Scarf, Light Ball, Black Sludge.

**The probe.** `rr_team_probe.team_mons`: the final six are copies of the cached mons holding
the items in hand at the boss, one each, greedy by `item_value` (deterministic in the team);
the chooser, stage A / B and the duels stay item-blind. `--no-items` is the old regime. Items
print in the sets and sit in the events (`sets[].item`).

**Paired on the 30 boxes (`rr_pp_boxes`, the same 60 sixes):**

| | Brock boxes: win / ≥50% / deaths per mon / deathless | Misty boxes |
|---|---|---|
| no items (`rr_pp_boxes`) | 74% / 87% / 0.57 / 8% | 41% / 40% / 0.78 / 3% |
| items in hand (`rr_items_boxes`) — **the baseline from here** | 83% / 90% / 0.46 / 13% | 53% / 57% / 0.70 / 4% |

Paired: Brock win **+8.3 ± 3.7**, deaths **−0.106 ± 0.029**, deathless +5.0 ± 2.5; Misty win
**+12.2 ± 4.2**, deaths **−0.084 ± 0.019**. The largest single lever since the cover chooser,
and the first one that moves deaths at both bosses. What is held: at Brock Silver Powder (46 of
180 slots), Mystic Water, Black Glasses, Sharp Beak, Twisted Spoon — the Route 2 package is type
boosters, so every mon with a STAB of a boosted type hits ×1.2 and the paper's 2-hit cells become
real; at Misty Sitrus (30: every box has one holder without a boosted STAB), Silk Scarf (28),
Mystic Water, Silver Powder. Kills by enemy: Brock's all fall (Cacnea 145 → 97, Lileep 128 →
102, Lunatone 161 → 136); at Misty Toxicroak 160 → 122 and Mantine 197 → 158 while Ludicolo
rises 204 → 226 (the fights last into it). Items are the default from here; every RR table
before this one has the player without items.

Next on the item side: the resist berries by the matrix (a Wacan on the Mantine answer is the
kind of pick `item_value` cannot make), and the duels / chooser with items, since a candidate's
cell changes with its item. Then D1 (Brock under sand) and D3, the box milestone.

## 2026-11-23 — Hardcore legality, normal mode, the ROM checklist, D1 revisited (`results/rr_{legal,normal}_boxes.*`, `results/rr_data_diff.log`, `results/rr_d1_brock_*.log`)

**Player legality.** `rr_probe.hardcore()` (under `--legal`) now applies all of the player's
hardcore rules, not just the 82 banned moves: Toxic / Leech Seed only on the 91 listed species
(`gauntlet.SPECIES_MOVES`) and the ability swaps (`player_ability`: Drizzle → Adaptability,
Speed Boost → Infiltrator, Tyranitar → Intimidate, keeps for Spinda / Comfey / Espathra);
`choose_ability` scores each slot as what the player gets. ~14 mons per boss pool change
(Yanma, Hippowdon, Bisharp, ...), Toxic falls off a few sets. 30 boxes, same flags as
`rr_items_boxes`: Brock 83 → 82% / 0.46 → 0.46 d/mon / 13 → 13% deathless, Misty 53 → 53% /
0.70 → 0.69 / 4 → 5%. Inside the noise: the rules were right to add and cost nothing yet.
**Baseline from here: `results/rr_legal_boxes.*`.**

**Normal mode (the easier setting, kept for comparison).** The tracker export has a `default`
(normal mode) boss list next to `hardcore`; `--normal` plays its sets and caps
(`rr_probe.normal_boss`; tracker names resolved against the mod's tables by `tracker_id`:
"Headlon. Rush", "Weakness Pol.", "HP Grass", "Ursaluna-BM"). Normal Brock is 4 mons at L12–14,
no sand; normal Misty 4 mons, no rain. Same pools, items and chooser, no legality:

| setting | Brock win / d/mon / deathless | Misty win / d/mon / deathless |
|---|---|---|
| normal mode | 100% / 0.05 / 77% | 100% / 0.09 / 68% |
| hardcore, full legality | 82% / 0.46 / 13% | 53% / 0.69 / 5% |

The hardcore bosses, not the player's rules, are the whole gap.

**A3 automatic half.** `scripts/rr_data_diff.py`: dump vs sheet for trainer order #0–#34, 98 / 149
shared mons identical; abilities 29 (dump slot 0 vs sheet hidden, e.g. Brock's Varoom Overcoat vs
Filter), levels 24 (dump +2 on Brendan #23 and the optional leaders), items 5, natures 3, moves 4.
Misty / Surge / Erika agree fully. `data/rr/FIDELITY.md` is the ROM checklist (sim value per row,
an empty ROM column).

**D1 revisited.** Same team, 12 fights, current code (`rr_deaths.py BROCK 12 --matchup
--death-weight 6`, ± `--blind-field`): field-aware 12/12, 24 deaths; blind 10/12, 34 deaths. The
2026-10-11 reversal (field 37 vs blind 20) is gone (CFRU AI, formula, items since). The deaths are
attrition: 22 / 24 at < 50% HP, 22 / 24 while already in; Cacnea (Power-Up Punch) 10, Lunatone 8.
That is D2 (death weight, rollout value) and the boost-aware threat, not field awareness.

**Enemy PP maxed (same day).** The official hardcore sheet: "All of the opposing Pokémon's moves
in this sheet are PP maxed" — the 2026-10-14 no-PP-Ups reset had taken them from the bosses too.
In a gen9rr format the bridge now resets only p1 (`sim_bridge.js basePP`; checked: our Rock Tomb
15, Brock's 24). 30 boxes on 10 workers (`results/rr_enemypp_boxes.*`): Brock 81% / 0.47 / 14%,
Misty 53% / 0.72 / 4% — inside the noise of `rr_legal_boxes`. **RR baseline from here:
`results/rr_enemypp_boxes.*`.** Sources checked the same day (data/rr/FIDELITY.md): the official
Drive folder has boss sheets only (both modes); the only per-trainer source is Rudo2204's dumps
(hardcore / normal / easy now in data/rr/). The dump has Varoom Overcoat in both modes, the sheets
Filter: a dump-wide ability reading question for the ROM.

**The official sheet as the truth for boss sets (same day, the user's call).** `rp.boss_team(sheet=True)`:
each mon the official hardcore sheet lists takes its level, item, ability, nature and moves; the
dump keeps IVs, gender and unlisted mons (the user's HTML export of the default-mode sheet matched
our copy: Varoom = Filter). Changes in the fights we run: Brock's Varoom Filter (was Overcoat), Lileep
Storm Drain, Brendan #1's Treecko Unburden, Brendan #23 at 28 (was 30); Misty / Surge / Erika
unchanged. 30 boxes, 10 workers (`results/rr_sheet_boxes.*`): **Brock 81 → 73% / 0.47 → 0.53 d/mon /
14 → 10% deathless**, Misty 53% / 0.72 / 4% (identical: same data, same seeds). Most missed at Brock
is now Ledyba vs Cacnea. **RR baseline from here: `results/rr_sheet_boxes.*`.**

## 2026-11-23 — Radical Red on the Rust engine (poke-engine `rr` build)

The user's call: put the whole RR setting on the Rust engine, Showdown as the fallback, built by
delegated agents (Gemini Neon for the bulk, Opus alongside). The battle stays on Showdown; the
player's search runs on poke-engine wherever the position converts. Details and every number:
`rl/README.md` → "Radical Red on poke-engine".

- **Build**: cargo feature `rr` (gen9 + the gen9rr mod), Python package `poke_engine_rr` next to
  the Gen 8 `poke_engine`. Data: `rl/rr_engine_data.py` writes the 149 changed moves' data into
  `poke-engine/src/rr_data.rs` and lists what is code. Code, on worktree branches merged into the
  fork's `nuzlock`: rr-moves (moves / items / species / conditions; RR freeze = frostbite, hail is
  snow-like), rr-abilities (28 new + 30 changed), rr-mega (66 stone megas, auto on first move),
  rr-conformance, rr-gaps (Rollout / Ice Ball for every gen, two-turn charge, Protosynthesis, Mega
  Rayquaza), rr-ai (the RR AI in Rust).
- **Conformance** (`rl/engine_diff.py --rr`, 400 positions over every hardcore boss): not
  convertible 45% → 0%, agreement on the rest 83% → **97%** (`results/engine_diff_rr.log`).
- **Search** (`rl/engine_search.py`, `rr_team_probe.py --engine 20000`): MCTS with a Nuzlocke death
  weight (60 points per fainted mon of ours; 200 lost nearly every box). 30 boxes Brock / Misty,
  paired against `results/rr_sheet_boxes`: wins −2.8 ± 10.3 / −2.2 ± 9.4 points, deaths per mon
  −0.04 / −0.04: **a tie, in 3 min on 4 workers (baseline 29 min on 10)**
  (`results/rr_engine_boxes30.*`). 96–100% of searched decisions run on the engine.
- **The RR AI in Rust**: 90.3% the same choice as `ai_util/rr_ai.py` (`results/rr_ai_diff.log`);
  searching against it (`--engine-opp ai`) loses badly (Misty −33 points; long passive battles, no
  turn-limit losses: the search exploits the model's errors). Not adopted.
  **Correction (2026-11-24, Fable + Neon review):** there *are* turn-limit losses: ≥ 10 / 30
  Misty boxes have deaths/mon < 1 − win (0 / 30 under minimax), so ≥ half the Misty drop is the
  100-turn force-loss; Brock's −13 has none (ordinary losses). `rl/RR_PLAN.md` plan v2, phase 1.
- **All six gyms, 30 boxes** (`results/rr_engine_gyms30.*`): Brock 66%, Misty 47%, Lt. Surge 44%,
  Erika 22%, Koga 0%, Blaine 0%. The Showdown search also loses all 10 Koga boxes
  (`results/rr_showdown_koga10.*`): a wall for the sampled boxes and the planner, not the engine.
- Fixed on the way: Hidden Power was typeless in every conversion (Showdown keeps the type in
  `hpType`); the calc bot crashed on a typeless ("???") move at Koga.

## 2026-11-23 — RL plan, Phase 0: is Koga / Blaine a team-choice wall? (`results/rr_*ceiling*`)

`rl/RR_PLAN.md` (the RL plan for RR, reviewed by Fable and Opus) gates everything on one question:
does *any* six from the whole pool beat Koga / Blaine? Pool mode, the gyms30 settings (`--legal
--arena --matchup --chooser cover --death-weight 6 --engine 20000`), `--n-search 12`, 10 workers,
~3 min a boss. Best of ~6 candidate sixes on the same seeds, so optimistic.

| setting | Erika | Koga | Blaine |
|---|---|---|---|
| hardcore, today's builder (`rr_pool_ceiling`) | 92% / 0.19 d/mon | **8%** / 0.93 | **0%** / 1.00 |
| normal-mode bosses (`rr_pool_ceiling_normal`) | | 8% / 0.94 | crashed (`heavydboots`: bad item id in `normal_boss`) |
| Rest out of best4 (`rr_ceiling_norest`) | | 8% (same teams) | 0% |
| + 85 EVs in every stat (`rr_ceiling_norest_ev85*`) | | **42%** / 0.63, 25% deathless | **58%** / 0.76 |
| stone / item evolutions in hand, 0 EVs (`rr_ceiling_stones`) | | 0% | 0% |

- **Gate failed:** with legal builds no pool six wins at Koga or Blaine. A better chooser or a value
  net cannot fix that; the RL plan stops at Phase 0 (as written).
- **EVs flip it** (≈ +14 stat points per stat at 68): the fights are decided on stat margins. EVs
  are illegal in hardcore (Minimal Grinding, `data/rr/hardcore_rules.json`), so this is a
  diagnostic, not a fix. The legal levers that move margins are the ones the builder lacks: mega
  stones, Choice Specs / Scarf, Life Orb before trainer 91, Focus Sash, Assault Vest
  (`data/rr/items.json` has 27 items, none of these) and set quality (best4 = 3 attacks + the
  first status move; Rest on 148/180 Koga mons in gyms30, never Sleep Talk; no setup moves unless
  learned by level).
- **Megas and the Choice Scarf (the user's catch: the items sheet has them).** `build_rr.py` kept only
  the items `item_value` could price; it now keeps the mod's mega stones (no stone before the Mega
  Ring, trainer 44; `X/Y` and two-stone cells split) and the Scarf: `items.json` 27 → 79 items,
  ~20 stones in hand by Koga, ~40 by Blaine, Scarf at 83. A stone counts 3.0 for its own species
  (`gauntlet.item_value`), `build_mon` hands it out up front so stages A / B see the mega
  (`gauntlet.mega_stone`), and the bridge megas any stone holder on its first move, both sides
  (`sim_bridge.js` `choose`; the RR AI already did, poke-engine's rr build models it). Tables
  before Sabrina are untouched (no stone before 44). **Ceiling with megas
  (`rr_ceiling_mega`): Koga 8%, Blaine 0%**; the chooser now fields Coalossal-Mega and
  Kangaskhan-Mega, and nothing changes.
- So far the ceiling means "the best of what stages A / B propose" (calc-bot solo scores, then
  greedy): at Blaine (Desolate Land, Ho-Oh, Charizard-Mega X) they propose Cherubi, Deerling,
  Seedot. A searched chooser over a wider candidate set (RR_PLAN.md Phase 1) is now the
  diagnostic, not the payoff.
- Experiments run through `scripts/rr_builder_probe.py` (env switches on top of the probe). Ditto
  and Smeargle are skipped there: poke-env's mirror asserts on Transform / Sketch.

## 2026-11-24 — Plan v3, phase 0: hygiene and instruments (`results/rr_p0_*`)

Plan v3 (`rl/RR_PLAN.md`, top): the final plan after a last pass by Fable and Gemini Neon on the
user's additions (advisor first, a battle-line planner, the Showdown cross-check, set search over
the legal movepool). Phase 0 changes, all in one step:

- **Engine (poke-engine fork, Opus delegate):** seeded MCTS (`seed=` in both bindings; replay
  holds for iteration limits); a Tailwind ≥ 90 turns (the permanent arena Tailwind) never counts
  down and scores as 4 turns; under `rr` a won leaf scores `0.55 + 0.45 · alive/party` (a costly
  win no longer scores below an open fight). Build both packages: `uv sync --reinstall-package
  poke-engine --reinstall-package poke-engine-rr`.
- **Search plumbing:** mega holders keep their `<move>-mega` options (stripped to the move; they
  could only switch before); a spent Phoenix Down is sent as no ability (`spent` in the dump); a
  live mon that must switch after U-turn / Volt Switch / Flip Turn is `force_switch` in
  `convert.to_engine` (the patch in `rr_ai_diff.py` is gone); the engine seed comes from the
  battle's searcher rng, so a seeded battle replays exactly.
- **Logs:** `timed_out` per battle (`capped` per box); `--states` writes `<out>.states.jsonl`
  (every engine decision: state string, each option's visits and value, the pick, the RR AI
  model's predicted options; every battle: both sides' chosen actions, seed, outcome; ~100 KB a
  battle); a `draft` event per cover pick (what it newly answers, gain, runner-up).
- **Chooser:** `--chooser` is honoured in box mode (searched was silently greedy); the lead is the
  best answer to the boss's lead (was alphabetical); one filter (`playable`: dupes, Truant,
  Ditto / Smeargle) in every stage; one mega by the stone, not by "holds an item".
- **Sets (hardcore):** no Rest (best4 never pairs it with Sleep Talk); no Choice item with a status
  move; a mixed set takes +Spe only when it outspeeds a boss mon it would not outspeed neutral,
  else + the stronger attack, − the defence the boss hits less; **abilities are rolled at catch**
  (slot 0 / 1, per run seed and species, kept by slot on evolution; the user: no capsules), not
  chosen per boss; `gauntlet.legal_pool` split out of `encounter_moves`.
- `scripts/sprt.py`: paired SPRT on per-box win differences of two runs (+ deaths/mon interval).

**Gate (Brock / Misty, 30 boxes, the gyms30 settings, n = 6 a box):**

| run | Brock win / d/mon | Misty win / d/mon | capped boxes |
|---|---|---|---|
| before (`rr_engine_boxes30`) | 70% / 0.49 | 51% / 0.68 | (Misty 0 under minimax) |
| phase 0 (`rr_p0_gate`) | 76% / 0.44 | **24%** / 0.87 | 0 / 0 |
| phase 0 + the old per-boss ability choice, Misty (`rr_p0_ablate_ability`) | | 48% / 0.70 | |

- Logging on / off: **60 / 60 boxes identical** (the first exact replay in the project).
- Zero battles lost at the turn cap.
- Brock +6 (3 teams changed by the new rows, 27 kept), within noise of the ±3 target (paired
  ±8 points).
- **Misty −27: abilities fixed at catch cost ~24 of it** (the ablation brings it back to 48%).
  The old builder handed every mon its best slot against the boss (Dry Skin / Water Absorb
  against Misty, Swift Swim in her rain): an illegal advantage if capsules are unobtainable. 24 of
  30 Misty sixes changed, and the changed ones carry the drop (58% → 26%).
- Reading: the earlier RR tables overstate every boss where an ability slot answers the boss.
  Whether an Ability Capsule / Patch is obtainable in hardcore decides which number is real; it
  is now a top question for the item research (Gemini Neon, running).
- **Resolved the same day (user, from the official sheets in `data/rr/sheets`):** raid dens on
  Route 5 / 6 / 10 / 11, Diglett Cave and Rock Tunnel drop Ability Capsules (25%) and need a badge;
  the Saffron City Fanclub sells the Ability Pill (2,500) and the Dream Patch (5,000, hidden
  ability); the Celadon Dept. Store 4F sells every evolution stone. Rule now
  (`rr_probe.ability_gates` / `run_ability`): abilities rolled at catch before trainer #20, the
  best slot 0 / 1 per boss from #20, the hidden slot too from #44; stone evolutions from #33
  (`gauntlet.STONE_GATE` is per game now; RR has no separate place / held-item gates). Brock and
  Misty are before #20, so the phase-0 table above is the hardcore number for them.
- Also from the sheets: Focus Sash (Chuck), Assault Vest (Jasmine) and Choice Specs (Blaine, after
  the fight) are quest rewards, given in hardcore for showing a mon with base stat ≥ 125 in the
  asked stat; Choice Scarf / Specs / Band come in a care package (×10). Item audit pending.
- **Item audit (Gemini Neon's online research, `rl/research/rr_items_teams.md`, checked against
  the official sheets):** `build_rr.py` dropped every EV-quest reward; in hardcore they ask for a
  base stat ≥ 125 (assumed met). Now in `items.json`: Focus Sash (#51, Chuck), Life Orb ×1 from
  Koga (#81; ×6 at #91 as before), Assault Vest (#85, Jasmine), Choice Specs (#91, after Blaine),
  Rocky Helmet (#32); items carry `steps` so a later care package raises the copies (Leftovers and
  Expert Belt ×6 from #75; `gauntlet.copies`). Sash / Vest / Helmet have rough flat values until
  phase 3 prices items by battle; resist berries and Heavy-Duty Boots stay unpriced. Natures: no
  mints in RR, a Nature Changer nurse in every Pokémon Center does it free (same rule as ours).
- **Pool ceiling after phase 0** (`rr_p0_ceiling*`, the same settings as `rr_pool_ceiling`, n = 12,
  best of ~6 sixes): Erika 92 → **100%**; Koga 8 → **8%** (25% before the pool filter let Truant
  Slaking go: 3 / 12 against 1 / 12, within noise); Blaine 0 → **8%**. Hygiene, items, abilities
  and stones do not break the Koga / Blaine wall; phases 1–3 and the reference teams are next.
  Blaine's calc-bot sixes are still Grass / Chlorophyll teams into Desolate Land (stages A / B).

## 2026-11-24 — Plan v3, phase 1: the opponent in the tree (`results/rr_p1_*`)

30 boxes each of Brock, Misty and Erika, the gyms30 settings (`--legal --arena --matchup
--chooser cover --death-weight 6 --engine 20000`), n = 6 a box, paired by box; `scripts/sprt.py`
with δ = 8 points.

| enemy that plays | search's enemy model | Brock | Misty | Erika | capped (Misty) |
|---|---|---|---|---|---|
| Python RR AI (the real one) | minimax (`rr_p0_gate`, `rr_p1_min_erika`) | 76% / 0.44 | 24% / 0.87 | 35% / 0.80 | 0% |
| Python RR AI | the Rust port (`rr_p1_ai`) | 64% / 0.50 | 22% / 0.80 | 44% / 0.67 | 6.7% |
| **Rust port (exact model)** | minimax (`rr_p1_oracle_min`) | 71% / 0.51 | 31% / 0.83 | 27% / 0.85 | 0% |
| **Rust port (exact model)** | the Rust port (`rr_p1_oracle_ai`) | 81% / 0.33 | 33% / 0.66 | 51% / 0.63 | 12.2% |

- **Against the real AI, following the port is a tie** after phase 0: −1.7 ± 5.7 points, deaths
  −0.05 ± 0.05 (SPRT: no difference). Before phase 0 it was −13 to −33.
- **The exact-model oracle says the method works:** with the port as both the enemy and the model,
  following it beats minimax by **+12.2 ± 6.2 points and −0.19 ± 0.05 deaths/mon** (SPRT: H1).
  So the gap against the real AI is the port's error, not the search.
- **On-policy agreement** (`rl/rr_ai_onpolicy.py`, Rust prediction vs the Python AI's actual choice
  at every searched turn): **70.5%** in minimax play, **74.6%** when the search follows the port
  (moves 76%, switches 61%); off-policy (`rr_ai_diff.py`, random play) it was 90%. Real battles
  sit where the port is worse.
- **The turn-cap stall survives the won-terminal floor under the AI model:** 12% of Misty battles
  capped even with the exact model, 0% under minimax. A search problem, not a model problem
  (the plan's fallback: a depth discount).

**Weighted port + mixture** (poke-engine `4855ee8`: `rr_ai_choice_weighted` with the Python AI's
random rolls as probabilities; `perform_mcts_vs_ai(p_model=...)`: per enemy decision follow the
port with probability p, else UCB over every enemy option; `POKE_ENGINE_BRANCH_ALL=1` branches
crits / KO rolls at every depth). Against the real (Python) AI, same 90 boxes as minimax:

| p_model | Brock | Misty | Erika | paired vs minimax (3 bosses) |
|---|---|---|---|---|
| minimax | 76% / 0.44 | 24% / 0.87 | 35% / 0.80 | |
| 1.0 (`rr_p1_mix1.0`) | 59% / 0.54 | 24% / 0.78 (6% capped) | 44% / 0.66 | tie |
| 0.95 | 59% / 0.53 | 30% / 0.77 | 41% / 0.70 | −1.7 ± 6.3, tie |
| **0.8** | **78% / 0.39** | **41% / 0.74** | **51% / 0.64** | **+11.3 ± 5.3, −0.12 d/mon: H1** |
| 0.5 | 80% / 0.37 | 30% / 0.83 | 42% / 0.72 | +5.4 ± 4.3, continue |

- Misty + Erika alone (the plan's gate): **+16.1 ± 6.9 points, −0.15 deaths/mon**, SPRT H1.
- **Fresh boxes and seeds (`--seed 1`, `rr_p1_s1_*`)**, against the optimizer's curse of picking
  0.8 out of four: minimax 78 / 45 / 24% → p = 0.8 82 / 57 / 41%: **+11.1 ± 4.8 points, −0.135 ±
  0.041 deaths/mon**, SPRT H1. Brock +4.
- Turn-cap losses at p = 0.8: 1.1% (seed 0) / 0.6% (seed 1) of Misty battles, 0 elsewhere.
- **Branching at every depth** (`rr_p1_mix0.8_branchall`, same iterations): +1.3 ± 3.6 points,
  −0.02 deaths: no measurable gain; stays off (default).
- On-policy agreement at p = 1.0 with the weighted build: 74.5% (moves 76%, switches 59%): the
  random rolls were not the main gap; the port's choices differ, switches most.
- Reading: following an imperfect model of a scripted enemy most of the time, with a 20% share of
  ordinary worst-case search per enemy decision as insurance, beats both pure minimax and pure
  model-following. The pure model loses because 1 in 4 of its predictions is wrong where it
  matters; the insurance share covers those.

**The model fixed (D4: "p = 0.8 + fix the model"; Opus delegate, poke-engine `9e605fd`).**
`rl/rr_ai_diff.py --onpolicy <events>` replays a probe's box teams with the real searcher and
diffs the Rust AI against the Python AI at every enemy decision (`--rescore` re-runs a saved
file on the current build). The main error was the damage model: the port used poke-engine's
full calc, the Python AI judges with calc_bot's simpler formula (no Gyro Ball power, Acrobatics
doubling, Scarf ...); the port now copies the Python AI's view of damage, speed, immunities and
the player's HP (Showdown's public percentage), and returns every tied switch-in. Two bugs on
our side, fixed by the lead: the Python AI read its *own* Hidden Power as the mod's Water entry
(`sim._typed_hidden_power` now types both sides; this changes the enemy, so earlier tables are
not comparable), and `convert` dropped Showdown's `stall` volatile (now `side_conditions.protect
= 1`). **On-policy agreement 70–83% → 98.0%** (moves 98.6%, voluntary switches 90.8%, forced
98.7%); off-policy 90 → 97%.

| run (new enemy, 30 boxes each) | Brock | Misty | Erika | vs minimax |
|---|---|---|---|---|
| minimax (`rr_p1b_min`) | 79% / 0.42 | 21% / 0.90 | 36% / 0.80 | |
| **p = 0.8** | **87% / 0.30** | **48% / 0.66** | **79% / 0.37** | **+26.3 ± 6.2, −0.26 d/mon** |
| p = 0.95 | 79% / 0.35 | 38% / 0.67 (7% capped) | 81% / 0.36 | +20.4 ± 6.9 |
| p = 1.0 | 78% / 0.35 | 30% / 0.71 (10% capped) | 79% / 0.36 | +17.0 ± 6.9 |
| fresh seeds (`--seed 2`): minimax → p = 0.8 | 62 → 83% | 36 → 60% | 23 → 69% | **+30.2 ± 7.0, −0.28 d/mon**, 0 capped |

- p = 0.8 stays (0.95 / 1.0 are −6 / −9 points against it, paired; their loss is Misty battles
  stalling into the turn cap). The 20% insurance share keeps the search from stalling.
- **Pool ceiling with the new search** (`rr_p1b_ceiling`, n = 12): Erika 100%, **Koga 8 → 17%,
  Blaine 8 → 25%**. The wall loosens, not yet broken.

## 2026-11-24 — Plan v3, phase 1.5: the battle line (`rl/line.py`), first numbers

The line is our search's policy written out as a tree against the Rust AI model: at our nodes
the action MCTS-vs-AI (p = 0.8, fixed seed) picks, at chance nodes every enemy choice the model
weighs × every outcome `generate_instructions` lists, with exact probabilities. Merged by state
string (a graph); cycles and branches below `eps` reach are unexplored and valued by playouts
from them under the same policy (Monte Carlo policy evaluation, drawn by reach). Exact-only
expansion does not reach the end of a 6v6 fight: at eps = 5e-5 still 94% unexplored (1k
iterations); with the full 20k-iteration search the policy ends fights sooner and at eps = 0.002
**42% of the probability is exact** (1,034 nodes + 200 playouts, 55 s).

**Erika, the cover six** (armarouge, beheeyem, swalot, noivern, corviknight, slowbro-galar):

| | P(win) | P(deathless) | E[deaths] | P(lead armarouge dies) |
|---|---|---|---|---|
| line (exact 42% + playouts) | 98.6% | 35.4% | 2.00 | 56% |
| engine-only playouts from the root | 98.5% | 39.0% | 1.82 | 53% |
| Showdown, 100 battles, same fixed-seed policy (`--check`) | 88 ± 6% | 56 ± 10% | 1.20 | 17% |
| Showdown with the Rust AI as enemy | 85 ± 7% | 52 ± 10% | 1.43 | |

- **Not calibrated yet:** the engine's whole-fight simulation is more optimistic about winning
  and more pessimistic about deaths than Showdown. Not the AI model (the Rust enemy gives the
  same gap); turn 1 agrees exactly (engine_diff: tv 0.002); the tree code agrees with plain
  engine playouts. Part of it is that small HP differences flip near-tied search choices (turn 2:
  slowbro 4744 vs swalot 4455 visits), so "the same policy" is not reproduced exactly on
  Showdown. The per-turn cross-check (`engine_diff --from-states`, Opus delegate) is to find the rest.

**The gap, found** (Opus delegate; `engine_diff.py --from-states` replays logged battles from their
seed and actions and compares every turn: engine outcome table vs 100 Showdown forks; classes
engine / AI / converter / luck / search, with a 5% control). Erika's flags were 31 / 34 engine:
- Synchronoise hit any target in the engine (Showdown: only one sharing a type with the user):
  the search used it and lost; Quick Draw never moved first (Showdown 30%); Roost skipped the
  Grassy Terrain heal; `last_used_move` was stale outside the MCTS (the AI model's first-turn
  rules); megas were recomputed without nature / EVs (`to_engine`, the bridge). poke-engine
  `624921c`, `rl/convert.py`, `ai_util/sim_bridge.js`.
- The line searched with death weight 0 (POKE_ENGINE_DEATH unset), the battler with 60: a
  different policy (35%-HP Armarouge into Mega Sceptile). `Line` now sets 60.

| Erika cover six, 200 Showdown battles | P(win) | P(deathless) | E[deaths] |
|---|---|---|---|
| line, before | 98.6% | 35.4% | 2.00 |
| line, after (`results/rr_line_erika.json`) | 96.0% | 38.8% | 1.01 |
| Showdown, after | 97.0 ± 2.4% | 47.5 ± 6.9% | 0.89 |

P(win) and deaths agree; P(deathless) is ~9 points pessimistic (Galarian Slowbro 47% vs 38%).
Delegate's runs (100 battles): Misty box 1 line 20% / 0 / 5.4 vs 11 ± 6% / 0 / 5.6; Brock box 2
98% / 36% / 1.0 vs 96 ± 4% / 24 ± 8% / 1.3. Residual engine gap: a damage roll is branched only
when it KOs directly, not when it decides whether poison / chip damage finishes a mon later; and
Showdown forks share a sleep counter, so the cross-check over-reports sleep turns as engine bugs.
The engine fixes change the battler too (Erika's Showdown wins 88 → 97%): the phase-1 tables
predate them.
- Views (Opus delegate): `ui/` Draft, Trace (from states.jsonl), Line, "what it uses".

## 2026-11-24 — Plan v3, phase 2: anchors and blame (`results/rr_p2_*`)

**Reference teams** (`data/rr/reference_teams.json`, players' Koga / Blaine teams from
`rl/research/rr_items_teams.md`; what the sources state, the rest from our set builder:
`rl/line.py --reference K2 --no-line --check 48`), our battler (20k, p = 0.8) vs the Python AI:

| team | boss | win | deathless | E[deaths] |
|---|---|---|---|---|
| K1 Crappo 4.1 (deathless for him) | Koga | 10 ± 9% | 4% | 5.5 |
| K2 Stephin Sunny 3.02 | Koga | 35 ± 14% | 8% | 4.3 |
| K3 Apecio 3.02 (5 mons) | Koga | 0% | 0% | 5.0 |
| K4 TheFetCet 3.1 (no moves given) | Koga | 38 ± 14% | 2% | 4.4 |
| B1 Crappo 4.1 (1 planned death) | Blaine | 46 ± 14% | 0% | 5.1 |
| **B3 Mega Sceptile + Cloud Nine Golduck (2 mons)** | Blaine | **100%** | **100%** | 0 |
| our best pool six (`rr_p1b_ceiling`, n = 12) | Koga / Blaine | 17% / 25% | 0% | |

- Blaine has a team-choice answer our chooser never finds: Cloud Nine (Golduck) cancels the
  permanent Desolate Land; the AI keeps clicking Solar Beam, which now charges a turn. The
  players' guides use exactly this. A builder wall at Blaine.
- Koga: the players' teams do 2× our best six on win (35–38%) but are nearly never deathless with
  our battler and filled-in sets (the players played planned lines with their own sets).

**Mirrors** (`scripts/rr_mirror.py --n 48`, the boss's six on both sides; now `--p1 search|calc|ai`,
the search = the D4 default), `results/rr_p2_mirrors.log`:

| boss | our search vs the AI | AI vs AI (baseline) | gain |
|---|---|---|---|
| Erika (arena) | 100% / 0.10 d/mon | 50% / 0.81 | +50 |
| Koga (no arena) | 77% / 0.54 | 60% / 0.81 | +17 |
| Blaine (arena) | 98% / 0.33 | 42% / 0.85 | +56 |

The battler is sound (≥ 75% everywhere; Koga just under the +20 margin).

**Iteration ladder** (reference teams, 48 battles, `results/rr_p2_ladder.log`): K2 at Koga 23 → 35
→ **54%** (2k / 20k / 200k), deaths 4.8 → 4.3 → 3.3; B1 at Blaine 21 → 46 → **56%**, deaths 5.6
→ 5.1 → 4.7. Search-bound: 20k → 200k still +19 / +10 points. Deathless stays 0–10% at every rung.

**Reading for D6:** (1) team choice: Blaine has a 100% deathless 2-mon answer (Cloud Nine) and
the Koga reference sixes double our best: a builder wall; (2) search budget / evaluation: more
iterations keep paying; (3) deathless play: even good teams with 10× search lose 3–5 mons, while
the players were deathless: their exact sets and planned lines, or fidelity (unknown).

## 2026-11-24 — Plan v3, phase 3: team and set search (`scripts/rr_search.py`, `results/rr_p3_*`)

Opus delegate, to the user's D7 (attack, utility and item swaps battle-tested in every halving
round) and D8 (score = progress − 0.5 · deaths/mon while a boss's best wins < 20%, then U).
Candidates: cover, `searched_cands`, the best six per archetype (negate weather / terrain,
speed, pivot, mega, priority, bulk), random sixes; sets from a per-boss shortlist (top-2
attacks per boss mon + one per utility class) by matchup-row gain; **sequential halving** on
paired seeds with mutations (box swap, lead, move, item) joining every round; archive of every
evaluated row (`<out>.archive.jsonl`, phase 4's data) and a plan sheet (`<out>.plan.{txt,json}`).
Rule change to review: `rr_probe.choose_ability` now scores Cloud Nine / Air Lock under an
arena weather (it picked Swift Swim for Golduck before).

| gate | before | phase 3 |
|---|---|---|
| Erika, 30 boxes, fast budget (~120 battles a box), paired | cover 78% / 0.40 d/mon | **89% / 0.28** (+11.1 ± 8.2, SPRT δ 10: H1) |
| Koga, pool, wide budget (best six, 48 fresh seeds) | 17% (best of ~6, n = 12) | **100%, deathless 90 ± 9%**, 0.03 d/mon |
| Blaine, pool, wide budget | 25% | **92 ± 8%, deathless 10 ± 9%** |

- **Lead's independent replay** (`rl/line.py check`, other seeds, 48 each): Koga 100% / 88% deathless
  / 0.17 deaths; Blaine 96% / 12% / 1.40. Confirmed.
- Koga six: Bronzong (lead), Annihilape, Chesnaught, Incineroar (Intimidate), Mr. Mime (RR level-up
  Psycho Boost / Sparkly Swirl), Grimmsnarl; every one carries Earthquake or a Steel / Psychic
  answer to Koga's Poison types. Blaine six: Scream Tail, Charizard @ Charizardite Y, Corviknight,
  Dondozo, Iron Hands, Galarian Rapidash @ Choice Band; Ho-Oh causes 45 of 48 battles' deaths.
- **Cloud Nine at Blaine was found** (the `negate` archetype: the same six with Golduck for
  Charizard, best in round 1, 90% win and 21–27% deathless) but cut in round 2 at 0.808 vs ~0.81–
  0.83: under U = win · (1 − 0.5 · deaths/6) a higher deathless rate does not outweigh a few
  points of win rate.
- The wall was team and set choice: the old chooser never fielded these sixes or sets. Cost:
  ~12k battles, ~28 min on 10 workers.
- Approximations (delegate): halving scores a round only on its own seeds; mutants take 1/3 of the
  slots; `--my-box` mons are not evolved; pool mode includes every starter.

**Realistic boxes** (30 sampled boxes each, fast budget ~120 battles a box, paired by box with the
cover chooser; `results/rr_p3b_*`):

| boss | cover | team search | boxes with a ≥ 50% six | deathless |
|---|---|---|---|---|
| Koga | 5% / 0.96 d/mon | **21% / 0.85** | 1 → 6 / 30 | 3% → 2% |
| Blaine | 28% / 0.83 | **69% / 0.53** | 10 → 25 / 30 | 1% → 6% |

Paired: **+28.3 ± 7.1 points, −0.20 ± 0.06 deaths/mon** (SPRT δ 10: H1). Blaine is now mostly
winnable from a typical box; Koga only from 1 box in 5 at this budget, and deathless stays rare
at both. The pool result (Koga 100%) needs species a typical box lacks; the advisor's wide budget
(4,000 battles) on one box should land between the two.

**Full budget on real-sized Koga boxes** (`rr_search.py --boxes 6 --battles 4000`, stopped by the
2 h limit after 5 boxes; finals replayed 48× with `line.check`, `results/rr_p3c_*`): win 6 / 38 /
40 / 12 / 2% on boxes 0 / 1 / 3 / 4 / 5 vs the fast budget's 0 / 17 / 50 / 33 / 17% (n = 6): no
clear gain from 30× the battles. Koga from a typical box is limited by the box's species.
**Deaths in won fights** (the user's question): mostly 0–1 (box 3: 58% of wins deathless, 32%
one death; box 1: 33% / 28%); losses are near-total wipes (5–6 of 6), so mean deaths track the
loss rate. A death's cost depends on who dies (the user's D14: a run-level value per mon).

## 2026-11-24 — Plan v3, phase 4: fitted leaf evaluation (Texel), first fit

D9 = battle outcomes, D10 = linear first (the user). Data (`logs/p4_data.*`, not in results: 459
MB): 16 mandatory singles bosses × 60 sampled boxes × 6 battles, `--states`, the D4 battler; every
4th boss held out (#18 Misty, #34 Erika, #77 Lt. Surge rematch, #95 Giovanni; #75 Brock rematch
was swapped out: its Ditto trips poke-env's Transform assertion, as did Ditto in our own pool
in box mode's solo stage, now filtered by `playable`). #74 / #77 / #93 are near-trivial fights
(6–12 turns, ~100% wins: weak dumped teams). Engine (Opus delegate, poke-engine `150b59d`): 43
features (`evaluate_features`, ours / theirs: HP, alive, items, statuses, hazards, boosts,
screens, Tailwind ...; our deaths; damage out / in of the actives, we move first, we KO first,
answers, unanswered), 2.8 µs a call; `POKE_ENGINE_EVAL_WEIGHTS=<file>` makes an open leaf
sigmoid(bias + w · f) and a finished battle U.

`rl/fit_eval.py` (soft-label logistic regression, L-BFGS, battle-level split; label = the battle's
U): 129,298 positions from 5,748 battles.

| model | train | validation | held-out bosses |
|---|---|---|---|
| constant (mean U) | 0.692 | 0.691 | 0.689 |
| **linear (43 features)** | 0.422 | **0.444** | **0.535** |
| 1×32 net (probe) | 0.377 | 0.419 | 0.575 |

- The net fits seen bosses better and generalises worse (held-out +7.5%): by D10, linear stays.
- Largest weights (standardised): our HP +1.49, their alive −0.87, their HP −0.71, their
  Tailwind −0.68 (only Koga's arena has it: a boss marker), our deaths −0.43, our alive +0.43;
  their paralysis −0.30 / burn −0.22, we move first +0.17, we KO first +0.13.
- Features constant in training (enemy hazards, screens, Substitute ...) got huge raw weights
  (sd ≈ 1e-6); now zeroed (`fit_eval.py`), so an unseen feature cannot explode a leaf.
- Held-out bosses are much harder to predict (0.535 vs 0.444): generalisation across bosses is
  the open question, as CLAUDE.md warns.

**Play test** (30 boxes each, `--seed 3`, paired; `results/rr_p4_{base,v1}.*`): hand evaluation →
fitted V1: Brock 91 → 83%, Misty 49 → 52%, Erika 75 → 62%, Koga 9 → 14%, Blaine 24 → 25%;
**−2.4 ± 3.7 points, +0.055 ± 0.029 deaths/mon** (SPRT δ 5: no difference; deaths worse).
**Gate failed**: a better predictor of outcomes is not a better guide for the search. Likely
reasons: (1) the label U charges a death 0.5 / 6 of a win, while the hand evaluation charges 60
points per death (POKE_ENGINE_DEATH), so the fitted search trades mons more readily (deaths up);
(2) Monte Carlo labels say who is ahead under the current policy, not which move is better
(positions differ mostly by features the search cannot change in one turn); (3) the floor /
terminal change and the leaf change are tested together.

**Hybrid test** (the user's D11 choice; same boxes and seeds; poke-engine: `POKE_ENGINE_EVAL_RELATIVE=1`
scores an open leaf sigmoid(logit − root logit), centred like the hand evaluation):

| variant | Brock | Misty | Erika | Koga | Blaine | vs hand (paired) |
|---|---|---|---|---|---|---|
| hand evaluation | 91% / 0.27 | 49% / 0.65 | 75% / 0.43 | 9% / 0.94 | 24% / 0.84 | |
| fitted, absolute (V1) | 83 / 0.41 | 52 / 0.66 | 62 / 0.53 | 14 / 0.93 | 25 / 0.88 | −2.4 ± 3.7, +0.055 d/mon |
| H1: fitted, relative | 88 / 0.36 | 51 / 0.68 | 62 / 0.60 | 12 / 0.93 | 27 / 0.88 | −1.8 ± 3.7, +0.061 |
| H2: H1 + the hand's death penalty | 76 / 0.42 | 34 / 0.75 | 57 / 0.56 | 4 / 0.97 | 23 / 0.87 | −10.7 ± 3.9, +0.084 |

- Neither saturation (H1) nor the death penalty (H2: worse, the search turns passive) explains it:
  the outcome-fitted features do not guide the search better than the hand evaluation. A model
  that predicts *who is winning* well (log-loss 0.44 vs 0.69) need not rank *moves* well: Monte
  Carlo labels mostly reflect the matchup and the policy, which the search cannot change by its
  choice at the leaf.

**Analysis** (the user asked for it; `results/rr_p4_diag*`, page https://claude.ai/artifact/DQrMhxF4RKjAYBpUekxniN):
- What the fit learned: a per-boss constant alone reaches log-loss 0.506 (one constant 0.692,
  linear 0.426): ~70% of the gain is boss identity. The fitted logit varies 4.5 between battles,
  1.4 within one; the search only compares positions within a battle.
- Calibration: exact on training bosses; held-out under-confident low (0.19 → 0.27), over-confident
  high (0.79 → 0.72).
- Search behaviour (192 positions, 20k, same seed): same pick in 56%; root value spread 0.189
  (hand) vs 0.096 (fitted); visits on the pick 48% vs 29%; entropy 1.43 vs 1.84; switches 38% vs
  30%. The UCB constant was tuned for the hand evaluation's scale: the fitted search explores more,
  commits less.
- Move quality where they disagree (60 positions, 40 playouts per pick, same continuation): fitted
  − hand U +0.000 ± 0.021 (15 better, 19 worse). The worst fitted picks stay in where a switch wins
  (Roost 0.15 vs switch Donphan 0.79); bench-answer features got ~0 weight (answers +0.02,
  unanswered −0.23 standardised).
- Options (for the user, next session): A rescale the logits ~2× / retune UCB (cheap test of the
  indecision mechanism); B fit the within-battle signal (the battle's start value as an offset or a
  TD-style difference target, bench-answer-after-switch features); C deep-search targets; D stop.

**Option A (2026-11-25): rescale logits 2× (`data/rr/eval_weights_v1_2x.txt`, `POKE_ENGINE_EVAL_RELATIVE=1`,
`results/rr_p4_a2.*`).**
- 192-position scale sweep: `abs_2x` still saturates (`spread 0.108, top_share 29.5%, entropy 1.80`),
  whereas `rel_1x` (H1) already recovered most of the spread (`0.175, 46.0%, 1.47`) and `rel_2x` matches/edges
  past hand (`0.212, 52.3%, 1.32`, agree with hand `64.1%`); `rel_3x`/`rel_4x` plateau (`0.223`/`0.225`).
  Switch rate stays stuck at `31–32%` vs hand `38.0%` at every scale.
- 30-box play test (same boxes & seed 3): Brock **92% / 0.31**, Misty **53% / 0.68**, Erika **62% / 0.55**,
  Koga **18% / 0.89**, Blaine **31% / 0.86**.
  - vs H1 (`rel_1x`): **+3.2 ± 3.2 pts, −0.031 ± 0.020 d/mon** (indecision hypothesis confirmed).
  - vs hand (`base`): **+1.4 ± 4.4 pts, +0.030 ± 0.031 d/mon** (SPRT δ 5: LLR −1.07, continue).
  - Sharp train vs held-out split: **seen bosses (Brock / Koga / Blaine) +5.6 ± 5.1 pts, +0.002 d/mon**
    (Koga 9 → 18%, Blaine 24 → 31%); **held-out (Misty / Erika) −4.7 ± 7.6 pts, +0.071 ± 0.055 d/mon**,
    entirely from Erika (−13.3 ± 12.3 pts, stuck at 62% across V1 / H1 / A2).
- Reading: rescaling recovered ~3 pts and halved the death regression, showing the fitted signal is
  real once MCTS isn't indecisive — but inverted/confounded between-battle weights (`our_items −0.33`,
  `our_boost_off −0.45`, `their_par −0.96`, `answers +0.01`) still hurt switching and held-out Erika.
  Strong case to test **Option B** (within-battle target + bench features) next.

**Option B (2026-11-25): within-matchup fixed-effects fit (`rl/fit_eval.py --within-team --scale 2.0`,
`POKE_ENGINE_EVAL_RELATIVE=1`, `results/rr_p4_{b1,b2}.*`).**
- **B1** (`data/rr/eval_weights_b1.txt`, existing 43 features, fitting `α[boss, team] + w · (f(s) − f(s₀))`):
  - Within-team log-loss: train `0.3594` (vs group-const `0.4118`), val `0.3894` (`0.4276`), **held-out
    bosses `0.4148` vs `0.4721` (−12.1%, r = 0.376)**. Confounded signs fixed (`our_items −0.33 → +0.16`,
    `their_brn −0.33 → +0.08`, `their_tailwind −0.57 → 0.00`); 192-pos switch rate `31.8 → 35.9%` (hand 38.0%).
  - 30-box play test (seed 3): Brock **93% / 0.27**, Misty **63% / 0.61**, Erika **74% / 0.48**,
    Koga **16% / 0.91**, Blaine **31% / 0.85**.
  - vs hand (`base`): **+5.9 ± 3.8 pts, −0.006 ± 0.030 d/mon** (`SPRT δ 5%: LLR +4.39 → H1`, gate passed);
    seen +5.4 ± 5.1, held-out +6.7 ± 6.0 (Misty +13.9, Erika recovered 62 → 74%).
- **B2** (`data/rr/eval_weights_b2.txt`, 46 features in `poke-engine/src/genx/evaluate.rs`): fixed
  `answers` / `unanswered` so the active mon is judged by direct 1v1 (`ko_first`) while bench mons are
  judged after eating a switch-in hit (previously switching in a counter and taking 35% chip could flip
  `answers` 1 → 0 and `unanswered` 0 → 1), and added `safest_switch_in`, `bench_answers`, `sweep_exposure`.
  - Within-team held-out log-loss improved further: **0.4109 vs 0.4721 (−13.0%, r = 0.386)**;
    `unanswered` strengthened (−0.56 → −0.66), `safest_switch_in` −0.21, `answers` −0.10 → −0.02.
  - 30-box play test (seed 3): Brock **95% / 0.24**, Misty **64% / 0.59**, Erika **75% / 0.47**,
    Koga **19% / 0.89**, Blaine **35% / 0.83** (beat B1 on all 5 bosses).
  - vs hand (`base`): **+8.0 ± 3.7 pts, −0.023 ± 0.028 d/mon** (`SPRT δ 5%: LLR +7.75 → H1`);
    seen **+8.3 ± 4.8 pts** (−0.030 d/mon), held-out **+7.5 ± 5.8 pts** (−0.011 d/mon) — virtually zero
    generalisation gap.
  - vs B1: +2.1 ± 3.2 pts, −0.017 ± 0.023 d/mon.
- **Mirrors check for B2** (`results/rr_p4_b2_mirrors.log`, n = 48): Erika (arena) **98% / 0.12** (hand
  100% / 0.10), Koga (no arena) **69% / 0.58** (hand 77% / 0.54; within n = 48 noise, 33 vs 37 of 48),
  Blaine (arena) **100% / 0.41** (hand 98% / 0.33).
- **Round 2 refit (2026-11-25, policy iteration under B2):** collected 2,874 on-policy B2 battles
  (66,865 positions) across the 12 non-trivial bosses (`--boxes 40 --seed 10`, `logs/p4_r2_data.*`;
  same held-out `#18` Misty, `#34` Erika). Fitted both **R2-only** (`data/rr/eval_weights_r2_only.txt`)
  and **R1 + R2 combined** (`data/rr/eval_weights_r2.txt`, 196,163 positions, 8,622 battles; held-out
  within-team log-loss `0.4370` vs `0.4979`, −12.2%, r = 0.401), play-tested on seed 3 (`results/rr_p4_r2*`):
  - **R2-only:** Brock 88% / 0.36, Misty 62% / 0.62, Erika 68% / 0.53, Koga 14% / 0.93, Blaine 30% / 0.86
    (**+2.8 ± 3.6 pts** vs hand, **−5.2 ± 3.9 pts** vs B2). Classic policy-iteration forgetting: throwing
    away Round 1 off-policy coverage let rare status features (`their_par +0.99`, `their_confusion +1.13`)
    over-inflate and bait MCTS into status moves.
  - **R1 + R2 combined (replay buffer):** Brock **95% / 0.26**, Misty **72% / 0.58**, Erika **78% / 0.45**,
    Koga **22% / 0.89**, Blaine **36% / 0.82** — beats hand on all 5 bosses!
    - vs hand (`base`): **+11.0 ± 3.5 pts, −0.028 ± 0.028 d/mon** (`SPRT δ 5%: LLR +13.17 → H1`);
      seen **+9.6 ± 4.4 pts** (−0.027 d/mon), held-out **+13.1 ± 5.8 pts** (−0.029 d/mon; Misty +22.8, Erika +3.3).
    - vs B2: **+3.0 ± 3.2 pts, −0.005 ± 0.022 d/mon**.
- **Round 3 refit (2026-11-25, ceiling check):** collected 2,874 more on-policy battles under R2
  (`--boxes 40 --seed 20`, `logs/p4_r3_data.*`), refitted on `R1 + R2 + R3` (260,576 positions, 11,496
  battles → `data/rr/eval_weights_r3.txt`, held-out within-team log-loss `0.4472` vs `0.5066`, r = 0.412).
  Play test on seed 3 (`results/rr_p4_r3.*`): Brock 96% / 0.24, Misty 64% / 0.61, Erika 78% / 0.45,
  Koga 18% / 0.90, Blaine 37% / 0.81 (**+8.9 ± 3.5 pts**, −0.026 d/mon vs hand, LLR +9.91 → H1;
  **−2.1 ± 2.7 pts vs R2**, SPRT H0: LLR −11.85). Linear ceiling reached; **R2 (`eval_weights_r2.txt`)
  stays the winner.**
- **Fresh-seed confirmation (`--seed 4`, untouched, `results/rr_p4_fresh_{hand,r2}.*`):**
  - Hand: Brock 87% / 0.27, Misty 46% / 0.66, Erika 77% / 0.42, Koga 2% / 0.99, Blaine 33% / 0.81.
  - **R2 (`eval_weights_r2.txt`):** Brock **96% / 0.27**, Misty **65% / 0.63**, Erika **84% / 0.41**,
    Koga **11% / 0.93**, Blaine **44% / 0.80** (beat hand on win rate AND deaths/mon on all 5 bosses!).
  - Paired vs hand: **+11.0 ± 4.1 pts, −0.022 ± 0.029 d/mon** (`SPRT δ 5%: LLR +9.61 → H1`);
    seen **+9.6 ± 4.7 pts**, held-out **+13.1 ± 7.6 pts** (reproduced seed 3 to the tenth of a point!).
- **Mirrors for R2** (`results/rr_p4_r2_mirrors.log`, n = 48): Erika (arena) **98% / 0.07** (hand
  100% / 0.10), Koga (no arena) **67% / 0.58** (hand 77% / 0.54; within noise), Blaine (arena)
  **100% / 0.36** (hand 98% / 0.33).
- **Erika line check (100 Showdown battles, `rr_p4_base` box 0):** hand Showdown 87.0 ± 6.6% win (line
  81.2%), 26.0 ± 8.6% deathless (line 21.5%), exact mass 0.5%; **R2** Showdown **93.0 ± 5.0% win** (line
  96.8%), 11.0 ± 6.1% deathless (line 13.9%), exact mass **18.0%**, 0% capped. Note: R2 trades a mon
  more readily to secure wins under U = win · (1 − 0.5 · deaths/6) (higher win rate, lower overall
  deaths/mon, slightly fewer deathless sweeps on already-won fights).
- **Adopted as default:** wired `data/rr/eval_weights_r2.txt` + `POKE_ENGINE_EVAL_RELATIVE=1` as default
  in `rl/engine_search.py` and `rl/line.py` (`POKE_ENGINE_EVAL_WEIGHTS=""` reverts to hand). Phase 4 done.

## 2026-11-25 — Plan v3, phase 6 (first pass): encounter routing, Celadon eggs & D14 (`scripts/rr_route.py`, `results/rr_p6_route*`)

Implemented `scripts/rr_route.py` (`tests/test_rr_route.py` 3/3):
- Models the previously unused `tracker_encounters.json` choice pools: **Celadon Mansion Starter Egg
  Vendor** (`eggVendor`, pre-Erika `#33`: Green/Blue/Red/Yellow Shards), Mt. Moon & Celadon shard
  fossils, Saffron Dojo (Tauros-Blaze vs Aqua), plus all 27 starters with Mega forms once their Mega
  Stone unlocks (`items.json`: Swampertite `#44`, Kanto starter megas `#53`, Blazikenite `#81`, Sceptilite `#102`).
- Starter cover ranking (Brock → Blaine, with Mega forms): **#1 Mudkip** (6.98, Swampertite `#44`),
  **#2 Bulbasaur** (6.67, Venusaurite `#53`), **#3 Fuecoco** (6.52), **#4 Fennekin** (6.25), **#5 Chikorita** (5.97);
  Celadon Eggs pre-Erika: **Red Shard** (E[gain] 4.99) > **Green** (4.67) > **Blue** (3.46) > **Yellow** (0.00).
- **D14 per-mon death costs** (`death_costs()`): ranks each mon in a box by drop in future-boss value when
  removed, naming the exact upcoming threats lost, normalised around `0.5/6`.

**Phase 6 gate test (30 boxes × 5 bosses, `--seed 5`, default R2 battler, `cover_team` chooser):**
- Pass 1 (binary `2·safe + race + 0.1·edge` cover, `results/rr_p6_route.log`): routed +2.4 ± 5.4% overall
  (Brock +2.2%, **Misty +10.6%** `72.8 → 83.3%`, Erika −8.9%, Koga +5.0% / +6.7% w/ egg, Blaine +3.3%).
  Two bugs found: binary score saturated at `3.1` on 38–74 mon boxes (alphabetical tie-breaking) and egg roll
  scrambled wild RNG.
- Pass 2 (unsaturated continuous `cell_score` + 3-deep cover + strictly paired per-location RNG, `results/rr_p6_route2.log`):
  - `routed` vs `random`: Brock +2.2%, **Misty +6.1%** (`64.4 → 70.6%`), Erika −7.2%, Koga −3.9%, Blaine −7.8%
    (**−2.1 ± 5.4% overall**, SPRT δ 10%: H0, LLR −9.37).
  - `routed+egg` vs `routed` (strictly identical wild boxes + 1 Celadon starter): Erika **+3.9%** (+6.5% on the
    18 boxes where the team changed), Koga −1.6%, Blaine −3.4% (−9.1% on 11 changed boxes because `cover_team`
    dropped `Floette-Eternal` to field `Bulbasaur`!).
  - `routed+full` (Bulbasaur starter + routed methods + egg + choice gifts) vs `random`: **Misty +12.8%**
    (`64.4 → 77.2%`, −0.102 d/mon), Brock +2.2% (−0.048 d/mon), Erika −7.8%, Koga −1.7%, Blaine 0.0% (+1.1 ± 5.5% overall).
- **Root cause found (Optimizer's Curse on the paper matrix):** correlation between static 1v1 paper score
  vs real R2/R3 battle win-rate lift across species:
  - Misty: paper **`+0.455`**, solo Showdown battle `+0.019` → paper routing worked (+6 to +13 pts).
  - Erika: paper `+0.071`, **solo battle `+0.274`**.
  - Koga: paper **`−0.052`**, **solo battle `+0.135`** (paper router chased `dondozo` 24/30 & `wailmer` 16/30; real winners are `bronzor +36%`, `dewpider +15%`, `heracross +14%`).
  - Blaine: paper **`−0.182`**, **solo battle `+0.236`** (paper router chased `slowpokegalar` & `exeggcute`).
- **Takeaway:** Static paper-matrix routing works early (Misty) but fails mid/late because the paper matrix
  correlates negatively with real 6v6 outcomes at Koga/Blaine — routing needs a battle-grounded scorer
  (`tp.solo_stage` + `rr_search`, or Phase 5's T1 learned team scorer).

### 2026-11-25 — Phase 6 Step A + Step B: branched evolutions, Nuzlocke gift/trade rules, duel-grounded routing, D14 in `rr_search`, ability audit

- **Step A (evolution, gift/trade, and ID fidelity):**
  - `scripts/build_rr.py` + rebuilt `data/rr/*.json`: included gifts/trades/fossils/eggVendor/statics in dex build
    (`kubfu` → `urshifu`/`urshifurapidstrike` now exists), normalized tracker IDs (`taurosblaze`/`taurosaqua` →
    `taurospaldeablaze`/`taurospaldeaaqua`, `ursalunabm` → `ursalunabloodmoon`, `flabébé` → `flabebe`,
    `anycappikachu` → `pikachuoriginal`), fixed Burmy cloak evos (`burmytrash` → `wormadamtrash`/`mothim`), tagged
    `order.json["gifts"]` with `kind` (`gift`/`trade`), `give`, and Saffron Dojo `choice_options`.
  - Branched evolutions (`gauntlet.evolve_options`, `expand_branches`, `evo_line`, `base>target` keys,
    `cover(..., limits, group_fn)`): all 35 branched species (Eevee ×1–2, Charcadet, Kirlia, Slowpoke, Applin,
    Kubfu, Hisuian starters, etc.) now expand across all legal branches with strict per-caught-mon slot limits
    (allowing 2 distinct Eeveelutions from Route 5 + Celadon 4F Eevees).
  - User's Nuzlocke rules wired into `rr_probe.py` & `rr_route.py`: free starter choice out of 27, 1 Celadon
    Starter Egg total (choose shard pool), accepted story gifts, in-game trades disabled unless the box actually
    owns a mon in `evo_line(give)` (and swapping out the traded mon if executed). Evolved names (`caught→evolved`)
    printed everywhere.
- **Step B (duel-grounded routing + D14 death costs in `rr_search.py`):**
  - Validated correlation against logged R2/R3/P3 real-battle win rates: Showdown 1v1 `duel(n=2)` beat paper on
    Misty (`+0.306` vs `+0.204`), Erika (`+0.261` vs `+0.167`), and Blaine (`+0.225` vs `+0.162`), while matching
    Koga (`+0.381` / `+0.424` w/ solo).
  - Built cached Showdown duel routing (`scripts/rr_route.py --scorer duel`, `logs/duel_cache/`, 33s cold / 9s warm)
    and wired D14 per-mon death costs (`--d14-bosses`) into `scripts/rr_search.py` (`one_battle` returns fainted
    species tuple; `score()` scores `win * (1 - Σ_dead c_i)`).
  - **Phase 6 paired 30-box × 5-boss benchmark (`results/rr_p6_duel.log`, duel scorer + duel chooser + R2 battler,
    seed 5, no free trades):**
    - `routed` vs `random`: **every boss improved on both win rate and deaths/mon** — Brock +1.1% (−0.073 d/mon),
      Misty **+11.1%** (−0.085), Erika **+5.6%** (−0.046), Koga **+5.6%** (14.4 → 20.0%, ≥50% 13 → 23%, −0.042),
      Blaine +0.6% (≥50% 30 → 47%, −0.003); **overall +4.8% ± 4.9%, −0.050 ± 0.038 d/mon** (`n=150`, SPRT δ 5%
      LLR +1.82; +6.9 pts turnaround over paper routing's −2.1%).
    - `routed+egg` (+1 Celadon Starter Egg) vs `random`: Brock +1.1%, Misty **+11.1%**, Erika **+8.9%** (88.9%,
      97% ≥50%), Koga +4.4%, Blaine +1.7% (47% ≥50%); **overall +5.4% ± 5.0%, −0.054 ± 0.039 d/mon** (LLR +2.28).
    - `routed+full` (Mudkip starter + routed + egg + choices) vs `random`: Misty **+15.0%**, Erika **+10.0%** (90.0%),
      Blaine **+15.6%** (48.3%, 63% ≥50%), but Brock −8.3% and Koga −10.0% (Mega Swampert drafted on 30/30 Koga
      teams crowded out non-Water answers into Iron Bundle Freeze-Dry / Tapu Lele).
- **Starter & Battle Bond / Emergency Exit ability audit:**
  - Verified `Emergency Exit` in RR triggers at **end of turn** (`gen9rr/abilities.ts:185-205`) and is faithfully
    implemented in `poke-engine/src/genx/rr_abilities.rs:768-806`; `Battle Bond` (`GRENINJABOND` → `GRENINJAASH` +
    3-hit Water Shuriken) is implemented in both Showdown and Rust (`rr_abilities.rs:153-170`), but `Mon.pack()`
    packed `greninja` instead of `Greninja-Bond` and `choose_ability()` defaulted 18/27 starters to
    `Overgrow`/`Blaze`/`Torrent` (missing `Unaware` Skeledirge, `Sharpness` Samurott, `Magic Guard` Delphox,
    `Tinted Lens` Decidueye, `Quill Rush` Decidueye-Hisui, `Self Sufficient` Torterra, `Iron Fist` Infernape,
    `Solar Power` Charizard in Sun, `Bad Company` Serperior, `Natural Cure` Meganium, etc.).
  - Fixed `Mon.pack()` (`Greninja-Bond`) and upgraded `rr_probe.choose_ability()` to use `mod_data.json` ability
    ratings below boss-specific immunities/weather/Cloud Nine/Intimidate (gating weather/drawback abilities to
    active weather); bumped `DUEL_CACHE_VERSION = 3`. Tests: 21/21 passing.

### 2026-11-25 — 6v6 starter halving (hardcore + normal), hazard probe, and three set-builder fixes

- **6v6 starter sequential halving** (`rr_route.py --starters-6v6`, 3 rounds, R3 = 16 boxes × 6 battles × 6
  bosses @ 20k iters; paired starter-less routed base box; lift = max(0, ΔU) per boss).
  - Hardcore (`results/rr_p6_starters_6v6.log`): Decidueye-Hisui **+0.076 U / +8.2% win** (Koga +0.164, Blaine
    +0.222), Incineroar +0.046, Cinderace +0.034, Meganium +0.028, Rillaboom +0.024, Infernape +0.018,
    Meowscarada +0.014. Swampert cut in R1 (#15).
  - Normal (`results/rr_p6_starters_6v6_normal.log`, ran from the now-deleted `rr_normal_starters.py`):
    Incineroar +0.050, Venusaur +0.033, Cinderace +0.025, Decidueye +0.021, Emboar +0.019, Quaquaval +0.012,
    Chesnaught +0.009. Lifts are smaller: the starter-less box already clears most normal gyms.
  - Incineroar is top 2 in both modes. R1 at 6 boxes × 2 battles is noisy (R1 and R2 orders disagree a lot).
  - user bet: "Swampert stays winning" — miss (cut in R1). Caveat: Swampert ran a dead Liquidation at Blaine
    (fix below), so its hardcore number was handicapped. Gemini's bet: Swampert top 2 (70%) miss, Greninja top 14
    (60%) hit.
- **Hazard probe (user's ask: force Stone Axe Kleavor / Ceaseless Edge Samurott-Hisui).** Stealth Rock, Spikes etc.
  are hardcore-banned; these two slicing moves are not and set rocks / spikes in Showdown and poke-engine. Paired
  8 boxes × 4 battles @ 10k, forced mon(s) lead, replacing the base team's last slot(s):
  - Koga: base 31.2%, Kleavor lead 12.5%, Samurott-H lead 21.9%, both 15.6% — fragile leads lose to Koga's fast
    Tapu Lele / Hoopa-U under permanent Tailwind.
  - Blaine: base 56.2%, Kleavor lead 62.5% (+6.2), Samurott-H lead 40.6% (−15.6), both 68.8% (+12.5). Blaine has
    Ho-Oh (4× weak to rocks) and a Focus Sash Venusaur, so rocks have real targets there.
  - n = 32 battles per arm: ±~9 pts, so only the Koga drop is clearly real. Blaine's +12.5 is a lead worth a
    proper test (more boxes, slot dropped by duel value rather than "the last one"), not a result.
- **Fixes** (tests 61/61; `DUEL_CACHE_VERSION = 4`, so duel tables recompute):
  - `best4` no longer picks attacks of a type the arena kills (`rr_probe.dead_types`: Water under Desolate Land,
    Fire under Primordial Sea, lifted by Cloud Nine / Air Lock). At Blaine: Swampert earthquake / superpower /
    takedown / dig (was liquidation), Greninja darkpulse / icebeam / gunkshot / nightslash (was surf + hydropump),
    Golduck (Cloud Nine) keeps Surf. User's catch.
  - 1v1 duels play Palafin as Palafin-Hero (`rr_team_probe.duel`): alone it can never switch out to transform
    (verified in the sim: 298 Atk vs 147). 6v6 keeps Palafin; poke-engine implements Zero to Hero. User's catch.
  - Normal mode folded into `rr_route.py --normal` (normal boss sets / caps, no hardcore bans or swaps, duel cache
    keyed on mode); `rr_normal_starters.py` deleted. The two data bugs it patched are fixed at the source:
    `rr_probe.tracker_id` maps "HeavyD. Boots" → heavydutyboots, and `rl/convert.py` drops Terrain Extender
    (not in the engine's item enum; without this normal Surge / Erika fell back to slow Showdown search).
  - Not re-run yet: the hardcore 6v6 halving with the Blaine fix (water starters were handicapped there).

### 2026-11-25 — Blaine hazard probe, proper (`scripts/rr_hazard_probe.py`, `results/rr_p6_hazards_blaine.log`)

- 40 starter-less routed boxes × 6 battles @ 20k iters, paired on box + seed (240 battles / arm). Forced Sharpness
  Jolly mon(s) lead, replacing the weakest base member by duel value. Controls swap the hazard move for a
  no-hazard move: Kleavor Stone Axe → Stone Edge, Samurott-H Ceaseless Edge → Knock Off.

  | arm | win | Δwin vs base (±se) | ΔU (±se) |
  |---|---:|---:|---:|
  | base | 44.2% | — | — |
  | Kleavor (Stone Axe) | 47.1% | +2.9 (4.5) | +0.022 (0.038) |
  | Kleavor ctrl (Stone Edge) | 34.6% | −9.6 (4.2) | −0.082 (0.034) |
  | Samurott-H (Ceaseless Edge) | 52.9% | **+8.8 (4.0)** | +0.056 (0.032) |
  | Samurott-H ctrl (Knock Off) | 45.0% | +0.8 (4.4) | −0.011 (0.036) |
  | both | 53.8% | **+9.6 (4.6)** | +0.049 (0.035) |

- Reading: **the hazards themselves pay at Blaine**: hazard arm minus its control is +12.5 pts (Kleavor) and
  +8.0 pts (Samurott-H). The Kleavor control is a bit unfair (Stone Edge 80% acc, no Sharpness: ~10% less expected
  damage than Stone Axe), the Samurott one is close (Knock Off 100% acc, removes items). Kleavor as a mon is no
  better than what it replaces; with rocks it breaks even. Samurott-H and both are ~2 se over base on win rate,
  ~1.5 se on U; five arms against one base, so treat as "likely real, about +9 pts", not settled.
- Hazards earn their keep here because Blaine's team has Ho-Oh (4× rocks) and a Focus Sash Venusaur. At Koga
  (first probe, n = 32) the fragile leads lost clearly.
- user bet: hazards underperform — miss (both hazard arms beat their controls; Samurott-H and both are the
  best arms). The user meant "the hazard mon is worse than the better mon it displaces" (vs base, not
  vs control): on that reading Kleavor held (+2.9 ± 4.5, a wash), Samurott-H missed (+8.8 ± 4.0). The probe
  displaced the *weakest* member; displacing a strong one would likely confirm the user's version. Note the 1v1
  duel scorer cannot see hazards at all (no switches in a duel), like Palafin's Zero to Hero. Claude's bet: both does not beat base by 2 se (7) — narrow miss on win (2.1 se), held on U
  (1.4 se); hazard ≈ control (5) — miss.

### 2026-11-25 — D15: teach the leaf evaluation about hazards we set

- Finding: in `eval_weights_r2.txt` the enemy-side `rocks` / `ground_hazards` weights are exactly 0 (hardcore bans
  our hazard moves, so the replay buffer never had enemy-side hazards; `fit_eval.py` zeroes constant features).
  The search credits Stone Axe / Ceaseless Edge only for switch-ins inside its tree. The features also count
  mons, not damage (a 4× weak Ho-Oh = a resisting mon). The RR AI's own switches are in the tree (Rust port,
  97% agreement), so baiting switches is reachable; valuing the hazard that punishes them is not.
- Options: A leave; B hand-set an effectiveness-weighted weight; C refit with hazard battles in the data;
  D measure switch-ins first.
- user call: C (Samurott-Hisui is a viable starter with guaranteed hazard access, so the battler should learn it).
- Claude: agree with C, with B's feature fix underneath (refitting a mon-count feature learns one average weight
  for Ho-Oh and resisters) and D's counts logged for free from the data run. Note Ceaseless Edge sets Spikes,
  which Flying Ho-Oh ignores; the Ho-Oh lever is Kleavor's Stone Axe.

### 2026-11-25 — Hardcore 6v6 starter halving re-run after the dead-move fix (`results/rr_p6_starters_6v6_v2.log`)

- Same schedule and seed as `rr_p6_starters_6v6.log`; changes: no dead Water moves under Desolate Land,
  Palafin-Hero duels (duel cache v4). R3 (16 boxes × 6 battles @ 20k): Decidueye-Hisui +0.052 (Koga +0.175),
  Cinderace +0.050 (Blaine +0.234), Incineroar +0.048, Primarina +0.041 (Koga +0.171), Inteleon +0.014,
  Quaquaval +0.013, Infernape +0.012. Swampert reached R2 (#4 in R1), cut at #12.
- Stable across both runs: Decidueye-H, Cinderace, Incineroar in the top 3 (now a three-way tie within noise).
  Meganium, Rillaboom, Meowscarada out; Primarina, Inteleon, Quaquaval in. The new Water entrants gain at
  Brock / Koga, not Blaine, so the swap is mostly R1 noise; the Blaine fix shows as smaller starter lifts there
  (Decidueye-H Blaine +0.222 → +0.037) because the base boxes' own Water mons now carry live moves.
- user bet: no change — miss (3 of 7 changed). Claude: no Water starter in the final 7 (7) — miss (3);
  at least one of the 7 swaps (7) — hit; Decidueye-H stays #1 (7) — hit, by 0.002.

### 2026-11-25 — Purposeful duping scan (`scripts/rr_dupe_scan.py`, `results/rr_dupe_scan.log`)

- The user's encounter rules (new): one encounter per area (any method), delayable, dupes clause by evolution
  line rerolling until new. `routed_box` differs (species-level dupes, 4 rerolls, free delay per boss box).
- Scan: per boss, each table's best mon by duel-table value, if it ranks top 15 among all species valued at that
  boss, with its odds owning 0–3 of the table's most common other lines.
- Grass tables have ~10 lines, so blocking barely helps (Gengar in Pokémon Tower 20 → 40% with 3 blockers).
  Duping works on the small fishing tables, and they hold Koga answers:
  - Golisopod (Koga #5): S.S. Anne old rod 70%, **100% owning the Finneon line**; Vermilion good rod 20 → 100%
    owning Frillish + Staryu lines.
  - Araquanid (Koga #3): Route 25 good rod 60 → 75 → **100% owning Mantine + Goldeen lines**; Route 13 good rod
    20 → 100% with Qwilfish + Skrelp lines.
  - Wailord (#7): Route 24 good rod 60 → 100% (Wishiwashi, Clobbopus); Dewgong (#9): Route 12 old rod 30 → 100%
    owning the Spheal line.
  - Blaine: nothing close to guaranteed; its top mons (Gengar, Scream Tail, Tropius, Centiskorch) sit in big
    grass tables.
- Caveats: value is the 1v1 duel row (same blind spots as routing); each blocker costs another area's encounter,
  so the cheap ones are blockers you would keep anyway. Not yet checked in 6v6.

### 2026-11-25 — D15 run: hazard data, feature fix, refit (gates pending)

- **Data** (`scripts/rr_hazard_probe.py --states`): Samurott-H / Kleavor-led routed teams at #26 #34 #38 #44 #46
  #80 #90, 20 boxes × 6 battles each half: on-policy R2 (`logs/d15_r2.*`, seed 10) and hand evaluation
  (`logs/d15_hand.*`, seed 11; the hand eval values hazards: −10 rocks / −7 Spikes a mon). 4,560 battles,
  ~105k positions; enemy rocks up 19% / 29%, Spikes 8% / 20%. Kleavor was also put in at #34–#46 where Scyther
  is not yet catchable (the setter only checks the evolution and the move); fine for a value fit.
- Found while building it: the ability picker gives Kleavor Sheer Force, which cancels Stone Axe's rocks
  (Showdown `!move.hasSheerForce`); the probe forces Sharpness once the Dream Patch is in hand. Under R2 the
  search often switched the setter out on turn 1 (enemy-hazard weights 0).
- **Feature** (`poke-engine` `evaluate.rs` `side_features`): `rocks` / `ground_hazards` = Σ switch-in damage /
  current HP, capped at 1 (rocks by type effectiveness, Mountaineer immune; Spikes by layers, grounded), Toxic
  Spikes / Sticky Web layers / 8. Was: the count of mons hit.
- **Refit** (`fit_eval.py` R1 + R2 + D15, `--within-team --scale 2.0` → `data/rr/eval_weights_d15.txt`, 301,724
  positions, 13,182 battles): their_rocks 0 → +0.46, their_ground_hazards 0 → +1.00 (their_hp −0.46: a unit of
  switch-in damage ≈ a unit of HP already lost). Held-out within-team log-loss 0.4473 (R2's fit: 0.4470), flat:
  hazard positions are a minority. `our_ground_hazards` came out +1.15 on Toxic Spikes alone (2% of positions,
  two bosses, ×8 from the /8 rescale; poisoning is already in `our_tox`): set to 0 by hand (hazards on our side
  cannot help).
- **Baselines under R2 on today's code** (`results/rr_d15_gate_r2.log`, `results/rr_d15_fresh_r2.*`):
  - Blaine probe with the run's own sets (setter = tp.mon_of + hazard move first): base 44.2%; Kleavor +1.2,
    Kleavor ctrl −16.2, **Samurott-H −9.2**, Samurott-H ctrl −6.7, both −5.4 (se ~4.5–5). The earlier +8.8 for
    Samurott-H came from the hand-picked Jolly set (Sucker Punch / Night Slash / Sacred Sword), not from Spikes:
    best4 gives it Foul Play / Megahorn. Kleavor's rocks beat its control under both sets (+12 / +17).
  - Fresh seed 4 (30 boxes, `--legal --arena --matchup --death-weight 6 --engine 20000`): Brock 93 / Misty 48 /
    Erika 78 / Koga 7 / Blaine 37. Below the Phase 4 fresh run (96 / 65 / 84 / 11 / 44) on unchanged seeds:
    the box pool and abilities changed since (Phase 6: 131 vs 129 species, ability ratings) — to look into.
- **Gates under D15** (`POKE_ENGINE_EVAL_WEIGHTS=data/rr/eval_weights_d15.txt`, same boxes and seeds):

  | Blaine probe arm (win) | R2 | D15 |
  |---|---:|---:|
  | base | 44.2% | 38.8% |
  | Kleavor (Stone Axe) | 45.4% | **55.8%** |
  | Kleavor ctrl (Stone Edge) | 27.9% | 33.3% |
  | Samurott-H (Ceaseless Edge) | 35.0% | 36.2% |
  | Samurott-H ctrl (Knock Off) | 37.5% | 30.8% |
  | both | 38.8% | **54.2%** |

  Kleavor +10.4 and both +15.4 over the same arms under R2 (unpaired se ~4.5); within D15 Kleavor is +17.1 ± 5.9
  over base. Samurott-H unchanged: its best4 set, not the evaluation, holds it back. Base −5.4 has no hazards
  in it; read as noise (fresh-seed Blaine moved the other way, 37 → 41).
  Fresh seed 4 (`results/rr_d15_fresh_d15.*`): Brock 91 / Misty 48 / Erika 73 / Koga 7 / Blaine 41; paired vs
  R2 **−0.7 ± 2.6 pts, +0.017 ± 0.017 d/mon, SPRT δ 5% H0 (no difference)**. No regression.
- Bets: Claude — Kleavor gains > 5 pts (5) hit; Samurott-H stays negative (7) hit; fresh within ±3 (7) hit.
  User did not bet.
- **Adopted (user, 2026-11-25):** `eval_weights_d15.txt` is the default in `rl/engine_search.py` and `rl/line.py`.

### 2026-11-25 — Why fresh-seed Misty fell 65 → 48% (investigation)

- Same seed, but no box is the same as in `rr_p4_fresh_r2`: the box sampler's draws changed with Phase 6
  (`order.json` gift kinds, branched evolutions), so seed 4 now means different starters and rolls.
- The structural change: **Carnivine** (in-game trade, give Snom at Cerulean) was in all 30 old Misty boxes and
  in 17 of 30 chosen teams. Phase 6 applies the user's trade rule (only if the box owns the give line), and no
  sampled box owns Snom, so it is gone; gifts per Misty box 1.8 → 0.8. The old Misty number was inflated by an
  illegal free Grass answer. Correction (same day): the D15 runs used the probe's default `--chooser greedy`,
  Phase 4's fresh runs `--chooser cover`, so the 65 → 48 gap also mixes in a chooser change; Carnivine's loss
  is real, its share of the gap is not measured. Owning a Snom early would make the trade a plannable action
  (like the dupe shaping).

### 2026-11-25 — Movesets: diagnosis (D16 pending)

- Samurott-H at Blaine: hand-picked Jolly Sharpness set (Ceaseless Edge / Sacred Sword / Sucker Punch / Night
  Slash) +8.8 vs base; best4 + Ceaseless Edge (Foul Play / Megahorn / Sacred Sword) −9.2: an 18-point swing from
  the moves alone.
- `best4` scores power × accuracy × STAB × category fit: no attacker abilities, no priority, no boss. The calc
  (`ai_util/calc_bot._dmg_frac`, used by duels, the calc bot and `rr_search.default_set`) models Technician,
  Sheer Force, Huge / Pure Power only. Showdown and poke-engine apply Sharpness etc. in battle; only the set
  builders ignore them.
- Scan of the Koga / Blaine pools (491 / 580 species as built): 19 / 23 have a flag ability (Sharpness, Iron
  Fist, Strong Jaw, Tough Claws, Mega Launcher, Punk Rock, Reckless) and use fewer than 2 boosted moves; ~37 / 41
  more carry Adaptability, -ate, Technician, Sheer Force etc. that best4 ignores; **38 / 46 pick Foul Play**,
  scored as a 95-power move off the user's own Attack (it uses the target's).

### 2026-11-25 — D16: movesets (user call: A + B)

- user call: A + B. Priority: the user notes it matters only against faster foes or foes with priority, so no
  flat priority bonus; `ai_util.matchup.pair`'s "faster" is speed only (priority not modelled), a later fix.
  Samurott-H as starter: the user is sure it is not the best pick; no re-run for that.
- **A** (`ai_util/calc_bot.ability_power`, used by `_dmg_frac` and `gauntlet.encounter_moves(ability=)`):
  Sharpness / Iron Fist (RR: ×1.3) / Strong Jaw / Tough Claws / Mega Launcher / Punk Rock / Reckless /
  Technician / Sheer Force / -ate / Steelworker-likes / Adaptability; Foul Play off the target's Attack (calc) or
  a base-100 Attack (best4); Huge / Pure Power in best4's fit. Calc vs Showdown, Sharpness Night Slash into Great
  Tusk: 13.7% vs 12.7–14.8% over 10 hits. New best4 at Blaine: Samurott-H Night Slash / Sacred Sword /
  X-Scissor / Foul Play, Kabutops Stone Axe, Clawitzer four pulses, Drednaw bites. Kleavor still gets Sheer Force
  from the ability picker. `DUEL_CACHE_VERSION = 5`.
- **B** (`rr_team_probe.py --sets boss`, `team_mons`): moves and nature from `rr_search.default_set` (calc damage
  on this boss's team) for every battle.
- Gates (fresh seed 4, `--chooser greedy` as the D15 runs): A vs `rr_d15_fresh_d15`, A+B vs A.
- **Results** (150 paired boxes each; `results/rr_d16_fresh_{a,ab}.*`; per boss: mean paired diff ± se):

  | boss | D15 | A | A+B | A − D15 | A+B − A |
  |---|---:|---:|---:|---:|---:|
  | Brock | 91% | 92% | 93% | +1.1 ± 0.8 | +1.7 ± 1.5 |
  | Misty | 48% | 42% | 41% | −6.1 ± 4.9 | −0.6 ± 4.6 |
  | Erika | 73% | 77% | 81% | +3.3 ± 5.3 | +4.4 ± 4.1 |
  | Koga | 7% | 12% | 8% | +5.0 ± 3.9 | −3.9 ± 2.2 |
  | Blaine | 41% | 38% | 46% | −3.3 ± 7.4 | +7.8 ± 6.9 |

  Overall (sprt.py, 95% intervals): A vs D15 **−0.0 ± 4.3 pts**, A+B vs A **+1.9 ± 3.8**, A+B vs D15 +1.9 ± 4.6;
  all SPRT "continue". No measurable change yet; A is kept on correctness (the calc now matches Showdown).
- Bets: Claude — A +2 to +4 (5) miss (0.0); B +3 more, mostly Erika / Blaine (5) miss on size (+1.9), right on
  where (Erika +4.4, Blaine +7.8). User did not bet.

### 2026-11-25 — D16 follow-up: Misty / Koga diagnosis, B′, ability tie-break

- **Misty −6 under A is team choice, not a set bug.** Same 30 boxes, but only 10/30 chosen teams match D15's
  (Koga 2/30): same-team boxes −1.7 pts, changed-team boxes −8.3. A shifts stage-A solo scores slightly
  (Marill 0.195 → under the top-24 cut at ~0.167; 12 seeded solos: A set 0.171, D15's Mild set 0.183, both
  0 wins), so the greedy chooser's candidates and its 4 screening battles land on other teams. Any change that
  touches the calc reshuffles teams: set comparisons are only clean on fixed teams (A → A+B: 30/30 same).
- **Koga −3.9 under B is a set bug: `default_set` is damage-only.** It dropped Thunder Wave (Aggron, Lanturn),
  Will-O-Wisp (Houndstone, Sableye, Runerigus) and Light Screen for a fourth attack. **B′**: keep best4's
  status moves, the calc picks only the attacks (`rr_search.default_set`; also the seed set of rr_search's
  set search). Gates queued: seed 7 A vs A+B′ (`results/rr_d16_s7_{a,ab}`), seed 4 A+B′ vs A
  (`results/rr_d16_fresh_abk`).
- **Ability tie-break** (parked in a patch until the queued runs start, so they stay clean): Sheer Force and
  Sharpness both rate 3.5 and slot order picked Sheer Force for Kleavor (which also cancels Stone Axe's
  rocks). `choose_ability` now breaks ties by best4's attack power under each ability (`encounter_moves`
  rule "strength"). Changes 3/372 at Erika, 12/490 at Koga, 11/579 at Blaine: Kleavor → Sharpness, Dracovish
  → Strong Jaw, Golem-Alola → Galvanize, Conkeldurr / Hariyama / Cetitan → Sheer Force, Feraligatr → Strong
  Jaw. Also the calc missed Sheer Force on Stone Axe / Ceaseless Edge / Diamond Storm (their data has
  `secondary: {}`, falsy in Python; Showdown boosts them) and Electro Shot / Order Up (`hasSheerForceBoost`).
- Bets (B′ gates): Claude — seed 7 A+B′ − A **+2 to +4 pts**, Koga ≥ 0 (5); pooled over seeds 4 + 7 still SPRT
  "continue" (7).
- **B′ results** (`results/rr_d16_s7_{a,ab}`, `results/rr_d16_fresh_abk`; paired diff ± se):

  | boss | seed 7 A | A+B′ | diff | seed 4 A+B′ − A | pooled (60 boxes) |
  |---|---:|---:|---:|---:|---:|
  | Brock | 93% | 95% | +2.2 ± 3.8 | +1.1 ± 2.1 | +1.7 ± 2.2 |
  | Misty | 34% | 29% | −5.0 ± 5.0 | −0.6 ± 4.0 | −2.8 ± 3.2 |
  | Erika | 69% | 74% | +4.4 ± 4.5 | +2.2 ± 5.2 | +3.3 ± 3.4 |
  | Koga | 8% | 10% | +1.7 ± 3.3 | −3.9 ± 2.6 | −1.1 ± 2.1 |
  | Blaine | 27% | 47% | **+20.0 ± 5.7** | +9.4 ± 6.9 | **+14.7 ± 4.5** |
  | all | | | +4.7 ± 4.1 (95%) | +1.7 ± 4.0 (95%) | **+3.2 ± 1.5 (se)** |

  SPRT (δ 5%) "continue" on each seed (the effect sits between 0 and 5). Pooled ~2.1 se: B′ helps, almost all
  at Blaine (Energy Ball / Rock Slide / Psychic in for Shadow Ball / Thunderbolt: coverage for this boss's
  team). **Koga correction:** B vs B′ on seed 4 is 0.0 at Koga (identical), so the dropped status moves were
  not Koga's −3.9; B vs B′ overall is noise (Erika −2.2 ± 4.4, Blaine +1.7 ± 6.1). Misty: the calc swaps
  U-turn / Dig / Thief / Sucker Punch for Bullet Seed / Dual Wingbeat / Aerial Ace (pivot and priority
  value invisible to a damage-only calc), −2.8 ± 3.2 pooled, not resolved.
- Bets settled: Claude seed 7 +2 to +4 (5) — miss, high (+4.7); Koga ≥ 0 on seed 7 hit (+1.7); pooled SPRT still
  "continue" (7) hit. User did not bet.
- Held-out check launched (seed 7, A vs A+B′, ability tie-break in both): Lt. Surge, Giovanni, Sabrina, Pryce,
  Clair (`results/rr_d16_ho_{a,ab}`). Two crashes on the way, both fixed: Ice Hammer's `"self": null` in
  mod_data broke poke-env's `Move.self_boost` (the strip moved from `rl/rr_ai_diff.py` into
  `rr_probe.inject_mod_data`), and D16's ability-changed move types skipped `effectiveness` (a "???" defender
  had no chart row; ability immunities were skipped too, e.g. Galvanize into Volt Absorb).
- Held-out first pass: Lt. Surge / Giovanni / Sabrina fully paired; Pryce / Clair lost 24 boxes to a poke-env
  assert on Transform (Ditto) and Sketch (Smeargle): the request lists moves the mirror never learned. Fixed in
  `ai_util/sim.py` (`_transformed_moves`); Pryce / Clair re-run. No earlier run had failed boxes.
- **Hidden Power was mistyped**: player IVs are all 31, so Showdown plays it as Dark; the calc read Water
  (mod_data) and best4 Normal (moves.json). B′ added it 28 times at Sabrina on a wrong type. Fixed at load
  (`rr_probe.inject_mod_data`); applied after the seed-4/7 and first held-out runs (the Pryce / Clair re-run
  has it in both arms).
- **Held-out result** (seed 7, 30 boxes each, all paired; `results/rr_d16_ho_*`, Pryce / Clair from
  `results/rr_d16_ho2_*`): A+B′ − A = Lt. Surge −7.2 ± 4.0, Giovanni 0.0 ± 3.5, Sabrina −2.8 ± 3.2,
  Pryce +5.6 ± 3.3, Clair −3.3 ± 5.2; **all −1.6 ± 1.8 (se)** (A 56.4%, B′ 54.9%). The first five bosses gave
  +3.2 ± 1.5. B′ does not generalize: it wins big where a damage-only coverage pick is the missing piece
  (Blaine +14.7, Pryce +5.6) and loses a little where best4's utility attacks matter (U-turn, Dig, Sucker
  Punch, Flash-likes; Lt. Surge, Misty). Note the 5 "train" bosses were where B′ was designed and debugged
  (Koga status-move fix), so +3.2 is optimistic. → D17 (what to do with B′).
- D17 posed (options: 1 keep best4 default, B′ opt-in; 2 battles choose best4 vs B′ per team, screening seeds
  apart from scoring seeds; 3 teach the calc utility moves, re-test held-out). Claude's lean: 1 now, 2 next.
  **user call: pending** (session wrapped up 2026-11-26 before the answer).
- User note (2026-11-26, confirmed by the user): in Radical Red, an NPC in the house next to Saffron's Poké Mart
  adjusts IVs so Hidden Power takes any type the player picks. Before Saffron it plays as Dark (all-31 IVs);
  from Saffron on, Hidden Power is a free-type coverage move, and the calc and best4 should choose its type per
  boss instead of fixing it to Dark (not modelled yet; gate on the Saffron progress point).

## 2026-11-26 — D17: what to do with B′ → the team builder must be learned

- Options posed: (1) keep best4 default, B′ opt-in; (2) simulated battles choose best4 vs B′ per team
  (separate choosing / scoring seeds); (3) hand-code utility-move value into the calc, re-test held-out.
- **user call: none of 1–3, conf not given.** Movesets are too complex for heuristics. The team builder
  (team + sets) sets the battler's floor and ceiling, so it must be good. It has to carry over to other games and
  ROM hacks where move choices are permanent (no relearner, one-use TMs), so it can hard-stick a run. If data is
  what blocks a learned builder, solve data next. best4 stays the baseline that a learned builder must beat
  held-out.
- Claude: agrees heuristics are near their limit (B′: +3.2 train, −1.6 held-out). Permanent choices make team
  building part of the run-level decision problem (roadmap step 2), and its evaluator is a team-vs-boss
  win-probability model (roadmap step 1's deliverable). Existing data: 7,787 box results in 89
  `results/rr_*.events.jsonl` (mixed configs, teams only from our own chooser). → D18 (where the data comes from).

## 2026-11-26 — D18: where the team builder's data comes from → A first (fit on logged boxes)

- Options: A fit a team-vs-boss win-probability model on the logged boxes; B generate data with deliberately
  varied teams / sets (exploration, poke-engine); C search picks between set options at scale and the model
  imitates it (expert iteration). **user call: A now, conf not given**, while researching where to get many
  more good enemy teams (competitive teams for RR and vanilla, other ROM hacks' bosses), where they sit on
  box progress, and how to design boxes so the builder learns well (research agent running).
- `rl/fit_teamwin.py`: 5,982 box events (hardcore legality, every run with a matrix), label = win rate over
  6 battles, 20 features (per-enemy "do we answer it" summaries of the 6x6 matchup matrix, team means, counts
  of pivots / priority attacks / status moves). Train on Brock / Misty / Erika / Koga / Blaine (validation =
  a fifth of their boxes), held out Lt. Surge / Giovanni / Sabrina / Pryce / Clair (402 boxes). Battle noise
  is 20–50% of each boss's win-rate variance (6 battles per label).

  | model | log-loss train / val / held-out | rank corr within boss, held-out | same box, which team/sets wins (held-out, 125 pairs) |
  |---|---|---:|---:|
  | constant | 0.692 / 0.692 / 0.688 | – | – |
  | one hand feature (share of enemy mons answered safely) | – | +0.19 | – |
  | linear | 0.544 / 0.545 / 0.577 | +0.28 | 61%, corr of differences +0.05 |
  | 1x32 probe | 0.484 / 0.494 / 0.656 | +0.26 | 53%, −0.08 |

- Meaning: the features tell strong boxes from weak ones a little better than one hand feature, on new bosses
  too. But the team builder's actual question, "for this box, which six / which sets", is near a coin flip
  on new bosses: the 125 held-out pairs are mostly the D16 A vs B′ boxes (same team, different sets), and
  the matrix features cannot see what made B′ win or lose (the calc's blind spots again). The probe fits the
  five training bosses well (val rank +0.58) and loses it on new ones: **five bosses are five environments**,
  more boxes of the same bosses will not fix that. Also the data has few same-box contrasts, and only among
  teams our chooser liked. Both point at the research question (many more distinct enemy teams) and at
  D18 B (varied teams per box), not at a bigger model.
- Enemy-team research (agent, sources fetched and parsed unless marked recalled):
  1. RR 4.1 itself: `data/rr/trainers_hardcore.json` (already parsed by `build_rr.py` from the dump) has 736
     trainers / 2,307 mons; 573 at fixed levels, 163 relative to the cap. `rr41_{normal,easy}_trainers.txt` are
     not parsed yet. So far only ~20 bosses are used as opponents.
  2. Dynamic-Calc (github hzla): trainer sets for ~44 ROM hacks, ~88k sets in one JSON schema (Run & Bun 1.8k,
     Renegade Platinum 2.8k, Emerald Kaizo 1.7k, Inclement, Imperium, BB2 Redux, ...), trainer order for ~20
     hacks; no license file (research use, don't redistribute). Other mechanics generations: played under RR
     mechanics they are diverse opponents, not faithful copies.
  3. Showdown's random-battle generator (MIT, already in our Node bridge): unlimited level-balanced teams,
     gen9 sets L64–100 and baby sets L5–9.
  4. metamon-teams (HF): 44k+ gen9ou teams, L100 (late game only). 5. Smogon chaos stats (LC fits early game).
  6. hzla/romhack-usage-statistics: real players' parties chosen from their boxes per trainer (no outcomes):
     a human prior for the team builder itself.
  Placement on progress: by absolute ace level → the RR checkpoint whose cap is the smallest ≥ it (badge counts
  don't transfer across hacks); don't level-scale L100 competitive teams to early caps. Hold out by source
  family (RR Normal shares sets with Hardcore). → D19 (enemy pool and holdout).

## 2026-11-26 — D19: enemy pool → RR only first

- Options: (1) all 736 RR hardcore trainers as opponents, hold out the gym leaders / big bosses; (2) plus other
  hacks (Dynamic-Calc), hold out hack families; (3) plus random-battle filler.
- **user call: 1, 70%.** **user bet:** "I expect overfitting and not getting what we want, but it still being
  better" than the 5-boss model (held-out same-box choice 61% / rank 0.28).
- Plumbing: `rr_team_probe.py --alt K` (box mode): also plays K variants of each box's six (1–3 mons swapped for
  other box mons, the main six's seeds: paired), logged as `box_alt` events; skips the threat duels.
  `rl/fit_teamwin.py --split d19` trains on non-boss trainers, holds out bosses (gym leaders incl. Johto,
  Giovanni, E4, Lance, rematches; rivals / Brendan / May / Rocket admins stay in training); `--label utility` =
  win × (1 − 0.5 × deaths/mon). Validation is now split by box (a box's variants together): the d18 probe's
  validation rank fell 0.58 → 0.38 (row split leaked boxes); held-out numbers barely moved (linear 0.27 / 60%).
- Data run `results/rr_d19_all` (seed 11): all 112 order trainers with a team (not #40 Ghost), 20 boxes each,
  `--alt 3`, greedy chooser, best4 sets, engine 20000, 12 workers (user gaming). Doubles fights play as singles.
- Claude bet (before results): the d19 model is **not** clearly better than d18 on the held-out bosses'
  same-box pairs (within ±5 points of 60%), but ranks boxes on bosses about as well (0.25–0.35); 70%.

## 2026-11-26 — First simulated Nuzlocke runs (`scripts/rr_run.py`, `results/rr_runs_s1`)

- User asked for a few small full runs "to get a feel for how far it would go". `rr_run.py`: random starter,
  box grows as locations open (one random encounter per location, dupes skipped, gifts), the box probe's
  settings per mandatory fight (93 of the order's entries: greedy chooser over the living box, cap level,
  items in hand, one engine-searched battle), fainted = dead, a loss ends the run, healed between fights.
  Not modelled: doubles (as singles), optional fights, eggs, trades, the player's route / catch choices.
- 6 runs (seeds 1–6), 0.3–3 min each: cleared 4, 4, 14, 14, 14, 26 of 93. Lost to Brock ×2 (box 4–5 alive),
  Misty ×3 (box 6–10 alive), Super Nerd Aidan ×1 (cap 36). Deaths before the loss: 8–24.
- **The killer is attrition, not bosses.** Routine trainers that are won 100% still cost ~0.3–0.6 deaths per
  fight early and ~1 at Rocket grunts / Archer (the run deaths match the D19 box probe's deaths/battle on the
  same trainers). 13 fights before Misty ≈ 10 dead, so the box is gutted when the boss comes. A real Nuzlocke
  needs ~0 deaths in routine fights. So: the team builder's label must weigh deaths (D19 `--label utility`),
  and the battler's play in won-anyway fights is the bigger lever (death weight 6 = one death costs a win
  still loses mons when the win is certain; the search does not price a mon's future value, cf. D14).

## 2026-11-26 — D20: cutting deaths in routine fights → the team picker minimises deaths

- Options: (1) the battle search prices deaths higher; (2) the team picker minimises expected deaths;
  (3) a per-mon future value in the search (D14 death costs).
- **user call: 2, 90%.** "Maximising win chance and not worrying about death amounts is the sole reason
  nuzlockers say the champion fights are usually one of the easiest in the game. Reducing deaths per fight via
  the battler would also start increasing the risk of a total wipe: not impossible but double-edged."
  (Deaths matter in every fight but the last; the battler should keep playing to win, the choice of six
  should carry the safety.)
- `--chooser safe`: the greedy chooser scored by the share of the team alive after a win (0 on a loss) over
  its n_team calc-bot battles, progress breaking ties (`team_value`). Test: `rr_run.py --chooser safe` on seeds
  1–12 vs greedy on the same seeds (1–6 already run).
- Claude bet (before results): safe cuts routine-fight deaths a little (deaths per fight before Misty down
  ~20–40%) but median fights cleared moves less than +5 (the calc-bot's 4 battles are a noisy death signal,
  and the searched battle, not the chooser, decides most deaths); 60%.
- **D20 result** (12 paired seeds, `results/rr_runs_safe` vs `results/rr_runs_s1`): fights cleared median 14 vs 14,
  paired −0.8 ± 1.1 (safe − greedy); deaths per won fight before Misty 0.53 vs 0.54. No effect: early boxes hold
  ~9 mons (median), so there is little to choose, and 6 of 12 runs played identically. Bets: Claude's "deaths
  −20–40%" missed (0%), "cleared < +5" hit. User did not bet.
- How mons die in routine fights (`rr_deaths.py`, run teams, death weight 6, matchup; NB this tool plays the
  Showdown search without items, not the engine search the runs use, so its win rates at #9 / #20 (50%) are not
  the runs' (~100%)): in WON fights the deaths are low-HP mons. #4 Catcher Sammy 12/12 wins, 9 deaths: 8 under
  half HP, 6 on the turn they switched in (sacked to bring another mon in). #16 Grunt 12/12 wins, 15 deaths:
  14 under half HP, 13 left in until KO'd. The 2026-10-11 preserve rule (retreat a mon under 40% when a healthy
  bench mon takes the hit) cut Misty's deaths 47 → 28 then; it lives in the calc bot (rollouts / Showdown
  search). The engine search the runs use has no such rule; it prices a fainted mon at 60 evaluation points
  (`rl/engine_search.py`, POKE_ENGINE_DEATH) and still leaves low mons in. → D21.
- User asked: doesn't a reward per surviving mon already prevent this? Checked: **the death price the runs use is
  weak and inconsistent.** The engine search (most decisions: e.g. 138 engine vs 9 Showdown in one box) scores
  leaves with the D15 fitted logit, trained on U = win × (1 − 0.5 × deaths/6): one death ≈ 1/12 of a win.
  `evaluate.rs` returns the fitted logit before the hand-set death term (POKE_ENGINE_DEATH = 60) is applied, so
  that term is off whenever fitted weights are on. The probe's `--death-weight 6` (one death = one win) reaches
  only the Showdown fallback search. Earlier evidence of the double edge: Misty 2026-11-23, engine death 0 → 53%,
  60 → 52%, 200 → nearly every box lost.

## 2026-11-26 — D21: re-price deaths in the engine's fitted evaluation

- Options: (1) retreat rule when the win is near-certain; (2) death cost up when the win is near-certain;
  (3) leave the battler; (4) refit the engine's leaf evaluation on a label where a death costs more.
- **user call: 4, a death = 1/6 of a win as the base** (conf not given; no bet "without knowing how (well) it
  will be implemented"). On (1): a hard rule makes no sense without the context the search has (the switch-in,
  its matchup, pivoting through several mons to a winning position vs staying in vs wiping). Later: per-mon
  death weights by run importance (some mons can be sacked without worry), unless testing how far a run gets
  deathless.
- Plan: `fit_eval.py --death-cost 1.0` (U = win × (1 − deaths/6); a loss 0) on D15's data and recipe (R1 + R2 +
  D15 states, `--within-team --scale 2.0`), first reproducing D15 with the old label; then (A) 12 paired run seeds
  D15 vs D21 weights (fights cleared, deaths per won fight, what ends runs) and (B) the box probe on bosses,
  paired (win rate and deaths: the double edge).
- Fits (301,724 positions, 13,182 battles, within-team, scale 2): reproducing D15 with the old label gives the
  same held-out fit (log-loss 0.4465 vs D15's 0.4473) but different weights where features are collinear:
  `our_alive` and `our_deaths` sum to the party size, so only their difference prices a death: **shipped D15
  0.17, reproduced 0.50, D21 (death = 1/6 win) 0.62** logit units per death (a full-HP mon's `our_hp` ≈ 1.0–1.3).
  D21 held-out log-loss 0.4905 vs group-constant 0.5541 (Δ −0.064; the D15-label Δ is −0.069). `our_ground_hazards`
  zeroed by hand in both, as in D15. So the shipped D15 prices deaths ~3× lower than its own recipe does today;
  arms: shipped D15 (`rr_runs_s1`), reproduced D15 (`eval_weights_d15_repro.txt`), D21 (`eval_weights_d21.txt`),
  12 run seeds each.
- Test B launched (user allowed up to 85% of the machine for ~1 h): box probe on the 10 bosses we have used,
  20 boxes each, seed 12, paired across arms: shipped D15, D21, reproduced D15 (`results/rr_d21b_*`).
- **Test A** (12 run seeds per arm; `results/rr_runs_{s1,d15r,d21}`): fights cleared median 14 / 14 / 13.5, mean
  12.4 / 10.7 / 10.8 (shipped D15 / reproduced D15 / D21); paired D21 − D15 −1.6 ± 1.4, D21 − D15r +0.2 ± 2.1.
  Deaths per won fight 0.58 / 0.57 / **0.46** (se ≈ 0.075 each, so D21 − D15 −0.12 ± 0.10: suggestive, not
  settled; deathless won fights 63 / 62 / 70%; before Misty 0.54 / 0.52 / 0.42). Every arm still stalls at
  Misty (or Brock): the D21 price takes off about a fifth of routine deaths, not enough to arrive at Misty with
  a full box. Runs are noisy (12 seeds, chaotic boxes); the deaths-per-fight number pools ~130–150 fights each.
- **Test B** (10 bosses × 20 boxes, seed 12, the same 200 teams in every arm, paired battle seeds;
  `results/rr_d21b_{d15,d21,d15r}`): boss win 50.1 / 50.2 / 46.6% (shipped D15 / reproduced D15 / D21), deaths per
  mon 0.68 / 0.68 / 0.68, deathless wins 5.0 / 5.4 / 10.1%. Paired D21 − D15: **win −3.5 ± 1.4 pts, deathless
  +5.1 ± 1.0 pts**, deaths/mon −0.004 ± 0.008. Reproduced − shipped D15: +0.2 ± 1.0 win, 0 deaths: the 3× death
  price gap between those two does nothing in play. Per boss D21 − D15 (win): Brock +2.5, Misty 0, Surge −5.8,
  Erika −4.2, Giovanni −8.3, Sabrina −6.7 ± 2.5, Koga 0, Pryce +1.7, Blaine −5.8, Clair −8.3 ± 4.4.
- Meaning: the user's double edge, measured. The D21 label makes the search keep more of a winning team (deathless
  wins double at bosses, routine deaths about −20% in runs) and lose a few more boss fights (−3.5). Its weights
  moved beyond the death term too (our_hp 1.28 → 1.00, safest_switch_in −0.42 → −0.07, ko_first 0.45 → 0.23), so
  "the death price" is not the only change. Runs cleared the same number of fights (they stall at Misty either
  way). Not shipped: the default stays D15. A natural next step is the user's own principle (bosses = play for
  the win, routine fights = keep the team): D21 weights in routine fights, D15 at bosses. The engine reads its
  weights once per process, so that needs a per-call weights switch in the fork. → D22 when the user is back.
- Test A extended to 36 run seeds per arm (seeds 13–36 added, user at lunch): deaths per won fight shipped D15
  0.63 ± 0.05 vs **D21 0.50 ± 0.04** (−0.13, ~2 se; on the same fight in the same seed −0.10 ± 0.07 over 391
  pairs), deathless won fights 61 → 69%. Fights cleared median 13 vs 14, paired +0.4 ± 0.9: no change; runs end
  at Misty (12 / 15 of 36), Rocket grunts, Brock, Archer. So D21's fewer routine deaths are now likely real, but
  not enough to change how far a run gets.

## 2026-11-26 — D19 result: a team model trained on route trainers judges boss teams as well as one trained on bosses

- Data run `results/rr_d19_all` (now `.xz`; raw in `logs/`): 112 trainers × 20 boxes = 2,240 boxes + 6,629 `--alt`
  variants (no trainer failed, ~4 h on 12 workers). `fit_teamwin.py` reads `.xz` logs.
- `--split d19` (train: every non-boss trainer; held out: all gym leaders, Giovanni ×3, E4, Lance, rematches),
  win label: held-out rank corr within boss linear 0.34 / probe 0.31; same-box pairs (3,970 on held-out bosses)
  picks the better team 59% / 58%. Absolute calibration does not transfer (held-out log-loss 0.91 vs constant
  1.14, train 0.28): route trainers are mostly won, bosses about half.
- **Fair comparison** (`rl/teamwin_compare.py`, `results/teamwin_compare_d19.log`): one held-out set for every
  model (all 1,342 logged boxes of Lt. Surge, Giovanni, Sabrina, Pryce, Clair; 713 same-box pairs), linear model,
  win label:

  | trained on | boxes | same-box picks better | pairs with ≥ 50-pt gap | rank corr |
  |---|---:|---:|---:|---:|
  | D18: Brock, Misty, Erika, Koga, Blaine | 5,179 | 64.5% | 77.0% (183) | 0.368 |
  | D19: 107 non-boss trainers | 5,440 | 66.2% | 80.3% (183) | 0.369 |
  | both | 10,619 | 64.4% | 78.1% | 0.319 |

  Utility label: 61.2 / 63.9 / 62.0%. The D19 model is +1.7 to +2.7 points (about 1 se: within noise), and it
  never saw a boss. (The earlier 61% on 125 pairs was mostly set-only pairs; team-swap pairs are easier.)
- Meaning: what the matchup features know about teams transfers from route trainers to bosses: no sign of
  overfitting to the training trainers, but no real gain either. The ceiling is the features (calc-matrix
  summaries): ~65% on same-box choices, ~0.37 rank. More of the same data will not lift it; richer inputs (the
  actual sets / per-mon matchups, not team summaries) or search-labelled data (D18 option C) would.
- Bets: user "overfitting and not getting what we want, but still being better" (70%): overfitting no; not
  what we want yes (~65%); better: marginally (+1.7, within noise) → mostly a miss on "overfit", hit on "not
  what we want". Claude "not clearly better, within ±5 of 60%; rank 0.25–0.35" (70%): within ±5 of the D18
  model yes (both ~65%), rank 0.37 slightly above the range → mostly hit.

## 2026-11-26 — D22: where the D21 death price applies → b (everywhere)

- Options: (a) D21 weights in routine fights, D15 at bosses (per-call weights switch in the fork); (b) D21 everywhere;
  (c) keep D15, attack deaths elsewhere (per-mon death weights, the team picker).
- **user call: b, conf not given.** "If we want to beat the complete gauntlet we might need to go with b."
- Claude: agrees (~65%). Every fight but the last carries its deaths forward; a boss won with 4 dead is close to a
  loss for the run (Test B: 0.68 deaths per mon at bosses in every arm; D21 doubles deathless boss wins 5 → 10%
  for −3.5 pts of wins). Runs cleared the same either way, so (a)'s fork change buys nothing measurable now. Note
  the tension with the D20 principle (the battler plays to win, the chooser carries safety): D20 showed the
  chooser cannot carry it with small early boxes. The principled version is a run-state death price (the last
  fight prices deaths at 0; per-mon value, D14), not a boss/route switch.
- Shipped: `rl/engine_search.DEFAULT_EVAL_WEIGHTS` and `rl/line.py` → `data/rr/eval_weights_d21.txt`. Every
  table before this entry is on D15.

## 2026-11-26 — Rivalry never chosen; typed Hidden Power from Saffron on

- User: never use Rivalry (no genders, bad anyway). `rr_probe.choose_ability` skips a Rivalry slot when another
  exists (a mon rolled Rivalry before capsules keeps it).
- Hidden Power: the NPC next to Saffron's Poké Mart sets its type (the user). The TM is in Silph Co. 1F
  (tms.json, `before_trainer` 44), so from #44 on: `rr_probe.HP_NPC`, the 16 typed entries in the game tables;
  best4 sets type an untyped Hidden Power against the opponent (`type_hidden_power`: summed effectiveness,
  a 4x weakness counts double); `--sets boss` (B′) gets all 16 typed Hidden Powers in its pool and the calc
  picks. Before #44 it stays Dark (all-31 IVs).
- Scope: best4 picks Hidden Power in 5 of 1,200 boss set slots (Test B logs), so the default is barely
  touched; B′ picks it often (smoke, 9 boxes at Koga / Blaine / Pryce: Gengar / Glaceon / Walrein HP Rock,
  Vikavolt HP Ground, Scizor HP Fighting). Runs end before #44 today, so no run number moves yet.

## 2026-11-26 — D23: the team builder's next gear → B (better data)

- Options: (A) richer inputs on the data we have (36 per-mon cells, the sets); (B) better data: more battles per
  team (label noise is 20–50% of a boss's variance at 6 battles) and teams the chooser would not pick (coverage);
  (C) search labels / expert iteration; (D) wire today's judge into the chooser and run.
- **user call: B, 70%.**
- Claude: agrees, with a correction to how B was pitched ("engine battles cost a tenth"): the data runs already
  search on poke-engine. Profile (2 route trainers × 2 boxes, 160 s, one process): greedy chooser 74 s (46%;
  calc-bot battles on Showdown), the searched label battles 37 s (23%; 35 s of it the engine's 20,000-iteration
  MCTS, ~75 ms a decision, ~20 decisions a battle), threat duels 29 s, stage A 19 s. So B's levers are
  (1) skip the chooser for exploration teams (random legal sixes), (2) a cheaper search for labels if it ranks
  teams the same, (3) more battles per team with what that saves. A on B's data afterwards.
- Gate for (2) running: `logs/d23_gate_2k` = D19's boxes and variants at #17 / #18 / #24 / #26 / #50 / #95
  (the trainers with the widest team spread), `--engine 2000`, D15 weights (D19's), compared by
  `rl/label_gate.py` against the D19 labels.
- **Gate result** (`logs/d23_gate_2k`, 6 min on 12 workers, 232 of 240 teams matched to D19): 2,000-iteration labels
  vs D19's 20,000: **r = 0.83, noise-only ceiling ~0.82** (two 6-battle labels of the same team): cheap labels rank
  teams as well as this noise level can tell. Per trainer r +0.73 (#95) to +0.89 (#24). Mean win is lower at 2k
  (−4 to −15 pts: the weaker search plays worse), so 2k labels are for ranking teams, not for absolute win rates.
  Bets: user skip. Claude "within 0.1 of the ceiling, 60%; mean win −3 to −10": hit; the drop ran to −15 (#17).
  Wall time barely moved (the chooser, not the search, was D19's cost).
- Plumbing: `rr_team_probe.py --rand K` (K random legal sixes per box as `box_alt` events with `rand: true`,
  after the `--alt` swaps; skips the threat duels). Timing (`--chooser cover --alt 3 --rand 2 --n-search 16
  --engine 2000`, #50, 4 boxes): 384 battles in 34 s on 8 workers, ~30 s per box-job. Cover six 56% mean win,
  swaps 0–56%, random sixes 0–38% (mostly 0: random teams lose; a floor, low contrast). The slow trainers are
  the ones where the engine falls back to the Showdown search (one Soul-Heart mon at #24: ~8% of decisions,
  ~90% of the time, 7x slower boxes); D19's fallback reasons: trapping volatiles and enemy Rivalry most often.
- User: go on the proposed run, and enemy Rivalry on the engine (`engine_search`: dropped from the fallback gaps;
  the engine has no genders = no effect, the mean of ±25%; Showdown rolled random genders anyway).
- **Data run launched** `logs/rr_d23_data` (seed 23): D19's 112 trainers × 20 boxes, `--chooser cover --alt 3
  --rand 2 --n-search 16 --engine 2000`, D21 weights, 12 workers. Plan: refit `fit_teamwin.py` on D23 labels and on
  D19's, score both on the same held-out boss pairs; spot-check boss labels at 20,000 iterations.
