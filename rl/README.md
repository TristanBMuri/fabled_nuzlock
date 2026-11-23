# rl/ — learning track

A ladder of small RL lessons on Pokémon battles. It is **off the planner's path**: nothing in
`ai_util/` or `scripts/` imports from here, and deleting the folder changes nothing else.
One file per lesson; each runs end to end in minutes.

## Why a Rust engine

The bots that changed the competitive scene in 2025–26 lean on a fast simulator
([research notes below](#the-2026-bot-wave-augnov-2026)). We use
[poke-engine](https://github.com/pmariglia/poke-engine) (`pip install poke-engine`, Rust,
the engine under Foul Play). Measured on this machine:

| | per turn |
|---|---|
| poke-engine `generate_instructions` | **13 µs** (and it returns every outcome with its probability) |
| our Showdown bridge, a rollout turn | ~1.7 ms |
| poke-engine MCTS | 680k iterations/s on 1 core, 2.4M on 8 |

~130x more turns per second. Attempt #3's PPO stalled at 6M Showdown steps; here 6M turns are
~80 s of engine time per core. Whether *that* fixes from-scratch RL is lesson 2's question.

**Our fork, built as Gen 8.** The PyPI wheel is a Gen 4 build (Shadow Ball does half damage
to Steel) and 0.0.48 lags upstream. `poke-engine/` (gitignored) is upstream `main` (f7b147f)
plus `rl/poke-engine.patch`, the conformance fixes below; `pyproject.toml` builds it with the
`gen8` feature (needs cargo, in the box). After editing the Rust: `uv sync
--reinstall-package poke-engine` (~30 s), and refresh the patch with `git diff f7b147f >
../rl/poke-engine.patch` inside `poke-engine/`. Same speed: 11-12 µs per turn.

## Can we trust it? — `engine_diff.py` + `convert.py`

`convert.py` turns a Showdown battle (the bridge's `dump` op) into a poke-engine State and
lists what it cannot carry over (poke-engine silently maps unknown names to NONE, so every
name is checked). `engine_diff.py` plays the same turn on both: poke-engine's exact outcome
table against 200 Showdown replays with fresh RNG, positions from random-move gen8 random
battles.

    uv run python rl/engine_diff.py --positions 1000     # ~3 min, logs/engine_diff.jsonl

`--seed` fixes the whole position set (the random teams too), so runs compare fix by fix.
Flagged positions keep their full dump in the log, to replay on poke-engine.

| 1000 positions | 0.0.48 wheel | upstream main | + our patch |
|---|---|---|---|
| fully agree | 81% | 69-71% | **97%** |
| not convertible | 8% | 25-27% | 0% |
| disagree (of the compared) | 11% | 6% | **3%** |

(The first two columns had unseeded teams; upstream main's "not convertible" is the same gaps
hit more often in that sample: Light Clay, Scope Lens, Leek, pinch berries.)

Fixed in the patch (each was a cluster in the diff): Defog's evasion drop; Sheer Force + Life
Orb recoil; Sticky Web vs Defiant / Competitive; Effect Spore odds and powder immunity;
Soul-Heart; Synchronize; Speed Boost on the switch-in turn; both sides switching (in the
outgoing mons' speed order, each switch-in complete: weather, Intimidate); Fishious Rend /
Bolt Beak vs a switch-in; Rest at full HP; crit stages (Scope Lens, Razor Claw, Super Luck,
Focus Energy, Leek); Light Clay and the pinch berries. Found by the BDSP gauntlet (low
levels, where rounding decides KOs): damage rolls in Showdown's order (the random factor on
the integer base, then STAB / type: a Lv 14 Tackle into 23 HP KOs on 1 roll of 16, not 5)
and their true mean; a non-immune hit does at least 1; drain rounds; Double Team / Minimize
raise evasion and accuracy counts the target's evasion (never-miss moves listed from
Showdown's dex). Fixed in the converter: last move (Encore), Wish, Future Sight, Healing
Wish, Slow Start, Rest's fixed sleep, "has not moved since switching in" (Fake Out), Bind
and confusion on their last turn.

Still open: Rollout / Ice Ball (no lock-in or doubling), Rivalry (poke-engine has no
genders), Ditto (no Imposter / Transform), Struggle (poke-engine refuses it), Doom Desire,
Rage, two-turn moves mid-charge, and a tail of single cases. Soul-Heart only counts its own
KOs. The first two are the gauntlet's big ones (see the router).

Approximations by design: Showdown fixes how long sleep, confusion and Outrage last when they
start; poke-engine rolls a chance each turn. The converter maps the fixed-length ones
(Taunt, Encore, Yawn, Slow Start, Rest) and the field timers. poke-engine keeps one damage
value per outcome, so its mean HP is good to ~0.5 HP per hit; `engine_diff` flags an HP gap
only from 5% of max HP and 1.5 HP.

## Router — `route.py`

Run a turn on poke-engine only where it agrees with Showdown; everything else stays on
Showdown. Two levels:

- `route(bridge, p1_team, p2_team, cache)` once per matchup, cached in `logs/route.jsonl`
  (the router's own log): every name known to poke-engine, then a probe of 50 random
  positions of this matchup, each replayed 200 times on Showdown (~4 s). Any disagreement
  sends the whole matchup to Showdown (`--tolerate` relaxes it).
- `turn_gaps(dump)` every turn, microseconds: a known gap (`GAPS`) on either active mon sends
  that turn to Showdown. A gap on the bench does not matter until it is in. The converter's
  `issues` do the same for a state it cannot carry.

    uv run python rl/route.py --harvest data/teams/gauntlet_harvest/battles_best4.jsonl --trainers 60

First 60 BDSP trainers (one harvested matchup each, `gen8customgamenogimmicks`):
**46/60 matchups pass**, and 45% of their probe turns run on poke-engine; the rest have
Rollout (645 positions) or Rivalry (462) on the field: early-game Bidoof and Shinx. The 14
failures are single cases (Rage, Disable, Rock Throw vs Protect, …).

Nothing in the planner calls the router yet: its search runs Showdown forks with `calc_bot`
rollouts, which read poke-env mirrors, not poke-engine states. The router is for code that
already runs on poke-engine (the lessons below); bringing it to the planner means a second
search backend.

What this means: fine for learning and for search as a first pass; a planner decision that
matters is still confirmed on Showdown. The format here is vanilla gen8 random battles; our
custom format and the RR / BDSP mods are not covered.

## Radical Red on poke-engine (started 2026-11-23)

The battle always runs on Showdown (the RR fork); the *search* behind each player decision runs
on poke-engine where it is trusted, and on the Showdown rollout search everywhere else.

- **Build.** Cargo feature `rr` = gen9 + the gen9rr mod. A second Python package,
  `poke_engine_rr` (`poke-engine/poke-engine-py-rr`, same bindings source), sits next to the
  Gen 8 `poke_engine`; `uv sync --reinstall-package poke-engine-rr` after a Rust edit.
- **Data.** `rl/rr_engine_data.py` diffs the mod against the gen9 it inherits (both in the fork)
  and writes `poke-engine/src/rr_data.rs`: the plain data of the 149 existing moves the mod
  changed. It also prints the worklist of what is code, not data (`--json
  logs/rr_engine_gaps.json`): 27 new moves, 28 new abilities, 30 changed abilities, 20 changed
  moves, 8 changed items, 5 changed conditions (freeze, hail, ...), 76 new species.
- **Mechanics** (Rust, `#[cfg(feature = "rr")]`, tests in `poke-engine/tests/test_rr_*.rs`), built
  2026-11-23 by delegated agents on worktree branches of the fork, merged into `nuzlock`:
  `rr-moves` (Gemini Neon: the 27 new moves, the changed moves / conditions / items, 76 new
  species, Oran Berry, Berry Juice; freeze is RR's frostbite, hail is snow-like), `rr-abilities`
  (Gemini Neon: all 28 new and 30 changed abilities), `rr-mega` (Opus: mega evolution from the
  mod's 66 stone megas; a stone holder megas on its first move, as the RR AI does; Gen 9 turn
  order on the mega turn), `rr-conformance` (Gemini Neon: the diff's tail; Speed Boost, Anger
  Shell after a multi-hit, Harsh Sun / Heavy Rain cases, permanent terrain, the terrain seeds,
  Sand Veil / Snow Cloak, Chloroblast, Covet, `switch:<slot>` choices for two mons of one species),
  `rr-gaps` (Opus: Rollout / Ice Ball lock-in and doubling, for every gen, gen8 suite green;
  Mega Rayquaza; the converter carries two-turn moves mid-charge and Protosynthesis / Quark
  Drive via the dump's `volatileData`). Left: Rivalry (no genders in poke-engine), a Booster
  Energy boost under sun / Electric Terrain (issue `volatile:<v>-booster`). Under `rr`, 38
  vanilla tests in test_battle_mechanics.rs and 9 lib tests fail on purpose (mechanics RR
  changes); the gen8 build passes everything.
- **Converter.** `convert.use_rr()`: typed Hidden Power (Showdown keeps the type in `hpType`)
  becomes poke-engine's 60 BP id; Z crystals (the RR AI never Z-moves) become no item; mega
  stones stay, a mega forme sets `mega_evolved`; `mega:pending` is left only for Mega Rayquaza.
  Arena-pinned side conditions (Koga's Tailwind) dump as 99 turns; `snow`; `recharge` = `none`.
- **Conformance.** `uv run python rl/engine_diff.py --rr --positions 400 --samples 100`:
  positions from every hardcore boss in trainer order (sampled player party vs the sheet set,
  arena on), 4 per boss. 2026-11-23: before the mechanics 45% not convertible and 89% of the
  rest agreeing; after the three merges 2% / 92%; after `rr-conformance` **2% not convertible,
  97% agree** (the rest are single cases: Metronome, HP means); after `rr-gaps` **0% not
  convertible, 97% agree**.
- **Search.** `rl/engine_search.py` `EngineSearcher(fallback, iterations, death=60)`: poke-engine
  MCTS (both sides search, full information) when the position converts and no known gap is on
  either active mon, else the fallback (`ai_util.evaluate.Searcher`). In the probe:
  `rr_team_probe.py ... --engine 20000` (box rows get `engine`: decisions on the engine / on
  Showdown and why). Under `rr` the evaluation charges `POKE_ENGINE_DEATH` points per fainted mon
  of ours and a won battle is scored by the evaluation (a win with deaths < a clean win).
  Misty, the same 10 boxes as `results/rr_sheet_boxes`: Showdown search 52% wins / 0.69 deaths
  per mon (172-487 s a box); engine, death 0: 48% / 0.77 (100k iterations: 43% / 0.76, so not
  depth); with the won-terminal change, death 0: 53% / 0.73, **death 60: 52% / 0.66 at 10-58 s a
  box**; death 200: nearly every box lost (too afraid to trade). On the merged engine, both bosses
  (`results/rr_engine_merged_boxes10.*`, 96-100% of searched decisions on the engine): **Brock 75% /
  0.44 / 17% deathless** (Showdown search 72% / 0.53 / 12%), **Misty 53% / 0.65** (52% / 0.69).
  10 boxes x 6 battles: a lead, not a result. **The 30-box table** (`results/rr_engine_boxes30.*`, the
  same boxes and teams as `results/rr_sheet_boxes`): Brock 70% / 0.49 / 16% deathless, Misty 51% /
  0.68 / 4%; paired per box vs the Showdown search: wins -2.8 +- 10.3 / -2.2 +- 9.4 points, deaths
  per mon -0.04 +- 0.09 / -0.04 +- 0.05 (95%). **A tie, in 3 min on 4 workers** (the baseline: 29
  min on 10).
- **The RR AI in Rust** (`rr-ai`, Gemini Neon: `src/genx/rr_ai.rs`, bindings `rr_ai_choice`,
  `monte_carlo_tree_search_vs_ai`): the same choice as `ai_util/rr_ai.py` on 90.3% of 600 boss
  decisions (moves 92%, forced switches 90%, voluntary switches 78%; `rl/rr_ai_diff.py`,
  `results/rr_ai_diff.log`; the Python AI breaks ties at random). **Searching against it is worse**
  (`--engine-opp ai`, `results/rr_engine_ai_boxes30.*`, paired vs the minimax engine search: Brock
  wins -13 +- 10 points, Misty -33 +- 12; no turn-limit losses, long passive battles of 70-90
  turns): the search exploits the model's errors. Not adopted; `minimax` stays the default.
- **All six gyms, 30 boxes, engine search** (`results/rr_engine_gyms30.*`, 10 workers): Brock 66% /
  0.52 d/mon, Misty 47% / 0.68, Lt. Surge 44% / 0.76, Erika 22% / 0.88, **Koga and Blaine 0% /
  1.00**. The Showdown search also loses all 10 Koga boxes (`results/rr_showdown_koga10.*`): a wall
  for the sampled boxes and the planner (Tapu Lele, Hoopa-U, Chi-Yu at 68), not the engine.

### The workload guess (2026-11-21, not measured)

What it would take to run RR search on the Rust engine. A guess, calibrated on the Gen 8
conformance sprint (one day, 2026-11-18: a 1177-line patch plus `convert.py` / `engine_diff.py`,
81% → 97%) and on the size of `pokemon-showdown-rr/data/mods/gen9rr`:

| | entries | with handlers (custom code) |
|---|---|---|
| species (`pokedex.ts`) | 905 | 0 (stats and types only) |
| moves | 488 | ~70 |
| abilities | 60 | ~118 |
| items | 220 | ~40 |
| conditions / `scripts.ts` | 99 + 625 lines | megas, `runMove` overrides |

poke-engine: 63k lines of Rust, a `gen9` feature exists, **no mega evolution as a choice**.

| Step | Work | Guess |
|---|---|---|
| 1. Data | script the mod's TS → Rust tables (stats, types, move power / accuracy / type, new names) | 1–2 days |
| 2. Mechanics | ~200 move / ability / item effects in Rust with tests, many reuse existing ones; arena rules (permanent weather) easy; megas: pre-evolved boss = cheap, a real choice type = not | 1½–3 weeks |
| 3. Conformance | `convert.py` + `engine_diff.py` on RR dumps, iterate to ~95% on boss positions | 3–5 days |
| 4. Enemy AI | the planner's enemy is scripted: `ai_util/rr_ai.py` (668 lines, CFRU) and the `calc_bot` parts it uses in Rust; a Python callback per node eats the 130x | 3–5 days |
| 5. Planner | the second search backend (above) and paired runs against the current tables | 3–5 days |

**Total 2½–4½ weeks** to planner-usable; steps 1–3 alone (a trustworthy engine) 2–3½ weeks.
The ~200 handlers decide where in the range it lands.

**Cheaper first step:** the RR work only reaches Brock and Misty. Cover only the early
bosses' species / moves / abilities and let `route.py` fall back to Showdown for the rest:
~4–7 days. Firm it up first (~1 h): count what `convert.py` cannot carry over on Brock / Misty
dumps.

**Caveat:** the RR sim itself is unverified against the ROM (TODO.md A1 / A2); a Rust copy
inherits its errors. Do it after, or alongside, that check.

## The ladder

### 1. Tabular Q-learning, one matchup — `lesson1_qlearning.py` (done, ~10 s)

Garchomp vs a random-move Rotom-Wash. The agent has to learn, from +1 / -1 alone, that
Earthquake does nothing (Levitate), Will-O-Wisp halves its attack, and that Swords Dance pays.
Run on the Gen 8 build (seed 0, 30k episodes): Q values Swords Dance highest at the start
(+0.98 vs Dragon Claw +0.88), the same move as poke-engine's expectiminimax; win rate **98.6%**,
vs 98.0% for always Dragon Claw and 74% for random. (On the Gen 4 wheel it was 95.9 / 94.8 / 64.)

**Bug fixed 2026-11-22:** the burn flag in `key()` compared against `"brn"`, but the engine
reports `"BURN"`, so it never fired: every number before this date (97.4%, Q +0.97 / +0.86)
came from a table that could not tell burned from not. With the flag working, Q-learning
beats always Dragon Claw instead of trailing it. That was the "drop the burn flag" exercise
run by accident. `--episodes 2000` gives 94% (the unconverged table).

Exercises, each a one-line change:
- `--episodes 2000`: what does an unconverged table look like? (Lesson from attempt #3:
  "the obs is missing X" was undertraining every time.)
- `--gamma 0.9`: the win is ~5 turns away; does discounting change the policy?
- Make `opponent()` always use Will-O-Wisp. Does Swords Dance stay the right opening?
- Drop the burn flag from `key()`. The table can no longer tell burned from not: what breaks?
- Rerun with other seeds: is 98.6 vs 98.0 a real gap, or noise? (Hint: 2000 battles at
  98% have a standard error of ~0.3 points.)
- Replace Q-learning's `max` target with the action actually taken next (SARSA). Compare.

### 2. PPO from scratch on 3v3, with a cockpit — `lesson2_ppo.py` + `cockpit.py` (2026-11-22)

3v3 with switching on poke-engine (Gen 8 sets, level 50, `POOL` in the file). A 156k-parameter
MLP (two layers of 256) reads a 344-number observation (HP, status, types, stats, boosts,
hazards, and per move its type, power, category, accuracy, effectiveness, STAB) and picks one
of 7 actions (4 moves, 3 switch slots; illegal ones masked). PPO, reward +1 / -1 at the end.
Everything goes to `logs/rl/<run>.jsonl`; the cockpit draws it live:

    uv run python rl/cockpit.py                                   # http://localhost:8766, leave it open
    uv run python rl/lesson2_ppo.py --run first                   # vs the greedy script, ~3.5k decisions/s
    uv run python rl/lesson2_ppo.py --run mix --opp greedy,mcts   # ~1k decisions/s (MCTS is 1.4 ms a move)
    uv run python rl/lesson2_ppo.py --run pool --pool             # 3 random mons a side each battle

The cockpit (one HTML page, polls every 3 s): eval win rate against random / greedy / MCTS,
training win rate per opponent, HP left when winning, battle length; the turn-1 policy over
time (lesson 1's "Q at the start"), the value at the opening, switch rate, action mix;
entropy, KL, clip fraction, explained variance, both losses, gradient norm, throughput; and
at every eval one whole battle turn by turn with the policy's probabilities and value.

Opponents (`--opp`, comma-separated, split across the 64 envs): `random`; `greedy` (highest
power × STAB × effectiveness, switches only when forced); `mcts` (poke-engine's MCTS, which
sees both full teams: the strong reference); `self` (the current policy, sampled).
`get_all_options` (the legal choices) is a binding we added to the fork.

**First result (fixed teams, `--opp greedy`, 300k decisions, 1.7 min):** 100% vs greedy and
97% vs random after 40k decisions, but **0–22% vs MCTS** (50 battles, noisy) the whole run.
The greedy script itself wins ~30% vs MCTS. The policy learned to beat *that script* (always
Stone Edge on turn 1, the right answer to a Volcarona that never switches), not to play. This
is attempt #3's ceiling seen live: a fixed scripted opponent is a narrow teacher.

**`--opp greedy,mcts` (MCTS 100 in training, 1M decisions, 18 min): 70–88% vs MCTS 1000.**
Same net, same budget; only the opponent changed.

#### Comparing runs: the levers and the cockpit

Pick several runs in the cockpit (click the chips) and every chart overlays them, one color
per run; a "What differs" table lists only the settings that differ, the inputs, and the
mean of each run's last three evals. Eval win rates carry a 95% band (200 battles: ±7 points
at 50%): overlapping bands mean the difference may be noise. The x axis switches between
decisions and minutes (BC needs an MCTS call per decision, so it is 2–3x slower per decision;
compare on minutes, and cap runs with `--minutes`). Every run is evaluated the same way:
random, greedy, and MCTS at `--eval-mcts 1000` iterations, 200 battles each, fixed seeds.

| lever | values |
|---|---|
| `--algo` | `ppo`; `a2c` (one gradient step per batch, value baseline); `reinforce` (same, no baseline, Monte Carlo return); `bc` (imitate MCTS `--mcts-iters` on the positions the net reaches itself = DAgger; the reward is not in the policy loss at all) |
| `--obs` | input groups: `core` (HP, alive, active), `status`, `types`, `stats`, `boosts`, `field` (hazards, screens, sub, Leech Seed, weather), `moves` (our active's) = the default; extra: `foe_moves` (their active's set: hidden information in real play), `matchup` (hand-made: best effectiveness each way, who is faster), `species` (identity). `all`, a list, or `+x,-y` |
| `--opp` | `random`, `greedy`, `mcts`, `self`, comma-separated |
| others | `--pool`, `--holdout`, `--init`, `--shape`, `--ent`, `--lr`, `--epochs`, `--hidden`, `--gamma`, `--seed`, `--note` |

**Starting runs from the cockpit.** "+ new run" (or "new run with these settings" on a run's
page) opens a form prefilled from a run's config; starting it pauses whatever is training
(SIGTSTP: the trainer logs `pause`, stops itself, and leaves the paused time out of `--minutes`
and the minutes axis). A paused run keeps its memory and uses no CPU; "resume" pauses the
others. A run started from the terminal pauses the training one too, and refuses a name
that exists (logs and nets are never overwritten). "Branch from this net" starts from that run's last checkpoint (`--init`, same `--obs`
and `--hidden`, fresh optimizer): e.g. `bc` for 10 min, then branch into `ppo` and see whether
RL gets past imitation. The note field asks what you expect: write the prediction first.

**Seeds and the queue.** The form's "seeds" field (`0,1,2`) makes one run per seed
(`<run>_s0`, …); the first starts, the rest wait in the queue (shown in the header), which
starts the next run whenever nothing is training; "add to queue" queues them all. The queue
lives in the cockpit's memory: restarting the cockpit forgets it. In the compare view, runs
that differ only in seed become one line (the mean) with a band from the lowest to the highest
seed. That band is the noise floor: two settings differ only if their seed bands separate.
The 95% bands on single runs cover the luck of the 200 eval battles, not training luck.

**Held-out teams (`--pool`).** Our team is one of the 84 trios of POOL; `--holdout 0.2` (the
default) keeps 17 of them out of training (fixed split, the same for every run; each mon still
appears in other trios). Every eval runs twice: training teams (`mcts`, …) and held-out teams
(`mcts held`, …). Held-out well below training = it memorized team combinations. Pool runs from
before 2026-11-22 trained on all trios.

**What MCTS would do.** The turn-by-turn battle shows next to the net's policy MCTS's visit
shares for our side (`--eval-mcts` iterations) and highlights the turns where their favourite
moves differ, with the net's value estimate next to the final result.

At the end of a run the finished net is re-evaluated with one input group zeroed at a time
(100 battles each vs greedy and MCTS): the cockpit's "What it looks at" shows the drop.

**Comparison 1 (2026-11-22, fixed teams, `--opp greedy,mcts`, 12 min each, mean of the last
3 evals vs MCTS 1000):**

| run | vs MCTS | decisions | notes |
|---|---|---|---|
| ppo, default inputs (344) | 74% | 573k | |
| ppo, `--obs all` (511) | 76% | 536k | |
| ppo, `--obs core,moves` (124) | 71% | 615k | |
| a2c | **2%** | 586k | one step per 4096 samples: too slow, beats greedy only |
| bc (imitate MCTS 100) | 64% (peak 77%) | 276k | agrees with the teacher on 76% of moves |

Information made no difference here, and the importance test says why: the default net
loses ≤ 3 points when any one group is zeroed. One fixed matchup can be recognized from
almost any input, so the information is redundant. The minimal net leans on `core` (−37
points without it). Information can only matter with `--pool` (many matchups). BC got there
on half the decisions; it uses `moves` (−57) and `core` (−49) heavily. The algorithm made
the biggest difference: PPO's 4 epochs over minibatches vs a2c's single step.

**Comparison 2 (`--pool`: 3 random mons a side from 9, 20 min each, mean of the last 3
evals, also on random teams):**

| inputs | vs MCTS | vs greedy | vs random | decisions |
|---|---|---|---|---|
| `all` (511) | 23% | 54% | 88% | 874k |
| default (344) | 16% | 50% | 82% | 949k |
| `core,moves` (124) | 19% | 45% | 81% | 1.02M |

Random teams are far harder: 16–23% vs MCTS where fixed teams reached ~75%. With the
matchup changing, the default net now uses its inputs (−17 points without `moves`, −11
without `core`, −10 without `types`). More information ranks first against every opponent,
but by 4–9 points, about one band width, so treat it as a lead, not a result: rerun with
another `--seed`. These evals use the training distribution (no held-out team combos yet).

Exercises:
- Longer `--pool` runs: are the curves still rising at 20 min? Hold out some team combos.
- `--algo bc --mcts-iters 1000`: a stronger teacher. Does the student follow it?
- `--opp self`: self-play. Train win stays near 50%; eval vs MCTS is the real signal.
- `--algo reinforce` vs `a2c`: what does the value baseline buy?
- `--ent 0.05` (attempt #2's value) vs 0.02; `--grad-clip 0.5` to watch learning silently stop.
- `--shape 1.0`: reward HP damage too. Faster? Does it change what it learns?
- Two seeds of the same config: how far apart are they? That is the noise floor for every
  other comparison.

### 3. Search teacher → small network, then search on top (next)

The Numbrion recipe. Its first half is now lesson 2's `--algo bc`. Next: a stronger
teacher, the value head trained to predict the winner, and the network playing *with* a
shallow search on top. Lesson 2's MCTS curve is the bar to clear.

### 4. AlphaZero-lite — `lesson4_az.py` (later)

Close the loop: pUCT search guided by the network's policy and value, played against itself,
and the search's visit counts become the next network's targets. This is Jaxcalibur's
method (self-play RL + pUCT, "similar to AlphaGo Zero"), the bot that held #1 in gen9 randbats.
Its value head is a learned win/death probability: the quantity the planner already asks for,
so this is where the track could rejoin the main project.

## The 2026 bot wave (Aug–Nov 2026)

From the Smogon [PS! Bot Projects](https://www.smogon.com/forums/forums/ps-bot-projects.1059/)
forum, read 2026-11-18. All results are self-reported; none of these bots is open source, and
most no longer play on the ladder (some still take challenges).

- **Jaxcalibur** ([thread](https://www.smogon.com/forums/threads/jaxcalibur-a-gen-9-randbats-bot-that-reached-1-on-the-ladder.3787537/),
  [write-up](https://jaxcalibur.github.io)): gen9 randbats, #1, peak 2557 Elo and 95.3 GXE (Foul Play:
  2341 / 88). A network trained by self-play RL plus pUCT search, "similar to AlphaGo Zero",
  over millions of games, in JAX. Over 6 months on one H100 (that figure is from a summary,
  not checked against the post). Left the ladder on 28 Aug, takes randbats challenges. Lost
  a 5–3 exhibition match to the human MichaelderBeste2 on 4 Oct.
- **Numbrion** ([thread](https://www.smogon.com/forums/threads/numbrion-reached-rank-1-in-gen-9-random-doubles-battle.3790146/)):
  gen9 random doubles, #1, peak 2136 Elo, average GXE 88.5. Copied Jaxcalibur's architecture.
  A search-backed model generates the training examples instead of PPO, and the bot searches
  again at play time. "Even 0.8M parameters was enough", "trained in about 10 hours on a
  macbook pro m1", about a week of work. Advice: "rewrite the showdown engine in rust or
  another fast language."
- **Others in the same forum:** MissingNo.Zero (pure self-play RL, #1 on six random-battle
  ladders), Pokerus (self-play plus a team builder, #1 on gen9 OU), and the VGC bots
  mikumiku37 (self-play), nessie123, Mew's Truck and RotomDotExe. Ninjask is an open-source
  Champions VGC simulator with Showdown parity, built for training bots.

The common thread: a small network plus search over a fast engine, trained by self-play or
by imitating a search teacher. That is lessons 3 and 4.

## What the field did (2025 – Mar 2026)

From the deep-research run on 2026-11-18; 19 sources; all 25 claims checked held up. Most
figures are self-reported by the authors.

- **Metamon** (UT Austin, [arXiv 2504.04395](https://arxiv.org/abs/2504.04395)): transformer
  policies (15M–200M parameters), no search. Trained by imitation, then offline RL on ~475k
  human replays rebuilt into first-person views, then offline fine-tuning on self-play data.
  Top 10% of the human ladder in Gen 1–4 OU; best: Gen 1 OU GXE 79.9%, peak #31.
- **Foul Play** ([blog](https://pmariglia.github.io/posts/foul-play),
  [repo](https://github.com/pmariglia/foul-play)): no learning. Root-parallel MCTS with
  decoupled UCT over sampled guesses of the hidden opponent state, on poke-engine. Moving from
  expectiminimax to MCTS took its search from ~5 to 10+ turns deep. Self-reported:
  gen9randombattle GXE 88%, peak 2341 (top 50); #1 in randbats blitz and Battle Factory.
- **PokéChamp** ([arXiv 2503.04094](https://arxiv.org/abs/2503.04094)): LLMs inside minimax.
  Projected Gen 9 OU 1300–1500 Elo; strong, not top-human.
- **PokéAgent Challenge, NeurIPS 2025** ([arXiv 2603.15563](https://arxiv.org/abs/2603.15563)):
  every top battling entry was RL or search; FoulPlay won Gen 9 OU, a Metamon-based entry won
  Gen 1 OU. The organisers (Mar 2026): "competitive with top human players", "less than
  superhuman". Later 2026 forum self-reports of bots at ladder #1 (Jaxcalibur in gen9randbats,
  MissingNo.Zero via self-play RL) are unverified.
- **pkmn/engine** ([repo](https://github.com/pkmn/engine), Zig): >1000x Showdown on its own
  Gen 1 benchmark, but only Gen 1–2.
- **Datasets** ([PokéAgent track 1](https://pokeagent.github.io/track1.html)): metamon-parsed-replays
  (3.5M+ first-person trajectories), pokechamp-replays (2M battles). The option if lesson 3
  stalls: offline RL from human play instead of self-play.
