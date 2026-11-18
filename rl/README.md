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

## The ladder

### 1. Tabular Q-learning, one matchup — `lesson1_qlearning.py` (done, ~10 s)

Garchomp vs a random-move Rotom-Wash. The agent has to learn, from +1 / -1 alone, that
Earthquake does nothing (Levitate), Will-O-Wisp halves its attack, and that Swords Dance pays.
Run on the Gen 8 build (seed 0, 30k episodes): Q values Swords Dance highest at the start
(+0.97 vs Dragon Claw +0.86), the same move as poke-engine's expectiminimax; win rate 97.4%,
vs 98.0% for always Dragon Claw and 74% for random. (On the Gen 4 wheel it was 95.9 / 94.8 / 64.)

Exercises, each a one-line change:
- `--episodes 2000`: what does an unconverged table look like? (Lesson from attempt #3:
  "the obs is missing X" was undertraining every time.)
- `--gamma 0.9`: the win is ~5 turns away; does discounting change the policy?
- Make `opponent()` always use Will-O-Wisp. Does Swords Dance stay the right opening?
- Drop the burn flag from `key()`. The table can no longer tell burned from not: what breaks?
- Q prefers Swords Dance yet loses to always Dragon Claw by 0.6 points. Is that noise
  (rerun with other seeds), the coarse `key()`, or is Swords Dance worse against a
  *random* opponent than against the worst-case one expectiminimax assumes?
- Replace Q-learning's `max` target with the action actually taken next (SARSA). Compare.

### 2. Search teacher → small network, 3v3 — `lesson2_distill.py` (next)

Random 3v3 sets from a small pool on poke-engine. poke-engine's own MCTS plays both sides at
a shallow setting and labels every turn (the move it picked, who won); a small MLP (~1M
parameters, the size Numbrion says was enough for #1) learns to imitate the move and predict
the winner. Adds torch back. Then the network plays alone, and plays with a shallow search on
top. This is the Numbrion recipe ("you need a teacher ... instead of PPO"), and it answers why
attempt #3's PPO stalled. Exercise: run plain self-play PPO on the same env with the attempt #2
values (grad_clip 5.0, entropy 0.05) and compare the curves. Evaluate on Showdown against the
calc bot via `ai_util/sim`.

### 3. AlphaZero-lite — `lesson3_az.py` (later)

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
by imitating a search teacher. That is lessons 2 and 3.

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
  (3.5M+ first-person trajectories), pokechamp-replays (2M battles). The option if lesson 2
  stalls: offline RL from human play instead of self-play.
