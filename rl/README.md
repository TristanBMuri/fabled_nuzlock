# rl/ — learning track

A ladder of small RL lessons on Pokémon battles. It is **off the planner's path**: nothing in
`ai_util/` or `scripts/` imports from here, and deleting the folder changes nothing else.
One file per lesson; each runs end to end in minutes.

## Why a Rust engine

The bots that changed the competitive scene in 2025–26 lean on a fast simulator
([deep-research notes, 2026-11-18](#what-the-field-did-2025-26)). We use
[poke-engine](https://github.com/pmariglia/poke-engine) (`pip install poke-engine`, Rust,
the engine under Foul Play). Measured on this machine:

| | per turn |
|---|---|
| poke-engine `generate_instructions` | **13 µs** (and it returns every outcome with its probability) |
| our Showdown bridge, a rollout turn | ~1.7 ms |
| poke-engine MCTS | 680k iterations/s on 1 core, 2.4M on 8 |

~130x more turns per second. Attempt #3's PPO stalled at 6M Showdown steps; here 6M turns are
~80 s of engine time per core. Whether *that* fixes from-scratch RL is lesson 2's question.

**Caveats.** poke-engine is its own model of the game: Gen 9 build, singles only, and by its
author's account less complete than Showdown. It is not our `gen8customgamenogimmicks`, nor
the RR / BDSP mods. Train on it, but any claim about the planner is checked on Showdown.

## The ladder

### 1. Tabular Q-learning, one matchup — `lesson1_qlearning.py` (done, ~10 s)

Garchomp vs a random-move Rotom-Wash. The agent has to learn, from +1 / -1 alone, that
Earthquake does nothing (Levitate), Will-O-Wisp halves its attack, and that Swords Dance pays.
First run (seed 0, 30k episodes): Q picks Swords Dance at the start (+0.90 vs Dragon Claw
+0.81), the same move as poke-engine's expectiminimax; win rate 95.9% vs 94.8% for always
Dragon Claw and 64% for random.

Exercises, each a one-line change:
- `--episodes 2000`: what does an unconverged table look like? (Lesson from attempt #3:
  "the obs is missing X" was undertraining every time.)
- `--gamma 0.9`: the win is ~5 turns away; does discounting change the policy?
- Make `opponent()` always use Will-O-Wisp. Does Swords Dance stay the right opening?
- Drop the burn flag from `key()`. The table can no longer tell burned from not: what breaks?
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
