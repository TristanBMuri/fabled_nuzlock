# Brief: independent research + plan for learning in the Radical Red nuzlocke project

You are one of two independent researchers (the other is a different model). Do your research and
write your plan **on your own**: do not read any other file in this `plan_round/` folder except this
brief. A later round will compare the plans.

## The project (repo: /home/tristan/CodingProjects/nuzlock_rl)

An agent for a Pokémon **Radical Red (RR) hardcore nuzlocke**: clear the game's bosses (gym leaders,
rivals, Elite Four) with no permanent deaths, using only mons caught so far (the "box"), no EVs, no
setup moves, items as they become available. Three linked decisions per boss: **which six (and which
lead) from the box, which sets (moves / item / ability), and how to play the battle.** The enemy is
a fixed, known, scripted AI (the RR / CFRU battle AI, ported to Python and Rust in this repo) with a
fixed, known team. Full information about the boss team is legitimately available to us.

Read first (skim what you need): `CLAUDE.md` (constraints + hard-won lessons), `HANDOFF.md`,
`rl/RR_PLAN.md` (the current plan, incl. "The battler plan after the mirrors"), `rl/README.md`,
NOTEBOOK.md entries from 2026-11-23, `rl/engine_search.py`, `poke-engine/src/mcts.rs`,
`poke-engine/src/genx/evaluate.rs`, `scripts/rr_team_probe.py`, `scripts/rr_mirror.py`,
`ai_util/matchup.py`, `ai_util/calc_bot.py`, `rl/lesson2_ppo.py`.

Where things stand (verify in the code; don't take this on faith):
- **No learned component** in the RR pipeline. Battler = poke-engine MCTS (decoupled UCB1 over
  simultaneous moves, hand-written leaf evaluation, ~20k iterations/move, a death penalty), with a
  Showdown rollout searcher as fallback. Team chooser = damage-calc matchup matrix (greedy / cover)
  plus a "searched" chooser that screens ~24 candidate sixes with 4 battles each. Set builder = rules.
- Results: searched chooser clears Erika 100%; **Koga 8% and Blaine 0%** for the best six found.
  Illegal 85 EVs in every stat → Koga 42%, Blaine 58% (so stat margin matters).
- Known engine bugs (being fixed separately, ~1 h): a mega-stone holder is offered only switches by
  our searcher; permanent Tailwind is scored per remaining turn (stall bias); a won battle with
  deaths scores below an open fight; Ho-Oh's Phoenix Down may revive forever in the engine.
- Searching with the RR AI model as the opponent (instead of generic decoupled search) **lost 13–33
  points**: unexplained, possibly a bug.
- Earlier (vanilla gauntlet, previous attempts): PPO from scratch capped ~14%; behaviour cloning 40%
  in ~1 h; search over the engine 52% → 95% clears; one round of expert iteration gained nothing.
- The open "chicken and egg": judging a team needs a good battler, judging the battler needs good teams.

Constraints: **minimal code** (one person must understand all of it; see CLAUDE.md), one heavy job at
a time, ~10 CPU workers of ~1 GB each, 60 GB RAM, one RTX 5090. Battles cost ~2.7 worker-seconds.

## Your task

1. **Research (general RL / search / ML literature, emphasis 2022–2026, not just Pokémon).** Use web
   search. Find methods that fit *this* problem: planning against a known fixed opponent policy,
   simultaneous-move search, search + learned value/prior at small budgets, learned combinatorial
   selection (picking a six from a box, conditioned on the enemy six), set / item selection,
   curricula and choosing which (box, boss) cases to train on, sample-efficient evaluation of
   candidates under noisy win rates, co-optimising two coupled components, offline data reuse.
   For each source: what it is, what it would give us concretely, its cost, and whether you read
   it or are recalling it (be honest; no invented citations or numbers).
2. **Feature engineering** is a required part. Specify concretely: the battle-state features for
   any value / prior / opponent model; the per-mon, per-set and team-vs-team (matchup) features for
   the builder and set builder; which hand features already exist in the repo (calc-bot damage,
   speed tiers, matchup rows, …) and should be reused; how to test that a feature set is adequate
   (ablations, probes) given the lesson that "the obs is missing X" was undertraining every time.
3. **Write a complete plan**: phases in order, each with goal, method, data, features, rough code
   size and file, machine-time cost, a measurable gate, and what you do if the gate fails. Say what
   you would not do and why. Address the chicken-and-egg loop and the RR-AI-model search question.
   Prefer cheap diagnostics before learning.

Rules: **read-only** on the repo — do not edit, create or commit repo files, and do not run battles
or anything over ~2 minutes of compute. Write your output to the one file named in your task line
below (research notes first, then the plan, with a feature spec table). End with a ≤15-line summary
in your final message.
