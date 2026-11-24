# Round 3: last pass before the final plan (independent)

You are one of two independent reviewers (the other is a different model). Do not read the
other reviewer's round-3 file (`final_fable.md` / `final_neon.md`). You may read everything else in
this folder (BRIEF.md, fable.md, neon.md, lead.md, *_review.md) and the repo.

The merged plan v2 is the top section of `/home/tristan/CodingProjects/nuzlock_rl/rl/RR_PLAN.md`
("The plan v2 (2026-11-24)"). Read it first, then `CLAUDE.md` (minimal-code rule, and the new
section "Working with the user on RL decisions").

## What the user added after reading plan v2 (address every point)

1. **Movepool for sets.** The user asked how many / which moves the planner should have available.
   Clarified: the *legal movepool the set search may choose 4 moves from*, instead of the fixed rule
   `encounter_moves(..., "best4")` in `scripts/gauntlet.py:193`. Legal sources in their hardcore run
   (user's answer): relearner (any level-up move ≤ current level), reusable TMs obtained so far,
   move tutors as reached. **No egg moves.** Questions: how big is that pool per mon at Koga/Blaine
   (measure from `data/rr` if cheap), should set search choose moves (and how, given the pool size:
   prefilter by calc damage vs this boss team, then mutations in phase 3?), and what does it do to
   search cost and the feature spec.
2. **Visualise exactly how the planner decides.** The user wants to see how the team draft works
   (e.g. which enemy threat each pick answers, which alternatives lost and why, counters), what
   information each component uses, and per-battle what the search thought (visit counts, values,
   predicted enemy move, deaths expected). This is a sanity tool for them. Propose the minimum set
   of logged fields + views, reusing what exists (`ui/` run dashboard, `rl/cockpit.py`,
   `<out>.events.jsonl`, phase-0 per-turn logging). Say what is cheap vs expensive.
3. **Showdown cross-check of bad battles.** Very bad battles (unexpected losses / deaths) should be
   re-run or replayed in the Showdown sim (`ai_util/sim*`, `rl/engine_diff.py`) instead of the Rust
   engine, to see whether the cause is an implementation bug. Design this check concretely (which
   battles are selected, what is compared: same seed replay of the chosen actions? per-turn damage
   / state diff? win rate on Showdown vs Rust for the same team?), and its cost.
4. **A "line" planner (battle flowchart).** Top nuzlockers write a "line" for a boss: a flowchart of
   the fight with branches on crits, misses, secondary effects, speed ties and the AI's random
   choices ("if Koga's Crobat crits, switch to X; else Y"). Design a separate module that produces
   such a tree from our engine + the exact RR AI model (the enemy AI is a known script, so given our
   policy the fight is a single-agent stochastic problem: expectimax / chance nodes). Include: how
   branches are pruned (probability threshold, merging equivalent states), how it reuses the MCTS
   or replaces it, output format (text tree / graph), how it reports the death and loss probability
   per branch, the cost, and how it could later help the autonomous bot. User decided: build it
   early, **right after phase 1** (it needs the exact AI model and oracle from phase 1).
5. **End use.** User decided: the system is first an **advisor** for the user's own playthrough
   (it plans teams, sets and lines; the user plays the real ROM), autonomy later. Say what this
   changes in the plan (e.g. explanations and lines matter more; what is out of scope for now).
6. **ML ambition check.** The user worries that closed models under-use ML out of caution. Assess
   honestly whether plan v2 is under-ambitious: for each place where learning was rejected or
   deferred (battle value net, policy prior, set-transformer builder, GRPO builder, offline RL,
   PPO), say whether the reason holds given the data and budget, and name the concrete evidence or
   data size that would make you switch to the stronger ML method. If you think something learned
   should come earlier, say so and why.
7. **Architecture.** Give the component list for a diagram of the planner and battler: each
   component with its inputs, outputs, what it is (search / rules / fitted model), where it lives
   (file), and the data flows between them (box → set builder → team chooser → battler → logs →
   fitted leaf eval / team filter → back). Mark which parts exist now vs which phase adds them.
8. **User involvement.** The user is learning RL and wants to decide each RL choice. List the
   decision points in the plan where the user should choose (with the options and what evidence
   will exist at that time).

## Also

- Anything in plan v2 you now think is wrong, missing or mis-ordered (be specific, cite file:line).
- A final phase list (short), each phase with its gate, integrating points 1–4.

Rules: read-only on the repo (no edits, no battles, nothing over ~2 minutes of compute). Write to
the file named in your task line, under ~300 lines. Mark sources as read vs recalled if you cite
literature. End with a ≤12-line summary in your final message.
