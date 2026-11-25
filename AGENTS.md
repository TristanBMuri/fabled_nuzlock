# AGENTS.md

Read and follow `CLAUDE.md` for all project rules, minimal-code constraints, commands, and hard-won
lessons.

## RL learning hooks (Antigravity / Neon parity with `CLAUDE.md`)
- **At a real `D#` RL decision:** Lay out options and basis first, then ask once before giving your
  recommendation: *"Your call first? (option + 5/7/9 confidence, or `rec`)"*. Record
  `user call: <choice> <conf>%` in that `NOTEBOOK.md` entry, then give your recommendation and let
  the user decide (auto-off after 3 consecutive `rec`/`skip`s until re-enabled).
- **When launching a multi-minute experiment run:** After starting it in the background, ask once:
  *"Bet before it finishes? (outcome + 5/7/9, or `skip`)"*. If answered, commit your own honest live
  bet before reading results and log `user bet: ...` in that run's `NOTEBOOK.md` entry so
  `/rl-tutor` can settle it later.
- **RL Tutor skill:** When the user invokes `/rl-tutor` or explicitly asks to learn/be quizzed on
  RL, follow `.agents/skills/rl-tutor/SKILL.md`. Never enter tutor mode during ordinary debugging,
  coding, or routine experiment explanations.
