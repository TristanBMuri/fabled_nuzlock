# How to Use `rl-tutor`

Your hands-on RL tutor built directly around `nuzlock_rl`. Works identically in **Claude** (Opus / Fable) and **Antigravity** (Neon) — both share the same progress file (`rl/LEARNING.md`).

---

## 1. Commands (How to Start a Session)

Type any of these in a fresh thread (or any chat in this repo):

| Command | Time | What you get |
|---|---|---|
| `/rl-tutor` | 5–10 min | **Quick check-in** (3 round trips): 1 recall question, 1 prediction bet, reveal + twist, 1-sentence takeaway + map pin. |
| `/rl-tutor lab` *(or `/rl-tutor lab gae`)* | 25–35 min | **Deep lab**: write a 10–15 line kata in `no_AI_code_here/rl/` (checked by running it against the repo), diagnose a planted bug in `rl/cockpit.py`, or derive an equation from a primary source. |
| `/rl-tutor why <D# \| commit \| file>` | 5–15 min | **Autopsy / rehearsal**: replay a real decision (e.g. `D4`, `D11`, `D14`) or commit with only the evidence from that day. |
| `/rl-tutor map` | 1 min | **Your RL atlas**: shows your Generalized Policy Iteration map, mastery progress across all 14 regions + 6 expansion topics, and calibration tally. |

*(In Antigravity, typing `/rl-tutor` or `"rl-tutor lab"` activates the skill automatically; plain prompts like `"quiz me on RL"` work too. Normal coding/debugging will NOT trigger tutor mode.)*

**Shortcuts mid-session:**
- `5`, `7`, `9` — your confidence on a bet (**50%** = coin flip / gut guess, **70%** = fairly sure, **90%** = would put real money on it).
- `stop` — end the session cleanly whenever you want.
- `show me` — ask for an interactive inline HTML slider/plot on a continuous knob (`γ`, `λ`, UCB `c`, scale, `p`).

---

## 2. How a 5–10 Minute Check-In Works

1. **Turn 1 (Instant):** Opens with 1 quick spaced-recall question from earlier + today's main bet. Reply with your answer, a 1-line *why*, and a confidence digit (`5` / `7` / `9`).
2. **Turn 2 (Reveal + Twist):**
   - If the bet is **`live`** (an unrun command or fresh `--seed`), the tutor posts its **own** bet before running it — so you can watch the tutor get surprised too.
   - If the bet is **`archive`** (`NOTEBOOK.md` / `results/`) or **`derivable`**, only you bet (no fake tutor bets).
   - Every reveal includes its uncertainty band (inside the band = *"a lead, not a result"*), followed by a free-form **twist** to test the real mechanism (about 1 in 3 twists confirm the obvious answer, so you can't game it by always picking "surprising").
3. **Turn 3 (Close):** Write a 1–2 sentence takeaway in your own words. The tutor sharpens it once if needed, adds a row to your map, queues a non-Pokémon transfer question for next time, and updates `rl/LEARNING.md`.

---

## 3. How Mastery Works (No XP, No Streaks, No Overdue Guilt)

Every region on the map tracks **real evidence** instead of "I read an explanation":

- **`P`** — **Predicted** an unseen variation/twist with the right mechanism.
- **`T`** — **Transferred** the concept to a non-Pokémon problem on a *later* day (automatically asked at the start of a future check-in).
- **`B`** — **Built or derived by hand** (a 10–15 line kata in `no_AI_code_here/rl/` verified by running against repo code, or a line-by-line derivation).
- **`D`** — **Decided first** on a real `D#` choice in this project before seeing the agent's recommendation.

A region counts as **owned** once you have **2 different letters on 2 different days**, with at least one of **`T` or `B`**. If you miss a week (or a month), you never see an "18 overdue cards" guilt counter — you just get 1 card and keep going.

---

## 4. Hooks During Normal Project Work (Both Enabled at Launch)

Outside `/rl-tutor`, both Claude (`CLAUDE.md`) and Neon/Antigravity (`AGENTS.md`) have two lightweight hooks enabled:

1. **At a real `D#` decision:** After laying out options and basis, the agent asks once before giving its recommendation:
   > *"Your call first? (option + 5/7/9 confidence, or `rec`)"*
   - Reply with your pick + `5`/`7`/`9`, or type `rec` to jump straight to the recommendation.
   - **Auto-off:** 3 `rec`/`skip`s in a row turns it off until you ask to turn it back on.
2. **When launching a multi-minute experiment run:** While the run is already executing in the background (zero slowdown), the agent asks once:
   > *"Bet before it finishes? (outcome + 5/7/9, or `skip`)"*
   - Rough `5` (50%) guesses are completely welcome — the agent bets too before seeing results, logs `user bet:` in `NOTEBOOK.md`, and your next `/rl-tutor` check-in settles it automatically.

**Want to change anything later?**
- **Turn off a hook:** Edit or delete the bullet in `CLAUDE.md` and `AGENTS.md` (or just say `"turn off the D# hook"`).
- **Turn on a weekly reminder:** Currently **OFF**. Say `"enable the weekly rl-tutor teaser"` anytime to add a single self-silencing weekly 2–4 line mystery prompt in one bound thread.

---

## 5. Where Everything Lives

| Path | Role |
|---|---|
| `rl/LEARNING.md` | Your personal ledger (map rows, calibration counts, misconceptions, next opener). **Gitignored** and automatically shared across all git worktrees. |
| `no_AI_code_here/rl/` | Your hand-written katas (`td.py`, `gae.py`, `bandit.py`, … → Lessons 3 & 4). The tutor runs tests against it and **never edits** files here. |
| `.agents/skills/rl-tutor/SKILL.md` | Tutor protocol & anti-illusion rules (canonical source). |
| `.agents/skills/rl-tutor/references/map.md` | The 14 RL regions + 6 "Why Not Here?" expansion topics (DQN, SAC, RND, MuZero, Offline RL, RLHF/GRPO). |
| `.claude/skills/rl-tutor` | Symlink so Claude Code loads the exact same skill as Antigravity. |
| `CLAUDE.md` & `AGENTS.md` | Holds the two work-session hooks (Claude + Antigravity parity). |

---

## 6. Four Tips to Make It Stick

1. **Don't hedge every bet at `5`.** Being confidently wrong (`9` miss) is the single highest-value learning moment — it surfaces the exact hidden assumption to fix.
2. **Mix in a `/rl-tutor lab` every ~4–5 check-ins.** Check-ins build sharp intuition (`P`/`T`); writing 12 lines of GAE or UCB1 yourself in `no_AI_code_here/rl/` (`B`) locks in the fine detail.
3. **Switch freely between Neon, Fable, and Opus.** Because `rl/LEARNING.md` lives in the repo root (shared across worktrees) and holds the pre-written `next:` opener, you can start Monday in Claude and pick up Thursday in Antigravity seamlessly.
4. **Use `/rl-tutor why` whenever a weird result lands.** Any confusing commit, phase result, or upcoming `D#` can be turned into an interactive autopsy on demand.
