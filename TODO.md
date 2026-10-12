# Todo (2026-10-11, evening): from "it wins" to "we trust why"

The state: every singles fight from Brendan to the Cerulean Rival is 30/30
from the planned pool, Misty 30/30 since the cover chooser, Brock 30/30 but
23% deathless, boxes 58–66% (NOTEBOOK.md 2026-10-11, HANDOFF.md → Radical
Red). The question behind all of it, which nothing so far answers: **how
close is our simulation to the game?** Four layers can be wrong, and each one
sets the ceiling for everything above it:

    4  play + chooser      (search, calc bot, matrix, cover)      ← measured
    3  matchup matrix      (hits-to-KO off the calc bot's model)  ← unmeasured
    2  enemy AI            (our port of the CFRU trainer AI)      ← unverified
    1  engine + data       (RadicalRedShowdown fork, ROM dump)    ← unverified

Order below: verify from the bottom up, because a 30/30 against the wrong
enemy is worth nothing, then raise the floor (items, counters, play), then
widen (analysis, doubles, routing). Minimal code throughout: every item is
a script or a test, not a system. Effort in half-days (hd).

## A. Simulation fidelity (layers 1–2)

**A1. Engine vs ROM: mechanics.** The sim is `pokemon-showdown-rr`
(github RadicalRedShowdown/pokemon-showdown, mod `gen9rr`, our head
5495579). It is the community's calc/sim, not the ROM, and its own fidelity
is unknown to us. Check the things our runs depend on, each as one scripted
battle in the fork vs the number in the ROM:
- damage: a Scald in Misty's rain, a Rock move in Brock's sand, a Grass move
  in Grassy Terrain, STAB/type chart for the mod's changed types, crit rate;
- speed: Swift Swim/Sand Rush doubling, the hardcore 31-IV / 0-EV stat line
  (compare `compute_raw_stats` with the in-game summary screen at cap);
- the mod's own moves and abilities we meet early (Draco Barrage, Supercell
  Slam, Drain Punch-likes, Disarming Voice), the gym arena rules
  (`battleEffect`, permanent weather) and the hardcore rules (no items for
  the player? which items the AI uses mid-battle, Set mode, ability swaps).
Ground truth = the ROM in an emulator (mGBA, save states; a Lua script can
read the battle RAM) — the user's side. Deliverable: `data/rr/FIDELITY.md`
with one row per check (sim value, ROM value, matches?), and a test per
mismatch we fix in `config/custom-formats.ts` or the fork. 2 hd + ROM time.

**A2. Enemy AI vs CFRU.** `ai_util/rr_ai.py` is a decision-skeleton port,
"not a 1:1". The real AI is CFRU's `battle_ai_util.c` / `ai_scripts.c`
(public; RR 4.1's engine is closed but CFRU-derived) plus RR's own flags
(smart switching, "AI knows your team"). Three checks, cheapest first:
1. **Read-through diff** of the port against CFRU's score tables: which
   positives/negatives exist in C and not in Python (and vice versa), with
   the RR-specific toggles the hardcore docs list. Output: a table in
   `ai_util/rr_ai.py`'s docstring, 1 hd.
2. **Decision replay.** Record ~20 real fights from the ROM (video or Lua
   log: state each turn + the AI's move/switch), rebuild each state in the
   sim, ask the port for its choice, score agreement per turn. Target: the
   port picks the same action ≥80% of turns, the same *kind* (attack /
   switch / status) ≥95%. Script `scripts/rr_ai_replay.py`, 1 hd + ROM time.
3. **Sensitivity.** Until (2) exists: rerun the Brock/Misty cover runs with
   the port's switching off, with its KO-preference weight ±50%, with the
   tie-break seeded differently. If the deathless rate moves by more than
   the seed noise, the AI's exact shape matters and (2) is urgent; if not,
   the result is robust to the port. 1 hd (machine time).

**A3. Data vs game.** The dump (`trainers_hardcore.json`) is one person's
export; the tracker's sets are another. Diff them for the first 20 bosses
(species, level, item, moves) and spot-check five against the ROM. Already
known: 12-character move/item names (fixed in `rp.dump_id`), #14 Lass Relli
missing, relative levels "Max Level − n". Also confirm what we never
checked: the enemy's natures / abilities / IVs in hardcore, and whether
the pool's level gates (`order.json reach`) match the game's map order.
½ hd + the user's ROM eyes.

## B. The matrix (layer 3): floor and ceiling

**B1. Calibrate `pair()` against duels.** For every candidate × enemy at
Brock and Misty (≈ 150 × 6 cells each), `duel()` plays the real one-on-one
with the real sets under the arena, 10 seeds. Confusion table: tier (safe /
race / none) vs duel win rate and HP left. Report precision per tier ("safe
cells win the duel x%, with y% HP"), the false-safes by name (the Parasect
vs Starmie kind), and the false-nones (answers we discard). Script
`scripts/rr_matrix_calib.py`, results into NOTEBOOK. 1 hd.

**B2. Add the terms the false cells name.** Expected from the calibration,
in the order they are cheap: accuracy as expected hits (Hurricane out of
rain, Hypnosis); status and secondary effects of the enemy's set (Spore,
Toxic, Scald burn → a "disables" flag like `heals`); set-up on the enemy
side (one boost before our second hit); switch-in damage and Intimidate /
Sturdy / absorb abilities; the enemy's item (Focus Sash = one extra hit,
berry = one heal). Each term is a few lines in `ai_util/matchup.py` and is
accepted only if the B1 confusion table improves. 1–2 hd.

**B3. Take the duel winner when the paper answer dies.** The cover trusts
its cell; Starmie killed 11 times through a "safe" Parasect. Rule: after
stage C, for each threat with ≥ N deaths, if the pool holds a duel-winning
counter not in the six, swap it for the weakest member (by total edge) and
replay paired; keep the swap only if deaths fall. This is `--repair` with
the duel as the judge instead of the greedy seed. ½ hd.

## C. Items (both sides)

**C1. Player items from the care-package timeline.** `data/rr/README.md`
has the item timeline; `order.json` has trainer indices. Add
`items_before(t)` (berries, type items, the single Eviolite, Leftovers when
reached), let `choose_item(mon, boss)` give each of the six one held item
(attrition first: Sitrus / Leftovers / Eviolite on the mon the matrix says
takes the hits), print it with the nature/ability, and rerun Brock/Misty
paired. Items are a hardcore lever we have been playing without. 1 hd.

**C2. Enemy items in the matrix and the switch rule.** Focus Sash, berries,
Life Orb recoil, Choice lock: today they only enter through the damage
formula. Part of B2; listed here so it is not forgotten.

**C3. AI item use.** Does the hardcore AI use Potions / X items mid-fight?
Part of A1; if yes the port needs it.

## D. Play (layer 4): the deathless levers

**D1. Brock under sand.** Same team, field-aware 37 deaths vs blind 20
under the matchup rule: the correct model makes the search retreat into a
bench with no Cacnea answer. Trace 12 fights (`rr_trace.py`), classify the
switch-in deaths, and test two rules paired: never switch into a cell that
is not at least a race-winner even if the current mon is losing (stay and
trade), and the boost-aware threat (C4 in the plan: the enemy's next-turn
Power-Up Punch / Swords Dance counted before it happens). Decide
field-aware as the default from the result. 1 hd.

**D2. Death weight and the rollout value.** `outcome_value` still scores HP
lead when a rollout does not finish, which misreads stall (Lunatone, Toxic
Hippopotas). Try the matrix's edge sum as the terminal proxy instead; paired
on Brock. ½ hd.

**D3. The box milestone with B3 + C1 + D1.** Rerun 30 boxes at Brock and
Misty; the number to move is deathless per box (3% / 5% today). Machine time.

## E. Analysis and the dashboard

The evening's results exist as log lines; per-team lines since a47fd48.
`UI_TODO.md` is the plan; its first step is the one that matters for
everything above: **structured events** (`logs/<run>.events.jsonl`: start,
boss_start, stage lines, each team's result with sets, each box row, threats
and counters, end), then the generator and the run cards. The matrix
calibration (B1), the AI replay (A2) and the fidelity table (A1) all write
JSON next to their logs so the page can show them. Steps 1–2 of UI_TODO:
1–2 hd; the rest after.

## F. Doubles

#8, #11–13 before Misty, ~20 mandatory fights overall incl. Sabrina and
Lorelei: "not played". Staged as plan C6: bridge + mirror for two actives
(poke-env `DoubleBattle`), a doubles calc bot (best target, spread 0.75),
the port's doubles targeting from CFRU, search over joint actions only if
needed. 3–4 hd. After D3, before anything past Erika.

## G. Routing and the run layer

The 30-point gap between a drawn box and the planned pool is routing. First
pass needs no new machinery: for each box, the "most missed counter" column
already names species, location, method and rarity. `scripts/rr_route.py`:
given the box so far and the next boss, list the open locations whose draw
could still add a missing counter, with P(catch it) from rarity — the
expected gain of visiting each. Then the run layer (plan C5: `NuzlockeEnv`
on `data/rr`, bosses first, route trainers after `parse_dump` keeps the
relative-level mons). Routing 1 hd; run layer large, after F.

## What only the user can do

- ROM-side ground truth for A1–A3: an emulator with save states at each
  early boss, and either a Lua battle-RAM log or recorded fights (A2.2).
- Confirm the hardcore facts the sim assumes (HUMAN_TODO.md 7–12): mints /
  capsules free, hidden abilities reachable, DexNav, AI item use.

## Order

**2026-10-12:** A2.1 + A2.3 done (NOTEBOOK.md 2026-10-12, `data/rr/AI_DIFF.md`): the
result is sensitive to the AI's switching (no-switch enemy: Brock 37 → 85 deaths) and
the port's switching is not CFRU's. Next: port CFRU's switch triggers + switch-in
choice, then the tier ladder (AI_DIFF.md §3), each paired against `results/rr_aisens/`;
A2.2 needs the user's ROM recordings. **Switching ported the same day** (NOTEBOOK.md):
Misty's cover team 30/30 → 17/30, Brock's deaths 37 → 76. Every RR table before it is
against the old AI. **Tier ladder ported too** (NOTEBOOK.md): inside the noise. Next:
rerun the cover pool / boxes / early bosses once on this port — **done** (NOTEBOOK.md:
Misty 100 → 73%, Brock 100% with 63% deathless, early bosses unchanged); then A2.2 with ROM data.


A2.1 + A2.3 (AI sensitivity, read-through) → B1 (matrix calibration) → B2/B3
→ C1 (player items) → D1/D2 → D3 (box milestone) → E steps 1–2 → A1/A3 as
the ROM evidence arrives → F doubles → G routing. Every step is paired on
Brock + Misty, n=30, and gets a NOTEBOOK line citing its results file.
