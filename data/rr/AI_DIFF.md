# CFRU trainer AI vs `ai_util/rr_ai.py`: the diff

Sources read (CFRU `master`, fetched 2026-10-11 from github.com/Skeli789/Complete-Fire-Red-Upgrade). The AI lives in
`src/Battle_AI/`, not `src/battle_ai_*.c`.
- `src/Battle_AI/ai_master.c`: `BattleAI_SetupAIData`, `GetAIFlags`, `ChooseMoveOrAction_Singles`,
  `BattleAI_DoAIProcessing`, `AI_TrySwitchOrUseItem`, `ShouldAIUseItem`,
  `BattleAI_HandleItemUseBeforeAISetup`, `PredictMovesForBanks`, `UpdateStrongestMoves`,
  `IsPlayerTryingToCheeseAI` / `PickNewAIMove`
- `src/Battle_AI/ai_negatives.c`: `AIScript_Negatives` (ability-absorb block, `AI_STANDARD_DAMAGE`,
  `EFFECT_RESTORE_HP`, `EFFECT_TOXIC`, `EFFECT_PROTECT`, `EFFECT_STOCKPILE`, `EFFECT_FAKE_OUT`,
  `EFFECT_SUCKER_PUNCH`, `EFFECT_ATTACK_UP`)
- `src/Battle_AI/ai_positives.c`: `AIScript_Positives`, `DamageMoveViabilityIncrease`, `AIScript_SemiSmart`
- `src/Battle_AI/ai_advanced.c`: `PredictFightingStyle`, `ShouldRecover`, `ShouldProtect`, `ShouldPivot`,
  `IncreaseStatusViability`, `ShouldTryToSetUpStat`, `IncreaseStatViability`, `IncreasePivotViability`,
  `IncreaseViabilityForSlowKOMove`, `IncreaseFoeProtectionViability`, `IncreaseEntryHazardsViability`
- `src/Battle_AI/ai_switching.c`: `ShouldSwitch` and its triggers, `CalcMostSuitableMonToSwitchInto`
- `src/Battle_AI/ai_util.c`: `AIRandom`, `CanKnockOut`, `MoveKnocksOutXHits`, `CalcStrongestMoveIgnoringMove`,
  `MoveKnocksOutPossiblyGoesFirstWithBestAccuracy`, `IsSuckerPunchOkayToUseThisRound`, `BadIdeaToPoison`,
  `BadIdeaToRaise*`, `IncreaseViability` / `DecreaseViability`
- `src/damage_calc.c`: `AI_CalcDmg` (93% roll); `src/battle_controller_opponent.c`: `OpponentHandleChoosePokemon`
- `include/new/ai_scripts.h`, `include/new/ai_switching.h` (switch-scoring constants), `include/constants/battle_ai.h`

Not verified: the AI flags Radical Red gives its trainers, RR's difficulty variable, and any changes RR
made to CFRU (RR is closed source). The effect ID behind Roost is also unverified: there is no Roost
special case in positives or negatives, so it presumably uses `EFFECT_RESTORE_HP`.

## 0. How CFRU scores moves (context for the table)

- Every usable move starts at **100**. Unusable moves start at 0
  (`BattleAI_SetupAIData`).
- The scripts run in flag-bit order: bit0 `AI_SCRIPT_CHECK_BAD_MOVE` runs `AIScript_Negatives`,
  bit1 `AI_SCRIPT_SEMI_SMART` runs `AIScript_SemiSmart`, and bit2 `AI_SCRIPT_CHECK_GOOD_MOVE` runs
  `AIScript_Positives` (`sBattleAIScriptTable`). Vanilla names for bits 1 and 2 are TRY_TO_FAINT and CHECK_VIABILITY.
- A script only runs on moves whose score is still above 0 (`BattleAI_DoAIProcessing`).
- **Negatives are soft.** Most are `DECREASE_VIABILITY(10)`. An immune damaging move gets −15, and an
  absorb ability (Water Absorb, Volt Absorb and so on) gets −20. A few cases return 0: Gravity, Heal Block, Throat Chop.
  So a "bad" move sits at about 90 and is still picked when nothing else is viable.
- **Positives are small tiers, not continuous.** The deltas are +1 to +9 in singles, and the score is capped at 255.
- Final pick: the highest score wins, with a uniform random choice among ties
  (`ChooseMoveOrAction_Singles`). There is **no 80/20 "best move" coin flip** like vanilla GBA.
- **Fighting-style class** (`PredictFightingStyle`) scales most bonuses. Classes for the early bosses
  (my derivation from that function plus `data/rr/trainers_hardcore.json`):

| Mon (cap) | Moves | CFRU class |
|---|---|---|
| Hippopotas (16) | Bulldoze, Rock Tomb, Toxic, Slack Off | SWEEPER_SETUP_STATUS (2 attacks, ≥1 status move) |
| Cacnea | Power-Up Punch, Bullet Seed, Thunder Punch, Pin Missile | SWEEPER_KILL |
| Lunatone / Archen | 4 attacks | SWEEPER_KILL |
| Lileep | Stockpile, Recover, HP Rock, Mega Drain | SWEEPER_SETUP_STATUS |
| Varoom | Bulldoze, Gyro Ball, Toxic, Protect | SWEEPER_SETUP_STATUS |
| Politoed (28) | Flip Turn, Scald, Earthquake, Toxic | SWEEPER_SETUP_STATUS |
| Ludicolo / Starmie | 4 attacks | SWEEPER_KILL |
| Mantine | Scald, Hurricane, Roost, HP Grass | SWEEPER_SETUP_STATUS |
| Clodsire | Poison Jab, Bulldoze, Recover, Toxic | SWEEPER_SETUP_STATUS |
| Toxicroak | Swords Dance, Drain Punch, Poison Jab, Sucker Punch | SWEEPER_SETUP_STATS |

For SETUP_STATUS, `IncreaseStatusViability(boost)` adds **4 + boost**. Toxic (boost 1) is **+5**.
Recover, Slack Off, Roost and Protect (boost 3) are **+7**. The resulting ladder for these bosses is:

- KO and moves first: +9
- KO but moves second: +8 (KILL / SETUP_STATUS) or +6 (SETUP_STATS)
- Recover / Protect when the good-idea check passes: +7
- Toxic, or Swords Dance / Power-Up Punch when safe: +5
- Strongest attack: +2
- Every other attack: 0

**In CFRU, any status, setup or recovery move whose good-idea check passes beats every attack that does not KO.**

Real damage numbers drive all of this. `CanKnockOut`, `Can2HKO` and `MoveKnocksOutXHits` use the full
damage calc at a fixed **93% roll** (`AI_CalcDmg`: `damage * 93 / 100`), with items, abilities, weather
and the **player's real moveset and stats**. The AI reads `gBattleMons` for the player directly; that is
the "AI knows your team" part. `PredictMovesForBanks` runs Negatives+Positives on the *player's* moves to
predict the player's move. If the player's best viability is below 100, it predicts a **switch** instead.

## 1. Behaviour table (priority order: most likely to change an early boss first)

| # | Behaviour | CFRU (file:function, delta / condition) | Our port (rr_ai.py) | Impact |
|---|---|---|---|---|
| 1 | **Relative weight of attacks vs status/setup/recovery** | Tier ladder above (`ai_positives.c:DamageMoveViabilityIncrease`, `ai_advanced.c:IncreaseStatusViability`). A non-KO attack is never worth more than +2. | Continuous. KO is +30 (L33, L217-218). A non-KO attack gets `int(min(dmg/hp,1.5)*18)` up to +27, plus ±4 for super-effective / resisted (L222-226). Toxic gets +11 (L304-306), setup +12-16, recovery +6/+14. An attack doing more than about 40-60% of the foe's remaining HP beats Toxic or Recover. | **High.** Hippopotas, Varoom, Politoed and Clodsire Toxic, and Toxicroak Swords Dances, far more reliably in CFRU. |
| 2 | **Fighting-style class** | `ai_advanced.c:PredictFightingStyle` sets the class (table above). Every `INCREASE_STATUS/STAT_VIABILITY`, slow-KO, pivot and protect bonus depends on it. | Missing: one flat scale for every mon. | **High** (it drives #1, #4, #6, #7) |
| 3 | **KO bonus** | `DamageMoveViabilityIncrease`: **+9** if the move KOs and moves first, accuracy ≥70 or no accuracy helper or the AI dies anyway, the foe is **not predicted to switch**, and the foe is not predicted to Sucker Punch. A slow KO gets `IncreaseViabilityForSlowKOMove` (+8 for KILL / SETUP_STATUS, +6 for SETUP_STATS) **only if the AI does not know Protect** and the foe won't die to residual damage first. Among KOing moves, ties go to priority, then accuracy, then non-contact, then non-recoil, then random. | +30 if expected damage ≥ 0.92 × HP (L217). Expected damage is **accuracy-weighted** (L407), so Hydro Pump or Rock Tomb can miss the KO threshold. No predicted-switch or Protect-in-moveset exception. +4 for a priority KO (L219). | **High.** Varoom (has Protect) gets no slow-KO bonus in CFRU. The AI also stops going for the KO when it predicts the player will switch. |
| 4 | **Recovery (Slack Off, Recover, Roost, Mega Drain, Drain Punch)** | Negatives: **−10 at 100% HP, −9 at ≥90%** (`ai_negatives.c` EFFECT_RESTORE_HP). Positives: `ShouldRecover` decides, not an HP threshold. If the AI moves first: heal when the foe can KO and healing prevents it; or when the foe can 2HKO, the heal is larger than current HP and a 50% coin (`simulatedRNG[1]&1`) lands. If the AI moves second: heal when the foe can't KO, can 2HKO, and can't KO after the heal. It then gets +7 (SETUP_STATUS; +8 for stall). Drain moves get +3 when `ShouldRecover` passes. | HP thresholds: <35% +14, <55% +6, otherwise −10; Rest −12 more above 50% (L259-268). Drain moves +4 below 70% HP (L229). Neither the foe's damage nor speed order is considered. | **High.** Hippopotas, Lileep, Mantine and Clodsire heal at different times. In CFRU they heal at roughly 40-60% HP whenever the player's hit can be outhealed, and **never** heal into a guaranteed KO they can't outheal. |
| 5 | **Setup (Swords Dance, Power-Up Punch, Stockpile)** | Swords Dance: `INCREASE_STAT_VIABILITY(ATK, 8, 2)`, so **+5** for KILL/SETUP_STATS and only up to +2 stages (statLimit 8), via `ShouldTryToSetUpStat`. If faster: don't set up if the foe can KO, counting chip damage. If slower: don't set up if the foe can 2HKO (Def/SpD boosts excepted). Also no setup vs Unaware, a predicted phazer, or a sleep move from a faster foe (`BadIdeaToRaiseStatAgainst`). The limit lifts to +6 vs stall or support classes or a predicted switch (`ShouldMaxStatBuffLimit`). Power-Up Punch (`EFFECT_ATTACK_UP_HIT`, 100% ≥75) takes the same path: **+5 when safe, otherwise +1**. Stockpile without Swallow or Spit Up is treated as Cosmic Power: +5 for SETUP_STATUS when the foe is likely physical and setup is safe. At 3 stacks it gets −10. | Setup: +6 + 3 per stage, +4 on turns 1-2, only if HP >50% and *type* matchup ≥0; −8 below 40% HP (L234-241). Capped only at +6 (L179-182). **Power-Up Punch is not seen as setup**: poke-env gives `self_boost=None, boosts=None` (checked). **Stockpile** has no boosts, so it gets the generic +2 (L291). | **High** for Toxicroak (Swords Dance once to +2, then attacks) and Cacnea (Power-Up Punch). **Medium** for Lileep (Stockpile). |
| 6 | **Switching: when** | `ai_master.c:AI_TrySwitchOrUseItem` calls `ai_switching.c:ShouldSwitch`, which checks in order: absorb the predicted move with Water Absorb, Dry Skin, Volt Absorb or similar (all trainers; 50% vs non-damager foes; 25/75% repeat limit); pass Wish; stop a locked move; Perish Song; Wonder Guard; only bad moves left; Natural Cure / Regenerator; semi-invulnerable; Yawn; asleep. The next group needs **flags > CHECK_BAD_MOVE**: **annoying residual damage** (Toxic counter >6 turns, Curse, Nightmare, Leech Seed 25%); **avoid death** (AI slower, foe's predicted move KOs, HP >20%, and a switch-in is immune, or walls / resists everything, or outspeeds and KOs); **offensive stat ≤ −3** (damager class); **save sweeper for later** (damager can't KO; 50% / 75% random parts). If a fast pivot move is known, it **pivots instead** (`goodToPivot`). There is no generic "bad type matchup" switch. | Switches when type matchup < −1.5 and a bench mon is 0.5 better, or **any** of Atk / SpA / **Def / SpD** ≤ −3, or a type-based "incoming KO risk" with best score <110. Never switches when the best move KOs (L344-365). Also switches whenever every move scores 0 (L141). | **Med-High.** The port switches on type matchups CFRU ignores (for example Politoed vs a Grass mon), and misses absorb switches and avoid-death switches. |
| 7 | **Switching: which mon (also used after a faint)** | `CalcMostSuitableMonToSwitchInto`, per bench mon: outspeeds +14, can KO +31, revenge kill (priority KO or Moxie) +8, can 2HKO +2 (only if no KO), resists everything (no foe move 3HKOs it) +17, walls +2, weak to a move −1, **faints to a foe move −39** (−15 if it outspeeds but can't KO). A hazard remover is preferred. Ties are 50% random. Smart AI uses real damage calcs; the basic flag uses type charts. `OpponentHandleChoosePokemon` uses the same pick after a faint, or the stored `pivotTo` target after U-turn / Flip Turn. | `_switch_score` = type matchup × 10 + HP × 4 − hazard terms (L377-385). No damage calc, no KO / outspeed check. | **High.** It decides which Brock or Misty mon comes in after every KO. CFRU leads with the bench mon that outspeeds and KOs the player's active mon. |
| 8 | **Toxic / other status** | Negatives: −10 if the target can't be poisoned (status, type, ability, Misty Terrain, Substitute). Positives: +5 (SETUP_STATUS) **unless `BadIdeaToPoison`** (Guts, Poison Heal, Toxic Boost, Marvel Scale vs physical, Natural Cure, Shed Skin, Quick Feet, Facade, Synchronize and so on); +2 extra with Venoshock or Merciless. KILL class gets **0**. Paralysis: +1/+2 class bonus, +2 when paralysis flips the speed order. Damaging moves with a status secondary below 75% chance (Scald 30%) get no status bonus. | +8 base, +3 Toxic / sleep, +4 paralysis if slower, +3 burn vs physical (L304-311). −6 if the best move KOs and is faster, −4 if foe HP <25% (L298-303). Ability blocks only for immunity abilities (L77-84, L313-340). No Guts / Facade / Natural Cure checks. | **Med.** Toxic choice is #1. The BadIdea list matters for the player's Guts or Facade mons. |
| 9 | **Protect (Varoom)** | Negatives: −10 if the foe is recharging. On a consecutive use: −10 at 2+ uses, and at 1 use a **50% roll for −6**. Positives: `ShouldProtect` returns USE_PROTECT when the foe will faint to residual damage, or is trapped and taking residual, or is predicted to explode or come down from Fly, or **when the foe can KO or 2HKO the AI and is taking secondary damage (Toxic, burn)**. That gives +7 for SETUP_STATUS. | Never twice in a row (L163). Otherwise generic status +2 (L291). Protect has priority 4 in poke-env but `dmg=0`, so there's no priority bonus. | **Med.** The Toxic-then-Protect stall on Varoom is missing. |
| 10 | **Sucker Punch (Toxicroak)** | Negatives: −10 if the foe has no attacks. −9 unless `IsSuckerPunchOkayToUseThisRound`: the player's **predicted move is damaging** and Sucker Punch goes first. Smart AI with Sucker Punch already revealed, against a player who has status moves or faster priority: only 50% of turns (`suckerPunchOkay = AIRandom()&1`). | A normal priority-1 attack: KO +4, +6 under type-based "incoming KO risk" (L219-220, L273-274). Fires into status moves. | **Med** (Misty's last mon) |
| 11 | **Flip Turn / U-turn** | `ShouldPivot` → `IncreasePivotViability`. PIVOT: **+9 for SETUP_STATUS** (Politoed), +3 for KILL. PIVOT_IMMEDIATELY (switch logic wanted out): +9. DONT_PIVOT (for example ≥25% hazard damage on re-entry, or none of the PIVOT cases apply): **−9**. PIVOT cases: foe predicted to switch; foe 2HKOs and a bench mon KOs and outspeeds / walls; break a Sash or Sturdy; bench mon resists everything; and others. | +8 if type matchup <0 and a switch is available (L270-271). | **Med** (Politoed's Flip Turn is likely its top-scored move in many states) |
| 12 | **Priority / desperation** | When the foe's predicted move KOs the AI, the **strongest move that still goes first** gets +9 (third branch of `DamageMoveViabilityIncrease`) unless the AI knows Protect, can stall-heal, or is a hazard / phaze / stall class. Repeats are limited by `usingDesperateMove` plus a `simulatedRNG[3]` 25/75% roll. | +6 for any damaging priority move under type-based KO risk (L273). | Low-Med (few early-boss priority moves besides Sucker Punch) |
| 13 | **Strongest attack** | `CalcStrongestMoveIgnoringMove`: highest 93%-roll damage, **capped at target HP**, so all KOing moves tie. Ties go to priority, then accuracy, then avoiding contact into Rough Skin / Rocky Helmet, then non-recoil, then no stat drop, then random. +2 for that one move only. | `max` over continuous scores (SE ±4, recoil −2, accuracy −(1−acc)×8, accuracy-weighted damage), random among exact ties (L119-125). | Low-Med (usually the same move; differs on accuracy and resist edges) |
| 14 | **Accuracy** | No general penalty ("Put Acc check here" is an empty comment at the end of `AIScript_Negatives`). Accuracy is used only for the KO tie-break and the ≥70 KO gate. | Damage × accuracy (L407) **and** −int((1−acc)×8) (L279-281). Double-counted. | Low |
| 15 | **Immunities / absorbs** | −15 for an immune damaging move, −20 for an absorb ability, −10 for Soundproof. These are not zero. | Returns 0 (L166-168) via `calc_bot.effectiveness` (full-info ability immunities). | Low (same pick unless every move is bad, and then CFRU still clicks one instead of switching) |
| 16 | **Hazards (Stealth Rock etc.)** | `IncreaseEntryHazardsViability`, by class: SETUP_STATUS / SETUP_STATS get +6 if `IsWorthSettingHazards` (foe has more than 2 mons left), KILL gets 0. | +10 if 3+ foes left, +4 if 2, −8 otherwise (L243-250). Counts 6 minus fainted, which assumes a 6-mon player. | Low for these two fights (no hazard setters); matters from Brock 68 on |
| 17 | **Item use (Potions, X items)** | `BattleAI_HandleItemUseBeforeAISetup` loads `gTrainers[].items` **only if `!IsBagDisabled()`** (or `VAR_ITEM_RESTRICTIONS` = AI-only). `ShouldAIUseItem` then handles Full Restore / Potion (<25% HP for the basic AI; <50% HP + `ShouldRecover` for smart AI), status cures, and X items on the first turn only. The RR hardcore sheet says "no bag against gym leaders and select bosses", so **under CFRU logic those bosses use no items.** | No item use. | Low (consistent for bosses; ordinary trainers with items are unverified) |
| 18 | **RNG in scoring** | Deterministic tiers plus: random choice among top ties; `simulatedRNG[i] = AIRandom()%100` per move (Protect double 50%, ShouldRecover 50%, desperation 25/75%); Sucker Punch coin; strongest-move random tie-break; switch-in 50% ties; save-sweeper 50% / 75%; absorb-switch 50% / 25% / 75%. `AI_TRY_TO_KILL_RATE` applies **only to the basic CHECK_BAD_MOVE AI**: on a % roll, +7 for KO-first or +2 for the strongest move. `AIRandom` is an LCG reseeded every frame. | Only the random tie-break (L124). Continuous scores make ties rare, so the port is near-deterministic. | Low-Med (CFRU's 50% recover and protect rolls show up within a single fight) |
| 19 | **Move prediction of the player** | `PredictMovesForBanks` scores the player's 4 moves with Negatives+Positives. It predicts a switch if the best is below 100, otherwise picks a random move among the best. Used by KO (no +9 on a predicted switch), Sucker Punch, Protect, setup, pivot, absorb-switch, avoid-death, and item use. | Missing. Opponent threat is only type-based (`_incoming_ko_risk`, L430-436). | **High** (indirect: gates #3, #4, #6, #9, #10, #11) |
| 20 | **Anti-cheese** | `RechooseAIMoveAfterSwitchIfNecessary`: with CHECK_GOOD_MOVE **and Expert difficulty**, the AI re-picks its move after the player double- or triple-switches or tries a turn-1 choice-lock cheese. A hard-mode chance to predict a normal switch exists (Unbound-only gating in this file). | Missing. | Low (only if the run layer switch-stalls; RR difficulty unknown) |
| 21 | **Mega / Dynamax / Z** | Mega when `megaPotential` is set (the AI temp-megas everyone while thinking); Dynamax via `ShouldAIDynamax`; Z via `ShouldAIUseZMove`. | Mega whenever possible (L136); Dynamax on last mon or full HP with ≤3 alive (L488-494); Tera if super-effective (L496-503). | Low for Brock / Misty 16-28 |

### AI flags and "smart switching" / "AI knows your team" in CFRU terms

- Trainer flags come from `gTrainers[id].aiFlags` (`GetAIFlags`). With `VAR_GAME_DIFFICULTY`, Hard and Expert
  OR in `AI_SCRIPT_SEMI_SMART` for every trainer, and Easy downgrades.
- `AIScript_SemiSmart` only works when CHECK_GOOD_MOVE is absent. It sends Toxic, Restore HP, Protect,
  Attack Up 2, Attack Up Hit, Stockpile, Baton Pass (which covers Flip Turn), absorb moves and others straight
  to `AIScript_Positives`, and gives everything else the damage tiers. **For these bosses, semi-smart and smart score moves almost identically.**
- The "smart switching" in RR's hardcore docs maps to the `ShouldSwitch` triggers gated on
  `aiFlags > AI_SCRIPT_CHECK_BAD_MOVE` (avoid death, residual damage, save sweeper, only bad moves left), plus
  the damage-calc branch of `CalcMostSuitableMonToSwitchInto` (`isOneFoeOnField && aiFlags > CHECK_BAD_MOVE`).
  The basic AI uses type charts there.
- "Knows your team": every `CanKnockOut(player, ai)`, `Can2HKO`, `PredictMovesForBanks` and switch-in damage
  calc reads the player's real `gBattleMons` moves, stats, ability and item. The port can only match this if
  poke-env's `opponent_active_pokemon` carries the full moveset and stats, which `calc_bot.effectiveness`'s
  docstring suggests it does under full info.
- I could not determine which flags RR 4.1 gives Brock or Misty. The sheet's "smart AI" wording implies at
  least SEMI_SMART.

## 2. Things in our port that CFRU does not do

1. **Type-matchup switching** (`_matchup` < −1.5, L353-365) and switching on **Def / SpD ≤ −3** (L351). CFRU
   switches only on the specific triggers in #6 and only checks offensive stats.
2. **A continuous damage bonus** (`dmg/hp × 18`, L222), **±4 for super-effective / resisted** (L223-226),
   **recoil −2** (L227), and an **accuracy-weighted** damage formula (L407) plus an accuracy penalty (L279-281).
3. **HP-threshold recovery** (<35% / <55% / else −10, L259-268). CFRU only penalises ≥90% HP; otherwise it decides with damage calcs.
4. **Setup gated on HP >50% and type matchup ≥0**, with a "turn ≤2" +4 bonus (L234-241). CFRU gates on the
   foe's ability to KO or 2HKO, and caps Swords Dance / Power-Up Punch at +2 stages.
5. **A status penalty when the best move KOs and is faster** (−6) **or the foe is below 25% HP** (−4) (L298-303).
   CFRU gets the same effect through the tier order: KO +9 beats status +5.
6. **Hazard scoring by count of opponent mons left** (+10 / +4 / −8). **Screens +6 above 40% HP** (L256).
   **Fake Out +12** (L276); CFRU uses `IncreaseFakeOutViability` by class.
7. **Hard dumps to 0** instead of −10/−15/−20, and **switching when every move is 0** (L141-142). CFRU still
   clicks a −10 move rather than switching, unless `OnlyBadMovesLeftInMoveset` (smart AI) finds a safe switch-in.
8. **Protect never twice** (L163). CFRU allows a second Protect 50% of the time, and refuses only at 2+ uses.
9. **Tera when super-effective** (L496-503). This is Showdown-only; CFRU has no Tera.
10. **Fainted-count heuristics** that assume 6 player mons (L244).

## 3. Verdict: the five differences to fix first

1. **Replace the continuous scale with CFRU's tier ladder, keyed by fighting-style class** (rows 1-3).
   - Attacks: KO-first +9, slow KO +8 or +6 (none if the AI knows Protect), strongest move +2, other attacks 0.
   - Status / setup / recovery: good-idea moves get class-scaled +5 to +7.
   - This is roughly 60 lines (`PredictFightingStyle` plus tier constants), and it flips the Toxic, Recover
     and Swords Dance vs attack choice on 8 of the 12 early-boss mons.
2. **Damage-calc gates using the player's real moves at a 93% roll:** foe-can-KO, foe-can-2HKO, and speed
   order. Then port `ShouldRecover` (with the 50% roll and the 100% / ≥90% penalties) and
   `ShouldTryToSetUpStat` (Swords Dance / Power-Up Punch +2 cap; detect Power-Up Punch and Stockpile, which
   poke-env reports with no boost). Rows 4-5.
3. **Switch-in choice:** port `CalcMostSuitableMonToSwitchInto`'s scores (+31 KO, +14 outspeed, +17 resists
   all, −39 faints) for post-faint replacements and switches (row 7). This picks the mon the player faces after every KO.
4. **Switch triggers:** drop the type-matchup switch and the Def/SpD −3 switch. Add the avoid-death switch (AI
   slower, predicted move KOs, HP >20%, a safe switch-in exists), the absorb-ability switch, and the Toxic
   counter >6 turns switch (row 6).
5. **Move-specific gates:**
   - Sucker Punch: −9 unless the predicted player move is damaging.
   - Protect: +7 when the player takes residual damage, 50% double-Protect.
   - Flip Turn: `ShouldPivot`, with +9 / −9 for Politoed.
   - These need a cheap predicted-player-move (best of the player's moves by the same scorer) (rows 9-11, 19).
