# Radical Red 4.1 data (Hardcore mode)

Inventory of what a full Radical Red hardcore Nuzlocke run needs, where each
piece comes from, and what is still missing. Written 2026-10-10 after the two
boss probes (NOTEBOOK.md → Radical Red). Hardcore mode auto-enables Minimal
Grinding Mode: that single fact changes several assumptions the probes made.

## Sources in hand

| file | source | what it holds |
|---|---|---|
| `sheets/rr_hardcore_info.xlsx` | official "Restricted/Hardcore Mode Info & Hardcore Bosses v4.1" (Google Sheet `1jDbKFA30…`) | Main tab: hardcore rules, level caps, banned moves, ability swaps, Toxic/Leech Seed whitelist; Trainer Order tab: the 115 bosses with caps and locations; one tab per boss category with full sets and the arena effect |
| `sheets/rr_items_tms_tutors.xlsx` | official "Item, TM, and Move Tutor Locations v4.1" (`16vBrWJ…`) | TMs/HMs, tutors (red = not in hardcore), overworld items, shops, **care packages** (every held item the game hands out, per progression point; red fill = not in hardcore, blue = not in MGM/hardcore) |
| `sheets/rr_locations_raids.xlsx` | official "Pokémon Locations & Raid Dens v4.1" (`15mUFUc…`) | grass/cave, fishing/surf, safari, fossils, statics, raids, egg vendor, trades, gifts |
| `tracker_*.json` | nuzlocke tracker export (same data as the sheets above, already JSON) | bosses by category with sets + `battleEffect`, 83 locations with encounter methods/rarity/levels, items |
| `sheets/rr_default_bosses.xlsx` | official "Default Mode Bosses v4.1 (with EVs)" (`1ES8L4Oz…`, added 2026-11-23) | the normal-mode bosses, same tabs as the hardcore sheet; `tracker_bosses.json → default` is the same data as JSON |
| `rr41_normal_trainers.txt`, `rr41_easy_trainers.txt` | Rudo2204's 4.1 dumps, normal (no min-grind, gist `7f9e4a3c…`) and easy (`99e8425c…`), added 2026-11-23 | all 738 slots for the other two difficulties (not parsed yet) |
| `rr41_hardcore_trainers.txt` | Rudo2204's ROM dump, hardcore (gist `ed23cfda…`) | all 738 trainer slots with sets, IVs, natures, items; levels either absolute or "Max Level − n" |
| `hardcore_rules.json` | extracted from the Main tab | caps (18), general rules, 82 banned moves, 31 ability swaps + 5 species swaps, 91 Toxic/Leech-allowed species |
| `fr_slots.json` | pret/pokefirered decomp (`opponents.h` + map scripts) | ROM trainer slot → FireRed constant → map(s). RR reuses FireRed's trainer table (609/738 slot names match; Brock = 0x19E, Misty = 0x19F, …), so 470 slots have a map; the 268 without are RR's own additions in old Hoenn/placeholder slots (Johto leaders, Brendan/May, admins, doubles partners, rematches) — those are all bosses the tracker already places |
| mod tables (`species/moves/learnsets/evolutions/tms.json`, `mod_data.json`) | RadicalRedShowdown fork, mod `gen9rr` | built by `scripts/build_rr.py` |

## Hardcore facts that change the model (from the Main tab)

- **No EVs anywhere** (player or trainers; Ghost Marowak 252 HP is the one
  exception); **all IVs 31**. Verified against the sheet's own speed stats:
  84/92 boss mons match EV 0, none match 252. `build_rr.trainer_evs` gives every
  trainer EVs 0 (Ghost: 252 HP) since 2026-10-10; the player gets 0 too.
- **Player bans** (82 moves): every set-up move, weather and terrain moves,
  Trick Room, Tailwind, hazards, Protect, Substitute, Taunt, Destiny Bond,
  Perish Song, Power-Up Punch, Flame Charge, Scale Shot, Fiery Dance, Charge
  Beam, Meteor Beam …; Toxic and Leech Seed only on 91 weak species. Ability
  swaps for the player: Drought/Desolate Land → Sheer Force, Drizzle/Primordial
  Sea → Adaptability, Sand Stream → Sand Force, the four Surges, Speed Boost →
  Infiltrator, Moxie/Beast Boost/… → Unnerve, Magic Bounce → Magic Guard, etc.
  `gauntlet.BANNED_MOVES` (set by `rr_team_probe --legal`) removes the moves;
  the ability swaps are still not applied (ability slot 0 as the dex lists it).
- **AI weather/terrain is permanent**; Defog & co. do not clear terrain.
- Set mode, no bag against gym leaders and selected bosses (the sim already
  has no bag).
- Level caps: 16 Brock · 23 Mt. Moon Archer · 28 Misty · 36 Surge · 44 Erika ·
  47 Game Corner Giovanni · 56 Silph Archer & Ariana · 57 Silph Giovanni · 59
  Sabrina · 68 Koga · 73 May · 76 Blaine · 79 Cave Archer & Ariana · 80 Cave
  Giovanni · 81 Clair · 82 Brendan · 85 Elite 4.
- Held items are **not** freely available: Eviolite is one copy from Whitney
  (optional, Route 11, pre-Surge), Life Orb one copy from Koga (needs a 125
  base-Speed mon), Expert Belt/Focus Sash from Chuck, Choice items from the
  late Johto leaders; care packages give berries, type boosters and the like
  per stage (Sitrus from Route 3, Leftovers and Expert Belt ×10 after Brendan
  in Fuchsia, Life Orb ×10 after Blaine). `items.json` (build_rr.progression, 2026-10-14) holds
  the held items the item model can price with the trainer index they are in hand at: the
  care-package stages (red / blue fills left out) and the overworld's single rewards; the
  probes hand one to each of the final six (gauntlet.assign_items). Resist berries are not
  priced yet (the matrix knows them, the item model does not).
- Trade evolutions use the Link Cable item (2100 in Celadon 4F); evolution
  stones are all buyable there.

## Arena effects per mandatory boss (tracker `battleEffect`)

Sandstorm (Brock, Silph Giovanni), rain (Misty), Electric Terrain (Surge),
Grassy Terrain (Erika), **doubles + Trick Room (Sabrina)**, Tailwind (Koga),
Desolate Land (Blaine), **doubles + Primordial Sea/Snow (Lorelei)**, Misty
Terrain (Bruno), Psychic Terrain (Agatha), no hazard removal (Lance, Clair),
plus 16 mandatory mini-bosses with permanent weather and ~20 **doubles**
fights (Game Corner and Hideout guards, Silph Ariana & Archer with partner
Brendan, several route mini-bosses). Weather / terrain / Tailwind / Trick Room
are in the sim since 2026-10-11 (`SimBattle(arena=)`, `rr_probe.arena()`);
doubles and the custom rules (Brock −33%, Erika's grass moves, Swamp, Magma
Storm) are not.

## Gaps, in the order they matter (3-5 partly closed 2026-10-11: `build_rr.progression()` gives trainer order, encounter gates and TM/tutor availability by trainer index; route trainers, gifts/statics and the care-package item timeline are still open)

- **#14 LASS RELLI (Nugget Bridge) has no team data**: the trainer order lists her but
  the tracker has no boss block for her and the dump has no "Lass Relli"; the probe
  skips her ("no team data"). The tracker's Nugget Bridge mini bosses are Cale, Ali,
  Timmy (doubles) and Ethan. (2026-10-11)

1. **Doubles.** Sabrina, Lorelei and ~20 mandatory fights are doubles. Showdown
   simulates them; the mirror, calc bot, searcher and RR-AI port are singles
   only. Either the run skips them (not a real answer) or the battler grows a
   doubles mode.
2. **Custom arena rules** (Brock −33% on super-effective hits, Erika's grass
   moves, Swamp, Magma Storm, pre-burned / pre-boosted mons): per-boss scripts
   in the bridge, or left out and declared. The generic effects are done.
3. **Trainer sequence with levels.** Bosses: tracker order (95 mandatory) with
   caps; "Highest Lv − n" resolves against the cap. Route trainers: 470 dump
   slots have a FireRed map via `fr_slots.json`; map → progression segment is a
   hand-written table (Kanto route order), and "Max Level − n" then resolves
   against that segment's cap. ~160 dump trainers are rematch/unused slots.
4. **Encounters with progression**: 83 tracker locations with rarity and
   levels, no order. Same map → segment table as (3); gifts/statics/fossils/
   trades from the locations sheet as fixed-level additions.
5. **Items/TMs/tutors per segment**: care packages (stage headers + red/blue
   fills), overworld items, shops (Cerulean TMs, Celadon stones/Link Cable),
   tutors; TM availability per segment replaces today's `before_trainer: 0`.
6. **Player legality**: drop `hardcore_rules.banned_moves` from the player's
   move pool, apply ability swaps, Toxic/Leech whitelist, no EVs, 31 IVs,
   natures (free to choose? mints are not in the item sheet — assume the
   caught nature, or ask).
7. **Enemy sets**: EVs 0 and the dump's IVs are in; `parse_dump` must still
   keep "Max Level − n" mons (796 of 2308 are dropped today — all route
   trainers and the scaled bosses), resolving them against the segment cap.
8. **AI fidelity**: `ai_util/rr_ai.py` is a CFRU port; doubles targeting and
   the arena rules are not in it.

## Build plan (code, once the user confirms the mode)

`scripts/build_rr.py` grows readers for the three sheets (zip + regex, no
openpyxl) and `fr_slots.json`, and emits the same shape as `data/bdsp/`:
`order.json` (segments → trainers with resolved levels, doubles flag, arena),
`encounters.json` (routes with `before_trainer`), `items.json` / `tms.json`
with `before_trainer`, `trainers.json` with EV-0 sets. The env then needs the
legality filter, the arena carry edits, and a decision on doubles.
