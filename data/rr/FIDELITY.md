# Sim vs ROM: the fidelity checklist (TODO.md A1 / A3, started 2026-11-23)

Every Radical Red number in `results/` comes from `pokemon-showdown-rr` (mod `gen9rr`) playing
the ROM dump's sets (`rr41_hardcore_trainers.txt`) against the CFRU AI port. None of it has
been checked against the game. This file is the list of checks: what the sim does now, and an
empty **ROM** column to fill from an emulator (mGBA + save states before each boss is enough
for most rows; the in-battle summary screen shows the enemy's ability and level).

Fill a row with what the game shows; a mismatch becomes a fix (data: `rr_probe.boss_team` /
`DUMP_NAMES`; mechanics: `config/custom-formats.ts` or the fork) plus a line in NOTEBOOK.md.

## 1. Data: dump vs official sheet (A3, automatic part done)

`scripts/rr_data_diff.py` → `results/rr_data_diff.log`: trainer order #0–#34 (to Erika),
**98 / 149 shared mons identical**; the rest differ in ability 29, level 24, item 5, nature 3,
moves 4. Brock, Misty, Lt. Surge and Erika agree on species, levels, items, natures and moves.
The sim plays the **dump's** value in every row below. Rows ordered by what the box tables use:

| # | fight | mon | field | sim (dump) | sheet | ROM |
|---|---|---|---|---|---|---|
| 6 | Brock | Varoom | ability | Overcoat | Filter (−25% super-effective damage taken) | |
| 6 | Brock | Lileep | ability | Suction Cups | Storm Drain | |
| 1 | Brendan | Treecko | ability | Overgrow | Unburden | |
| 7 | Lass Sally | Pikachu-Flying | ability | Static | Lightning Rod | |
| 10 | Rival (Cerulean) | Simisear | ability | Gluttony | Sage Power | |
| 12 | Lass Ali | Brionne | ability, nature | Torrent, Modest | Liquid Voice, Bashful | |
| 15 | Camper Ethan | Mabosstiff | nature | Bashful | Adamant | |
| 19 | Grunt | Veluza | ability | Mold Breaker | Sharpness | |
| 23 | Brendan (S.S. Anne) | all six | level | 30 | 28 (= cap) | |
| 28 | Picnicker Caitlin | Spinda | ability | Own Tempo | Contrary | |
| 30 | Maniac Herman | Ninetales / Sylveon | ability | Flash Fire / Cute Charm | Drought / Pixilate | |
| 17, 24, 31 | Bugsy, Whitney, Morty (optional) | most | level, items, moves | dump sets 2–8 levels under the sheet | | |

Sources, 2026-11-23: the official Drive folder ("Radical Red Docs", `1YaYM-8dz…`) holds the
boss sheets for both modes, items, locations, evolution and Pokémon changes, changelogs. **No
official document lists every trainer**; the only complete per-trainer source is Rudo2204's ROM
dump (hardcore / normal / easy gists), whose header says abilities and levels are not 100%.
Brock's Varoom is Overcoat in the normal *and* hardcore dumps and Filter in both official
sheets, so the dump reads abilities one way throughout: one ROM look settles it for all 29.
Both sheets also say "Even if something looks VERY wrong, that's how it is in-game".

**Enemy PP (fixed 2026-11-23):** the hardcore sheet says "All of the opposing Pokémon's moves
in this sheet are PP maxed". The 2026-10-14 bridge reset took the bosses' PP Ups away too; in a
gen9rr format only the player is reset now (`ai_util/sim_bridge.js basePP`).

The pattern: where they disagree, the dump usually has ability slot 0 and the sheet the
hidden ability. The dump's own header warns that its max-level guess is "still wrong a lot
for bosses; check the official doc". **If the ROM sides with the sheet on Varoom and Treecko,
switch the sim to the sheet's abilities wholesale** and rerun the box tables.

Also open: #14 Lass Relli has no team in either source; #21 / #22 (doubles) list only one of
the two trainers in the dump.

## 2. Mechanics (A1): one scripted moment each

For each, note the numbers the game shows (HP bar in pixels / HP text for your own mon, or
the damage % from a calc on your recorded fight). The sim side is computed from the same
setup by `SimBattle` with a fixed seed when you report the ROM number.

| check | setup in the ROM | sim | ROM |
|---|---|---|---|
| stat line, 0 EV / 31 IV | any of your mons at the cap: the summary-screen stats | Showdown's stat formula (EV 0, IV 31, nature) | |
| sand chip + Rock boost | Brock: damage of Hippopotas' Rock Tomb on your lead, sand up | Showdown: 1/16 chip, Rock SpD +50% in sand | |
| permanent AI weather | Misty: does rain outlast 5 turns / a switch-in of your own weather? | permanent (`arena.ai_permanent`) | |
| rain Water boost | Misty: Politoed / Mantine Water move on your mon | ×1.5 | |
| Swift Swim / Sand Rush | turn order Cacnea vs your mon in sand | speed ×2 | |
| Wacan Berry | Electric hit on Misty's Mantine | halved once, berry gone | |
| Brock −33% rule | a super-effective hit on Brock's mons | **not modelled** (full damage) | |
| AI item use | do bosses use Full Restores / X items in hardcore? | never | |
| AI switching | when does the AI switch out (record ~20 turns, TODO A2.2) | CFRU port (`ai_util/rr_ai.py`) | |
| crit rate | — (use Showdown's Gen 9 1/24) unless the game shows otherwise | 1/24 | |

## 3. Rules the sim now applies for the player (`--legal`, 2026-11-23)

`rr_probe.hardcore()`: the 82 banned moves, Toxic / Leech Seed only on the 91 listed
species, and the ability swaps (Drizzle → Adaptability, Speed Boost → Infiltrator, Sand
Stream → Sand Force with Tyranitar → Intimidate, keeps for Spinda / Comfey / Espathra).
Confirm in-game that a swapped ability shows as the replacement (e.g. a caught Yanma shows
Infiltrator, not Speed Boost) and that Toxic is unavailable to a mon off the list.
