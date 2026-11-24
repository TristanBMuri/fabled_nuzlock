# Radical Red 4.1 (Hardcore Mode): Item Availability & Koga/Blaine Reference Teams

Research compiled 2026-11-24 for the Radical Red 4.1 Hardcore Mode Nuzlocke planner.  
Repo inspected read-only: `/home/tristan/CodingProjects/nuzlock_rl` (`data/rr/items.json`, `data/rr/hardcore_rules.json`, `data/rr/order.json`, `scripts/build_rr.py`, `data/rr/sheets/*.xlsx`).

## Primary Sources Verified Live

| Label | URL | What was verified |
|---|---|---|
| **Official RR Docs Drive Folder** | https://drive.google.com/drive/folders/1YaYM-8dzRlBRuJm1bmYrjJC6HGwTwl-x | Directory of all official v4.1 docs |
| **Items/TMs/Tutors Sheet v4.1** (`ITEMS_SHEET`) | https://docs.google.com/spreadsheets/d/16vBrWJDrsw5QsZyiJjD8ACH7079ZCkQ5BaPtioJOPTk/edit | Overworld items, Shops, Care Packages (`SO2Toxic`), Other Useful Items (matches local `data/rr/sheets/rr_items_tms_tutors.xlsx`) |
| **Locations & Raid Dens Sheet v4.1** (`RAIDS_SHEET`) | https://docs.google.com/spreadsheets/d/15mUFUcN8250hRL7iUOJPX0s1rMcgVuJPuHANioL4o2o/edit | Raid Den badge star unlocks & item drop tables (matches local `data/rr/sheets/rr_locations_raids.xlsx`) |
| **Hardcore Mode Info & Bosses v4.1** (`HC_SHEET`) | https://docs.google.com/spreadsheets/d/1jDbKFA30xo8csPHZNLtsmqs781bW_Xb9mKoPYyE6KK8/edit | Hardcore restrictions, level caps, Koga & Blaine v4.1 boss teams (matches local `data/rr/sheets/rr_hardcore_info.xlsx`) |
| **Evolution Changes v4.1** (`EVO_DOC`) | https://docs.google.com/document/d/1hJCmOWxWA3H70S5zCHuEMNlsQEQL22IiOEAJoFpfEwM/edit | Evolution stones & Link Cable availability in Celadon Dept. Store 4F |
| **4.1 Changelog** (`LOG_41`) | https://docs.google.com/document/d/15q2O-OgtaqucghOEINeefYYWu8mAepuXLCOt6_yHnyc/edit | Care Package retroactive redemption, Eviolite behavior |
| **Changelog up to v4.0** (`LOG_40`) | https://docs.google.com/document/d/1Iudi1QxBSHOH-jhVrNF-IdaK_ixqbmacvgMmBce8n1c/edit | Focus Sash non-consumption/Chuck swap, Nature Changer Nurse, Raid 5★/6★ Choice item rules, removal of wild Life Orb/Rocky Helmet/Focus Sash |
| **Official RR 4.1 Pokédex Data** (`RR_DEX`) | https://raw.githubusercontent.com/JwowSquared/Radical-Red-Pokedex/master/data.js | Complete wild Pokémon held-item tables (1,343 species) & item registry (749 items; confirms 0 Mint items) |
| **Official RR HC Nuzlocke Hall of Fame** (`HC_HOF`) | https://docs.google.com/spreadsheets/d/1l9Vmohnlkz8KgHIFEzCQ2UK5-PRGT16dNDYnsJgrH7A/htmlview/sheet?headers=true&gid=1916946302 | Official winners list (13 players through Nov 2025, 5 on v4.10: Zen1th, Byonemil, camplaysoboe, Eizan Will, MLS) |

---

## Part 1: Item Availability in Radical Red 4.1 (Hardcore Mode)

### Key Mechanics & Hardcore Differences (All `stated in the source`)

1. **Care Packages (`SO2Toxic` NES code in Pallet Town bedroom):** Gives bundles (usually ×10) of every item otherwise farmable from wild Pokémon or Raid Dens up to each milestone (`ITEMS_SHEET` Care Packages tab; `LOG_40`; `LOG_41`). Items highlighted **red** (`#FFC2C2`) are omitted in Hardcore (Psychic/Misty/Electric Seeds, Throat Spray, Weather Rocks, Air Balloon, Terrain Extender, Weakness Policy); **blue** (`#D2EFFF`) are omitted in MGM/Hardcore (EV vitamins/wings). None of the items requested in Part 1 are red/blue-omitted in Hardcore.
2. **Raid Den Star Scaling by Badge Count (`RAIDS_SHEET` Raid Dens tab):**
   - 1 badge (post-Brock, `t=7`): 1★
   - 2 badges (post-Misty, `t=19`): 1★ or 2★
   - 3 badges (post-Surge, `t=27`): 2★
   - 4 badges (post-Erika, `t=35`): 2★ or 3★
   - 5 badges (post-Sabrina, `t=52`): 3★ or 4★
   - 6 badges (post-Koga, `t=81`): 4★
   - 7 badges (post-Blaine, `t=91`): 4★ or 5★
   - 8 badges (post-Clair, `t=98`): 5★
   - Post-Hall of Fame: 5★ or 6★
3. **Stat-Check NPC Rewards in Hardcore / MGM (`ITEMS_SHEET` Overworld Items tab):** Because Hardcore auto-enables Minimal Grinding Mode (0 EVs), the EV-threshold NPCs (Chuck, Koga, Jasmine, Blaine) replace their 150 EV check with a **125 Base Stat** check in the relevant stat.
4. **Focus Sash Persistence (`LOG_40`):** Focus Sash is **returned after battle** (even wild/raid battles) rather than permanently consumed, and only 1 copy exists per run (wild Machamp no longer holds it).
5. **Hardcore Unobtainables (matching `data/rr/hardcore_rules.json`):** Weakness Policy, Throat Spray, Air Balloon, Light Clay, Weather Rocks (Damp/Heat/Icy/Smooth), Booster Energy in Cerulean Cave, Power Plant follower Eject Button, and pre-badge Terrain Seeds (Electric pre-Surge, Misty pre-Sabrina, Psychic pre-Koga) (`HC_SHEET` Main tab).

### Reference Boss Indices in `data/rr/order.json`
`t=0` Rival (Route 22) · `t=1` Brendan (Viridian Forest) · `t=5` Falkner · `t=6` Brock · `t=7` Lass Sally (Route 3) · `t=8` Miguel (Mt. Moon) · `t=10` Rival (Cerulean) · `t=17` Bugsy · `t=18` Misty · `t=20` Camper Jeff · `t=21` S.S. Anne · `t=24` Whitney (Route 11) · `t=25` Tucker · `t=26` Lt. Surge · `t=27` Alicia (Route 9) · `t=28` Caitlin (Route 9) · `t=33` Ace Mary (Celadon Gym) · `t=34` Erika · `t=44` Rival (Silph Co.) · `t=46` Giovanni (Silph Co.) · `t=47` Ace Cole (Route 8) · `t=50` Chuck (Saffron Dojo) · `t=51` Sabrina · `t=53` Rocker Luca (Route 12) · `t=74` Brendan (Safari Zone) · `t=75` Brock Rematch · `t=80` Koga · `t=81` Jojo Fan (Power Plant, Surf unlocked) · `t=82` Pryce · `t=83` May · `t=84` Jasmine · `t=85` Scientist Ted · `t=90` Blaine · `t=91` Cerulean Cave Grunt 1 · `t=97` Clair · `t=102` Ace Naomi (Victory Road) · `t=110` Lorelei (E4).

---

### Full Item Availability Table (RR 4.1 Hardcore)

| Item | Earliest Main-Game Location & Gate (RR 4.1 HC) | Additional Main-Game / Post-Game Sources | Confidence & Source |
|---|---|---|---|
| **Focus Sash** | **Saffron City Fighting Dojo (×1):** Defeat Silph Co. Giovanni for **Mega Ring** (`t=46` → `t=47`), defeat **Chuck** (`t=50`), and show a Pokémon with **125+ Base Attack** (MGM/HC; 150 Atk EVs in normal). **Gate: `t=51` (pre-Sabrina, 4 badges, post-Silph Giovanni & Chuck).** Reusable (not consumed permanently). | None (explicitly capped at 1/run; wild Machamp removed in `LOG_40`). | **Stated in the source** (`ITEMS_SHEET` Overworld; `LOG_40`) |
| **Assault Vest** | **Cinnabar Island Pokémon Lab (×1):** After Koga + Surf (`t=81`), defeat **Jasmine** (`t=84`) and show a Pokémon with **125+ Base Defense** (MGM/HC). **Gate: `t=85` (pre-Blaine, 6 badges, post-Jasmine).** *(Note: Jasmine is inside Cinnabar Lab, reachable via Surf before fighting May at Pokémon Mansion, so technically reachable at `t=81` if doing Jasmine first.)* | None pre-E4. | **Stated in the source** (`ITEMS_SHEET` Overworld; `order.json` timing *inferred* for Lab-before-May) |
| **Choice Specs** | **Cinnabar Island Gym (×1):** Defeat **Blaine** (`t=90`) and show a Pokémon with **125+ Base Sp. Atk** (MGM/HC). **Gate: `t=91` (post-Blaine / 7 badges, pre-Cerulean Cave).** | Post-game only: 6★ Raid Dens (Cerulean Cave / Power Plant, 5% total) & Post-Game Care Package ×10 (`ITEMS_SHEET`; `RAIDS_SHEET`; `LOG_40`). | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RAIDS_SHEET`; `LOG_40`) |
| **Choice Band** | **Cinnabar Island Pokémon Lab (×1):** Defeat **Jasmine** (`t=84`). **Gate: `t=85` (pre-Blaine, 6 badges, post-Jasmine).** *(Same Lab-before-May note as Assault Vest.)* | Post-game only: 6★ Raid Dens (5% total) & Post-Game Care Package ×10. | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RAIDS_SHEET`; `LOG_40`) |
| **Choice Scarf** | **Route 20 East / Seafoam entrance (×1):** Defeat **Pryce** (`t=82`, requires Surf after Koga). **Gate: `t=83` (pre-May & pre-Blaine, 6 badges, post-Pryce).** | Post-game only: 6★ Raid Dens (5% total) & Post-Game Care Package ×10. | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RAIDS_SHEET`; `LOG_40`) |
| **Life Orb** | **(A) 4★ Raid Dens (farmable, 5% total = 4%+1%):** Available as soon as 4★ dens unlock at **5 Gym Badges (post-Sabrina, `t=52`, pre-Koga!)** (Route 2, Route 10, Route 25, etc.). **(B) Guaranteed static copy (×1): Fuchsia City** from **Koga** after defeating him (`t=80`) and showing a Pokémon with **125+ Base Speed** (MGM/HC) → **Gate: `t=81` (post-Koga / 6 badges, pre-May & pre-Blaine).** | **Care Package after defeating Blaine (`t=91`):** ×10 copies (`ITEMS_SHEET`). Also 5★ Raid Dens (7+ badges, 5%). Wild Absol no longer holds it (`LOG_40`). | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RAIDS_SHEET`; `LOG_40`; `RR_DEX`) |
| **Expert Belt** | **(A) Saffron City Dojo (×1):** Defeat **Chuck** (`t=50`). **Gate: `t=51` (pre-Sabrina, post-Chuck).** *(Note: Chuck can be fought as soon as Saffron opens after Erika/Hideout if Mega Ring is only required for Focus Sash, or at `t=50` in `order.json`.)* **(B) Bulk ×10:** Care Package after **Defeating Brendan in Fuchsia City (`t=74` → `t=75`, pre-Koga)**. | **Important correction vs. older Reddit posts:** Wild Makuhita (Black Belt 5%), Sawk (None), and Throh (None) do **NOT** hold Expert Belt in RR 4.1 (`RR_DEX`). | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `LOG_40`; `RR_DEX`) |
| **Leftovers** | **(A) Route 16 & Route 12 Snorlax (×2 static + wild held):** Use Itemfinder on the exact tiles where both Snorlax slept (`t=53`, post-Sabrina/Poké Flute, pre-Koga), plus wild Snorlax/Munchlax hold Leftovers (50%/5% in `RR_DEX`). **(B) Bulk ×10:** Care Package after **Defeating Brendan in Fuchsia City (`t=74` → `t=75`, pre-Koga)**. | None further needed (×10 Care Package covers full team of 6 before Koga). | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RR_DEX`) |
| **Sitrus Berry** | **(A) Care Package: Route 3 After Lass Sally (`t=8`, ×10, pre-Mt. Moon Archer).** **(B) 1★ Raid Dens (30%):** Unlocked after **1 badge (post-Brock, `t=7`)** on Route 3, Route 4, Mt. Moon, etc. **(C) Wild held (5%):** Teddiursa, Furret, Bibarel, Audino, Gulpin, Trumbeak, Greedent, etc. (`RR_DEX`). | Hidden overworld pickups on Route 6 (`t=20`) & Route 23 (`t=91`). | **Stated in the source** (`ITEMS_SHEET` Care Packages/Overworld; `RAIDS_SHEET`; `RR_DEX`) |
| **Eviolite** | **Route 11 (×1):** Defeat **Whitney** (`t=24`). **Gate: `t=25` (pre-Lt. Surge, 2 badges, post-Whitney).** | **Victory Road 5★ Raid Den (5%):** Available at `t=102` (8 badges / Victory Road). **Post-game:** 6★ Raid Dens (5%) & Post-Game Care Package ×10. Not held by wild mons (`RR_DEX`). | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RAIDS_SHEET`; `RR_DEX`) |
| **Heavy-Duty Boots** | **Victory Road 2F (×2):** Western side of floor near a boulder. **Gate: `t=102` (post-Clair & Route 23 Brendan, 8 badges, pre-Victory Road Ace Naomi / pre-E4).** | None earlier (not in Care Packages, Raid Dens, Shops, or wild held items). | **Stated in the source** (`ITEMS_SHEET` Overworld; `RAIDS_SHEET`; `RR_DEX`) |
| **Rocky Helmet** | **Route 8 (×1):** Defeat **Ace Trainer Cole** (`t=47`) — **requires Mega Ring** (obtained after defeating Giovanni in Silph Co., `t=46`). **Gate: `t=48` (post-Silph Giovanni & Ace Cole, pre-Sabrina).** | **Victory Road 5★ Raid Den (25%):** Available at `t=102` (8 badges). **Post-game:** 6★ Raid Dens (25%) & Post-Game Care Package ×10. Wild Shelmet no longer holds it (`LOG_40`; `RR_DEX`). | **Stated in the source** (`ITEMS_SHEET` Overworld/Care Packages; `RAIDS_SHEET`; `LOG_40`; `RR_DEX`) |
| **All 18 Resist Berries** | **ALL 18 available ×10 via Care Packages (`SO2Toxic`) by Route 9 (`t=28`, pre-Erika), none restricted in Hardcore:** See exact breakdown table below. Also ALL 18 drop from 1★–4★ Raid Dens (`RAIDS_SHEET`), and 13/18 are farmable from wild Pokémon (`RR_DEX`). | See per-berry table below. | **Stated in the source** (`ITEMS_SHEET` Care Packages; `RAIDS_SHEET`; `RR_DEX`) |
| **Evolution Stones (all 10) + Link Cable** | **All 10 stones (Fire, Water, Thunder, Leaf, Sun, Moon, Ice, Dusk, Dawn, Shiny) + Link Cable buyable unlimited at Celadon Dept. Store 4F ($3,000 each; Link Cable $2,100) → Gate: `t=33` (entering Celadon City, 3 badges, pre-Erika).** Earlier single pickups exist for 7/10 stones (see breakdown below); **Dusk, Dawn, and Ice Stones** have **no** earlier overworld/raid/wild source before Celadon 4F (`t=33`). | Early pickups: **Moon Stone** ×10 Care Package Exiting Pewter (`t=7`) + ×2 Mt. Moon (`t=8`) + Mt. Moon 1★ Raid (30%) + wild Clefairy/Jigglypuff (5%); **Thunder Stone** Route 3 (`t=7`); **Water Stone** Route 4 West (`t=8`); **Sun Stone** Route 4 East (`t=10`); **Leaf Stone** Route 25 (`t=17`); **Fire Stone** Route 11 (`t=24`); **Shiny Stone** S.S. Anne from Brendan (`t=24`). | **Stated in the source** (`ITEMS_SHEET` Shops/Overworld/Care Packages; `EVO_DOC`; `RAIDS_SHEET`; `RR_DEX`) |
| **Mints / Nature Changer** | **NO Mint items exist in RR 4.1** (`0` mints in `RR_DEX` item table). Instead, the **Nature Changer Nurse** next to Nurse Joy inside **every Pokémon Center** changes any Pokémon's nature **for free, unlimited times**. **Gate: `t=0` (Viridian City Pokémon Center, reachable before Rival Route 22 / Brendan / Brock).** | Available at every Pokémon Center throughout the run (`LOG_40` also confirms she offers Bashful/Gentle/Lax, Instant HP toggle, Gender Swap, and Level Cap display). | **Stated in the source** (`RR_DEX`; `LOG_40`; Reddit/Wiki consensus) |

#### Per-Berry Breakdown: All 18 Resist Berries (RR 4.1 Hardcore — All `stated in the source`)

| Resist Berry (Type) | Care Package Stage (`ITEMS_SHEET`) | Care Package `before_trainer` | Raid Den Source (`RAIDS_SHEET`) | Wild Held 5% (`RR_DEX`, 50% noted) |
|---|---|---|---|---|
| **Charti** (Rock) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Mt. Moon 1★/3★ (50%) | Starly/Staravia/Staraptor (50%), Taillow/Swellow (5%) |
| **Chilan** (Normal) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 6 1★ (50%), Route 11 1★ (80%) | Rattata/Raticate, Lechonk/Oinkologne |
| **Coba** (Flying) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 3 1★ (50%) | Caterpie/Metapod, Weedle/Kakuna |
| **Kebia** (Poison) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 13 3★ (50%) | Shroomish/Breloom |
| **Occa** (Fire) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 2/10/25/Safari/Mansion 4★ (50%) | Pansage/Simisage |
| **Passho** (Water) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 24 1★ (30%) | Pansear/Simisear |
| **Payapa** (Psychic) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 16 2★ (80%) | Mankey/Primeape/Annihilape |
| **Rindo** (Grass) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 8 2★ (80%) | Panpour/Simipour |
| **Yache** (Ice) | Entering Route 2 (×10) | `t=1` (pre-Brendan/Brock) | Route 12 2★ (50%) | Starly/Staravia/Staraptor |
| **Tanga** (Bug) | Exiting Viridian Forest (×10) | `t=5` (pre-Falkner/Brock) | Route 25 1★ (30%) | Spoink/Grumpig |
| **Chople** (Fighting) | Exiting Pewter City (×10) | `t=7` (pre-Lass Sally/Archer) | Route 9 2★ (80%) | Buneary/Lopunny |
| **Colbur** (Dark) | Route 4 East after Mt. Moon (×10) | `t=10` (pre-Cerulean Rival/Misty) | Route 4 1★ (50%) | None |
| **Wacan** (Electric) | Nugget Bridge after Grunt (×10) | `t=17` (pre-Bugsy/Misty) | Route 5 1★ (50%) | Buizel/Floatzel |
| **Haban** (Dragon) | Cerulean City after Grunt (×10) | `t=20` (pre-Camper Jeff/Surge) | Route 6 1★ (80%) | Gible/Gabite/Garchomp |
| **Roseli** (Fairy) | Before Entering Vermilion City (×10) | `t=21` (pre-S.S. Anne/Surge) | Route 11 1★ (50%) | None |
| **Shuca** (Ground) | Before Entering Vermilion City (×10) | `t=21` (pre-S.S. Anne/Surge) | Diglett Cave 1★ (50%) | None |
| **Babiri** (Steel) | Route 9 after 1st mandatory trainer (×10) | `t=28` (pre-Caitlin/Erika) | Rock Tunnel 2★ (80%) | None |
| **Kasib** (Ghost) | Route 9 after 1st mandatory trainer (×10) | `t=28` (pre-Caitlin/Erika) | Route 10 2★ (80%) | None |

---

### Disagreements vs. `data/rr/items.json` (Every Difference Listed)

Root causes found in `scripts/build_rr.py`:
- **Bug A (line 353):** `if "EV" not in it["location"]:` drops every overworld NPC reward whose tooltip mentions EVs (Focus Sash, Life Orb from Koga, Assault Vest, Choice Specs), even though Hardcore/MGM uses 125 Base Stat instead of EVs!
- **Bug B (lines 157–166):** `HELD` omits Focus Sash, Assault Vest, Rocky Helmet, Heavy-Duty Boots, all 18 resist berries, evolution stones, and nature changing.
- **Bug C (line 329):** `if iid not in held or w < held[iid]["before_trainer"]:` keeps only the earliest source and never records later copy upgrades (e.g., 1 → 6 copies from Fuchsia Care Package for Expert Belt and Leftovers).
- **Bug D (lines 223–229 & 355–361, bonus):** Mega Ring is gated at `before_trainer: 44` (Silph Rival) in `items.json` instead of `47` (after Silph Giovanni `t=46`), because `when()` failed to match `"Giovanni in Silph Co."` to `t=46` and fell back to `loc_index("SAFFRON")` (`t=44`).

| Item | Our Gate in `data/rr/items.json` | Official Source Location & Boss Gate (RR 4.1 HC) | Disagreement Summary |
|---|---|---|---|
| **Focus Sash** | **Missing** | Saffron Dojo: post-Silph Giovanni (Mega Ring, `t=46`) + Chuck (`t=50`) + 125 Base Atk → **`t=51` (pre-Sabrina), ×1 reusable** | Completely missing (Bug A + Bug B). Huge for Sabrina, Koga, Blaine, E4. |
| **Assault Vest** | **Missing** | Cinnabar Lab: post-Koga/Surf + Jasmine (`t=84`) + 125 Base Def → **`t=85` (pre-Blaine; or `t=81` if Jasmine before May), ×1** | Completely missing (Bug A + Bug B). |
| **Choice Specs** | **Missing** | Cinnabar Gym: post-Blaine (`t=90`) + 125 Base SpA → **`t=91` (pre-Cerulean Cave), ×1** | Already in `HELD` (line 165) but silently dropped by Bug A (`"EV"` filter)! |
| **Life Orb** | `before_trainer: 91` (after Blaine care package, `copies: 6`) | **1) 4★ Raid Dens:** 5% drop post-Sabrina (**`t=52`, pre-Koga**); **2) Static ×1 from Koga** (`t=80` + 125 Base Spe → **`t=81`, pre-May & pre-Blaine**); **3) ×10 Care Package post-Blaine (`t=91`)** | **10 trainers too late** for guaranteed ×1 (Bug A dropped Koga's Life Orb, preventing Life Orb use vs. May/Blaine!) and misses 4★ raids pre-Koga. |
| **Rocky Helmet** | **Missing** | Route 8 Ace Trainer Cole (`t=47`, requires Mega Ring from Silph Giovanni `t=46`) → **`t=48` (pre-Sabrina), ×1** (+ Victory Road 5★ Raid 25% at `t=102`) | Completely missing (Bug B). |
| **Heavy-Duty Boots** | **Missing** | Victory Road 2F → **`t=102` (post-Clair / Route 23 Brendan, pre-E4), ×2** | Completely missing (Bug B). |
| **Expert Belt** | `before_trainer: 51` (Chuck, `copies: 1`) | Static ×1 from Chuck (`t=51`, matches), **PLUS ×10 Care Package after Defeating Brendan in Fuchsia (`t=75`, pre-Koga)** | Copy count stuck at `1` instead of upgrading to `6` at `t=75` (Bug C). |
| **Leftovers** | `before_trainer: 53` (Route 16 Snorlax, `copies: 1`) | Route 16 & Route 12 Snorlax (×2 static + wild held at `t=53`), **PLUS ×10 Care Package after Defeating Brendan in Fuchsia (`t=75`, pre-Koga)** | Copy count stuck at `1` instead of `2` at `t=53` and `6` at `t=75` (Bug C). |
| **All 18 Resist Berries** | **Missing** | Care Packages (`SO2Toxic`) give ×10 of all 18 between **`t=1` (Route 2) and `t=28` (Route 9, pre-Erika)**; also 1★–4★ Raid Dens & 13/18 wild held | All 18 completely missing (Bug B). Heavily used in real Koga/Blaine HC nuzlocke teams. |
| **Evolution Stones (all 10) + Link Cable** | **Missing** | All 10 stones + Link Cable buyable unlimited at **Celadon Dept. Store 4F (`t=33`, pre-Erika)**; earlier single pickups for Thunder (`t=7`), Moon (`t=7` CP / `t=8`), Water (`t=8`), Sun (`t=10`), Leaf (`t=17`), Fire (`t=24`), Shiny (`t=24`) | Not tracked in `items.json` (`Shops` tab not parsed by `build_rr.py`). |
| **Mints / Nature Changer** | **Missing** (`README.md` L98 unsure) | **No mint items**; free unlimited **Nature Changer Nurse** in every Pokémon Center starting **Viridian City (`t=0`)** | Planner can legally assign **any nature freely from `t=0`** (no mint gating). |
| **Choice Scarf** | `before_trainer: 83` (Pryce, `copies: 1`) | Route 20 East Pryce (`t=82` → `t=83`, ×1) | **Agrees** (matches). |
| **Choice Band** | `before_trainer: 85` (Jasmine, `copies: 1`) | Cinnabar Lab Jasmine (`t=84` → `t=85`, ×1; Lab reachable at `t=81` before May) | **Agrees** with `order.json` order (note Lab is reachable pre-May). |
| **Eviolite** | `before_trainer: 25` (Whitney, `copies: 1`) | Route 11 Whitney (`t=24` → `t=25`, ×1); Victory Road 5★ Raid 5% (`t=102`) | **Agrees** for earliest copy (misses 5★ Raid extra copies at `t=102`). |
| **Sitrus Berry** | `before_trainer: 8` (Route 3 CP, `copies: 6`) | Route 3 After Lass Sally Care Package (`t=8`, ×10); 1★ Raids at `t=7` | **Agrees** (1★ Raids give 30% farmable copies 1 trainer earlier at `t=7`). |

---

## Part 2: Reference Player Teams for Koga & Blaine (Radical Red Hardcore Mode Nuzlockes)

> **Important Version & Mode Caveat:** Many YouTube/Reddit runs titled *"Radical Red 4.1 Hardcore Nuzlocke"* play **Normal Mode** with Hardcore Nuzlocke rules (where Koga has Swellow/Drapion/Greninja and Blaine has Armarouge/Slither Wing/Typhlosion — e.g., ShadyPenguinn, Waterbird, MasterPoucine, DTMBBrown's Reddit guides). All of those were verified and **excluded** below. Every team listed here was verified against **Radical Red Hardcore Mode** bosses:
> - **v4.0/4.1 HC Koga (Cap 68, Permanent Tailwind + Tapu Lele Psychic Terrain):** Tapu Lele, Hoopa-Unbound, Krookodile, Chi-Yu, Nidoking, Mega Toxtricity (`HC_SHEET`). *(v3.02/3.1 HC Koga was identical except Kyurem instead of Chi-Yu.)*
> - **v4.0/4.1 HC Blaine (Cap 76, Permanent Desolate Land):** Sandy Shocks, Great Tusk, Walking Wake, Venusaur (Focus Sash), Ho-Oh (Phoenix Down — revives once at 50% HP), Mega Charizard X (`HC_SHEET`). *(v3.02/3.1 HC Blaine had Heliolisk, Darmanitan, Typhlosion, Groudon, Venusaur, Mega Charizard Y.)*

---

### A. Koga Teams (Level Cap 68 — Permanent Tailwind + Psychic Terrain)

#### Team K1 — Crappo (RR 4.1 Hardcore Mode Nuzlocke — **Deathless**)
- **Source:** https://www.youtube.com/watch?v=3Ai_X-IxK30 (Koga fight at 11:28–16:31; full run recap https://www.youtube.com/watch?v=uq8pKMP_3SQ)
- **Version & Rules:** Radical Red **4.1 Hardcore Mode Nuzlocke** (Italian streamer; beat the full 4.1 HC Nuzlocke).
- **Level Cap:** 68 | **Deathless:** **Yes (0 deaths)** (`stated in the source` — transcript confirms full survival and explicitly jokes Gyarados never even entered).
- **Player's Six:**
  1. **Snorlax** — Def-boosting nature (item/moves not all read aloud; tanks Tapu Lele's Psyshock safely through crits and handles lead Tapu Lele). *(Confidence: species & Def+ nature `stated in the source`)*
  2. **Meganium** — Item: **Leftovers**; Moves shown: **Draining Kiss** (walls Krookodile completely, forces Krookodile switch into Hoopa-U, later finishes Krookodile). *(Confidence: `stated in the source`)*
  3. **Incineroar** — Ability: **Intimidate** (from recap `uq8pKMP_3SQ`); Moves shown: **Leech Life**, **Drain Punch** (switches into Hoopa-U's Expanding Force, checkmates & KOs Hoopa-U, then takes back-to-back KO on Chi-Yu). *(Confidence: `stated in the source`)*
  4. **Probopass** — Handles **Nidoking** and **Mega Toxtricity** (closes out both; team packed Ground moves so Toxtricity comes out last). *(Confidence: species & matchup `stated in the source`)*
  5. **Gyarados** — Brought on the 6-mon roster as backup; **unused** ("warmed the bench"). *(Confidence: `stated in the source`)*
  6. **Runerigus** *(or Golisopod/Weezing — 6th slot not sent out)* — In pre-battle planning (11:38) Crappo specifically highlights Runerigus with Earthquake for Mega Toxtricity alongside Snorlax/Meganium/Incineroar/Probopass/Gyarados; Golisopod+Weezing was discussed as an alternate Hoopa-U line. *(Confidence: `inferred` for exact 6th slot since only 4 mons entered battle + Gyarados named on bench)*

#### Team K2 — Stephin Sunny (Official HC Nuzlocke HOF #5 — **Deathless**)
- **Source:** https://www.youtube.com/watch?v=qBrjV712epA (Koga at ~29:15–31:30; HOF #5 on `HC_HOF`)
- **Version & Rules:** Radical Red **v3.02 Hardcore Mode Nuzlocke** (completed full run; Koga had Kyurem instead of 4.1's Chi-Yu, other 5 mons + Tailwind + Psychic Terrain identical).
- **Level Cap:** 68 | **Deathless:** **Yes (0 deaths)** (`stated in the source`).
- **Player's Six (all 6 explicitly named in transcript):**
  1. **Forretress** — Ability: **Sturdy**; Move: **Gyro Ball** (leads vs. Tapu Lele, survives HP Fire via Sturdy, OHKOs with Gyro Ball).
  2. **Chesnaught** — Handles Krookodile, pivots Nidoking's Earth Power.
  3. **Incineroar** — Item: **Chople Berry**; Move: **Leech Life** (switches into Hoopa-U Expanding Force, baits Z-move with Chople Berry, KOs with Leech Life; pivots Kyurem).
  4. **Jellicent** — Move: **Scald** (pivots in via Chesnaught and KOs Nidoking).
  5. **Sylveon** — Moves: **Draining Kiss**, **Quick Attack** (switches into Kyurem & finishes it off).
  6. **Runerigus** — Move: **Earthquake** (switches into Mega Toxtricity's Boomburst immunity & OHKOs with Earthquake).
- **Confidence:** All 6 species, abilities/items/moves listed above `stated in the source`.

#### Team K3 — Apecio Unbound ("Guaranteed 0 Deaths" HC Nuzlocke Guide — **Deathless**)
- **Source:** https://www.youtube.com/watch?v=9iFA7EMQgRo (0:16–6:29)
- **Version & Rules:** Radical Red **v3.02 Hardcore Mode Nuzlocke** (deterministic 0-death strategy guide; Koga had Kyurem instead of Chi-Yu).
- **Level Cap:** 68 | **Deathless:** **Yes (0 deaths)** (`stated in the source`).
- **Player's Six (5 of 6 named in spoken transcript):**
  1. **Delphox** — Move: **Wish** (leads vs. Tapu Lele; Wish AI-baits switches).
  2. **Meganium** — Move: **Bulldoze** (pivots on burned Krookodile, drops incoming Kyurem's Speed -1).
  3. **Mega Sceptile** — Moves: **Detect**, **Dual Chop** (OHKOs -1 Speed Kyurem, chips Krookodile).
  4. **Cinderace** — Moves: **Iron Head**, **U-turn** (outspeeds -1 Tapu Lele, U-turns on AI's Hoopa-U switch-in).
  5. **Donphan** — Ability: **Sturdy**; carries a Steel-type move to deter Tapu Lele switch-in while baiting Nidoking.
  6. *6th slot:* Not named in the audio transcript (could not view video frames; honestly omitted).
- **Confidence:** 5 species & listed moves/abilities `stated in the source`; 6th slot `unsure` (not spoken).

#### Team K4 — TheFetCet (Full 6-Mon Hardcore Mode Koga Sets in Video Description)
- **Source:** https://www.youtube.com/watch?v=ETEkE3bf1ZU (published Apr 17, 2023 → v3.1 Hardcore Mode; themed run using girlfriend's favorite Pokémon, not explicitly a Nuzlocke).
- **Level Cap:** 68 (`inferred` from HC cap) | **Deathless:** **Unsure** (no spoken transcript or comments confirming deaths).
- **Player's Six (verbatim from video description — showcases standard HC Koga bulk/Counter/resist-berry archetypes):**
  1. **Forretress** — Brave, **Sturdy**, **Sitrus Berry**
  2. **Pyukumuku** — Sassy, **Innards Out**, **Sitrus Berry**
  3. **Dragonite** — Adamant, **Multiscale**, **Focus Sash**
  4. **Runerigus** — Sassy, **Wandering Spirit**, **Rindo Berry**
  5. **Wailord** — Modest, **Blubber Defense**, **Wacan Berry**
  6. **Blissey** — Calm, **Natural Cure**, **Chople Berry**
- **Confidence:** All 6 species, natures, abilities, and items `stated in the source` (video description); movesets & death count `unsure`.

*Additional 4.0/4.1 HC Koga videos accessed (titles/descriptions/comments only; no English captions or team lists in description):*
- Sav (`https://www.youtube.com/watch?v=2SViuSPuG6k` — RR 4.1 HC Koga, Cap 68; comments mention Sylveon and paralyzing Hoopa-U).
- BANG KHAFID (`https://www.youtube.com/watch?v=nEkLc_UaBLQ` — 2-slot deathless RR 4.1 HC Koga; comments mention Light Screen & Salac Berry).
- Pikachu's Playbook (`https://www.youtube.com/watch?v=1DVSbHUybwI` — RR 4.0 HC Koga, No Deaths).

---

### B. Blaine Teams (Level Cap 76 — Permanent Desolate Land)

#### Team B1 — Crappo (RR 4.1 Hardcore Mode Nuzlocke — **1 Planned Death**)
- **Source:** https://www.youtube.com/watch?v=uq8pKMP_3SQ (Blaine fight at ~1:06:00–1:16:00)
- **Version & Rules:** Radical Red **4.1 Hardcore Mode Nuzlocke** (verified against exact 4.1 Blaine roster: Sandy Shocks, Walking Wake, Focus Sash Venusaur, Phoenix Down Ho-Oh, Great Tusk, Mega Charizard X).
- **Level Cap:** 76 | **Deathless:** **No — 1 planned sacrifice (Drednaw)** (`stated in the source`).
- **Player's Six (all 6 fully verified from transcript):**
  1. **Emboar** — Ability: **Bull Rush**; Item: **Choice Band**; Move: **Flare Blitz** (Sun-boosted Banded Bull Rush Flare Blitz OHKOs lead Sandy Shocks cleanly without triggering a Walking Wake switch).
  2. **Lanturn** — Move: **Volt Switch** (switches into Walking Wake's Hydro Steam/Draco Barrage and safely Volt Switches into Dragapult). *(ASR rendered name as "l'ur" in Italian, confirmed as Lanturn from earlier mentions in same video at 799s/1210s/1446s/2773s; `inferred` from ASR match.)*
  3. **Dragapult** — Nature: **Quiet**; Item: **Life Orb**; Moves: **Dragon Darts**, **Psychic Fangs**, **Sucker Punch**, **Solar Beam** (outspeeds & KOs Walking Wake with Dragon Darts; baits Chlorophyll Focus Sash Venusaur, breaks Sash with Psychic Fangs + finishes with Sucker Punch; KOs Mega Charizard X with Dragon Darts; Quiet + Solar Beam threatens Great Tusk to manipulate AI switch order).
  4. **Drednaw** — Nature: **Bold** (0 Atk IV); Ability: **Shell Armor**; Item: **Choice Scarf**; Move: **Rock Tomb** (intentionally weakened so Ho-Oh does not see a KO and switch to Great Tusk; absorbs Flyinium-Z / Earthquake and dies as a planned sacrifice).
  5. **Golduck** — Ability: **Cloud Nine**; Item: **Focus Sash**; Moves: **Simple Beam**, **Flip Turn** (Cloud Nine bypasses Desolate Land's Water immunity; Focus Sash guarantees living Ho-Oh's Sacred Fire to click **Simple Beam**, overwriting **Phoenix Down** so Ho-Oh cannot revive, then KOs permanently with Flip Turn).
  6. **Mega Slowbro** — Ability: **Shell Armor**; Move: **Psychic** (crit-immune Shell Armor, cleanly OHKOs Great Tusk to finish the fight).
- **Confidence:** Emboar, Dragapult, Drednaw, Golduck, Mega Slowbro + all listed items/moves/abilities/natures `stated in the source`; Lanturn `inferred` (Italian ASR "l'ur" cross-checked against Crappo's earlier Lanturn references in the same video).

#### Team B2 — VersacePokemon (RR 4.0 Hardcore Mode — Same Exact 6-Mon Blaine Roster as 4.1; Non-Nuzlocke)
- **Source:** https://www.youtube.com/watch?v=DK7ZlMDIm6g (full commentary transcript)
- **Version & Rules:** Radical Red **4.0 Hardcore Mode** (verified exact same 6 Blaine mons as 4.1: Sandy Shocks, Walking Wake, Mega Charizard X, Great Tusk, Ho-Oh with Phoenix Down, Focus Sash Venusaur; **standard Hardcore Mode playthrough, not a Nuzlocke**).
- **Level Cap:** 76 (`inferred` from 4.0/4.1 HC cap) | **Deathless:** **No** (lost Mega Alakazam, Iron Thorns, Iron Moth, and Dragonite was fainted/swapped mid-battle — `stated in the source`).
- **Player's Six (all 6 explicitly named in commentary):**
  1. **Sandy Shocks** — Speed+ nature; Ability: **Protosynthesis**; Item: **Shuca Berry**; Move: **Earth Power** (wins the lead Sandy Shocks mirror).
  2. **Mega Alakazam** — Move: **Psychic** (2HKOs Walking Wake while surviving Draco Barrage; chunks Mega Charizard X before fainting to Dragon Claw).
  3. **Iron Thorns** — Item: **Shuca Berry**; Moves: **Stone Edge**, **Accelerock** (survives Ho-Oh's Sacred Fire + Solar Beam, KOs Ho-Oh with Stone Edge, then priority Accelerock KOs the revived 50% HP Ho-Oh; chips Great Tusk before fainting).
  4. **Dragonite** — Ability: **Multiscale**; Item: **Silk Scarf**; Move: **Dual Wingbeat** (switches into Great Tusk's Headlong Rush immunity, chips with Dual Wingbeat).
  5. **Iron Jugulis** — Move: **Air Slash** (outspeeds & KOs defense-dropped Great Tusk; survives Mega Charizard X Flare Blitz so Flare Blitz recoil secures the win).
  6. **Iron Moth** — Moves: **Flamethrower**, **Discharge** (4× resists Venusaur's Solar Beam, breaks Focus Sash with Sun-boosted Flamethrower, tanks Weather Ball, finishes Venusaur with Discharge).
- **Confidence:** All 6 species and listed moves/items/abilities `stated in the source`.

#### Team B3 — Stephin Sunny (HOF #5) & Apecio Unbound (RR v3.02 Hardcore Mode Nuzlocke — **Deathless 2-Mon**)
- **Sources:** Stephin Sunny https://www.youtube.com/watch?v=qBrjV712epA (35:30–37:40) & Apecio Unbound https://www.youtube.com/watch?v=ooMo6D3YI9s (0:10–1:15)
- **Version & Rules:** Radical Red **v3.02 Hardcore Mode Nuzlocke** *(included for historical comparison & because Cloud Nine Golduck carried straight into 4.1 meta; note v3.02 Blaine had Heliolisk, Darmanitan, Typhlosion/Charizard-Y, Groudon, Venusaur instead of 4.1's Paradox trio + Ho-Oh)*.
- **Level Cap:** 76 (`inferred`) | **Deathless:** **Yes (0 deaths, only 2 Pokémon used)** (`stated in the source`).
- **Player's Team (2-mon core):**
  1. **Mega Sceptile** — Moves: **Dual Chop**, **Rock Tomb** (Apecio ran 0 Def IV to bait Charizard kill score; sweeps Heliolisk, Darmanitan, Charizard/Typhlosion).
  2. **Golduck** — Ability: **Cloud Nine**; Moves: **Encore**, **Surf** (Cloud Nine shuts off Desolate Land & Venusaur's Chlorophyll, Encores Venusaur into Growth, KOs Groudon and Venusaur with Surf).
- **Confidence:** Both species, abilities, and moves `stated in the source`.

#### Team B4 — Zen1th (`u/Dragonking732`, Official HOF #9 — World's First 4.1 HC Nuzlocke Winner — **Deathless Blaine**)
- **Sources:** https://reddit.com/r/pokemonradicalred/comments/1cemjm3/radical_red_nuzlocke_deathless_blaine_with_only_1/ (Reddit video `v.redd.it/oruq8f98o2xc1`, raw footage) & https://reddit.com/r/pokemonradicalred/comments/1b4tkjx/rr_41_hc_1_death_equals_reset_plans_and_nuzlocke/
- **What could / could not be accessed:** The Reddit video (`v.redd.it`) could not be decoded visually (no `ffmpeg`/video frames), and Zen1th's linked spreadsheet (`https://docs.google.com/spreadsheets/d/1Nbs1javjbOUhcGt1jugVNeIwQq_T2mZpdBhEwcntY3U/edit`, `Boss Fight Guide` tab) was blank. From his posts/comments we confirmed:
  - **Version & Outcome:** RR **4.1 Hardcore Mode Nuzlocke**, **Deathless Blaine risking only 1 crit** (`stated in the source`).
  - **Noted tech:** He specifically highlighted **Wide Lens Kleavor (Stone Axe)** as a recurring lead across his 4.0/4.1 deathlocke splits to set Stealth Rock immediately (breaking sashes/chipping Fire-Flying threats like Ho-Oh/Charizard), while warning Blaine was one of the highest-risk fights in 4.1 HC (`stated in the source`).

*Additional 4.0/4.1 HC Blaine videos accessed (titles/descriptions/comments only; no spoken English transcript):*
- Sav (`https://www.youtube.com/watch?v=emqsfCi1si4` — RR 4.1 HC Blaine, Cap 76; comment notes **Regirock** was used).
- BANG KHAFID (`https://www.youtube.com/watch?v=1Z4w-cId91I` — 2-slot deathless RR 4.1 HC Blaine; comment mentions **Hitmonlee**).
- ZayneAlmighty (`https://www.youtube.com/watch?v=3hy6tZgdug8` — 3-slot deathless RR 4.0 HC Blaine; comment mentions **Hitmonlee**).

---

## Summary of Access Limitations (Transparency Note)
- **Accessed directly & verified:** All official Radical Red 4.1 Google Docs/Sheets (live exports/titles), the official RR 4.1 Pokédex (`data.js` from `JwowSquared/Radical-Red-Pokedex`), the official RR HC Nuzlocke Hall of Fame spreadsheet, Reddit posts/comments via `api.pullpush.io` (direct `reddit.com` returned 403), and YouTube video pages/descriptions/comments/captions via YouTube Innertube API.
- **Could not access directly:** Visual video frames (no `yt-dlp`/`ffmpeg`), `v.redd.it` video streams (Zen1th's deathless Blaine clip), Twitch VODs (`twitch.tv/eizanwill`, `twitch.tv/zen1thhcn`, etc.), and `pokecommunity.com` / `fandom.com` (HTTP 403). Where a video had no captions (e.g., Sav, BANG KHAFID, Masterball, Pikachu's Playbook), only visible title/description/comment metadata is reported—no species or moves were guessed.
