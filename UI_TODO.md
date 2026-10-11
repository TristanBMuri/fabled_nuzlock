# Run overview web UI — todo (2026-10-11)

**Built 2026-10-11: `ui/`** (`ui/build.py` + `ui/index.html`). Serve with
`uv run python ui/build.py --serve` → http://localhost:8765 (data.json is rebuilt on
every fetch; the page refetches every 30 s). Only runs started after this have an
events file; older logs are not parsed. Tabs: Runs (cards, per-boss / per-box
tables, matrix per team and box), Compare, Deaths, Traces, Matrices; the
"results/" box adds the history. Traces → "watch" opens the fight in Showdown's own
replay viewer (`/replay?file=…&turn=N`, p1's protocol log; sprites and the viewer load
from play.pokemonshowdown.com, so it needs internet). The plan below is kept as the spec.

A local page that shows how every Radical Red run is going, in detail, instead
of `scripts/rr_status.py`'s one line per log. Governing constraint still holds
(CLAUDE.md: minimal code): one static HTML file with vanilla JS, one small
Python generator, served with `python -m http.server` on a port outside
8000–8007 (say 8765). No framework, no database, no live server process beyond
that; the page refetches a JSON file every 30 s.

## What it shows

1. **Run cards** — one per run in `logs/` (and `results/` for finished ones):
   name, state (running / done / stopped), started and last-updated times,
   the full configuration (bosses, `--legal --arena --boxes --chooser --matchup
   --death-weight --n-search`, seed), a progress bar (bosses done of N, or
   boxes done of N·bosses), and the headline numbers so far.
2. **Per-boss table** inside a card: cap, arena, pool size, the chosen team
   with evolved species / nature / ability / moves, win %, deaths per mon,
   deathless %, the top-six spread, the two threats with their counters and
   where to catch them, the repaired team if any.
3. **Per-box table** (box runs): box number, the starter, the whole box with
   each mon's source (route / method / rarity), the six fielded, what was left
   in the box, win / deaths / deathless, the killers, the missing counters.
   Sortable by win rate; click a box to see its matchup matrix.
4. **Matchup matrix** for any team against the boss: the `rr_matchups` grid as
   a heat map (hits we need / hits they need, speed, safe cells highlighted),
   with the enemy sets (item, ability, moves) in the header.
5. **Fight trace viewer**: one fight turn by turn — both actives with HP bars
   and status, the bench, every candidate with its search value, the pick,
   the engine's events for the turn; deaths marked. Fed by `rr_trace.py`.
6. **Death analysis panel** per boss: deaths by enemy mon and move, HP before
   the killing hit, switched-in deaths, deaths by turn — the `rr_deaths`
   summaries as small bar charts, side by side for variants (baseline /
   preserve / matchup / death weight).
7. **Comparisons**: pick two runs with the same seeds and see the per-boss
   paired difference (win, deaths, deathless) and which team each chose.
8. **History**: everything under `results/` with the same cards, read-only.

## What has to change in the code first

- **Structured events from the probe.** `rr_team_probe.py` prints lines and
  writes its jsonl only at the end. Add `logs/<run>.events.jsonl`: one JSON
  object per event — `start` (config), `boss_start` (cap, arena, pool, starter,
  boss sets), `stage_a` (top solos), `stage_b` (greedy / cover team), `team`
  (searched team result with sets, threats, counters), `box` (the row the
  box mode already builds), `boss_done`, `end`. The print lines stay.
- **Trace as data.** `rr_trace.py` writes `logs/trace_<boss>_<seed>.json`:
  per turn {ours, theirs, bench, candidates: {choice: value}, pick, events}.
- **Death analysis as data.** `rr_deaths.py` writes its counters to JSON next
  to the log.
- **Matrix as data.** `rr_matchups.py --json` writes the rows.
- **Generator** `ui/build.py`: folds `logs/*.events.jsonl` and
  `results/*.events.jsonl`, traces, death summaries, matrices and the process list into
  `data.json` (runs → bosses → boxes, plus traces / deaths /
  matrices). `--serve` rebuilds it on every fetch.
- **Page** `ui/index.html`: fetches `data.json`, renders the
  panels above; tables sortable, matrix as a coloured grid, trace as a
  scrollable list, charts as plain divs with widths (no chart library).

## Order

1. Events jsonl in the probe + generator + run cards and per-boss tables
   (what the status script shows, properly).
2. Box tables with contents / fielded / left-in-box and missing counters.
3. Matrix heat map (reuse `rr_matchups` rows).
4. Trace viewer and death panels.
5. Comparisons and history.

Serve with `uv run python ui/build.py --serve` and open `http://localhost:8765/`.
Ports 8000–8007 belong to the old project. Producers: `rr_team_probe.py --out X.jsonl`
writes `X.events.jsonl` next to it; `rr_trace.py`, `rr_deaths.py`, `rr_matchups.py`
take `--json` and write `logs/trace_* / deaths_* / matrix_*.json`. To keep a run in
the history, move its `.events.jsonl` to `results/` with the `.jsonl`.
