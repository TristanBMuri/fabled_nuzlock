# Human todo (2026-10-10)

Things only you can decide or do. Tick them off and say so in chat.

## Remote setup — done 2026-10-11

`origin` = https://github.com/TristanBMuri/fabled_nuzlock.git, branch `main`
(tracking), tags `pre-cleanup` / `cleanup-done` pushed. The harvest history
was purged (five 119–135 MB versions of `battles.jsonl` were over GitHub's
limit): 102 commits rewritten, pack 13.4 → 4.1 MB, forced push over the four
September commits whose content is merged in (PLAN.md kept at the root).
A bundle of the pre-purge repo is in the agent's scratchpad for this session
only. Pushes run from here via the host's git; no SSH key needed (HTTPS).

## Clean-up decisions

5. **Behaviour-cloning pipeline** — **deleted 2026-10-12** (code under the `pre-cleanup` tag,
   recipe in NOTEBOOK.md; `models/` and `data/teams/` on disk are yours to remove). Was:
   (`ai_util/collect_game.py`, `pretrain_bc.py`,
   `battler_policy.py`, `battler_obs.py`, the prior branches in `evaluate.py` and the
   runner, `--harvest`): **delete** (recommended; the calc bot replaced the prior and
   the notebook holds the recipe) or **keep** as a trimmed fallback. Delete also drops
   torch and stable-baselines3 from the dependencies.

6. **`no_AI_code_here/`** — **decided 2026-10-12: leave it alone** (in the repo, untouched)
   unless told otherwise or it becomes a problem.

## Radical Red: what only you can answer (2026-10-10) — answered 2026-10-12

7. RR 4.1 Hardcore, species/dupes clause on, nothing else.
8. Starter random across all nine generations (as the probes do); choosing it by some
   learned process is a later idea.
9. Singles only for now; doubles stay "not played" (TODO.md F later).
10. The 95 mandatory bosses, expanding to route trainers when the capabilities are
    there; optional bosses may be fought for their items.
11. Natures free, any ability slot after Cerulean.
12. First encounter only: the boxes mode is the real run, the planned pool an upper bound.

All six match what the sim assumes today (`rr_probe.sample_box`, `choose_nature`,
`choose_ability`). The original questions, for the record:

The data is in hand (`data/rr/README.md`: official sheets, ROM dump, FireRed
slot map, hardcore rules). These are the facts about *your* run that the docs
cannot settle; one line each in chat is enough.

7. **Version and mode.** Radical Red 4.1, Hardcore (= Minimal Grinding) is
   assumed. Anything else on (randomizer, "Restricted" differences, Nuzlocke
   patch, species/dupes clause, shiny clause)?

8. **Starter** (Bulbasaur / Charmander / Squirtle, or another option your
   patch offers) — the rival's team depends on it.

9. **Doubles.** ~20 mandatory fights incl. Sabrina and Lorelei are doubles.
   Is "skip them, score singles only" acceptable as a first answer, or must
   the battler learn doubles before the result counts?

10. **Which fights count.** Only the 95 mandatory tracker bosses, or every
    route trainer in the way as well (470 dump slots have a map; a run that
    counts them needs the route-order table, a day of work)? Optional bosses
    (Johto leaders, Chuck, Pryce …) give the only Eviolite / Expert Belt /
    Choice items: fight them or not?

11. **Natures and abilities.** Are natures free (mints / nature changer in
    your patch?) or the caught one? Ability Pills are in the Cerulean care
    package (×306), so any slot is assumed available after Cerulean.

12. **DexNav / encounter choice.** The probes assumed the player can pick any
    species in a route's pool. In your rules: first encounter only, or DexNav
    search allowed?

## Done by Claude once 1–3 are ticked

- `git remote add origin <url>`, `git branch -M main`, `git push -u origin main --tags`
- verify with `ssh -T git@github.com` from the host
