# Human todo (2026-10-10)

Things Claude cannot do from inside the container. Tick them off and say so in chat;
everything after them (remote add, branch rename, push, tags) is done from here.

## Remote setup — status 2026-10-11

Done from here: `origin` = https://github.com/TristanBMuri/fabled_nuzlock.git,
branch renamed to `main`, the September README/PLAN.md from GitHub merged in
(local wins; PLAN.md came along as the original plan). **The push was rejected
by GitHub**: five old versions of `data/teams/gauntlet_harvest/battles.jsonl`
in the history are 119–135 MB each (GitHub's limit is 100 MB per file), so the
harvest history must be purged before the first push — decision 4 below, now
forced. The purge is a history rewrite, which the agent is not allowed to run
unasked. A full backup bundle of the repo as it is now sits at
`/tmp/claude-1001/.../scratchpad/nuzlock_rl_pre_filter.bundle` (copy it
somewhere safe first if you want a belt and braces).

Either say "purge the harvest history" in chat (the agent runs the commands
below), or run them yourself on the host:

```bash
cd ~/CodingProjects/nuzlock_rl && FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --index-filter 'git rm -r -q --cached --ignore-unmatch data/teams/gauntlet_harvest' --prune-empty --tag-name-filter cat -- --all
```
```bash
cd ~/CodingProjects/nuzlock_rl && git for-each-ref --format='%(refname)' refs/original/ | xargs -n1 git update-ref -d && git reflog expire --expire=now --all && git gc --prune=now --aggressive
```
```bash
cd ~/CodingProjects/nuzlock_rl && git push -u origin main --force --tags
```
(`--force` because the rewritten `main` no longer shares history with the
4 commits on GitHub; their content is already merged into ours.) Expected pack
after the purge: well under 100 MB; the 44 MB `results/rr_probe.log` stays.

Also still yours: the `ssh-add` note is moot (HTTPS works from the host).

## Clean-up decisions

5. **Behaviour-cloning pipeline** (`ai_util/collect_game.py`, `pretrain_bc.py`,
   `battler_policy.py`, `battler_obs.py`, the prior branches in `evaluate.py` and the
   runner, `--harvest`): **delete** (recommended; the calc bot replaced the prior and
   the notebook holds the recipe) or **keep** as a trimmed fallback. Delete also drops
   torch and stable-baselines3 from the dependencies.

6. **`no_AI_code_here/`** (your hand-written track): keep in the repo, move out, or delete.

## Radical Red: what only you can answer (2026-10-10)

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
