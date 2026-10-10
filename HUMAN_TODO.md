# Human todo (2026-10-10)

Things Claude cannot do from inside the container. Tick them off and say so in chat;
everything after them (remote add, branch rename, push, tags) is done from here.

## Remote setup

1. **Create the repository.** On GitHub, an empty *private* repo: no README, no
   license, no .gitignore. Any name. Paste the SSH URL into chat:
   `git@github.com:<you>/<name>.git`

2. **Register the host key.** GitHub → Settings → SSH and GPG keys → New key, paste:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   Fingerprint to confirm: `SHA256:7yUGJ63+vu3rMZ3v/e6KDtH8oF9+8HvTRNUJASKLmcE`
   (the `xWare_DevOps` key looks like a work key; do not use it here).

3. **Load the key into the agent.** The key has a passphrase, so a push from the
   container would hang on the prompt. In a *host* terminal (outside any distrobox):
   ```bash
   ssh-add ~/.ssh/id_ed25519
   ```
   Alternative if you do not want to do this every login: say so and a passphrase-free
   deploy key for this one repo gets set up instead.

4. **Decide: harvest history.** The tracked harvest jsonl files are untracked now but
   stay in git history (49 MB). Keep (recommended, costs one 49 MB push) or purge
   before the first push, which needs on the host:
   ```bash
   brew install git-filter-repo
   ```

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
