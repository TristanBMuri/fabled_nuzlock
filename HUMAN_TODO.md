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

## Done by Claude once 1–3 are ticked

- `git remote add origin <url>`, `git branch -M main`, `git push -u origin main --tags`
- verify with `ssh -T git@github.com` from the host
