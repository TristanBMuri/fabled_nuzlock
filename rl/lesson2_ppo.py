"""Lesson 2: deep RL (PPO) on 3v3 battles with switching, on poke-engine, watched live.

Lesson 1's table is replaced by a small network that reads the raw state, and the
action space grows to 4 moves + switches. Trained from scratch against a scripted
opponent; every update and every evaluation goes to logs/rl/<run>.jsonl, which the
cockpit draws while it trains:

    uv run python rl/cockpit.py                          # http://localhost:8766, leave it open
    uv run python rl/lesson2_ppo.py --run first          # fixed teams, ~3.5k decisions/s on one core
    uv run python rl/lesson2_ppo.py --run pool --pool    # 3 random mons a side each battle

The levers (each run's settings are in its log; the cockpit overlays runs to compare):
--algo ppo|a2c|reinforce|bc, --obs (which input groups the net sees), --opp (who it trains
against), --pool (+ --holdout), --init, --shape, and the PPO knobs. --minutes caps a run for equal-time comparisons.

Read top to bottom: teams -> observation -> env -> network -> update -> eval -> log.
Attempt #3's from-scratch PPO stalled at ~14% vs SimpleHeuristics (Showdown, 6M
steps): here the engine is ~100x faster, so you can watch whether it stalls again.
"""

import argparse
import itertools
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from cockpit import pause_all
from poke_engine import (
    Move,
    Pokemon,
    Side,
    State,
    generate_instructions,
    get_all_options,
    monte_carlo_tree_search,
)
from poke_env.data import GenData
from torch import nn

GEN = GenData.from_gen(8)
TYPES = sorted(t.lower() for t in GEN.type_chart)  # 18
STATUS = ["none", "burn", "paralyze", "poison", "toxic", "sleep", "freeze"]  # the engine's names
WEATHER = ["none", "sun", "rain", "sand", "hail", "snow"]

# --- teams: (species, ability, item, moves). Level 50, 85 EVs, 31 IVs, neutral nature.
POOL = [
    ("garchomp", "roughskin", "leftovers", ["earthquake", "dragonclaw", "swordsdance", "stoneedge"]),
    ("rotomwash", "levitate", "leftovers", ["hydropump", "willowisp", "thunderbolt", "protect"]),
    ("ferrothorn", "ironbarbs", "leftovers", ["powerwhip", "gyroball", "leechseed", "stealthrock"]),
    ("volcarona", "flamebody", "leftovers", ["quiverdance", "fierydance", "bugbuzz", "gigadrain"]),
    ("gyarados", "intimidate", "leftovers", ["dragondance", "waterfall", "earthquake", "icefang"]),
    ("tyranitar", "sandstream", "leftovers", ["stoneedge", "crunch", "earthquake", "dragondance"]),
    ("toxapex", "regenerator", "blacksludge", ["scald", "toxic", "recover", "haze"]),
    ("alakazam", "magicguard", "lifeorb", ["psychic", "shadowball", "focusblast", "nastyplot"]),
    ("scizor", "technician", "leftovers", ["bulletpunch", "uturn", "swordsdance", "knockoff"]),
]
FIXED = ([0, 1, 2], [3, 4, 5])  # garchomp, rotomwash, ferrothorn vs volcarona, gyarados, tyranitar
N_MONS = 3


def make_mon(i):
    species, ability, item, moves = POOL[i]
    d = GEN.pokedex[species]
    b = d["baseStats"]
    stat = lambda base: (2 * base + 31 + 21) * 50 // 100 + 5
    hp = (2 * b["hp"] + 31 + 21) * 50 // 100 + 60
    types = tuple(t.lower() for t in d["types"]) + (("typeless",) if len(d["types"]) == 1 else ())
    return Pokemon(id=species, level=50, types=types, base_types=types, ability=ability, base_ability=ability,
                   item=item, hp=hp, maxhp=hp, attack=stat(b["atk"]), defense=stat(b["def"]),
                   special_attack=stat(b["spa"]), special_defense=stat(b["spd"]), speed=stat(b["spe"]),
                   weight_kg=d.get("weightkg", 50.0), moves=[Move(id=m) for m in moves])


def split_teams(holdout):
    """Our possible teams (every trio of POOL) split into train and held-out, the same way every run.
    Held-out trios are never our team in training (each mon still is, in other trios): the held-out
    eval asks whether the net learned to play or memorized its teams."""
    trios = list(itertools.combinations(range(len(POOL)), N_MONS))
    random.Random(123).shuffle(trios)
    k = round(holdout * len(trios))
    return trios[k:], trios[:k]


def new_battle(rng, teams):
    """teams: False = the FIXED matchup; else a list of our possible trios, the foe gets 3 of the rest."""
    if teams:
        ours = list(rng.choice(teams))
        theirs = rng.sample([i for i in range(len(POOL)) if i not in ours], N_MONS)
    else:
        ours, theirs = FIXED
    pad = lambda team: [make_mon(i) for i in team] + [Pokemon.create_fainted() for _ in range(6 - N_MONS)]
    return State(side_one=Side(pokemon=pad(ours)), side_two=Side(pokemon=pad(theirs)))


# --- observation: what the network sees, in named groups (OBS below). Abilities and items
# are in no group: does that matter?
def types(p):
    """The engine hands back what we set (lower case) until a turn runs, then upper case."""
    return [t.lower() for t in p.types if t.lower() != "typeless"]


def effectiveness(move_type, defender_types):
    m = 1.0
    for t in defender_types:
        m *= GEN.type_chart[t.upper()][move_type.upper()]
    return m


def move_features(user, target, moves):
    """Per move: type, power, category, accuracy, effectiveness on the target, STAB, usable."""
    f = []
    for mv in moves[:4]:
        d = GEN.moves[mv.id.lower()]
        acc = 1.0 if d["accuracy"] is True else d["accuracy"] / 100
        eff = effectiveness(d["type"], types(target))
        f += [float(d["type"].lower() == t) for t in TYPES]
        f += [d["basePower"] / 150, float(d["category"] == "Physical"), float(d["category"] == "Special"),
              float(d["category"] == "Status"), acc, math.log2(eff) / 2 if eff else -1.5,
              float(d["type"].lower() in types(user)), float(mv.pp > 0 and not mv.disabled)]
    return f


def best_hit(attacker, defender):
    """log2 of the best effectiveness among the attacker's damaging moves (-2 = immune to all)."""
    effs = [effectiveness(GEN.moves[m.id.lower()]["type"], types(defender)) for m in attacker.moves[:4]
            if GEN.moves[m.id.lower()]["category"] != "Status"]
    best = max(effs, default=1.0)
    return math.log2(best) / 2 if best else -2.0


def active(side):
    return side.pokemon[int(side.active_index)]


def team(side):
    return side.pokemon[:N_MONS]


def boosts(side):
    return [b / 6 for b in (side.attack_boost, side.defense_boost, side.special_attack_boost,
                            side.special_defense_boost, side.speed_boost, side.accuracy_boost, side.evasion_boost)]


def field(side):
    c, vol = side.side_conditions, {v.lower() for v in side.volatile_statuses}
    return [float(c.stealth_rock > 0), c.spikes / 3, c.toxic_spikes / 2, float(c.reflect > 0),
            float(c.light_screen > 0), float(side.substitute_health > 0), float("leechseed" in vol)]


SPECIES = [p[0] for p in POOL]
# Observation groups: what the network may see. --obs picks them; the cockpit compares runs.
# Each takes (state, me, foe) and returns a list of floats.
OBS = {
    "core": lambda st, me, foe: [x for sd in (me, foe) for i, p in enumerate(team(sd)) for x in
                                 (p.hp / p.maxhp if p.maxhp else 0.0, float(p.hp > 0), float(int(sd.active_index) == i))]
                                + [float(me.force_switch), float(foe.force_switch)],
    "status": lambda st, me, foe: [float(p.status.lower() == s) for sd in (me, foe) for p in team(sd) for s in STATUS],
    "types": lambda st, me, foe: [float(t in types(p)) for sd in (me, foe) for p in team(sd) for t in TYPES],
    "stats": lambda st, me, foe: [x / 400 for sd in (me, foe) for p in team(sd) for x in
                                  (p.maxhp, p.attack, p.defense, p.special_attack, p.special_defense, p.speed)],
    "boosts": lambda st, me, foe: boosts(me) + boosts(foe),
    "field": lambda st, me, foe: field(me) + field(foe) + [float(st.weather.lower() == w) for w in WEATHER],
    "moves": lambda st, me, foe: move_features(active(me), active(foe), active(me).moves),
    # beyond the default: what the foe's active can do to ours (hidden information in real play),
    # hand-made matchup features, and plain identity (which species is in which slot)
    "foe_moves": lambda st, me, foe: move_features(active(foe), active(me), active(foe).moves),
    "matchup": lambda st, me, foe: [x for p in team(me) for x in
                                    (best_hit(p, active(foe)), best_hit(active(foe), p), float(p.speed > active(foe).speed))],
    "species": lambda st, me, foe: [float(p.id.lower() == sp) for sd in (me, foe) for p in team(sd) for sp in SPECIES],
}
DEFAULT_OBS = ["core", "status", "types", "stats", "boosts", "field", "moves"]


def parse_obs(spec):
    """'default', 'all', a list ('core,moves'), or changes to the default ('+matchup,-types')."""
    if spec in ("default", "all"):
        return list(OBS) if spec == "all" else list(DEFAULT_OBS)
    parts = spec.split(",")
    if all(p[0] in "+-" for p in parts):
        groups = list(DEFAULT_OBS)
        for p in parts:
            groups = [g for g in groups if g != p[1:]] + ([p[1:]] if p[0] == "+" else [])
    else:
        groups = parts
    bad = [g for g in groups if g not in OBS]
    if bad:
        raise SystemExit(f"unknown --obs groups {bad}; known: {list(OBS)}")
    return [g for g in OBS if g in groups]  # fixed order


def observe(state, side, groups):
    """The state from one side's point of view (side 1 = us, 2 = them), as a flat vector."""
    me, foe = (state.side_one, state.side_two) if side == 1 else (state.side_two, state.side_one)
    return [x for g in groups for x in OBS[g](state, me, foe)]


# --- actions: 4 moves of the active mon, then "switch to team slot i".
N_ACT = 4 + N_MONS


def legal(state, side):
    """{action index: engine choice string}; empty when the side has nothing to choose."""
    me = state.side_one if side == 1 else state.side_two
    opts = get_all_options(state)[side - 1]
    act = me.pokemon[int(me.active_index)]
    out = {}
    for o in opts:
        if o.startswith("switch "):
            out[4 + [p.id.lower() for p in me.pokemon[:N_MONS]].index(o[7:])] = o
        elif o != "No Move":  # the engine's "nothing to choose" (the other side switches)
            out[[m.id.lower() for m in act.moves].index(o)] = o
    return out


def step(state, c1, c2, rng):
    bare = lambda c: "none" if c == "No Move" else c.removeprefix("switch ")  # a switch = bare species
    outcomes = generate_instructions(state, bare(c1), bare(c2))
    pick = rng.choices(outcomes, weights=[o.percentage for o in outcomes])[0]
    return state.apply_instructions(pick)


def winner(state):
    alive = lambda s: any(p.hp > 0 for p in s.pokemon[:N_MONS])
    a, b = alive(state.side_one), alive(state.side_two)
    return 0 if a and b else (1 if a else -1)


def team_hp(side):
    return sum(p.hp / p.maxhp for p in side.pokemon[:N_MONS]) / N_MONS


# --- scripted opponents. Each takes (state, side, rng) and returns a choice string.
def random_player(state, side, rng):
    opts = legal(state, side)
    return rng.choice(list(opts.values())) if opts else "none"


def greedy_player(state, side, rng):
    """Highest base power x STAB x type effectiveness x accuracy; switches only when forced
    (to the bench mon whose types take the least from the foe's STAB)."""
    me, foe = (state.side_one, state.side_two) if side == 1 else (state.side_two, state.side_one)
    opts = legal(state, side)
    if not opts:
        return "none"
    act, foe_act = me.pokemon[int(me.active_index)], foe.pokemon[int(foe.active_index)]
    moves = {a: o for a, o in opts.items() if a < 4}
    if not moves:  # forced switch
        threat = lambda a: max(effectiveness(t, types(me.pokemon[a - 4])) for t in types(foe_act))
        return opts[min(opts, key=threat)]
    def score(a):
        d = GEN.moves[act.moves[a].id.lower()]
        acc = 1.0 if d["accuracy"] is True else d["accuracy"] / 100
        stab = 1.5 if d["type"].lower() in types(act) else 1.0
        return d["basePower"] * stab * effectiveness(d["type"], types(foe_act)) * acc + rng.random()
    return moves[max(moves, key=score)]


def mcts_player(iterations):
    """poke-engine's MCTS, which sees everything (both teams, sets): a strong, slow reference."""
    def play(state, side, rng):
        res = monte_carlo_tree_search(state, 10_000, iterations, 1)
        r = res.side_one if side == 1 else res.side_two
        return max(r, key=lambda x: x.visits).move_choice if len(r) > 1 else (r[0].move_choice if r else "none")
    return play


# --- network: one MLP trunk, a policy head over N_ACT (illegal actions masked) and a value head.
class Net(nn.Module):
    def __init__(self, groups, n_obs, hidden):
        super().__init__()
        self.groups, self.blank = groups, None  # blank: input columns zeroed (the importance test)
        self.trunk = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi, self.v = nn.Linear(hidden, N_ACT), nn.Linear(hidden, 1)

    def observe(self, state, side):
        return observe(state, side, self.groups)

    def forward(self, obs, mask):
        if self.blank is not None:
            obs = obs.clone()
            obs[:, self.blank] = 0.0
        h = self.trunk(obs)
        return self.pi(h).masked_fill(~mask, -1e9), self.v(h).squeeze(-1)


def to_tensors(obs_list, legal_list):
    mask = torch.zeros(len(legal_list), N_ACT, dtype=torch.bool)
    for i, l in enumerate(legal_list):
        mask[i, list(l)] = True
    return torch.tensor(np.array(obs_list, dtype=np.float32)), mask


def policy_player(net, greedy):
    @torch.no_grad()
    def play(state, side, rng):
        opts = legal(state, side)
        if not opts:
            return "none"
        logits, _ = net(*to_tensors([net.observe(state, side)], [opts]))
        a = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
        return opts[a]
    return play


# --- rollout: N parallel battles, our side samples from the policy, each env's opponent plays its own.
# A transition is recorded only when we had a choice; a battle's last transition gets +1 / -1.
# With a teacher (--algo bc), every decision is also labelled with the teacher's choice. We still
# play the student's move, so it is labelled in the positions it reaches itself (DAgger).
def collect(net, envs, n_steps, shape, rng, stats, teacher=None):
    buf = {k: [] for k in ("obs", "legal", "act", "logp", "val", "rew", "done", "env", "target")}
    while len(buf["act"]) < n_steps:
        ready = [(e, legal(e["state"], 1)) for e in envs]
        mine = [(e, l) for e, l in ready if l]
        if mine:
            obs, mask = to_tensors([net.observe(e["state"], 1) for e, _ in mine], [l for _, l in mine])
            with torch.no_grad():
                logits, val = net(obs, mask)
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)
        choices = {}
        for k, (e, l) in enumerate(mine):
            a = int(acts[k])
            choices[id(e)] = l[a]
            target = a
            if teacher:
                target = {v: i for i, v in l.items()}.get(teacher(e["state"], 1, rng), a)
                stats["agree"].append(target == a)
            if e["last"] is not None:
                buf["done"][e["last"]] = False
            e["last"] = len(buf["act"])
            for key, x in (("obs", obs[k]), ("legal", mask[k]), ("act", a), ("logp", float(logps[k])),
                           ("val", float(val[k])), ("rew", 0.0), ("done", False), ("env", e["i"]), ("target", target)):
                buf[key].append(x)
            name = l[a] if a < 4 else "switch"
            stats["actions"][name] = stats["actions"].get(name, 0) + 1
        for e, _ in ready:
            s = e["state"]
            before = team_hp(s.side_one) - team_hp(s.side_two)
            s = step(s, choices.get(id(e), "none"), e["opp"][1](s, 2, rng), rng)
            e["turns"] += 1
            w = winner(s)
            if e["last"] is not None:
                buf["rew"][e["last"]] += shape * (team_hp(s.side_one) - team_hp(s.side_two) - before)
            if w or e["turns"] >= 100:
                if e["last"] is not None:
                    buf["rew"][e["last"]] += float(w)
                    buf["done"][e["last"]] = True
                stats["results"].setdefault(e["opp"][0], []).append(w)
                stats["lengths"].append(e["turns"])
                s, e["turns"], e["last"] = new_battle(rng, e["pool"]), 0, None
            e["state"] = s
    for e in envs:  # battles still running: cut the trajectory, bootstrap from the value
        if e["last"] is not None:
            buf["done"][e["last"]] = None
        e["last"] = None
    return buf


def advantages(buf, gamma, lam, use_value=True):
    """GAE per battle. done=None marks a cut trajectory: bootstrap from that step's own value.
    use_value=False with lam=1 gives the plain Monte Carlo return (REINFORCE, no baseline)."""
    n = len(buf["act"])
    vals = buf["val"] if use_value else [0.0] * n
    adv, nxt_adv, nxt_val = [0.0] * n, {}, {}
    for t in reversed(range(n)):
        e, d = buf["env"][t], buf["done"][t]
        if d is None:
            nv, na = vals[t], 0.0
        elif d:
            nv, na = 0.0, 0.0
        else:
            nv, na = nxt_val[e], nxt_adv[e]
        delta = buf["rew"][t] + gamma * nv - vals[t]
        adv[t] = delta + gamma * lam * na
        nxt_adv[e], nxt_val[e] = adv[t], vals[t]
    adv = torch.tensor(adv)
    return adv, adv + torch.tensor(vals)


# --- the update. ppo: clipped ratio, several epochs over minibatches. a2c and reinforce are the
# same code with one epoch over one batch (the ratio is then 1, the clip never acts): a plain
# policy gradient, with the value as baseline (a2c) or without (reinforce). bc: cross-entropy
# to the teacher's choice, no reward in the policy loss at all.
def update(net, opt, buf, adv, ret, args):
    obs, mask = torch.stack(buf["obs"]), torch.stack(buf["legal"])
    act, old_logp, old_val = torch.tensor(buf["act"]), torch.tensor(buf["logp"]), torch.tensor(buf["val"])
    target = torch.tensor(buf["target"])
    n, logs = len(act), []
    norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # over the whole batch: a 1-sample minibatch has no std
    for _ in range(args.epochs):
        for idx in torch.randperm(n).split(args.minibatch):
            logits, v = net(obs[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[idx])
            ratio = (logp - old_logp[idx]).exp()
            if args.algo == "bc":
                pi_loss = nn.functional.cross_entropy(logits, target[idx])
            else:
                a = norm_adv[idx]
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            v_loss = 0.5 * (v - ret[idx]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + args.vf * v_loss - args.ent * ent
            opt.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                logs.append([pi_loss.item(), v_loss.item(), ent.item(), (old_logp[idx] - logp).mean().item(),
                             ((ratio - 1).abs() > args.clip).float().mean().item(), gn.item()])
    m = np.mean(logs, axis=0)
    ev = 1 - (ret - old_val).var() / (ret.var() + 1e-8)
    return {"pi_loss": m[0], "v_loss": m[1], "entropy": m[2], "kl": m[3], "clipfrac": m[4], "grad_norm": m[5],
            "explained_var": float(ev), "value_mean": float(old_val.mean()), "adv_std": float(adv.std())}


# --- evaluation: the greedy (argmax) policy against fixed opponents on fixed seeds, the same for
# every run, so runs compare. One battle written out turn by turn.
def evaluate(net, opponents, n, pool, seed):
    me, out = policy_player(net, greedy=True), {}
    for name, opp in opponents.items():
        rng, wins, hp_left = random.Random(seed), 0, 0.0
        for _ in range(n):
            s = new_battle(rng, pool)
            for _ in range(100):
                s = step(s, me(s, 1, rng), opp(s, 2, rng), rng)
                if winner(s):
                    break
            wins += winner(s) > 0
            hp_left += team_hp(s.side_one) if winner(s) > 0 else 0.0
        out[name] = {"win": wins / n, "hp_left": hp_left / max(wins, 1), "n": n}
    return out


def importance(net, opponents, n, pool, seed, sizes):
    """Which inputs the trained net uses: zero one observation group at a time, re-evaluate."""
    out, col = {}, 0
    for g in net.groups:
        net.blank = list(range(col, col + sizes[g]))
        out[g] = {k: v["win"] for k, v in evaluate(net, opponents, n, pool, seed).items()}
        col += sizes[g]
    net.blank = None
    return out


@torch.no_grad()
def show_battle(net, opponent, pool, seed, search_iters):
    """One battle, sampled policy vs the opponent: per turn the actives, HP, our policy and value,
    and what MCTS would play in our place (its visit share per choice)."""
    rng, turns = random.Random(seed), []
    s = new_battle(rng, pool)
    for t in range(100):
        a1, a2 = active(s.side_one), active(s.side_two)
        row = {"turn": t + 1, "us": a1.id.lower(), "us_hp": round(a1.hp / a1.maxhp, 3),
               "them": a2.id.lower(), "them_hp": round(a2.hp / a2.maxhp, 3),
               "us_boost": s.side_one.attack_boost + s.side_one.special_attack_boost,
               "them_boost": s.side_two.attack_boost + s.side_two.special_attack_boost,
               "us_team": round(team_hp(s.side_one), 3), "them_team": round(team_hp(s.side_two), 3)}
        opts = legal(s, 1)
        c1 = "none"
        if opts:
            logits, v = net(*to_tensors([net.observe(s, 1)], [opts]))
            p = torch.softmax(logits, -1)[0]
            row["policy"] = {opts[a]: round(float(p[a]), 3) for a in opts}
            row["value"] = round(float(v), 3)
            if search_iters and len(opts) > 1:
                r = monte_carlo_tree_search(s, 10_000, search_iters, 1).side_one
                total = sum(x.visits for x in r) or 1
                row["search"] = {x.move_choice: round(x.visits / total, 3) for x in r}
            c1 = opts[int(torch.distributions.Categorical(probs=p).sample())]
        c2 = opponent(s, 2, rng)
        row.update(ours=c1, theirs=c2)
        turns.append(row)
        s = step(s, c1, c2, rng)
        if winner(s):
            break
    return {"turns": turns, "result": winner(s)}


@torch.no_grad()
def opening(net):
    """The policy at the first turn of the fixed matchup: lesson 1's "Q at the start"."""
    s = new_battle(random.Random(0), False)
    opts = legal(s, 1)
    logits, v = net(*to_tensors([net.observe(s, 1)], [opts]))
    p = torch.softmax(logits, -1)[0]
    return {"policy": {opts[a]: round(float(p[a]), 4) for a in opts}, "value": round(float(v), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=time.strftime("run_%m%d_%H%M%S"))
    ap.add_argument("--note", default="", help="free text shown in the cockpit's run table")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "reinforce", "bc"],
                    help="bc = imitate MCTS (--mcts-iters) on the positions the net reaches")
    ap.add_argument("--obs", default="default", help=f"observation groups: default, all, a list, or +x,-y; {list(OBS)}")
    ap.add_argument("--pool", action="store_true", help="3 random mons a side from POOL each battle")
    ap.add_argument("--holdout", type=float, default=0.2, help="with --pool: share of our team trios kept for the held-out eval")
    ap.add_argument("--opp", default="greedy",
                    help="training opponents, comma-separated, split across the envs: random, greedy, mcts, self")
    ap.add_argument("--mcts-iters", type=int, default=100, help="training opponent / bc teacher strength")
    ap.add_argument("--steps", type=int, default=2_000_000, help="our decisions in total")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = no limit): equal-time comparisons")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=4096, help="decisions per update")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=5.0)  # attempt #2: 0.5 silently stopped learning
    ap.add_argument("--shape", type=float, default=0.0, help="reward per unit of team-HP difference gained")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=10, help="updates")
    ap.add_argument("--eval-n", type=int, default=200, help="battles per eval opponent")
    ap.add_argument("--eval-mcts", type=int, default=1000, help="the eval MCTS's iterations; keep it fixed across runs; 0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="start from another run's last checkpoint (same --obs and --hidden)")
    args = ap.parse_args()
    if args.algo in ("a2c", "reinforce"):  # one plain gradient step per batch (see update())
        args.epochs, args.minibatch = 1, args.rollout
    if args.algo == "reinforce":
        args.lam = 1.0
    args.obs = ",".join(parse_obs(args.obs))

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = random.Random(args.seed)
    log_path = Path(__file__).resolve().parent.parent / "logs" / "rl" / f"{args.run}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():  # never overwrite a run: its log and net are the record
        raise SystemExit(f"run {args.run!r} exists: pick another --run, or delete logs/rl/{args.run}.*")
    log = log_path.open("w")
    # paused time (the cockpit's pause button) is left out of --minutes and the minutes axis: a = active seconds
    t0, paused = time.time(), [0.0]
    active = lambda: time.time() - t0 - paused[0]
    emit = lambda kind, **kw: (log.write(json.dumps(dict(kind=kind, t=round(time.time(), 1), a=round(active(), 1), **kw)) + "\n"), log.flush())

    def on_pause(*_):  # SIGTSTP from the cockpit: stop here until SIGCONT
        emit("pause")
        s = time.time()
        os.kill(os.getpid(), signal.SIGSTOP)
        paused[0] += time.time() - s
        emit("resume")
    signal.signal(signal.SIGTSTP, on_pause)

    train_teams, held_teams = split_teams(args.holdout) if args.pool else (False, [])
    groups, probe = args.obs.split(","), new_battle(rng, train_teams)
    sizes = {g: len(OBS[g](probe, probe.side_one, probe.side_two)) for g in groups}
    net = Net(groups, sum(sizes.values()), args.hidden)
    if args.init:
        net.load_state_dict(torch.load(log_path.parent / f"{Path(args.init).name}.pt"))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    every = {"random": random_player, "greedy": greedy_player, "mcts": mcts_player(args.mcts_iters),
             "self": policy_player(net, greedy=False)}
    train_opps = [(name, every[name]) for name in args.opp.split(",")]
    teacher = mcts_player(args.mcts_iters) if args.algo == "bc" else None
    eval_opps = {"random": random_player, "greedy": greedy_player}
    if args.eval_mcts:
        eval_opps["mcts"] = mcts_player(args.eval_mcts)
    emit("start", pid=os.getpid(), config=vars(args), n_obs=sum(sizes.values()), obs_sizes=sizes,
         params=sum(p.numel() for p in net.parameters()), mcts_label=f"mcts {args.eval_mcts} iters",
         teams=[POOL[i][0] for i in FIXED[0]] + ["vs"] + [POOL[i][0] for i in FIXED[1]] if not args.pool else SPECIES,
         held=[[POOL[i][0] for i in t] for t in held_teams])
    print(f"run {args.run}: {args.algo}, inputs {sizes}, log {log_path}")
    pause_all(but=args.run)  # one run trains at a time, also when started from the terminal

    envs = [{"i": i, "state": new_battle(rng, train_teams), "turns": 0, "last": None, "pool": train_teams,
             "opp": train_opps[i % len(train_opps)]} for i in range(args.envs)]
    done_steps, n_update = 0, 0
    out_of_time = lambda: args.minutes and active() > 60 * args.minutes

    def do_eval():
        te = time.time()
        ev = evaluate(net, eval_opps, args.eval_n, train_teams, 10_000)
        if held_teams:  # the same opponents, our team never seen in training: "<opp> held"
            ev |= {f"{k} held": v for k, v in evaluate(net, eval_opps, args.eval_n, held_teams, 10_000).items()}
        emit("eval", step=done_steps, update=n_update, elapsed=round(active()), vs=ev, secs=round(time.time() - te, 1))
        show = "mcts" if "mcts" in eval_opps else "greedy"
        battle = show_battle(net, eval_opps[show], train_teams, n_update, args.eval_mcts or args.mcts_iters)
        emit("battle", step=done_steps, update=n_update, vs=show, **battle)
        torch.save(net.state_dict(), log_path.with_suffix(".pt"))
        print(f"step {done_steps:>9,}  " + "  ".join(f"{k} {v['win']:.0%}" for k, v in ev.items()))

    while done_steps < args.steps and not out_of_time():
        if n_update % args.eval_every == 0:
            do_eval()
        tc = time.time()
        stats = {"actions": {}, "results": {}, "lengths": [], "agree": []}
        buf = collect(net, envs, args.rollout, args.shape, rng, stats, teacher)
        tu = time.time()
        adv, ret = advantages(buf, args.gamma, args.lam, use_value=args.algo != "reinforce")
        m = update(net, opt, buf, adv, ret, args)
        done_steps += len(buf["act"])
        n_update += 1
        r = [w for ws in stats["results"].values() for w in ws]
        emit("update", step=done_steps, update=n_update, sps=round(len(buf["act"]) / (time.time() - tc)),
             collect_s=round(tu - tc, 2), learn_s=round(time.time() - tu, 2), elapsed=round(active()),
             battles=len(r), train_win={k: sum(x > 0 for x in ws) / len(ws) for k, ws in stats["results"].items()},
             ep_len=(sum(stats["lengths"]) / len(r)) if r else None,
             teacher_agree=(sum(stats["agree"]) / len(stats["agree"])) if stats["agree"] else None,
             actions={k: v / len(buf["act"]) for k, v in stats["actions"].items()},
             opening=opening(net), **{k: round(float(v), 5) for k, v in m.items()})
    do_eval()
    if len(groups) > 1:
        print("importance: zeroing one input group at a time ...")
        imp_opps = {k: v for k, v in eval_opps.items() if k != "random"}
        emit("importance", step=done_steps, n=args.eval_n // 2,
             base={k: v["win"] for k, v in evaluate(net, imp_opps, args.eval_n // 2, train_teams, 20_000).items()},
             groups=importance(net, imp_opps, args.eval_n // 2, train_teams, 20_000, sizes))
    emit("end", step=done_steps, elapsed=round(active()))


if __name__ == "__main__":
    main()
