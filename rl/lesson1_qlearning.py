"""Lesson 1: tabular Q-learning on one 1v1 matchup, on the Rust engine (poke-engine).

Garchomp (us) vs Rotom-Wash (a random-move opponent). The things to discover:
Earthquake does nothing (Levitate), Will-O-Wisp halves our attack, Swords Dance
may or may not pay for itself. Nobody tells the agent; it learns from +1 / -1.

    uv run python rl/lesson1_qlearning.py                 # ~1 min
    uv run python rl/lesson1_qlearning.py --episodes 2000 # watch it fail to converge

Read the code top to bottom: matchup -> env step -> state abstraction -> Q-learning
-> evaluation. Every knob is a module constant or a flag.
"""

import argparse
import random
from collections import defaultdict

from poke_engine import (
    Move,
    Pokemon,
    Side,
    State,
    generate_instructions,
    iterative_deepening_expectiminimax,
)

US = ["earthquake", "dragonclaw", "swordsdance", "stoneedge"]
THEM = ["hydropump", "willowisp", "thunderbolt", "protect"]


def mon(species, types, ability, stats, moves):
    hp, atk, df, spa, spd, spe = stats  # level-50 stats, set by hand
    return Pokemon(id=species, level=50, types=types, base_types=types, ability=ability,
                   base_ability=ability, hp=hp, maxhp=hp, attack=atk, defense=df,
                   special_attack=spa, special_defense=spd, speed=spe,
                   moves=[Move(id=m) for m in moves])


def new_battle():
    empty = [Pokemon.create_fainted() for _ in range(5)]
    us = mon("garchomp", ("dragon", "ground"), "roughskin", (183, 150, 115, 100, 105, 122), US)
    them = mon("rotomwash", ("electric", "water"), "levitate", (125, 85, 127, 125, 127, 106), THEM)
    return State(side_one=Side(pokemon=[us] + empty), side_two=Side(pokemon=[them] + list(empty)))


# --- environment: one turn = both sides pick, the engine lists every outcome with its
# probability (damage rolls, crits, misses), we sample one. 13 us per turn in Rust.
def step(state, ours, theirs, rng):
    outcomes = generate_instructions(state, ours, theirs)
    pick = rng.choices(outcomes, weights=[o.percentage for o in outcomes])[0]
    state = state.apply_instructions(pick)
    a, b = state.side_one.pokemon[0].hp, state.side_two.pokemon[0].hp
    done = a <= 0 or b <= 0
    reward = (1.0 if b <= 0 else -1.0) if done else 0.0  # sparse: win or die, nothing else
    return state, reward, done


# --- state abstraction: the table needs a small discrete key. This choice IS the model;
# lesson 2 replaces it with a network that reads the raw state.
def key(state):
    me, foe = state.side_one, state.side_two
    bucket = lambda p: min(4, int(5 * p.hp / p.maxhp))  # HP in fifths
    return (bucket(me.pokemon[0]), bucket(foe.pokemon[0]), min(me.attack_boost, 6) // 2,
            me.pokemon[0].status.lower() == "burn")  # the engine says "BURN"


def opponent(rng):
    return rng.choice(THEM)


def train(episodes, alpha, gamma, eps_start, seed):
    rng = random.Random(seed)
    q = defaultdict(lambda: [0.0] * len(US))
    for ep in range(episodes):
        eps = eps_start * (1 - ep / episodes)  # linear decay to greedy
        state = new_battle()
        for _ in range(50):
            s = key(state)
            a = rng.randrange(len(US)) if rng.random() < eps else max(range(len(US)), key=q[s].__getitem__)
            state, r, done = step(state, US[a], opponent(rng), rng)
            target = r if done else r + gamma * max(q[key(state)])
            q[s][a] += alpha * (target - q[s][a])  # the whole algorithm is this line
            if done:
                break
    return q


def win_rate(policy, n, seed):
    rng, wins = random.Random(seed), 0
    for _ in range(n):
        state = new_battle()
        for _ in range(50):
            state, r, done = step(state, policy(state, rng), opponent(rng), rng)
            if done:
                wins += r > 0
                break
    return wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=1.0)  # short episodes: no discount needed
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    q = train(args.episodes, args.alpha, args.gamma, args.eps, args.seed)
    greedy = lambda st, rng: US[max(range(len(US)), key=q[key(st)].__getitem__)]
    print(f"{len(q)} table entries after {args.episodes} episodes\n")

    print("Q at the start (full HP, no boost, not burned):")
    for m, v in zip(US, q[key(new_battle())]):
        print(f"  {m:12s} {v:+.3f}")
    worst_case = iterative_deepening_expectiminimax(new_battle(), 500).get_safest_move()
    print(f"poke-engine expectiminimax (opponent plays its best, not randomly): {worst_case}\n")

    print("Win rate vs the random Rotom, 2000 battles each:")
    for name, pol in [("random", lambda st, rng: rng.choice(US)),
                      ("always dragonclaw", lambda st, rng: "dragonclaw"),
                      ("Q-learning greedy", greedy)]:
        print(f"  {name:18s} {win_rate(pol, 2000, args.seed + 1):.1%}")


if __name__ == "__main__":
    main()
