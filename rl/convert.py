"""Showdown battle state (sim_bridge.js `dump`) -> poke-engine State, and back to a
comparable outcome summary.

poke-engine maps an unknown species / move / ability / item / volatile to NONE without
an error, so every name is checked by a round trip through its serializer and the misses
are returned as `issues`: a state with issues is not the same battle on both engines.
Approximations the engines cannot share are kept, not hidden: Showdown knows how many
turns a sleep or confusion lasts, poke-engine models it as a chance per turn.
"""

import re
from functools import cache

from poke_engine import (
    Move,
    Pokemon,
    Side,
    SideConditions,
    State,
    VolatileStatusDurations,
)

STATUS = {"": "none", "brn": "burn", "par": "paralyze", "slp": "sleep", "frz": "freeze",
          "psn": "poison", "tox": "toxic"}
BACK = {v: k for k, v in STATUS.items()}
WEATHER = {"": "none", "sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "hail": "hail",
           "desolateland": "harshsun", "primordialsea": "heavyrain"}
TERRAIN = {"": "none", "electricterrain": "electricterrain", "grassyterrain": "grassyterrain",
           "mistyterrain": "mistyterrain", "psychicterrain": "psychicterrain"}
SIDE_CONDITIONS = {"auroraveil": "aurora_veil", "lightscreen": "light_screen", "reflect": "reflect",
                   "mist": "mist", "safeguard": "safeguard", "spikes": "spikes",
                   "stealthrock": "stealth_rock", "stickyweb": "sticky_web", "tailwind": "tailwind",
                   "toxicspikes": "toxic_spikes"}
BOOSTS = {"atk": "attack_boost", "def": "defense_boost", "spa": "special_attack_boost",
          "spd": "special_defense_boost", "spe": "speed_boost", "accuracy": "accuracy_boost",
          "evasion": "evasion_boost"}
# Showdown bookkeeping volatiles with no effect poke-engine needs to know about
IGNORED_VOLATILES = {"choicelock", "stall"}
# Showdown counts these down (turns left), poke-engine up (turns done) to a fixed end;
# confusion and lockedmove get a random length in Showdown and a per-turn chance in
# poke-engine, so they stay approximations
ELAPSED = {"taunt": lambda left: 3 - left, "encore": lambda left: 3 - left,
           "yawn": lambda left: 2 - left}


def _tokens(state: State) -> set[str]:
    return set(re.split(r"[,;=:]", state.to_string()))


@cache
def known(kind: str, name: str) -> bool:
    """Does poke-engine (this build) know this species / move / ability / item / volatile?"""
    fields = {"species": dict(id=name), "ability": dict(ability=name, base_ability=name),
              "item": dict(item=name), "move": dict(moves=[Move(id=name)])}
    side = (Side(volatile_statuses={name}, pokemon=[Pokemon()]) if kind == "volatile"
            else Side(pokemon=[Pokemon(**fields[kind])]))
    return name.upper() in _tokens(State(side_one=side, side_two=Side(pokemon=[Pokemon()])))


def _mon(p: dict, issues: list) -> Pokemon:
    for kind, name in [("species", p["species"]), ("ability", p["ability"])] + \
                      ([("item", p["item"])] if p["item"] else []) + [("move", m) for m, _ in p["moves"]]:
        if not known(kind, name):
            issues.append(f"{kind}:{name}")
    types = [t.lower() for t in p["types"]][:2] + ["typeless"]
    st = p["stats"]
    sl = p["sleep"] or {}
    rest = sl.get("time", 0) if sl.get("rest") else 0  # both engines count Rest sleep down from 3
    slept = 0 if rest or not sl else sl["start"] - sl["time"]
    return Pokemon(id=p["species"], level=p["level"], types=tuple(types[:2]), base_types=tuple(types[:2]),
                   hp=p["hp"], maxhp=p["maxhp"], ability=p["ability"], base_ability=p["ability"],
                   item=p["item"] or "none", attack=st["atk"], defense=st["def"], special_attack=st["spa"],
                   special_defense=st["spd"], speed=st["spe"], status=STATUS[p["status"]],
                   sleep_turns=max(0, slept), rest_turns=rest, weight_kg=p["weightkg"], moves=[Move(id=m, pp=pp) for m, pp in p["moves"]])


def _side(d: dict, foe: dict, issues: list) -> Side:
    mons = [_mon(p, issues) for p in d["pokemon"]]
    active = d["pokemon"][0]
    vols, durations = set(), {}
    for v, left in active["volatiles"].items():
        if v in IGNORED_VOLATILES:
            continue
        if known("volatile", v):
            vols.add(v)
            if v in ELAPSED:
                durations[v] = ELAPSED[v](left)
        else:
            issues.append(f"volatile:{v}")
    if active["ability"] == "slowstart" and active["abilityCounter"] > 0:  # Showdown keeps it in abilityState
        vols.add("slowstart")
        durations["slowstart"] = active["abilityCounter"]  # both count turns left
    # Wish heals this slot; a future move sits on the target's slot in Showdown, on the user's
    # side in poke-engine. Both count the same turns.
    extra = {}
    if "wish" in d["slot"]:
        extra["wish"] = (1, int(d["slot"]["wish"]["hp"]))  # Showdown: always resolves this turn
    fm = foe["slot"].get("futuremove")
    if fm and fm["move"] == "futuresight":
        extra["future_sight"] = (fm["duration"], str(fm["source"]))
    issues += [f"slot:{k}" for k in d["slot"] if k not in ("wish", "healingwish")
               and not (k == "futuremove" and d["slot"][k]["move"] == "futuresight")]
    cond = {}
    for k, n in d["conditions"].items():
        if k in SIDE_CONDITIONS:
            cond[SIDE_CONDITIONS[k]] = n
        else:
            issues.append(f"side:{k}")
    cond["toxic_count"] = active["toxic"]
    if "healingwish" in d["slot"]:  # Gen 8: waits in the slot for a mon that needs it
        cond["healing_wish"] = 1
    names = [m for m, _ in active["moves"]]
    # poke-engine marks "switched in, has not moved yet" with a switch as the last move
    last = ("switch:0" if active["moveActions"] == 0 else
            f"move:{names.index(active['lastMove'])}" if active["lastMove"] in names else "move:none")
    return Side(pokemon=mons + [Pokemon.create_fainted() for _ in range(6 - len(mons))],
                volatile_statuses=vols, side_conditions=SideConditions(**cond),
                volatile_status_durations=VolatileStatusDurations(**durations), last_used_move=last,
                **extra,
                **{BOOSTS[k]: v for k, v in active["boosts"].items()})


def to_engine(dump: dict) -> tuple[State, list[str]]:
    """The dumped battle as a poke-engine State, plus what could not be carried over."""
    issues: list[str] = []
    for key, table in (("weather", WEATHER), ("terrain", TERRAIN)):
        if dump[key] not in table:
            issues.append(f"{key}:{dump[key]}")
    pseudo = set(dump["pseudo"]) - {"sleepclausemod"}  # a clause, not a field effect
    issues += [f"field:{p}" for p in pseudo - {"trickroom"}]
    turns = lambda n: n if n > 0 else -1  # poke-engine: 0 turns left = never ends, -1 too
    state = State(side_one=_side(dump["p1"], dump["p2"], issues), side_two=_side(dump["p2"], dump["p1"], issues),
                  weather=WEATHER.get(dump["weather"], "none"), weather_turns_remaining=turns(dump["weatherTurns"]),
                  terrain=TERRAIN.get(dump["terrain"], "none"), terrain_turns_remaining=turns(dump["terrainTurns"]),
                  trick_room="trickroom" in pseudo, trick_room_turns_remaining=dump["pseudo"].get("trickroom", 0))
    return state, issues


# ---------------------------------------------------------------- outcome summaries
# One turn's result in terms both engines report the same way. `key` is the discrete
# part (which team slot is in, its status and boosts, which slots are alive, hazards,
# weather); `hp` the active mons' HP as fractions, compared on average (poke-engine
# collapses the 16 damage rolls). Slots, not species: formes change (Schooling, Disguise).

def outcome_showdown(d: dict) -> tuple[tuple, tuple]:
    def side(s):
        a = s["pokemon"][0]
        alive = tuple(sorted(p["slot"] for p in s["pokemon"] if p["hp"] > 0))
        cond = tuple(sorted((SIDE_CONDITIONS[k], 1 if k not in ("spikes", "toxicspikes") else n)
                            for k, n in s["conditions"].items() if k in SIDE_CONDITIONS))
        if a["hp"] <= 0:  # a fainted mon: status and boosts are bookkeeping that differs
            return (a["slot"], "fnt", (), alive, cond), 0.0
        boosts = tuple(a["boosts"][k] for k in BOOSTS)
        return (a["slot"], a["status"], boosts, alive, cond), a["hp"] / a["maxhp"]
    (k1, h1), (k2, h2) = side(d["p1"]), side(d["p2"])
    return (k1, k2, WEATHER.get(d["weather"], d["weather"])), (h1, h2)


def outcome_engine(s: State, before: dict) -> tuple[tuple, tuple]:
    """`before`: the dump the state was converted from (engine index -> team slot)."""
    def side(sd, slots):
        a = int(sd.active_index)
        mon = sd.pokemon[a]
        alive = tuple(sorted(slots[i] for i, p in enumerate(sd.pokemon) if i < len(slots) and p.hp > 0))
        sc = sd.side_conditions
        cond = tuple(sorted((attr, 1 if attr not in ("spikes", "toxic_spikes") else getattr(sc, attr))
                            for attr in SIDE_CONDITIONS.values() if getattr(sc, attr) > 0))
        if mon.hp <= 0:
            return (slots[a], "fnt", (), alive, cond), 0.0
        boosts = tuple(getattr(sd, BOOSTS[k]) for k in BOOSTS)
        return (slots[a], BACK[mon.status.lower()], boosts, alive, cond), mon.hp / mon.maxhp
    slots = {k: [p["slot"] for p in before[k]["pokemon"]] for k in ("p1", "p2")}
    (k1, h1), (k2, h2) = side(s.side_one, slots["p1"]), side(s.side_two, slots["p2"])
    return (k1, k2, s.weather.lower()), (h1, h2)
