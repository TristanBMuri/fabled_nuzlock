"""In-process Showdown battles (via sim_bridge.js) mirrored into poke-env Battle
objects, so existing brains (policy nets, SimpleHeuristics) act on them unchanged.

No websocket server, no logins, no hung challenges. Fork = Showdown state
serialize/deserialize + fresh RNG; the poke-env mirrors are pickle-copied
(under a millisecond, flat in the fight's length; 2026-10-11).
"""

import itertools
import json
import logging
import os
import pickle
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Callable

from poke_env.battle import Battle, Move, Pokemon
from poke_env.data import GenData, to_id_str
from poke_env.stats import compute_raw_stats
from poke_env.player import Player
from poke_env.teambuilder import Teambuilder

BRIDGE = Path(__file__).with_name("sim_bridge.js")
SIDES = ("p1", "p2")
_LOG = logging.getLogger("sim")
_LOG.setLevel(logging.CRITICAL)
_IDS = itertools.count()
_IGNORE = Player.MESSAGES_TO_IGNORE | {"message"}
_DEX = GenData.from_gen(9).pokedex  # base species for nickname dedupe (formes)

# (battle, side) -> Showdown choice string ("move tackle", "switch 2", "default")
Brain = Callable[["SimBattle", str], str]


def dedupe_nicknames(packed: str) -> str:
    """Give repeated species distinct nicknames (Bidoof1, Bidoof2, ...).

    poke-env keys a team by nickname, so two same-species mons collapse into one
    entry: no active mon, no switches, 'switch Bidoof' resolving to the active one.
    Harvest parties have duplicates often; the server path has the same defect.
    Every member of a duplicated species gets a suffix: a bare 'Bidoof' next
    to a 'Bidoof2' made Showdown resolve 'switch Bidoof' to the *active*
    Bidoof2 by species and reject the choice (found 2026-09-19 with all-legal
    search candidates; rare under the prior, which seldom proposed it)."""
    parts = [mon.split("|") for mon in packed.split("]")]
    species_of = lambda f: (f[1] or f[0]) if len(f) >= 2 else None
    # Showdown names a mon whose nickname equals its species after the *base* species
    # ('Raticate-Alola' -> 'p1: Raticate'), so a forme next to its base form collides too
    def base_of(s):  # the dex stores baseSpecies as an id for base forms and a name for formes
        if not s:
            return s
        bid = to_id_str(_DEX.get(to_id_str(s), {}).get("baseSpecies", s))
        return _DEX.get(bid, {}).get("name", s)
    count = Counter(base_of(species_of(f)) for f in parts if len(f) >= 2)
    seen: dict[str, int] = {}
    mons = []
    for f in parts:
        species = species_of(f)
        base = base_of(species)
        if species is not None and count[base] > 1:  # every member: Raticate1, Raticate2
            n = seen[base] = seen.get(base, 0) + 1
            f[0], f[1] = f"{base}{n}", species
        mons.append("|".join(f))
    return "]".join(mons)


class SimBridge:
    def __init__(self, showdown: str | None = None):
        """showdown: a Showdown build directory other than the vendored clone (the
        Radical Red fork); also settable with SHOWDOWN_DIR in the environment."""
        env = dict(os.environ)
        if showdown:
            env["SHOWDOWN_DIR"] = str(Path(showdown).resolve())
        self.proc = subprocess.Popen(
            ["node", str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)

    def call(self, **msg) -> dict:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("sim bridge exited")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


class SimBattle:
    def __init__(self, bridge: SimBridge, fmt: str, p1_team: str | None, p2_team: str | None,
                 seed: list[int] | None = None, dedupe: bool = True, carry: list | None = None,
                 full_info: bool = False, arena: dict | None = None):
        """Packed teams; None lets the format's generator build one (randombattle).
        carry: per p1 mon {hp, status, pp} to start from (a fight without a heal before it).
        full_info: each mirror knows the other side's whole team from turn one (sets,
        levels, items, abilities, real stats) — the full-docs regime, both ways.
        arena: field effects fixed for the fight — {weather, terrain, pseudo: [...],
        p2: [side conditions], ai_permanent} (Radical Red hardcore gyms; sim_bridge.js)."""
        self.bridge, self.fmt, self.full_info, self.arena = bridge, fmt, full_info, arena
        self.gen = GenData.from_format(fmt).gen
        self.id = f"b{next(_IDS)}"
        if dedupe:
            p1_team = dedupe_nicknames(p1_team) if p1_team else p1_team
            p2_team = dedupe_nicknames(p2_team) if p2_team else p2_team
        self.teams = {"p1": p1_team, "p2": p2_team}
        self.history = {s: [] for s in SIDES}
        self.pending = {s: False for s in SIDES}
        self.errors = {s: 0 for s in SIDES}
        self.stats = {"choices": 0, "rejected": 0, "forced_default": 0, "brain_default": 0}
        self.why: Counter = Counter()  # rejection messages / default-choice contexts
        self.ended, self.winner, self.turn = False, "", 0
        self.timed_out = False  # lost at the turn cap (play: max_turns)
        self.chosen: list = []  # (turn, side, choice) as sent: the replay log
        self.mirrors = {s: self._mirror(s) for s in SIDES}
        if bridge is not None and fmt is not None:
            self._absorb(bridge.call(
                op="new", id=self.id, format=fmt, seed=seed, carry=carry, arena=arena,
                p1={"name": "p1", "team": p1_team}, p2={"name": "p2", "team": p2_team}))

    # ------------------------------------------------------------ mirroring

    def _mirror(self, side: str) -> Battle:
        b = Battle(battle_tag=self.id, username=side, logger=_LOG, gen=self.gen)
        if self.teams[side]:
            tb = Teambuilder.parse_packed_team(self.teams[side])
            b._teambuilder_team = tb
            # Pre-populate our own team keyed by nickname. poke-env's lazy path
            # (get_pokemon) renames an existing same-species entry when it meets a
            # new nickname, which corrupts teams with deduped duplicates.
            for m in tb:
                m.nickname, m.species = m.nickname or None, m.species or None
                name = m.nickname or m.species
                b._team[f"{side}: {name}"] = Pokemon(gen=self.gen, teambuilder=m, name=name)
        opp = "p2" if side == "p1" else "p1"
        if self.full_info and self.teams[opp]:
            # both sides know the other's sets (the env built them; the game's trainer AI
            # knows the player's team too): moves, item, ability, level and the real
            # stats, keyed as the protocol will name them
            data = GenData.from_gen(self.gen)
            for m in Teambuilder.parse_packed_team(self.teams[opp]):
                m.nickname, m.species = m.nickname or None, m.species or None
                name = m.nickname or m.species
                m.moves = [x for x in m.moves if to_id_str(x) in data.moves]  # a mod's own moves stay unknown
                mon = Pokemon(gen=self.gen, teambuilder=m, name=name)
                for ms in m.moves:  # a typed Hidden Power: poke-env keeps only "hiddenpower" (Normal); the engine plays the type
                    if to_id_str(ms).startswith("hiddenpower") and to_id_str(ms) != "hiddenpower":
                        mon._moves.base_moves.pop("hiddenpower", None)
                        typed = Move("hiddenpower", raw_id=to_id_str(ms), gen=self.gen)
                        mon._moves.base_moves[typed.id] = typed
                raw = compute_raw_stats(to_id_str(m.species or m.nickname), list(m.evs), list(m.ivs),
                                        m.level or 100, (m.nature or "serious").lower(), data)
                mon.stats = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), raw))
                b._opponent_team[f"{opp}: {name}"] = mon
        return b

    def _absorb(self, reply: dict) -> None:
        for side in SIDES:
            for line in reply[side]:
                self.history[side].append(line)
                self._feed(self.mirrors[side], side, line)
        self.ended, self.winner, self.turn = reply["ended"], reply["winner"], reply["turn"]

    def _feed(self, mirror: Battle, side: str, line: str) -> None:
        if not line.startswith("|"):
            return
        if line.startswith("|request|"):
            payload = line[len("|request|"):]
            if payload:
                req = json.loads(payload)
                mirror.parse_request(req)
                self.pending[side] = not req.get("wait", False)
                self.errors[side] = 0
            return
        parts = line.split("|")
        cmd = parts[1]
        if cmd == "error":  # invalid/unavailable choice: this side must choose again
            self.pending[side] = True
            self.errors[side] += 1
            self.stats["rejected"] += 1
            # strip mon/move names so identical causes aggregate
            self.why["rejected: " + re.sub(r"'s \S.*?(?= is| can| must|$)", "'s X", parts[2])[:90]] += 1
        elif cmd in _IGNORE:
            pass
        elif cmd == "win":
            mirror.won_by(parts[2])
        elif cmd == "tie":
            mirror.tied()
        else:
            try:
                mirror.parse_message(parts)
            except NotImplementedError:
                pass

    # -------------------------------------------------------------- playing

    def step(self, choices: dict[str, str]) -> None:
        for s in choices:
            self.pending[s] = False
        self._absorb(self.bridge.call(op="choose", id=self.id, **choices))

    def choice(self, side: str, brain: Brain) -> str:
        self.stats["choices"] += 1
        if self.errors[side] >= 2:  # brain keeps producing rejected choices
            self.stats["forced_default"] += 1
            return "default"
        c = brain(self, side)
        self.chosen.append((self.turn, side, c))
        if c == "default":
            self.stats["brain_default"] += 1
            m = self.mirrors[side]
            self.why[f"brain default: moves={len(m.available_moves)} switches="
                     f"{len(m.available_switches)} force={m.force_switch} trapped={m.trapped} "
                     f"active={m.active_pokemon is not None} turn={min(self.turn, 1)}"] += 1
        return c

    def play(self, brains: dict[str, Brain], max_turns: int = 100,
             until: int | None = None) -> "SimBattle":
        """Run to the end, or stop (unresolved) once turn >= until. Past max_turns
        p1 is declared the loser (some low-level pairs cannot damage each other)."""
        while not self.ended and (until is None or self.turn < until):
            if self.turn > max_turns:
                self._absorb(self.bridge.call(op="forcelose", id=self.id, side="p1"))
                self.timed_out = True
                break
            choices = {s: self.choice(s, brains[s]) for s in SIDES if self.pending[s]}
            if not choices:
                raise RuntimeError(f"{self.id}: nobody to move, battle not ended")
            self.step(choices)
        return self

    def fork(self, reseed: bool = True, seed: list[int] | None = None) -> "SimBattle":
        """Copy of the battle at this point. seed (4 ints) fixes the copy's RNG so
        two forks with the same seed replay the same chance events (common random
        numbers across search candidates); reseed=True rolls a fresh one."""
        new = SimBattle.__new__(SimBattle)
        new.bridge, new.fmt, new.gen, new.full_info, new.arena = self.bridge, self.fmt, self.gen, self.full_info, self.arena
        new.id = f"b{next(_IDS)}"
        new.teams = self.teams
        new.history = {s: list(self.history[s]) for s in SIDES}
        new.pending = dict(self.pending)
        new.errors = {s: 0 for s in SIDES}
        new.stats = {k: 0 for k in self.stats}
        new.why = Counter()
        new.ended, new.winner, new.turn = self.ended, self.winner, self.turn
        new.timed_out = self.timed_out
        new.chosen = list(self.chosen)
        self.bridge.call(op="fork", id=self.id, new=new.id, reseed=reseed, seed=seed)
        # the mirrors are copied, not replayed: a pickle round trip is ~0.6 ms flat, the
        # replay of the whole history was 1.6 -> 5.8 ms as a fight went on (72 forks per
        # search decision; 2026-10-11). The copy reads identically to a replayed mirror.
        new.mirrors = pickle.loads(pickle.dumps(self.mirrors))
        return new

    def state(self) -> dict:
        return self.bridge.call(op="state", id=self.id)

    def outcome(self) -> tuple[bool, list[bool]]:
        """(p1 won, fainted flags per p1 mon in packed-team order)."""
        st = self.state()
        return st["winner"] == "p1", [m["fainted"] for m in st["p1"]]

    def close(self) -> None:
        self.bridge.call(op="close", id=self.id)


# ---------------------------------------------------------------- brains

def order_to_choice(order) -> str:
    msg = order.message
    return msg[len("/choose "):] if msg.startswith("/choose ") else "default"


def player_brain(player: Player) -> Brain:
    """Wrap any poke-env Player (SimpleHeuristics, PolicyPlayer...) as a Brain."""
    return lambda sb, side: order_to_choice(player.choose_move(sb.mirrors[side]))
