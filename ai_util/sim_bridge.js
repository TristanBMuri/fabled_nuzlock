// JSON-lines driver for Showdown's in-process simulator (no websocket server).
// One request per stdin line, one JSON reply per stdout line.
//
//   {"op":"new","id":"b0","format":"gen8customgamenogimmicks",
//    "p1":{"name":"p1","team":"<packed>"},"p2":{...},"seed":[..]?}   -> flush
//   {"op":"choose","id":"b0","p1":"move 1","p2":"switch 2"}          -> flush
//   {"op":"fork","id":"b0","new":"b1","reseed":true,"seed":[..]?}    -> {"ok":true}
//   new also takes "arena": {weather, terrain, pseudo:[], p2:[], ai_permanent} (see applyArena)
//     seed = 4 ints (Gen5 PRNG) fixes the fork's RNG; reseed=true rolls a fresh one
//   {"op":"forcelose","id":"b0","side":"p1"}                         -> flush
//   {"op":"state","id":"b0"}   -> hp/fainted per mon in original team order
//   {"op":"close","id":"b0"}
//
// flush = {"p1":[lines],"p2":[lines],"ended":bool,"winner":"p1"|"p2"|"","turn":n}
// where lines are the per-side protocol messages (|split| already resolved,
// |request| included), i.e. exactly what each player would get from a server.
'use strict';
const path = require('path');
const readline = require('readline');

// SHOWDOWN_DIR picks another build (e.g. the Radical Red fork); default: the vendored clone
const ROOT = path.join(process.env.SHOWDOWN_DIR ? path.resolve(process.env.SHOWDOWN_DIR) : path.join(__dirname, '..', 'pokemon-showdown'), 'dist');
const { Battle, Teams, TeamValidator } = require(path.join(ROOT, 'sim'));
const { extractChannelMessages } = require(path.join(ROOT, 'sim', 'battle'));

const battles = new Map();

// The server validates teams before a battle, which also fills defaults —
// notably 252 EVs in every stat when the format has no EV limit (Custom Game).
// Without this the sim's mons are weaker than the server's (verified 2026-09-15).
function prepPlayer(format, opts) {
  if (!opts || typeof opts.team !== 'string') return opts;
  const team = Teams.unpack(opts.team);
  if (!team) throw new Error(`cannot unpack team: ${opts.team}`);
  const problems = new TeamValidator(format).validateTeam(team);
  if (problems) throw new Error(`invalid team: ${problems.join('; ')}`);
  return { ...opts, team };
}

// The channel must exist before the battle starts: the opening requests are
// emitted at start (older builds, e.g. the Radical Red fork, do not re-emit them
// on the first flush), so `new` passes it in the constructor and fork attaches it.
function sender(entry) {
  return (type, data) => {
    if (Array.isArray(data)) data = data.join('\n');
    if (type === 'update') {
      const ch = extractChannelMessages(data, [1, 2]);
      entry.out.p1.push(...ch[1]);
      entry.out.p2.push(...ch[2]);
    } else if (type === 'sideupdate') {
      const i = data.indexOf('\n');
      entry.out[data.slice(0, i)].push(data.slice(i + 1));
    }
  };
}

function attach(entry) {
  entry.battle.send = sender(entry);
}

// arena: field effects fixed for the whole fight (Radical Red hardcore gym rules):
// {weather, terrain, pseudo: [...], p2: [side conditions], ai_permanent}. Set once the battle
// has started (leads in, so the messages reach the mirrors) with no duration, and re-asserted
// after every turn; ai_permanent also freezes whatever weather / terrain p2's own mons set.
function applyArena(e) {
  const a = e.arena, b = e.battle, f = b.field;
  if (!a || !b.started || b.ended) return;
  const src = b.sides[1].active[0];
  const pin = (st) => { if (st) st.duration = 0; };
  if (a.weather) { if (f.weatherState.id !== a.weather) f.setWeather(a.weather, src); pin(f.weatherState); }
  if (a.terrain) { if (f.terrainState.id !== a.terrain) f.setTerrain(a.terrain, src); pin(f.terrainState); }
  for (const id of a.pseudo || []) { if (!f.pseudoWeather[id]) f.addPseudoWeather(id, src); pin(f.pseudoWeather[id]); }
  for (const id of a.p2 || []) { if (!b.sides[1].sideConditions[id]) b.sides[1].addSideCondition(id, src); pin(b.sides[1].sideConditions[id]); }
  if (a.ai_permanent) for (const st of [f.weatherState, f.terrainState]) if (st.source && st.source.side && st.source.side.id === 'p2') pin(st);
}

function winnerId(b) {
  if (!b.ended || !b.winner) return '';
  const side = b.sides.find(s => s.name === b.winner);
  return side ? side.id : '';
}

function flush(entry) {
  applyArena(entry);
  const b = entry.battle;
  b.sendUpdates();
  const out = entry.out;
  entry.out = { p1: [], p2: [] };
  return { p1: out.p1, p2: out.p2, ended: b.ended, winner: winnerId(b), turn: b.turn };
}

function teamState(side) {
  // original team order (what the packed string listed), not switch order
  return side.team.map((set, i) => {
    const p = side.pokemon.find(q => q.set === set) || side.pokemon[i];
    return {
      species: p.species.name, hp: p.hp, maxhp: p.maxhp, fainted: p.fainted, level: p.level,
      stats: p.storedStats, moves: p.moveSlots.map(m => m.id), ability: p.ability, item: p.item,
      status: p.status, pp: p.moveSlots.map(m => m.pp),
    };
  });
}

function handle(msg) {
  const entry = msg.id !== undefined ? battles.get(msg.id) : undefined;
  if (msg.op !== 'new' && !entry) throw new Error(`no battle ${msg.id}`);
  switch (msg.op) {
    case 'new': {
      const e = { battle: null, out: { p1: [], p2: [] } };
      // carry: per p1 mon in team order {hp, status, pp} from a previous battle
      // (a fight that starts without a heal). The battle starts only once both
      // players are set, so p1's state is edited before p2 is added.
      e.battle = new Battle({ formatid: msg.format, seed: msg.seed, send: sender(e), p1: prepPlayer(msg.format, msg.p1) });
      if (msg.carry) {
        const side = e.battle.sides[0];
        msg.carry.forEach((c, i) => {
          const p = side.pokemon.find(q => q.set === side.team[i]) || side.pokemon[i];
          if (!c || !p) return;
          if (c.hp != null) { p.hp = Math.max(0, Math.min(p.maxhp, c.hp)); if (p.hp === 0) p.fainted = true; }
          if (c.status && !p.setStatus(c.status, null, null, true)) {  // before start: set directly
            p.status = c.status;
            p.statusState = { id: c.status, target: p, startTime: 3, time: 3 };
          }
          if (c.pp) c.pp.forEach((pp, j) => { if (p.moveSlots[j]) p.moveSlots[j].pp = pp; });
        });
      }
      e.battle.setPlayer('p2', prepPlayer(msg.format, msg.p2));
      e.arena = msg.arena || null;
      battles.set(msg.id, e);
      return flush(e);
    }
    case 'choose':
      for (const s of ['p1', 'p2']) if (msg[s] != null) entry.battle.choose(s, msg[s]);
      return flush(entry);
    case 'fork': {
      // stringify: toJSON shares array refs (log) with the live battle
      const b = Battle.fromJSON(JSON.stringify(entry.battle.toJSON()));
      const e = { battle: b, out: { p1: [], p2: [] }, arena: entry.arena };
      attach(e);
      if (msg.seed) b.resetRNG(msg.seed);
      else if (msg.reseed !== false) b.resetRNG(null);
      battles.set(msg.new, e);
      return { ok: true };
    }
    case 'forcelose':
      entry.battle.lose(msg.side);
      return flush(entry);
    case 'state':
      return {
        turn: entry.battle.turn, ended: entry.battle.ended, winner: winnerId(entry.battle),
        p1: teamState(entry.battle.sides[0]), p2: teamState(entry.battle.sides[1]),
      };
    case 'close':
      entry.battle.destroy();
      battles.delete(msg.id);
      return { ok: true };
    default:
      throw new Error(`unknown op ${msg.op}`);
  }
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', line => {
  if (!line.trim()) return;
  let reply;
  try {
    reply = handle(JSON.parse(line));
  } catch (err) {
    reply = { error: String(err && err.stack || err) };
  }
  process.stdout.write(JSON.stringify(reply) + '\n');
});
rl.on('close', () => process.exit(0));
