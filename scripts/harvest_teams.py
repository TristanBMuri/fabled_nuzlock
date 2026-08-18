"""Harvest random teams from the local Showdown's own generator.

Writes showdown-export team files to data/teams/pool/team_NN.txt (1-indexed by
--start). Deterministic per (seed-prefix, index) — same scheme as the old
project's validation manifest.

Usage: uv run python scripts/harvest_teams.py --count 15 --start 1
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SHOWDOWN_DIR = Path(__file__).resolve().parents[1] / "pokemon-showdown"
POOL_DIR = Path(__file__).resolve().parents[1] / "data" / "teams" / "pool"

NODE_SCRIPT = """
const path = require('path');
const crypto = require('crypto');
const {Teams} = require(path.join(process.argv[1], 'dist', 'sim'));
const battleFormat = 'gen8randombattle';
const [start, count, seedPrefix] = process.argv.slice(2).map((v, i) => i === 2 ? v : Number(v));
function seedArray(text) {
  const d = crypto.createHash('sha256').update(text).digest();
  return [d.readUInt32BE(0), d.readUInt32BE(4), d.readUInt32BE(8), d.readUInt32BE(12)];
}
const out = [];
for (let i = 0; i < count; i++) {
  const seed = seedArray(`${seedPrefix}-${String(start + i).padStart(2, '0')}`);
  out.push(Teams.export(Teams.generate(battleFormat, {seed})));
}
console.log(JSON.stringify(out));
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--seed-prefix", default="nuzlock-pool-v1")
    args = parser.parse_args()

    raw = subprocess.run(
        ["node", "-e", NODE_SCRIPT, str(SHOWDOWN_DIR), str(args.start),
         str(args.count), args.seed_prefix],
        capture_output=True, text=True, check=True,
    ).stdout
    teams = json.loads(raw)
    assert len(teams) == args.count and all(5 <= t.count("\n\n") <= 6 for t in teams)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for i, team in enumerate(teams):
        path = POOL_DIR / f"team_{args.start + i:02d}.txt"
        path.write_text(team)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
