"""Smoke test: two random bots battle on the local Showdown server.

Usage: uv run python scripts/smoke_test.py [--host localhost --port 8020]

Proves the full stack: poke-env connects, battles run, and we can read
back the results. This is the pattern the RL env will reuse.
"""

import argparse
import asyncio

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer

BATTLE_FORMAT = "gen8randombattle"


async def main(host: str, port: int) -> None:
    server = ServerConfiguration(
        websocket_url=f"ws://{host}:{port}/showdown/websocket",
        authentication_url=f"http://{host}:{port}/action.php",
    )

    player_1 = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        account_configuration=AccountConfiguration("SmokeTestP1", None),
        server_configuration=server,
    )
    player_2 = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        server_configuration=server,
    )

    print(f"Starting battles on {host}:{port} ({BATTLE_FORMAT})...")
    await player_1.battle_against(player_2, n_battles=3)

    print("\n=== Results ===")
    for p in (player_1, player_2):
        print(f"{p.username:>15}: {p.n_won_battles}/{p.n_finished_battles} wins")

    # Show what battle data is available — this is what the RL env will observe.
    battle = player_1.battles[next(iter(player_1.battles))]
    print("\n=== Sample battle data ===")
    print(f"turn: {battle.turn}, won: {battle.won}, lost: {battle.lost}")
    print(f"active mon: {battle.active_pokemon}")
    print(f"team: {[str(m) for m in battle.team.values()]}")
    print(f"opponent team: {[str(m) for m in battle.opponent_team.values()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port))
