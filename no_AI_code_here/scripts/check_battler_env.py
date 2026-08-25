import asyncio
from unittest import result
from poke_env import AccountConfiguration, RandomPlayer, ServerConfiguration
import uuid

async def mock_check_battler_env(num_servers: int, starting_port: int=8000):
    tasks = [
            _battle_on_server(server_id, starting_port)
            for server_id in range(num_servers)
    ]

    results = await asyncio.gather(*tasks)
    for result in results:
        if result["error"] != None:
            raise Exception(f"Battle on server {result['port']} failed: {result['error']}")

    return True

async def _battle_on_server(server_id: int, starting_port: int=8000):
    server = ServerConfiguration(
        websocket_url=f"ws://localhost:{starting_port + server_id}/showdown/websocket",
        authentication_url=f"http://localhost:{starting_port + server_id}/action.php",
    )

    account_configuration = AccountConfiguration(
    username=f"smoke_{str(uuid.uuid4())[:12]}",
        password=None,
    )
    player1 = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=server,
        account_configuration=account_configuration,
    )

    account_configuration = AccountConfiguration(
        username=f"smoke_{str(uuid.uuid4())[:12]}",
        password=None,
    )
    player2 = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=server,
        account_configuration=account_configuration,
    )

    try: 
        result = await asyncio.wait_for(
            player1.battle_against(player2, n_battles=3),
            timeout=10
        )
    except Exception as e:
        return {
            "port": starting_port + server_id,
            "error": e,
            "result": False,
        }
    print(f"Battle on server {starting_port + server_id} completed")
    return {
        "port": starting_port + server_id,
        "error": None,
        "result": result,
    }

def main():
    asyncio.run(mock_check_battler_env(10))


if __name__ == "__main__":
    main()