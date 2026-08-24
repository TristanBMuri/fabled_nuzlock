import asyncio
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent  
sys.path.insert(0, str(project_root))

from no_AI_code_here.scripts.check_battler_env import mock_check_battler_env
import project_config
from no_AI_code_here.config.battler.train_config import BattlerTrainConfig
from no_AI_code_here.util.showdown.set_up_battler_env import setup_battler_env


async def check_train_environment(num_servers: int):
    try:
        await mock_check_battler_env(num_servers)
    except Exception as e:
        print(f"Error checking train environment: {e}")
        return False, e

    # check if tensorboard is running
    if b'tensorboard' in subprocess.check_output('pgrep -f tensorboard', shell=True):
        return True
    else:
        return False, "Tensorboard is not running"

async def train_battler(config: BattlerTrainConfig):
    # check if train environment is set up correctly
    env_check_result, env_check_error = await check_train_environment(config.num_servers)

    # if not, try setting up the environment
    if not env_check_result:
        print(f"Train environment is not set up correctly: {env_check_error}")
        try:
            setup_battler_env(config.num_servers)
            env_check_result, env_check_error = await check_train_environment(config.num_servers)
            if not env_check_result:
                print(f"Train environment is not set up correctly: {env_check_error}")
                return False, env_check_error
        except Exception as e:
            print(f"Error setting up train environment: {e}")
            return False, e
    
    # start training

    return True

def evaluate_battler():
    pass

async def main():
    config = BattlerTrainConfig(num_servers=12)
    await train_battler(config)
    evaluate_battler()

if __name__ == "__main__":
    asyncio.run(main())