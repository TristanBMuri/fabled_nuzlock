import subprocess

import project_config

_SCRIPTS = project_config.PROJECT_ROOT / "scripts"


def setup_battler_env(num_servers: int, starting_port: int = 8020):
    subprocess.run(
        [str(_SCRIPTS / "start_showdown.sh"), str(num_servers), str(starting_port)],
        check=True,
    )

    # check if tensorboard is running
    res = subprocess.run("tensorboard --logdir logs", shell=True, capture_output=True)
    if res.returncode != 0:
        print("Tensorboard is not running")
        # start tensorboard
        try:
            subprocess.run("uv run tensorboard --logdir logs --port 6006", shell=True, check=True)
        except Exception as e:
            print(f"Error starting tensorboard: {e}")
            return False
    return True


def teardown_env(num_servers: int, starting_port: int = 8020):
    subprocess.run(
        [str(_SCRIPTS / "stop_showdown.sh"), str(num_servers), str(starting_port)],
        check=True,
    )
