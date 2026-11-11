"""Runs the random-action check for one task in a fresh Isaac Sim process (used by the env test files).

Isaac Sim does not release all GPU memory (PhysX/RTX) when an env is closed (~0.65 GiB per env on a 16 GiB GPU),
so running all envs in one process ends in GPU out-of-memory errors. Each task therefore runs in its own process.

Results are written to a JSON file instead of relying on the exit code, because Kit may crash at shutdown.

Usage:
    python run_env_check.py --list [--factory] --out tasks.json
    python run_env_check.py --task <id> --num_envs 32 1 --out result.json
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--list", action="store_true", help="Write the list of task IDs to test to --out.")
parser.add_argument("--factory", action="store_true", help="With --list: list factory envs instead of the others.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, nargs="+", default=[32, 1])
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

# import pinocchio before the app, to use the dependencies installed by Isaac Lab (needed by the Pink IK controller)
if sys.platform != "win32":
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pass

from isaaclab.app import AppLauncher

headless = os.environ.get("TACEX_TEST_HEADLESS", "1") != "0"
simulation_app = AppLauncher(headless=headless, enable_cameras=True).app

"""Rest everything follows."""

import tacex_tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_test_utils import _check_random_actions, setup_environment  # noqa: E402


def _write(data):
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)


if args.list:
    _write(setup_environment(include_play=False, factory_envs=args.factory, multi_agent=False))
else:
    results = {}
    for num_envs in args.num_envs:
        try:
            _check_random_actions(args.task, args.device, num_envs, num_steps=args.num_steps)
            results[str(num_envs)] = {"passed": True, "message": ""}
        except BaseException as e:  # pytest.fail raises a BaseException subclass
            results[str(num_envs)] = {"passed": False, "message": f"{type(e).__name__}: {e}"}
        # write after every case, so results survive a later crash
        _write(results)

simulation_app.close()
