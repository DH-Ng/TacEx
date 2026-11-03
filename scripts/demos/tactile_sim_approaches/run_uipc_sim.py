from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Ball rolling experiment with a Franka, which is equipped with one GelSight Mini Sensor."
)
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="If system utilization should be tracked.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="If tactile images should shown in Isaac.",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime
import json
import time
import traceback
from pathlib import Path

import carb
import numpy as np
import psutil
import pynvml
import torch

from envs.ball_roll_uipc_env import BallRollingUIPCEnv
from envs.ball_roll_uipc_env_cfg import BallRollingUIPCEnvCfg
from envs.sensor_configs import TAXIM_FEM_CFG
from envs.utils import run_simulator


def main():
    """Main function."""
    # Define simulation env
    env_cfg = BallRollingUIPCEnvCfg()
    env_cfg.gsmini = TAXIM_FEM_CFG

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = 1  # args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    env = BallRollingUIPCEnv(env_cfg)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(env=env, simulation_app=simulation_app)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim apply
        simulation_app.close()
