"""Ball rolling demo with a UIPC gelpad attached to the sensor case through a UIPC rigid "carrier" body.

Same as ``run_uipc_sim.py``, but the gel is stitched to a UIPC affine body with the shape of the sensor case, which
the robot moves (``tacex_uipc/objects/constraints/uipc_carrier_attachment.py``), instead of a soft position
constraint on the gel vertices. The gel's underside stays on the sensor under load.

Run with the Isaac Lab python (from this directory):
    python run_uipc_carrier_sim.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Ball rolling experiment with a Franka and one GelSight Mini; UIPC gelpad attached via a rigid carrier body."
    )
)
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="If tactile images and the UIPC carrier body should be shown in Isaac.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback

import carb

from envs.ball_roll_uipc_carrier_env import BallRollingUIPCCarrierEnv, BallRollingUIPCCarrierEnvCfg
from envs.sensor_configs import TAXIM_FEM_CFG
from envs.utils import run_simulator


def main():
    """Main function."""
    env_cfg = BallRollingUIPCCarrierEnvCfg()
    env_cfg.gsmini = TAXIM_FEM_CFG

    # the UIPC attachments (soft position constraint and carrier) currently support a single env
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis
    env_cfg.carrier_cfg.debug_vis = args_cli.debug_vis  # render the UIPC carrier body

    env = BallRollingUIPCCarrierEnv(env_cfg)

    print("[INFO]: Setup complete...")
    run_simulator(env=env, simulation_app=simulation_app)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
