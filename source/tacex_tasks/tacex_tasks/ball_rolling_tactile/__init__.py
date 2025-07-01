"""
Ball Rolling Enviornments:
Goal is to roll a ball to a random target position.
"""

import gymnasium as gym
from . import agents



##
# Register Gym environments.
##

#isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-Base-v0 --num_envs 1000 --enable_cameras
# from .base_env import BallRollingEnv, BallRollingEnvCfg # need to import BallRollingEnv here, otherwise class will not be detected for entry point
# gym.register(
#     id="TacEx-Ball-Rolling-Tactile-Base-v0",
#     entry_point=f"{__name__}.base_env:BallRollingEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": BallRollingEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
#         # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
#     },
# )

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-Base-v1 --num_envs 1024 --enable_cameras
from .ball_rolling_base import BallRollingEnv, BallRollingEnvCfg # need to import BallRollingEnv here, otherwise class will not be detected for entry point
gym.register(
    id="TacEx-Ball-Rolling-Tactile-Base-v1",
    entry_point=f"{__name__}.ball_rolling_base:BallRollingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
from .ball_rolling_tactile_rgb import BallRollingTactileRGBEnv, BallRollingTactileRGBCfg 
gym.register(
    id="TacEx-Ball-Rolling-Tactile-RGB-v0",
    entry_point=f"{__name__}.ball_rolling_tactile_rgb:BallRollingTactileRGBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingTactileRGBCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_rgb_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

from .ball_rolling_tactile_rgb_uipc import BallRollingTactileRGBUipcEnv, BallRollingTactileRGBUipcCfg 
gym.register(
    id="TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0",
    entry_point=f"{__name__}.ball_rolling_tactile_rgb_uipc:BallRollingTactileRGBUipcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingTactileRGBUipcCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_rgb_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
