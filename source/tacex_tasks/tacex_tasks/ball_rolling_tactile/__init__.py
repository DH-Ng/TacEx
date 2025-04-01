"""
Ball Rolling Enviornments:
Goal is to roll a ball to a random target position.
"""

import gymnasium as gym
from . import agents



##
# Register Gym environments.
##

# ball_rolling_task_entry = "tacex_tasks.ball_rolling"

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
# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-No-Privileged-v0 --num_envs 1000 --enable_cameras
from .base_env_no_privileged import BallRollingEnvNoPrivileged, BallRollingEnvNoPrivilegedCfg 
gym.register(
    id="TacEx-Ball-Rolling-Tactile-No-Privileged-v0",
    entry_point=f"{__name__}.base_env_no_privileged:BallRollingEnvNoPrivileged",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvNoPrivilegedCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-Base-v1 --num_envs 1000 --enable_cameras
from .base_env_new import BallRollingEnv, BallRollingEnvCfg # need to import BallRollingEnv here, otherwise class will not be detected for entry point
gym.register(
    id="TacEx-Ball-Rolling-Tactile-Base-v1",
    entry_point=f"{__name__}.base_env_new:BallRollingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

#isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-Depth_Map-v0 --num_envs 1000 --enable_cameras
# from .depth_map_env import BallRollingDepthMapEnv, BallRollingDepthMapEnvCfg 
# gym.register(
#     id="TacEx-Ball-Rolling-Tactile-Depth_Map-v0",
#     entry_point=f"{__name__}.depth_map_env:BallRollingDepthMapEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": BallRollingDepthMapEnvCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
#         "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
#     },
# )
