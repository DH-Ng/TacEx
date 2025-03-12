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

#isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-v0 --num_envs 1000 --enable_cameras
from .base_env import BallRollingEnv, BallRollingEnvCfg # need to import BallRollingEnv here, otherwise class will not be detected for entry point
gym.register(
    id="TacEx-Ball-Rolling-Tactile-v0",
    entry_point=f"{__name__}.base_env:BallRollingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

#isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Height_Map-v0 --num_envs 1000 --enable_cameras
from .height_map_env import BallRollingHeightMapEnv, BallRollingHeightMapEnvCfg 
gym.register(
    id="TacEx-Ball-Rolling-Height_Map-v0",
    entry_point=f"{__name__}.height_map_env:BallRollingHeightMapEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingHeightMapEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

#isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Height_Map-with-IK_v0 --num_envs 1000 --enable_cameras
from .reset_with_IK_solver import BallRollingIKResetEnv, BallRollingIKResetEnvCfg 
gym.register(
    id="TacEx-Ball-Rolling-Height_Map-with-IK_v0",
    entry_point=f"{__name__}.reset_with_IK_solver:BallRollingIKResetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingIKResetEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)