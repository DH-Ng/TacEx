"""
Ball Rolling Enviornments:
Goal is to roll a ball to a random target position.
"""

import gymnasium as gym
from . import agents



##
# Register Gym environments.
##

# ball_rolling_task_entry = "tacex_tasks.ball_rolling_priviledged"
from .base_env import BallRollingEnv, BallRollingEnvCfg # need to import BallRollingEnv here, otherwise class will not be detected for entry point
gym.register(
    id="TacEx-Ball-Rolling-Priviledged-v0",
    entry_point=f"{__name__}.base_env:BallRollingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg, #! we are using this one
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

from .without_reaching import BallRollingWithoutReachingEnv, BallRollingEnvWithoutReachingCfg # need to import BallRollingEnv here, otherwise class will not be detected for entry point
gym.register(
    id="TacEx-Ball-Rolling-Priviledged-without-Reach_v0",
    entry_point=f"{__name__}.without_reaching:BallRollingWithoutReachingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvWithoutReachingCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg, #! we are using this one
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
