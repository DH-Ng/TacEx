import gymnasium as gym

from .. import agents
from .factory_tactile_marker_env import FactoryTactileMarkerEnv
from .factory_tactile_marker_env_cfg import FactoryTaskPegInsertTactileMarkerCfg

# from .factory_tactile_single_sensor_env import FactoryTactileSingleEnv
# from .factory_tactile_single_sensor_env_cfg import (
#     FactoryTaskGearMeshTactileSingleCfg,
#     FactoryTaskNutThreadTactileSingleCfg,
#     FactoryTaskPegInsertTactileSingleCfg,
#     FactoryTaskPegInsertTactileSinglePlayCfg,
# )

# ---
# Register Gym environments.
# ---

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-Tactile-Marker-v0 --num_envs 128 --enable_cameras agent.params.config.full_experiment_name=peg_insert_tactile_marker
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-Marker-v0",
    entry_point=f"{__name__}.factory_tactile_marker_env:FactoryTactileMarkerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactileMarkerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-Direct-Tactile-Play-v0 --num_envs 128 --enable_cameras --checkpoint
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-Marker-Play-v0",
    entry_point=f"{__name__}.factory_tactile_marker_env:FactoryTactileMarkerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactileMarkerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

# # --- Single Sensor envs
# # isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-Tactile-Single-v0 --num_envs 128 --enable_cameras agent.params.config.full_experiment_name=peg_insert_tactile_single_sensor
# gym.register(
#     id="TacEx-Factory-PegInsert-Direct-Tactile-Single-v0",
#     entry_point=f"{__name__}.factory_tactile_single_sensor_env:FactoryTactileSingleEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskPegInsertTactileSingleCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
#     },
# )

# gym.register(
#     id="TacEx-Factory-PegInsert-Direct-Tactile-Single-Play-v0",
#     entry_point=f"{__name__}.factory_tactile_single_sensor_env:FactoryTactileSingleEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskPegInsertTactileSinglePlayCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
#     },
# )
