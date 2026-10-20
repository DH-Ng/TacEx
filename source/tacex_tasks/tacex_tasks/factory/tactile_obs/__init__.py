import gymnasium as gym

from .. import agents
from .factory_tactile_env import FactoryTactileEnv
from .factory_tactile_env_cfg import (
    FactoryTaskGearMeshTactileCfg,
    FactoryTaskNutThreadTactileCfg,
    FactoryTaskPegInsertTactileCfg,
    FactoryTaskPegInsertTactilePlayCfg,
)

from .factory_tactile_single_sensor_env import FactoryTactileSingleEnv
from .factory_tactile_single_sensor_env_cfg import (
    FactoryTaskGearMeshTactileSingleCfg,
    FactoryTaskNutThreadTactileSingleCfg,
    FactoryTaskPegInsertTactileSingleCfg,
    FactoryTaskPegInsertTactileSinglePlayCfg,
)

# ---
# Register Gym environments.
# ---

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-Tactile-v0 --num_envs 128 --enable_cameras agent.params.config.full_experiment_name=peg_insert_tactile
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-v0",
    entry_point=f"{__name__}.factory_tactile_env:FactoryTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactileCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-Direct-Tactile-Play-v0 --num_envs 128 --enable_cameras --checkpoint
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-Play-v0",
    entry_point=f"{__name__}.factory_tactile_env:FactoryTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactilePlayCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearInsert-Direct-Tactile-v0 --num_envs 128 --enable_cameras agent.params.config.full_experiment_name=gear_insert_tactile
gym.register(
    id="TacEx-Factory-GearInsert-Direct-Tactile-v0",
    entry_point=f"{__name__}.factory_tactile_env:FactoryTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshTactileCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-Tactile-v0 --num_envs 64 --enable_cameras agent.params.config.full_experiment_name=nut_thread_tactile
gym.register(
    id="TacEx-Factory-NutThread-Direct-Tactile-v0",
    entry_point=f"{__name__}.factory_tactile_env:FactoryTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadTactileCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_nut_cfg.yaml",
    },
)

# --- Single Sensor envs
# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-Tactile-Single-v0 --num_envs 128 --enable_cameras agent.params.config.full_experiment_name=peg_insert_tactile_single_sensor
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-Single-v0",
    entry_point=f"{__name__}.factory_tactile_single_sensor_env:FactoryTactileSingleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactileSingleCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-Single-Play-v0",
    entry_point=f"{__name__}.factory_tactile_single_sensor_env:FactoryTactileSingleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactileSinglePlayCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)
