# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import (
    FactoryTaskGearMeshCfg,
    FactoryTaskNutThreadCfg,
    FactoryTaskPegInsertCfg,
)

from .factory_env_modified_obs import FactoryModifiedObsEnv
from .factory_env_modified_obs_cfg import (
    FactoryTaskPegInsertModifiedObsCfg,
    FactoryTaskNutThreadModifiedObsCfg,
    FactoryTaskGearMeshModifiedObsCfg,
)

from .factory_ik_joint_control_env import FactoryIKJointControlEnv
from .factory_ik_joint_control_env_cfg import (
    FactoryTaskGearMeshIKJointControlCfg,
    FactoryTaskNutThreadIKJointControlCfg,
    FactoryTaskPegInsertIKJointControlCfg,
)

from .factory_tactile_env import FactoryTactileEnv
from .factory_tactile_env_cfg import (
    FactoryTaskGearMeshTactileCfg,
    FactoryTaskNutThreadTactileCfg,
    FactoryTaskPegInsertTactileCfg,
    FactoryTaskPegInsertTactilePlayCfg
)

# ---
# Register Gym environments.
# ---

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 100 --enable_cameras
gym.register(
    id="TacEx-Factory-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Direct-v0 --num_envs 100 --enable_cameras
gym.register(
    id="TacEx-Factory-GearMesh-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-v0 --num_envs 20 --enable_cameras
gym.register(
    id="TacEx-Factory-NutThread-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Factory-PegInsert-Direct-ModifiedObs-v0",
#     entry_point="isaaclab_tasks.direct.factory:FactoryModifiedObsEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FactoryTaskPegInsertModifiedObsCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )


# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-IK-JointControl-v0 --num_envs 128 --enable_cameras
gym.register(
    id="TacEx-Factory-PegInsert-Direct-IK-JointControl-v0",
    # entry_point=f"{__name__}.factory_ik_joint_control_env:FactoryIKJointControlEnv",
    entry_point="tacex_tasks.factory:FactoryIKJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertIKJointControlCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-Tactile-v0 --num_envs 128 --enable_cameras
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-v0",
    entry_point="tacex_tasks.factory:FactoryTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactileCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-Direct-Tactile-Play-v0 --num_envs 128 --enable_cameras
gym.register(
    id="TacEx-Factory-PegInsert-Direct-Tactile-Play-v0",
    entry_point="tacex_tasks.factory:FactoryTactileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertTactilePlayCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_cfg.yaml",
    },
)