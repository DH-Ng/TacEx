import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import (
    FactoryTaskGearMeshCfg,
    FactoryTaskNutThreadCfg,
    FactoryTaskPegInsertCfg,
)

from .factory_ik_joint_control_env import FactoryIKJointControlEnv
from .factory_ik_joint_control_env_cfg import (
    FactoryTaskGearMeshIKJointControlCfg,
    FactoryTaskNutThreadIKJointControlCfg,
    FactoryTaskPegInsertIKJointControlCfg,
)


# ---
# Register Gym environments.
# ---

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 128 agent.params.config.full_experiment_name=factory_peg_insert
gym.register(
    id="TacEx-Factory-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Direct-v0 --num_envs 128 agent.params.config.full_experiment_name=factory_gear_insert
gym.register(
    id="TacEx-Factory-GearMesh-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-v0 --num_envs 64 agent.params.config.full_experiment_name=factory_nut_thread
gym.register(
    id="TacEx-Factory-NutThread-Direct-v0",
    entry_point=f"{__name__}.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-PegInsert-Direct-IK-JointControl-v0 --num_envs 128 agent.params.config.full_experiment_name=peg_insert_ik
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


### Gear Insertion
# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearInsert-Direct-IK-JointControl-v0 --num_envs 64 agent.params.config.full_experiment_name=gear_insert_ik
gym.register(
    id="TacEx-Factory-GearInsert-Direct-IK-JointControl-v0",
    entry_point="tacex_tasks.factory:FactoryIKJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskGearMeshIKJointControlCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


### Nut-Thread
# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-IK-JointControl-v0 --num_envs 64 agent.params.config.full_experiment_name=nut_thread_ik
gym.register(
    id="TacEx-Factory-NutThread-Direct-IK-JointControl-v0",
    entry_point="tacex_tasks.factory:FactoryIKJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadIKJointControlCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_tactile_nut_cfg.yaml",
    },
)
