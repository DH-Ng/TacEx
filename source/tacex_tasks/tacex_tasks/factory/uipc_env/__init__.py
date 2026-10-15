import gymnasium as gym

from .. import agents
from .factory_uipc_env import FactoryUipcEnv
from .factory_uipc_env_cfg import (
    FactoryUipcTaskGearMeshCfg,
    FactoryUipcTaskNutThreadCfg,
    FactoryUipcTaskPegInsertCfg,
)

# ---
# Register Gym environments.
# ---

# isaaclab -p ./scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-Uipc-PegInsert-Direct-v0 --num_envs 1 agent.params.config.full_experiment_name=factory_uipc_peg_insert
gym.register(
    id="TacEx-Factory-Uipc-PegInsert-Direct-v0",
    entry_point=f"{__name__}.factory_uipc_env:FactoryUipcEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryUipcTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_uipc_cfg.yaml",
    },
)
