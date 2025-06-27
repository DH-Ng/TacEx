from dataclasses import MISSING

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

from tacex_uipc import UipcSim, UipcSimCfg, UipcObject, UipcObjectCfg

@configclass
class UipcEnvCfg(DirectRLEnvCfg):
    """Configuration for an RL environment defined with the direct workflow.

    Please refer to the :class:`isaaclab.envs.direct_rl_env.DirectRLEnv` class for more details.
    """

    # UIPC simulation settings

    uipc_sim: UipcSimCfg = UipcSimCfg()