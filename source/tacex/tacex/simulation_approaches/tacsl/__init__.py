# TacSL needs isaaclab_contrib (and trimesh); keep TacEx importable without it.
try:
    from .tacsl_sim import TacSLSimulator
    from .tacsl_sim_cfg import TacSLForceFieldCfg, TacSLSimulatorCfg

    __all__ = ["TacSLSimulator", "TacSLSimulatorCfg", "TacSLForceFieldCfg"]
except ImportError:
    pass
