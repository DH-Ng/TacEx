from .envs import UipcInteractiveScene, UipcRLEnv
from .objects import (
    UipcObject,
    UipcObjectCfg,
    UipcConstraint,
    UipcConstraintCfg,
    UipcDeformableObject,
    UipcDeformableObjectCfg,
    UipcDeformableObjectData,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcRigidObject,
    UipcRigidObjectCfg,
    UipcRigidObjectData,
)
from .sim import UipcSim, UipcSimCfg

# Register UI extensions (GUI only: headless apps lack isaacsim.gui / debug_draw, which the extension needs).
try:
    from .ui_extension import *  # noqa: F403
except ImportError:
    pass
