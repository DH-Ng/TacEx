from __future__ import annotations

import warp as wp

from isaaclab.utils import configclass

wp.init()


from ..uipc_object import UipcObjectCfg
from .uipc_rigid_object import UipcRigidObject


@configclass
class UipcRigidObjectCfg(UipcObjectCfg):
    class_type: type = UipcRigidObject

    # contact_model:

    @configclass
    class AffineBodyConstitutionCfg:
        # class_type = AffineBodyConstitution # doesn't work, cause no builtin signature found for AffineBodyConstitution class
        m_kappa: float = 100.0
        """Stiffness (hardness) of the object
        in [MPa]

        E.g. 100.0 MPa = hard-rubber-like material
        """

        kinematic: bool = False
        """Makes the DoF of the ABD body fixed.

        """

    constitution_cfg: AffineBodyConstitutionCfg = AffineBodyConstitutionCfg()
