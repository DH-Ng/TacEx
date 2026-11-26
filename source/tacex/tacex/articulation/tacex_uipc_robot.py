"""TacEx robot with UIPC objects (e.g., gelpads) that belong to the robot.

Requires ``tacex_uipc``, which is imported lazily: ``tacex`` works without it as long as this class is not used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tacex_robot import TacExRobot

if TYPE_CHECKING:
    from tacex_uipc import UipcObject

    from .tacex_uipc_robot_cfg import TacExUipcRobotCfg


class TacExUipcRobot(TacExRobot):
    """:class:`TacExRobot` that creates the UIPC objects of :attr:`TacExUipcRobotCfg.uipc_objects` with the robot."""

    cfg: TacExUipcRobotCfg

    def __init__(self, cfg: TacExUipcRobotCfg) -> None:
        """Initialize the robot.

        The objects are added to ``UipcSim.instance()``, so the robot must be created after the ``UipcSim``
        (``UipcRLEnv`` creates it before ``_setup_scene``).
        """
        super().__init__(cfg)
        self._uipc_objects: dict[str, UipcObject] = {}
        if cfg.uipc_objects:
            self._create_uipc_objects()

    @property
    def uipc_objects(self) -> dict[str, UipcObject]:
        """UIPC objects that belong to the robot, by name (see :attr:`TacExUipcRobotCfg.uipc_objects`)."""
        return self._uipc_objects

    def _create_uipc_objects(self):
        from tacex_uipc import UipcDeformableObject, UipcIsaacAttachments, UipcIsaacAttachmentsCfg, UipcRigidObject
        from tacex_uipc.objects import UipcDeformableObjectCfg, UipcRigidObjectCfg

        # Robot prims exist after Articulation.__init__ (spawned), so attachments can be computed right away.
        for name, obj_cfg in self.cfg.uipc_objects.items():
            obj_cfg = obj_cfg.copy()
            if not obj_cfg.prim_path.startswith("/"):
                obj_cfg.prim_path = f"{self.cfg.prim_path}/{obj_cfg.prim_path}"
            constraint_cfg = obj_cfg.constraint_cfg
            if isinstance(constraint_cfg, UipcIsaacAttachmentsCfg):
                if constraint_cfg.isaaclab_rigid_body_prim_path is None:
                    constraint_cfg.isaaclab_rigid_body_prim_path = self.cfg.prim_path

            if isinstance(obj_cfg, UipcDeformableObjectCfg):
                obj_type = UipcDeformableObject
            elif isinstance(obj_cfg, UipcRigidObjectCfg):
                obj_type = UipcRigidObject
            else:
                raise ValueError(f"Unsupported UIPC object cfg for '{name}': {type(obj_cfg)}")
            uipc_obj = obj_type(obj_cfg)
            # the attachment needs the articulation (see UipcIsaacAttachments._initialize_impl)
            if isinstance(uipc_obj.constraint, UipcIsaacAttachments):
                uipc_obj.constraint.isaaclab_rigid_object = self
            self._uipc_objects[name] = uipc_obj
