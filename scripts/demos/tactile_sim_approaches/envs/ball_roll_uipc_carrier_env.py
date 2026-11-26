"""UIPC ball rolling env whose gelpad is attached to the sensor case through a UIPC rigid "carrier" body.

Same as :class:`BallRollingUIPCEnv`, but instead of the soft position constraint on the gel vertices
(``UipcIsaacAttachments``), the gel is stitched to a UIPC affine body with the shape of the sensor case, which the
robot moves (``UipcIsaacCarrierAttachments``, see ``tacex_uipc/objects/constraints/uipc_carrier_attachment.py``).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor, TacExUipcRobot
from tacex_uipc.objects import UipcRigidObject
from tacex_uipc.objects.constraints.uipc_carrier_attachment import (
    UipcIsaacCarrierAttachments,
    UipcIsaacCarrierAttachmentsCfg,
)

from .ball_roll_uipc_env import BallRollingUIPCEnv
from .ball_roll_uipc_env_cfg import BallRollingUIPCEnvCfg


@configclass
class BallRollingUIPCCarrierEnvCfg(BallRollingUIPCEnvCfg):
    carrier_cfg: UipcIsaacCarrierAttachmentsCfg = UipcIsaacCarrierAttachmentsCfg(
        isaaclab_rigid_body_prim_path="/World/envs/env_.*/Robot",
        body_name="gelsight_mini_case",
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # the gel is attached by the carrier, not by a constraint on its vertices
        gelpad_cfg = self.robot.uipc_objects["gelpad"].replace(constraint_cfg=None)
        self.robot = self.robot.replace(uipc_objects={**self.robot.uipc_objects, "gelpad": gelpad_cfg})


class BallRollingUIPCCarrierEnv(BallRollingUIPCEnv):
    cfg: BallRollingUIPCCarrierEnvCfg

    def _setup_scene(self):
        # same as BallRollingUIPCEnv._setup_scene, except for the gelpad attachment
        self._robot = TacExUipcRobot(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.131)),
                ),
            ],
        )

        RigidObject(self.cfg.plate)

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---
        # the gelpad (cfg.robot.uipc_objects) is created by the robot, without a constraint on its vertices
        self._uipc_gelpad = self._robot.uipc_objects["gelpad"]
        self.object = UipcRigidObject(self.cfg.ball)
        # carrier last: it registers its surface vertices after all other UIPC objects
        self._carrier = UipcIsaacCarrierAttachments(
            self.cfg.carrier_cfg, self._uipc_gelpad, isaaclab_rigid_object=self._robot
        )

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini
