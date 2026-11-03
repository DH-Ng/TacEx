from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from tacex import GelSightSensor
from tacex_uipc import (
    UipcRLEnv,
)
from tacex_uipc.objects import UipcDeformableObject, UipcRigidObject

from .ball_roll_base_env_cfg import BallRollingEnvCfg


class BallRollingUIPCEnv(UipcRLEnv):
    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- for IK ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("TCP")
        # save only the first body index
        self._body_tcp_idx = body_ids[0]
        self._body_name = body_names[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_tcp_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._ee_pos_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # ---

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        # ee orientation should always be (0,1,0,0)
        self.ik_commands[:, 3:] += torch.tensor([0, 1, 0, 0], device=self.device)

        # Define goals for the end effector of the franka arm by adding offsets to the current ball position
        # (for moving ee and ball in a specific pattern)
        ball_radius = self.cfg.ball_radius
        gel_length = self.cfg.gsmini.gelpad_dimensions.length
        gel_width = self.cfg.gsmini.gelpad_dimensions.width
        gel_height = self.cfg.gsmini.gelpad_dimensions.height
        z_offset = ball_radius - gel_height / 2

        above_ball = torch.tensor([0, 0, ball_radius * 2], device=self.device)
        center = torch.tensor([0, 0, z_offset], device=self.device)
        backward = torch.tensor([-gel_length / 2, 0, z_offset], device=self.device)
        forward = torch.tensor([gel_length, 0, z_offset], device=self.device)
        left = torch.tensor([0, gel_width / 2, z_offset], device=self.device)
        right = torch.tensor([0, -gel_width, z_offset], device=self.device)

        self.pattern_offsets = [
            above_ball,  # first, place ee above ball
            center,  # then ee to the center of the ball, so that there is contact
            backward,  # move ee backwards
            forward,
            center,  # back to the center
            left,  # move ee to the left
            right,
            # repeat pattern
            center,
            backward,  # move ee backwards
            forward,
            center,  # back to the center
            left,  # move ee to the left
            right,
            center,
        ]
        # Track the given command
        self.current_goal_idx = 0
        self.num_step_goal_change = 50
        self.step_count = 0

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
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
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.131),  # 0.1034
                    ),
                ),
            ],
        )

        RigidObject(self.cfg.plate)

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- UIPC simulation setup ---

        # gelpad simulated via uipc
        self._uipc_gelpad = UipcDeformableObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.scene.uipc_objects["uipc_gelpad"] = self._uipc_gelpad
        # set rigid object for attachment-constraint
        self._uipc_gelpad.constraint.isaaclab_rigid_object = self.scene.articulations["robot"]

        self.object = UipcRigidObject(self.cfg.ball, self.uipc_sim)

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini, gelpad_obj=self._uipc_gelpad)
        self.scene.sensors["gsmini"] = self.gsmini

    def _pre_physics_step(self, actions: torch.Tensor):
        # update movement pattern according to the ball position
        ball_pos = self.object.data.root_pos_w - self.scene.env_origins
        # change goal
        if self.step_count % self.num_step_goal_change == 0:
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.pattern_offsets)
        self.ik_commands[:, :3] = ball_pos + self.pattern_offsets[self.current_goal_idx]

        # add some randomization for diverse tactile signals
        self.ik_commands[:, :2] += sample_uniform(-0.002, 0.002, (self.num_envs, 2), self.device)
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        # obtain quantities from simulation
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        ee_pose_w = self._robot.data.body_pose_w[:, self._body_tcp_idx]
        root_pose_w = self._robot.data.root_pose_w
        joint_pos = self._robot.data.joint_pos[:, :]

        # compute ee frame in root frame
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # apply ee offset
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._ee_pos_offset, self._ee_rot_offset
        )

        # compute the joint commands
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # # apply rotation for testing purposes
        if self.current_goal_idx == 3 or self.current_goal_idx == 4:
            joint_pos_des[:, 6] = -joint_pos_des[:, 6]  # joint with id 6 = hand

        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        obj_pos = self.object.data.default_root_state[env_ids]
        obj_pos[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(obj_pos, env_ids=env_ids)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # reset actions
        self.actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids)

        # reset goal
        self.current_goal_idx = 0

    # MARK: observations
    def _get_observations(self) -> dict:
        pass
