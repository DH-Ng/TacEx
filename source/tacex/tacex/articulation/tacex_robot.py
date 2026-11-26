"""Articulation that wraps the default IsaacLab Articulation class and extends it for TacEx.

- Adds IK movement of the robot (omits need of external DiffIK logic in environments).
- Adds robot reset with natural movement to reset pose, instead of "teleporting".
    - useful for Factory env's, UIPC env's or Real Env's
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController

if TYPE_CHECKING:
    from .tacex_robot_cfg import TacExRobotCfg


class TacExRobot(Articulation):
    """Articulation with built-in differential IK.

    Cartesian targets (in the robot root frame) are set with :meth:`set_ee_pose_target`. Envs with a Cartesian
    target get their joint targets computed by IK inside :meth:`write_data_to_sim`, i.e., on every physics step.
    """

    cfg: TacExRobotCfg

    def __init__(self, cfg: TacExRobotCfg) -> None:
        super().__init__(cfg)

    # MARK: initialization

    def _initialize_impl(self):
        super()._initialize_impl()
        self._init_ik_controller()

    def _init_ik_controller(self):
        cfg = self.cfg
        self._ik_controller = DifferentialIKController(
            cfg=cfg.ik_controller_cfg, num_envs=self.num_instances, device=self.device
        )

        body_ids, _ = self.find_bodies(cfg.ik_body_name)
        self._ik_body_idx = body_ids[0]
        self._ik_joint_ids, _ = self.find_joints(cfg.ik_joint_names)
        self._ik_joint_set = set(self._ik_joint_ids)

        # For a fixed base robot, the jacobian has no row for the root body and no columns for the 6 base DoFs.
        if self.is_fixed_base:
            self._jacobi_body_idx = self._ik_body_idx - 1
            self._jacobi_joint_ids = list(self._ik_joint_ids)
        else:
            self._jacobi_body_idx = self._ik_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._ik_joint_ids]

        self._ee_pos_offset = torch.tensor(cfg.ee_offset_pos, device=self.device).repeat(self.num_instances, 1)
        self._ee_rot_offset = torch.tensor(cfg.ee_offset_rot, device=self.device).repeat(self.num_instances, 1)

        self._ik_commands = torch.zeros((self.num_instances, self._ik_controller.action_dim), device=self.device)
        self._ik_commands[:, 3] = 1.0  # valid default quaternion
        self._cartesian_mask = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
        """False: env uses joint targets, True: env uses cartesian target (IK)."""

    # MARK: public API

    @property
    def ik_joint_ids(self) -> list[int]:
        """Ids of the joints that are controlled by IK."""
        return self._ik_joint_ids

    def _body_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose of the IK body (without offset) in the robot root frame."""
        body_pose_w = self.data.body_pose_w[:, self._ik_body_idx]
        root_pose_w = self.data.root_pose_w
        return math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], body_pose_w[:, 0:3], body_pose_w[:, 3:7]
        )

    @property
    def ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose (pos, quat) of the controlled frame (incl. offset) in the robot root frame."""
        body_pos_b, body_quat_b = self._body_pose_b()
        return math_utils.combine_frame_transforms(body_pos_b, body_quat_b, self._ee_pos_offset, self._ee_rot_offset)

    def set_ee_pose_target(
        self, pos: torch.Tensor, quat: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Set a cartesian target pose (root frame) and switch the corresponding envs to cartesian control.

        Args:
            pos: Target positions, shape (len(env_ids), 3).
            quat: Target orientations (w, x, y, z), shape (len(env_ids), 4).
            env_ids: Envs to apply the target to. Defaults to all.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._ik_commands[env_ids, :3] = pos
        self._ik_commands[env_ids, 3:] = quat
        self._ik_controller.set_command(self._ik_commands)
        self._cartesian_mask[env_ids] = True

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: Sequence[int] | slice | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set joint position targets into internal buffers.

        Same as the Isaac Lab method, but if the targets include IK joints, the given envs switch back to joint
        control (IK is no longer applied to them until the next :meth:`set_ee_pose_target`).
        Targets of other joints only (e.g., gripper fingers) don't affect IK.

        Note:
            ``env_ids`` should be a tensor (or None), like for the Isaac Lab method.
        """
        super().set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
        if self._includes_ik_joints(joint_ids):
            self._cartesian_mask[slice(None) if env_ids is None else env_ids] = False

    def _includes_ik_joints(self, joint_ids: Sequence[int] | slice | None) -> bool:
        if joint_ids is None or isinstance(joint_ids, slice):
            return True
        if isinstance(joint_ids, torch.Tensor):
            joint_ids = joint_ids.tolist()
        return not self._ik_joint_set.isdisjoint(joint_ids)

    def compute_ik_joint_targets(self) -> torch.Tensor:
        """Joint position targets of the IK joints for the current cartesian command, shape (N, len(ik_joints))."""
        jacobian = self.root_physx_view.get_jacobians()[:, self._jacobi_body_idx][:, :, self._jacobi_joint_ids]
        # rotate jacobian from world into base frame
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(self.data.root_quat_w))
        jacobian = jacobian.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        # account for the offset of the controlled frame: v_ee = v_body + w x r (r = offset in root frame)
        body_pos_b, body_quat_b = self._body_pose_b()
        offset_b = math_utils.quat_apply(body_quat_b, self._ee_pos_offset)
        jacobian[:, 0:3, :] -= torch.bmm(math_utils.skew_symmetric_matrix(offset_b), jacobian[:, 3:, :])

        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            body_pos_b, body_quat_b, self._ee_pos_offset, self._ee_rot_offset
        )
        joint_pos = self.data.joint_pos[:, self._ik_joint_ids]
        return self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    def write_data_to_sim(self):
        if self._cartesian_mask.any():
            joint_pos_des = self.compute_ik_joint_targets()
            ids = torch.nonzero(self._cartesian_mask).squeeze(-1)
            super().set_joint_position_target(joint_pos_des[ids], joint_ids=self._ik_joint_ids, env_ids=ids)
        super().write_data_to_sim()

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        self._ik_controller.reset(env_ids)
        self._cartesian_mask[slice(None) if env_ids is None else env_ids] = False

    def move_robot_till_target_reached(
        self,
        sim_context,
        dt: float,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        render: bool = False,
        time_budget: float = 0.25,
        tolerance: float = 1e-3,
    ) -> bool:
        """Step the simulation until the IK joints are close to their targets (or the time budget is used up).

        Useful for resets with natural movement (e.g., UIPC, real envs) instead of teleporting.
        Note: only this robot's data is updated, other scene entities (such as rigid objects or cameras) are not.

        Returns:
            True if the target was reached within the budget.
        """
        if env_ids is None:
            env_ids = slice(None)
        elapsed = 0.0
        while elapsed < time_budget:
            self.write_data_to_sim()
            sim_context.step(render)
            self.update(dt)
            elapsed += dt
            err = (
                self.data.joint_pos_target[env_ids][:, self._ik_joint_ids]
                - self.data.joint_pos[env_ids][:, self._ik_joint_ids]
            ).abs()
            if err.max() < tolerance:
                return True
        return False
