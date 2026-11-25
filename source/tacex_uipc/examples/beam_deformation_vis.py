"""Visualization of the elastic strain energy and the elastic forces of UIPC soft bodies.

A Franka with two GelSight Mini grippers squeezes a soft beam (attached to a plate), presses it down, stretches it
by lifting and releases it again. The beam (and optionally the gelpads) are colored by the chosen quantity
(blue = low, green, red = high), optionally with the elastic nodal forces drawn as arrows.

Usage:
    python source/tacex_uipc/examples/beam_deformation_vis.py --quantity energy --log_scale
    python source/tacex_uipc/examples/beam_deformation_vis.py --quantity force --force_arrows
    python source/tacex_uipc/examples/beam_deformation_vis.py --quantity energy --force_arrows --vis_gelpads
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize elastic energy/forces of a UIPC beam squeezed by a Franka.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--quantity",
    type=str,
    default="energy",
    choices=["energy", "force", "displacement"],
    help="Quantity used for coloring: elastic energy density, elastic nodal force or displacement.",
)
parser.add_argument("--log_scale", action="store_true", help="Map the quantity logarithmically onto the color ramp.")
parser.add_argument(
    "--value_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Fixed value range for the color ramp (unit of the quantity). Default: (0, max) of each frame.",
)
parser.add_argument("--force_arrows", action="store_true", help="Draw the elastic nodal forces as arrows.")
parser.add_argument("--vis_gelpads", action="store_true", help="Visualize the deformation of the gelpads as well.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many steps (default: run forever).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # loads omni.ui, which is needed by tacex (also in headless mode)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import traceback

import carb
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_uipc import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG

from tacex_uipc import UipcIsaacAttachmentsCfg, UipcRLEnv, UipcSimCfg
from tacex_uipc.objects import UipcDeformableObject, UipcDeformableObjectCfg
from tacex_uipc.utils import TetMeshCfg, VoxelTetMeshCfg

GOAL_POS_UP = (0.5, 0.0, 0.25)
GOAL_ROT_DOWN = (0.0, 0.0, -1.0, 0.0)

# scripted motion: step -> (goal z position, finger position, description)
SCHEDULE = {
    50: (0.15, 0.04, "Moving ee down."),
    70: (0.15, 0.0, "Squeezing beam."),
    130: (0.14, 0.0, "Pressing beam down."),
    190: (0.18, 0.0, "Stretching beam."),
    290: (0.18, 0.04, "Releasing beam."),
    340: (0.25, 0.04, "Moving ee up."),
}
RESET_STEP = 400


def deformation_vis_cfg() -> UipcDeformableObjectCfg.DeformationVisCfg:
    return UipcDeformableObjectCfg.DeformationVisCfg(
        quantity=args_cli.quantity,
        log_scale=args_cli.log_scale,
        value_range=tuple(args_cli.value_range) if args_cli.value_range is not None else None,
        draw_force_arrows=args_cli.force_arrows,
    )


@configclass
class BeamDeformationVisEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.8, 0.35, 0.3), lookat=(0.5, 0.0, 0.12))

    decimation = 1
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.0025)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            scale=(0.05, 0.05, 10),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    # filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0

    uipc_sim = UipcSimCfg(
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001, default_friction_ratio=2.5, default_contact_resistance=25.0),
    )

    gel_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 15)
    gelpad_left_cfg = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gsmini_gelpad_left",
        mesh_cfg=gel_mesh_cfg,
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=1),
        constraint_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=100.0,
            body_name="gelsight_mini_case_left",
            compute_attachment_data=True,
            isaaclab_rigid_body_prim_path="/World/envs/env_.*/Robot",
        ),
    )
    gelpad_right_cfg = gelpad_left_cfg.replace(
        prim_path="/World/envs/env_.*/Robot/gsmini_gelpad_right",
        constraint_cfg=gelpad_left_cfg.constraint_cfg.replace(body_name="gelsight_mini_case_right"),
    )

    beam = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/beam",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.005]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/beam.usd"),
        # voxel tet mesh: fast and robust meshing. The original USD mesh is rendered (tetrahedral render mapping),
        # the deformation vis values are interpolated from the tet vertices onto it.
        mesh_cfg=VoxelTetMeshCfg(resolution=10),
        usd_mesh_prim_name="beam",
        # E = 1 MPa: a softer 20 cm beam buckles under its own weight (critical height ~5 cm for 5 kPa)
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=1.0),
        constraint_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=100.0,
            compute_attachment_data=True,
            isaaclab_rigid_body_prim_path="/World/envs/env_.*/ground_plate",
        ),
        # -- deformation visualization
        debug_deformation_vis=True,
        deformation_vis_cfg=UipcDeformableObjectCfg.DeformationVisCfg(),  # set from the CLI in main()
    )


class BeamDeformationVis(UipcRLEnv):
    cfg: BeamDeformationVisEnvCfg

    def __init__(self, cfg: BeamDeformationVisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, _ = self._robot.find_bodies("panda_hand")
        self._body_tcp_idx = body_ids[0]
        self._finger_joint_ids, _ = self._robot.find_joints(["panda_finger.*"])
        # fixed base robot: root body is not included in the jacobians
        self._jacobi_body_idx = self._body_tcp_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._ee_pos_offset = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)

        self.step_count = 0
        self.finger_pos = 0.04
        self.goal_prim_view = None

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

        plate = RigidObject(self.cfg.plate)

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array(GOAL_POS_UP),
            orientation=np.array(GOAL_ROT_DOWN),
            color=np.array([255.0, 0.0, 0.0]),
        )

        # --- UIPC objects ---
        self.gsmini_gelpad_left = UipcDeformableObject(self.cfg.gelpad_left_cfg)
        self.gsmini_gelpad_right = UipcDeformableObject(self.cfg.gelpad_right_cfg)
        self.gsmini_gelpad_left.constraint.isaaclab_rigid_object = self.scene.articulations["robot"]
        self.gsmini_gelpad_right.constraint.isaaclab_rigid_object = self.scene.articulations["robot"]

        self.beam = UipcDeformableObject(self.cfg.beam)
        self.beam.constraint.isaaclab_rigid_object = plate

    def _pre_physics_step(self, actions: torch.Tensor | None):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        ee_pose_w = self._robot.data.body_pose_w[:, self._body_tcp_idx]
        root_pose_w = self._robot.data.root_pose_w
        joint_pos = self._robot.data.joint_pos[:, :]

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._ee_pos_offset, self._ee_rot_offset
        )
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des[:, self._finger_joint_ids] = self.finger_pos
        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    def _get_dones(self):
        pass

    def _get_rewards(self):
        pass

    def _get_observations(self):
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

        for obj in (self.gsmini_gelpad_left, self.gsmini_gelpad_right, self.beam):
            obj.write_nodal_pos_to_sim(obj.data.default_nodal_state_w[:, :, :3])

        self.step_count = 0
        self.finger_pos = 0.04
        if self.goal_prim_view is not None:
            self.goal_prim_view.set_world_poses(
                positions=torch.tensor(GOAL_POS_UP, device=self.device).repeat(self.num_envs, 1),
                orientations=torch.tensor(GOAL_ROT_DOWN, device=self.device).repeat(self.num_envs, 1),
            )


def print_stats(env: BeamDeformationVis):
    """Prints total elastic energy and max. elastic nodal force of the visualized objects."""
    msg = [f"[step {env.step_count:4d}]"]
    for name in ("beam", "gsmini_gelpad_left", "gsmini_gelpad_right"):
        obj: UipcDeformableObject = getattr(env, name)
        if not obj.cfg.debug_deformation_vis:
            continue
        state = obj.elastic_state if obj.elastic_state is not None else obj.compute_elastic_state()
        msg.append(
            f"{name}: E_el = {state.total_energy * 1e3:.4f} mJ, max |f_el| = {state.forces.norm(dim=-1).max():.4f} N"
        )
    print(" | ".join(msg))


def run_simulator(env: BeamDeformationVis):
    print(f"Starting simulation with {env.num_envs} envs")
    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

    total_steps = 0
    while simulation_app.is_running():
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.uipc_sim.update_render_meshes()
        env.scene.update(dt=env.physics_dt)
        env.sim.render()
        total_steps += 1

        if env.step_count in SCHEDULE:
            z, env.finger_pos, description = SCHEDULE[env.step_count]
            print(description)
            env.goal_prim_view.set_world_poses(
                positions=torch.tensor((GOAL_POS_UP[0], GOAL_POS_UP[1], z), device=env.device).repeat(env.num_envs, 1)
            )
        if env.step_count % 20 == 0:
            print_stats(env)
        if env.step_count == RESET_STEP:
            print("Reset\n")
            env.reset()

        if args_cli.max_steps is not None and total_steps >= args_cli.max_steps:
            break

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations
    env.close()


def main():
    env_cfg = BeamDeformationVisEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # deformation vis settings from the CLI
    env_cfg.beam.deformation_vis_cfg = deformation_vis_cfg()
    if args_cli.vis_gelpads:
        for gelpad_cfg in (env_cfg.gelpad_left_cfg, env_cfg.gelpad_right_cfg):
            gelpad_cfg.debug_deformation_vis = True
            gelpad_cfg.deformation_vis_cfg = deformation_vis_cfg()

    env = BeamDeformationVis(env_cfg)
    print("[INFO]: Setup complete...")
    run_simulator(env)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
