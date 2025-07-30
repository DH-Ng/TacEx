import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_adapter_uipc_textured import (
    FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG,
)
from tacex_assets.sensors.gelsight_mini.gelsight_mini_cfg import GelSightMiniCfg
from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)
from tacex_uipc.utils import TetMeshCfg

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg
from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg

try:
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except:
    draw = None

from .ball_rolling_physx_rigid import PhysXRigidEnv, PhysXRigidEnvCfg


@configclass
class UipcTexturedEnvCfg(PhysXRigidEnvCfg):

    debug_vis = True

    decimation = 1
    # simulationff
    sim: SimulationCfg = SimulationCfg(
        dt=1/60, #0.01, #1 / 120, #0.001
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True, # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        # render=RenderCfg(
        #     rendering_mode="performance"
        # )
    )

    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        ground_height=0.001,
        contact=UipcSimCfg.Contact(
            d_hat=0.0005
        )
    )

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1/5,
        # epsilon_r=0.01
    )
    ball = UipcObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.01]), #rot=(0.72,-0.3,0.42,-0.45)
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
        ),
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg() # UipcObjectCfg.StableNeoHookeanCfg() #
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_SINGLE_ADAPTER_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
            },
        ),
    )
    # simulate the gelpad as uipc mesh
    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1/5,
        # epsilon_r=0.01
    )
    gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        # mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=1000.0,
        body_name="gelsight_mini_case",
        debug_vis=False,
        compute_attachment_data=True
    )

    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix = "/Camera",
            update_period= 0,
            resolution = (480,640), #(120, 160),
            data_types = ["depth", "rgb"],
            clipping_range = (0.02, 0.034), #(0.024, 0.034),
        ),
        device = "cuda",
        debug_vis=True, # for rendering sensor output in the gui
        # marker_motion_sim_cfg=None,
        data_types=["tactile_rgb", "marker_motion", "camera_depth", "camera_rgb"], #marker_motion
    )

    # settings for optical sim - update Taxim cfg
    gsmini.optical_sim_cfg = gsmini.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(240,320),
    )
    # update FOTS cfg
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    gsmini.marker_motion_sim_cfg = gsmini.marker_motion_sim_cfg.replace(
        device="cuda",
        tactile_img_res=(240,320),
        frame_transformer_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case", #"/World/envs/env_.*/Robot/gelsight_mini_case",
            # you have to make sure that the asset frame center is correct, otherwise wrong shear/twist motions
            source_frame_offset=OffsetCfg(
                rot=(0.0, 0.92388, -0.38268, 0.0) # values for the robot used here
            ),
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/ball")
            ],
            debug_vis=False,
            visualizer_cfg=marker_cfg,
        )
    )

class UipcTexturedEnv(PhysXRigidEnv):
    cfg: UipcTexturedEnvCfg

    def __init__(self, cfg: UipcTexturedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self._setup_base_scene()
        self.scene.clone_environments(copy_from_source=False)

        # sensors
        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        ### UIPC simulation setup

        # gelpad simulated via uipc
        self._uipc_gelpad: UipcObject = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)

        self.object: UipcObject = UipcObject(self.cfg.ball, self.uipc_sim)
        self.scene.uipc_objects["object"] = self.object

        # create attachment
        self.attachment = UipcIsaacAttachments(self.cfg.gelpad_attachment_cfg, self._uipc_gelpad, self.scene.articulations["robot"])

    def _pre_physics_step(self, actions: torch.Tensor):
        # update movement pattern according to the ball position
        ball_pos = self.object.data.root_pos_w - self.scene.env_origins

        draw.clear_points()
        points = ball_pos.cpu().numpy()

        draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])

        # change goal
        if (self.step_count+1) % self.num_step_goal_change == 0:
            self.current_goal_idx = (self.current_goal_idx+1)  % len(self.pattern_offsets)
        self.ik_commands[:, :3] = ball_pos + self.pattern_offsets[self.current_goal_idx]

        self._ik_controller.set_command(self.ik_commands)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.object.write_vertex_positions_to_sim(vertex_positions=self.object.init_vertex_pos)
        self._uipc_gelpad.write_vertex_positions_to_sim(vertex_positions=self._uipc_gelpad.init_vertex_pos)

        self.step_count = 0

        # reset goal
        self.current_goal_idx = 0
