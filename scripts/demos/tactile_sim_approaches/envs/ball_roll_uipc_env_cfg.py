from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_taxim import GELSIGHT_MINI_TAXIM_CFG
from tacex_uipc import (
    UipcIsaacAttachmentsCfg,
    UipcSimCfg,
)
from tacex_uipc.objects import UipcDeformableObjectCfg, UipcRigidObjectCfg
from tacex_uipc.utils import TetMeshCfg


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BallRollingUIPCEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    # viewer.origin_type = "env"
    # viewer.env_idx = 50

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60.0,
        render_interval=decimation,
        # device="cpu",
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        ground_height=0.005,
        contact=UipcSimCfg.Contact(d_hat=0.0001, default_friction_ratio=0.95),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=20,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,  # only update sensors when they are accessed
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),  # pos=[0, 0, -1.05]
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    plate = AssetBaseCfg(
        prim_path="/World/envs/env_.*/plate",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd"),
    )

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1 / 5,
        # epsilon_r=0.01
    )
    ball = UipcRigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.015]),  # rot=(0.72,-0.3,0.42,-0.45)
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
        ),
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(),
        debug_vis=True,
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(
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
        edge_length_r=1 / 20,
        # epsilon_r=0.01
    )
    gelpad_cfg = UipcDeformableObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
        constraint_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=100.0,
            body_name="gelsight_mini_case",
            debug_vis=debug_vis,
            compute_attachment_data=True,
            isaaclab_rigid_body_prim_path="/World/envs/env_.*/Robot",
        ),
    )

    gsmini = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    ball_radius = 0.005

    obj_pos_randomization_range = [-0.15, 0.15]

    # some filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0
