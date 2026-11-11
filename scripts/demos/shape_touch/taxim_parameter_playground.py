"""Collecting tactile data of the shapes from https://danfergo.github.io/gelsight-simulation/.

This script can be used to play around with the parameters of the Taxim simulation.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with one GelSight Mini Sensor, by moving the Frame in the GUI"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--path", type=str, help="Path to data folder.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="If tactile images should be rendered inside the GUI.",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
from contextlib import suppress
from pathlib import Path
import json

import omni.ui

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import numpy as np
import torch
import cv2

import carb
import dataclasses
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex.simulation_approaches.gpu_taxim import TaximSimulator, TaximSimulatorCfg
from tacex.simulation_approaches.gpu_taxim.sim.taxim_impl import SimulatorParameters

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG

from tacex_tasks.utils import DirectLiveVisualizer

# ui stuff
from isaacsim.gui.components.ui_utils import *
import omni.ui as ui


try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

INDENTER_FILE_PATHS = sorted(list(Path(f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/").glob("*.usd")))


def float_builder(
    label="", type="floatfield", default_val=0, tooltip="", min=-inf, max=inf, step=0.1, format="%.2f", label_width=160
):
    """Creates a Stylized Floatfield Widget

    Args:
        label (str, optional): Label to the left of the UI element. Defaults to "".
        type (str, optional): Type of UI element. Defaults to "floatfield".
        default_val (int, optional): Default Value of UI element. Defaults to 0.
        tooltip (str, optional): Tooltip to display over the UI elements. Defaults to "".

    Returns:
        AbstractValueModel: model
    """
    with ui.HStack():
        ui.Label(label, width=label_width, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip))
        float_field = ui.FloatDrag(
            name="FloatField",
            width=ui.Fraction(1),
            height=0,
            alignment=ui.Alignment.LEFT_CENTER,
            min=min,
            max=max,
            step=step,
            format=format,
        ).model
        float_field.set_value(default_val)
        add_line_rect_flourish(False)
        return float_field


class TaximParameterWindow:
    """GUI Window inside IsaacSim for playing around with Taxim Simulation parameters."""

    def __init__(self, window_name: str, sensor: GelSightSensor, delegate=None, **kwargs):
        self.ui_window = ui.Window(
            window_name, width=500, height=800, visible=True, dock_preference=ui.DockPreference.LEFT_BOTTOM
        )

        self.taxim_params = None
        self.new_params = False

        self.sensor = sensor

        self.num_calib_point: int | None = None
        self.current_calib_point_idx = 11

        with self.ui_window.frame:
            self.build_fn()

    def _build_initial_frame_processing_params_frame(self):
        """Build the frame for the parameters which control the initial frame processing"""
        with ui.CollapsableFrame(
            title="Initial Frame processing",
            height=0,
            collapsed=False,
            style=get_style(),
            name="groupFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            with ui.VStack(height=0, spacing=5):
                ui.Spacer(height=6)
                with ui.VStack(height=0, spacing=5):
                    # with ui.HStack(height=2, spacing=5):
                    #     ui.Label("x", alignment=ui.Alignment.CENTER)
                    #     ui.Label("y", alignment=ui.Alignment.CENTER)
                    with ui.HStack(height=2, spacing=0):
                        self.initial_frame_sigma_widget = [
                            float_builder(
                                default_val=0.078125,
                                min=0.0001,
                                max=0.015,
                                step=0.0001,
                                format="%.4f",
                                label="initial_frame_sigma_rel ",
                            ),
                            float_builder(
                                default_val=0.10416666666666667, min=0.0001, max=0.015, step=0.0001, format="%.4f"
                            ),
                        ]

                self.diff_threshold_widget = float_builder(
                    min=0.0001, max=20, label="diff_threshold", default_val=5, step=0.01
                )
                self.frame_mixing_percentage_widget = float_builder(
                    min=0.0001, max=1, label="frame_mixing_percentage", default_val=0.15, step=0.001
                )

    def _build_deformation_params_frame(self):
        """Build the frame for the parameters which control the deformation approximation"""
        with ui.CollapsableFrame(
            title="Deformation Approximation",
            height=0,
            collapsed=False,
            style=get_style(),
            name="groupFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            with ui.VStack(height=0, spacing=5):
                ui.Spacer(height=6)

                self.contact_scale_widget = float_builder(
                    default_val=0.4,
                    min=0.0001,
                    max=1,
                    step=0.001,
                    format="%.4f",
                    label="contact_scale",
                    tooltip="Contact Scale for approximated gel deformation.",
                )

                with ui.VStack(height=0, spacing=5):
                    # ui.Label("deform_final_sigma_rel")
                    # with ui.HStack(height=2, spacing=0):
                    #     ui.Label("", alignment=ui.Alignment.CENTER)
                    #     ui.Label("x", alignment=ui.Alignment.CENTER)
                    #     ui.Label("", alignment=ui.Alignment.CENTER)
                    #     ui.Label("y", alignment=ui.Alignment.CENTER)
                    with ui.HStack(height=2, spacing=0):
                        self.deform_final_sigma_rel = [
                            float_builder(
                                default_val=0.003125,
                                min=0.0001,
                                max=0.01,
                                step=0.00001,
                                format="%.5f",
                                label="deform_final_sigma_rel",
                            ),
                            float_builder(
                                default_val=0.004166666666666667, min=0.0001, max=0.01, step=0.00001, format="%.5f"
                            ),
                        ]

                self.deform_pyramid_sigma_rel_widgets = []
                with ui.CollapsableFrame(
                    title="deform_pyramid_sigma_rel",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    name="subFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                ):
                    with ui.VStack(height=0, spacing=5):
                        # with ui.HStack(height=2, spacing=5):
                        #     ui.Label("x", alignment=ui.Alignment.CENTER)
                        #     ui.Label("y", alignment=ui.Alignment.CENTER)

                        with ui.HStack(height=2, spacing=0):
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.04765625,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label="Kernel 0",
                                    label_width=60,
                                )
                            )
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.06354166666666666,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label_width=60,
                                )
                            )

                        with ui.HStack(height=2, spacing=0):
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.02421875,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label="Kernel 1",
                                    label_width=60,
                                )
                            )
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.03229166666666667,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label_width=60,
                                )
                            )

                        with ui.HStack(height=0, spacing=0):
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.012499999999999999,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label="Kernel 2",
                                    label_width=60,
                                )
                            )
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.016666666666666666,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label_width=60,
                                )
                            )

                        with ui.HStack(height=0, spacing=0):
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.00546875,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label="Kernel 3",
                                    label_width=60,
                                )
                            )
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.007291666666666667,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label_width=60,
                                )
                            )

                        with ui.HStack(height=0, spacing=0):
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.003125,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label="Kernel 4",
                                    label_width=60,
                                )
                            )
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.004166666666666667,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label_width=60,
                                )
                            )

                        with ui.HStack(height=0, spacing=0):
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.0017187500000000002,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label="Kernel 5",
                                    label_width=60,
                                )
                            )
                            self.deform_pyramid_sigma_rel_widgets.append(
                                float_builder(
                                    default_val=0.0022916666666666667,
                                    min=0.0001,
                                    max=0.1,
                                    step=0.0001,
                                    format="%.5f",
                                    label_width=60,
                                )
                            )

    def _build_shadow_sim_params_frame(self):
        """Build the frame for the parameters which control the simulation of the shadows in Taxim"""
        with ui.CollapsableFrame(
            title="Shadow Simulation",
            height=0,
            collapsed=False,
            style=get_style(),
            name="groupFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            with ui.VStack(height=0, spacing=5):
                ui.Spacer(height=6)

                with ui.VStack(height=0, spacing=5):
                    # ui.Label("shadow_step_rel")
                    # with ui.HStack(height=2, spacing=5):
                    #     ui.Label("x", alignment=ui.Alignment.CENTER)
                    #     ui.Label("y", alignment=ui.Alignment.CENTER)
                    with ui.HStack(height=2, spacing=0):
                        self.shadow_step_widgets = [
                            float_builder(
                                default_val=0.001953125,
                                min=0.0001,
                                max=0.01,
                                step=0.0001,
                                format="%.4f",
                                label="shadow_step_rel",
                            ),
                            float_builder(
                                default_val=0.0026041666666666665, min=0.0001, max=0.01, step=0.0001, format="%.4f"
                            ),
                        ]

                with ui.VStack(height=0, spacing=5):
                    # ui.Label("shadow_blur_sigma_rel")
                    # with ui.HStack(height=2, spacing=5):
                    #     ui.Label("x", alignment=ui.Alignment.CENTER)
                    #     ui.Label("y", alignment=ui.Alignment.CENTER)
                    with ui.HStack(height=2, spacing=0):
                        self.shadow_blur_sigma_rel = [
                            float_builder(
                                default_val=0.0017187500000000002,
                                min=0.0001,
                                max=0.01,
                                step=0.0001,
                                format="%.4f",
                                label="shadow_blur_sigma_rel",
                            ),
                            float_builder(
                                default_val=0.0022916666666666667, min=0.0001, max=0.01, step=0.0001, format="%.4f"
                            ),
                        ]

                with ui.VStack(height=0, spacing=5):
                    ui.Label("shadow_attachment_kernel_size_rel")
                    with ui.HStack(height=0, spacing=0):
                        self.shadow_attachment_kernel_size_rel = [
                            float_builder(
                                default_val=0.0078125,
                                min=0.0001,
                                max=0.015,
                                step=0.0001,
                                format="%.4f",
                            ),
                            float_builder(default_val=0.010416667, min=0.0001, max=0.015, step=0.0001, format="%.4f"),
                        ]

                self.discritize_precision_widget = float_builder(
                    min=0.0001, max=1, label="discretize_precision", default_val=0.1, format="%.4f", step=0.001
                )
                self.height_precision_widget = float_builder(
                    min=0.0001, max=1, label="height_precision", default_val=0.1, format="%.4f", step=0.001
                )
                self.fan_angle_widget = float_builder(
                    min=0.0001, max=1, label="fan_angle", default_val=0.1, format="%.4f", step=0.001
                )

                # only used during init
                self.fan_precision_widget = float_builder(
                    min=0.0001, max=1, label="fan_precision", default_val=0.05, format="%.4f", step=0.001
                )

    def build_fn(self):
        """
        The method that is called to build all of the UI once the window is visible.
        """
        with ui.VStack(height=0):
            ui.Spacer(height=3)
            self.calib_point_label = ui.Label(f"Current calib_point_idx: {self.current_calib_point_idx}")
            with ui.HStack():
                ui.Button(
                    "Prev Calib point",
                    name="tool_button",
                    clicked_fn=lambda: self.prev_calib_point(),
                )
                ui.Button(
                    "Next Calib point",
                    name="tool_button",
                    clicked_fn=lambda: self.next_calib_point(),
                )

            self.indentation_depth_widget = float_builder(
                default_val=0.001,
                min=0.000,
                max=0.002,
                step=0.00001,
                format="%.5f",
                label="indentation_depth",
            )

            ui.Spacer(height=6)
            self._build_initial_frame_processing_params_frame()
            ui.Button(
                "Create new Taxim Sim",
                name="tool_button",
                clicked_fn=lambda: self.create_new_taxim_sim(self.sensor),
            )
            ui.Spacer(height=3)
            self._build_deformation_params_frame()
            ui.Spacer(height=3)
            self._build_shadow_sim_params_frame()
            ui.Spacer(height=12)

    def collect_taxim_parameters(self):
        """Method gets called when 'Update Taxim Parameters' button is pressed.

        The returned dictionary contains the current values for the Taxim simulation parameters
        from the GUI.
        """

        # initial frame params
        initial_frame_sigma_rel = (
            self.initial_frame_sigma_widget[0].as_float,
            self.initial_frame_sigma_widget[1].as_float,
        )
        diff_threshold = self.diff_threshold_widget.as_float
        frame_mixing_percentage = self.frame_mixing_percentage_widget.as_float

        # deformation params
        contact_scale = self.contact_scale_widget.as_float
        deform_final_sigma_rel = (self.deform_final_sigma_rel[0].as_float, self.deform_final_sigma_rel[1].as_float)

        deform_pyramid_sigma_rel_0 = []
        deform_pyramid_sigma_rel_1 = []
        for m in range(0, len(self.deform_pyramid_sigma_rel_widgets) - 1, 2):
            deform_pyramid_sigma_rel_0.append((self.deform_pyramid_sigma_rel_widgets[m].as_float))
            deform_pyramid_sigma_rel_1.append(self.deform_pyramid_sigma_rel_widgets[m + 1].as_float)

        deform_pyramid_sigma_rel = (tuple(deform_pyramid_sigma_rel_0), tuple(deform_pyramid_sigma_rel_1))

        # shadow sim params
        shadow_step_rel = (self.shadow_step_widgets[0].as_float, self.shadow_step_widgets[1].as_float)
        shadow_blur_sigma_rel = (self.shadow_blur_sigma_rel[0].as_float, self.shadow_blur_sigma_rel[1].as_float)
        shadow_attachment_kernel_size_rel = (
            self.shadow_attachment_kernel_size_rel[0].as_float,
            self.shadow_attachment_kernel_size_rel[1].as_float,
        )
        discretize_precision = self.discritize_precision_widget.as_float
        fan_angle = self.fan_angle_widget.as_float
        fan_precision = self.fan_precision_widget.as_float
        height_precision = self.height_precision_widget.as_float

        self.taxim_params = {
            "initial_frame_sigma_rel": initial_frame_sigma_rel,
            "diff_threshold": diff_threshold,
            "frame_mixing_percentage": frame_mixing_percentage,
            "contact_scale": contact_scale,
            "deform_final_sigma_rel": deform_final_sigma_rel,
            "deform_pyramid_sigma_rel": deform_pyramid_sigma_rel,
            "shadow_step_rel": shadow_step_rel,
            "shadow_blur_sigma_rel": shadow_blur_sigma_rel,
            "shadow_attachment_kernel_size_rel": shadow_attachment_kernel_size_rel,
            "discretize_precision": discretize_precision,
            "fan_angle": fan_angle,
            "fan_precision": fan_precision,
            "height_precision": height_precision,
        }

        # update Taxim parameters that do not require new Taxim Sim
        taxim = self.sensor.optical_simulator._taxim
        taxim.sim_params.contact_scale = contact_scale
        taxim.sim_params.deform_final_sigma_rel = deform_final_sigma_rel
        taxim.sim_params.deform_pyramid_sigma_rel = deform_pyramid_sigma_rel
        taxim.sim_params.shadow_step_rel = shadow_step_rel
        taxim.sim_params.shadow_blur_sigma_rel = shadow_blur_sigma_rel
        taxim.sim_params.shadow_attachment_kernel_size_rel = shadow_attachment_kernel_size_rel
        taxim.sim_params.discretize_precision = discretize_precision
        taxim.sim_params.fan_angle = fan_angle
        taxim.sim_params.fan_precision = fan_precision
        taxim.sim_params.height_precision = height_precision

    def create_new_taxim_sim(self, sensor: GelSightSensor):
        """Uses the latest given TaximParameters and creates a new TaximSimulation with these parameters.

        Args:
            sensor: The sensor whose Taxim simulation should be replaced.
        """
        if not isinstance(sensor.optical_simulator, TaximSimulator):
            print("Simulation Approach is not GPU-Taxim! Cannot update parameters.")
            return
        taxim = sensor.optical_simulator._taxim

        self.collect_taxim_parameters()

        params = self.taxim_params

        # create new Taxim sim instance with new parameters
        cfg: TaximSimulatorCfg = sensor.cfg.optical_sim_cfg.copy()
        cfg.taxim_parameters = {"simulator": params, "sensor": dataclasses.asdict(taxim.sensor_params)}
        # use new Taxim sim in our sensor and remove previous Taxim instance
        del sensor.optical_simulator
        sensor.optical_simulator = TaximSimulator(sensor, cfg)
        sensor.optical_simulator._set_debug_vis_impl(sensor.cfg.debug_vis)
        sensor.optical_simulator._initialize_impl()
        sensor.reset()

        print("Create new Taxim Sim: ")
        print(
            f"initial_frame_sigma_rel : {taxim.sim_params.initial_frame_sigma_rel}  \n"
            f"frame_mixing_percentage : {taxim.sim_params.frame_mixing_percentage} \n"
            f"diff_threshold : {taxim.sim_params.diff_threshold}  \n"
            f"contact_scale : {taxim.sim_params.contact_scale}  \n"
            f"deform_pyramid_sigma_rel : {taxim.sim_params.deform_pyramid_sigma_rel}  \n"
            f"shadow_blur_sigma_rel : {taxim.sim_params.shadow_blur_sigma_rel}  \n"
            f"deform_final_sigma_rel : {taxim.sim_params.deform_final_sigma_rel}  \n"
            f"shadow_step_rel : {taxim.sim_params.shadow_step_rel}  \n"
            f"height_precision : {taxim.sim_params.height_precision}  \n"
            f"discretize_precision : {taxim.sim_params.discretize_precision}  \n"
            f"fan_angle : {taxim.sim_params.fan_angle}  \n"
            f"fan_precision : {taxim.sim_params.fan_precision}  \n"
            f"shadow_attachment_kernel_size_rel : {taxim.sim_params.shadow_attachment_kernel_size_rel}  \n"
        )

    def prev_calib_point(self):
        self.current_calib_point_idx -= 1
        if self.current_calib_point_idx < 0:
            self.current_calib_point_idx = self.num_calib_point - 1
        self.calib_point_label.text = f"Current calib_point_idx: {self.current_calib_point_idx}"

    def next_calib_point(self):
        self.current_calib_point_idx += 1
        if self.current_calib_point_idx == self.num_calib_point:
            self.current_calib_point_idx = 0
        self.calib_point_label.text = f"Current calib_point_idx: {self.current_calib_point_idx}"


def create_indenters_cfg(base_center_x, base_center_y, base_center_z) -> RigidObjectCfg:
    """Creates RigidObjectCfg's for each usd file in the `{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/` directory.

    The objects are spawned on a line along the y axis.
    Returns:
        shapes: dict of names and corresponding RigidObjectCfg's
    """

    usd_files_path = sorted(list(Path(f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/").glob("*.usd")))
    usd_files_path = [str(path) for path in usd_files_path]
    pos = [base_center_x, base_center_y, base_center_z]

    spawn_cfgs = sim_utils.MultiUsdFileCfg(
        usd_path=usd_files_path,
        random_choice=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            kinematic_enabled=True,
            disable_gravity=False,
        ),
    )

    indenters_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/indenter",
        spawn=spawn_cfgs,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )

    return indenters_cfg


def create_calib_points_in_gelpad_frame(
    cylinder_radius_mm,
    calib_area_height: float = 0.02075,
    calib_area_width: float = 0.02525,
    num_points_x=3,
    num_points_y=4,
    border_fraction=0.15,
):
    """Creates (x,y) positions for the data collection.

    The robot has to be manually calibrated so that the calib indenter appears at these positions
    during data collection.
    "Calibration" here means, that we move the robot per hand to the desired positions and save the
    corresponding joint states. Since the Franka has high repeatabiltiy, we can use these exact joint
    states for each indenter of ours and easiliy replicate the same indentation locations in Sim.

    Regarding the gelpad frame: x axis of the gelpad frame shows up and the y axis shows to the left.
    Args:
        gelpad_height: _description_. Defaults to 0.02075. Units [m]
        gelpad_width: _description_. Defaults to 0.02525. Units [m]

    Returns:
        calib_points: (x,y) positions in the gelad frame of the sensor (Units: [m]). Shape (num_calib_points, 2).
                    Amount of calib points = num_points_x * num_points_y + (num_points_x - 1)*(num_points_y - 1)
    """
    calib_area_height = calib_area_height * (1 - border_fraction)
    calib_area_width = calib_area_width * (1 - border_fraction)

    # grid pattern
    cylinder_radius = cylinder_radius_mm / 1000
    x_coor = np.linspace(
        -calib_area_height / 2 + cylinder_radius,
        calib_area_height / 2 - cylinder_radius,
        num_points_x,
    )
    y_coor = np.linspace(
        -calib_area_width / 2 + cylinder_radius,
        calib_area_width / 2 - cylinder_radius,
        num_points_y,
    )
    xy_coor = np.array(np.meshgrid(x_coor, y_coor)).T.reshape(-1, 2)  # x direction is up/down, and y dir is left/right

    # additional pattern inbetween
    num_points_x_inbetween = num_points_x - 1
    num_points_y_inbetween = num_points_y - 1
    x_coor_inbetween = np.linspace(
        (x_coor[0] - x_coor[1]) / 2 + x_coor[1],
        (x_coor[-1] - x_coor[-2]) / 2 + x_coor[-2],
        num_points_x_inbetween,
    )
    y_coor_inbetween = np.linspace(
        (y_coor[0] - y_coor[1]) / 2 + y_coor[1],
        (y_coor[-1] - y_coor[-2]) / 2 + y_coor[-2],
        num_points_y_inbetween,
    )

    xy_coor_inbetween = np.array(np.meshgrid(x_coor_inbetween, y_coor_inbetween)).T.reshape(-1, 2)

    # combine patterns
    calib_points = np.vstack((xy_coor, xy_coor_inbetween))

    # sort calib points for easier calibration

    # first sort according to x values
    calib_points = calib_points[calib_points[:, 0].argsort()]
    # extract subarrays where x components are equal
    _, indices = np.unique(calib_points[:, 0], return_index=True)
    subarrays = []
    for i in range(0, indices.shape[0] - 1):
        subarray = calib_points[indices[i] : indices[i + 1]]
        # sort subarray column-wise (= according to y components)
        subarray = subarray[subarray[:, 1].argsort()]
        subarrays.append(subarray)
    # last subarray
    subarray = calib_points[indices[-1] :]
    subarray = subarray[subarray[:, 1].argsort()]
    subarrays.append(subarray)
    calib_points = np.vstack(subarrays)

    # for i in range(calib_points.shape[0]):
    #     plt.plot(
    #         calib_points[i, 0],
    #         calib_points[i, 1],
    #         marker="o",
    #         color="b",
    #         linestyle="none",
    #     )
    # plt.show()
    return calib_points


def create_calib_points_in_image_frame(calib_points: np.array, imgw=320, imgh=240, pixmm=0.0634):
    """Transforms calib points defined in the gelpad frame to the image frame and filters out points that are not inside the image.

    Args:
        calib_points (_type_): Units are [m].
        imgw (_type_, optional): _description_. Defaults to 320.
        imgh (_type_, optional): _description_. Defaults to 240.
        pixmm (_type_, optional): _description_. Defaults to 0.0634.
    """
    # convert to mm
    calib_points = calib_points * 1000
    calib_points_pix = calib_points.copy()

    # calib_points[:, 1] = y-coor. in gelpad frame (left/right direction), this corresponds to x coor. in img-frame (i.e. in left/right direction of img)
    calib_points_pix[:, 0] = calib_points[:, 1] / pixmm + imgw / 2
    calib_points_pix[:, 1] = calib_points[:, 0] / pixmm + imgh / 2
    calib_points_pix = calib_points_pix.astype(int)

    # import matplotlib.pyplot as plt
    # for i in range(calib_points_pix.shape[0]):
    #     plt.plot(
    #         calib_points_pix[i, 0],
    #         calib_points_pix[i, 1],
    #         marker="o",
    #         color="red",
    #         linestyle="none",
    #     )
    # plt.show()

    # filter out points that are outside the image
    calib_points_pix_filtered = calib_points_pix[(calib_points_pix >= 0).all(axis=1)]
    calib_points_pix_filtered = calib_points_pix_filtered[calib_points_pix_filtered[:, 0] <= imgw]
    calib_points_pix_filtered = calib_points_pix_filtered[calib_points_pix_filtered[:, 1] <= imgh]

    # first sort according to y values
    calib_points_pix_filtered = calib_points_pix_filtered[calib_points_pix_filtered[:, 1].argsort()]
    # extract subarrays where y components are equal
    _, indices = np.unique(calib_points_pix_filtered[:, 1], return_index=True)
    subarrays = []
    for i in range(0, indices.shape[0] - 1):
        subarray = calib_points_pix_filtered[indices[i] : indices[i + 1]]
        # sort subarray according to x components)
        subarray = subarray[subarray[:, 0].argsort()]
        subarrays.append(subarray)
    # last subarray
    subarray = calib_points_pix_filtered[indices[-1] :]
    subarray = subarray[subarray[:, 0].argsort()]
    subarrays.append(subarray)
    calib_points_pix_filtered = np.vstack(subarrays)

    # import matplotlib.pyplot as plt
    # for i in range(calib_points_pix_filtered.shape[0]):
    #     plt.plot(
    #         calib_points_pix_filtered[i, 0],
    #         calib_points_pix_filtered[i, 1],
    #         marker="o",
    #         color="red",
    #         linestyle="none",
    #     )
    # plt.show()

    # filter out corresponding points in gelpad frame
    calib_points_filtered = calib_points[(calib_points_pix >= 0).all(axis=1)]
    calib_points_filtered = calib_points_filtered[calib_points_filtered[:, 0] <= imgw]
    calib_points_filtered = calib_points_filtered[calib_points_filtered[:, 1] <= imgh]

    calib_points_filtered /= 1000
    return calib_points_filtered, calib_points_pix_filtered


@configclass
class ShapeTouchEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (2.45, -1.2, 0.01869)
    viewer.lookat = (-4.8, 6.0, -0.2)

    debug_vis = True

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        render=RenderCfg(enable_translucency=True),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.75,
        replicate_physics=False,
        lazy_sensor_update=True,  # only update sensors when they are accessed
    )

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # spawn indenter_holder for cool visuals
    indenter_holder_plate = AssetBaseCfg(
        prim_path="/World/envs/env_.*/indenter_holder",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.5,
                0,
                0.02,
            ),
            rot=(0.7071068, 0, 0, 0.7071068),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/indenter_holder_plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
                disable_gravity=False,
            ),
        ),
    )

    indenters: RigidObjectCfg = create_indenters_cfg(
        base_center_x=0.5,
        base_center_y=0,
        base_center_z=0.02,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    gsmini = GELSIGHT_MINI_TAXIM_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GELSIGHT_MINI_TAXIM_CFG.SensorCameraCfg(
            prim_name="Camera",
            update_period=0,
            resolution=(320, 240),
            data_types=["depth"],
            clipping_range_m=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for being able to see sensor output in the gui
        # update FOTS cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],
    )
    # change settings for optical sim
    gsmini.optical_sim_cfg = gsmini.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(320, 240),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ee_pos_offset = [0.0, 0.0, 0.0]
    ee_rot_offset = [0.0, 1.0, 0.0, 0.0]

    base_pose = [
        0.5,
        0,
        0.02,
        1,
        0,
        0,
        0,
    ]

    # some filler values, needed for DirectRLEnv class
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class ShapeTouchEnv(DirectRLEnv):
    cfg: ShapeTouchEnvCfg

    def __init__(self, cfg: ShapeTouchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- for IK ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, _ = self._robot.find_bodies("TCP")
        # save only the first body index
        self._body_tcp_idx = body_ids[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_tcp_idx - 1

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.ik_commands[:, 3:] = torch.tensor([1, 0, 0, 0], device=self.device)

        # ee offset w.r.t TCP -> TCP is defined so that z-axis shows down. In our case here we want z to show upwards
        self._ee_pos_offset = torch.tensor(self.cfg.ee_pos_offset, device=self.device).repeat(self.num_envs, 1)
        self._ee_rot_offset = torch.tensor(self.cfg.ee_rot_offset, device=self.device).repeat(self.num_envs, 1)
        # ---

        self.step_count = 0

        self.base_pose = torch.tensor([self.cfg.base_pose], device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # create visualizer for real frame etc.
        if self.cfg.debug_vis:
            # add plots
            self.visualizers = {
                "Images": DirectLiveVisualizer(self.cfg.debug_vis, self.num_envs, None, visualizer_name="Images"),
                # "Metrics": DirectLiveVisualizer(
                #     self.cfg.debug_vis, self.num_envs, None, visualizer_name="Metrics"
                # ),
            }

            self.visualizers["Images"].terms["tactile_rgb"] = torch.zeros(
                (
                    self.num_envs,
                    self.gsmini.tactile_image_shape[0],
                    self.gsmini.tactile_image_shape[1],
                    self.gsmini.tactile_image_shape[2],
                )
            )
            self.visualizers["Images"].terms["calib_points"] = torch.zeros(
                (
                    self.num_envs,
                    self.gsmini.tactile_image_shape[0],
                    self.gsmini.tactile_image_shape[1],
                    self.gsmini.tactile_image_shape[2],
                )
            )
            # self.visualizers["Metrics"].terms["SSIM"] = torch.zeros((self.num_envs, 1))

            for vis in self.visualizers.values():
                vis.create_visualizer()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        indenter_holder_plate = self.cfg.indenter_holder_plate
        indenter_holder_plate.spawn.func(
            indenter_holder_plate.prim_path,
            indenter_holder_plate.spawn,
            translation=indenter_holder_plate.init_state.pos,
            orientation=indenter_holder_plate.init_state.rot,
        )
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=True)

        self.scene.rigid_objects["indenters"] = RigidObject(self.cfg.indenters)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/TCP",
                    name="end_effector",
                    offset=OffsetCfg(pos=self.cfg.ee_pos_offset, rot=self.cfg.ee_rot_offset),
                ),
            ],
        )

        self.object = list(self.scene.rigid_objects.values())[0]

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        # -- gui window for playing around with Taxim Simulation parameters
        self.param_window: TaximParameterWindow = TaximParameterWindow(
            "Taxim Parameters", self.gsmini, width=500, height=800
        )

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
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

        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    # post-physics step calls

    # MARK: dones
    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # which environment is done
        pass

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # MARK: observations
    def _get_observations(self) -> dict:
        pass


def step_env(env: ShapeTouchEnv):
    env._pre_physics_step(None)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)

    env.param_window.collect_taxim_parameters()

    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def run_data_collection(env: ShapeTouchEnv):
    """Runs the simulation loop."""

    # for convenience, we directly turn on debug_vis
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    # compute calibration pattern
    calib_area_height = 0.02075  # 0.0143  # # we use camera fov instead of gelpad area
    calib_area_width = 0.02525  # 0.0186
    # calib_area_height = 0.0143  # # we use camera fov instead of gelpad area
    # calib_area_width = 0.0186
    pixmm = 0.0634

    cylinder_radius_mm = 2.0
    calib_points = create_calib_points_in_gelpad_frame(
        cylinder_radius_mm,
        calib_area_height,
        calib_area_width,
        num_points_x=3,
        num_points_y=5,
    )
    # filter out calib points that are not inside the tactile img
    calib_points_filtered, calib_points_pix = create_calib_points_in_image_frame(
        calib_points, imgw=320, imgh=240, pixmm=0.0634
    )

    base_center_x = 0.5
    base_center_y = 0
    calib_points_filtered[:, 0] += base_center_x
    calib_points_filtered[:, 1] += base_center_y
    calib_pattern = torch.tensor(calib_points_filtered, device=env.device)

    # draw the calib points
    calib_points_filtered = calib_points
    calib_points_filtered[:, 0] += base_center_x
    calib_points_filtered[:, 1] += base_center_y
    calib_points_filtered = np.concatenate(
        [calib_points_filtered, np.full((calib_points_filtered.shape[0], 1), 0.02)], axis=1
    )
    calib_points_3d = []
    for i in range(env.num_envs):
        points_3d = calib_points_filtered.copy()
        points_3d[:, :2] += env.scene.env_origins[i, :2].cpu().numpy()
        calib_points_3d.append(points_3d)

    draw.clear_points()
    points = np.array(calib_points_3d).flatten().reshape((-1, 3))
    draw.draw_points(points, [(255, 0, 255, 0.5)] * points.shape[0], [30] * points.shape[0])

    env.param_window.num_calib_point = calib_pattern.shape[0]

    print(f"Starting simulation with {env.num_envs} envs")

    env.reset()

    # move to ee initial position
    for _ in range(50):
        env.ik_commands[:] = env.base_pose[:]
        step_env(env)
    prev_calib_point_idx = None
    # Data collection loop
    while simulation_app.is_running():
        calib_point_idx = env.param_window.current_calib_point_idx
        if prev_calib_point_idx is None:
            prev_calib_point_idx = calib_point_idx
        elif prev_calib_point_idx != calib_point_idx:
            print("Moving to calib point ", calib_point_idx)
            # move ee up first
            for _ in range(15):
                env.ik_commands[:] = env.base_pose[:]
                env.ik_commands[:, 2] += 0.005
                step_env(env)
            # then move ee to calib point
            for _ in range(20):
                env.ik_commands[:, :2] = calib_pattern[calib_point_idx]
                step_env(env)
            prev_calib_point_idx = calib_point_idx

        # -- move ee down
        depth = env.param_window.indentation_depth_widget.as_float
        env.ik_commands[:, 2] = 0.02 - depth
        env.ik_commands[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

        step_env(env)
        # update data visualization
        env.visualizers["Images"].terms["tactile_rgb"] = env.gsmini.data.output["tactile_rgb"]

        # draw all calib points of the calibration pattern for debug purposes for all envs
        for i in range(env.num_envs):
            tactile_rgb = env.gsmini.data.output["tactile_rgb"][i].cpu().numpy()
            for j in range(calib_points_pix.shape[0]):
                # mark the current goal pos with blue circle
                if calib_point_idx == j:
                    color = (0, 0, 255)
                else:
                    color = (50, 50, 50)

                cv2.circle(
                    tactile_rgb,
                    (calib_points_pix[j, 0], calib_points_pix[j, 1]),
                    int(cylinder_radius_mm / pixmm),
                    color,
                    1,
                )
                cv2.circle(
                    tactile_rgb,
                    (calib_points_pix[j, 0], calib_points_pix[j, 1]),
                    1,
                    color,
                    1,
                )
            # create image with the calib points positions drawn onto it
            env.visualizers["Images"].terms["calib_points"][i] = torch.tensor(tactile_rgb)

    env.close()


def main():
    """Main function."""

    # Define simulation env
    env_cfg = ShapeTouchEnvCfg()

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        len(INDENTER_FILE_PATHS)  # one env for each indenter shape
    )
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    experiment = ShapeTouchEnv(env_cfg)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_data_collection(env=experiment)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim apply
        simulation_app.close()
