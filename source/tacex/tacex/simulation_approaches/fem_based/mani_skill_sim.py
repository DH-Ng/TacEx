from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import omni.usd
import torch
import torchvision.transforms.functional as F
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.utils.math import euler_xyz_from_quat

from tacex_uipc import UipcObject
from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator

from .sim import VisionTactileSensorUIPC

if TYPE_CHECKING:
    from .mani_skill_sim_cfg import ManiSkillSimulatorCfg

class ManiSkillSimulator(GelSightSimulator):
    """Wrapper for ManiSkill-ViTac simulator for GelSight sensors.

    The original code can be found here https://github.com/chuanyune/ManiSkill-ViTac2025.git
    """
    cfg: ManiSkillSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: ManiSkillSimulatorCfg):
        self.sensor: GelSightSensor = sensor

        # needed for VisionTactileSensorUIPC class
        self.camera = None
        self.gelpad_uipc: UipcObject = self.sensor.gelpad_obj

        super().__init__(sensor=sensor, cfg=cfg)


    def _initialize_impl(self):
        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everythings
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """

        self.camera = self.sensor.camera 
        self.marker_motion_sim: VisionTactileSensorUIPC = VisionTactileSensorUIPC(
            self.gelpad_uipc,
            self.camera,
            self.cfg.marker_interval_range
        )

        self.marker_motion_sim._gen_marker_grid()


    def marker_motion_simulation(self):
        marker_flow = self.marker_motion_sim.gen_marker_flow()
        self.marker_data = marker_flow
        return self.marker_data

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        # self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """ Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlayed.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "_debug_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._debug_windows = {}
                self._debug_img_providers = {}
                #todo check if we can make implementation more efficient than dict of dicts
                if "marker_motion" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows_prim_view
        for i, prim in enumerate(self.sensor._prim_view.prims):
            if "marker_motion" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_marker_motion").Get()
                if show_img==True:
                    if not (str(i) in self._debug_windows):
                        # create a window
                        window = omni.ui.Window(self.sensor._prim_view.prim_paths[i] + "/fots_marker", height=640, width=480)
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = omni.ui.ByteImageProvider() # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = np.zeros((self.cfg.tactile_img_res[1],self.cfg.tactile_img_res[0])).astype(np.uint8)

                    # like the `_generate` function of FOTS MarkerMotion sim
                    marker_data = self.sensor.data.output["marker_motion"][i]
                    # position values are in pix
                    x_pos_of_all_markers = marker_data[:,:,0].cpu().numpy() # = columns, shape (num_markers_row, num_markers_col)
                    y_pos_of_all_markers  = marker_data[:,:,1].cpu().numpy() # = row
                    color = (255, 255, 255)

                    num_markers_row = x_pos_of_all_markers.shape[0]
                    num_markers_col = x_pos_of_all_markers.shape[1]
                    for k in range(num_markers_col):
                        for j in range(num_markers_row):
                            init_x_pos = int(self.init_marker_pos[0][j,k])    # get initial x position of marker [j,k]
                            init_y_pos = int(self.init_marker_pos[1][j,k]) # get initial y position of marker [j,k]
                            x_pos = int(x_pos_of_all_markers[j, k]) # x is column-wise definied
                            y_pos = int(y_pos_of_all_markers[j, k]) # y row-wise
                            if ((x_pos >= frame.shape[1])
                                or (x_pos < 0)
                                or (y_pos >= frame.shape[0])
                                or (y_pos < 0)):
                                continue
                            # cv2.circle(frame,(column,row), 6, (255,255,255), 1, lineType=8)

                            arrow_scale = 0.001 #10 #0.0001 #0.25
                            pt1 = (init_x_pos, init_y_pos)
                            # pt2 = (column+int(arrow_scale*(column-init_column)), row+int(arrow_scale*(row-init_row)))
                            pt2 = (x_pos, y_pos)
                            cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.1)
                    # visualize contact center with red cross
                    # if len(self._data.output["traj"][i]) > 1:
                    #     traj = []
                    #     traj.append(self._data.output["traj"][i][0])
                    #     cv2.circle(frame,(int(traj[0][0]/self.taxim.sensor_params.pixmm + frame.shape[0]/2), int(traj[0][1]/self.taxim.sensor_params.pixmm + frame.shape[1]/2)), 5, (255,0,0), -1)
                    #     # cv2.circle(frame,(int(traj[0][0]/self.taxim.sensor_params.pixmm + 320), int(traj[0][1]/self.taxim.sensor_params.pixmm + 240)), 5, (255,0,0), -1)

                        # should = self.should_be.cpu().numpy()[0]
                        # cv2.circle(frame,(should[1], should[0]), 4, (0,255,0), -1)
                    frame = cv2.normalize(frame, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA) #cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        #self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._debug_img_providers[str(i)].set_bytes_data(frame.flatten().data, [width, height]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        image = omni.ui.ImageWithProvider(self._debug_img_providers[str(i)]) #, fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()
