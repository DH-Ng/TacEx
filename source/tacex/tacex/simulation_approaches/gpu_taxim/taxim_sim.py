from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import omni.usd
import torchvision.transforms.functional as F

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from .sim import Taxim

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg


class TaximSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors
    inside Isaac Sim.

    """

    cfg: TaximSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TaximSimulatorCfg):
        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """

        # torch backend is required: this wrapper (and FOTS) operate on torch tensors
        self._taxim: Taxim = Taxim(
            calib_folder=calib_folder, backend="torch", device=self._device, params=self.cfg.taxim_parameters
        )

        # -- NOTE: Taxim sim uses (channels, height, width) format

        # tactile rgb image without indentation
        self.background_img = self._taxim.background_img
        #  up/downscale height map if different than tactile img res
        if self.background_img.shape != (3, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            self.background_img = F.resize(
                self.background_img, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])
            )
        # last dim should be channels for isaac
        self.background_img = self.background_img.movedim(0, 2)

        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

        # no-contact image (written into the output by reset): the simulation of an undeformed gel, i.e. a zero height
        # map without indentation (rendering is independent per env, so one env suffices)
        w, h = self.cfg.tactile_img_res
        self._rest_img = self._render(
            torch.zeros((1, h, w), device=self._device), torch.zeros(1, device=self._device)
        )[0]

    def get_output_shape(self, data_type: str) -> tuple[int, ...]:
        if data_type != "tactile_rgb":
            return super().get_output_shape(data_type)
        return (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0], 3)

    def _render(self, height_map: torch.Tensor, press_depth: torch.Tensor) -> torch.Tensor:
        """Taxim images (N, height, width, 3) for height maps (N, height, width) at the tactile image resolution."""
        return self._taxim.render_direct(
            height_map,
            with_shadow=self.cfg.with_shadow,
            press_depth=press_depth,
            orig_hm_fmt=False,
        ).movedim(1, 3)  # *255).type(torch.uint8)

    def optical_simulation(self):
        """Simulates the Taxim tactile images into the sensor's ``tactile_rgb`` buffer.

        Images have the shape (num_envs, height, width, channels).
        """
        height_map = self.sensor._data.output["height_map"]

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()

        # only simulate in case of indentation
        # self.tactile_rgb_img[self._indentation_depth <= 0][:] = self.background_img
        # if height_map[self._indentation_depth > 0].shape[0] > 0:
        #     self.tactile_rgb_img[self._indentation_depth > 0] = self._taxim.render_direct(
        #         height_map[self._indentation_depth > 0],
        #         with_shadow=self.cfg.with_shadow,
        #         press_depth=self._indentation_depth[self._indentation_depth > 0],
        #         orig_hm_fmt=False,
        #     ).movedim(1, 3) #*255).type(torch.uint8)

        out = self._output_buffers["tactile_rgb"]
        out[:] = self._render(height_map[:], self.sensor._indentation_depth).to(out.device)

    def compute_indentation_depth(self):
        height_map = self.sensor._data.output["height_map"]
        # for old height map
        # min_distance_obj = height_map.amin((1, 2))
        # # smallest distance between object and sensor case
        # dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance * 1000
        # dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

        # self._indentation_depth[:] = torch.where(
        #     dist_obj_sensor_case <= self.cfg.gelpad_height, (self.cfg.gelpad_height * 1000 - dist_obj_sensor_case), 0
        # )

        self._indentation_depth[:] = -height_map.amin((1, 2))
        return self._indentation_depth

    def reset(self, env_ids: Sequence[int] | None = None):
        """Writes the no-contact tactile image (simulated undeformed gel) into the env_ids rows of the output.

        Assumes that sensor is in no-contact (users responsibility)."""

        if env_ids is None:
            env_ids = self.sensor._ALL_INDICES

        self._indentation_depth[env_ids] = 0.0
        out = self._output_buffers.get("tactile_rgb")
        if out is not None:
            out[env_ids] = self._rest_img.to(out.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then two windows will be shown.
        """
        # NOTE: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "_debug_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._debug_windows = {}
                self._debug_img_providers = {}
                # todo check if we can make implementation more efficient than dict of dicts
                if "tactile_rgb" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows
        for i, prim in enumerate(self.sensor.prim_view.prims):
            if "tactile_rgb" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_tactile_rgb").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/taxim_rgb",
                            height=self.cfg.tactile_img_res[1],
                            width=self.cfg.tactile_img_res[0],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self.sensor.data.output["tactile_rgb"][i].cpu().numpy()  # * 255
                    # frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    # frame = frame.astype(np.uint8)

                    # update image of the debug window
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._debug_img_providers[str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._debug_img_providers[str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()
