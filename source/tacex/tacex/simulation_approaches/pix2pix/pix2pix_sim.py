# This sim approach loads a trained pix2pix model and uses it to run inference.
#
# Code to train pix2pix model: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master


from __future__ import annotations

import copy
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from pathlib import Path
import json

from collections import OrderedDict

import cv2
import omni
import torchvision.transforms.functional as F

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat


from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator

from .pytorch_cyclegan_and_pix2pix.networks import define_G, patch_instance_norm_state_dict

if TYPE_CHECKING:
    from .pix2pix_sim_cfg import Pix2PixSimulatorCfg


class Pix2PixSimulator(GelSightSimulator):
    """This simulator class uses a trained pix2pix model to map height images to tactile_rgb images."""

    cfg: Pix2PixSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: Pix2PixSimulatorCfg):
        # if self.cfg.calib_folder_path == "":
        ## cannot use tacex_asset class here due to circular import
        #     self.cfg.calib_folder_path = f"{TACEX_SENSORS_DATA_DIR}/GelSight_Mini/calibs/pix2pix"

        self.calib_folder_path = Path(cfg.calib_folder_path)

        # take the json file
        config_path = list(self.calib_folder_path.rglob("*.json"))

        assert len(config_path) == 1

        with open(str(config_path[0])) as config_file:
            self.config = json.load(config_file)
        print("Model Config: ", self.config)

        # new_dict = OrderedDict()
        # for k, v in self.model_dict.items():
        #     # load_state_dict expects keys with prefix 'module.'
        #     new_dict["module." + k] = v
        # self.model_dict = new_dict

        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs

        # load background image
        # f0 = self.__bgr_to_rgb(self.__np_img_to_torch(data_file["f0"] / 255))
        bg_img = cv2.imread(str(self.calib_folder_path / "0.png"), cv2.IMREAD_COLOR)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_img = cv2.resize(
            bg_img, [self.cfg.tactile_img_res[0], self.cfg.tactile_img_res[1]], interpolation=cv2.INTER_LINEAR
        )  # (height, width)
        bg_img = torch.tensor(bg_img, device=self.cfg.device) / 255.0

        # self.__bg_proc = self.__process_initial_frame(f0)
        self._bg_img = bg_img

        # Define the generator model
        # -> make sure you use the config.json with the same parameters that were used for training the model
        pix2pix_config = self.config["pix2pix_model"]
        self.generator_model = define_G(
            input_nc=pix2pix_config["input_nc"],
            output_nc=pix2pix_config["output_nc"],
            ngf=pix2pix_config["ngf"],
            netG=pix2pix_config["netG"],
            norm=pix2pix_config["norm"],
            use_dropout=pix2pix_config["use_dropout"],
        )

        if isinstance(self.generator_model, torch.nn.parallel.DistributedDataParallel):
            self.generator_model = self.generator_model.module

        load_path = str(self.calib_folder_path / self.config["checkpoint"])
        print(f"Loading the model from {load_path}.")

        # reference for loading model: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py#L221
        state_dict = torch.load(load_path)  # , map_location=str(self._device), weights_only=True)
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        self.generator_model.load_state_dict(state_dict)
        if self._device != "cpu":
            self.generator_model.cuda()
        self.generator_model.eval()  # - somehow worse image simulation with eval mode -> due to skip connections disabled

        self.tactile_rgb_img = torch.zeros(
            (self.sensor._num_envs, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0], 3),
            device=self._device,
        )

    def optical_simulation(self):
        height_map = self.sensor._data.output["height_map"]
        # up/downscale height map if camera res different than tactile img res
        # if (height_map.shape[1], height_map.shape[2]) != (self.config["img_height"], self.config["img_width"]):
        #     height_map = F.resize(height_map, (self.config["img_height"], self.config["img_width"]))

        # need to match dim which were used for training
        if self.config["pix2pix_model"]["load_size"] is not None:
            height_map = F.resize(
                height_map, [self.config["pix2pix_model"]["load_size"], self.config["pix2pix_model"]["load_size"]]
            )

        # normalize map [-gelpad_height, 0] to [0,255] for proper depth image of height_map
        # (generate same img as what was used for training)
        height_map = (
            (
                torch.clamp(height_map, -self.sensor.cfg.max_indentation_depth * 1000, 0)
                + self.sensor.cfg.max_indentation_depth * 1000
            )
            / (self.sensor.cfg.max_indentation_depth * 1000)
            * 255
        ).type(torch.uint8)

        # translation into [-1,1] range, which is needed for the model
        height_map = (height_map / 255.0) * 2.0 - 1.0
        height_map = height_map.clamp(-1.0, 1.0)

        if self._device == "cpu":
            height_map = height_map.cpu()

        # add 1 color channel for gray scale
        height_map = height_map[:, None, :, :]

        with torch.no_grad():
            pix2pix_output = self.generator_model(height_map)

        if (pix2pix_output.shape[1], pix2pix_output.shape[2]) != (
            self.cfg.tactile_img_res[1],
            self.cfg.tactile_img_res[0],
        ):
            pix2pix_output = F.resize(pix2pix_output, [self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]])

        # move color channel to last dim
        pix2pix_output = torch.movedim(pix2pix_output, 1, 3)

        # model generates image tensor with values [-1, 1], convert to [0,1]
        pix2pix_output = (pix2pix_output + 1.0) / 2.0

        # model generally gives us pretty good image, but background is bad -> due to training on random cropped area
        # so we blend real background image and simulated image

        # pix2pix_output = (pix2pix_output + self._bg_img).clamp(0.0, 1.0).type(torch.float32)
        # bg_weight = 0.3
        # pix2pix_output = (bg_weight * self._bg_img + pix2pix_output * (1 - bg_weight)).clip(0, 1)

        # # only add background were height map != 0
        # bg_repeated = self._bg_img.unsqueeze(0).repeat(self._num_envs, 1, 1, 1)
        # pix2pix_output[height_map[:, 0, :, :] == 1.0] = bg_repeated[height_map[:, 0, :, :] == 1.0]

        # # todo rmv debug line
        # test = pix2pix_output.cpu().numpy()
        # test = test.squeeze(0)
        # test = (test * 255.0).clip(0, 255).astype(np.uint8)
        # test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("test.png", test)

        self.tactile_rgb_img[:] = pix2pix_output
        return self.tactile_rgb_img

    def reset(self, env_ids: Sequence[int] | None = None):
        """Pix2Pix is stateless: the no-contact image is produced by the sensor calling
        :meth:`optical_simulation` with the zeroed height map of ``env_ids``."""
        pass

    # def __process_initial_frame(self, f0: torch.Tensor):
    #     """
    #     Conduct some preprocessing on the initial frame.
    #     :param f0: A 3xHxW torch tensor containing the initial frame.
    #     :return: A 3xHxW torch tensor containing the processed initial frame.
    #     """
    #     # gaussian filtering with square kernel
    #     f0_blurred = self.__gaussian_blur(f0, self.sim_params.initial_frame_sigma(f0.shape[1:]))
    #     # Checking the difference between original and filtered image
    #     d_i = torch.mean(f0_blurred - f0, dim=0)

    #     # Mixing image based on the difference between original and filtered image
    #     fmp = self.sim_params.frame_mixing_percentage
    #     thresh = self.sim_params.diff_threshold

    #     return torch.where((d_i < thresh).unsqueeze(0), fmp * f0_blurred + (1 - fmp) * f0, f0)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.
        """
        # note: parent only deals with callbacks. not their visibility
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

    # TODO make unified "tactile_rgb sim" class, which implements this general debug vis -> right now its just copy-pasted from TaximSimulator
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
