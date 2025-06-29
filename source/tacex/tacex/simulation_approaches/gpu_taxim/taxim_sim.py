from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as F

from .sim import Taxim
from ..gelsight_simulator import GelSightSimulator
from ...gelsight_sensor import GelSightSensor

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg

class TaximSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors 
    inside Isaac Sim.
    
    """
    cfg: TaximSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TaximSimulatorCfg):
        self.sensor = sensor

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everythings
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """

        super().__init__(sensor=sensor, cfg=cfg)


    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else: 
            self._device = self.cfg.device
            
        self._taxim: Taxim = Taxim(calib_folder=calib_folder, device=self._device)
        # update Taxim settings via settings from cfg class
        # print(self._taxim.width)
        # self._taxim.width = self.cfg.tactile_img_res[0]
        # self._taxim.height = self.cfg.tactile_img_res[1]
        
        # tactile rgb image without indentation
        self.background_img = self._taxim.background_img
        #  up/downscale height map if different than tactile img res
        if self.background_img.shape != (3, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            self.background_img = F.resize(self.background_img, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))
        # last dim should be channels for isaac
        self.background_img = self.background_img.movedim(0, 2)
    
        self.tactile_rgb_img = torch.zeros(
            (self._num_envs, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0], 3),
            device=self._device, 
        )
        self.tactile_rgb_img[:] = self.background_img

        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

    def optical_simulation(self):
        """ Returns simulation output of Taxim optical simulation.
        
        Images have the shape (num_envs, height, width, channels) and values in range [0,255].
        """
        height_map = self.sensor._data.output["height_map"]

        # up/downscale height map if camera res different than tactile img res
        if height_map.shape != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
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

        self.tactile_rgb_img[:] = self._taxim.render_direct(
            height_map[:],
            with_shadow=self.cfg.with_shadow,
            press_depth=self._indentation_depth,
            orig_hm_fmt=False,
        ).movedim(1, 3) #*255).type(torch.uint8) 

        return self.tactile_rgb_img

    def compute_indentation_depth(self):
        height_map = self.sensor._data.output["height_map"] / 1000 # convert height map from mm to meter
        min_distance_obj = height_map.amin((1,2))
        # smallest distance between object and sensor case
        dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

        # print("dist_obj_sensor_case", dist_obj_sensor_case)
        # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
        #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
        #     dist_obj_sensor_case = 0
        dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

        self._indentation_depth[:] = torch.where(
            dist_obj_sensor_case <= self.cfg.gelpad_height, 
            (self.cfg.gelpad_height - dist_obj_sensor_case)*1000, 
            0
        )

        return self._indentation_depth
    
    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self.tactile_rgb_img[:] = self.background_img