from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .sim import Taxim

from ..optical_sim import OpticalSimulator

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg

class TaximSimulator(OpticalSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors 
    inside Isaac Sim.
    
    """
    cfg: TaximSimulatorCfg

    def __init__(self, cfg, height_map: torch.Tensor):
        self.height_map: torch.Tensor = height_map

        super().__init__(cfg)

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        self._device = self.cfg.device
        self._num_envs = self.cfg.num_envs

        self._taxim = Taxim(calib_folder=calib_folder, device=self._device)

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everythings
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """
    
        # if camera resolution is different than the tactile RGB res, scale img

        self.img_res = self.cfg.tactile_img_res

    def optical_simulation(self):
        
        #shifted_height_map = self._taxim._get_shifted_height_map(self._indentation_depth, self.height_map)
        #todo only render img where indentation_depth > 0
        tactile_rgb_img = self._taxim.render(
            self.height_map,
            with_shadow=self.cfg.with_shadow,
            press_depth=self._indentation_depth,
            orig_hm_fmt=False,
        )
        return tactile_rgb_img

    def compute_indentation_depth(self):
        self.height_map = self.height_map / 1000 # convert height map from mm to meter
        min_distance_obj = self.height_map.amin((1,2))
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