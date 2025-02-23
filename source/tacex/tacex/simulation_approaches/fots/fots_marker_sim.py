from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .sim import MarkerMotion
from ..gelsight_simulator import GelSightSimulator
from ..gpu_taxim import TaximSimulator
from ...gelsight_sensor import GelSightSensor

if TYPE_CHECKING:
    from .fots_marker_sim_cfg import FOTSMarkerSimulatorCfg

class FOTSMarkerSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors 
    inside Isaac Sim.
    
    The class uses an instance of the gpu_taxim simulator for some of it operations.
    """
    cfg: FOTSMarkerSimulatorCfg

    def _initialize_impl(self):
        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else: 
            self._device = self.cfg.device

        # use Taxim for gpu based operations
        if ((self.sensor.optical_simulator is not None)
            and (type(self.sensor.optical_simulator) is TaximSimulator)):
            self._taxim = self.sensor.optical_simulator._taxim
        else:
            raise RuntimeError("Currently FOTS simulation approach has to be used in combination with GPU-Taxim as optical-simulator.")

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everythings
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """

        bg_img = self._taxim._bg_proc.permute(1, 2, 0).cpu().numpy()
        self.marker_motion_sim = MarkerMotion(
            frame0_blur=bg_img,
            mm2pix=self.cfg.mm_to_pixel,
            num_markers_col=self.cfg.marker_params.num_markers_col, #20, #11
            num_markers_row=self.cfg.marker_params.num_markers_row, #15, #9
            tactile_img_height=self.cfg.tactile_img_res[0], # 480 #TODO fix, so that resolution is always WxH (e.g for Taxim thats kidna messed up)
            tactile_img_width=self.cfg.tactile_img_res[1], # 640
            lamb=[0.00125,0.0021,0.0038], #[0.00125,0.00021,0.00038],
            x0=self.cfg.marker_params.x0,
            y0=self.cfg.marker_params.y0
        )

        self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)
        
        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

        # create buffers
        self.marker_data = torch.zeros((self._num_envs, self.cfg.marker_params.num_markers_row, self.cfg.marker_params.num_markers_col, 2), device=self._device)
        
        self.sensor._data.output["traj"] = []
        for _ in range(self._num_envs):
            self.sensor._data.output["traj"].append([])
        self.theta = torch.zeros((self._num_envs), device=self._device)

    def marker_motion_simulation(self):
        self._indentation_depth = self.sensor._indentation_depth
        height_map_shifted = self._taxim._get_shifted_height_map(self._indentation_depth, self.sensor._data.output["height_map"])
        deformed_gel, contact_mask = self._taxim._compute_gel_pad_deformation(height_map_shifted)
        deformed_gel = deformed_gel.max() - deformed_gel

       
        for h in range(deformed_gel.shape[0]):
            if self._indentation_depth[h].item() > 0.0:
                # compute contact center based on contact_mask
                contact_points = torch.argwhere(contact_mask[h])
                mean = torch.mean(contact_points.float(), dim=0).cpu().numpy()
                # angle = 0
                # if len(self.sensor._data.output["traj"][h]) == 0:
                #     should_be = (self._data.output["height_map"]==self._data.output["height_map"].min()).nonzero()[0]
                #     print("should be in pix: ", should_be)
                #     print("should be in mm: ", (should_be[0]).cpu().numpy()*self.taxim.sensor_params.pixmm, (should_be[1]).cpu().numpy()*self.taxim.sensor_params.pixmm)
                #     self.should_be = (self._data.output["height_map"]==self._data.output["height_map"].min()).nonzero()                
                #     print("mean", mean)

                mean[0] = (mean[0]-self.marker_motion_sim.tactile_img_width/2)/self.marker_motion_sim.mm2pix
                mean[1] = (mean[1]-self.marker_motion_sim.tactile_img_height/2)/self.marker_motion_sim.mm2pix
                
                # if len(self.sensor._data.output["traj"]) >= 2:
                #     self._data.output["motion_vectors"] = self.camera.data.output["motion_vectors"][:, :, :, :2]
                #     # compute rotation angle
                #     # a = mean - np.array(self.sensor._data.output["traj"][0][0], self.sensor._data.output["traj"][0][1])
                #     # #TODO create fixed variable for init contact, when it comes to contact -> no need to create array everytime
                #     # b = np.array([self.sensor._data.output["traj"][-1][0], self.sensor._data.output["traj"][-1][1]]) - np.array([self.sensor._data.output["traj"][0][0], self.sensor._data.output["traj"][0][1]])
                #     # print("a ", a)
                #     # print("b ", b)
                #     # angle = np.arccos((np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))))
                #     # print("angle ", np.rad2deg(angle))
                self.sensor._data.output["traj"][h].append([mean[1], mean[0], self.theta[h].cpu().numpy()])
            else:
                self.sensor._data.output["traj"][h] = []
        
            #todo vectorize with pytorch 
            marker_x_pos, marker_y_pos = self.marker_motion_sim.marker_sim(
                deformed_gel[h].cpu().numpy(), 
                contact_mask[h].cpu().numpy(), 
                self.sensor._data.output["traj"][h]
            )
            self.marker_data[h, :, :, 0] = torch.tensor(marker_x_pos, device=self._device)
            self.marker_data[h, :, :, 1] = torch.tensor(marker_y_pos, device=self._device)
        return self.marker_data

    # def compute_indentation_depth(self):
    #     self.height_map = self.height_map / 1000 # convert height map from mm to meter
    #     min_distance_obj = self.height_map.amin((1,2))
    #     # smallest distance between object and sensor case
    #     dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

    #     # print("dist_obj_sensor_case", dist_obj_sensor_case)
    #     # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
    #     #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
    #     #     dist_obj_sensor_case = 0
    #     dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

    #     self._indentation_depth[:] = torch.where(
    #         dist_obj_sensor_case <= self.cfg.gelpad_height, 
    #         (self.cfg.gelpad_height - dist_obj_sensor_case)*1000, 
    #         0
    #     )

    #     return self._indentation_depth
    
    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)