from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import omni.usd
import numpy as np

import torch
import torchvision.transforms.functional as F

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg

from .sim import MarkerMotion
from ..gelsight_simulator import GelSightSimulator
from ..gpu_taxim import TaximSimulator
from ..gpu_taxim.sim import TaximTorch

from ...gelsight_sensor import GelSightSensor

if TYPE_CHECKING:
    from .fots_marker_sim_cfg import FOTSMarkerSimulatorCfg

class FOTSMarkerSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors 
    inside Isaac Sim.
    
    The class uses an instance of the gpu_taxim simulator for some of it operations.
    """
    cfg: FOTSMarkerSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: FOTSMarkerSimulatorCfg):
        self.sensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)
        
        # use IsaacLab FrameTransformer for keeping track of relative positon/rotation
        self.frame_transformer: FrameTransformer = FrameTransformer(self.cfg.frame_transformer_cfg)
            
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

        # use Taxim for gpu based operations
        if ((self.sensor.optical_simulator is not None)
            and (type(self.sensor.optical_simulator) is TaximSimulator)):
            self._taxim: TaximTorch = self.sensor.optical_simulator._taxim
        else:
            raise RuntimeError("Currently FOTS simulation approach has to be used in combination with GPU-Taxim as optical-simulator.")

        # tactile rgb image without indentation
        bg_img = self._taxim.background_img.movedim(0, 2).cpu().numpy()
        self.marker_motion_sim = MarkerMotion(
            frame0_blur=bg_img,
            mm2pix=self.cfg.mm_to_pixel,
            num_markers_col=self.cfg.marker_params.num_markers_col, #20, #11
            num_markers_row=self.cfg.marker_params.num_markers_row, #15, #9
            tactile_img_width=self.cfg.tactile_img_res[0], # default 480
            tactile_img_height=self.cfg.tactile_img_res[1], # default 640
            lamb=[0.00125,0.0021,0.0038], #[0.00125,0.00021,0.00038],
            x0=self.cfg.marker_params.x0,
            y0=self.cfg.marker_params.y0
        )

        self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)
        
        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

        # create buffers
        self.marker_data = torch.zeros((self.sensor._num_envs, self.cfg.marker_params.num_markers_row, self.cfg.marker_params.num_markers_col, 2), device=self._device)
        
        self.sensor._data.output["traj"] = []
        for _ in range(self.sensor._num_envs):
            self.sensor._data.output["traj"].append([])
        self.theta = torch.zeros((self.sensor._num_envs), device=self._device)

        # need to initialize manually
        self.frame_transformer._initialize_impl() 
        self.frame_transformer._is_initialized = True
        print("Frame transformer for FOTS: ", self.frame_transformer)

    def marker_motion_simulation(self):
        self.frame_transformer.update(dt=0)
        self._indentation_depth = self.sensor._indentation_depth
        height_map = self.sensor._data.output["height_map"] # height map has shape (height, width) cause row-column format

        # up/downscale height map if camera res different than tactile img res
        if height_map.shape != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()
            self._indentation_depth = self.sensor._indentation_depth.cpu()

        height_map_shifted = self._taxim._TaximTorch__get_shifted_height_map(self._indentation_depth, height_map)
        deformed_gel, contact_mask = self._taxim._TaximTorch__compute_gel_pad_deformation(height_map_shifted)
        deformed_gel = deformed_gel.max() - deformed_gel

       
        for h in range(deformed_gel.shape[0]):
            if self._indentation_depth[h].item() > 0.0:
                # compute contact center based on contact_mask
                contact_points = torch.argwhere(contact_mask[h])
                mean = torch.mean(contact_points.float(), dim=0).cpu().numpy()
                #print("should be pix ", mean[1], mean[0])
                # rows = height = y values
                mean[0] = (mean[0]-self.marker_motion_sim.tactile_img_height/2)/self.marker_motion_sim.mm2pix
                # columns = width = x values
                mean[1] = (mean[1]-self.marker_motion_sim.tactile_img_width/2)/self.marker_motion_sim.mm2pix
                #self.sensor._data.output["traj"][h].append([mean[1], mean[0], self.theta[h].cpu().numpy()])
                #print("should be ", mean[1], mean[0])
                
                # rel position/orientation of obj to sensor
                rel_pos = self.frame_transformer.data.target_pos_source.cpu().numpy()[h,0,:] # target_pos_source shape is (num_envs, num_targets, 3)
                rel_pos *= 1000 # convert to mm
                # print("rel_pos in pix ", self.cfg.mm_to_pixel*rel_pos[0] + self.marker_motion_sim.tactile_img_width/2, self.cfg.mm_to_pixel*rel_pos[1] + self.marker_motion_sim.tactile_img_height/2)
                # print("rel_pos ", rel_pos)
                rel_orient = self.frame_transformer.data.target_quat_source[h] #-> only one target_frame
                roll, pitch, yaw = euler_xyz_from_quat(rel_orient)
                theta = yaw.cpu().numpy()
                # angle = 0

                #self.sensor._data.output["traj"][h].append([rel_pos[0], rel_pos[1], theta])
                # print("")
                
                # traj = contaxt data (x,y,theta)
                self.sensor._data.output["traj"][h].append([mean[1], mean[0], theta])
                
                #todo vectorize with pytorch 
                marker_x_pos, marker_y_pos = self.marker_motion_sim.marker_sim(
                    deformed_gel[h].cpu().numpy(), 
                    contact_mask[h].cpu().numpy(), 
                    self.sensor._data.output["traj"][h]
                )
            else:
                self.sensor._data.output["traj"][h] = []
                marker_x_pos = self.marker_motion_sim.init_marker_x_pos
                marker_y_pos = self.marker_motion_sim.init_marker_y_pos
                
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
