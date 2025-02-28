from __future__ import annotations

import numpy as np
from omni import physx
import scipy.spatial.transform as tf
from dataclasses import dataclass, MISSING
from collections.abc import Sequence
from typing import Any, Dict, Tuple, Union, List, TYPE_CHECKING

from omni.physx.scripts import utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils

import omni.kit.commands
import omni.usd

from isaacsim.core.prims import XFormPrim

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from omni.physx.scripts import deformableUtils, physicsUtils

from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
# from torchvision.transforms import v2

from isaaclab.utils import class_to_dict, to_camel_case, configclass
from isaaclab.utils.math import convert_quat

from isaaclab.sensors import SensorBase, SensorBaseCfg, Camera, CameraCfg, TiledCamera, TiledCameraCfg
# from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
# from isaaclab.sensors.camera.camera import Camera
# from isaaclab.sensors.camera.camera_cfg import CameraCfg


import time


from .simulation_approaches.gelsight_simulator import GelSightSimulator

from .gelsight_sensor_data import GelSightSensorData
if TYPE_CHECKING:
    from .gelsight_sensor_cfg import GelSightSensorCfg


class GelSightSensor(SensorBase):
    cfg: GelSightSensorCfg
    
    def __init__(self, cfg: GelSightSensorCfg):
        self.cfg = cfg

        # initialize base class
        super().__init__(self.cfg)

        # sensor camera
        self.camera = None

        self._indentation_depth: torch.tensor = None

        # simulation approaches for simulating GelSight sensor output
        self.optical_simulator: GelSightSimulator = None
        self.marker_motion_simulator: GelSightSimulator = None
        self.compute_indentation_depth_func = None

        # Create empty variables for storing output data
        self._data = GelSightSensorData()
        self._data.output = dict.fromkeys(self.cfg.data_types, None)

        # Flag to check that sensor is spawned.
        self._is_spawned = False

        #todo remove
        self.test = 0
    
    def __del__(self):
        """Unsubscribes from callbacks."""
        # unsubscribe callbacks
        super().__del__()

    # def __str__(self) -> str:
    #     """Returns: A string containing information about the instance."""
    #     # message for class
    #     return (
    #         f"Gelsight Mini @ '{self.cfg.prim_path}': \n"
    #         f"\tdata types   : {list(self._data.output.keys())} \n"
    #         f"\tupdate period (s): {self.cfg.update_period}\n"
    #         f"\tframe        : {self.frame}\n"
    #         f"\tresolution        : {self.image_resolution}\n"
    #         f"\twith shadows        : {self._simulate_shadows}\n"
    #         f"\tposition     : {self._data.position} \n"
    #         f"\torientation  : {self._data.orientation} \n"
    #     )
    
    """
    Properties
    """
    @property
    def data(self) -> GelSightSensorData:
        """Data related to Camera sensor."""
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data
    
    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame
    
    # @property
    # def image_resolution(self) -> Tuple[int, int]:
    #     """A tuple containing (height, width) of the camera sensor."""
    #     return self.cfg.resolution #TODO fix, image shape should be 3 dim, [width, height, channels (= 3)] -> change attribute name to "resolution" 

    @property
    def camera_resolution(self) -> Tuple[int, int]:
        """Shape of the simulated tactile RGB image, i.e. (height, width, channels)."""
        return self.cfg.sensor_camera_cfg.resolution[0], self.cfg.sensor_camera_cfg.resolution[1]  # type: ignore

    @property
    def press_depth(self):
        """How deep objects are inside the gel pad of the sensor"""
        return self._indentation_depth

    """
    Operations
    """
    #MARK: reset
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES # type: ignore
        
        # reset camera
        if self.camera is not None:
            self.camera.reset()

        # reset the buffer
        # self._data.position = None
        # self._data.orientation = None
        #self._data.image_resolution = self.image_resolution

        self._indentation_depth[env_ids] = 0 

        # reset height map
        self._data.output["height_map"][env_ids] = 0
        # torch.zeros(
        #     (env_ids.size(), self.camera_cfg.height, self.camera_cfg.width), 
        #     device=self.cfg.device
        # )

        # if self._interpolate_height_map:
        #     resized = F.resize(self._data.output["height_map"], (self.taxim.sensor_params.height,self.taxim.sensor_params.width))
        #     #TODO should I compute press depth after interpolation or before?
        #     self._data.output["height_map"] = resized

        # simulate optical/marker output, but without indentation
        if (self.optical_simulator is not None) and ("tactile_rgb" in self._data.output.keys()) :
            self._data.output["tactile_rgb"][:] = self.optical_simulator.optical_simulation()
            self.optical_simulator.reset()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self._data.output.keys()):
            # height_map_shifted = self.taxim._get_shifted_height_map(self._indentation_depth, self._data.output["height_map"])
            self._data.output["marker_motion"][:] = self.marker_motion_simulator.marker_motion_simulation() #TODO adjust mm2pix value 19.58 #/19.58
            # (yy_init_pos, xx_init_pos), i.e. along height x width of tactile img
            self._data.output["init_marker_pos"] = ([0], [0])
            
            self.marker_motion_simulator.reset()

        # Reset the frame count
        self._frame[env_ids] = 0

 
####
# Implemenation of abstract methods of base sensor class
#### 
    #MARK: _init_impl      
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers."""
        # Initialize parent class
        super()._initialize_impl()

        # set device, if specified (per default the same as the simulation)
        if self.cfg.device is not None:
            self._device = self.cfg.device

        prim_paths_expr = self.cfg.prim_path
        print(f"Initializing GelSight Sensor {prim_paths_expr}...")
        self._view = XFormPrim(prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}")
        self._view.initialize()
        # Check that sizes are correct
        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of sensor prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )
        # need to create the attribute for the debug_vis here since it depends on self._view
        if self.cfg.debug_vis:
            for prim in self._view.prims:
                # creates an USD attribut, which can be found in the Isaac GUI under "Raw Usd Properties -> "Extra Properties"
                attr = prim.CreateAttribute("show_tactile_image", Sdf.ValueTypeNames.Bool)
                attr.Set(False)

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)

        if self.cfg.sensor_camera_cfg is not None:
            # self.camera_cfg: TiledCameraCfg = TiledCameraCfg(
            #         prim_path= prim_paths_expr + self.cfg.sensor_camera_cfg.prim_path_appendix, 
            #         update_period= self.cfg.sensor_camera_cfg.update_period,
            #         height= self.cfg.sensor_camera_cfg.resolution[0],
            #         width= self.cfg.sensor_camera_cfg.resolution[1],
            #         data_types= self.cfg.sensor_camera_cfg.data_types,
            #         spawn= None, # use camera which is part of the GelSight Mini Asset
            #         # spawn=sim_utils.PinholeCameraCfg(
            #         #    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            #         # ),      
            #         #depth_clipping_behavior="max", # doesnt work, cause "max" value is taking from spawn config, which we dont have
            # )
            # self.camera = TiledCamera(cfg=self.camera_cfg)
            
            # use normal camera
            self.camera_cfg: CameraCfg = CameraCfg(
                    prim_path= prim_paths_expr + self.cfg.sensor_camera_cfg.prim_path_appendix, 
                    update_period= self.cfg.sensor_camera_cfg.update_period,
                    height= self.cfg.sensor_camera_cfg.resolution[0],
                    width= self.cfg.sensor_camera_cfg.resolution[1],
                    data_types= self.cfg.sensor_camera_cfg.data_types,
                    spawn= None, # use camera which is part of the GelSight Mini Asset
                    # spawn=sim_utils.PinholeCameraCfg(
                    #    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                    # ),      
                    #depth_clipping_behavior="max", # doesnt work, cause "max" value is taking from spawn config, which we dont have
            )
            self.camera = Camera(cfg=self.camera_cfg)

            # need to initialize the camera manually. It only happens automatically if we define the camera sensor in the environment config
            self.camera._initialize_impl() 
            self.camera._is_initialized = True

        self._data.output["height_map"] = torch.zeros(
            (self._num_envs, self.camera_cfg.height, self.camera_cfg.width), 
            device=self.cfg.device
        )

        # Check that sensor has been spawned
        # if sensor_prim_path is None:
        #     if not self._is_spawned:
        #         raise RuntimeError("Initializing the camera failed! Please provide a valid argument for `prim_path`.")
        #     sensor_prim_path = self.prim_path


        # initialize classes for GelSight simulation approaches for simulating GelSight sensor output
        if self.cfg.optical_sim_cfg is not None:
            # initialize class we set in the cfg class of the sim approach
            self.optical_simulator = self.cfg.optical_sim_cfg.simulation_approach_class(
                sensor = self,
                cfg = self.cfg.optical_sim_cfg,
            )
            
        if self.cfg.marker_motion_sim_cfg is not None:
            if ((self.optical_simulator is not None) and 
                (self.cfg.optical_sim_cfg.simulation_approach_class == self.cfg.marker_motion_sim_cfg.simulation_approach_class)):
                # if same class for optical and marker sim, then use same obj
                self.marker_motion_simulator = self.optical_simulator
            else:
                self.marker_motion_simulator = self.cfg.marker_motion_sim_cfg.simulation_approach_class(
                    sensor = self,
                    cfg = self.cfg.marker_motion_sim_cfg
                )
        
        # create buffers for output
        if "tactile_rgb" in self._data.output.keys():
            self._data.output["tactile_rgb"] = torch.zeros(
                (self._num_envs, 3, self.cfg.optical_sim_cfg.tactile_img_res[0], self.cfg.optical_sim_cfg.tactile_img_res[1]), 
                device=self.cfg.device
            )

        if "marker_motion" in self._data.output.keys():
            self._data.output["marker_motion"]= torch.zeros(
                (
                    self._num_envs,
                    self.cfg.marker_motion_sim_cfg.marker_params.num_markers_row, 
                    self.cfg.marker_motion_sim_cfg.marker_params.num_markers_col,
                    2 # two, because each marker at (row,col) has position value (x,y)
                ), 
                device=self.cfg.device
            )

        # set how the indentation depth should be computed
        if self.cfg.compute_indentation_depth_class == "optical_sim":
            self.compute_indentation_depth_func = self.optical_simulator.compute_indentation_depth
        else:
            self.compute_indentation_depth_func = self.marker_motion_simulator.compute_indentation_depth

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # reset internal buffers
        self.reset()

        #todo print init data

    #MARK: _update_buffers_impl
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Updates the internal buffer with the latest data from the sensor.

        This function reads ...

        """
        # -- pose
        # self._data.position = self._sensor_prim.GetAttribute("xformOp:translate").Get()
        # self._data.orientation = self._sensor_prim.GetAttribute(
        #     "xformOp:rotation"
        # ).Get()

        self._frame[env_ids] += 1

        # -- update camera buffer
        if self.camera is not None:
            self.camera.update(dt=0.1)

        # -- height_map
        self._get_height_map()
        
        # -- pressing depth
        self._indentation_depth[:] = self.compute_indentation_depth_func() # type: ignore

        if (self.optical_simulator is not None) and ("tactile_rgb" in self._data.output.keys()) :
            # self.optical_simulator.height_map = self._data.output["height_map"]
            self._data.output["tactile_rgb"][:] = self.optical_simulator.optical_simulation()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self._data.output.keys()):
            self._data.output["marker_motion"][:] = self.marker_motion_simulator.marker_motion_simulation()


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
            if not hasattr(self, "_windows"):
                # list of windows that show the simulated tactile images, if the attribute of the Sensor asset is turned on
                self._windows = {}
                self._img_providers = {}
        else:
            # if hasattr(self, "frame_visualizer"):
            #     self.frame_visualizer.set_visibility(False)
            pass

    def _debug_vis_callback(self, event):
        if not self._windows:
            return
        
        # Update the GUI windows
        for i, prim in enumerate(self._view.prims):
            # creates an attribut, which can be found in the GUI under "Raw Usd Properties -> "Extra Properties"
            show_img = prim.GetAttribute("show_tactile_image").Get()
            if show_img==True:
                if not (str(i) in self._windows.keys()):
                    # create a window
                    #window = omni.ui.Window(self._view.prim_paths[i], width=self.cfg.resolution[1], height=self.cfg.resolution[0]) # +5 to have a small border around the tactile images   
                    window = omni.ui.Window(self._view.prim_paths[i], auto_resize=True)
                    self._windows[str(i)] = window
                    # create image provider
                    self._img_providers[str(i)] = omni.ui.ByteImageProvider() # default format omni.ui.TextureFormat.RGBA8_UNORM

                if "tactile_rgb" in self._data.output.keys():
                    # get tactile image
                    frame = self.data.output["tactile_rgb"][i].permute(1, 2, 0).cpu().numpy()
                    frame = cv2.normalize(frame, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(f"tact_rgb{self._frame[i]}.jpg", frame)
                    frame = frame.astype(np.uint8)
                
                if "marker_motion" in self._data.output.keys():
                    if not "tactile_rgb" in self._data.output.keys():
                        frame = np.zeros(self.marker_motion_simulator.cfg.tactile_img_res).astype(np.uint8) 

                    # like the `_generate` function of FOTS MarkerMotion sim
                    marker_data = self.data.output["marker_motion"][i]
                    # position values are in pix
                    x_pos_of_all_markers = marker_data[:,:,0].cpu().numpy() # = columns, shape (num_markers_row, num_markers_col)
                    y_pos_of_all_markers  = marker_data[:,:,1].cpu().numpy() # = row
                    color = (255, 255, 255)

                    num_markers_row = x_pos_of_all_markers.shape[0]
                    num_markers_col = x_pos_of_all_markers.shape[1]
                    for k in range(num_markers_col):
                        for j in range(num_markers_row):
                            init_x_pos = int(self.marker_motion_simulator.init_marker_pos[0][j,k])    # get initial x position of marker [j,k]
                            init_y_pos = int(self.marker_motion_simulator.init_marker_pos[1][j,k]) # get initial y position of marker [j,k]
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
                    #frame = cv2.normalize(frame, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)                

                #TODO remove this tmp workaround for different res
                # frame = cv2.resize(frame, (120, 160)) #(self._sensor_params.height, self._sensor_params.width)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA) #cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                height, width, channels = frame.shape

                # update image of the window
                with self._windows[str(i)].frame:
                    self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                    image = omni.ui.ImageWithProvider(self._img_providers[str(i)], width=width, height=height) #, fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size

            elif str(i) in self._windows.keys():
                # remove window/img_provider from dictionary and destroy them
                self._windows.pop(str(i)).destroy()
                self._img_providers.pop(str(i)).destroy()
    """ 
    Private Helper methods
    """
    #TODO implement
    # def _create_buffers(self):
    #     """Create buffers for storing data."""
    #     # create the data object
    #     # -- pose of the cameras
    #     self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
    #     self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
    #     # -- intrinsic matrix
    #     self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
    #     self._data.image_shape = self.image_shape
    #     # -- output data
    #     # lazy allocation of data dictionary
    #     # since the size of the output data is not known in advance, we leave it as None
    #     # the memory will be allocated when the buffer() function is called for the first time.
    #     self._data.output = TensorDict({}, batch_size=self._view.count, device=self.device)
    #     self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._view.count)]

    #TODO implement properly
    # def _update_poses(self, env_ids: Sequence[int]):
    #     """Computes the pose of the camera in the world frame with ROS convention.

    #     This methods uses the ROS convention to resolve the input pose. In this convention,
    #     we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

    #     Returns:
    #         A tuple of the position (in meters) and quaternion (w, x, y, z).
    #     """
    #     # check camera prim exists
    #     if len(self._sensor_prims) == 0:
    #         raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

    #     # get the poses from the view
    #     poses, quat = self._view.get_world_poses(env_ids)
    #     self._data.pos_w[env_ids] = poses
    #     self._data.quat_w_world[env_ids] = convert_orientation_convention(quat, origin="opengl", target="world")
    

    def _get_height_map(self):
        if self.camera is not None:
            self._data.output["height_map"][:] = self.camera.data.output["depth"][:,:,:,0] # tiled camera gives us data with shape (num_cameras, height, width, num_channels),
            # clip camera values that are = inf
            self._data.output["height_map"][torch.isinf(self._data.output["height_map"])] = self.cfg.sensor_camera_cfg.clipping_range[1]
            # default unit is meter -> convert to mm for optical sim
            self._data.output["height_map"] *= 1000
            return self._data.output["height_map"]
        else:
            # not setting camera cfg means "no need for camera"
            # e.g. use soft body deformation as height map? -> not implemented yet
            # or that we dont need a height map in general
            pass

    def _show_height_map_inside_gui(self, index):
        plt.close()
        height_map = self._data.output["height_map"][0].cpu().numpy()
        X = np.arange(0, height_map.shape[0])
        Y = np.arange(0, height_map.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = height_map
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X, Y, Z.T)
        # plt.show()
        print("saving img")
        plt.savefig(f"height_map{index}.png")
    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None

        self.camera._invalidate_initialize_callbacks()
        self.camera.__del__()

        if hasattr(self, "_windows"):
            self._windows = None
            self._img_providers = None
        