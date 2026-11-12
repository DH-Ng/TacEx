from __future__ import annotations
from source.tacex.tacex.simulation_approaches.gelsight_simulator import GelSightSimulator

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import logging

import cv2
import omni.kit.commands
import omni.usd
import isaacsim.core.utils.prims as prims_utils
from pxr import Sdf

from isaaclab.sensors import SensorBase, TiledCamera, TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView

from .gelsight_sensor_data import GelSightSensorData
from .simulation_approaches.gelsight_simulator import GelSightSimulator

# from torchvision.transforms import v2


# from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
# from isaaclab.sensors.camera.camera import Camera
# from isaaclab.sensors.camera.camera_cfg import CameraCfg

if TYPE_CHECKING:
    from .gelsight_sensor_cfg import GelSightSensorCfg
    from .simulation_approaches.gelsight_simulator_cfg import GelSightSimulatorCfg

logger = logging.getLogger(__name__)


class GelSightSensor(SensorBase):
    cfg: GelSightSensorCfg

    def __init__(self, cfg: GelSightSensorCfg, gelpad_obj=None) -> None:
        # initialize base class
        super().__init__(cfg)
        self._check_gel_geometry()

        self._prim_view = None

        # sensor camera
        self.camera = None

        # object which represents the gelpad
        self.gelpad_obj = gelpad_obj

        self._indentation_depth: torch.tensor = None

        # simulation approaches for simulating GelSight sensor output
        self.optical_simulator: GelSightSimulator | None = None
        self.marker_motion_simulator: GelSightSimulator | None = None
        self.force_field_simulator: GelSightSimulator | None = None
        self.compute_indentation_depth_func = None

        # Create empty variables for storing output data
        self._data = GelSightSensorData()
        self._data.output = dict.fromkeys(self.cfg.data_types, None)

        # Flag to check that sensor is spawned.
        self._is_spawned = False

        # initialize classes for GelSight simulation approaches for simulating GelSight sensor output
        # (a slot whose cfg names an already created simulation class reuses that instance)
        self.optical_simulator = self._get_or_create_simulator(self.cfg.optical_sim_cfg)
        self.marker_motion_simulator = self._get_or_create_simulator(self.cfg.marker_motion_sim_cfg)
        self.force_field_simulator = self._get_or_create_simulator(self.cfg.force_field_sim_cfg)

        # set how the indentation depth should be computed
        if (self.cfg.compute_indentation_depth_class) == "optical_sim" and (self.optical_simulator is not None):
            self.compute_indentation_depth_func = self.optical_simulator.compute_indentation_depth
        elif (self.cfg.compute_indentation_depth_class == "marker_motion_sim") and (
            self.marker_motion_simulator is not None
        ):
            self.compute_indentation_depth_func = self.marker_motion_simulator.compute_indentation_depth
        else:
            # use default implementation
            self.compute_indentation_depth_func = self.compute_indentation_depth

        # check if desired output is possible with given config:
        if "tactile_rgb" in self.cfg.data_types and self.cfg.optical_sim_cfg is None:
            raise RuntimeError("Output [tactile_rgb] not possible with given config: optical_sim_cfg = None]")

        if "marker_motion" in self.cfg.data_types and self.cfg.marker_motion_sim_cfg is None:
            raise RuntimeError("Output [marker_motion] not possible with given config: marker_motion = None")

        if "force_field" in self.cfg.data_types and self.cfg.force_field_sim_cfg is None:
            raise RuntimeError("Output [force_field] not possible with given config: force_field_sim_cfg = None")

        self._check_camera_aspect_ratio()

        self._set_debug_vis_flag = False
        self._debug_vis_is_initialized = False

    def _get_or_create_simulator(self, sim_cfg: GelSightSimulatorCfg | None) -> GelSightSimulator | None:
        """Returns the simulator for a cfg slot, reusing an existing instance of the same simulation class."""
        if sim_cfg is None:
            return None
        for simulator in (self.optical_simulator, self.marker_motion_simulator, self.force_field_simulator):
            if simulator is not None and type(simulator) is sim_cfg.simulation_approach_class:
                return simulator
        return sim_cfg.simulation_approach_class(sensor=self, cfg=sim_cfg)

    @property
    def simulators(self) -> list[GelSightSimulator]:
        """The distinct simulators of this sensor (a simulator used for several outputs appears once)."""
        unique_simulator = []
        for simulator in (self.optical_simulator, self.marker_motion_simulator, self.force_field_simulator):
            if simulator is not None and all(simulator is not s for s in unique_simulator):
                unique_simulator.append(simulator)
        return unique_simulator

    def _check_camera_aspect_ratio(self):
        """Ensures the sensor camera has the aspect ratio of the real tactile image (``reference_img_res``).

        The renderer uses square pixels: with another aspect ratio the view height follows from the view width and
        the resolution aspect, so the camera would not see the configured physical area (``view_height_m``).
        A different output shape is fine, but has to be set via the simulation approach (e.g. ``tactile_img_res``),
        which resizes the rendered height map.
        """
        cam_w, cam_h = self.cfg.sensor_camera_cfg.resolution
        ref_w, ref_h = self.cfg.sensor_camera_cfg.reference_img_res
        expected_h = cam_w * ref_h / ref_w
        if abs(cam_h - expected_h) > 1.0:  # tolerance for integer rounding, e.g. 43x32 for 4:3
            raise ValueError(
                f"Sensor camera resolution {cam_w}x{cam_h} must have the aspect ratio of reference_img_res"
                f" {ref_w}x{ref_h} (expected height ~{expected_h:.1f} px for width {cam_w}). Otherwise the camera"
                " does not see the configured physical view. To get a different output shape, set the resolution"
                " of the simulation approach (e.g. tactile_img_res) instead."
            )

    def __del__(self):
        """Unsubscribes from callbacks."""
        # unsubscribe callbacks
        super().__del__()

    # def __str__(self) -> str:
    #     """Returns: A string containing information about the instance."""
    #     # message for class
    #     return (
    #         f"Gelsight Mini @ '{self.cfg.prim_path}': \n"
    #         f"\tdata types   : {list(self._data.output)} \n"
    #         f"\tupdate period (s): {self.cfg.update_period}\n"
    #         f"\tframe        : {self.frame}\n"
    #         f"\camera resolution        : {self.camera_resolution}\n"
    #         f"\ttactile resolution        : {self.tactile_image_resolution}\n"
    #         f"\twith shadows        : {self._simulate_shadows}\n"
    #         # f"\tposition     : {self._data.position} \n"
    #         # f"\torientation  : {self._data.orientation} \n"
    #     )

    """
    Properties
    """

    @property
    def data(self) -> GelSightSensorData:
        """Data related to Camera sensor."""
        # update sensors if needed
        self._update_outdated_buffers()
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def tactile_image_shape(self) -> tuple[int, int, int]:
        """Shape of the simulated tactile RGB image, i.e. (height, width, channels)."""
        return (self.cfg.optical_sim_cfg.tactile_img_res[1], self.cfg.optical_sim_cfg.tactile_img_res[0], 3)

    @property
    def camera_resolution(self) -> tuple[int, int]:
        """The resolution (width x height) of the camera used by this sensor."""
        return self.cfg.sensor_camera_cfg.resolution[0], self.cfg.sensor_camera_cfg.resolution[1]  # type: ignore

    @property
    def indentation_depth(self):
        """How deep objects are inside the gel pad of the sensor.

        Units are in millimeter.
        """
        return self._indentation_depth

    @property
    def prim_view(self):
        return self._prim_view

    """
    Operations
    """

    # MARK: reset
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES  # type: ignore

        # reset camera
        if self.camera is not None:
            self.camera.reset()

        # reset the buffer
        # self._data.position = None
        # self._data.orientation = None
        # self._data.image_resolution = self.image_resolution

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

        if "camera_depth" in self.cfg.data_types:
            self._data.output["camera_depth"][env_ids] = 0

        # reset every simulator once (one instance can serve several outputs). Each writes the no-contact state into
        # the env_ids rows of its output buffers (no re-simulation, other envs stay untouched).
        for simulator in self.simulators:
            simulator.reset(env_ids)

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self.cfg.data_types):
            # (yy_init_pos, xx_init_pos), i.e. along height x width of tactile img
            self._data.output["init_marker_pos"] = ([0], [0])

        # Reset the frame count
        self._frame[env_ids] = 0

    def _requested_simulated_outputs(self) -> list[tuple[str, GelSightSimulator]]:
        """Requested outputs that a simulation approach produces, as ``(data_type, simulator)`` pairs.

        ``data_type`` is the output's name (``tactile_rgb``, ``marker_motion`` or ``force_field``, if listed in
        ``cfg.data_types``), not its buffer; the buffers are in ``self._data.output[data_type]``. Camera-based outputs
        (``height_map``, ``camera_depth``, ``camera_rgb``) are not included.
        """
        outputs = []
        for data_type, simulator in (
            ("tactile_rgb", self.optical_simulator),
            ("marker_motion", self.marker_motion_simulator),
            ("force_field", self.force_field_simulator),
        ):
            if simulator is not None and data_type in self.cfg.data_types:
                outputs.append((data_type, simulator))
        return outputs

    def _allocate_simulator_output(self, data_type: str, simulator: GelSightSimulator):
        """Allocates the output buffer of ``data_type`` on the sensor's device and hands it to its simulator."""
        shape = simulator.get_output_shape(data_type)
        if isinstance(shape, dict):  # e.g. force_field: one tensor per key
            buffer = {key: torch.zeros((self._num_envs, *s), device=self._device) for key, s in shape.items()}
        else:
            buffer = torch.zeros((self._num_envs, *shape), device=self._device)
        self._data.output[data_type] = buffer
        simulator.set_output_buffer(data_type, buffer)

    def compute_indentation_depth(self):
        """Computes the indentation depth by taking the minimum value of the height map.

        The height map format is as follow:
        - value 0 is at the gelpad's highest point (= no indentation)
        - value min(height_map) is the point closest to sensor camera, i.e. the point with the biggest indentation depth
        """
        height_map = self._data.output["height_map"]
        self._indentation_depth[:] = -height_map.amin((1, 2))
        return self._indentation_depth

    ####
    # Implementation of abstract methods of base sensor class
    ####
    # MARK: _init_impl
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers."""
        print(f"Initializing GelSight Sensor `{self.cfg.prim_path}`...")

        # Initialize parent class
        super()._initialize_impl()

        # NOTE: use Isaac Lab's view, not isaacsim.core.prims.XFormPrim. The Isaac Sim view registers a prim-deletion
        # callback that cannot be deregistered; once the view is dropped, the next prim deletion (e.g. the camera's
        # annotator detach on env close) hits a dead weakref and Kit terminates.
        self._prim_view = XformPrimView(
            self.cfg.prim_path, device=self._device, validate_xform_ops=False, stage=self.stage
        )
        # Check that sizes are correct
        if self._prim_view.count != self._num_envs:
            raise RuntimeError(
                f"Number of sensor prims in the view ({self._prim_view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # set device, if specified (per default the same as the simulation)
        if self.cfg.device is not None:
            self._device = self.cfg.device

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)

        # need to initialize the camera manually, since its not part of the scene cfg
        self.camera = self._create_tiled_camera()
        self.camera._initialize_impl()
        self.camera._is_initialized = True

        self._data.output["height_map"] = torch.zeros(
            (self._num_envs, self.camera_cfg.height, self.camera_cfg.width), device=self.cfg.device
        )

        # Check that sensor has been spawned
        # if sensor_prim_path is None:
        #     if not self._is_spawned:
        #         raise RuntimeError("Initializing the camera failed! Please provide a valid argument for `prim_path`.")
        #     sensor_prim_path = self.prim_path

        # create buffers for output
        if "camera_depth" in self.cfg.data_types:
            self._data.output["camera_depth"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1), device=self.cfg.device
            )
        if "camera_rgb" in self.cfg.data_types:
            self._data.output["camera_rgb"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 3), device=self.cfg.device
            )
        # Initialize sim approaches
        # NOTE: important, that this happens after the height map buffer was created, because the simulators read it.
        # Each simulator is initialized once, even if it serves several outputs.
        for simulator in self.simulators:
            simulator._initialize_impl()

        # The sensor owns the simulator outputs: allocate them on the sensor's device (shapes from the simulators, which
        # may depend on their initialization, e.g. the number of force field points) and hand them over. The
        # simulators write into them in place; the no-contact state is written by reset() below.
        for data_type, simulator in self._requested_simulated_outputs():
            self._allocate_simulator_output(data_type, simulator)

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Reset internal buffers
        self.reset()

        # Create debug visualization
        self._initialize_debug_vis(self._initialize_debug_vis_flag)

        # todo print init data
        # print(self)

    # MARK: _update_buffers_impl
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

        # -- update camera data
        if self.camera is not None:
            self.camera._timestamp = self._timestamp
            self.camera.update(dt=0.0, force_recompute=True)  # time comes from the parent sensor (synced above)

            if "camera_depth" in self.cfg.data_types:
                self._get_camera_depth()

            if "camera_rgb" in self.cfg.data_types:
                self._data.output["camera_rgb"][:] = self.camera.data.output["rgb"]

        if self.compute_indentation_depth_func is not None:
            # get height map and indentation depth data
            self._get_height_map()
            self._indentation_depth[:] = self.compute_indentation_depth_func()

        # the simulators write into the sensor's output buffers in place (see _allocate_simulator_output)
        if (self.optical_simulator is not None) and ("tactile_rgb" in self.cfg.data_types):
            self.optical_simulator.optical_simulation()

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self.cfg.data_types):
            self.marker_motion_simulator.marker_motion_simulation()

        if (self.force_field_simulator is not None) and ("force_field" in self.cfg.data_types):
            self.force_field_simulator.force_field_simulation()

    def _simulators_with_requested_outputs(self) -> list[GelSightSimulator]:
        """Distinct simulators that produce at least one of the requested outputs (for debug visualization)."""
        # dict keeps the order and deduplicates by identity (simulators do not override __eq__/__hash__)
        return list[GelSightSimulator](dict.fromkeys(simulator for _, simulator in self._requested_simulated_outputs()))

    def _set_debug_vis_impl(self, debug_vis: bool):
        # we actually set the debug_vis in _initialize_impl, since we need the _prim_view, which
        # is only correctly initialized after _initialize_impl method (thats the case in ManagerBased workflow - in Direct workflow you can control it yourself)
        self._initialize_debug_vis_flag: bool = debug_vis

    def _initialize_debug_vis(self, debug_vis: bool):
        """Creates an USD attribute for the sensor assets, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.

        > Method has to be called after the prim_view was initialized.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # need to create the attribute for the debug_vis here since it depends on self._prim_view
            for prim in self._prim_view.prims:
                # creates USD attributes for each data type, which can be found in the Isaac GUI under "Raw Usd Properties -> "Extra Properties"
                if "camera_depth" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_depth", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "camera_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "height_map" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_height_map", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)

                if "tactile_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_tactile_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "marker_motion" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_marker_motion", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "force_field" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_force_field", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)

            if not hasattr(self, "_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._windows = {}
                self._img_providers = {}
                # todo check if there is a more efficient implementation than dict of dicts
                if "camera_depth" in self.cfg.data_types:
                    self._windows["camera_depth"] = {}
                    self._img_providers["camera_depth"] = {}
                if "camera_rgb" in self.cfg.data_types:
                    self._windows["camera_rgb"] = {}
                    self._img_providers["camera_rgb"] = {}
                if "height_map" in self.cfg.data_types:
                    self._windows["height_map"] = {}
                    self._img_providers["height_map"] = {}

            for simulator in self._simulators_with_requested_outputs():
                simulator._set_debug_vis_impl(debug_vis)

            self._debug_vis_is_initialized = True

    def _debug_vis_callback(self, event):
        if not self._debug_vis_is_initialized:
            return

        # Update the GUI windows
        for i, prim in enumerate(
            self._prim_view.prims
        ):  # note: bad that we iterate through all prims multiple times (once per sim data type)
            if "camera_rgb" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_rgb").Get()
                if show_img:
                    if str(i) not in self._windows["camera_rgb"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_rgb",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_rgb"][str(i)] = window
                        # create image provider
                        self._img_providers["camera_rgb"][str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["camera_rgb"][i].cpu().numpy()

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape
                    with self._windows["camera_rgb"][str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._img_providers["camera_rgb"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_rgb"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["camera_rgb"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["camera_rgb"].pop(str(i)).destroy()
                    self._img_providers["camera_rgb"].pop(str(i)).destroy()

            if "camera_depth" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_depth").Get()
                if show_img:
                    if str(i) not in self._windows["camera_depth"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_depth",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_depth"][str(i)] = window
                        # create image provider
                        self._img_providers["camera_depth"][str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["camera_depth"][i].cpu().numpy()
                    # # image is channel first, convert to channel last
                    # frame = np.moveaxis(frame, 0, -1)
                    # convert to 3 channel image, to later turn it into 4 channel RGBA for Isaac Widget
                    frame = np.dstack((frame, frame, frame)).astype(np.uint8)
                    # update image of the window
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape
                    with self._windows["camera_depth"][str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._img_providers["camera_depth"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_depth"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["camera_depth"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["camera_depth"].pop(str(i)).destroy()
                    self._img_providers["camera_depth"].pop(str(i)).destroy()

            if "height_map" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_height_map").Get()
                if show_img:
                    if str(i) not in self._windows["height_map"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/height_map",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["height_map"][str(i)] = window
                        # create image provider
                        self._img_providers["height_map"][str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["height_map"][i].cpu().numpy()
                    # map [-gelpad_height, 0] to [0,255] for proper depth image of height_map
                    frame = (
                        (
                            np.clip(frame, -self.cfg.max_indentation_depth * 1000, 0)
                            + self.cfg.max_indentation_depth * 1000
                        )
                        / (self.cfg.max_indentation_depth * 1000)
                        * 255
                    ).astype(np.uint8)
                    # convert to 3 channel image, to later turn it into 4 channel RGBA for Isaac Widget
                    frame = np.dstack((frame, frame, frame)).astype(np.uint8)

                    # update image of the window
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape
                    with self._windows["height_map"][str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._img_providers["height_map"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["height_map"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["height_map"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["height_map"].pop(str(i)).destroy()
                    self._img_providers["height_map"].pop(str(i)).destroy()

        for simulator in self._simulators_with_requested_outputs():
            simulator._debug_vis_callback(event)

    """
    Private Helper methods
    """
    # TODO implement
    # def _create_buffers(self):
    #     """Create buffers for storing data."""
    #     # create the data object
    #     # -- pose of the cameras
    #     self._data.pos_w = torch.zeros((self._prim_view.count, 3), device=self._device)
    #     self._data.quat_w_world = torch.zeros((self._prim_view.count, 4), device=self._device)
    #     # -- intrinsic matrix
    #     self._data.intrinsic_matrices = torch.zeros((self._prim_view.count, 3, 3), device=self._device)
    #     self._data.image_shape = self.image_shape
    #     # -- output data
    #     # lazy allocation of data dictionary
    #     # since the size of the output data is not known in advance, we leave it as None
    #     # the memory will be allocated when the buffer() function is called for the first time.
    #     self._data.output = TensorDict({}, batch_size=self._prim_view.count, device=self.device)
    #     self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._prim_view.count)]

    # TODO implement properly
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
    #     poses, quat = self._prim_view.get_world_poses(env_ids)
    #     self._data.pos_w[env_ids] = poses
    #     self._data.quat_w_world[env_ids] = convert_orientation_convention(quat, origin="opengl", target="world")

    def _get_camera_depth(self):
        if self.camera is not None:
            depth_output = self.camera.data.output["depth"][
                :, :, :, 0
            ]  # tiled camera gives us data with shape (num_cameras, height, width, num_channels),
            # clip camera values that are = inf
            depth_output[torch.isinf(depth_output)] = self.cfg.sensor_camera_cfg.clipping_range_m[1]

            # add a channel to the depth image for debug_vis
            self._data.output["camera_depth"] = depth_output[:, :, :, None]

            # normalize the depth image in clipping range
            normalized = (self._data.output["camera_depth"] - self.cfg.sensor_camera_cfg.clipping_range_m[0]) / (
                self.cfg.sensor_camera_cfg.clipping_range_m[1] - self.cfg.sensor_camera_cfg.clipping_range_m[0]
            )
            normalized = (normalized * 255).type(dtype=torch.uint8)
            self._data.output["camera_depth"] = normalized

            # self._data.output["camera_depth"] *= 1000.0

        return self._data.output["camera_depth"]

    def _get_height_map(self):
        if self.camera is not None:
            self._data.output["height_map"][:] = self.camera.data.output["depth"][
                :, :, :, 0
            ]  # tiled camera gives us data with shape (num_cameras, height, width, num_channels),
            # clip camera values that are = inf
            self._data.output["height_map"][torch.isinf(self._data.output["height_map"])] = (
                self.cfg.sensor_camera_cfg.clipping_range_m[1]
            )
            # depth relative to the undeformed gel surface: 0 = no indentation, min(height_map) = biggest indentation.
            # Two subtractions (underside, then gelpad height) instead of one by get_gel_surface_to_camera_distance_m():
            # same value, but float32 rounds the sub-micrometer depth noise of the flat gel differently, and Taxim
            # amplifies that into visible background texture -> keeps the output bit-identical to earlier versions.
            self._data.output["height_map"] -= self.cfg.gelpad_underside_to_camera_distance_m
            self._data.output["height_map"] -= self.cfg.gelpad_dimensions.height

            self._data.output["height_map"] = self._data.output["height_map"].clamp(
                -self.cfg.gelpad_dimensions.height, 0
            )

            # default unit is meter -> convert to mm for optical sim
            self._data.output["height_map"] *= 1000

            return self._data.output["height_map"]
        else:
            # not setting camera cfg means "no need for camera"
            # e.g. use soft body deformation as height map? -> not implemented yet
            # or that we dont need a height map in general
            pass

    def _check_gel_geometry(self):
        """Warns about sensor camera settings that do not match the gel geometry."""
        cam_cfg = self.cfg.sensor_camera_cfg
        if cam_cfg is None:
            return
        gel_surface = self.cfg.get_gel_surface_to_camera_distance_m()
        if cam_cfg.clipping_range_m[1] < gel_surface - 1e-6:
            logger.warning(
                f"Sensor camera far clipping plane ({cam_cfg.clipping_range_m[1]} m) is in front of the gel surface"
                f" ({gel_surface} m = gelpad_underside_to_camera_distance_m + gelpad height): the undeformed gel"
                " reads as indented."
            )

    def _create_tiled_camera(self) -> TiledCamera:
        prim_path = self.cfg.prim_path + f"/{self.cfg.sensor_camera_cfg.prim_name}"
        # delete existing camera prim
        # sim_utils.delete_prim(prim_path)

        if self.cfg.sensor_camera_cfg.update_period != self.cfg.update_period:
            logger.warning(
                f"Sensor camera configuration update period ({self.cfg.sensor_camera_cfg.update_period}) is not equal to sensor"
                f" update period ({self.cfg.update_period}), changing camera update period to match sensor update perod."
            )
            self.cfg.sensor_camera_cfg.update_period = self.cfg.update_period

        # The physical width/height of the viewable area at the undeformed gel surface (in meters).
        view_width_m = self.cfg.sensor_camera_cfg.view_width_m
        view_height_m = self.cfg.sensor_camera_cfg.view_height_m
        # The view is defined at the gel surface (mm_per_pixel is measured on real images of it), so the (undeformed)
        # gel surface is the focus distance. With f_stop = 0 (PinholeCameraCfg default) focusing is off: the USD
        # focus distance does not change the image, it is only the reference distance for the aperture calculation.
        focus_distance_m = self.cfg.get_gel_surface_to_camera_distance_m()

        focal_length = self.cfg.sensor_camera_cfg.focal_length_cm

        # Calculate aperture using the pinhole camera ratio:
        # aperture / focal_length = view_width / focus_distance
        horizontal_aperture = (view_width_m * focal_length) / focus_distance_m
        vertical_aperture = (view_height_m * focal_length) / focus_distance_m

        self.camera_cfg: TiledCameraCfg = TiledCameraCfg(
            prim_path=prim_path,
            update_period=self.cfg.sensor_camera_cfg.update_period,
            offset=TiledCameraCfg.OffsetCfg(
                pos=self.cfg.sensor_camera_cfg.camera_pos_offset,
                rot=self.cfg.sensor_camera_cfg.camera_rot_offset,
                convention="opengl",
            ),  # need to use opengl convention, since IsaacLab converts it into the opengl convention for the camera -> x and y are switched
            width=self.cfg.sensor_camera_cfg.resolution[0],
            height=self.cfg.sensor_camera_cfg.resolution[1],
            data_types=self.cfg.sensor_camera_cfg.data_types,
            update_latest_camera_pose=self.cfg.sensor_camera_cfg.update_latest_camera_pose,  # True is needed for FEM based marker sim
            spawn=sim_utils.PinholeCameraCfg(
                # use user-defined focal length to set proper zoom factor
                focal_length=focal_length,
                focus_distance=focus_distance_m,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture,
                clipping_range=self.cfg.sensor_camera_cfg.clipping_range_m,
            ),
            depth_clipping_behavior="max",
        )
        return TiledCamera(cfg=self.camera_cfg)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._prim_view = None

        if self.camera is not None:
            self.camera = None

        if hasattr(self, "_windows"):
            self._windows = None
            self._img_providers = None
