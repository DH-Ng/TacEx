"""TacSL simulation approach: wraps Isaac Lab's TacSL visuo-tactile sensor for TacEx.

Two outputs, each enabled by the GelSight sensor's cfg slots and data types:

- optical (``"tactile_rgb"``): Isaac Lab's ``GelsightRender`` (Taxim-style polynomial lookup) applied to the sensor's
  height map. TacSL itself renders the difference between a "nominal" (no-contact) depth image and the current one;
  TacEx's height map already is that difference (0 = undeformed gelpad, negative = pressed in), so no nominal
  render is needed. Note: Looks more like that Tacto than Taxim, because no Gaussian Kernel blur.
- force field (``"force_field"``): Isaac Lab's SDF-based penalty forces at a grid of points on the gelpad surface.
  The Isaac Lab code is reused through ``_TacSLForceField``, a ``VisuoTactileSensor`` subclass that is not
  registered as a sensor (no callbacks, no camera); only the tactile point generation is replaced to support
  TacEx gelpad assets.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import trimesh
from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaacsim.core.simulation_manager import SimulationManager

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorData
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import GelsightRender

from ..gelsight_simulator import GelSightSimulator

if TYPE_CHECKING:
    from ...gelsight_sensor import GelSightSensor
    from .tacsl_sim_cfg import TacSLForceFieldCfg, TacSLSimulatorCfg

logger = logging.getLogger(__name__)


class TacSLSimulator(GelSightSimulator):
    """TacSL optical and force field simulation for GelSight sensors (see module docstring)."""

    cfg: TacSLSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TacSLSimulatorCfg):
        super().__init__(sensor=sensor, cfg=cfg)

        self._renderer: GelsightRender | None = None
        self._render_res: tuple[int, int] | None = None  # (width, height) of the calibration
        self._rest_img: torch.Tensor | None = None  # no-contact tactile image (height, width, 3)

        self._force_field: _TacSLForceField | None = None

        self._debug_windows: dict = {}
        self._debug_img_providers: dict = {}

    """
    Which outputs this instance provides (decided by the sensor's cfg slots and data types).
    """

    @property
    def optical_enabled(self) -> bool:
        return self.sensor.optical_simulator is self and "tactile_rgb" in self.sensor.cfg.data_types

    @property
    def force_field_enabled(self) -> bool:
        return self.sensor.force_field_simulator is self and "force_field" in self.sensor.cfg.data_types

    """
    Initialization.
    """

    def _initialize_impl(self):
        self._device = self.sensor._device if self.cfg.device is None else self.cfg.device
        self._num_envs = self.sensor._num_envs

        if self.optical_enabled:
            self._initialize_optical()
        if self.force_field_enabled:
            self._initialize_force_field()

    def _initialize_optical(self):
        render_cfg = self.cfg.render_cfg
        self._render_res = (render_cfg.image_width, render_cfg.image_height)

        if self.cfg.mm_per_pixel_from_sensor:
            cam_cfg = self.sensor.cfg.sensor_camera_cfg
            mm_per_px_w = cam_cfg.view_width_m * 1000 / render_cfg.image_width
            mm_per_px_h = cam_cfg.view_height_m * 1000 / render_cfg.image_height
            if abs(mm_per_px_w / mm_per_px_h - 1.0) > 0.01:
                logger.warning(
                    f"TacSL calibration resolution {render_cfg.image_width}x{render_cfg.image_height} does not have"
                    f" the aspect ratio of the sensor view ({cam_cfg.view_width_m * 1000:.2f} x"
                    f" {cam_cfg.view_height_m * 1000:.2f} mm); using the width scale {mm_per_px_w:.4f} mm/px."
                )
            if abs(mm_per_px_w / render_cfg.mm_per_pixel - 1.0) > 0.1:
                logger.warning(
                    f"TacSL render scale from the sensor view ({mm_per_px_w:.4f} mm/px) differs by more than 10% from"
                    f" the calibration's mm_per_pixel ({render_cfg.mm_per_pixel:.4f})."
                )
            render_cfg = render_cfg.replace(mm_per_pixel=mm_per_px_w)

        self._renderer = GelsightRender(render_cfg, device=self._device)

        if self.cfg.background_override_path is not None:
            background = _load_background(self.cfg.background_override_path)
            if background.shape != (render_cfg.image_height, render_cfg.image_width, 3):
                raise ValueError(
                    f"Background {self.cfg.background_override_path} has shape {background.shape}, expected the"
                    f" calibration resolution ({render_cfg.image_height}, {render_cfg.image_width}, 3)."
                )
            self._renderer.background_tensor = torch.tensor(background, device=self._device)

        # no-contact image (written into the output by reset): the rendering of an undeformed gel, i.e. a zero height
        # map at the camera resolution (rendering is independent per env, so one env suffices)
        cam_h, cam_w = self.sensor._data.output["height_map"].shape[1:]
        self._rest_img = self._render(torch.zeros((1, cam_h, cam_w), device=self.sensor._device))[0]

    def _initialize_force_field(self):
        ff_cfg = self.cfg.force_field
        if ff_cfg.elastomer_prim_path_expr is None:
            raise ValueError(
                "TacSL force field: set force_field.elastomer_prim_path_expr (the gelpad rigid body) to use the"
                " 'force_field' output."
            )
        elastomer_expr = self._resolve_env_regex_ns(ff_cfg.elastomer_prim_path_expr)
        contact_expr = self._resolve_env_regex_ns(ff_cfg.contact_object_prim_path_expr)

        elastomer_prims = sim_utils.find_matching_prims(elastomer_expr)
        if len(elastomer_prims) != self._num_envs:
            raise RuntimeError(
                f"TacSL force field: found {len(elastomer_prims)} elastomer prims for '{elastomer_expr}', expected"
                f" {self._num_envs} (one per env)."
            )
        if not elastomer_prims[0].HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"TacSL force field: elastomer prim '{elastomer_prims[0].GetPath()}' must be a rigid body, since"
                " its pose and velocity are read from the physics simulation."
            )

        self._force_field = _TacSLForceField(
            cfg=ff_cfg.replace(elastomer_prim_path_expr=elastomer_expr, contact_object_prim_path_expr=contact_expr),
            elastomer_prim=elastomer_prims[0],
            num_envs=self._num_envs,
            device=self._device,
            physics_sim_view=SimulationManager.get_physics_sim_view(),
        )
        self._force_field._initialize_force_field()

    def get_output_shape(self, data_type: str) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        if data_type == "tactile_rgb":
            w, h = self.cfg.tactile_img_res
            return (h, w, 3)
        if data_type == "force_field":
            num_pts = self._force_field.num_tactile_points
            return {
                "points_pos_w": (num_pts, 3),
                "penetration_depth": (num_pts,),
                "normal_force": (num_pts,),
                "shear_force": (num_pts, 2),
            }
        return super().get_output_shape(data_type)

    def _resolve_env_regex_ns(self, expr: str | None) -> str | None:
        """Replaces ``{ENV_REGEX_NS}`` (not resolved by the scene inside simulator cfgs) using the sensor prim path."""
        if expr is None or "{ENV_REGEX_NS}" not in expr:
            return expr
        match = re.match(r"^(.*?/env_\.\*)", self.sensor.cfg.prim_path)
        if match is None:
            raise ValueError(
                f"Cannot resolve '{{ENV_REGEX_NS}}' in '{expr}': the sensor prim path '{self.sensor.cfg.prim_path}'"
                " contains no 'env_.*'. Use the full prim path expression instead."
            )
        return expr.format(ENV_REGEX_NS=match.group(1))

    """
    Simulation.
    """

    def optical_simulation(self):
        """Renders the tactile RGB images into the sensor's ``tactile_rgb`` buffer (num_envs, height, width, 3),
        values in [0, 1]."""
        out = self._output_buffers["tactile_rgb"]
        out[:] = self._render(self.sensor._data.output["height_map"]).to(out.device)

    def _render(self, height_map: torch.Tensor) -> torch.Tensor:
        """Tactile images (N, height, width, 3) in [0, 1] for height maps (N, H, W) in mm (0 = undeformed, < 0 =
        pressed in)."""
        render_w, render_h = self._render_res
        if tuple(height_map.shape[1:]) != (render_h, render_w):
            height_map = F.resize(height_map, [render_h, render_w])

        # TacSL renders "nominal depth - current depth" in meters, i.e. how far the gelpad is pushed in
        press_depth_m = torch.clamp(-height_map, min=0.0).to(self._device) / 1000.0
        img = self._renderer.render(press_depth_m).float() / 255.0  # (N, H, W, 3)

        w, h = self.cfg.tactile_img_res
        if (render_w, render_h) != (w, h):
            img = F.resize(img.permute(0, 3, 1, 2), [h, w]).permute(0, 2, 3, 1)
        return img

    def force_field_simulation(self):
        """Simulates the force field into the sensor's ``force_field`` buffer (dict of tensors, updated in place).

        Keys: ``points_pos_w`` (N, P, 3) tactile point positions in the world frame [m];
        ``penetration_depth`` (N, P) [m]; ``normal_force`` (N, P) and ``shear_force`` (N, P, 2) in the tactile frame
        (z = contact normal), in the units of the penalty stiffnesses.
        """
        ff = self._force_field
        ff._update_force_field(slice(None))
        out = self._output_buffers["force_field"]
        out["points_pos_w"][:] = ff._data.tactile_points_pos_w
        out["penetration_depth"][:] = ff._data.penetration_depth
        out["normal_force"][:] = ff._data.tactile_normal_force
        out["shear_force"][:] = ff._data.tactile_shear_force

    def compute_indentation_depth(self) -> torch.Tensor:
        """Deepest point of the height map (mm), same as the sensor's default."""
        return -self.sensor._data.output["height_map"].amin((1, 2))

    def reset(self, env_ids: Sequence[int] | None = None):
        """Stateless: writes the no-contact state into the env_ids rows of the outputs (tactile image of the undeformed
        gel, zero forces; the tactile point positions are recomputed on the next update)."""
        idx = slice(None) if env_ids is None else env_ids
        if "tactile_rgb" in self._output_buffers:
            out = self._output_buffers["tactile_rgb"]
            out[idx] = self._rest_img.to(out.device)
        if "force_field" in self._output_buffers:
            for key in ("penetration_depth", "normal_force", "shear_force"):
                self._output_buffers["force_field"][key][idx] = 0.0

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if self._force_field is not None:
            self._force_field._set_debug_vis_impl(debug_vis)
        if not debug_vis:
            for window in self._debug_windows.values():
                window.destroy()
            self._debug_windows, self._debug_img_providers = {}, {}

    def _debug_vis_callback(self, event):
        if self._force_field is not None:
            self._force_field._debug_vis_callback(event)
        if self.optical_enabled and self.sensor.prim_view is not None:
            self._update_tactile_rgb_windows()

    def _update_tactile_rgb_windows(self):
        """Shows the tactile RGB image of every sensor whose USD attribute ``debug_tactile_rgb`` is on."""
        import omni.ui

        w, h = self.cfg.tactile_img_res
        for i, prim in enumerate(self.sensor.prim_view.prims):
            key = str(i)
            attr = prim.GetAttribute("debug_tactile_rgb")
            if attr and attr.Get():
                if key not in self._debug_windows:
                    self._debug_windows[key] = omni.ui.Window(
                        self.sensor.prim_view.prim_paths[i] + "/tacsl_rgb", width=w, height=h
                    )
                    self._debug_img_providers[key] = omni.ui.ByteImageProvider()
                frame = (self._output_buffers["tactile_rgb"][i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                with self._debug_windows[key].frame:
                    self._debug_img_providers[key].set_bytes_data(frame.flatten().data, [w, h])
                    omni.ui.ImageWithProvider(self._debug_img_providers[key])
            elif key in self._debug_windows:
                self._debug_windows.pop(key).destroy()
                self._debug_img_providers.pop(key).destroy()


class _TacSLForceField(VisuoTactileSensor):
    """Isaac Lab's TacSL force field code, used without the sensor machinery.

    ``VisuoTactileSensor.__init__`` is deliberately not called: this object is not a sensor (no SensorBase callbacks,
    no camera). It only provides the attributes the reused methods read (see ``VisuoTactileSensor.__init__``):
    ``_initialize_force_field``, ``_create_physx_views``, ``_update_force_field`` and the force computation, as well as
    the tactile point markers. ``_generate_tactile_points`` is replaced (see there).
    """

    def __init__(self, cfg: TacSLForceFieldCfg, elastomer_prim, num_envs: int, device: str, physics_sim_view):  # noqa: D107
        self.cfg = cfg
        self._data = VisuoTactileSensorData()
        self._device = device
        self._num_envs = num_envs
        self._parent_prims = [elastomer_prim]  # TacSL's elastomer is the parent prim of its sensor prim
        self._physics_sim_view = physics_sim_view

        self._tactile_pos_local = None
        self._tactile_quat_local = None
        self._elastomer_com_b = None
        self._contact_object_com_b = None
        self._elastomer_body_view = None
        self._contact_object_body_view = None
        self._contact_object_sdf_view = None
        self._tactile_visualizer = None
        self.num_tactile_points = 0

    def __del__(self):
        pass  # not registered as a sensor, nothing to unsubscribe

    def _generate_tactile_points(self, num_divs: list, margin: float, visualize: bool):
        """Generates the tactile point grid on the gelpad surface (in the elastomer body frame).

        Same ray-casting scheme as TacSL, extended for TacEx gelpad assets:

        - uses the first visual mesh below the elastomer, else any mesh (TacEx gelpads may only have a collision mesh)
        - triangulates polygon faces and applies the mesh's transform relative to the elastomer body
        - contact side from ``cfg.contact_normal_local`` (TacSL's dome heuristic fails for flat gelpads)
        - tactile frame z = outward contact normal for any gelpad orientation (TacSL assumes the normal along z)
        """
        elastomer_prim = self._parent_prims[0]
        mesh_prim = sim_utils.get_first_matching_child_prim(
            elastomer_prim.GetPath(), predicate=lambda p: p.IsA(UsdGeom.Mesh) and not p.HasAPI(UsdPhysics.CollisionAPI)
        ) or sim_utils.get_first_matching_child_prim(elastomer_prim.GetPath(), predicate=lambda p: p.IsA(UsdGeom.Mesh))
        if mesh_prim is None:
            raise RuntimeError(f"TacSL force field: no mesh found below elastomer '{elastomer_prim.GetPath()}'.")

        points, faces = _mesh_in_body_frame(mesh_prim, elastomer_prim)
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        dims = bounds[1] - bounds[0]
        slim_axis = int(np.argmin(dims))
        center = bounds.mean(axis=0)

        if self.cfg.contact_normal_local is not None:
            normal = np.asarray(self.cfg.contact_normal_local, dtype=float)
            normal /= np.linalg.norm(normal)
            if abs(normal[slim_axis]) < 0.99:
                raise ValueError(
                    f"contact_normal_local {tuple(self.cfg.contact_normal_local)} is not parallel to the gelpad's"
                    f" thinnest axis ({'xyz'[slim_axis]}) in the elastomer frame."
                )
            tip_sign = float(np.sign(normal[slim_axis]))
        else:
            offset = mesh.center_mass[slim_axis] - center[slim_axis]
            if abs(offset) < 1e-3 * dims[slim_axis]:
                raise ValueError(
                    f"Cannot detect the contact side of the flat gelpad '{mesh_prim.GetPath()}'. Set"
                    " force_field.contact_normal_local (outward normal in the elastomer frame)."
                )
            tip_sign = 1.0 if offset > 0 else -1.0

        # grid on the plane, rays cast from outside the contact side toward the gelpad (as in TacSL)
        axes = [a for a in range(3) if a != slim_axis]
        spacing = min((dims[axes] - 2.0 * margin) / (np.array(num_divs) + 1))
        grid_axes = []
        for axis_i in range(3):
            if axis_i == slim_axis:
                grid_axes.append([center[slim_axis] + tip_sign * (dims[slim_axis] + 1.0)])
            else:
                n = num_divs[axes.index(axis_i)]
                ticks = np.linspace(
                    center[axis_i] - spacing * (n + 1) / 2.0, center[axis_i] + spacing * (n + 1) / 2.0, n + 2
                )
                grid_axes.append(ticks[1:-1])
        origins = np.array(list(itertools.product(*grid_axes)))
        ray_dir = np.zeros(3)
        ray_dir[slim_axis] = -tip_sign
        _, index_ray, locations = trimesh.ray.ray_triangle.RayMeshIntersector(mesh).intersects_id(
            origins, np.tile(ray_dir, (len(origins), 1)), return_locations=True, multiple_hits=False
        )
        if visualize:
            trimesh.Scene([mesh, trimesh.PointCloud(locations, colors=(0.0, 0.0, 1.0))]).show()

        tactile_points = locations[index_ray.argsort()]
        self._tactile_pos_local = torch.tensor(tactile_points, dtype=torch.float32, device=self._device)
        self.num_tactile_points = len(tactile_points)
        if self.num_tactile_points != num_divs[0] * num_divs[1]:
            raise RuntimeError(
                f"TacSL force field: {self.num_tactile_points} of {num_divs[0] * num_divs[1]} tactile points hit the"
                f" gelpad mesh '{mesh_prim.GetPath()}'. Increase tactile_margin or check contact_normal_local."
            )

        # tactile frame: z = outward normal; x, y chosen so that a z-normal gelpad gets TacSL's frame (Rz(-pi))
        a, b = (slim_axis + 1) % 3, (slim_axis + 2) % 3
        rot = np.zeros((3, 3))
        rot[a, 0] = -1.0  # x = -e_a
        rot[b, 1] = -tip_sign  # y = -sign * e_b  (keeps the frame right-handed)
        rot[slim_axis, 2] = tip_sign  # z = sign * e_slim
        quat = math_utils.quat_from_matrix(torch.tensor(rot, dtype=torch.float32, device=self._device))
        self._tactile_quat_local = quat.unsqueeze(0).repeat(self.num_tactile_points, 1)

        logger.info(f"TacSL force field: {self.num_tactile_points} tactile points on '{mesh_prim.GetPath()}'.")


def _mesh_in_body_frame(mesh_prim, body_prim) -> tuple[np.ndarray, np.ndarray]:
    """Mesh vertices in the body frame (without the body's scale, like physics poses) and triangulated faces."""
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())

    faces, start = [], 0
    for count in counts:  # fan triangulation of polygons
        poly = indices[start : start + count]
        faces.extend([poly[0], poly[k], poly[k + 1]] for k in range(1, count - 1))
        start += count

    cache = UsdGeom.XformCache()
    body_tf = Gf.Transform(cache.GetLocalToWorldTransform(body_prim))
    body_pose = Gf.Matrix4d().SetRotate(body_tf.GetRotation()).SetTranslateOnly(body_tf.GetTranslation())
    mesh_to_body = np.array(cache.GetLocalToWorldTransform(mesh_prim) * body_pose.GetInverse())  # row-vector conv.
    points_h = np.hstack([points, np.ones((len(points), 1))])
    return (points_h @ mesh_to_body)[:, :3], np.asarray(faces, dtype=np.int64)


def _load_background(path: str) -> np.ndarray:
    """Loads an RGB uint8 background from an image file or a Taxim dataPack (.npz, key ``f0``, BGR)."""
    if path.endswith(".npz"):
        bgr = np.load(path, allow_pickle=True)["f0"]
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read background image '{path}'.")
    return np.ascontiguousarray(bgr[..., ::-1]).astype(np.uint8)
