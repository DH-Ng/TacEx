###
# Implementation of the simulation approach from
# @ARTICLE{chen2024tactilesim2real,
#     author={Chen, Weihang and Xu, Jing and Xiang, Fanbo and Yuan, Xiaodi and Su, Hao and Chen, Rui},
#     journal={IEEE Transactions on Robotics},
#     title={General-Purpose Sim2Real Protocol for Learning Contact-Rich Manipulation With Marker-Based Visuotactile Sensors},
#     year={2024},
#     volume={40},
#     pages={1509-1526},
#     doi={10.1109/TRO.2024.3352969}
# }
# Original code can be found here
# https://github.com/chuanyune/ManiSkill-ViTac2025/blob/a3d7df54bca9a2e57f34b37be3a3df36dc218915/Track_1/envs/tactile_sensor_sapienipc.py
##

import math
import numpy as np
import torch

import omni.usd
import usdrt
import usdrt.UsdGeom
from sklearn.neighbors import NearestNeighbors

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from ....gelsight_sensor import GelSightSensor
from tacex_uipc.objects import UipcObject
from tacex_uipc.sim import UipcSim

from .utils.geometry import in_hull


class FemBasedMarkerSim:
    def __init__(
        self,
        uipc_gelpad: UipcObject,
        sensor: GelSightSensor,
        tactile_img_width=320,
        tactile_img_height=240,
        marker_interval_range: tuple[float, float] = (2.0625, 2.0625),
        marker_rotation_range: float = 0.0,
        marker_translation_range: tuple[float, float] = (0.0, 0.0),
        marker_pos_shift_range: tuple[float, float] = (0.0, 0.0),
        marker_random_noise: float = 0.0,
        marker_lose_tracking_probability: float = 0.0,
        normalize: bool = False,
        num_markers: int = 128,
        camera_distort_coeff: float = 0.0,
        **kwargs,
    ):
        """
        param: marker_interval_rang, in mm.
        param: marker_rotation_range: overall marker rotation, in radian.
        param: marker_translation_range: overall marker translation, in mm. first two elements: x-axis; last two elements: y-xis.
        param: marker_pos_shift_range: independent marker position shift, in mm, in x- and y-axis. caused by fabrication errors.
        param: marker_random_noise: std of Gaussian marker noise, in pixel. caused by CMOS noise and image processing.
        param: loss_tracking_probability: the probability of losing tracking, appled to each marker
        param: normalize: whether to normalize the output marker flow
        param: marker_flow_size: the size of the output marker flow

        The camera intrinsics for projecting the markers into the tactile image are computed from the sensor config
        (view size at the gel surface, gel surface distance) and the tactile image resolution.
        """

        self.gelpad_obj = uipc_gelpad
        self.uipc_sim: UipcSim = uipc_gelpad.uipc_sim
        self.scene = self.uipc_sim.scene

        self.sensor = sensor
        self.camera = sensor.camera
        # overwrite config so that pose is always updated,
        # which is needed for projection of markers from gelpad frame into img frame
        self.camera.cfg.update_latest_camera_pose = True

        self.init_surface_vertices = self.get_surface_vertices_world()

        self.tactile_img_width = tactile_img_width
        self.tactile_img_height = tactile_img_height

        self.marker_interval_range = marker_interval_range
        self.marker_rotation_range = marker_rotation_range
        self.marker_translation_range = marker_translation_range
        self.marker_pos_shift_range = marker_pos_shift_range
        self.marker_random_noise = marker_random_noise
        self.marker_lose_tracking_probability = marker_lose_tracking_probability
        self.normalize = normalize
        self.num_markers = num_markers

        # camera frame to gel center
        # NOTE: camera frame follows opencv coordinate system
        self.camera_intrinsic = self.compute_camera_intrinsic(sensor, tactile_img_width, tactile_img_height)

        self.camera_distort_coeffs = np.array([camera_distort_coeff, 0, 0, 0, 0], dtype=np.float32)

        self.init_vertices_camera = self.get_vertices_camera()
        self.init_surface_vertices_gelpad = self.get_surface_vertices_world().clone()

        # undeformed gel surface in the camera frame, from consistent spawn poses: used to place the marker grid on the
        # surface (_gen_marker_weight) and as reference for the marker motion (gen_marker_flow)
        self.init_surface_vertices_camera = self.compute_rest_surface_vertices_camera()
        self.reference_surface_vertices_camera = self.init_surface_vertices_camera.clone()

        # self.phong_shading_renderer = PhongShadingRenderer()
        self.curr_marker_pts = None

    def get_vertices_world(self):
        v = self.gelpad_obj._data.nodal_pos_w
        return v

    def get_surface_vertices_world(self):
        surf_v = self.gelpad_obj._data.surf_nodal_pos_w
        return surf_v

    # todo find out what's wrong with this method -> frame coor. sys. seems to be wrong
    def transform_camera_to_world_frame(self, input_vertices):
        # self.camera._update_poses(self.camera._ALL_INDICES)
        # math_utils.convert_camera_frame_orientation_convention
        cam_pos_w = self.camera._data.pos_w
        cam_quat_w = self.camera._data.quat_w_ros  # quat_w_opengl #quat_w_world

        # NOTE: I think there is a slight delay between camera pose data, thats why marker sim with FEM bad in case of fast movement
        # self._ALL_INDICES = torch.arange(self.camera._num_envs, device=self.camera._device, dtype=torch.long)
        # cam_pos_w, cam_quat_w = self.camera._view.get_world_poses(self._ALL_INDICES)

        v_cv = math_utils.transform_points(input_vertices, pos=cam_pos_w, quat=cam_quat_w)
        return v_cv

    def transform_world_to_camera_frame(self, input_vertices, cam_pos_w=None, cam_quat_w=None):
        """World -> camera frame (ROS convention). Uses the current camera pose data unless a pose is given."""
        # self.camera._update_poses(self.camera._ALL_INDICES)
        # math_utils.convert_camera_frame_orientation_convention
        if cam_pos_w is None:
            cam_pos_w = self.camera._data.pos_w
            cam_quat_w = self.camera._data.quat_w_ros
        cam_quat_w_inv = math_utils.quat_inv(cam_quat_w)

        rot_inv = math_utils.matrix_from_quat(cam_quat_w_inv)
        # convert to batched
        if rot_inv.dim() == 2:
            rot_inv = rot_inv[None]  # (3, 3) -> (1, 3, 3)

        t_target = input_vertices - cam_pos_w
        # convert to batched #todo fix for multi env
        t_target = t_target[None, :, :]  # (N, 3) -> (N, 1, 3)

        rot_inv = rot_inv.to(t_target.dtype)
        v_cv = torch.matmul(rot_inv, t_target.transpose_(1, 2))
        v_cv = v_cv.transpose_(1, 2)
        # todo fix for multi env
        v_cv = v_cv[0]
        return v_cv

    def get_init_surface_vertices_camera(self):
        return self.transform_world_to_camera_frame(self.get_surface_vertices_world()).clone()

    def transform_to_init_gelpad_frame(self, input_vertices):
        world_tf = self.gelpad_obj.init_world_transform
        pos_w, rot_mat = math_utils.unmake_pose(world_tf)
        quat_w = math_utils.quat_from_matrix(rot_mat).type(torch.float32)

        vertices_gelpad_frame = math_utils.transform_points(input_vertices, pos=pos_w, quat=quat_w)
        return vertices_gelpad_frame

    def get_surface_vertices_in_gelpad_frame(self):
        v = self.get_surface_vertices_world()
        v_cv = self.transform_to_init_gelpad_frame(v)
        return v_cv

    def get_init_surface_vertices_gelpad(self):
        return self.init_surface_vertices_gelpad.clone()

    def get_vertices_camera(self):
        v = self.get_vertices_world()
        v_cv = self.transform_world_to_camera_frame(v)
        return v_cv

    def get_surface_vertices_camera(self):
        v = self.get_surface_vertices_world()
        v_cv = self.transform_world_to_camera_frame(v)
        return v_cv

    def set_reference_surface_vertices_camera(self):
        self.reference_surface_vertices_camera = self.get_surface_vertices_camera().clone()

    def compute_rest_surface_vertices_camera(self) -> torch.Tensor:
        """Undeformed gel surface in the camera frame, from the spawn poses of gel and camera.

        The current poses are not usable here: during initialization the camera pose data still holds the spawn pose,
        while the gelpad may already follow the physics pose of the sensor case (attachment), which shifts the
        reference (and the marker grid) in the image. The spawn poses of gel (UIPC rest geometry) and camera (USD)
        come from the same USD hierarchy, so they are consistent at any time.
        """
        # surface vertices are copies of tet node positions -> map each surface vertex to its node
        nodes_now = self.get_vertices_world().reshape(-1, 3).cpu().numpy()
        surf_now = self.get_surface_vertices_world().reshape(-1, 3).cpu().numpy()
        dist, node_idx = NearestNeighbors(n_neighbors=1).fit(nodes_now).kneighbors(surf_now)
        if dist.max() > 1e-6:
            raise RuntimeError(f"Gel surface vertices do not coincide with tet nodes (max distance {dist.max()} m).")

        # rest (spawn) positions of these nodes
        _, geo_slot_rest = self.uipc_sim.scene.geometries().find(self.gelpad_obj.obj_id)
        nodes_rest = np.asarray(geo_slot_rest.geometry().positions().view()).reshape(-1, 3)
        surf_rest = torch.tensor(nodes_rest[node_idx[:, 0]], device=self.camera._device, dtype=torch.float32)

        # spawn pose of the camera (USD), in ROS convention like the camera data
        cam_prim = sim_utils.find_matching_prims(self.camera.cfg.prim_path)[0]
        tf = omni.usd.get_world_transform_matrix(cam_prim)
        q = tf.ExtractRotation().GetQuaternion()
        cam_pos = torch.tensor([list(tf.ExtractTranslation())], device=self.camera._device, dtype=torch.float32)
        cam_quat_opengl = torch.tensor(
            [[q.GetReal(), *q.GetImaginary()]], device=self.camera._device, dtype=torch.float32
        )
        cam_quat_ros = math_utils.convert_camera_frame_orientation_convention(
            cam_quat_opengl, origin="opengl", target="ros"
        )
        return self.transform_world_to_camera_frame(surf_rest, cam_pos_w=cam_pos, cam_quat_w=cam_quat_ros)

    def _gen_marker_grid(self):
        marker_interval = (self.marker_interval_range[1] - self.marker_interval_range[0]) * np.random.rand(1)[
            0
        ] + self.marker_interval_range[0]
        marker_rotation_angle = 2 * self.marker_rotation_range * np.random.rand(1) - self.marker_rotation_range
        marker_translation_x = (
            2 * self.marker_translation_range[0] * np.random.rand(1)[0] - self.marker_translation_range[0]
        )
        marker_translation_y = (
            2 * self.marker_translation_range[1] * np.random.rand(1)[0] - self.marker_translation_range[1]
        )

        marker_x_start = (
            -math.ceil((8 + marker_translation_x) / marker_interval) * marker_interval + marker_translation_x  # 16.5
        )  # 8
        marker_x_end = (
            math.ceil((8 - marker_translation_x) / marker_interval) * marker_interval + marker_translation_x
        )  # 16.5
        marker_y_start = (
            -math.ceil((6 + marker_translation_y) / marker_interval) * marker_interval + marker_translation_y
        )
        marker_y_end = math.ceil((6 - marker_translation_y) / marker_interval) * marker_interval + marker_translation_y

        marker_x = np.linspace(
            marker_x_start,
            marker_x_end,
            round((marker_x_end - marker_x_start) / marker_interval) + 1,
            True,
        )
        marker_y = np.linspace(
            marker_y_start,
            marker_y_end,
            round((marker_y_end - marker_y_start) / marker_interval) + 1,
            True,
        )

        marker_xy = np.array(np.meshgrid(marker_x, marker_y)).reshape((2, -1)).T
        marker_num = marker_xy.shape[0]
        # print(marker_num)

        marker_pos_shift_x = (
            np.random.rand(marker_num) * self.marker_pos_shift_range[0] * 2 - self.marker_pos_shift_range[0]
        )

        marker_pos_shift_y = (
            np.random.rand(marker_num) * self.marker_pos_shift_range[1] * 2 - self.marker_pos_shift_range[1]
        )

        marker_xy[:, 0] += marker_pos_shift_x
        marker_xy[:, 1] += marker_pos_shift_y

        rot_mat = np.array(
            [
                [math.cos(marker_rotation_angle), -math.sin(marker_rotation_angle)],
                [math.sin(marker_rotation_angle), math.cos(marker_rotation_angle)],
            ]
        )

        marker_rotated_xy = marker_xy @ rot_mat.T

        return marker_rotated_xy / 1000.0

    def _gen_marker_weight(self, marker_pts):
        # filter out markers which cannot be on the mesh surface
        # surface_pts = self.get_init_surface_vertices_gelpad()
        surface_pts = self.init_surface_vertices_camera

        # surface_pts = self.transform_world_to_camera_frame(self.get_surface_vertices_world())[:, :2]

        surface_pts = surface_pts.cpu().numpy()
        # draw.clear_points()
        # points = np.array(surface_pts)
        # draw.draw_points(points, [(255,0,255,0.5)]*points.shape[0], [30]*points.shape[0])

        # set marker to be at gelpad by adding corresponding z values
        z = np.max(
            surface_pts[:, 2]
        )  # to set pattern at the bottom of the gelpad, i.e. closer to camera: np.min(surface_pts[:, 2])
        marker_pts = np.hstack((marker_pts, np.ones((marker_pts.shape[0], 1), dtype=np.float32) * z))
        # marker_pts = self.transform_to_camera_frame(torch.tensor(marker_pts, device="cuda:0", dtype=torch.float32)).cpu().numpy()
        # draw.draw_points(marker_pts, [(255,0,0,0.5)]*marker_pts.shape[0], [30]*marker_pts.shape[0])

        # check if x,y coor. of markers are plausible
        marker_on_surface = in_hull(marker_pts[:, :2], surface_pts[:, :2])
        marker_pts = marker_pts[marker_on_surface]

        # extract faces of the mesh surface
        # simulation surface triangles (the render mesh may have a different topology, see `RenderMeshCfg`)
        triangles = self.gelpad_obj.sim_surf_triangles
        f_centers = []
        for tri in triangles:
            vertices = surface_pts[tri]
            f_center = np.mean(vertices, axis=0)
            f_centers.append(f_center)

        # points = np.array(f_centers)
        # draw.draw_points(points, [(0,255,255,0.5)]*points.shape[0], [30]*points.shape[0])

        nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(f_centers)
        distances, face_idx = nbrs.kneighbors(marker_pts)

        # after nearest neighbors, just look at xy components
        marker_pts = marker_pts[:, :2]

        marker_pts_surface_idx = []
        marker_pts_surface_weight = []
        valid_marker_idx = []

        # compute barycentric weight of each vertex
        for i in range(marker_pts.shape[0]):
            possible_face_ids = face_idx[i]
            p = marker_pts[i]
            for possible_face_id in possible_face_ids.tolist():
                vertices_of_face_i = surface_pts[triangles[possible_face_id]]
                p0, p1, p2 = vertices_of_face_i[:, :2]
                A = np.stack([p1 - p0, p2 - p0], axis=1)
                w12 = np.linalg.inv(A) @ (p - p0)
                if possible_face_id == possible_face_ids[0]:
                    marker_pts_surface_idx.append(triangles[possible_face_id])
                    marker_pts_surface_weight.append(np.array([1 - w12.sum(), w12[0], w12[1]]))
                    valid_marker_idx.append(i)
                    if w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                        break
                elif w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                    marker_pts_surface_idx[-1] = triangles[possible_face_id]
                    marker_pts_surface_weight[-1] = np.array([1 - w12.sum(), w12[0], w12[1]])
                    valid_marker_idx[-1] = i
                    break

        valid_marker_idx = np.array(valid_marker_idx).astype(np.int32)
        marker_pts = marker_pts[valid_marker_idx]
        marker_pts_surface_idx = np.stack(marker_pts_surface_idx)
        marker_pts_surface_weight = np.stack(marker_pts_surface_weight)
        assert np.allclose(
            (surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1)[:, :2],
            marker_pts,
        ), (
            "max err:"
            f" {np.abs((surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1)[:, :2] - marker_pts).max()}"
        )

        return marker_pts_surface_idx, marker_pts_surface_weight

    @staticmethod
    def compute_camera_intrinsic(sensor: GelSightSensor, img_width: int, img_height: int) -> np.ndarray:
        """Pinhole intrinsics (3, 3) of the sensor camera for an image of ``img_width`` x ``img_height`` pixels.

        Same geometry as the sensor camera (see ``GelSightSensor._create_tiled_camera``): the view
        ``view_width_m`` x ``view_height_m`` is defined at the undeformed gel surface, so
        ``fx = img_width * d / view_width_m`` with ``d`` = gel surface distance (likewise ``fy``). The principal point
        is the image center (``img_width / 2``, ``img_height / 2``), as for the Isaac Lab camera.
        """
        cam_cfg = sensor.cfg.sensor_camera_cfg
        d = sensor.cfg.get_gel_surface_to_camera_distance_m()
        return np.array(
            [
                [img_width * d / cam_cfg.view_width_m, 0, img_width / 2],
                [0, img_height * d / cam_cfg.view_height_m, img_height / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def gen_marker_uv(self, marker_pts):
        # marker_uv = cv2.projectPoints(
        #     marker_pts,
        #     np.zeros(3, dtype=np.float32),
        #     np.zeros(3, dtype=np.float32),
        #     self.camera_intrinsic,
        #     self.camera_distort_coeffs,
        # )[0].squeeze(1)
        # marker_uv = cv2.projectPoints(
        #     marker_pts,
        #     np.zeros(3, dtype=np.float32),
        #     np.zeros(3, dtype=np.float32),
        #     self.camera_intrinsic,
        #     self.camera_distort_coeffs,
        # )[0].squeeze(1)
        # intrinsic_matrices = self.camera._data.intrinsic_matrices #todo intrinisc matrix is currently wrong
        intrinsic_matrices = torch.tensor(self.camera_intrinsic, device="cuda:0")
        vertices_img_plane = math_utils.project_points(
            torch.tensor(marker_pts, device="cuda:0", dtype=torch.float32), intrinsic_matrices
        )
        marker_uv = vertices_img_plane[0, :, :2].cpu().numpy()
        return marker_uv

    def gen_marker_flow(self):
        marker_grid = self._gen_marker_grid()

        marker_pts_surface_idx, marker_pts_surface_weight = self._gen_marker_weight(marker_grid)

        init_marker_pts = (
            self.reference_surface_vertices_camera[marker_pts_surface_idx].cpu().numpy()
            * marker_pts_surface_weight[..., None]
        ).sum(1)
        self.curr_marker_pts = (
            self.get_surface_vertices_camera()[marker_pts_surface_idx].cpu().numpy()
            * marker_pts_surface_weight[..., None]
        ).sum(1)

        init_marker_uv = self.gen_marker_uv(init_marker_pts)
        curr_marker_uv = self.gen_marker_uv(self.curr_marker_pts)
        marker_mask = np.logical_and.reduce(
            [
                init_marker_uv[:, 0] > 5,
                init_marker_uv[:, 0] < self.tactile_img_width,  # u: image x (columns)
                init_marker_uv[:, 1] > 5,
                init_marker_uv[:, 1] < self.tactile_img_height,  # v: image y (rows)
            ]
        )
        marker_flow = np.stack([init_marker_uv, curr_marker_uv], axis=0)
        marker_flow = marker_flow[:, marker_mask]

        # post processing
        no_lose_tracking_mask = np.random.rand(marker_flow.shape[1]) > self.marker_lose_tracking_probability
        marker_flow = marker_flow[:, no_lose_tracking_mask, :]
        noise = np.random.randn(*marker_flow.shape) * self.marker_random_noise
        marker_flow += noise

        original_point_num = marker_flow.shape[1]

        if original_point_num >= self.num_markers:
            chosen = np.random.choice(original_point_num, self.num_markers, replace=False)
            ret = marker_flow[:, chosen, ...]
        else:
            ret = np.zeros((marker_flow.shape[0], self.num_markers, marker_flow.shape[-1]))
            ret[:, :original_point_num, :] = marker_flow.copy()
            ret[:, original_point_num:, :] = ret[:, original_point_num - 1 : original_point_num, :]

        if self.normalize:
            ret /= self.tactile_img_width / 2
            ret -= 1.0

        ret = torch.tensor(ret, device="cuda:0")
        self.curr_marker_uv = curr_marker_uv
        return ret
