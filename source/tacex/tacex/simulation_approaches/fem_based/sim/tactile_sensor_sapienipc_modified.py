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

import copy
import math
from typing import Tuple

import cv2
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import usdrt
import usdrt.UsdGeom
from sklearn.neighbors import NearestNeighbors
from tacex_uipc.objects import UipcObject
from tacex_uipc.sim import UipcSim

from .utils.geometry import in_hull

try:
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except:
    draw = None

class VisionTactileSensorUIPC():
    def __init__(
        self,
        uipc_gelpad: UipcObject,
        camera,
        marker_interval_range: Tuple[float, float] = (2.0625, 2.0625),
        marker_rotation_range: float = 0.0,
        marker_translation_range: Tuple[float, float] = (0.0, 0.0),
        marker_pos_shift_range: Tuple[float, float] = (0.0, 0.0),
        marker_random_noise: float = 0.0,
        marker_lose_tracking_probability: float = 0.0,
        normalize: bool = False,
        num_markers: int = 128,
        # camera_params: Tuple[float, float, float, float, float] = (
        #     340,
        #     325,
        #     160,
        #     125,
        #     0.0,
        # ),
        # for upright tactile img
        camera_params: Tuple[float, float, float, float, float] = (
            325,
            340,
            125,
            160,
            0.0,
        ),
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
        param: camera_params: (fx, fy, cx, cy, distortion)
        """

        self.gelpad_obj = uipc_gelpad
        self.uipc_sim: UipcSim = uipc_gelpad.uipc_sim
        self.scene = self.uipc_sim.scene

        self.camera = camera

        self.init_surface_vertices = self.get_surface_vertices_world()

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
        self.camera_intrinsic = np.array(
            [
                [camera_params[0], 0, camera_params[2]],
                [0, camera_params[1], camera_params[3]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.camera_distort_coeffs = np.array(
            [camera_params[4], 0, 0, 0, 0], dtype=np.float32
        )

        self.init_vertices_camera = self.get_vertices_camera()
        self.init_surface_vertices_gelpad = self.get_surface_vertices_world().clone()

        self.init_surface_vertices_camera = self.get_surface_vertices_camera().clone()
        self.reference_surface_vertices_camera = self.get_surface_vertices_camera().clone()

        # self.phong_shading_renderer = PhongShadingRenderer()

        self.patch_array_dict = copy.deepcopy(generate_patch_array())

    def get_vertices_world(self):
        v = self.gelpad_obj._data.nodal_pos_w
        return v

    def get_surface_vertices_world(self):
        surf_v = self.gelpad_obj._data.surf_nodal_pos_w
        return surf_v

    #todo find out whats wrong with this method -> frame coor. sys. seems to be wrong
    def transform_camera_to_world_frame(self, input_vertices):
        self.camera._update_poses(self.camera._ALL_INDICES)
        # math_utils.convert_camera_frame_orientation_convention
        cam_pos_w = self.camera._data.pos_w
        cam_quat_w =  self.camera._data.quat_w_ros #quat_w_opengl#quat_w_world
        v_cv = math_utils.transform_points(input_vertices, pos=cam_pos_w, quat=cam_quat_w)
        return v_cv

    def transform_world_to_camera_frame(self, input_vertices):
        self.camera._update_poses(self.camera._ALL_INDICES)
        # math_utils.convert_camera_frame_orientation_convention
        cam_pos_w = self.camera._data.pos_w
        cam_quat_w = self.camera._data.quat_w_ros
        cam_quat_w_inv = math_utils.quat_inv(cam_quat_w)

        rot_inv = math_utils.matrix_from_quat(cam_quat_w_inv)
        # convert to batched
        if rot_inv.dim() == 2:
            rot_inv = rot_inv[None]  # (3, 3) -> (1, 3, 3)

        t_target = input_vertices - cam_pos_w
        # convert to batched #todo fix it for multi env
        t_target = t_target[None, :, :]  # (N, 3) -> (N, 1, 3)

        v_cv = torch.matmul(rot_inv, t_target.transpose_(1, 2))
        v_cv = v_cv.transpose_(1, 2)
        #todo fix it for multi env
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
        self.reference_surface_vertices_camera = (
            self.get_surface_vertices_camera().clone()
        )

    def _gen_marker_grid(self):
        marker_interval = (
            self.marker_interval_range[1] - self.marker_interval_range[0]
        ) * np.random.rand(1)[0] + self.marker_interval_range[0]
        marker_rotation_angle = (
            2 * self.marker_rotation_range * np.random.rand(1)
            - self.marker_rotation_range
        )
        marker_translation_x = (
            2 * self.marker_translation_range[0] * np.random.rand(1)[0]
            - self.marker_translation_range[0]
        )
        marker_translation_y = (
            2 * self.marker_translation_range[1] * np.random.rand(1)[0]
            - self.marker_translation_range[1]
        )

        marker_x_start = (
            -math.ceil((6 + marker_translation_x) / marker_interval)  # 16.5
            * marker_interval
            + marker_translation_x
        )
        marker_x_end = (
            math.ceil((6 - marker_translation_x) / marker_interval) * marker_interval
            + marker_translation_x
        )
        marker_y_start = (
            -math.ceil((8 + marker_translation_y) / marker_interval) * marker_interval
            + marker_translation_y
        )
        marker_y_end = (
            math.ceil((8 - marker_translation_y) / marker_interval) * marker_interval
            + marker_translation_y
        )

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
            np.random.rand(marker_num) * self.marker_pos_shift_range[0] * 2
            - self.marker_pos_shift_range[0]
        )

        marker_pos_shift_y = (
            np.random.rand(marker_num) * self.marker_pos_shift_range[1] * 2
            - self.marker_pos_shift_range[1]
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
        z = np.max(surface_pts[:, 2]) # to set pattern at the bottom of the gelpad, i.e. closer to camera: np.min(surface_pts[:, 2])
        marker_pts = np.hstack((marker_pts, np.ones((63,1), dtype=np.float32)*z))
        # marker_pts = self.transform_to_camera_frame(torch.tensor(marker_pts, device="cuda:0", dtype=torch.float32)).cpu().numpy()
        # draw.draw_points(marker_pts, [(255,0,0,0.5)]*marker_pts.shape[0], [30]*marker_pts.shape[0])

        # check if x,y coor. of markers are plausible
        marker_on_surface = in_hull(marker_pts[:, :2], surface_pts[:, :2])
        marker_pts = marker_pts[marker_on_surface]

        # extract faces of the mesh surface
        triangles = np.array(usdrt.UsdGeom.Mesh(self.gelpad_obj.fabric_prim).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        f_centers = []
        for tri in triangles:
            vertices = surface_pts[tri]
            f_center = np.mean(vertices, axis=0)
            f_centers.append(f_center)

        # points = np.array(f_centers)
        # draw.draw_points(points, [(0,255,255,0.5)]*points.shape[0], [30]*points.shape[0])

        nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(
            f_centers
        )
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
                    marker_pts_surface_weight.append(
                        np.array([1 - w12.sum(), w12[0], w12[1]])
                    )
                    valid_marker_idx.append(i)
                    if w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                        break
                elif w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                    marker_pts_surface_idx[-1] = triangles[possible_face_id]
                    marker_pts_surface_weight[-1] = np.array(
                        [1 - w12.sum(), w12[0], w12[1]]
                    )
                    valid_marker_idx[-1] = i
                    break

        valid_marker_idx = np.array(valid_marker_idx).astype(np.int32)
        marker_pts = marker_pts[valid_marker_idx]
        marker_pts_surface_idx = np.stack(marker_pts_surface_idx)
        marker_pts_surface_weight = np.stack(marker_pts_surface_weight)
        assert np.allclose(
            (
                surface_pts[marker_pts_surface_idx]
                * marker_pts_surface_weight[..., None]
            ).sum(1)[:, :2],
            marker_pts,
        ), f"max err: {np.abs((surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1)[:, :2] - marker_pts).max()}"

        return marker_pts_surface_idx, marker_pts_surface_weight

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
        intrinsic_matrices = self.camera._data.intrinsic_matrices #todo intrinisc matrix is currently wrong
        intrinsic_matrices = torch.tensor(self.camera_intrinsic, device="cuda:0")
        vertices_img_plane = math_utils.project_points(torch.tensor(marker_pts, device="cuda:0", dtype=torch.float32), intrinsic_matrices)
        marker_uv = vertices_img_plane[0, :, :2].cpu().numpy()
        return marker_uv

    def gen_marker_flow(self):
        marker_grid = self._gen_marker_grid()

        marker_pts_surface_idx, marker_pts_surface_weight = self._gen_marker_weight(marker_grid)

        init_marker_pts = (
            self.reference_surface_vertices_camera[marker_pts_surface_idx].cpu().numpy()
            * marker_pts_surface_weight[..., None]
        ).sum(1)
        curr_marker_pts = (
            self.get_surface_vertices_camera()[marker_pts_surface_idx].cpu().numpy()
            * marker_pts_surface_weight[..., None]
        ).sum(1)

        # draw markers in sim world
        draw.clear_points()
        curr_marker_pts_3d = curr_marker_pts.copy()
        curr_marker_pts_3d = self.transform_camera_to_world_frame(torch.tensor(curr_marker_pts_3d, device="cuda:0", dtype=torch.float32)).cpu().numpy()
        draw.draw_points(curr_marker_pts_3d, [(255,0,0,0.5)]*curr_marker_pts_3d.shape[0], [30]*curr_marker_pts_3d.shape[0])

        init_marker_uv = self.gen_marker_uv(init_marker_pts)
        curr_marker_uv = self.gen_marker_uv(curr_marker_pts)
        marker_mask = np.logical_and.reduce(
            [
                init_marker_uv[:, 0] > 5,
                init_marker_uv[:, 0] < 320, # img height
                init_marker_uv[:, 1] > 5,
                init_marker_uv[:, 1] < 240, # img width
            ]
        )
        marker_flow = np.stack([init_marker_uv, curr_marker_uv], axis=0)
        marker_flow = marker_flow[:, marker_mask]

        # post processing
        no_lose_tracking_mask = (
            np.random.rand(marker_flow.shape[1]) > self.marker_lose_tracking_probability
        )
        marker_flow = marker_flow[:, no_lose_tracking_mask, :]
        noise = np.random.randn(*marker_flow.shape) * self.marker_random_noise
        marker_flow += noise

        original_point_num = marker_flow.shape[1]

        if original_point_num >= self.num_markers:
            chosen = np.random.choice(
                original_point_num, self.num_markers, replace=False
            )
            ret = marker_flow[:, chosen, ...]
        else:
            ret = np.zeros(
                (marker_flow.shape[0], self.num_markers, marker_flow.shape[-1])
            )
            ret[:, :original_point_num, :] = marker_flow.copy()
            ret[:, original_point_num:, :] = ret[
                :, original_point_num - 1 : original_point_num, :
            ]

        if self.normalize:
            ret /= 160.0 # img height/2
            ret -= 1.0

        ret = torch.tensor(ret, device="cuda:0")
        self.curr_marker_uv = curr_marker_uv
        return ret

    def get_marker_img(self):
        curr_marker_uv = self.curr_marker_uv
        curr_marker_img = self.draw_markers(curr_marker_uv)
        # cv2.imwrite("curr_marker_img.png", curr_marker_img)
        return curr_marker_img

    # def gen_rgb_image(self):
    #     # generate RGB image from depth
    #     depth = self._gen_depth()
    #     rgb = self.phong_shading_renderer.generate(depth)
    #     rgb = rgb.astype(np.float64)

    #     # generate markers
    #     marker_grid = self._gen_marker_grid()
    #     marker_pts_surface_idx, marker_pts_surface_weight = self._gen_marker_weight(
    #         marker_grid
    #     )
    #     curr_marker_pts = (
    #         self.get_surface_vertices_in_camera_frame()[marker_pts_surface_idx]
    #         * marker_pts_surface_weight[..., None]
    #     ).sum(1)
    #     curr_marker_uv = self.gen_marker_uv(curr_marker_pts)

    #     curr_marker = self.draw_marker(curr_marker_uv)
    #     rgb = rgb.astype(np.float64)
    #     rgb *= np.dstack([curr_marker.astype(np.float64) / 255] * 3)
    #     rgb = rgb.astype(np.uint8)
    #     return rgb

    # def _gen_depth(self):
    #     # hide the gel to get the depth of the object in contact
    #     self.render_component.disable()
    #     self.cam_entity.set_pose(cv2ex2pose(self.get_camera_pose()))
    #     self.scene.update_render()
    #     ipc_update_render_all(self.scene)
    #     self.cam.take_picture()
    #     position = self.cam.get_picture("Position")  # [H, W, 4]
    #     depth = -position[..., 2]  # float in meter
    #     fem_smooth_sigma = 2
    #     depth = gaussian_filter(depth, fem_smooth_sigma)
    #     self.render_component.enable()

    #     return depth

    def draw_markers(self, marker_uv, marker_size=3, img_w=240, img_h=320):
        marker_uv_compensated = marker_uv + np.array([0.5, 0.5])

        marker_image = np.ones((img_h + 24, img_w + 24), dtype=np.uint8) * 255
        for i in range(marker_uv_compensated.shape[0]):
            uv = marker_uv_compensated[i]
            u = uv[0] + 12
            v = uv[1] + 12
            patch_id_u = math.floor(
                (u - math.floor(u)) * self.patch_array_dict["super_resolution_ratio"]
            )
            patch_id_v = math.floor(
                (v - math.floor(v)) * self.patch_array_dict["super_resolution_ratio"]
            )
            patch_id_w = math.floor(
                (marker_size - self.patch_array_dict["base_circle_radius"])
                * self.patch_array_dict["super_resolution_ratio"]
            )
            current_patch = self.patch_array_dict["patch_array"][
                patch_id_u, patch_id_v, patch_id_w
            ]
            patch_coord_u = math.floor(u) - 6
            patch_coord_v = math.floor(v) - 6
            if (
                marker_image.shape[1] - 12 > patch_coord_u >= 0
                and marker_image.shape[0] - 12 > patch_coord_v >= 0
            ):
                marker_image[
                    patch_coord_v : patch_coord_v + 12,
                    patch_coord_u : patch_coord_u + 12,
                ] = current_patch
        marker_image = marker_image[12:-12, 12:-12]

        return marker_image

def generate_patch_array(super_resolution_ratio=10):
    circle_radius = 3
    size_slot_num = 50
    base_circle_radius = 1.5

    patch_array = np.zeros(
        (
            super_resolution_ratio,
            super_resolution_ratio,
            size_slot_num,
            4 * circle_radius,
            4 * circle_radius,
        ),
        dtype=np.uint8,
    )
    for u in range(super_resolution_ratio):
        for v in range(super_resolution_ratio):
            for w in range(size_slot_num):
                img_highres = (
                    np.ones(
                        (
                            4 * circle_radius * super_resolution_ratio,
                            4 * circle_radius * super_resolution_ratio,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )
                center = np.array(
                    [
                        circle_radius * super_resolution_ratio * 2,
                        circle_radius * super_resolution_ratio * 2,
                    ],
                    dtype=np.uint8,
                )
                center_offseted = center + np.array([u, v])
                radius = round(base_circle_radius * super_resolution_ratio + w)
                img_highres = cv2.circle(
                    img_highres,
                    tuple(center_offseted),
                    radius,
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                    lineType=cv2.LINE_AA,
                )
                img_highres = cv2.GaussianBlur(img_highres, (17, 17), 15)
                img_lowres = cv2.resize(
                    img_highres,
                    (4 * circle_radius, 4 * circle_radius),
                    interpolation=cv2.INTER_CUBIC,
                )
                patch_array[u, v, w, ...] = img_lowres

    return {
        "base_circle_radius": base_circle_radius,
        "circle_radius": circle_radius,
        "size_slot_num": size_slot_num,
        "patch_array": patch_array,
        "super_resolution_ratio": super_resolution_ratio,
    }
