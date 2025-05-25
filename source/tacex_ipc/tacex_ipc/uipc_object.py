# Copyright (c) 2022-2025, The TacEx Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import omni.usd
import usdrt
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils

from isaaclab.utils import configclass
from isaaclab.assets import AssetBase, AssetBaseCfg



import uipc
from uipc import builtin, view
from uipc.core import Engine, World, Scene, SceneIO
from uipc import Vector3, Vector2, Transform, Logger, Quaternion, AngleAxis
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa

import numpy as np
import warp as wp
wp.init()

if TYPE_CHECKING:
    from tacex_ipc import UipcSim

@configclass
class UipcObjectCfg(AssetBaseCfg):
    constitution_type: str = "AffineBodyConstitution"

    # contact_model: 

class UipcObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: UipcObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcObjectCfg, uipc_sim: UipcSim):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        self._uipc_sim = uipc_sim
        super().__init__(cfg)
        
    """
    Properties
    """

    @property
    def data(self):
        return None
        #return self._data

    @property
    def num_instances(self) -> int:
        return self._prim_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def uipc_sim(self) -> physx.RigidBodyView:
        """uipc simulation instance of this uipc object.

        """
        return self._uipc_sim

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # reset external wrench
        self._external_force_b[env_ids] = 0.0
        self._external_torque_b[env_ids] = 0.0

    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # write external wrench
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=False,
            )

    def update(self, dt: float):
        pass
        #self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """

        # set into simulation
        self.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_com_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # set into simulation
        self.write_root_com_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_link_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # set into simulation
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_link_state_w[env_ids, :7] = root_pose.clone()
        self._data.root_state_w[env_ids, :7] = self._data.root_link_state_w[env_ids, :7]
        # convert root quaternion from wxyz to xyzw
        root_poses_xyzw = self._data.root_link_state_w[:, :7].clone()
        root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # set into simulation
        self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)

    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            local_env_ids = slice(env_ids)
        else:
            local_env_ids = env_ids

        com_pos = self.data.com_pos_b[local_env_ids, 0, :]
        com_quat = self.data.com_quat_b[local_env_ids, 0, :]

        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[..., :3],
            root_pose[..., 3:7],
            math_utils.quat_rotate(math_utils.quat_inv(com_quat), -com_pos),
            math_utils.quat_inv(com_quat),
        )

        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
        self.write_root_link_pose_to_sim(root_pose=root_link_pose, env_ids=env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, 7:] = root_velocity.clone()
        self._data.body_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_velocities(self._data.root_state_w[:, 7:], indices=physx_env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """

        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_com_state_w[env_ids, 7:] = root_velocity.clone()
        self._data.root_state_w[env_ids, 7:] = self._data.root_com_state_w[env_ids, 7:]
        self._data.body_acc_w[env_ids] = 0.0
        # set into simulation
        self.root_physx_view.set_velocities(self._data.root_com_state_w[:, 7:], indices=physx_env_ids)

    """
    Internal helper.
    """

    def _initialize_impl(self):
        
        prim_paths_expr = self.cfg.prim_path + "/mesh"
        print(f"Initializing uipc objects {prim_paths_expr}...")
        self._prim_view = XFormPrim(prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}", usd=False)
        self._prim_view.initialize()
        # Check that sizes are correct
        # if self._view.count != self._num_envs:
        #     raise RuntimeError(
        #         f"Number of sensor prims in the view ({self._view.count}) does not match"
        #         f" the number of environments ({self._num_envs})."
        #     )

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        
        self.uipc_meshes = []
        self.objects = []
        # setup tet meshes for uipc
        for prim in self._prim_view.prims:
            tet_points = np.array(prim.GetAttribute("tet_points").Get())
            tet_indices = np.array(prim.GetAttribute("tet_indices").Get()).reshape(-1,4) # uipc wants 2D array
            tet_surf_indices = np.array(prim.GetAttribute("tet_surf_indices").Get()).reshape(-1,3)

            # setup mesh updates via Fabric
            fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(str(prim.GetPrimPath())))
            if not fabric_prim:
                print(f"Prim at path {prim.GetPrimPath()} is not in Fabric")
            if not fabric_prim.HasAttribute("points"):
                print(f"Prim at path {prim.GetPrimPath()} does not have points attribute")

            # Tell OmniHydra to render points from Fabric
            if not fabric_prim.HasAttribute("Deformable"):
                fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

            mesh = tetmesh(tet_points, tet_indices)
            self.uipc_meshes.append(mesh)

            # create constitution and contact model
            abd = AffineBodyConstitution()
            # friction ratio and contact resistance
            self._uipc_sim.scene.contact_tabular().default_model(0.5, 1.0 * GPa)
            default_element = self._uipc_sim.scene.contact_tabular().default_element()
            
            # apply the constitution and contact model to the base mesh
            abd.apply_to(mesh, 100 * MPa) # stiffness (hardness) of 100 MPa (= hard-rubber-like material)
            # apply the default contact model to the base mesh
            default_element.apply_to(mesh)

            # enable the contact by labeling the surface 
            label_surface(mesh)

            # create objects
            obj = self._uipc_sim.scene.objects().create(self.cfg.prim_path)
            obj_geo_slot, _ = obj.geometries().create(mesh)
            self.objects.append(obj_geo_slot)

        # log information about the rigid body
        omni.log.info(f"UIPC body initialized at: {self.cfg.prim_path}.")
        omni.log.info(f"Number of instances: {self.num_instances}")

        # create buffers
        # self._create_buffers()
        # process configuration
        # self._process_cfg()
        # update the rigid body data
        self.update(0.0)

        self._uipc_sim.uipc_objects.append(self)
        self.sio = SceneIO(self._uipc_sim.scene)

    # def _create_buffers(self):
    #     """Create buffers for storing data."""
    #     # constants
    #     self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

    #     # external forces and torques
    #     self.has_external_wrench = False
    #     self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
    #     self._external_torque_b = torch.zeros_like(self._external_force_b)

    #     # set information about rigid body into data
    #     self._data.body_names = self.body_names
    #     self._data.default_mass = self.root_physx_view.get_masses().clone()
    #     self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    # def _process_cfg(self):
    #     """Post processing of configuration parameters."""
    #     # default state
    #     # -- root state
    #     # note: we cast to tuple to avoid torch/numpy type mismatch.
    #     default_root_state = (
    #         tuple(self.cfg.init_state.pos)
    #         + tuple(self.cfg.init_state.rot)
    #         + tuple(self.cfg.init_state.lin_vel)
    #         + tuple(self.cfg.init_state.ang_vel)
    #     )
    #     default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
    #     self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    def _update_meshes(self):
        for i, prim_path in enumerate(self._prim_view.prim_paths):
            #uipc_points = self.objects[i].geometry().positions().view()[:,:,0] #todo check how it can be optimized -> how can we use third dim effectively?
            trimesh = self.sio.simplicial_surface(2).positions().view().reshape(-1,3)
            print("Updated points ", trimesh)

            # tf_view = view(self.objects[i].geometry().transforms())[0]
            # print("Transformation matrix ")
            # print(tf_view)

            fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))
            mesh_points = fabric_prim.GetAttribute("points")
            mesh_points.Set(usdrt.Vt.Vec3fArray(trimesh))
            # # update Transform
            # fabric_id = self.stage.GetFabricId()
            # hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(fabric_id, omni.usd.get_context().get_stage_id())
            # local_xform = hier.get_local_xform(usdrt.Sdf.Path(prim_path))
            # print("local xform ", local_xform)
            # print("new xform: ", usdrt.Gf.Matrix4d(tf_view))
            # hier.set_local_xform(usdrt.Sdf.Path(prim_path), usdrt.Gf.Matrix4d(tf_view))
            # # hier.set_world_xform(path, Gf.Matrix4d(1))
            # hier.update_world_xforms()

            # rtxformable = usdrt.Rt.Xformable(fabric_prim)
            # # Generate a random orientation quaternion
            # import random, math
            # angle = random.random()*math.pi*2
            # axis = usdrt.Gf.Vec3f(random.random(), random.random(), random.random()).GetNormalized()
            # halfangle = angle/2.0
            # shalfangle = math.sin(halfangle)
            # rotation = usdrt.Gf.Quatf(math.cos(halfangle), axis[0]*shalfangle, axis[1]*shalfangle, axis[2]*shalfangle)

            # rtxformable.GetWorldOrientationAttr().Set(rotation)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
