from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.usd
from pxr import Gf, UsdGeom

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np

import warp as wp
from uipc import (
    Matrix4x4,
    Quaternion,
    Transform,
    builtin,
    view,
)
from uipc.constitution import AffineBodyConstitution
from uipc.core import (
    AffineBodyStateAccessorFeature,
)
from uipc.geometry import (
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
    tetmesh,
)
from uipc.unit import MPa

import isaaclab.utils.math as math_utils

wp.init()


from tacex_uipc.utils import MeshGenerator, TetMeshCfg

from ..uipc_object import UipcObject
from .uipc_rigid_object_data import UipcRigidObjectData

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim

    from .uipc_rigid_object_cfg import UipcRigidObjectCfg


class UipcRigidObject(UipcObject):
    """A rigid object simulated with UIPC.

    The class handles initialization of Affine Bodies in UIPC and setups their rendering in Isaac Sim.

    """

    cfg: UipcRigidObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcRigidObjectCfg, uipc_sim: UipcSim):
        """Initialize the uipc object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg, uipc_sim)

    """
    Properties
    """

    @property
    def data(self) -> UipcRigidObjectData:
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 right now, since each object is a single rigid body.
        """
        return 1

    @property
    def backend_system_body_offset(self) -> int:
        geo_slot = self.geo_slot_list[0].geometry()
        abd_body_offset = geo_slot.meta().find(builtin.backend_abd_body_offset)

        return abd_body_offset.view()

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # TODO implement this
        pass
        # # resolve all indices
        # if env_ids is None:
        #     env_ids = slice(None)

    def write_data_to_sim(self):
        pass

    """
    Operations - Write to simulation.
    """

    def write_root_pose_to_sim(
        self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Sets pose [pos, quat] of the UIPC rigid body into the UIPC simulation.

        Args:
            pose: Pose that should be written into the UIPC sim for the object.
                  Pose consists of cartesian position and quaternion orientation in (w,x,y,z).
                  Shape is (len(env_ids), 7).
        """
        # resolve all indices
        # physx_env_ids = env_ids
        # if env_ids is None:
        #     env_ids = slice(None)
        #     physx_env_ids = self._ALL_INDICES

        # write pose
        # self._state_accessor.copy_to(self._state_geo)  # Get state data from backend
        pose_matrix = math_utils.make_pose(
            root_pose[:, :3], math_utils.matrix_from_quat(root_pose[:, 3:])
        )
        view(self._state_geo.transforms())[:, :] = pose_matrix.cpu()
        # Copy modified data back to backend
        self._state_accessor.copy_from(self._state_geo)

    def write_root_velocity_to_sim(
        self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None
    ):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame.
                           The velocities will be written into the UIPC sim for the object.
                           Shape is (len(env_ids), 6).
        """
        # resolve all indices
        # physx_env_ids = env_ids
        # if env_ids is None:
        #     env_ids = slice(None)
        #     physx_env_ids = self._ALL_INDICES

        # write velocity
        vel_buffer = view(self._state_geo.instances().find(builtin.velocity))
        vel_buffer = root_velocity.numpy().cpu()

        self._state_accessor.copy_to(self._state_geo)

    def update(self, dt: float):
        self._data.update(dt)

    """
    Internal helper.
    """

    def _setup_uipc_mesh(self):
        if self.cfg.mesh_cfg is None:
            # Load precomputed mesh data from USD prim.
            tet_points = np.array(self._usd_mesh_prim.GetAttribute("tet_points").Get())
            tet_indices = self._usd_mesh_prim.GetAttribute("tet_indices").Get()
            surf_indices = self._usd_mesh_prim.GetAttribute("tet_surf_indices").Get()

            if tet_indices is None:
                print(
                    f"No precomputed tet mesh data found for prim at {self._usd_mesh_prim.GetPath()}... Creating a tet mesh with default config..."
                )
                mesh_gen = MeshGenerator(config=TetMeshCfg())
                tet_points, tet_indices, surf_points, surf_indices = (
                    mesh_gen.generate_tet_mesh_for_prim(self._usd_geom_mesh)
                )
            else:
                print(
                    f"Using precomputed mesh data; num_tet_points={tet_points.shape[0]}"
                )
        else:
            mesh_gen = MeshGenerator(config=self.cfg.mesh_cfg)
            if type(self.cfg.mesh_cfg) is TetMeshCfg:
                tet_points, tet_indices, surf_points, surf_indices = (
                    mesh_gen.generate_tet_mesh_for_prim(self._usd_geom_mesh)
                )

        tf_world = omni.usd.get_world_transform_matrix(self._usd_geom_mesh)

        translation = np.array(tf_world.ExtractTranslation())
        rotation = math_utils.quat_from_matrix(
            torch.tensor(np.array(tf_world.ExtractRotationMatrix()))
        )
        scale = np.array(
            Gf.Vec3d(*(v.GetLength() for v in tf_world.ExtractRotationMatrix()))
        )

        # workaround for bug with kinematic objects -> somehow mesh is scaled additionally/twice?
        if self.cfg.constitution_cfg.kinematic:
            scale = [1, 1, 1]

        scale_mat = np.array(
            [
                [scale[0], 0, 0],
                [0, scale[1], 0],
                [0, 0, scale[2]],
            ]
        )
        # scale the local mesh points
        tet_points = tet_points @ scale_mat

        # uipc wants 2D array
        tet_indices = np.array(tet_indices).reshape(-1, 4)
        surf_indices = np.array(surf_indices).reshape(-1, 3)

        # create uipc mesh with scaled (local) tet points
        uipc_mesh = tetmesh(tet_points.copy(), tet_indices.copy())

        # enable contact for uipc meshes etc.
        label_surface(uipc_mesh)
        label_triangle_orient(uipc_mesh)
        # flip the triangles inward for better rendering
        uipc_mesh = flip_inward_triangles(
            uipc_mesh
        )  # NOTE idk if this makes a difference for us

        # set transform of uipc ABD body
        trans_view = view(uipc_mesh.transforms())
        t = Transform.Identity()
        t.translate(translation)
        t.rotate(Quaternion(np.array(rotation)))

        trans_view[0] = np.array(tf_world).T.copy()

        # uipc_mesh = self.uipc_meshes[0] #todo code properly cloned envs (i.e. for instanced objects?)

        # create and apply the constitution for the affine body
        stiffness = self.cfg.constitution_cfg.m_kappa
        constitution = AffineBodyConstitution()
        constitution.apply_to(
            uipc_mesh, stiffness * MPa, mass_density=self.cfg.mass_density
        )

        if self.cfg.constitution_cfg.kinematic:
            # make ABD body kinematic
            is_fixed_attr = uipc_mesh.instances().find(builtin.is_fixed)
            view(is_fixed_attr)[0] = 1

        # apply the default contact model to the base mesh
        default_element = self._uipc_sim.scene.contact_tabular().default_element()
        default_element.apply_to(uipc_mesh)

        return uipc_mesh

    def _initialize_impl(self):
        # log information the uipc body
        omni.log.info(f"UIPC Rigid Body initialized at: {self.cfg.prim_path}.")
        omni.log.info(f"Number of instances: {self.num_instances}")

        self._state_accessor: AffineBodyStateAccessorFeature = (
            self._uipc_sim.world.features().find(AffineBodyStateAccessorFeature)
        )

        # create buffers
        self._data = UipcRigidObjectData(
            self._uipc_sim, self, self.device
        )  # container for data access
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the uipc_object data
        self.update(0.0)

        # add this object to the list of all uipc objects in the world
        self._uipc_sim.uipc_objects.append(self)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(
            self.num_instances, dtype=torch.long, device=self.device
        )

        # Create a state_geo to contain data
        self._state_geo = self._state_accessor.create_geometry(
            body_offset=self.backend_system_body_offset, body_count=self.num_bodies
        )
        # tell the backend we need transform information
        self._state_geo.instances().create(builtin.transform, Matrix4x4.Zero())
        # # also need velocity information
        # self._state_geo.instances().create(builtin.velocity, Vector3.Zero())

        #     # set information about rigid body into data
        #     self._data.body_names = self.body_names
        #     self._data.default_mass = self.root_physx_view.get_masses().clone()
        #     self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    # self._data.default_root_state = (self.num_instances,)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state

        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            # + tuple(self.cfg.init_state.lin_vel)
            # + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(
            default_root_state, dtype=torch.float, device=self.device
        )
        self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

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
