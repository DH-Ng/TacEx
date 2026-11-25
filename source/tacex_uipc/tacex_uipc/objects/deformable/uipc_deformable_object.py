from __future__ import annotations

import torch
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING
import weakref

import omni.log
import omni.usd
import usdrt
import usdrt.UsdGeom
from pxr import UsdGeom, Sdf


import numpy as np

import warp as wp
from uipc import Vector3, builtin, view
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.core import FiniteElementStateAccessorFeature, ContactSystemFeature
from uipc.geometry import (
    SimplicialComplex,
    Geometry,
    flip_inward_triangles,
    label_surface,
    label_triangle_orient,
    tetmesh,
)
from uipc.unit import MPa

wp.init()


from tacex_uipc.utils import ContactInfo, DebugDraw, MeshGenerator, TetMeshCfg, VoxelTetMeshCfg
from tacex_uipc.utils.create_deformation_vis_material import (
    StableNeoHookeanField,
    create_deform_vis_material,
    add_deform_primvar,
    compute_residuals_and_magnitudes,
    normalize_deform_values,
)
# from tacex_assets import TACEX_ASSETS_DATA_DIR cannot import things from tacex_assets, otherwise circular import problem

from ..uipc_object import UipcObject
from .uipc_deformable_object_data import UipcDeformableObjectData

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim
    from .uipc_deformable_object_cfg import UipcDeformableObjectCfg


class UipcDeformableObject(UipcObject):
    """A deformable object simulated in UIPC.

    The class handles initialization of Deformable Bodies in UIPC and setups their rendering in Isaac Sim.

    """

    cfg: UipcDeformableObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcDeformableObjectCfg):
        """Initialize the uipc object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        self._elastic_field: StableNeoHookeanField | None = None
        self.rest_points: np.ndarray | None = None
        if self.cfg.debug_deformation_vis:
            self._setup_deformation_vis()

    """
    Properties
    """

    @property
    def data(self) -> UipcDeformableObjectData:
        return self._data

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def backend_system_vertex_offset(self) -> int:
        geo_slot = self.geo_slot_list[0].geometry()
        fem_vertex_offset = geo_slot.meta().find(builtin.backend_fem_vertex_offset)

        return fem_vertex_offset.view()

    @property
    def default_nodal_state_w(self) -> torch.Tensor:
        """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.

        Shape is (num_instances, max_sim_vertices_per_body, 6).
        """

        return self._data.default_nodal_state_w

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # TODO implement this

        # # resolve all indices
        # if env_ids is None:
        #     env_ids = slice(None)

        # take current pos for rest_points_pos for debug vis:
        if self.cfg.debug_deformation_vis:
            fabric_prim = self.fabric_prim
            self.rest_points = np.array(fabric_prim.GetAttribute("points").Get())

            # make sure that primvar interpolation is vertex - need to set it explicitly for the usdrt prim!
            usdrt.UsdGeom.PrimvarsAPI(fabric_prim).GetPrimvar(self.cfg.deformation_vis_cfg.primvar_name).SetInterpolation(
                usdrt.UsdGeom.Tokens.vertex
            )

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Write to simulation.
    """

    # def write_nodal_state_to_sim(self, nodal_state: torch.Tensor, env_ids: Sequence[int] | None = None):
    #     """Set the nodal state over selected environment indices into the simulation.

    #     The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
    #     a translational component. All the quantities are in the simulation frame.

    #     Args:
    #         nodal_state: Nodal state in simulation frame.
    #             Shape is (len(env_ids), max_sim_vertices_per_body, 6).
    #         env_ids: Environment indices. If None, then all indices are used.
    #     """
    #     # set into simulation
    #     self.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)
    #     self.write_nodal_velocity_to_sim(nodal_state[..., 3:], env_ids=env_ids)

    def write_nodal_pos_to_sim(self, nodal_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal positions over selected environment indices into the simulation.

        The nodal position comprises of individual nodal positions of the simulation mesh for the deformable body.
        The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions in simulation frame.
                Shape is (len(env_ids), max_sim_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        # physx_env_ids = env_ids
        # if env_ids is None:
        #     env_ids = slice(None)
        #     physx_env_ids = self._ALL_INDICES

        # # note: we need to do this here since tensors are not set into simulation until step.
        # # set into internal buffers
        # self._data.nodal_pos_w[env_ids] = nodal_pos.clone()
        # # set into simulation
        # self.root_physx_view.set_sim_nodal_positions(self._data.nodal_pos_w, indices=physx_env_ids)

        self._state_accessor.copy_to(self._state_geo)  # Get state data from backend
        view(self._state_geo.positions())[:] = nodal_pos.reshape(-1, 3, self.num_instances).cpu().numpy()
        self._state_accessor.copy_from(self._state_geo)  # Copy modified data back to backend

    # def write_nodal_velocity_to_sim(self, nodal_vel: torch.Tensor, env_ids: Sequence[int] | None = None):
    #     """Set the nodal velocity over selected environment indices into the simulation.

    #     The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
    #     body. Since these are nodes, the velocity only has a translational component. The velocities are in the
    #     simulation frame.

    #     Args:
    #         nodal_vel: Nodal velocities in simulation frame.
    #             Shape is (len(env_ids), max_sim_vertices_per_body, 3).
    #         env_ids: Environment indices. If None, then all indices are used.
    #     """
    #     # resolve all indices
    #     physx_env_ids = env_ids
    #     if env_ids is None:
    #         env_ids = slice(None)
    #         physx_env_ids = self._ALL_INDICES
    #     # note: we need to do this here since tensors are not set into simulation until step.
    #     # set into internal buffers
    #     self._data.nodal_vel_w[env_ids] = nodal_vel.clone()
    #     # set into simulation
    #     self.root_physx_view.set_sim_nodal_velocities(self._data.nodal_vel_w, indices=physx_env_ids)

    def write_nodal_kinematic_target_to_sim(self, targets: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the kinematic targets of the simulation mesh for the deformable bodies indicated by the indices.

        The kinematic targets comprise of individual nodal positions of the simulation mesh for the deformable body
        and a flag indicating whether the node is kinematically driven or not. The positions are in the simulation frame.

        Note:
            The flag is set to 0.0 for kinematically driven nodes and 1.0 for free nodes.

        Args:
            targets: The kinematic targets comprising of nodal positions and flags.
                Shape is (len(env_ids), max_sim_vertices_per_body, 4).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # store into internal buffers
        self._data.nodal_kinematic_target[env_ids] = targets.clone()
        # set into simulation
        self.root_physx_view.set_sim_kinematic_targets(self._data.nodal_kinematic_target, indices=physx_env_ids)

    """
    Operations - Finders.
    """

    # def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
    #     """Find bodies in the rigid body based on the name keys.

    #     Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
    #     information on the name matching.

    #     Args:
    #         name_keys: A regular expression or a list of regular expressions to match the body names.
    #         preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

    #     Returns:
    #         A tuple of lists containing the body indices and names.
    #     """
    #     return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Internal helper.
    """

    def _setup_uipc_mesh(self) -> SimplicialComplex:
        # Load precomputed mesh data from USD prim.
        if self.cfg.mesh_cfg is None:
            tet_points = np.array(self._usd_mesh_prim.GetAttribute("tet_points").Get())
            tet_indices = self._usd_mesh_prim.GetAttribute("tet_indices").Get()
            surf_indices = self._usd_mesh_prim.GetAttribute("tet_surf_indices").Get()

            if tet_indices is None:
                print(
                    f"No precomputed tet mesh data found for prim at {self._usd_mesh_prim.GetPath()}... Creating a tet mesh with default config..."
                )
                mesh_gen = MeshGenerator(config=TetMeshCfg())
                tet_points, tet_indices, surf_points, surf_indices = mesh_gen.generate_tet_mesh_for_prim(
                    self._usd_geom_mesh
                )
        else:
            mesh_gen = MeshGenerator(config=self.cfg.mesh_cfg)
            if isinstance(self.cfg.mesh_cfg, (TetMeshCfg, VoxelTetMeshCfg)):
                tet_points, tet_indices, surf_points, surf_indices = mesh_gen.generate_tet_mesh_for_prim(
                    self._usd_geom_mesh
                )

        # transform local tet points to world coor for uipc
        tf_world = omni.usd.get_world_transform_matrix(self._usd_geom_mesh)

        tet_points_world = np.array(tf_world).T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
        tet_points_world = tet_points_world[:-1].T

        self.init_world_transform = torch.tensor(np.array(tf_world).T.copy(), device=self.uipc_sim.cfg.device)

        # uipc wants 2D array
        tet_indices = np.array(tet_indices).reshape(-1, 4)
        # surf_indices = np.array(surf_indices).reshape(-1, 3)

        # create uipc mesh
        uipc_mesh = tetmesh(tet_points_world.copy(), tet_indices.copy())

        # enable contact for uipc meshes etc.
        label_surface(uipc_mesh)

        label_triangle_orient(uipc_mesh)
        # flip the triangles inward for better rendering
        uipc_mesh = flip_inward_triangles(uipc_mesh)  # NOTE idk if this makes a difference for us

        # uipc_mesh = self.uipc_meshes[0] #todo code properly cloned envs (i.e. for instanced objects?)

        # create and apply the constitution for the deformable body
        youngs = self.cfg.constitution_cfg.youngs_modulus
        poisson = self.cfg.constitution_cfg.poisson_rate
        moduli = ElasticModuli.youngs_poisson(youngs * MPa, poisson)

        constitution = StableNeoHookean()
        constitution.apply_to(uipc_mesh, moduli, mass_density=self.cfg.mass_density)

        # apply the default contact model to the base mesh
        default_element = self._uipc_sim.scene.contact_tabular().default_element()
        default_element.apply_to(uipc_mesh)

        return uipc_mesh

    def _initialize_impl(self):
        # log information the uipc body
        omni.log.info(f"UIPC Deformable Body initialized at: {self.cfg.prim_path}.")
        omni.log.info(f"Number of instances: {self.num_instances}")

        self._state_accessor: FiniteElementStateAccessorFeature = self._uipc_sim.world.features().find(
            FiniteElementStateAccessorFeature
        )
        # create buffers
        self._data = UipcDeformableObjectData(self, self.device)  # container for data access
        self._create_buffers()

        # process configuration
        self._process_cfg()

        # update the uipc_object data
        self.update(0.0)

        # reset internal buffers
        self.reset()

        # add our uipc object to the list of all uipc objects of our uipc simulation
        self._uipc_sim.uipc_objects.append(self)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # Create a state_geo to contain data
        self._state_geo = self._state_accessor.create_geometry(
            vertex_offset=self.backend_system_vertex_offset, vertex_count=self.vertex_count
        )
        self._state_geo.vertices().create(
            builtin.position, Vector3.Zero()
        )  # tell the backend we need position information

        # self._data._nodal_pos_w = torch.zeros(self.num_instances, self._vertex_count)

        #     # set information about rigid body into data
        #     self._data.body_names = self.body_names
        #     self._data.default_mass = self.root_physx_view.get_masses().clone()
        #     self._data.default_inertia = self.root_physx_view.get_inertias().clone()

        # save initial world vertex positions
        geom = self._uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(self.obj_id)
        nodal_positions = torch.tensor(
            geo_slot.geometry().positions().view().copy().reshape(self.num_instances, -1, 3), device=self.device
        )
        nodal_velocities = torch.zeros_like(nodal_positions)
        self._data.default_nodal_state_w = torch.cat((nodal_positions, nodal_velocities), dim=-1)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
            # + tuple(self.cfg.init_state.lin_vel)
            # + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        # self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    def _sim_nodal_positions_w(self) -> np.ndarray:
        """Current positions of all tet mesh vertices in world frame (FEM positions are stored in world frame)."""
        geo_slot, _ = self._uipc_sim.scene.geometries().find(self.obj_id)
        return geo_slot.geometry().positions().view().reshape(-1, 3)

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

    """
    Deformation visualization.
    """

    def _setup_deformation_vis(self):
        """Creates the deformation vis material + primvar and the data for computing the elastic quantities."""
        vis_cfg = self.cfg.deformation_vis_cfg
        if vis_cfg.quantity not in ("energy", "force", "displacement"):
            raise ValueError(
                f"Unknown deformation vis quantity '{vis_cfg.quantity}'. Options: 'energy', 'force', 'displacement'."
            )

        add_deform_primvar(self._render_geom_mesh, primvar_name=vis_cfg.primvar_name)
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(vis_cfg.material_path).IsValid():
            create_deform_vis_material(vis_cfg.material_path, primvar_name=vis_cfg.primvar_name)

        # bind material with fabric
        rel = self.fabric_prim.GetRelationship(usdrt.UsdShade.Tokens.materialBinding)
        rel.SetTargets([vis_cfg.material_path])

        # topology of the simulation mesh: tets and the tet vertices that form the simulation surface.
        # `extract_surface` keeps the vertices with `is_surf` in ascending order -> same order as the surface points
        uipc_mesh = self.uipc_meshes[0]
        is_surf = np.asarray(uipc_mesh.vertices().find(builtin.is_surf).view()).reshape(-1)
        self._surf_vertex_ids = np.flatnonzero(is_surf)
        tets = np.asarray(uipc_mesh.tetrahedra().topo().view()).reshape(-1, 4)
        rest_points = np.asarray(uipc_mesh.positions().view()).reshape(-1, 3)

        constitution_cfg = self.cfg.constitution_cfg
        self._elastic_field = StableNeoHookeanField(
            rest_points,
            tets,
            youngs_modulus=constitution_cfg.youngs_modulus * MPa,
            poisson_rate=constitution_cfg.poisson_rate,
            device=self._uipc_sim.cfg.device if torch.cuda.is_available() else "cpu",
        )
        self._surf_vertex_ids_t = torch.as_tensor(self._surf_vertex_ids, device=self._elastic_field.device)
        self._bbox_diag = float(np.linalg.norm(np.ptp(rest_points, axis=0)))

        self.elastic_state = None
        """Elastic quantities of the last deformation vis update (:class:`ElasticState`), None for "displacement"."""

    def compute_elastic_state(self):
        """Computes elastic strain energy (density) and elastic nodal forces of the current simulation state.

        Requires ``debug_deformation_vis``.

        Returns:
            :class:`~tacex_uipc.utils.create_deformation_vis_material.ElasticState` for all tet mesh vertices.
        """
        if self._elastic_field is None:
            raise RuntimeError(f"Deformation vis is not enabled for {self.cfg.prim_path} (`debug_deformation_vis`).")
        return self._elastic_field.compute(self._sim_nodal_positions_w())

    def update_deformation_vis(self, surf_points_w: np.ndarray, render_points_w: np.ndarray):
        """Updates the deformation vis primvar of the render mesh (and the force arrows).

        Called by :meth:`UipcSim.update_render_meshes` after the render mesh points were updated.

        Args:
            surf_points_w: Current simulation surface vertex positions in world frame.
            render_points_w: Current render mesh vertex positions in world frame.
        """
        if self._elastic_field is None:
            return
        vis_cfg = self.cfg.deformation_vis_cfg

        if vis_cfg.quantity == "displacement":
            # displacement after rigid alignment (Kabsch) of the render mesh
            render_points_w = np.asarray(render_points_w, dtype=np.float64).reshape(-1, 3)
            if getattr(self, "rest_points", None) is None:
                self.rest_points = render_points_w.copy()
            _, mags, _ = compute_residuals_and_magnitudes(self.rest_points, render_points_w)
            render_values = torch.as_tensor(mags)
            self.elastic_state = None
        else:
            self.elastic_state = self.compute_elastic_state()
            if vis_cfg.quantity == "energy":
                sim_values = self.elastic_state.energy_density
            else:
                sim_values = self.elastic_state.forces.norm(dim=-1)
            render_values = self._sim_values_to_render_vertices(sim_values)

        floor = vis_cfg.auto_range_floor
        if floor is None:
            floor = {"energy": 1e-3, "force": 1e-6, "displacement": 1e-6}[vis_cfg.quantity]
        normalized = normalize_deform_values(
            render_values,
            value_range=vis_cfg.value_range,
            log_scale=vis_cfg.log_scale,
            log_decades=vis_cfg.log_decades,
            auto_range_floor=floor,
        )
        UsdGeom.PrimvarsAPI(self._render_geom_mesh).GetPrimvar(vis_cfg.primvar_name).Set(
            normalized.to(torch.float32).cpu().numpy()
        )

        if vis_cfg.draw_force_arrows:
            self._draw_force_arrows(surf_points_w)

    def _sim_values_to_render_vertices(self, sim_values: torch.Tensor) -> torch.Tensor:
        """Maps per tet vertex values onto the render mesh vertices."""
        if self._render_mapper is None:
            # render mesh == simulation surface
            return sim_values[self._surf_vertex_ids_t]
        if self._render_mapper.method == "surface":
            sim_values = sim_values[self._surf_vertex_ids_t]
        return self._render_mapper.interpolate(sim_values)

    def _draw_force_arrows(self, surf_points_w: np.ndarray):
        """Draws the elastic forces of the simulation surface vertices as lines (own :class:`DebugDraw`)."""
        if self.elastic_state is None:
            self.elastic_state = self.compute_elastic_state()
        vis_cfg = self.cfg.deformation_vis_cfg
        forces = self.elastic_state.forces[self._surf_vertex_ids_t].cpu().numpy()
        mags = np.linalg.norm(forces, axis=1)
        max_mag = float(mags.max()) if mags.size else 0.0
        if getattr(self, "_force_draw", None) is None:
            name = re.sub(r"[^A-Za-z0-9_]", "_", self.cfg.prim_path.strip("/"))
            self._force_draw = DebugDraw(f"/Visuals/UipcForces/{name}")
        if max_mag <= 0.0:
            self._force_draw.set_lines(np.zeros((0, 3)), np.zeros((0, 3)))  # no forces -> no arrows
            return
        mask = mags >= vis_cfg.force_arrow_min_ratio * max_mag
        scale = vis_cfg.force_arrow_scale
        if scale is None:
            scale = 0.1 * self._bbox_diag / max_mag
        starts = np.asarray(surf_points_w, dtype=np.float64)[mask]
        ends = starts + scale * forces[mask]
        width = vis_cfg.force_arrow_width if vis_cfg.force_arrow_width is not None else 3e-3 * self._bbox_diag
        self._force_draw.set_lines(starts, ends, colors=vis_cfg.force_arrow_color, sizes=width)
