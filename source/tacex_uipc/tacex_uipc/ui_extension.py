import omni.ext
from isaacsim.util.debug_draw import _debug_draw

draw = None

# import omni.usd
from omni.physx import get_physx_interface

import numpy as np

import pxr
from pxr import Sdf, UsdGeom

# ui stuff
from isaacsim.gui.components.ui_utils import *
import omni.ui as ui

from tacex_uipc.objects.constraints import UipcIsaacAttachments
from tacex_uipc.utils import MeshGenerator, TetMeshCfg
from tacex_uipc.utils.create_surf_triangle_vis_material import (
    assign_material_to_mesh_with_usd,
    create_surf_tri_vis_material,
    add_barycentric_primvar,
)

from omni.physx.scripts import deformableUtils


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class TacexIPCExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[tacex_uipc] startup")

        global draw
        draw = _debug_draw.acquire_debug_draw_interface()

        self._window = omni.ui.Window(
            "TacEx UIPC Extension",
            width=500,
            height=400,
            dockPreference=omni.ui.DockPreference.RIGHT_BOTTOM,
        )
        self.sub = None
        self.playing = False

        with self._window.frame:
            with omni.ui.VStack(height=3):
                label = omni.ui.Label("Select a prim and push one of the buttons", alignment=omni.ui.Alignment.LEFT_TOP)
                ui.Spacer(height=6)
                self._build_tet_mesh_cfg_frame()

                ui.Spacer(height=6)

                def update_surf_mesh():
                    self._update_surf_mesh(get_selected_prim_path())

                omni.ui.Button("Update Surface Mesh", clicked_fn=update_surf_mesh, height=0)

                def create_attachment():
                    _create_attachment(get_selected_prim_paths())

                omni.ui.Button(
                    "Create Attachment \n(Select rigid body, then tet mesh, then press button)",
                    clicked_fn=create_attachment,
                    height=0,
                )

                # experimental
                def extract_primvar_st():
                    _extract_primvar_st(get_selected_prim_path())

                omni.ui.Button("Extract primvars:st values (uv map)", clicked_fn=extract_primvar_st, height=0)

                def set_primvar_st():
                    _set_primvar_st(get_selected_prim_path())

                omni.ui.Button("Set primvars:st values (uv map)", clicked_fn=set_primvar_st, height=0)

                ui.Spacer(height=6)

    def on_shutdown(self):
        print("[tacex_uipc] shutdown")

    """UI Builder Functions"""

    def _build_tet_mesh_cfg_frame(self):
        """Build the frame for the parameters which control the initial frame processing"""
        with ui.CollapsableFrame(
            title="Generate Tet Mesh",
            height=0,
            collapsed=False,
            style=get_style(),
            name="groupFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            with ui.VStack(height=0, spacing=5):
                ui.Spacer(height=6)

                self.stop_quality = int_builder(
                    default_val=8,
                    min=1,
                    max=20,
                    label="stop_quality",
                    tooltip="Max AMIPS energy for stopping mesh optimization. Larger means less optimization and sooner stopping.",
                )

                self.max_its = int_builder(
                    default_val=80,
                    min=1,
                    max=200,
                    label="max_its",
                    tooltip="Max number of mesh optimization iterations.",
                )

                self.epsilon_r = float_builder(
                    min=0.001,
                    max=0.05,
                    label="epsilon_r",
                    default_val=1e-2,
                    step=0.0001,
                    format="%.4f",
                    tooltip="Smaller = features are better preserves. Larger eplsion_r + larger edge_length_r = tetmesh with low res",
                )
                self.edge_length_r = float_builder(
                    min=0.0001,
                    max=0.1,
                    label="edge_length_r",
                    default_val=0.05,
                    step=0.0001,
                    format="%.4f",
                    tooltip="Relative target edge length. Smaller edge length gives denser mesh.",
                )

                self.skip_simplify = cb_builder(
                    type="checkbox",
                    label="skip_simplify",
                    default_val=False,
                    tooltip="If the tri mesh simplification prior to the tet mesh computation should be skipped.",
                    on_clicked_fn=None,
                )

                self.coarsen = cb_builder(
                    type="checkbox",
                    label="coarsen",
                    default_val=True,
                    tooltip="If the output mesh should be coarsend as much as possible.",
                    on_clicked_fn=None,
                )

                def generate_tet_mesh():
                    self._generate_tet_mesh(get_selected_prim_path())

                omni.ui.Button("Generate Tet Mesh", clicked_fn=generate_tet_mesh, height=0)

    def _generate_tet_mesh(self, path):
        """Generates a tetrahedra mesh for a USD trimesh.

        You need to make sure that the USD path belongs to the geom_mesh and not just the Xform of the prim.
        """
        tet_cfg = TetMeshCfg(
            stop_quality=self.stop_quality.as_int,
            max_its=self.max_its.as_int,
            epsilon_r=self.epsilon_r.as_float,
            edge_length_r=self.edge_length_r.as_float,
            skip_simplify=self.skip_simplify.as_bool,
            coarsen=self.coarsen.as_bool,
            log_level=6,
        )

        self.mesh_gen = MeshGenerator(tet_cfg)

        stage = omni.usd.get_context().get_stage()
        geom_mesh = UsdGeom.Mesh.Get(stage, path)
        tet_points, tet_indices, surf_points, surf_indices = self.mesh_gen.generate_tet_mesh_for_prim(geom_mesh)

        tf_world = np.array(omni.usd.get_world_transform_matrix(geom_mesh))
        world_tet_points = tf_world.T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
        world_tet_points = world_tet_points[:-1].T

        world_tet_surf_points = tf_world.T @ np.vstack((surf_points.T, np.ones(surf_points.shape[0])))
        world_tet_surf_points = world_tet_surf_points[:-1].T

        # no need for that anymor, because we now have the Triangle_outline material
        # draw.clear_points()
        # draw.clear_lines()
        # _draw_tets(world_tet_points, tet_indices)
        # _draw_surface_trimesh(world_tet_surf_points, surf_indices)

        # Dont save the transformed points ->  we want to save the local points. Transformations happens during scene creation
        # Otherwise, Transformation during scene init becomes messed up.
        _create_tet_data_attributes(
            path,
            tet_points=tet_points,
            tet_indices=tet_indices,
            tet_surf_points=surf_points,
            tet_surf_indices=surf_indices,
        )
        return (
            f"Amount of tet points {len(tet_points)},\nAmount of tetrahedra: {int(len(tet_indices) / 4)},\nAmount of"
            f" surface points: {int(len(surf_indices) / 3)}"
        )

    def _update_surf_mesh(self, path):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(pxr.Sdf.Path(path))
        print("Update surface mesh of prim: ", prim)

        # extract surface data of tet mesh
        surf_points = prim.GetAttribute("tet_surf_points").Get()
        tet_surf_indices = prim.GetAttribute("tet_surf_indices").Get()

        surf_points = np.array(surf_points)
        triangles = tet_surf_indices

        # MeshGenerator.update_usd_mesh(UsdGeom.Mesh(prim), surf_points=surf_points, triangles=triangles)
        # print("Updated Surface Mesh of ", path)

        # update surface based on uipc_mesh surface
        # MeshGenerator.update_usd_mesh_with_uipc_surface(prim)

        geom_mesh = UsdGeom.Mesh(prim)
        # _update_surf_mesh(path)
        self.mesh_gen.update_usd_mesh_remap_uv(geom_mesh, surf_points=surf_points, triangles=triangles)

        add_barycentric_primvar(UsdGeom.Mesh.Get(stage, path))
        print("Update of surface Mesh: ", path)

        # Create material that visualizes the tet mesh resolution

        mat_path = "/Materials/TriangleOutlineMat"
        mat = create_surf_tri_vis_material(mat_path)
        # bind material with normal usd api
        assign_material_to_mesh_with_usd(geom_mesh, mat)


"""Helper Functions"""


def get_selected_prim_path():
    """Return the path of the first selected prim"""
    context = omni.usd.get_context()
    selection = context.get_selection()
    paths = selection.get_selected_prim_paths()

    return None if not paths else paths[0]


def get_selected_prim_paths():
    """Return the paths of the selected prims"""
    context = omni.usd.get_context()
    selection = context.get_selection()
    paths = selection.get_selected_prim_paths()

    return paths


def get_stage_id():
    """Return the stage Id of the current stage"""
    context = omni.usd.get_context()
    return context.get_stage_id()


def _draw_tets(all_vertices, tet_indices):
    draw.clear_lines()

    # first draw the tet mesh nodes
    # draw.draw_points(all_vertices, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices))

    # connect nodes according to tet_indices
    color = [(125, 0, 0, 0.5)]
    line_size = 20
    for i in range(0, len(tet_indices), 4):
        tet_points_idx = tet_indices[i : i + 4]
        tet_points = [all_vertices[i] for i in tet_points_idx]
        # draw.draw_points(tet_points, [(255,0,0,1)]*len(all_vertices), [line_size]*len(all_vertices))
        draw.draw_lines(
            [tet_points[0]] * 3, tet_points[1:], color * 3, [line_size] * 3
        )  # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tet_points[1]] * 2, tet_points[2:], color * 2, [line_size] * 2)
        draw.draw_lines([tet_points[2]], [tet_points[3]], color, [line_size])  # draw line between the other 2 points


def _draw_surface_trimesh(all_vertices, tet_surf_indices):
    color = [(0, 0, 125, 0.5)]
    point_size = 15
    line_size = 20
    # draw surface mesh
    for i in range(0, len(tet_surf_indices), 3):
        tri_points_idx = tet_surf_indices[i : i + 3]
        tri_points = [all_vertices[j] for j in tri_points_idx]
        draw.draw_lines(
            [tri_points[0]] * 2, tri_points[1:], color * 2, [line_size] * 2
        )  # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tri_points[1]] * 1, tri_points[2:], color * 1, [line_size] * 1)
        draw.draw_points(tri_points, [(255, 255, 255, 1)] * len(tri_points), [point_size] * len(tri_points))


def _create_tet_data_attributes(path, tet_points, tet_indices, tet_surf_points, tet_surf_indices):
    """
    Creates an attribute for a prim that holds a boolean.
    See: https://graphics.pixar.com/usd/release/api/class_usd_prim.html.
    The attribute can then be found in the GUI under "Raw USD Properties" of the prim.
    Args:
        prim: A prim that should be holding the attribute.
        attribute_name: The name of the attribute to create.
    Returns:
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    attr_tet_points = prim.CreateAttribute("tet_points", pxr.Sdf.ValueTypeNames.Vector3fArray)
    attr_tet_points.Set(tet_points)
    attr_tet_points.SetCustom(True)

    attr_tet_indices = prim.CreateAttribute("tet_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_indices.Set(tet_indices)
    attr_tet_indices.SetCustom(True)

    attr_tet_points = prim.CreateAttribute("tet_surf_points", pxr.Sdf.ValueTypeNames.Vector3fArray)
    attr_tet_points.Set(tet_surf_points)
    attr_tet_points.SetCustom(True)

    attr_tet_surf_indices = prim.CreateAttribute("tet_surf_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_surf_indices.Set(tet_surf_indices)

    print("*" * 40)
    print("Created tet data: ")
    print(f"tet_points (num {tet_points.shape[0]})")
    print(f"tet_indices (num {len(tet_indices)})")
    print(f"tet_surf_points (num {len(tet_surf_points)})")
    print(f"tet_surf_indices (num {len(tet_surf_indices)})")
    print("*" * 40)


def _update_surf_mesh(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(pxr.Sdf.Path(path))
    print("prim ", prim)
    # extract surface data of tet mesh
    surf_points = prim.GetAttribute("tet_surf_points").Get()
    tet_surf_indices = prim.GetAttribute("tet_surf_indices").Get()

    surf_points = np.array(surf_points)

    triangles = tet_surf_indices
    MeshGenerator.update_usd_mesh(UsdGeom.Mesh(prim), surf_points=surf_points, triangles=triangles)
    # print("Updated Surface Mesh of ", path)

    # update surface based on uipc_mesh surface
    # MeshGenerator.update_usd_mesh_with_uipc_surface(prim)

    # add_barycentric_primvar(UsdGeom.Mesh.Get(stage, path))
    # print("Update of surface Mesh via UIPC: ", path)


def _create_attachment(paths):
    print("paths are ", paths)

    isaac_mesh_path = paths[0]
    tet_mesh_path = paths[1]

    # extract data of tet mesh
    stage = omni.usd.get_context().get_stage()
    tet_prim = stage.GetPrimAtPath(pxr.Sdf.Path(tet_mesh_path))
    tet_points = np.array(tet_prim.GetAttribute("tet_points").Get())
    tet_indices = tet_prim.GetAttribute("tet_indices").Get()

    # convert to world coordinates
    tf_world = np.array(omni.usd.get_world_transform_matrix(tet_prim))
    print("tf world ", tf_world)
    world_tet_points = tf_world.T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
    world_tet_points = world_tet_points[:-1].T

    # disable collision of the mesh that should be simulated by uipc -> otherwise raycasts are only detecting the tet mesh
    try:
        collision_enabled = tet_prim.GetAttribute("physics:collisionEnabled")
        collision_enabled.Set(False)
    except RuntimeError:
        pass

    attachment_offsets, idx, rigid_prims, attachment_points_positions, obj_pos = (
        UipcIsaacAttachments.compute_attachment_data(isaac_mesh_path, world_tet_points, tet_indices)
    )
    _create_attachment_data_attributes(tet_mesh_path, attachment_offsets, idx)

    # draw attachment data
    draw.draw_points(
        attachment_points_positions,
        [(255, 0, 0, 0.5)] * attachment_points_positions.shape[0],
        [30] * attachment_points_positions.shape[0],
    )  # the new positions
    obj_center = obj_pos

    for j in range(0, attachment_points_positions.shape[0]):
        draw.draw_lines([obj_center], [attachment_points_positions[j, :]], [(255, 255, 0, 0.5)], [10])

    get_physx_interface().release_physics_objects()


def _create_attachment_data_attributes(path, attachment_offsets, attachment_indices):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    attr_tet_points = prim.CreateAttribute("attachment_offsets", pxr.Sdf.ValueTypeNames.Vector3fArray)
    attr_tet_points.Set(attachment_offsets)

    attr_attachment_indices = prim.CreateAttribute("attachment_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_attachment_indices.Set(attachment_indices)

    print("*" * 40)
    print("Created tet data: ")
    print(f"attachment_offsets (num {attachment_offsets.shape[0]})")
    print(f"attachment_indices (num {len(attachment_indices)})")
    print("*" * 40)


# --- for some funky UV texture stuff (just experimental) ---
def _extract_primvar_st(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    pv_api = UsdGeom.PrimvarsAPI(UsdGeom.Mesh(prim))
    if not pv_api.HasPrimvar("primvars:st"):
        print("No primvars:st")
        return

    primvars_st = np.array(pv_api.GetPrimvar("primvars:st").Get())
    print("primvars:st has shape ", primvars_st.shape)
    np.save("./primvars_st.npy", primvars_st)


def _set_primvar_st(path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)

    # load uv coor from array
    uv_coor = np.load("./primvars_st.npy")

    pv_api = UsdGeom.PrimvarsAPI(UsdGeom.Mesh(prim))
    if not pv_api.HasPrimvar("primvars:st"):
        pv = pv_api.CreatePrimvar(
            "primvars:st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying, uv_coor.size
        )
    else:
        pv = pv_api.GetPrimvar("primvars:st")
    pv.Set(uv_coor)
    print("Set uv values for primvars:st")


# ---
