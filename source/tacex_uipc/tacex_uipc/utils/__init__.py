from .mesh_gen import MeshGenerator, TetMeshCfg, TriMeshCfg, VoxelTetMeshCfg
from .spawn_from_msh import create_prim_for_tet_data, create_prim_for_uipc_scene_object
from .create_surf_triangle_vis_material import (
    add_barycentric_primvar,
    create_surf_tri_vis_material,
    assign_material_to_mesh_with_usd,
)
from .create_deformation_vis_material import (
    add_deform_primvar,
    create_deform_vis_material,
)
from .uipc_contact_info import ContactInfo
from .debug_draw import DebugDraw

import os

TACEX_UIPC_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
