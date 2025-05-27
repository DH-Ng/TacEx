"""
Script for computing precomputing tet mesh data with Wildmeshing.

Usage:
1. Select the **xForm** of the mesh for which you want to compute a tet mesh
2. Call the script from the script editor in the Isaac Sim GUI.
3. Run the script

Important:
Currently the script can only be called once!
(Otherwise gipc simulation won't work properly and no attachment points are computed)
If you want to call it again, you need to restart Isaac Sim.



Returns:
    _type_: _description_
"""

import torch
import traceback
import numpy as np

import carb
import omni
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.utils import export_prim_to_file
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import transform_points
import omni.isaac.lab.utils.math as orbit_math

from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

import omni.isaac.core.utils.xforms as xform_utils
from omni.physx.scripts import deformableUtils

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt
from omni.isaac.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrimView
import omni.isaac.core.utils.stage as stage_utils

from omni.isaac.core.utils.collisions import ray_cast
#from omni.isaac.core.utils.bounds import create_bbox_cache, compute_aabb
from omni.physx import get_physx_cooking_interface, get_physx_interface, get_physx_scene_query_interface


#! own stuff
from pathlib import Path
parent_path = str(Path(__file__).resolve().parent.parent)   
print("parent path is ", parent_path) # /workspace/isaaclab/data_storage/TacEx 

path = parent_path + "/examples"
print("Path to robot config: ", path)
import sys
sys.path.insert(0, path)

import gipc_isaac
from gipc_isaac.sim_gipc import SimGIPC, CfgGIPC 
from gipc_isaac.object_gipc import ObjectsGIPC
from gipc_isaac.tet_mesh_generation import TetMeshGenerator, CfgTetMesh


def standalone_create_objects(gipc_sim, prim_paths_expr: str, name: str, debug_view=False, tet_cfg=None, mesh_name="mesh"):
    objects_gipc = ObjectsGIPC(prim_paths_expr, name, debug_view)
    #self._objs_gipc_list.append(obj_gipc)
    _load_meshes(gipc_sim, objects_gipc, tet_cfg, mesh_name=mesh_name)
    return objects_gipc

def compute_tet_mesh_for_usd_prim(objects_gipc, tet_cfg=None, mesh_name="mesh"):        
    """
    Need to make sure that we get the mesh in USD and not just the Xform of the mesh
    """
    prim_view = objects_gipc.prim_view
    objects_gipc.object_idx_offsets.append(gipc_sim.body_vertex_idx_range[-1]) # take last idx as first 

    tet_mesh_points = None
    tet_indices = None

    stage = omni.usd.get_context().get_stage()
    for i, prim_path in enumerate(prim_view.prim_paths):
        mesh_path = prim_path + f"/{mesh_name}"
        geom_mesh = UsdGeom.Mesh.Get(stage, mesh_path)
        tf_matrix = omni.usd.get_world_transform_matrix(prim_view.prims[i])
        points = _transform_points(geom_mesh.GetPointsAttr().Get(), tf_matrix)

        # triangles is a list of indices, every 3 consecutive indices form a triangle
        triangles = deformableUtils.triangulate_mesh(geom_mesh)

        # tet_mesh contains 2 lists,
        # 1. conforming_tet_points: the nodal points of the tet mesh -> Gf
        # 2. conforming_tet_indices: the indices of the points of a tet (4 consecutive numbers -> the points for a tet)
        #conforming_tet_points, tet_indices = deformableUtils.compute_conforming_tetrahedral_mesh(points, triangles) #TODO custom tet mesh computation
        
        # convert Gf.Vec3f to list, which is compatible with c++
        #tet_mesh_points = [[gf_vec[0], gf_vec[1], gf_vec[2]] for gf_vec in conforming_tet_points] # use nested list, cause easy to use with pybind

        tet_gen = TetMeshGenerator()
        tet_mesh_points, tet_indices = tet_gen.compute_tet_mesh(points, triangles, config=tet_cfg)
        
        current_vertex_num = gipc_sim.sim.load_tet_mesh_from_data(tet_points=tet_mesh_points, tet_indices=tet_indices)

        print("Current vertex number: ", current_vertex_num)
        gipc_sim.body_vertex_idx_range.append(current_vertex_num) 
        objects_gipc.object_idx_offsets.append(current_vertex_num)      #! update idx offsets for this object type
       
        #! Don't update the points, otherwise we break the normal GIPC simulation setup
        # # remove xForm operations and update points manually
        # xform_utils.clear_xform_ops(prim_view.prims[i])
        # geom_mesh.GetPointsAttr().Set(points)
        # idx = np.array(tet_indices).reshape(-1,3)
        # geom_mesh.GetFaceVertexCountsAttr().Set([3] * len(idx)) # how many vertices each face has (3, cause we use triangles)
        # geom_mesh.GetFaceVertexIndicesAttr().Set(idx)
        # #geom_mesh.GetNormalsAttr().Set([]) # set to be empty, cause we use catmullClark and this gives us normals
        # #geom_mesh.GetSubdivisionSchemeAttr().Set("catmullClark") #none
        # self.geom_meshes.append(geom_mesh)
        
        _draw_tets(tet_mesh_points, tet_indices)
        _create_tet_data_attributes(objects_gipc=objects_gipc, tet_points=tet_mesh_points, tet_indices=tet_indices)
        
    print("obj idx offsets", objects_gipc.object_idx_offsets)
    print(f"Init {len(prim_view.prim_paths)} GIPC tet mesh(s) for gipc objects {objects_gipc.name} done.")
    gipc_sim.num_objects += len(prim_view.prim_paths)

def _transform_points(points, transformation_matrix):
    # need a Gf matrix, otherwise matrix multiplication is going to yield wrong result
    # transformation_matrix = Vt.Matrix4fArray.FromNumpy(transformation_matrix)[0]
    transformed_points = []
    # transform points by making them homogenous first and then applying the transformation matrix
    # after that, convert to normal coor.
    for p in points:
        transformed_vector = Gf.Vec4f(p[0], p[1], p[2], 1) * transformation_matrix # usd applies transform to row vectors: y^T = p^T*M^T, ref. https://nvidia.github.io/warp/modules/functions.html#warp.transform_point
        transformed_vector = Gf.Vec3f(transformed_vector[0], transformed_vector[1], transformed_vector[2])/transformed_vector[3]
        transformed_points.append(transformed_vector)
    return transformed_points

def _draw_tets(all_vertices, tet_indices):
        
    # first draw the tet mesh nodes
    # draw.draw_points(all_vertices, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices))
    
    # connect nodes according to tet_indices
    color = [(0,0,0,1)]
    for i in range(0, len(tet_indices), 4):
        tet_points_idx = tet_indices[i:i+4]
        tet_points = [all_vertices[i] for i in tet_points_idx]
        #draw.draw_points(tet_points, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices)) 
        draw.draw_lines([tet_points[0]]*3, tet_points[1:], color*3, [10]*3) # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tet_points[1]]*2, tet_points[2:], color*2, [10]*2)
        draw.draw_lines([tet_points[2]], [tet_points[3]], color, [10]) # draw line between the other 2 points

def _create_tet_data_attributes(objects_gipc: ObjectsGIPC, tet_points, tet_indices):
    """
    Creates an attribute for a prim that holds a boolean.
    See: https://graphics.pixar.com/usd/release/api/class_usd_prim.html.
    The attribute can then be found in the GUI under "Raw USD Properties" of the prim.
    Args:
        prim: A prim that should be holding the attribute.
        attribute_name: The name of the attribute to create.
    Returns:
        An attribute created at specific prim.
    """
    prim_view = objects_gipc.prim_view
    for prim in prim_view.prims:
        att_tet_points = prim.CreateAttribute("tet_points", Sdf.ValueTypeNames.Vector3fArray)
        att_tet_points.Set(tet_points)

        attr_tet_indices = prim.CreateAttribute("tet_indices", Sdf.ValueTypeNames.UIntArray)
        attr_tet_indices.Set(tet_indices)

        print("*"*40)
        print("Created tet data ")
        print("*"*40)

    return prim_view.prim_paths



############################################################
### Main
############################################################
print("")
print("")
print("#"*40)
print("#"*40)
print("Computing tet data")
gipc_cfg = CfgGIPC()
gipc_sim = SimGIPC(gipc_cfg)

draw.clear_points()
draw.clear_lines()

#--------------------- change paths for corresponding objects -----------------------------------------------
usd_context = omni.usd.get_context()
# returns a list of prim path strings
selection = usd_context.get_selection().get_selected_prim_paths()

if len(selection) > 1:
    print("Error, only select 1 mesh for attachment creation. ")
print("selected ", selection[0])
gipc_prim_path = selection[0]

       
#--------------- Compute Tet data ------------------
#!
cfg = CfgTetMesh()
cfg = cfg.replace(edge_length_r=1/25)
cfg = cfg.replace(epsilon=1e-3)
# cfg = cfg.replace(verbosity=1)

gipc_object = standalone_create_objects(gipc_sim, prim_paths_expr=gipc_prim_path, name="obj", tet_cfg=cfg) # use different object creation method than the one from gipc_sim

# update usd meshes
# gipc_sim.create_objects(prim_paths_expr=gipc_prim_path, name="beam", debug_view=True, tet_cfg=cfg)
# gipc_sim.init_scene()

# visualize all gipc points
positions = np.array(gipc_sim.sim.get_vertices())
draw.draw_points(positions, [(0,255,0,0.5)]*positions.shape[0], [25]*positions.shape[0])

print("#"*40)
print("#"*40)
