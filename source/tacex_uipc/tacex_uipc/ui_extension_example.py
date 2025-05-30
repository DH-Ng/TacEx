## Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##
import math
import random
from ctypes import alignment


import carb
import carb.events
import omni.ext
# import omni.usd

import isaacsim.core.utils.xforms as xform_utils
from isaacsim.core.prims import XFormPrim

from isaacsim.util.debug_draw import _debug_draw
draw = _debug_draw.acquire_debug_draw_interface()

from omni.physx.scripts import deformableUtils

from usdrt import Gf, Rt, Sdf, Usd, Vt
from pxr import UsdGeom, Gf
import pxr

import numpy as np
import wildmeshing as wm

from tacex_uipc.utils import MeshGenerator, TetMeshCfg

try:
    wp = None
    import warp as wp

    wp.init()

    @wp.kernel
    def deform(positions: wp.array(dtype=wp.vec3), t: float):
        tid = wp.tid()

        x = positions[tid]
        offset = -wp.sin(x[0])
        scale = wp.sin(5.0*t) * 0.05

        x = x + wp.vec3(0.0, offset * scale, 0.0)

        positions[tid] = x

except ImportError:
    pass


def get_selected_prim_path():
    """Return the path of the first selected prim"""
    context = omni.usd.get_context()
    selection = context.get_selection()
    paths = selection.get_selected_prim_paths()

    return None if not paths else paths[0]


def get_stage_id():
    """Return the stage Id of the current stage"""
    context = omni.usd.get_context()
    return context.get_stage_id()


def is_vtarray(obj):
    """Check if this is a VtArray type

    In Python, each data type gets its own
    VtArray class i.e. Vt.Float3Array etc.
    so this helper identifies any of them.
    """
    return hasattr(obj, "IsFabricData")


def condensed_vtarray_str(data):
    """Return a string representing VtArray data

    Include at most 6 values, and the total items
    in the array
    """
    size = len(data)
    if size > 6:
        datastr = "[{}, {}, {}, .. {}, {}, {}] (size: {})".format(
            data[0], data[1], data[2], data[-3], data[-2], data[-1], size
        )
    else:
        datastr = "["
        for i in range(size - 1):
            datastr += str(data[i]) + ", "
        datastr += str(data[-1]) + "]"

    return datastr


def get_fabric_data_for_prim(stage_id, path):
    """Get the Fabric data for a path as a string"""
    if path is None:
        return "Nothing selected"

    stage = Usd.Stage.Attach(stage_id)

    # If a prim does not already exist in Fabric,
    # it will be fetched from USD by simply creating the
    # Usd.Prim object. At this time, only the attributes that have
    # authored opinions will be fetch into Fabric.
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim:
        return f"Prim at path {path} is not in Fabric"

    # This diverges a bit from USD - only attributes
    # that exist in Fabric are returned by this API
    attrs = prim.GetAttributes()

    result = f"Fabric data for prim at path {path}\n\n\n"
    for attr in attrs:
        try:
            data = attr.Get()
            datastr = str(data)
            if data is None:
                datastr = "<no value>"
            elif is_vtarray(data):
                datastr = condensed_vtarray_str(data)

        except TypeError:
            # Some data types not yet supported in Python
            datastr = "<no Python conversion>"

        result += "{} ({}): {}\n".format(attr.GetName(), str(attr.GetTypeName().GetAsToken()), datastr)

    return result


### MESH STUFF
def _generate_tet_mesh(path, tet_cfg=None):        
    """
    Need to make sure that we load the geom mesh in USD and not just the Xform of the prim
    """
    if tet_cfg is None:
        tet_cfg = TetMeshCfg(
            edge_length_r=1/10
        )
    mesh_gen = MeshGenerator(tet_cfg)

    stage = omni.usd.get_context().get_stage()
    # prim = stage.GetPrimAtPath(Sdf.Path(path))
    geom_mesh = UsdGeom.Mesh.Get(stage, path)
    tet_points, tet_indices, surf_points, tet_surf_indices = mesh_gen.generate_tet_mesh_for_prim(geom_mesh)

    # Dont transform ->  we want to save the local points. Transformations happens during loading of the obj
    # tf_world = np.array(omni.usd.get_world_transform_matrix(geom_mesh))
    # tet_points = tf_world.T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
    # tet_points = (tet_points[:-1].T)

    draw.clear_lines()
    _draw_tets(tet_points, tet_indices)
    _draw_surface_trimesh(surf_points, tet_surf_indices)

    _create_tet_data_attributes(path, tet_points=tet_points, tet_indices=tet_indices, tet_surf_points=surf_points,tet_surf_indices=tet_surf_indices)
    return f"Amount of tet points {len(tet_points)},\nAmount of tetrahedra: {int(len(tet_indices)/4)},\nAmount of surface points: {int(len(tet_surf_indices)/3)}"

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

    draw.clear_lines()
    # first draw the tet mesh nodes
    # draw.draw_points(all_vertices, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices))
    
    # connect nodes according to tet_indices
    color = [(125,0,0,0.5)]
    for i in range(0, len(tet_indices), 4):
        tet_points_idx = tet_indices[i:i+4]
        tet_points = [all_vertices[i] for i in tet_points_idx]
        #draw.draw_points(tet_points, [(255,0,0,1)]*len(all_vertices), [10]*len(all_vertices)) 
        draw.draw_lines([tet_points[0]]*3, tet_points[1:], color*3, [10]*3) # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tet_points[1]]*2, tet_points[2:], color*2, [10]*2)
        draw.draw_lines([tet_points[2]], [tet_points[3]], color, [10]) # draw line between the other 2 points

def _draw_surface_trimesh(all_vertices, tet_surf_indices):
    color = [(0,0,125,0.5)]
    #draw surface mesh
    for i in range(0, len(tet_surf_indices), 3):
        tet_points_idx = tet_surf_indices[i:i+3]
        tet_points = [all_vertices[i] for i in tet_points_idx]
        draw.draw_points(tet_points, [(255,255,255,1)]*len(tet_points), [40]*len(tet_points)) 
        draw.draw_lines([tet_points[0]]*2, tet_points[1:], color*2, [10]*2) # draw from point 0 to every other point (3 times 0, cause line from 0 to the other 3 points)
        draw.draw_lines([tet_points[1]]*1, tet_points[2:], color*1, [10]*1)


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

    attr_tet_indices = prim.CreateAttribute("tet_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_indices.Set(tet_indices)

    attr_tet_surf_points = prim.CreateAttribute("tet_surf_points", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_surf_points.Set(tet_surf_points)

    attr_tet_surf_indices = prim.CreateAttribute("tet_surf_indices", pxr.Sdf.ValueTypeNames.UIntArray)
    attr_tet_surf_indices.Set(tet_surf_indices)

    print("*"*40)
    print("Created tet data: ")
    print(f"tet_points (num {tet_points.shape[0]})")
    print(f"tet_indices (num {len(tet_indices)})")
    print(f"tet_surf_points (num {len(tet_indices)})")
    print(f"tet_surf_indices (num {len(tet_indices)})")
    print("*"*40)

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class TacexIPCExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[tacex_uipc] startup")

        self._window = omni.ui.Window(
            "Generate Tet Meshes for the IPC simulation:", width=300, height=300, dockPreference=omni.ui.DockPreference.RIGHT_BOTTOM
        )
        self._t = 0
        self._dt = 0.01
        self.sub = None
        self.playing = False

        with self._window.frame:
            with omni.ui.VStack():
                frame = omni.ui.ScrollingFrame()
                with frame:
                    label = omni.ui.Label("Select a prim and push a button", alignment=omni.ui.Alignment.LEFT_TOP)

                def compute_tet_mesh():
                    label.text = _generate_tet_mesh(get_selected_prim_path())

                omni.ui.Button("Compute Tet Mesh", clicked_fn=compute_tet_mesh, height=0)

    # def init_on_update(self):
    #     @carb.profiler.profile(zone_name="omni.example.python.usdrt.on_update")
    #     def on_update(e: carb.events.IEvent):
    #         if not self.playing:
    #             return
    #         try:
    #             deform_mesh_with_warp(get_stage_id(), get_selected_prim_path(), self._t)
    #             self._t += self._dt
    #         except Exception as e:
    #             carb.log_error(e)
    #         return

    #     update_stream = omni.kit.app.get_app().get_update_event_stream()
    #     self.sub = update_stream.create_subscription_to_pop(on_update, name="omni.example.python.usdrt.on_update")
    #     return
    
    def on_shutdown(self):
        print("[tacex_uipc] shutdown")