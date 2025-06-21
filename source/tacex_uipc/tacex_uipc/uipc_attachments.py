
import omni
from omni.physx import get_physx_cooking_interface, get_physx_interface, get_physx_scene_query_interface
from isaacsim.core.prims import XFormPrim

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt

import numpy as np
import torch
import re
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

from isaaclab.utils.math import transform_points
import isaaclab.utils.math as math_utils

try:
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except:
    draw = None

from uipc.constitution import SoftTransformConstraint

if TYPE_CHECKING:
    from tacex_uipc.uipc_attachments import UipcObject

class UipcIsaacAttachments():
    #todo fix init properly
    def __init__(self, isaac_rigid_object=None, uipc_object=None) -> None: 
        self.isaac_rigid_object = isaac_rigid_object
        self.uipc_object: UipcObject = uipc_object
        
        self.uipc_object_vertex_indices = []
        self.attachment_points_init_positions = []
        self.attachment_offsets = np.zeros(0) #[] # should be the same for every object of this class, are in local frame
        self.num_attachment_points_per_obj = 0

        self.body_id = None # used to query the position of the rigid body

        #self.attachments_offsets_idx_range = [0]
        self.new_pos = np.zeros(0)

    def get_new_positions(self):
        if self.body_id is None:
            raise RuntimeError(
                f"Id for attachment is None. Cannot compute new positions"
            )
        pose = self.isaac_rigid_object.data.body_state_w[:, self.body_id, 0:7].clone()
                
        new_pos = torch.tensor(self.attachment_offsets.reshape((self.objects_gipc.num_objects, self.num_attachment_points_per_obj, 3)), device="cuda:0").float()
        new_pos = transform_points(new_pos, pos=pose[:, 0:3], quat=pose[:, 3:]) 
        new_pos = new_pos.cpu().numpy()
        self.new_pos = new_pos.flatten().reshape(-1,3)
        
        #     # extract velocity
        #     lin_vel = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 7:10]
        #     lin_vel = lin_vel.cpu().numpy()
        #     lin_vel = np.tile(lin_vel, len(attachment_points)).reshape(len(attachment_points),3)
        idx = self.gipc_vertex_idx
        return idx, self.new_pos
    
    def draw_debug_view(self):
        
        draw.draw_points(self.new_pos, [(255,0,0,0.5)]*self.new_pos.shape[0], [30]*self.new_pos.shape[0]) # the new positions
        pose = self.isaac_rigid_object.data.body_state_w[:, self.body_id, 0:7].clone()
        obj_center = pose[:, 0:3]

        for i in range(self.objects_gipc.num_objects):
            for j in range(i, self.objects_gipc.num_objects*self.num_attachment_points_per_obj):
                draw.draw_lines([obj_center[i].cpu().numpy()], [self.new_pos[j]], [(255,255,0,0.5)], [10])

    # def create(self):
    #     soft_position_constraint = 
    
    def create_attachment(self, mesh_path=None, all_gipc_vertices=None):
        """_summary_

        Args:
            all_gipc_vertices (_type_): _description_
            scene_env_origins (): origions of the cloned env in the scene
            sphere_radius (float, optional): _description_. Defaults to 0.0005.
            max_dist (float, optional): _description_. Defaults to 0.0005.

        Returns:
            idx (list): List of vertex indices in GIPC for the attachment points
            attachment_points_positions (list[np.array]): List of positions (x,y,z) for the attachment points. 
        """
        #TODO remove
        # if mesh_path is not None and all_gipc_vertices is not None:
        #     print("Computing data for attachment with ", mesh_path)
        #     self._compute_attachment_data(mesh_path, all_gipc_vertices)
            
        # take one prim path to find out if the body is part of prim_expr
        attached_to_prim_path = self.objects_gipc.prim_view.prims[0].GetAttribute("attached_to").Get()
        body_names = self.isaac_rigid_object.body_names
        
        for i, name in enumerate(body_names):
            if name in attached_to_prim_path:
                self.body_id = i
        
        # the offsets in the attachment_offsets attribute are defined in local space, therefore all objects of this class share the same offsets
        self.attachment_offsets = np.array(self.objects_gipc.prim_view.prims[0].GetAttribute("attachment_offsets").Get())
        
        attachment_points_init = []
        for j in range(self.objects_gipc.num_objects):
            prim = self.objects_gipc.prim_view.prims[j]

            min_vertex_idx = self.objects_gipc.object_idx_offsets[j]
            max_vertex_idx = self.objects_gipc.object_idx_offsets[j+1] - 1

            obj_idx = np.array(prim.GetAttribute("attachment_vertex_idx").Get()) + min_vertex_idx # update vertex idx in correspondence to the gipc simulation class
            self.gipc_vertex_idx.append(obj_idx.tolist())
            attachment_points_init.append(np.array(prim.GetAttribute("initial_attachment_positions").Get()))
            
            #print("attachemnt idx ", self.gipc_vertex_idx)
            # print("attachemnt init pos ", self.attachment_points_init_positions) #TODO update initial positions for multi env, i.e. use scene offset correctly (I guess)
            # print("attachment offsets ", self.attachment_offsets)
            
        self.num_attachment_points_per_obj = self.attachment_offsets.shape[0]
        print(f"Number of attachment points per obj '{self.objects_gipc.name}': {self.num_attachment_points_per_obj}")

        # transform attachment offsets array into shape (N,P,3) for transform_points in get_new_positions,
        # N=number of env's, i.e. number of attachment points per obj
        self.attachment_offsets.reshape(self.num_attachment_points_per_obj*self.objects_gipc.num_objects, 3)
        self.attachment_points_init_positions = np.array(attachment_points_init).flatten().reshape(-1,3)

        # print("offsets, ", self.attachment_offsets)
        # print("init pos, ", self.attachment_points_init_positions)
        
        # at the end, make sure that idx list is a flat list
        self.gipc_vertex_idx = np.array(self.gipc_vertex_idx).flatten().tolist() # can use np flatten, cause every list here has the same number of elements (are basically the same objects with the same amount of attachment points)
        return self.gipc_vertex_idx, self.attachment_points_init_positions #TODO fix this messed up command here lol, we want a flat list
     
    @staticmethod
    def compute_attachment_data(isaac_mesh_path, tet_points, tet_indices, sphere_radius=5e-4, max_dist=1e-5): # really small distances to prevent intersection with unwanted geometries
        """
        Computes the attachment data and sets it as attribute values of the corresponding usd meshs.

        Args:
            all_gipc_vertices (_type_): _description_
            scene_env_origins (): origions of the cloned env in the scene
            sphere_radius (float, optional): _description_. Defaults to 0.0005.
            max_dist (float, optional): _description_. Defaults to 0.0005.

        Returns:
            idx (list): List of vertex indices in GIPC for the attachment points
            attachment_points_positions (list[np.array]): List of positions (x,y,z) for the attachment points. 
        """
        print(f"Creating Uipc x Isaac attachments for {isaac_mesh_path}")
        # force Physx to cook everything in the scene so it get cached
        get_physx_interface().force_load_physics_from_usd()
        # stage = omni.usd.get_context().get_stage()
        # init_prim = stage.GetPrimAtPath(isaac_mesh_path)
        #print("init prim ",init_prim)

        # check if base asset path is valid
        # note: currently the spawner does not work if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Robot_[1,2]" since the spawner will not
        #   know which prim to spawn. This is a limitation of the spawner and not the asset.
        asset_path = isaac_mesh_path.split("/")[-1]
        asset_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", asset_path) is None
        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(isaac_mesh_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {isaac_mesh_path}.")
        init_prim = matching_prims[0]

        pose = omni.usd.get_world_transform_matrix(init_prim) #omni.usd.utils.get_world_transform_matrix(init_prim)
        obj_position = pose.ExtractTranslation()
        obj_position = np.array([obj_position])

        q = pose.ExtractRotation().GetQuaternion()
        obj_orientation = [q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
        obj_orientation = torch.tensor(np.array([obj_orientation]), device="cuda:0").float()

        idx = []
        attachment_points_positions = []
        attachment_offsets = []

        # get position of current object for offset computation
        obj_pos = obj_position[0,:]
        print("obj_pos ", obj_pos)
        print("orient ", obj_orientation)
        
        # uipc_mesh = self.uipc_object.uipc_meshes[0]
        # vertex_positions = torch.tensor(uipc_mesh.positions().view().copy().reshape(-1,3), device=self.device)
        # print("vertex_positions ", vertex_positions)

        # topology = uipc_mesh.topo().view().reshape(-1).tolist()
        # print("topology ", topology)

        vertex_positions = tet_points
        indices = tet_indices
        
        for i, v in enumerate(vertex_positions):
            # print("raycast ", i)
            ray_dir = [0,0,1] # unit direction of the sphere ray cast -> doesnt really matter here, cause we only use a super short ray. 
            hitInfo = get_physx_scene_query_interface().sweep_sphere_closest(radius=sphere_radius, origin=v, dir=ray_dir, distance=max_dist, bothSides=True)
            if hitInfo["hit"]:
                print("hiiiit, ", hitInfo["collision"])
                if str(init_prim.GetPath()) in hitInfo["collision"] : # prevent attaching to unrelated geometry
                    attachment_points_positions.append(v)
                    # idx.append(i+min_vertex_idx) unlike the gipc simulation, we use the object specific idx here
                    idx.append(i)
                    #TODO do this at the end and compute in a vectorized fashion?
                    # compute offsets from object position to attachment points 
                    offset = v - obj_pos
                    offset = torch.tensor(offset, device="cuda:0").float()
                    offset = math_utils.quat_rotate_inverse(obj_orientation[0].reshape((1,4)), offset.reshape((1,3)))[0]
                    offset = offset.cpu().numpy()
                    attachment_offsets.append(offset)

        #get_physx_interface().release_physics_objects() #! using this in standalone script, breaks sim
        
        # print(f"Number of attachment points per obj '{self.objects_gipc.name}': {self.num_attachment_points_per_obj}")

        attachment_points_positions = np.array(attachment_points_positions).reshape(-1,3)

        # offset to later compute the `should-be` positions of the attachment point
        attachment_offsets = np.array(attachment_offsets).reshape(-1, 3) 
        assert len(idx) == attachment_offsets.shape[0]

        print("offsets, ", attachment_offsets)
        # print("attachment local idx, ", idx)
        # print("Init pos, ", attachment_points_positions)

        # self._create_attachment_data_attributes(isaac_mesh_path, self.objects_gipc, idx, attachment_points_positions, attachment_offsets)

        # draw attachment data
        draw.draw_points(attachment_points_positions, [(255,0,0,0.5)]*attachment_points_positions.shape[0], [30]*attachment_points_positions.shape[0]) # the new positions
        obj_center = obj_position[0]

        for j in range(0, attachment_points_positions.shape[0]):
            draw.draw_lines([obj_center], [attachment_points_positions[j,:]], [(255,255,0,0.5)], [10])

        return attachment_offsets, idx    

    # def _create_attachment_data_attributes(self, mesh_name, objects_gipc: ObjectsGIPC, gipc_vertex_idx, initial_attachment_points_positions, attachment_offsets):
    #     """
    #     Creates an attribute for a prim that holds a boolean.
    #     See: https://graphics.pixar.com/usd/release/api/class_usd_prim.html.
    #     The attribute can then be found in the GUI under "Raw USD Properties" of the prim.
    #     Args:
    #         prim: A prim that should be holding the attribute.
    #         attribute_name: The name of the attribute to create.
    #     Returns:
    #         An attribute created at specific prim.
    #     """

    #     prim_view = objects_gipc.prim_view
    #     #TODO use scene_env offsets to adjust the data here!
    #     for prim in prim_view.prims:
    #         attr_attached_to = prim.CreateAttribute("attached_to", Sdf.ValueTypeNames.String)
    #         attr_idx = prim.CreateAttribute("attachment_vertex_idx", Sdf.ValueTypeNames.UIntArray)
    #         attr_initial = prim.CreateAttribute("initial_attachment_positions", Sdf.ValueTypeNames.Vector3fArray)
    #         attr_offsets = prim.CreateAttribute("attachment_offsets", Sdf.ValueTypeNames.Vector3fArray)

    #         prim.GetAttribute("attached_to").Set(mesh_name)
    #         prim.GetAttribute("attachment_vertex_idx").Set(gipc_vertex_idx)
    #         prim.GetAttribute("initial_attachment_positions").Set(initial_attachment_points_positions)
    #         prim.GetAttribute("attachment_offsets").Set(attachment_offsets)

    #         print("*"*40)
    #         print("Created attachment with data: ")
    #         print("attached to ", attr_attached_to.Get())
    #         print("idx ", attr_idx.Get())
    #         # print("inital pos ", attr_initial.Get())
    #         # print("offsets ", attr_offsets.Get())
    #         print("*"*40)

    #     return prim_view.prim_paths