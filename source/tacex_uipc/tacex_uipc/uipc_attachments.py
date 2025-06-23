from __future__ import annotations

import weakref
import inspect

import omni
from omni.physx import get_physx_cooking_interface, get_physx_interface, get_physx_scene_query_interface
from isaacsim.core.prims import XFormPrim

from pxr import UsdGeom, Usd, Sdf, PhysxSchema, UsdPhysics, Gf, UsdShade, Vt

import numpy as np
import torch
import re
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
try:
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except:
    draw = None

from isaaclab.utils import configclass
from isaaclab.utils.math import transform_points
import isaaclab.utils.math as math_utils

from isaaclab.assets import AssetBase, AssetBaseCfg, RigidObject

from uipc.constitution import SoftPositionConstraint
from uipc import view
from uipc import builtin
from uipc import Animation, Vector3
from uipc.geometry import GeometrySlot, SimplicialComplex

if TYPE_CHECKING:
    from tacex_uipc.uipc_attachments import UipcObject

@configclass
class UipcIsaacAttachmentsCfg:
    constraint_strength_ratio: float = 100.0
    """
    E.g., 100.0 means the stiffness of the constraint is 100 times of the mass of the uipc object.
    """

    debug_vis: bool = False
    """Draw attachment offsets and aim_position via IsaacSim's _debug_draw api.
    
    """
    
    body_name: str = None
    """Name of the body in the rigid object that should be used for the attachment.

    Useful, e.g. when attaching to a part of an articulation.
    """

    attachment_points_radius: float = 5e-4
    """Distance between tet points and isaac collider, which is used to determine the attachment points.
    
    If the collision mesh of the isaaclab_rigid_object is in the radius of a point, then the
    point is considered "attached" to the isaaclab_rigid_object.
    """
class UipcIsaacAttachments():
    cfg: UipcIsaacAttachmentsCfg

    #todo code init properly
    def __init__(self, cfg: UipcIsaacAttachmentsCfg, uipc_object: UipcObject, isaaclab_rigid_object: RigidObject) -> None: 
        # check that the config is valid
        cfg.validate()
        self.cfg = cfg.copy()

        self.uipc_object: UipcObject = uipc_object
        self.isaaclab_rigid_object: RigidObject = isaaclab_rigid_object
        
        self.rigid_body_id = None # used to query the position of the rigid body

        self.uipc_object_vertex_indices = []
        self.attachment_points_init_positions = []

        #self.attachments_offsets_idx_range = [0]
        self.aim_positions = np.zeros(0)

        # create the attachment
        attachment_points_radius = self.cfg.attachment_points_radius

        isaac_rigid_prim_path = self.isaaclab_rigid_object.cfg.prim_path
        if self.cfg.body_name is not None:
            isaac_rigid_prim_path += "/.*" + self.cfg.body_name
        print("isaac_rigid_prim ", isaac_rigid_prim_path)

        mesh = self.uipc_object.uipc_meshes[0]
        tet_points_world = mesh.positions().view()[:,:,0]
        tet_indices = mesh.tetrahedra().topo().view()[:,:,0]

        attachment_offsets, idx, rigid_prims = self.compute_attachment_data(isaac_rigid_prim_path, tet_points_world, tet_indices, attachment_points_radius)
        
        # set attachment data
        self.attachment_offsets = attachment_offsets
        self.attachment_points_idx = idx
        self.num_attachment_points_per_obj = len(idx)

        # set uipc constraint
        soft_position_constraint = SoftPositionConstraint()
        #todo handle multiple meshes properly (currently just single mesh)
        soft_position_constraint.apply_to(self.uipc_object.uipc_meshes[0], self.cfg.constraint_strength_ratio) 

        # flag for whether the asset is initialized
        self._is_initialized = False

        # note: Use weakref on all callbacks to ensure that this object can be deleted when its destructor is called.
        # add callbacks for stage play/stop
        # The order is set to 10 which is arbitrary but should be lower priority than the default order of 0
        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            lambda event, obj=weakref.proxy(self): obj._initialize_callback(event),
            order=10,
        )
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        # clear physics events handles
        if self._initialize_handle:
            self._initialize_handle.unsubscribe()
            self._initialize_handle = None
        if self._invalidate_initialize_handle:
            self._invalidate_initialize_handle.unsubscribe()
            self._invalidate_initialize_handle = None
        # clear debug visualization
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None
    """
    Properties
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the asset is initialized.

        Returns True if the asset is initialized, False otherwise.
        """
        return self._is_initialized

    @property
    def num_instances(self) -> int:
        """Number of instances of the asset.

        This is equal to the number of asset instances per environment multiplied by the number of environments.
        """
        return NotImplementedError

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self._device

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the asset has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the asset data.

        Args:
            debug_vis: Whether to visualize the asset data.

        Returns:
            Whether the debug visualization was successfully set. False if the asset
            does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    def get_new_positions(self):
        if self.body_id is None:
            raise RuntimeError(
                f"Id for attachment is None. Cannot compute new positions"
            )
        pose = self.isaac_rigid_prims.data.body_state_w[:, self.body_id, 0:7].clone()
                
        new_pos = torch.tensor(self.attachment_offsets.reshape((self.objects_gipc.num_objects, self.num_attachment_points_per_obj, 3)), device="cuda:0").float()
        new_pos = transform_points(new_pos, pos=pose[:, 0:3], quat=pose[:, 3:]) 
        new_pos = new_pos.cpu().numpy()
        self.aim_positions = new_pos.flatten().reshape(-1,3)
        
        #     # extract velocity
        #     lin_vel = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 7:10]
        #     lin_vel = lin_vel.cpu().numpy()
        #     lin_vel = np.tile(lin_vel, len(attachment_points)).reshape(len(attachment_points),3)
        idx = self.gipc_vertex_idx
        return idx, self.aim_positions
    
    def _create_animation(self):
        animator = self.uipc_object._uipc_sim.scene.animator()
        def animate_tet(info: Animation.UpdateInfo): # animation function
            print("animated points ", self.aim_positions)
            geo_slots:list[GeometrySlot] = info.geo_slots()
            geo: SimplicialComplex = geo_slots[0].geometry()
            rest_geo_slots:list[GeometrySlot] = info.rest_geo_slots()
            rest_geo:SimplicialComplex = rest_geo_slots[0].geometry()

            is_constrained = geo.vertices().find(builtin.is_constrained)
            is_constrained_view = view(is_constrained)
            aim_position = geo.vertices().find(builtin.aim_position)
            aim_position_view = view(aim_position)
            rest_position_view = rest_geo.positions().view()

            is_constrained_view[0] = 1

            t = info.dt() * info.frame()
            theta = np.pi * t
            z = -np.sin(theta)

            aim_position_view[0] = rest_position_view[0] + Vector3.UnitZ() * z

        animator.insert(self.uipc_object.uipc_scene_objects[0], animate_tet)

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
        body_names = self.isaac_rigid_prims.body_names
        
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
            raise RuntimeError(f"Could not find prim with path {isaac_mesh_path}. The body_name in the cfg might not exist.")
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
        # print("obj_pos ", obj_pos)
        # print("orient ", obj_orientation)
        
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
                # print("hiiiit, ", hitInfo["collision"])
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

        # print(f"Number of attachment points per obj '{self.objects_gipc.name}': {self.num_attachment_points_per_obj}")
        attachment_points_positions = np.array(attachment_points_positions).reshape(-1,3)

        # offset to later compute the `should-be` positions of the attachment point
        attachment_offsets = np.array(attachment_offsets).reshape(-1, 3) 
        assert len(idx) == attachment_offsets.shape[0]

        # print("offsets, ", attachment_offsets)
        # print("attachment local idx, ", idx)
        # print("Init pos, ", attachment_points_positions)

        # self._create_attachment_data_attributes(isaac_mesh_path, self.objects_gipc, idx, attachment_points_positions, attachment_offsets)

        # draw attachment data
        draw.draw_points(attachment_points_positions, [(255,0,0,0.5)]*attachment_points_positions.shape[0], [30]*attachment_points_positions.shape[0]) # the new positions
        obj_center = obj_position[0]

        for j in range(0, attachment_points_positions.shape[0]):
            draw.draw_lines([obj_center], [attachment_points_positions[j,:]], [(255,255,0,0.5)], [10])

        return attachment_offsets, idx, matching_prims    

    def _compute_aim_positions(self):
        pose = self.isaac_lab_rigid_bodies.data.body_state_w[:, self.body_id, 0:7].clone()
                
        # aim_pos = torch.tensor(self.attachment_offsets.reshape((self.objects_gipc.num_objects, self.num_attachment_points_per_obj, 3)), device="cuda:0").float()
        aim_pos = torch.tensor(self.attachment_offsets.reshape((self.objects_gipc.num_objects, self.num_attachment_points_per_obj, 3)), device="cuda:0").float()
        aim_pos = transform_points(aim_pos, pos=pose[:, 0:3], quat=pose[:, 3:]) 
        aim_pos = aim_pos.cpu().numpy()
        self.aim_positions = aim_pos.flatten().reshape(-1,3)
        
        #     # extract velocity
        #     lin_vel = scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[1], 7:10]
        #     lin_vel = lin_vel.cpu().numpy()
        #     lin_vel = np.tile(lin_vel, len(attachment_points)).reshape(len(attachment_points),3)
        idx = self.gipc_vertex_idx
        return self.aim_positions     
       
    """
    Internal helper.
    """

    def _initialize_impl(self):
        if self.cfg.body_name is not None:
            self.rigid_body_id , found_body_name = self.isaaclab_rigid_object.find_bodies(self.cfg.body_name)

        self._create_animation()


    """
    Internal simulation callbacks. 
    
    Same as AssetBase class from asset_base.py
    """

    def _initialize_callback(self, event):
        """Initializes the scene elements.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.
        """
        if not self._is_initialized:
            # obtain simulation related information
            sim = sim_utils.SimulationContext.instance()
            if sim is None:
                raise RuntimeError("SimulationContext is not initialized! Please initialize SimulationContext first.")
            self._backend = sim.backend
            self._device = sim.device
            # initialize attachments
            self._initialize_impl()
            # set flag
            self._is_initialized = True

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        self._is_initialized = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self._draw = _debug_draw.acquire_debug_draw_interface()
            except:
                self._draw = None
                print("No debug_vis for attachment. Reason: Cannot import _debug_draw")
                return

            # # draw attachment data
            # draw.draw_points(attachment_points_positions, [(255,0,0,0.5)]*attachment_points_positions.shape[0], [30]*attachment_points_positions.shape[0]) # the new positions
            # obj_center = obj_position[0]

            # for j in range(0, attachment_points_positions.shape[0]):
            #     draw.draw_lines([obj_center], [attachment_points_positions[j,:]], [(255,255,0,0.5)], [10])

    def _debug_vis_callback(self, event):
        # # draw attachment data
        # draw.draw_points(attachment_points_positions, [(255,0,0,0.5)]*attachment_points_positions.shape[0], [30]*attachment_points_positions.shape[0]) # the new positions
        # obj_center = obj_position[0]

        # for j in range(0, attachment_points_positions.shape[0]):
        #     draw.draw_lines([obj_center], [attachment_points_positions[j,:]], [(255,255,0,0.5)], [10])
        print("debug vis")        
        # self._draw.draw_points(self.aim_positions, [(255,0,0,0.5)]*self.aim_positions.shape[0], [30]*self.aim_positions.shape[0]) # the new positions
        # pose = self.isaac_rigid_prims.data.body_state_w[:, self.body_id, 0:7].clone()
        # obj_center = pose[:, 0:3]

        # for i in range(self.objects_gipc.num_objects):
        #     for j in range(i, self.objects_gipc.num_objects*self.num_attachment_points_per_obj):
        #         self._draw.draw_lines([obj_center[i].cpu().numpy()], [self.aim_positions[j]], [(255,255,0,0.5)], [10])
