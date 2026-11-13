from __future__ import annotations

import re

import inspect
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import omni

import isaaclab.sim as sim_utils

from uipc import Animation

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from ..uipc_object import UipcObject


@configclass
class UipcConstraintCfg:
    """Constraint Config class for simple UIPC constraint"""

    constraint_strength_ratio: float = 100.0
    """
    E.g., 100.0 means the stiffness of the constraint is 100 times of the mass of the uipc object.
    """

    # constraint_type: SoftPositionConstraint | SoftTransformConstraint = None
    constraint_type: type = None


class UipcConstraint:
    cfg: UipcConstraintCfg

    # todo code init properly
    def __init__(self, cfg: UipcConstraintCfg, uipc_object: UipcObject) -> None:
        """Simple UIPC constraint for UIPC objects.

        Args:
            cfg (UipcConstraintCfg): _description_
            uipc_object (UipcObject): _description_
        """
        # check that the config is valid
        cfg.validate()
        self.cfg: UipcConstraintCfg = cfg.copy()

        self.uipc_object: UipcObject = uipc_object

        self._num_instances = 1

        # todo handle multiple meshes properly (currently just single mesh)

        # create uipc constraint
        constraint = self.cfg.constraint_type()
        # `apply` has to happen **before** the uipc_scene_object is created!
        constraint.apply_to(self.uipc_object.uipc_meshes[0], self.cfg.constraint_strength_ratio)

        # the function used to animate the vertices
        self.animate_func: Callable[Animation.UpdateInfo] = None
        # self._create_animation()

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
        return self._num_instances

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
                self._debug_vis_handle = app_interface.get_pre_update_event_stream().create_subscription_to_pop(  # get_post_update_event_stream get_pre_update_event_stream
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    """
    Internal helper.
    """

    # def _initialize_impl(self):
    #     sim: sim_utils.SimulationContext = sim_utils.SimulationContext.instance()
    #     # sim.add_physics_callback(
    #     #     f"{self.uipc_object.cfg.prim_path}_X_{self.isaaclab_rigid_object.cfg.prim_path}_attachment_update",
    #     #     self._compute_aim_positions,
    #     # )

    def _create_animation(self):
        animator = self.uipc_object.uipc_sim.scene.animator()
        animator.insert(self.uipc_object.uipc_scene_objects[0], self.animate_func)

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
        if debug_vis and getattr(self, "_draw", None) is None:
            from tacex_uipc.utils.debug_draw import DebugDraw

            name = re.sub(r"[^A-Za-z0-9_]", "_", self.uipc_object.cfg.prim_path.strip("/"))
            self._draw = DebugDraw(f"/Visuals/UipcConstraints/{name}", point_radius=3e-4, color=(1.0, 0.0, 0.0))
        if getattr(self, "_draw", None) is not None:
            self._draw.set_visibility(debug_vis)

    # def _debug_vis_callback(self, event):
    #     if self.aim_positions.shape[0] == 0:
    #         return

    #     # self._compute_aim_positions()

    #     # # draw attachment data
    #     self._draw.clear_points()
    #     self._draw.clear_lines()

    #     # drawing with the debug method leads to render delay
    #     self._draw.draw_points(
    #         self.aim_positions, [(255, 0, 0, 0.5)] * self.aim_positions.shape[0], [60] * self.aim_positions.shape[0]
    #     )  # the new positions
    #     # pose = self.isaaclab_rigid_object.data.body_state_w[:, self.rigid_body_id, 0:7].clone()
    #     pose = self.obj_pose.clone()

    #     for i in range(self._num_instances):
    #         obj_center = pose[i, 0, 0:3].cpu().numpy()
    #         # self._draw.clear_points()
    #         # print("Obj_center_debug ", obj_center)
    #         # draw.draw_points([obj_center], [(255,255,0,0.5)]*obj_center.shape[0], [50]*obj_center.shape[0]) # the new positions
    #         # print("")
    #         for j in range(i, self._num_instances * self.num_attachment_points_per_obj):
    #             self._draw.draw_lines([obj_center], [self.aim_positions[j]], [(255, 255, 0, 0.5)], [10])
