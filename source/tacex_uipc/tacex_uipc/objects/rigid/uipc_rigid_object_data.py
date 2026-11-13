from __future__ import annotations

import torch
import weakref
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcSim

    from .uipc_rigid_object import UipcRigidObject


class UipcRigidObjectData:
    """Data container for a rigid uipc object.

    This class contains the data for a rigid object in the simulation. The data includes the state of
    the root rigid body and the state of all the bodies in the object. The data is stored in the simulation
    world frame unless otherwise specified.

    For a rigid body, there are two frames of reference that are used:

    - Actor frame: The frame of reference of the rigid body prim. This typically corresponds to the Xform prim
      with the rigid body schema.
    - Center of mass frame: The frame of reference of the center of mass of the rigid body.

    Depending on the settings of the simulation, the actor frame and the center of mass frame may be the same.
    This needs to be taken into account when interpreting the data.

    The data is lazily updated, meaning that the data is only updated when it is accessed. This is useful
    when the data is expensive to compute or retrieve. The data is updated when the timestamp of the buffer
    is older than the current simulation timestamp. The timestamp is updated whenever the data is updated.
    """

    def __init__(self, uipc_sim: UipcSim, uipc_rigid_object: UipcRigidObject, device: str):
        """Initializes the rigid object data.

        Args:
            root_physx_view: The root rigid body view.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device

        self._uipc_sim: UipcSim = weakref.proxy(uipc_sim)

        # Set the root rigid body view
        # note: this is stored as a weak reference to avoid circular references between the asset class
        #  and the data container. This is important to avoid memory leaks.
        self._uipc_rigid_object: UipcRigidObject = weakref.proxy(uipc_rigid_object)

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Initialize the lazy buffers.
        self._root_state_w = TimestampedBuffer()

    def update(self, dt: float):
        """Updates the data for the rigid object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Names.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Defaults.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat]`` in world frame. Shape is (num_instances, 13).
    #TODO add , lin_vel, ang_vel
    The position and quaternion are of the rigid body's actor frame. Meanwhile, the linear and angular velocities are
    of the center of mass frame.
    """

    ##
    # Properties.
    ##

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal positions.
        """
        # return self.nodal_pos_w.mean(dim=1)
        geom = self._uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(
            self._uipc_rigid_object.obj_id
        )  # todo instead of finding obj, lets just save ref to geo_slot in uipc_object

        # NOTE: transformation is w.r.t. to initial pose -> so, how do we get the initial pose tf matrix?
        trans = geo_slot.geometry().transforms().view()
        trans = torch.tensor(trans, device=self.device).reshape(1, 4, 4)
        root_pos_w, root_orient_w = math_utils.unmake_pose(trans)
        # print("Root pos w of abd body: ", root_pos_w)
        return root_pos_w.reshape(1, 3)  # todo need to adjust once we go multi env

    # @property
    # def root_vel_w(self) -> torch.Tensor:
    #     """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
    #     Shape is (num_instances, 3).

    #     This quantity is computed as the mean of the nodal velocities.
    #     """
    #     return self.nodal_vel_w.mean(dim=1)
