"""Base class for optical simulation approaches.

This class defines an interface for simulating the optical response of GelSight tactile sensors.
Each simulation method class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from ..gelsight_sensor import GelSightSensor
    from ..simulation_approaches.gelsight_simulator_cfg import GelSightSimulatorCfg


class GelSightSimulator(ABC):
    """Base class for implementing an optical simulation approach.

    Output buffers: the sensor owns the output tensors. After :meth:`_initialize_impl`, it asks the simulator for the
    per-env shape of each output it produces (:meth:`get_output_shape`), allocates the buffer on its own device and
    hands it over with :meth:`set_output_buffer`. The simulation methods write into these buffers in place (computing
    on the simulator's device and moving the result to the buffer's device), so nothing is copied by the sensor.
    :meth:`reset` writes the no-contact state into the reset envs' rows.
    """

    def __init__(self, sensor: GelSightSensor, cfg: GelSightSimulatorCfg):
        self.cfg = cfg
        self.sensor = sensor

        self._output_buffers: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        """Output buffers owned by the sensor, by data type (``tactile_rgb``, ``marker_motion``, ``force_field``)."""

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        # self._num_envs = self.sensor._num_envs

        # self._initialize_impl()

    @abstractmethod
    def _initialize_impl(self):
        raise NotImplementedError

    """
    Output buffers.
    """

    def get_output_shape(self, data_type: str) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        """Per-env shape (without the num_envs dimension) of an output this simulator produces.

        Called by the sensor after :meth:`_initialize_impl`, to allocate the buffer passed to :meth:`set_output_buffer`.

        Args:
            data_type: ``tactile_rgb`` -> (height, width, 3); ``marker_motion`` -> (2, num_markers, 2);
                ``force_field`` -> dict of per-env shapes, one tensor per key.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not produce the output [{data_type}].")

    def set_output_buffer(self, data_type: str, buffer: torch.Tensor | dict[str, torch.Tensor]):
        """Hands over the sensor-owned output buffer for ``data_type`` (first dim = num_envs).

        The simulator writes into it in place and must not replace it.
        """
        self._output_buffers[data_type] = buffer

    """
    Simulation. Each method writes into its output buffer in place (all envs) and returns None.
    """

    # @abstractmethod
    def optical_simulation(self) -> None:
        """Simulates the optical output of a Tactile sensor into the ``tactile_rgb`` buffer
        (num_envs, height, width, 3)."""
        raise NotImplementedError

    # @abstractmethod
    def marker_motion_simulation(self) -> None:
        """Simulates the marker motion of a Tactile sensor into the ``marker_motion`` buffer
        (num_envs, 2, num_markers, 2): dim 1 = initial/current, last dim = marker position."""
        raise NotImplementedError

    # @abstractmethod
    def force_field_simulation(self) -> None:
        """Simulates per-point contact forces on the gelpad surface into the ``force_field`` buffer
        (dict of tensors, first dim = num_envs, keys and shapes from :meth:`get_output_shape`)."""
        raise NotImplementedError

    # @abstractmethod make it optional, in case another method is used for computing indentation depth
    def compute_indentation_depth(self):
        """Computes how deep the indenter is pressed into the gelpad"""
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the internal state of the simulator for the given environments and writes the no-contact state
        into the ``env_ids`` rows of its output buffers (e.g. the tactile image of an undeformed gel, the initial
        marker positions, zero forces).

        Args:
            env_ids: Environment indices to reset. If None, all environments are reset.

        Note:
            Only the state and output rows of ``env_ids`` may be modified. The sensor does not re-simulate on reset.
        """
        raise NotImplementedError

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    """
    Internal simulation callbacks.
    """
