from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import warp as wp

from isaaclab.utils import configclass

wp.init()


from ..uipc_object import UipcObjectCfg
from .uipc_deformable_object import UipcDeformableObject

if TYPE_CHECKING:
    pass


@configclass
class UipcDeformableObjectCfg(UipcObjectCfg):
    class_type: type = UipcDeformableObject

    # contact_model:

    @configclass
    class StableNeoHookeanCfg:
        # class_type = StableNeoHookean
        youngs_modulus: float = 0.01
        """
        in [MPa]
        """

        poisson_rate: float = 0.49
        """ Poission Rate

        Has to be < 0.5.
        """

    constitution_cfg: StableNeoHookeanCfg = StableNeoHookeanCfg()

    @configclass
    class DeformationVisCfg:
        """Settings of the deformation visualization (``debug_deformation_vis``).

        The chosen quantity is computed per simulation vertex, interpolated onto the render mesh and mapped onto a
        color ramp (blue = low, green, red = high). Optionally, the elastic nodal forces are drawn as arrows.
        """

        quantity: Literal["energy", "force", "displacement"] = "energy"
        """Per-vertex quantity used for coloring.

        - ``"energy"``: elastic strain energy density [J/m^3 = Pa] (Stable Neo-Hookean as in libuipc).
        - ``"force"``: magnitude of the elastic (internal) nodal force [N]. Shows where the body is loaded, i.e. at
          contacts and attachments. Close to zero in the unloaded interior at equilibrium.
        - ``"displacement"``: displacement from the rest shape after rigid alignment (Kabsch) [m].
        """

        value_range: tuple[float, float] | None = None
        """Values mapped to the start and the end of the color ramp, in the unit of :attr:`quantity`.

        If None, the range is ``(0, max)`` of the current frame (relative visualization).
        """

        auto_range_floor: float | None = None
        """Min. upper bound of the automatic range (unit of :attr:`quantity`).

        Avoids that numerical noise of an undeformed body is stretched over the whole color ramp.
        If None: 1e-3 Pa (energy), 1e-6 N (force), 1e-6 m (displacement).
        """

        log_scale: bool = False
        """Map the values logarithmically. Useful since the energy is concentrated around contacts."""

        log_decades: float = 3.0
        """Number of decades below the upper bound that are shown with :attr:`log_scale`."""

        draw_force_arrows: bool = False
        """Draw the elastic nodal forces of the simulation surface vertices as lines (:class:`DebugDraw`).

        The arrows point in the direction of the elastic restoring force, i.e. against the external load.
        """

        force_arrow_scale: float | None = None
        """Arrow length per force [m/N]. If None, the longest arrow is 10% of the bounding box diagonal."""

        force_arrow_min_ratio: float = 0.05
        """Arrows of forces smaller than ``force_arrow_min_ratio * max force`` are not drawn."""

        force_arrow_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 1.0)
        """RGBA color of the force arrows."""

        force_arrow_width: float | None = None
        """Line width of the force arrows [m]. If None, 0.3 % of the bounding box diagonal."""

        material_path: str = "/World/Materials/DeformationVisMat"
        """Path of the deformation vis material. Shared by all objects with the same :attr:`primvar_name`."""

        primvar_name: str = "deformValue"
        """Name of the per-vertex primvar that holds the normalized values."""

    debug_deformation_vis: bool = False
    """Color the render mesh by its deformation, see :attr:`deformation_vis_cfg`."""

    deformation_vis_cfg: DeformationVisCfg = DeformationVisCfg()
    """Settings of the deformation visualization. Only used if :attr:`debug_deformation_vis` is True."""
