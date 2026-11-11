from __future__ import annotations

from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import VISUO_TACTILE_SENSOR_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab_contrib.sensors.tacsl_sensor import GelSightRenderCfg

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .tacsl_sim import TacSLSimulator

"""Configuration for the TacSL simulation approach (Isaac Lab's visuo-tactile sensor) in TacEx."""


@configclass
class TacSLForceFieldCfg:
    """Force field part of TacSL: penalty-based contact forces at a grid of points on the gelpad surface.

    Field names match Isaac Lab's ``VisuoTactileSensorCfg``, whose force field code is reused.
    """

    elastomer_prim_path_expr: str | None = None
    """Prim path expression of the gelpad (elastomer) rigid body, e.g. ``{ENV_REGEX_NS}/Robot/gelsight_mini_gelpad``.

    Required when the sensor requests ``"force_field"``. Must be a rigid body; the tactile points are generated on the
    first mesh below it. ``{ENV_REGEX_NS}`` is resolved from the sensor's prim path.
    """

    contact_object_prim_path_expr: str | None = None
    """Prim path expression of the object touching the gelpad, e.g. ``{ENV_REGEX_NS}/object``.

    The object needs an SDF collision mesh (``SDFMeshPropertiesCfg``). If None, only the tactile point positions are
    computed (forces stay zero).
    """

    tactile_array_size: tuple[int, int] = (20, 25)
    """Number of tactile points (rows, cols)."""

    tactile_margin: float = 0.002
    """Margin (m) between the outermost tactile points and the gelpad mesh edges."""

    contact_normal_local: tuple[float, float, float] | None = None
    """Outward contact normal of the gelpad surface in the elastomer body frame, e.g. ``(0.0, 0.0, 1.0)``.

    Must be parallel to the gelpad's thinnest axis. If None, it is detected like in TacSL from the mesh's dome
    (center of mass vs. bounding box center), which fails for flat gelpads: set it explicitly for those.
    """

    normal_contact_stiffness: float = 1.0
    """Normal contact stiffness for the penalty force F_n = k_n * penetration_depth."""

    friction_coefficient: float = 2.0
    """Friction coefficient mu for the shear force limit mu * F_n."""

    tangential_stiffness: float = 0.1
    """Tangential stiffness for shear forces F_t = k_t * |v_t| (before the friction limit)."""

    visualizer_cfg: VisualizationMarkersCfg = VISUO_TACTILE_SENSOR_MARKER_CFG.replace(
        prim_path="/Visuals/TacSLTactilePoints"
    )
    """Markers for the tactile points (shown when the sensor's debug visualization is on)."""

    visualize_sdf_closest_pts: bool = False
    """Show the closest points on the contact object instead of the tactile points (debugging)."""

    trimesh_vis_tactile_points: bool = False
    """Open a trimesh window with the generated tactile points at initialization (debugging)."""


@configclass
class TacSLSimulatorCfg(GelSightSimulatorCfg):
    """Config for the TacSL simulation approach.

    TacSL provides two outputs; the sensor decides which ones are used via its cfg slots and data types:

    - optical (``optical_sim_cfg`` + ``"tactile_rgb"``): Taxim-style polynomial rendering (Isaac Lab ``GelsightRender``)
      of the sensor's height map.
    - force field (``force_field_sim_cfg`` + ``"force_field"``): SDF-based penalty forces at points on the gelpad.

    Pass the same cfg object to both slots to use both outputs from one simulator instance.
    """

    simulation_approach_class: type = TacSLSimulator

    device: str = "cuda"

    tactile_img_res: tuple[int, int] = (320, 240)
    """Resolution (width, height) of the tactile RGB output. The image is rendered at the calibration resolution
    (``render_cfg.image_width`` x ``image_height``) and resized to this resolution."""

    render_cfg: GelSightRenderCfg = MISSING
    """TacSL renderer config: calibration data (``polycalib.npz``, background image) and its image resolution.

    The calibration is tied to its image resolution, so ``image_width``/``image_height`` must be the resolution the
    calibration was made for (not the output resolution).
    """

    background_override_path: str | None = None
    """Optional no-contact background replacing ``render_cfg.background_path``: an image file, or a Taxim
    ``dataPack.npz`` (key ``f0``, BGR). Must have the calibration resolution."""

    mm_per_pixel_from_sensor: bool = True
    """If True, the render scale is the sensor camera view divided by the render resolution, so that the gelpad
    slopes are computed in the same physical units as the simulated view. Otherwise ``render_cfg.mm_per_pixel``."""

    force_field: TacSLForceFieldCfg = TacSLForceFieldCfg()
    """Force field settings (only used when the sensor requests ``"force_field"``)."""
