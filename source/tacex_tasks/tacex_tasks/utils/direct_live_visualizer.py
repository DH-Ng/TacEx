from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING

import carb
import omni.kit.app
import isaacsim
from isaacsim.core.api.simulation_context import SimulationContext

from isaaclab.ui.widgets import ManagerLiveVisualizer

import torch

from .image_plot import ImagePlot
from .line_plot import LiveLinePlot

if TYPE_CHECKING:
    import omni.ui

# import logger
logger = logging.getLogger(__name__)


def _has_gui() -> bool:
    """Whether the app has a GUI (local or streamed); reads Isaac Lab's simulation context if one exists."""
    from isaaclab.sim import SimulationContext as LabSimulationContext

    sim = LabSimulationContext.instance()
    if sim is not None and hasattr(sim, "has_gui"):
        return sim.has_gui()
    # no simulation context yet: fall back to whether the GUI components (used for the window styling) are loaded
    return omni.kit.app.get_app().get_extension_manager().is_extension_enabled("isaacsim.gui.components")


class ExtraVisualizerWindow:
    def __init__(self, num_env, window_name: str = "VisualizerWindow"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        print("Creating window.")
        # create window for UI
        self.ui_window = omni.ui.Window(
            window_name, width=400, height=500, visible=True, dock_preference=omni.ui.DockPreference.RIGHT_TOP
        )

        self.num_envs = num_env
        self._env_idx = 0
        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_window_elements = dict()
        # create main frame
        self.ui_window_elements["main_frame"] = self.ui_window.frame
        with self.ui_window_elements["main_frame"]:
            # create main stack
            self.ui_window_elements["main_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["main_vstack"]:
                # create frame for switching currently viewed env
                self._build_viewer_frame()

                # create collapsable frame for debug visualization
                self._build_debug_vis_frame()

    def _build_viewer_frame(self):
        """Build the viewer-related control frame for the UI."""
        # create collapsable frame for viewer
        self.ui_window_elements["viewer_frame"] = omni.ui.CollapsableFrame(
            title="Viewer Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=isaacsim.gui.components.ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["viewer_frame"]:
            # create stack for controls
            self.ui_window_elements["viewer_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["viewer_vstack"]:
                # create a number slider set from which env the data should be visualized
                # NOTE: slider is 1-indexed, whereas the env index is 0-indexed
                viewport_origin_cfg = {
                    "label": "Environment Index",
                    "type": "button",
                    "default_val": self._env_idx,
                    "min": 0,
                    "max": self.num_envs,
                    "tooltip": "The environment index to follow. Only effective if follow mode is not 'World'.",
                }
                self.ui_window_elements["viewer_env_index"] = isaacsim.gui.components.ui_utils.int_builder(
                    **viewport_origin_cfg
                )
                self.ui_window_elements["viewer_env_index"].add_value_changed_fn(self._set_viewer_env_index_fn)

    def _set_viewer_env_index_fn(self, model: omni.ui.SimpleIntModel):
        self._env_idx = model.as_int

    def _build_debug_vis_frame(self):
        # create collapsable frame for debug visualization
        self.ui_window_elements["debug_frame"] = omni.ui.CollapsableFrame(
            title="Scene Debug Visualization",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=isaacsim.gui.components.ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["debug_frame"]:
            # create stack for debug visualization
            self.ui_window_elements["debug_vstack"] = omni.ui.VStack(spacing=5, height=0)

    def _create_debug_vis_ui_element(self, name: str, elem: object):
        """Create a checkbox for toggling debug visualization for the given element.

        Same function as in `base_env_window.py`.
        """
        from omni.kit.window.extensions import SimpleCheckBox

        with omni.ui.HStack():
            # create the UI element
            text = (
                "Toggle debug visualization."
                if elem.has_debug_vis_implementation
                else "Debug visualization not implemented."
            )
            omni.ui.Label(
                name.replace("_", " ").title(),
                width=isaacsim.gui.components.ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip=text,
            )
            has_cfg = hasattr(elem, "cfg") and elem.cfg is not None
            is_checked = False
            if has_cfg:
                is_checked = (hasattr(elem.cfg, "debug_vis") and elem.cfg.debug_vis) or (
                    hasattr(elem, "debug_vis") and elem.debug_vis
                )
            self.ui_window_elements[f"{name}_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=elem.has_debug_vis_implementation,
                checked=is_checked,
                on_checked_fn=lambda value, e=weakref.proxy(elem): e.set_debug_vis(value),
            )
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()

        # Create a panel for the debug visualization
        if isinstance(elem, ManagerLiveVisualizer):
            self.ui_window_elements[f"{name}_panel"] = omni.ui.Frame(width=omni.ui.Fraction(1))
            if not elem.set_vis_frame(self.ui_window_elements[f"{name}_panel"]):
                print(f"Frame failed to set for ManagerLiveVisualizer: {name}")


class DirectLiveVisualizer(ManagerLiveVisualizer):
    def __init__(
        self, debug_vis: bool, num_envs: int, parent_window: omni.ui.Window = None, visualizer_name: str = "Direct Vis"
    ):
        """Initialize Visualizer Widget for a `Direct Workflow`.

        Basically the same as "ManagerLiveVisualizer" but omits the need for the manager classes.
        Instead, you need set the data manually for the visualizer.
        Args:
            debug_vis: If the visualizer should be rendered
            num_envs: Amount of envs, used to set the correct buffer sized.
            parent_window: The window to which the Visualizer should be attached.
                           Use `None` to create separate window with size (640x640).
            visualizer_name: Name of the visualizer.
        """

        self.visualizer_name = visualizer_name
        self.debug_vis = debug_vis
        self.num_envs = num_envs
        self._env_idx: int = 0
        self._viewer_env_idx = 0
        self._vis_frame: omni.ui.Frame

        # evaluate chosen terms if no terms provided - >use all available.
        self._term_visualizers = {}
        self.terms: dict[str, torch.tensor] = {}
        self.terms_names: dict[str, list[str]] = {}

        # Without a GUI (headless), omni.ui windows and the isaacsim.gui styling are not available. The visualizer
        # then becomes a no-op: envs can still write into `terms`, but nothing is created or drawn.
        self._has_gui = _has_gui()
        self._vis_window: omni.ui.Window | None = None
        if not self._has_gui:
            return

        if parent_window is None:
            # create a window
            self._vis_window = ExtraVisualizerWindow(self.num_envs, f"{visualizer_name} Window")
        else:
            self._vis_window = parent_window

    def create_visualizer(self):
        if not self._has_gui:
            return
        with self._vis_window.ui_window_elements["debug_frame"]:
            with self._vis_window.ui_window_elements["debug_vstack"]:
                self._vis_window._create_debug_vis_ui_element(self.visualizer_name, self)

    def _set_env_selection_impl(self, env_idx: int):
        """Update the index of the selected environment to display.

        Args:
            env_idx: The index of the selected environment.
        """
        if env_idx >= 0 and env_idx < self.num_envs:
            self._env_idx = env_idx
        else:
            logger.warning(f"Environment index is out of range [0, {self.num_envs - 1}]")

    #
    # Implementations
    #

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization implementation.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """
        if not self._has_gui:
            return

        if not hasattr(self, "_vis_frame"):
            raise RuntimeError("No frame set for debug visualization.")

        # Clear internal visualizers
        self._term_visualizers = {}
        self._vis_frame.clear()

        if debug_vis:
            # if enabled create a subscriber for the post update event if it doesn't exist
            if not hasattr(self, "_debug_vis_handle") or self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # if disabled remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

            self._vis_frame.visible = False
            return

        self._vis_frame.visible = True

        with self._vis_frame:
            with omni.ui.VStack():
                # Add a plot in a collapsible frame for each term available
                # self._env_idx
                for name, values in self.terms.items():
                    frame = omni.ui.CollapsableFrame(
                        name,
                        collapsed=False,
                        style={"border_color": 0xFF8A8777, "padding": 4},
                    )
                    with frame:
                        value = values[self._env_idx]

                        terms_names = self.terms_names[name] if name in self.terms_names else None
                        # create line plot for single or multivariable signals
                        len_term_shape = len(value.shape)
                        if len_term_shape == 0:
                            value = value.reshape(1)
                        if len_term_shape <= 1:
                            plot = LiveLinePlot(
                                y_data=[[elem] for elem in value.T.tolist()],
                                plot_height=150,
                                show_legend=True,
                                legends=terms_names,
                            )
                            self._term_visualizers[name] = plot
                        # create an image plot for 2d and greater data (i.e. mono and rgb images)
                        elif len_term_shape == 2 or len_term_shape == 3:
                            image = ImagePlot(
                                image=value.cpu().numpy(),
                                label=name,
                            )
                            self._term_visualizers[name] = image
                        else:
                            carb.log_warn(
                                f"DirectLiveVisualizer: Term ({name}) is not a supported data type for visualization."
                            )
                    frame.collapsed = True
        self._debug_vis = debug_vis

    def _debug_vis_callback(self, event):
        """Callback for the debug visualization event."""

        if SimulationContext.instance() is None or not SimulationContext.instance().is_playing():
            # Visualizers have not been created yet.
            return
        self._env_idx = self._vis_window.ui_window_elements["viewer_env_index"].as_int
        # get updated data and update visualization
        for name, values in self.terms.items():
            # E.g. terms = actions: Actions values have the shape (num_envs, num_actions).
            # This means we have `num_actions` amount of plots in our 'actions' timeserie.
            # To plot this, we need to pass over a list of lists.
            # Specifically, `num_actions` amount of lists, where each inner list contains the datapoint for the corresponding timeserie
            value = values[self._env_idx]
            if len(value.shape) == 0:
                value = value.reshape(1)

            vis = self._term_visualizers[name]
            if isinstance(vis, LiveLinePlot):
                vis.add_datapoint(value.T.tolist())
            elif isinstance(vis, ImagePlot):
                vis.update_image(value.cpu().numpy())

        # # get updated data and update visualization
        # for name, values in self.terms.items():
        #     # E.g. terms = actions: Actions values have the shape (num_envs, num_actions).
        #     # This means we have `num_actions` amount of plots in our 'actions' timeserie.
        #     # To plot this, we need to pass over a list of lists.
        #     # Specifically, `num_actions` amount of lists, where each inner list contains the datapoint for the corresponding timeserie
        #     value = values[self._env_idx]
        #     if len(value.shape) == 0:
        #         value = value.reshape(1)

        #     vis = self._term_visualizers[name]
        #     if isinstance(vis, LiveLinePlot):
        #         vis.add_datapoint(value.T.tolist())
        #     elif isinstance(vis, ImagePlot):
        #         vis.update_image(value.cpu().numpy())
