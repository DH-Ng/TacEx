"""Debug drawing of points and lines as USD prims (replacement for ``isaacsim.util.debug_draw``).

``isaacsim.util.debug_draw`` is one global, viewport-only overlay: ``clear_points()`` / ``clear_lines()`` erase what
every other user has drawn, it is not available headless and does not show up in recordings.

:class:`DebugDraw` keeps the same calls (``draw_points``, ``draw_lines``, ``clear_points``, ``clear_lines``) but draws
into its own prims below ``prim_path`` (by convention under ``/Visuals``). For points/lines that move every frame, use
``set_points`` / ``set_lines``: they replace the previous ones in one update (instead of clear + draw).

- points: Isaac Lab :class:`~isaaclab.markers.VisualizationMarkers` (spheres), one marker set per color,
- lines: one ``UsdGeom.BasisCurves`` prim (linear segments).

Both are invisible to secondary rays, i.e. they do not show up in depth images (e.g. GelSight height maps), but they
are visible in RGB camera images.

Differences to ``debug_draw``: sizes are in world units (m) instead of pixels; the alpha of colors is ignored.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Color = Sequence[float]


def _to_rgb(color: Color) -> tuple[float, float, float]:
    """RGB in [0, 1] from an RGB(A) color in [0, 1] or [0, 255] (alpha ignored)."""
    c = np.asarray(color, dtype=np.float64).reshape(-1)[:3]
    if c.max() > 1.0:
        c = c / 255.0
    return tuple(float(v) for v in np.clip(c, 0.0, 1.0))


def _expand_colors(colors, n: int, default: Color) -> list[tuple[float, float, float]]:
    """None / one color / one color per item -> list of n RGB colors in [0, 1]."""
    if colors is None:
        return [_to_rgb(default)] * n
    arr = np.asarray(colors, dtype=np.float64)
    if arr.ndim == 1 or len(arr) == 1:  # a single color
        return [_to_rgb(arr.reshape(-1))] * n
    if len(arr) != n:
        raise ValueError(f"Got {len(arr)} colors for {n} items.")
    return [_to_rgb(c) for c in arr]


def _expand_sizes(sizes, n: int, default: float) -> np.ndarray:
    """None / one size / one size per item -> array of n sizes."""
    if sizes is None:
        return np.full(n, default, dtype=np.float64)
    arr = np.asarray(sizes, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, arr[0])
    if arr.size != n:
        raise ValueError(f"Got {arr.size} sizes for {n} items.")
    return arr


class DebugDraw:
    """Draws points and lines into its own USD prims (see module docstring)."""

    def __init__(
        self,
        prim_path: str = "/Visuals/DebugDraw",
        point_radius: float = 2e-4,
        line_width: float = 1e-4,
        color: Color = (1.0, 0.0, 1.0),
    ):
        """
        Args:
            prim_path: Parent prim of the drawing prims. Use one path per user.
            point_radius: Default radius of the points [m].
            line_width: Default width of the lines [m].
            color: Default color (RGB(A), [0, 1] or [0, 255]).
        """
        self.prim_path = prim_path
        self.point_radius = point_radius
        self.line_width = line_width
        self.color = color
        self._visible = True
        # points: color -> list of (positions (N, 3), radii (N,))
        self._points: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._markers: dict[tuple, object] = {}
        self._counts: dict[tuple, int] = {}  # number of instances last written per color
        # lines: list of (starts (N, 3), ends (N, 3), colors (N, 3), widths (N,))
        self._lines: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._curves = None

    """
    Drawing.
    """

    def draw_points(self, points, colors: Color | Sequence[Color] | None = None, sizes=None):
        """Adds points (spheres). They stay until :meth:`clear_points`.

        Args:
            points: Positions in world frame. Shape (N, 3).
            colors: One color for all points or one per point. Defaults to :attr:`color`.
            sizes: Radius [m], one for all points or one per point. Defaults to :attr:`point_radius`.
        """
        for color in self._add_points(points, colors, sizes):
            self._update_points(color)

    def set_points(self, points, colors: Color | Sequence[Color] | None = None, sizes=None):
        """Replaces all points of this instance (one update; use this for moving points instead of clear + draw).

        Args: see :meth:`draw_points`.
        """
        self._points.clear()
        touched = self._add_points(points, colors, sizes)
        for color in dict.fromkeys([*self._markers, *touched]):
            self._update_points(color)

    def draw_lines(self, starts, ends, colors: Color | Sequence[Color] | None = None, sizes=None):
        """Adds line segments ``starts[i] -> ends[i]``. They stay until :meth:`clear_lines`.

        Args:
            starts: Start points in world frame. Shape (N, 3).
            ends: End points in world frame. Shape (N, 3).
            colors: One color for all lines or one per line. Defaults to :attr:`color`.
            sizes: Width [m], one for all lines or one per line. Defaults to :attr:`line_width`.
        """
        if self._add_lines(starts, ends, colors, sizes):
            self._update_lines()

    def set_lines(self, starts, ends, colors: Color | Sequence[Color] | None = None, sizes=None):
        """Replaces all lines of this instance (one update; use this for moving lines instead of clear + draw).

        Args: see :meth:`draw_lines`.
        """
        self._lines.clear()
        added = self._add_lines(starts, ends, colors, sizes)
        if added or self._curves is not None:
            self._update_lines()

    def clear_points(self):
        """Removes all points drawn by this instance."""
        self._points.clear()
        for color in list(self._markers):
            self._update_points(color)

    def clear_lines(self):
        """Removes all lines drawn by this instance."""
        self._lines.clear()
        if self._curves is not None:
            self._update_lines()

    def clear(self):
        """Removes all points and lines drawn by this instance."""
        self.clear_points()
        self.clear_lines()

    def set_visibility(self, visible: bool):
        """Shows/hides everything drawn by this instance (drawing while hidden is kept for later)."""
        self._visible = visible
        for color, markers in self._markers.items():
            if markers.is_visible() != visible:
                markers.set_visibility(visible)
            if visible:
                self._update_points(color)  # catch up on what was drawn while hidden
        if self._curves is not None and self._curves.GetPrim().IsValid():
            from pxr import UsdGeom

            imageable = UsdGeom.Imageable(self._curves)
            imageable.MakeVisible() if visible else imageable.MakeInvisible()

    """
    Internal helpers.
    """

    def _add_points(self, points, colors, sizes) -> list[tuple]:
        """Stores points (grouped by color), returns the colors that got points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = points.shape[0]
        if n == 0:
            return []
        colors = _expand_colors(colors, n, self.color)
        radii = _expand_sizes(sizes, n, self.point_radius)
        touched = list(dict.fromkeys(colors))
        for color in touched:
            mask = np.array([c == color for c in colors])
            self._points.setdefault(color, []).append((points[mask], radii[mask]))
        return touched

    def _add_lines(self, starts, ends, colors, sizes) -> bool:
        """Stores line segments, returns whether any were added."""
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        n = starts.shape[0]
        if n == 0:
            return False
        if ends.shape[0] != n:
            raise ValueError(f"Got {n} start but {ends.shape[0]} end points.")
        colors = np.array(_expand_colors(colors, n, self.color))
        widths = _expand_sizes(sizes, n, self.line_width)
        self._lines.append((starts, ends, colors, widths))
        return True

    def _update_points(self, color: tuple):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        import isaaclab.sim as sim_utils

        markers = self._markers.get(color)
        if markers is None or not markers._instancer_manager.GetPrim().IsValid():  # (re)create, e.g. after a new stage
            markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path=f"{self.prim_path}/points_{len(self._markers)}",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=1.0, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                        )
                    },
                )
            )
            markers.set_visibility(self._visible)
            self._markers[color] = markers
            self._counts.pop(color, None)
        if not self._visible:
            return  # written when shown again
        chunks = self._points.get(color, [])
        positions = np.concatenate([p for p, _ in chunks]) if chunks else np.zeros((0, 3))
        radii = np.concatenate([r for _, r in chunks]) if chunks else np.zeros(0)
        # write the instancer directly instead of `VisualizationMarkers.visualize`: it does not accept zero markers,
        # and clearing must not toggle the visibility (clear + draw every frame would make the prim blink)
        from pxr import Vt

        instancer = markers._instancer_manager
        instancer.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions.astype(np.float32)))
        instancer.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(radii[:, None], 3, axis=1).astype(np.float32)))
        n = positions.shape[0]
        if self._counts.get(color) != n:
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.zeros(n, dtype=np.int32)))
            self._counts[color] = n

    def _update_lines(self):
        from pxr import Sdf, UsdGeom, Vt

        import omni.usd

        if self._curves is None or not self._curves.GetPrim().IsValid():  # (re)create, e.g. after a new stage
            stage = omni.usd.get_context().get_stage()
            curves = UsdGeom.BasisCurves.Define(stage, f"{self.prim_path}/lines")
            curves.CreateTypeAttr(UsdGeom.Tokens.linear)
            # not visible in depth images (like VisualizationMarkers)
            UsdGeom.PrimvarsAPI(curves).CreatePrimvar("invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
            if not self._visible:
                UsdGeom.Imageable(curves).MakeInvisible()
            self._curves = curves
        if self._lines:
            starts = np.concatenate([s for s, _, _, _ in self._lines])
            ends = np.concatenate([e for _, e, _, _ in self._lines])
            colors = np.concatenate([c for _, _, c, _ in self._lines])
            widths = np.concatenate([w for _, _, _, w in self._lines])
        else:
            starts = ends = colors = np.zeros((0, 3))
            widths = np.zeros(0)
        points = np.stack([starts, ends], axis=1).reshape(-1, 3).astype(np.float32)
        curves = self._curves
        curves.GetCurveVertexCountsAttr().Set([2] * starts.shape[0])
        curves.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        curves.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.repeat(widths, 2).astype(np.float32)))
        curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
