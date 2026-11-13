"""DebugDraw: debug points/lines as USD prims (replacement for isaacsim.util.debug_draw).

Run with the Isaac Lab python:
    python -m pytest source/tacex_uipc/test/utils/test_debug_draw.py -v
"""

import debug_vis_helper as dv

simulation_app = dv.launch_app()

"""Rest everything follows."""

import numpy as np
import pytest
from pxr import UsdGeom

import isaaclab.sim as sim_utils

from tacex_uipc.utils import DebugDraw


@pytest.fixture(scope="module")
def sim():
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda"))
    sim.reset()
    return sim


def _instancer(path):
    stage = sim_utils.get_current_stage()
    prims = [p for p in stage.Traverse() if str(p.GetPath()).startswith(path) and p.IsA(UsdGeom.PointInstancer)]
    return {str(p.GetPath()): UsdGeom.PointInstancer(p) for p in prims}


def _count(instancer: UsdGeom.PointInstancer) -> int:
    pos = instancer.GetPositionsAttr().Get()
    return 0 if pos is None else len(pos)


def test_points_grouped_by_color_and_accumulate(sim):
    draw = DebugDraw("/Visuals/TestDraw/points", point_radius=1e-3)
    draw.draw_points([[0, 0, 0], [1, 0, 0]], [(255, 0, 255, 0.5)] * 2, [30, 40])  # debug_draw style args
    draw.draw_points([[0, 1, 0]], colors=(0.0, 1.0, 0.0))  # other color, default size
    draw.draw_points([[0, 2, 0]], colors=(1.0, 0.0, 1.0))  # same color as first call -> same set, accumulates
    sets = _instancer("/Visuals/TestDraw/points")
    assert len(sets) == 2
    counts = sorted(_count(i) for i in sets.values())
    assert counts == [1, 3]
    # sizes are radii (world units): explicit sizes and the default
    magenta = next(i for i in sets.values() if _count(i) == 3)
    np.testing.assert_allclose(np.array(magenta.GetScalesAttr().Get())[:, 0], [30, 40, 1e-3], rtol=1e-6)


def test_clear_only_affects_own_instance(sim):
    a = DebugDraw("/Visuals/TestDraw/a")
    b = DebugDraw("/Visuals/TestDraw/b")
    a.draw_points([[0, 0, 0]])
    b.draw_points([[1, 1, 1]])
    a.clear_points()
    ia, ib = (next(iter(_instancer(f"/Visuals/TestDraw/{k}").values())) for k in "ab")
    assert _count(ia) == 0 and len(ia.GetProtoIndicesAttr().Get()) == 0
    assert _count(ib) == 1
    a.draw_points([[2, 2, 2]])  # drawing after a clear starts from scratch
    assert _count(ia) == 1


def test_clear_and_draw_does_not_toggle_visibility(sim):
    """clear + draw every frame (like the FEM marker drawing) must not touch the visibility (prim blinks otherwise)."""
    draw = DebugDraw("/Visuals/TestDraw/frames")
    draw.draw_points([[0, 0, 0]])
    instancer = next(iter(_instancer("/Visuals/TestDraw/frames").values()))
    vis_attr = instancer.GetVisibilityAttr()
    before = vis_attr.Get()
    for k in range(5):
        draw.clear_points()
        assert vis_attr.Get() == before  # also not in between
        draw.draw_points([[k, 0, 0], [k, 1, 0]])
    assert vis_attr.Get() == before and instancer.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert _count(instancer) == 2 and len(instancer.GetScalesAttr().Get()) == 2


def test_lines(sim):
    draw = DebugDraw("/Visuals/TestDraw/lines", line_width=2e-4)
    draw.draw_lines([[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 1]], [(255, 255, 0, 0.5)] * 2, [10, 20])
    draw.draw_lines([[0, 0, 2]], [[1, 1, 2]])  # default color and width
    curves = UsdGeom.BasisCurves(sim_utils.get_current_stage().GetPrimAtPath("/Visuals/TestDraw/lines/lines"))
    assert list(curves.GetCurveVertexCountsAttr().Get()) == [2, 2, 2]
    np.testing.assert_allclose(np.array(curves.GetPointsAttr().Get())[[0, 1, 5]], [[0, 0, 0], [1, 0, 0], [1, 1, 2]])
    np.testing.assert_allclose(list(curves.GetWidthsAttr().Get()), [10, 10, 20, 20, 2e-4, 2e-4], rtol=1e-6)
    colors = np.array(curves.GetDisplayColorPrimvar().Get())
    np.testing.assert_allclose(colors[0], [1, 1, 0], atol=1e-6)  # 0-255 color converted, alpha dropped
    assert UsdGeom.PrimvarsAPI(curves).GetPrimvar("invisibleToSecondaryRays").Get() is True  # not in depth images
    draw.clear_lines()
    assert len(curves.GetCurveVertexCountsAttr().Get()) == 0


def test_visibility_keeps_drawing(sim):
    draw = DebugDraw("/Visuals/TestDraw/vis")
    draw.draw_points([[0, 0, 0]])
    draw.set_visibility(False)
    draw.draw_points([[1, 0, 0]])  # drawn while hidden
    instancer = next(iter(_instancer("/Visuals/TestDraw/vis").values()))
    assert instancer.ComputeVisibility() == UsdGeom.Tokens.invisible
    draw.set_visibility(True)
    assert instancer.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert _count(instancer) == 2  # points drawn while hidden show up


def test_invalid_sizes(sim):
    draw = DebugDraw("/Visuals/TestDraw/invalid")
    with pytest.raises(ValueError):
        draw.draw_points([[0, 0, 0], [1, 1, 1]], sizes=[1, 2, 3])


def test_set_replaces(sim):
    """set_points / set_lines replace everything drawn before (moving points without clear + draw)."""
    draw = DebugDraw("/Visuals/TestDraw/set")
    draw.draw_points([[0, 0, 0], [1, 0, 0]], colors=(1.0, 0.0, 0.0))
    draw.draw_points([[0, 1, 0]], colors=(0.0, 0.0, 1.0))
    draw.set_points([[5, 5, 5]], colors=(1.0, 0.0, 0.0))  # replaces both colors
    counts = {p: _count(i) for p, i in _instancer("/Visuals/TestDraw/set").items()}
    assert sorted(counts.values()) == [0, 1]
    draw.draw_lines([[0, 0, 0]], [[1, 1, 1]])
    draw.set_lines([[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]])
    curves = UsdGeom.BasisCurves(sim_utils.get_current_stage().GetPrimAtPath("/Visuals/TestDraw/set/lines"))
    assert len(curves.GetCurveVertexCountsAttr().Get()) == 2
    draw.set_lines(np.zeros((0, 3)), np.zeros((0, 3)))  # empty -> no lines
    assert len(curves.GetCurveVertexCountsAttr().Get()) == 0
