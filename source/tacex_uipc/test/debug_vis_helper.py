"""Live visualization for the UIPC scene tests.

Set ``TACEX_TEST_DEBUG_VIS=1`` to watch a test in the Isaac Sim GUI instead of running headless:

    TACEX_TEST_DEBUG_VIS=1 TACEX_TEST_DEBUG_VIS_HOLD=30 python -m pytest source/tacex_uipc/test/render_mesh/test_render_mesh_voxel.py -v -s

With the flag, the tests
- launch the GUI (not headless) and render every simulation step,
- add a dome light and point the viewport camera at the scene,
- enable ``debug_vis`` of the UIPC objects (e.g. simulation mesh wireframe, carrier mesh),
- keep the window open after the run for ``TACEX_TEST_DEBUG_VIS_HOLD`` seconds (default 30) or until it is closed. Simulation keeps playing till closed if negative.

Must be importable before the Isaac Sim app is launched (only standard library imports at module level).
"""

from __future__ import annotations

import os
import time

DEBUG_VIS: bool = os.environ.get("TACEX_TEST_DEBUG_VIS", "0").strip().lower() in ("1", "true", "yes", "on")
"""Whether the tests run with live visualization."""

HOLD_S: float = float(os.environ.get("TACEX_TEST_DEBUG_VIS_HOLD", "30"))
"""Seconds the window stays open after the run (live visualization only).

If < 0: simulation keeps running till window is closed.
"""


def launch_app():
    """Launches the Isaac Sim app: GUI with ``TACEX_TEST_DEBUG_VIS``, headless otherwise."""
    from isaaclab.app import AppLauncher

    # cameras: loads omni.ui, which tacex_uipc imports
    return AppLauncher(headless=not DEBUG_VIS, enable_cameras=True).app


def setup_view(sim, eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Adds lights and points the viewport camera at the scene (live visualization only).

    A weak dome light (ambient, also the background) plus a directional light from above at the side for shading.
    A bright dome light alone washes the scene out.
    """
    if not DEBUG_VIS:
        return
    import isaaclab.sim as sim_utils

    dome_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(1.0, 1.0, 1.0))
    dome_cfg.func("/World/DebugVisLight", dome_cfg)
    sun_cfg = sim_utils.DistantLightCfg(intensity=1500.0, angle=1.0)
    sun_cfg.func("/World/DebugVisSun", sun_cfg, orientation=(0.8536, 0.3536, 0.3536, 0.1464))  # ~45 deg from above
    sim.set_camera_view(eye, target)


def step(sim):
    """Steps the simulation; renders the step with live visualization."""
    sim.step(render=DEBUG_VIS)


def hold(sim):
    """Keeps rendering the final state for ``HOLD_S`` seconds or until the window is closed (live visualization)."""
    if not DEBUG_VIS:
        return
    import omni.kit.app

    app = omni.kit.app.get_app()
    if HOLD_S < 0.0:
        print("[debug vis] playing the simulation till window is closed).", flush=True)
        while app.is_running():
            sim.render()

    print(f"[debug vis] holding the final state for {HOLD_S:.0f} s (close the window to continue).", flush=True)
    t_end = time.time() + HOLD_S
    while app.is_running() and time.time() < t_end:
        sim.render()
