"""UIPC gel attached to an Isaac body via the soft position constraint (``UipcIsaacAttachments``).

Pure UIPC test (no tactile simulation): see ``uipc_attachment_scene.py`` for the scene, trajectory and checks.

Run with the Isaac Lab python (-s prints the measured values):
    [TACEX_TEST_DEBUG_VIS=1] python -m pytest source/tacex_uipc/test/attachment/test_soft_attachment.py -v -s
"""

import debug_vis_helper as dv

simulation_app = dv.launch_app()  # headless, or the GUI with TACEX_TEST_DEBUG_VIS=1 (see debug_vis_helper.py)

"""Rest everything follows."""

import pytest

import uipc_attachment_scene as scene_lib

ATTACHMENT = "soft_position"


@pytest.fixture(scope="module")
def result():
    """One scene and one run for the whole module (one Isaac/UIPC world per process)."""
    return scene_lib.run(scene_lib.build_scene(ATTACHMENT))


def test_attachment_points(result):
    scene_lib.check_attachment_points(result)


def test_gel_follows_case(result):
    scene_lib.check_gel_follows_case(result)


def test_gel_deforms_like_ball(result):
    scene_lib.check_gel_deforms_like_ball(result)


@pytest.mark.xfail(
    reason="the soft position constraint (stiffness ~ gel vertex mass) lets the underside slide ~1.5 mm under load; see the carrier attachment",
    strict=False,
)
def test_underside_stays_on_case(result):
    scene_lib.check_underside_stays_on_case(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
