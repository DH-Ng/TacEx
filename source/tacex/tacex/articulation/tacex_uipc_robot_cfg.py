from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .tacex_robot_cfg import TacExRobotCfg
from .tacex_uipc_robot import TacExUipcRobot

if TYPE_CHECKING:
    from tacex_uipc import UipcObjectCfg


@configclass
class TacExUipcRobotCfg(TacExRobotCfg):
    """Configuration for :class:`TacExUipcRobot`."""

    class_type: type = TacExUipcRobot

    uipc_objects: dict[str, UipcObjectCfg] = {}
    """UIPC objects that belong to the robot (e.g., gelpads), by name.

    They are created together with the robot and are accessible via :attr:`TacExUipcRobot.uipc_objects`.
    Prim paths are relative to the robot's prim path (e.g. ``"gsmini_gelpad_left"``); absolute paths (leading
    ``/``) are kept. For a :class:`UipcIsaacAttachmentsCfg` constraint, ``isaaclab_rigid_body_prim_path`` defaults
    to the robot's prim path.
    """
