from __future__ import annotations

import dataclasses

from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from .tacex_robot import TacExRobot


@configclass
class TacExRobotCfg(ArticulationCfg):
    """Configuration for :class:`TacExRobot`."""

    class_type: type = TacExRobot

    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    """Differential IK controller settings."""

    ik_body_name: str = "TCP"
    """Body whose frame is controlled by IK."""

    ik_joint_names: list[str] = ["panda_joint.*"]
    """Joints (regex) driven by IK. Other joints stay under joint control (e.g., gripper fingers)."""

    ee_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the controlled frame relative to :attr:`ik_body_name`."""

    ee_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation (w, x, y, z) of the controlled frame relative to :attr:`ik_body_name`."""

    @classmethod
    def from_articulation_cfg(cls, cfg: ArticulationCfg, **kwargs) -> TacExRobotCfg:
        """Wrap an existing :class:`ArticulationCfg` (e.g., a Franka asset cfg); ``kwargs`` override fields."""
        values = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg) if f.name != "class_type"}
        values.update(kwargs)
        print("val ", values)
        return cls(**values)
