from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas

from .ball_roll_base_env import BallRollingEnv


class BallRollingTacSLEnv(BallRollingEnv):
    """Ball rolling env for the TacSL sensor.

    Same as :class:`BallRollingEnv`, but the ball gets an SDF collision mesh: TacSL's force field queries the SDF of
    the contact object (the ball asset only has a bounding-sphere collision).
    """

    def _setup_scene(self):
        super()._setup_scene()
        # physics has not started yet, so every (cloned) env's ball can still be changed
        for mesh_prim in sim_utils.find_matching_prims("/World/envs/env_.*/ball/mesh"):
            schemas.define_mesh_collision_properties(
                mesh_prim.GetPath().pathString, sim_utils.SDFMeshPropertiesCfg(sdf_resolution=256)
            )
