from __future__ import annotations

from typing import Any, Dict, Tuple, Union, List, TYPE_CHECKING

from isaaclab.utils import configclass

import uipc
from uipc import Logger, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.unit import MPa, GPa

if TYPE_CHECKING:
    from tacex_ipc import UipcObject

@configclass
class UipcSimCfg:
    device: str | None = "cuda"


class UipcSim():
    cfg: UipcSimCfg

    def __init__(self, cfg: UipcSimCfg = None):
        """Initialize the uipc simulation.
        
        """
        if cfg is None:
            cfg = UipcSimCfg()
        self.cfg = cfg

        self.engine: Engine = Engine(self.cfg.device)
        self.world: World = World(self.engine)
        self.config = Scene.default_config()

        dt = 0.02
        self.config["dt"] = dt
        
        self.config["gravity"] = [[0.0], [0.0], [-9.8]]
        self.scene = Scene(self.config)

        self.uipc_objects: list[UipcObject] = []

        Logger.set_level(Logger.Error)

    def setup_scene(self):
        self.world.init(self.scene)

    def step(self):
        self.world.advance()
        self.world.retrieve()
    
    def update_meshes(self):
        for uipc_obj in self.uipc_objects:
            uipc_obj._update_meshes()
