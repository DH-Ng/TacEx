from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, Union

import torch
from isaaclab.utils import class_to_dict, configclass, to_camel_case

"""Configuration for a tactile RGB simulation with Taxim."""
@configclass
class GelSightSimulatorCfg():
    """Parent Class for Simulation Approach Cfg classes.

    Basically, only `simulation_approach_class` is important (right now at least).
    It could very well be that this class is pretty much useless/overkill.
    """
    simulation_approach_class: type = None
    """"""
    device: str = "cuda"
