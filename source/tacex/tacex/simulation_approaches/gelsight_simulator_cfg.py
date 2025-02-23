import torch

from dataclasses import dataclass, MISSING
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union, List
from isaaclab.utils import class_to_dict, to_camel_case, configclass


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