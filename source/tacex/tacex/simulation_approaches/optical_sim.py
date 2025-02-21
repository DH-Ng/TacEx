"""Base class for optical simulation approaches.

This class defines an interface for simulating the optical response of GelSight tactile sensors.
Each simulation method class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import ABC, abstractmethod

class OpticalSimulator(ABC):
    """Base class for implementing an optical simulation approach.
    
    
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self._initialize_impl()

    @abstractmethod
    def _initialize_impl(self):
        raise NotImplementedError

    @abstractmethod
    def optical_simulation(self):
        """Simulates the optical output of a Tactile sensor.

        """
        raise NotImplementedError

    #@abstractmethod make it optional, in case another method is used for computing indentation depth
    def compute_indentation_depth(self):
        """Computes how deep the indenter is pressed into the gelpad"""
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError