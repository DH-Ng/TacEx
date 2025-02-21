"""Base class for marker motion simulation approaches.

This class defines an interface for simulating the marker motion of GelSight tactile sensors.
Each simulation method class should inherit from this class and implement the abstract methods.
"""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import ABC, abstractmethod

class MarkerMotionSimulator(ABC):
    """Base class for implementing a marker motion simulation approach.
    
    
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self._initialize_impl()

    @abstractmethod
    def _initialize_impl(self):
        raise NotImplementedError

    @abstractmethod
    def marker_motion_simulation(self):
        """Simulates the marker motion of a Tactile sensor.

        """
        raise NotImplementedError

    #@abstractmethod
    def compute_indentation_depth(self):
        """Computes how deep the indenter is pressed into the gelpad"""
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError