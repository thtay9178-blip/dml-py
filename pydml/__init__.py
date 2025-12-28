"""
DML-PY - A Collaborative Deep Learning Library

Main package for collaborative neural network training.
"""

__version__ = "0.1.0"
__author__ = "DML-PY Contributors"
__license__ = "MIT"

from pydml.trainers.dml import DMLTrainer, DMLConfig
from pydml.core.base_trainer import BaseCollaborativeTrainer

__all__ = [
    "DMLTrainer",
    "DMLConfig",
    "BaseCollaborativeTrainer",
]
