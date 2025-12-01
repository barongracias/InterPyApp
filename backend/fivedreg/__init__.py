"""Public interface for the TensorFlow-based 5D→1D regressor package."""

from .tf_model import build_tf_model
from .trainer_tf import TrainerTF
from .tester_tf import TesterTF

__all__ = ["build_tf_model", "TrainerTF", "TesterTF"]
