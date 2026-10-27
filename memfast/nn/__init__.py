"""Analog inference, hardware-aware training, and recurrent layers."""

from .conversion import convert_model, reset_read_events
from .layers import AnalogConv2d, AnalogLinear
from .recurrent import AnalogRNN, AnalogRNNCell
from .training import TrainableAnalogConv2d, TrainableAnalogLinear

HATLinear = TrainableAnalogLinear
HATConv2d = TrainableAnalogConv2d

__all__ = [
    "AnalogLinear",
    "AnalogConv2d",
    "TrainableAnalogLinear",
    "TrainableAnalogConv2d",
    "HATLinear",
    "HATConv2d",
    "AnalogRNNCell",
    "AnalogRNN",
    "convert_model",
    "reset_read_events",
]
