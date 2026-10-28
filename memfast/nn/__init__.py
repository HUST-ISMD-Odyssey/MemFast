"""Analog inference, hardware-aware training, and recurrent layers."""

from .conversion import convert_model, reset_read_events
from .convolution import AnalogConv2d
from .linear import AnalogLinear
from .modules import Conv2d, Linear
from .recurrent import AnalogGRU, AnalogGRUCell, AnalogRNN, AnalogRNNCell
from .training import TrainableAnalogConv2d, TrainableAnalogLinear

HATLinear = TrainableAnalogLinear
HATConv2d = TrainableAnalogConv2d

__all__ = [
    "Linear",
    "Conv2d",
    "AnalogLinear",
    "AnalogConv2d",
    "TrainableAnalogLinear",
    "TrainableAnalogConv2d",
    "HATLinear",
    "HATConv2d",
    "AnalogRNNCell",
    "AnalogRNN",
    "AnalogGRUCell",
    "AnalogGRU",
    "convert_model",
    "reset_read_events",
]
