"""MemFast: bit-sliced analog in-memory computing simulation."""

from .config import Config, Nonidealities
from .nn import AnalogConv2d, AnalogLinear, convert_model, reset_read_events
from .simulator import AnalogMatrix, Simulator

__version__ = "0.1.0"
__all__ = [
    "Config",
    "Nonidealities",
    "Simulator",
    "AnalogMatrix",
    "AnalogLinear",
    "AnalogConv2d",
    "convert_model",
    "reset_read_events",
]
