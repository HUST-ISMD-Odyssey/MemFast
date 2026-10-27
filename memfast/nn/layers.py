"""Compatibility imports for the separate Linear and Conv2d modules."""

from .convolution import AnalogConv2d
from .linear import AnalogLinear

__all__ = ["AnalogLinear", "AnalogConv2d"]
