"""Linear inference using the simulator's programmed matrices."""

from torch import nn

from ..simulator import AnalogMatrix


class AnalogLinear(AnalogMatrix):
    """Map a PyTorch Linear, preserving its bias and matrix orientation."""

    def __init__(self, source, simulator, *, name="linear"):
        if not isinstance(source, nn.Linear):
            raise TypeError("source must be torch.nn.Linear")
        super().__init__(source.weight, simulator, bias=source.bias, name=name)
