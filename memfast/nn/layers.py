"""Inference layers using the simulator's programmed matrices."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..simulator import AnalogMatrix


class AnalogLinear(AnalogMatrix):
    """Map a PyTorch Linear, preserving its bias and matrix orientation."""

    def __init__(self, source, simulator, *, name="linear"):
        if not isinstance(source, nn.Linear):
            raise TypeError("source must be torch.nn.Linear")
        super().__init__(source.weight, simulator, bias=source.bias, name=name)


class AnalogConv2d(nn.Module):
    """Conv2d lowered to independently programmed matrices for each group."""

    def __init__(self, source, simulator, *, name="conv"):
        super().__init__()
        if not isinstance(source, nn.Conv2d):
            raise TypeError("source must be torch.nn.Conv2d")
        if isinstance(source.padding, str):
            raise ValueError("use numeric Conv2d padding before conversion")
        for attr in (
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "padding_mode",
        ):
            setattr(self, attr, getattr(source, attr))
        self.matrices = nn.ModuleList()
        per_group = source.out_channels // source.groups
        for group in range(source.groups):
            sl = slice(group * per_group, (group + 1) * per_group)
            bias = None if source.bias is None else source.bias[sl]
            self.matrices.append(
                simulator.map(source.weight[sl].flatten(1), bias=bias, name=f"{name}.group{group}")
            )

    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("expected NCHW input with matching channels")
        original_h, original_w = x.shape[-2:]
        padding = self.padding
        if self.padding_mode != "zeros":
            ph, pw = padding
            x = F.pad(x, (pw, pw, ph, ph), mode=self.padding_mode)
            padding = (0, 0)
        patches = F.unfold(x, self.kernel_size, self.dilation, padding, self.stride)
        size = self.in_channels // self.groups * math.prod(self.kernel_size)
        values = [
            matrix(patches[:, index * size : (index + 1) * size].transpose(1, 2))
            for index, matrix in enumerate(self.matrices)
        ]
        output = torch.cat(values, dim=-1).transpose(1, 2)
        height, width = [
            (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                (original_h, original_w), self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        return output.reshape(x.shape[0], self.out_channels, height, width)
