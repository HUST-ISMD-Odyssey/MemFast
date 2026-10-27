"""Inference modules and conversion of PyTorch networks."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .simulator import AnalogMatrix


class AnalogLinear(AnalogMatrix):
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


def convert_model(model, simulator, *, exclude=(), inplace=False):
    """Convert Linear and Conv2d; other operations retain their PyTorch behavior."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch module")
    if isinstance(exclude, str):
        raise TypeError("exclude must be a sequence of module paths")
    excluded = set(exclude)
    names = {name for name, _ in model.named_modules(remove_duplicate=False)}
    missing = excluded - names
    if missing:
        raise ValueError(f"unknown excluded modules: {sorted(missing)}")
    for name, module in model.named_modules():
        if name in excluded or any(name.startswith(path + ".") for path in excluded):
            continue
        if isinstance(module, nn.Conv2d) and isinstance(module.padding, str):
            raise ValueError(f"{name}: numeric Conv2d padding is required")
    target = model if inplace else copy.deepcopy(model)
    target.to(simulator.device).eval()
    replacements = {}

    def convert(module, name):
        if name in excluded or isinstance(module, (AnalogMatrix, AnalogConv2d)):
            return module
        if id(module) in replacements:
            return replacements[id(module)]
        if isinstance(module, nn.Linear):
            result = AnalogLinear(module, simulator, name=name or "linear")
        elif isinstance(module, nn.Conv2d):
            result = AnalogConv2d(module, simulator, name=name or "conv")
        else:
            result = module
            for attr, child in list(module._modules.items()):
                if child is not None:
                    child_name = f"{name}.{attr}" if name else attr
                    setattr(module, attr, convert(child, child_name))
        replacements[id(module)] = result
        return result

    return convert(target, "").eval().requires_grad_(False)


def reset_read_events(model, event=0):
    """Replay a read event across all programmed matrices."""
    for module in model.modules():
        if isinstance(module, AnalogMatrix):
            module.reset_read(event)
