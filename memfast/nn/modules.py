"""PyTorch-style layer constructors with analog forward and FP32 backward."""

import itertools
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..config import Config
from ..simulator import AnalogMatrix, Simulator
from .training import TrainableAnalogConv2d, TrainableAnalogLinear

_layer_ids = itertools.count()


def _engine(simulator, config, device, mode, cache_dir):
    if simulator is None:
        return Simulator(
            config,
            device=device if device is not None else torch.empty(0).device,
            mode=mode,
            cache_dir=cache_dir,
        )
    if not isinstance(simulator, Simulator):
        raise TypeError("simulator must be Simulator")
    if config is not None and config != simulator.config:
        raise ValueError("config differs from the supplied simulator")
    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        if requested.type == "cpu":
            requested = torch.device("cpu")
        if requested != simulator.device:
            raise ValueError("device differs from the supplied simulator")
    if mode != "default" and mode != simulator.mode:
        raise ValueError("mode differs from the supplied simulator")
    if cache_dir is not None and Path(cache_dir) != simulator.cache_dir:
        raise ValueError("cache_dir differs from the supplied simulator")
    return simulator


class _Layer:
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if len(args) > 3:
            raise TypeError("state_dict accepts at most three positional arguments")
        if args:
            destination = args[0]
        if len(args) > 1:
            prefix = args[1]
        if len(args) > 2:
            keep_vars = args[2]
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        internal = ("matrix.",) if isinstance(self, TrainableAnalogLinear) else ("analog.",)
        for key in list(state):
            if any(key.startswith(prefix + name) for name in internal):
                del state[key]
        return state

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self.update_weight()
        child_name = "matrix" if isinstance(self, TrainableAnalogLinear) else "analog"
        child = getattr(self, child_name)
        for key, value in child.state_dict().items():
            state_dict.setdefault(prefix + child_name + "." + key, value)

    @property
    def simulator(self):
        matrix = self.matrix if isinstance(self, TrainableAnalogLinear) else self.analog.matrices[0]
        return matrix.simulator

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        matrices = [module for module in self.modules() if isinstance(module, AnalogMatrix)]
        if matrices and matrices[0].simulator.device != self.weight.device:
            old = matrices[0].simulator
            engine = Simulator(
                old.config,
                device=self.weight.device,
                mode=old.mode,
                cache_dir=old.cache_dir,
                workspace_mb=old.workspace_bytes / 1024**2,
            )
            for matrix in matrices:
                matrix.simulator = engine
                matrix._prepared.clear()
        return self

    @torch.no_grad()
    def update_weight(self, weight=None):
        """Copy optional PyTorch-layout weights and refresh physical programming.

        Forward also refreshes programming from the trainable master parameters,
        so ordinary optimizer steps do not require an explicit call.
        """
        replacement = self.weight
        if weight is not None:
            if not isinstance(weight, torch.Tensor) or weight.shape != self.weight.shape:
                raise ValueError("weight must have the layer's original PyTorch shape")
            if not weight.is_floating_point() or not torch.isfinite(weight).all():
                raise ValueError("weight must be real floating and finite")
            replacement = weight.to(self.weight)
            if not torch.isfinite(replacement).all():
                raise ValueError("weight exceeds the layer parameter dtype")
            if not self.simulator.config.weight_signed and (replacement < 0).any():
                raise ValueError("unsigned weights must be nonnegative")
        if isinstance(self, TrainableAnalogLinear):
            plans = [(self.matrix, replacement, self.bias)]
        else:
            size = self.out_channels // self.groups
            plans = []
            for index, matrix in enumerate(self.analog.matrices):
                block = slice(index * size, (index + 1) * size)
                bias = None if self.bias is None else self.bias[block]
                plans.append((matrix, replacement[block].flatten(1), bias))
        # Prepare every group before committing either parameters or programming.
        prepared = [
            AnalogMatrix(values, matrix.simulator, bias=bias, name=matrix.name)
            for matrix, values, bias in plans
        ]
        if weight is not None:
            self.weight.copy_(replacement)
        for (matrix, _, _), programmed in zip(plans, prepared):
            matrix.packed.copy_(programmed.packed)
            matrix.weight_scale.copy_(programmed.weight_scale)
            if matrix.bias is not None:
                matrix.bias.copy_(programmed.bias)
            matrix._prepared.clear()
        return self

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if not self.simulator.config.weight_signed:
            with torch.no_grad():
                self.weight.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.update_weight()


class Linear(_Layer, TrainableAnalogLinear):
    """Linear(in_features, out_features, bias=True, device=None, dtype=None).

    Additional simulator options are keyword-only. Both operands retain
    gradients through the FP32 surrogate; forward values include device effects.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        *,
        simulator=None,
        config: Config | None = None,
        mode="default",
        cache_dir=None,
        gradient_mode="quantized",
    ):
        engine = _engine(simulator, config, device, mode, cache_dir)
        source = nn.Linear(in_features, out_features, bias, device=engine.device, dtype=dtype)
        if not engine.config.weight_signed:
            with torch.no_grad():
                source.weight.abs_()
        name = f"layer.{next(_layer_ids)}"
        super().__init__(source, engine, name=name, gradient_mode=gradient_mode)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Conv2d(_Layer, TrainableAnalogConv2d):
    """Conv2d with PyTorch argument order and keyword-only simulator options."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        *,
        simulator=None,
        config: Config | None = None,
        mode="default",
        cache_dir=None,
        gradient_mode="quantized",
    ):
        engine = _engine(simulator, config, device, mode, cache_dir)
        source = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=engine.device,
            dtype=dtype,
        )
        if not engine.config.weight_signed:
            with torch.no_grad():
                source.weight.abs_()
        explicit_padding = None
        if isinstance(padding, str):
            explicit_padding = (
                tuple(source._reversed_padding_repeated_twice),
                "constant" if padding_mode == "zeros" else padding_mode,
            )
            source.padding = (0, 0)
            source.padding_mode = "zeros"
        name = f"layer.{next(_layer_ids)}"
        super().__init__(source, engine, name=name, gradient_mode=gradient_mode)
        self._input_padding = explicit_padding
        self._requested_padding = padding

    def forward(self, input):
        unbatched = input.ndim == 3
        if unbatched:
            input = input.unsqueeze(0)
        if self._input_padding is not None:
            padding, mode = self._input_padding
            if any(padding):
                input = F.pad(input, padding, mode=mode)
        output = super().forward(input)
        return output.squeeze(0) if unbatched else output

    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self._requested_padding!r}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )


__all__ = ["Linear", "Conv2d"]
