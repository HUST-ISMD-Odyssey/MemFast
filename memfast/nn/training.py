"""Analog forward values with selectable quantized-operand or digital STE.

Master weights and biases are ordinary Parameters used by the surrogate graph.
Each invocation reprograms persistent analog buffers from those parameters;
parameter and buffer identities remain stable for optimizers and DDP.
Surrogate arithmetic and its derivatives use FP32, including wide/BFP mapping
and autocast contexts. Quantization and programmed scale buffers retain their
physical precision. Leaf gradients accumulate in their Parameters' own dtype.

The default ``quantized`` mode follows the legacy HAT quantized-operand
convention: dx uses dequantized weights, dw uses dequantized inputs, and the
quantizer itself has identity derivatives. ``digital`` differentiates the
unquantized digital layer instead. Neither mode differentiates device effects.
Activations remain signed; weight operands follow ``Config.weight_signed``.
Unsigned HAT requires nonnegative master weights on every programming step.

Conv2d quantizes unfolded patches and each group's physical [K,O] weights.
Its gradients use those patches directly. The old Conv2d implementation first
folded and averaged overlapping dequantized patches into an image; that can
change weight gradients when patch scales differ, so the two convolution
backwards are not claimed to be numerically identical.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..quantization import grouped_quantize
from .convolution import AnalogConv2d


class _SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, analog, surrogate):
        ctx.surrogate_dtype = surrogate.dtype
        return analog

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.to(ctx.surrogate_dtype)


@torch.no_grad()
def _program(matrix, weight, bias):
    programmed = matrix.simulator.map(weight, bias=bias, name=matrix.name)
    # Keep the original buffers, sampling identity, read counter, and selections.
    matrix.packed.copy_(programmed.packed)
    matrix.weight_scale.copy_(programmed.weight_scale)
    if matrix.bias is not None:
        matrix.bias.copy_(programmed.bias)


def _parameter(value, device):
    if value is None:
        return None
    return nn.Parameter(value.detach().to(device).clone(), requires_grad=value.requires_grad)


def _validate_gradient_mode(mode):
    if not isinstance(mode, str) or mode not in ("quantized", "digital"):
        raise ValueError("gradient_mode must be quantized or digital")
    return mode


def _quantized_operand(value, bits, granularity, config, reference=None, *, signed=True):
    operand = value.float()
    if value.numel() == 0:
        return operand
    with torch.no_grad():
        original = value if reference is None else reference
        code, scale = grouped_quantize(
            original.detach(), bits, granularity, config.bw_e, signed=signed
        )
        dequantized = (code * scale).float()
    return operand + (dequantized - operand).detach()


def _quantized_linear(x, weight, bias, config, input_reference=None):
    input_operand = _quantized_operand(
        x.reshape(-1, weight.shape[1]),
        config.activation_bits,
        config.input_quant_gran,
        config,
        reference=None if input_reference is None else input_reference.reshape(-1, weight.shape[1]),
    )
    weight_operand = _quantized_operand(
        weight.T,
        config.weight_bits,
        config.weight_quant_gran,
        config,
        signed=config.weight_signed,
    )
    output = input_operand @ weight_operand
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output.reshape(*x.shape[:-1], weight.shape[0])


class TrainableAnalogLinear(nn.Module):
    """Hardware-aware Linear with an exact analog forward and selectable STE.

    ``gradient_mode="quantized"`` uses dequantized grouped operands in backward.
    ``"digital"`` uses the unquantized F.linear derivatives. Leading input
    dimensions are flattened before quantization, as in the physical engine.
    """

    def __init__(self, source, simulator, *, name="linear", gradient_mode="quantized"):
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError("source must be torch.nn.Linear")
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight = _parameter(source.weight, simulator.device)
        self.register_parameter("bias", _parameter(source.bias, simulator.device))
        self.matrix = simulator.map(self.weight, bias=self.bias, name=name)
        self.train(source.training)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            _program(self.matrix, self.weight, self.bias)
            analog = self.matrix(x.detach())
            if not torch.is_grad_enabled():
                return analog
            if self.gradient_mode == "quantized":
                surrogate = _quantized_linear(x, self.weight, self.bias, self.matrix.config)
            else:
                bias = None if self.bias is None else self.bias.float()
                surrogate = F.linear(x.float(), self.weight.float(), bias)
        return _SurrogateGradient.apply(analog, surrogate)


class TrainableAnalogConv2d(nn.Module):
    """Grouped Conv2d with an analog forward and selectable STE gradients.

    Default gradients use quantized unfolded [N*patches,K] inputs separately
    for each convolution group and physical [K,O] weight quantization. Digital
    mode uses F.conv2d derivatives. Biases are never quantized.
    """

    def __init__(self, source, simulator, *, name="conv", gradient_mode="quantized"):
        super().__init__()
        if not isinstance(source, nn.Conv2d):
            raise TypeError("source must be torch.nn.Conv2d")
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self.weight = _parameter(source.weight, simulator.device)
        self.register_parameter("bias", _parameter(source.bias, simulator.device))
        self.analog = AnalogConv2d(source, simulator, name=name)
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
        self.train(source.training)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x)

    def _forward(self, x):
        per_group = self.out_channels // self.groups
        for group, matrix in enumerate(self.analog.matrices):
            sl = slice(group * per_group, (group + 1) * per_group)
            bias = None if self.bias is None else self.bias[sl]
            _program(matrix, self.weight[sl].flatten(1), bias)
        analog = self.analog(x.detach())
        if not torch.is_grad_enabled():
            return analog
        original = x.detach()
        x = x.float()
        padding = self.padding
        if self.padding_mode != "zeros":
            ph, pw = padding
            x = F.pad(x, (pw, pw, ph, ph), mode=self.padding_mode)
            original = F.pad(original, (pw, pw, ph, ph), mode=self.padding_mode)
            padding = (0, 0)
        if self.gradient_mode == "quantized":
            patches = F.unfold(x, self.kernel_size, self.dilation, padding, self.stride)
            reference = (
                patches.detach()
                if original.dtype == torch.float32
                else F.unfold(original, self.kernel_size, self.dilation, padding, self.stride)
            )
            size = self.weight[0].numel()
            values = []
            for group, matrix in enumerate(self.analog.matrices):
                sl = slice(group * per_group, (group + 1) * per_group)
                bias = None if self.bias is None else self.bias[sl]
                values.append(
                    _quantized_linear(
                        patches[:, group * size : (group + 1) * size].transpose(1, 2),
                        self.weight[sl].flatten(1),
                        bias,
                        matrix.config,
                        input_reference=reference[:, group * size : (group + 1) * size].transpose(
                            1, 2
                        ),
                    )
                )
            surrogate = torch.cat(values, dim=-1).transpose(1, 2).reshape(analog.shape)
        else:
            bias = None if self.bias is None else self.bias.float()
            surrogate = F.conv2d(
                x, self.weight.float(), bias, self.stride, padding, self.dilation, self.groups
            )
        return _SurrogateGradient.apply(analog, surrogate)
