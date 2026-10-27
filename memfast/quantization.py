"""Quantization and packing with independent rectangular scale groups."""

import math

import torch
from torch.nn import functional as F


def grouped_quantize(value, bits, granularity, bw_e=None, *, signed=True):
    dtype = torch.float64 if bits > 8 or bw_e is not None else torch.float32
    value = value.to(dtype)
    if not torch.isfinite(value).all():
        raise ValueError(f"quantized values must be finite in {dtype}")
    if not signed and ((value < 0).any() or bw_e is not None):
        raise ValueError(
            "unsigned floating weights must be nonnegative INT values; "
            "use map_quantized with zero_points for affine weights"
        )
    rows, columns = value.shape
    gr, gc = (size if group is None else group for size, group in zip(value.shape, granularity))
    gr, gc = min(gr, rows), min(gc, columns)
    nr, nc = math.ceil(rows / gr), math.ceil(columns / gc)
    padded = F.pad(value, (0, nc * gc - columns, 0, nr * gr - rows))
    grouped = padded.reshape(nr, gr, nc, gc)
    maximum = grouped.abs().amax((1, 3), keepdim=True)
    upper = (1 << (bits - int(signed))) - 1
    if bw_e is None:
        scale = torch.where(maximum > 0, maximum, 1.0) / upper
        minimum = torch.nextafter(
            torch.zeros((), dtype=dtype, device=value.device),
            torch.ones((), dtype=dtype, device=value.device),
        )
        scale = scale.clamp_min(minimum)
        lower = -upper if signed else 0
    else:
        exponent_limit = (1 << (bw_e - 1)) - 1
        exponent = maximum.clamp_min(torch.finfo(dtype).tiny).log2().floor()
        exponent = exponent.clamp(1 - exponent_limit, exponent_limit)
        scale = torch.pow(2.0, exponent + 2 - bits)
        lower = -(1 << (bits - 1))
    code = (grouped / scale).round().clamp_(lower, upper)
    code = code.reshape(nr * gr, nc * gc)[:rows, :columns].to(torch.int64)
    expanded_scale = scale.expand(nr, gr, nc, gc).reshape(nr * gr, nc * gc)
    return code, expanded_scale[:rows, :columns]


def expanded_parameter(value, shape, device, dtype, name, *, positive=False):
    original = (
        torch.as_tensor(value, device=device)
        if torch.is_tensor(value)
        else torch.as_tensor(value, device=device, dtype=torch.float64)
    )
    if original.is_complex() or not torch.isfinite(original).all():
        raise ValueError(f"{name} must be real and finite")
    if positive and (original < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    value = original.to(dtype)
    try:
        value = torch.broadcast_to(value, shape)
    except RuntimeError:
        raise ValueError(f"{name} must broadcast to {tuple(shape)}") from None
    if not torch.isfinite(value).all() or (positive and (value < 0).any()):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if positive else ""))
    return value


def check_read_scales(value, config, name):
    """Each column has a single scale or zero point within one analog read."""
    indices = torch.arange(value.shape[0], device=value.device)
    physical_start = indices // config.array_size[0] * config.array_size[0]
    read_offset = (indices % config.array_size[0]) // config.read_rows * config.read_rows
    if not torch.equal(value, value[physical_start + read_offset]):
        raise ValueError(f"{name} changes inside an ADC read; reduce rows_per_read")


def read_starts(partitions, config, device):
    physical = torch.arange(partitions, device=device)[:, None] * config.array_size[0]
    reads = torch.arange(config.reads_per_array, device=device)[None] * config.read_rows
    return (physical + reads).flatten()


def weight_tiles(weight, config):
    # PyTorch weights [O,K] become physical [K,O] before grouping.
    code, scale = grouped_quantize(
        weight.T.contiguous(),
        config.weight_bits,
        config.weight_quant_gran,
        config.bw_e,
        signed=config.weight_signed,
    )
    return pack_weight(code, scale, config)


def pack_weight(code, scale, config):
    """Pack logical [K,O] integer codes without estimating or changing their scales."""
    k, o = code.shape
    r, c = config.array_size
    m, p = math.ceil(k / r), math.ceil(o / c)
    scale = scale.to(torch.float64 if config.wide else torch.float32)
    padded = F.pad(code, (0, p * c - o, 0, m * r - k))
    dtype = torch.int64 if max(config.activation_bits, config.weight_bits) > 8 else torch.uint8
    packed = (padded & ((1 << config.weight_bits) - 1)).to(dtype)
    packed = packed.reshape(m, r, p, c).permute(0, 2, 1, 3).contiguous()
    # Dequantize each ADC partial before summing physical arrays or subreads.
    # Physical padding belongs to the final logical scale group.
    scales = F.pad(scale[None, None], (0, p * c - o, 0, m * r - k), mode="replicate")[0, 0]
    scales = (
        scales[read_starts(m, config, code.device)]
        .reshape(m * config.reads_per_array, p, c)
        .contiguous()
    )
    return packed, scales


def input_tiles(value, config, fixed_scale=None):
    n, k = value.shape
    r = config.array_size[0]
    m = math.ceil(k / r)
    if fixed_scale is None:
        code, scale = grouped_quantize(
            value, config.activation_bits, config.input_quant_gran, config.bw_e
        )
    else:
        dtype = torch.float64 if config.wide else torch.float32
        if not torch.isfinite(value).all():
            raise ValueError("quantized values must be finite")
        scale = fixed_scale.expand_as(value)
        upper = (1 << (config.activation_bits - 1)) - 1
        lower = -(1 << (config.activation_bits - 1)) if config.bw_e is not None else -upper
        code = (value.to(dtype) / scale).round().clamp(lower, upper).to(torch.int64)
    scale = scale.to(torch.float64 if config.wide else torch.float32)
    padded = F.pad(code, (0, m * r - k))
    dtype = torch.int64 if max(config.activation_bits, config.weight_bits) > 8 else torch.uint8
    packed = (padded & ((1 << config.activation_bits) - 1)).to(dtype)
    scales = F.pad(scale[None], (0, m * r - k), mode="replicate")[0]
    scales = scales[:, read_starts(m, config, value.device)]
    return packed.reshape(n, m, r).contiguous(), scales.contiguous()
