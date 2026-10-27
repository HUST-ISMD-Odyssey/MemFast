"""Symmetric quantization with independent rectangular scale groups."""

import math

import torch
from torch.nn import functional as F


def grouped_quantize(value, bits, granularity):
    rows, columns = value.shape
    gr, gc = (size if group is None else group for size, group in zip(value.shape, granularity))
    nr, nc = math.ceil(rows / gr), math.ceil(columns / gc)
    padded = F.pad(value.float(), (0, nc * gc - columns, 0, nr * gr - rows))
    grouped = padded.reshape(nr, gr, nc, gc)
    maximum = grouped.abs().amax((1, 3), keepdim=True)
    scale = torch.where(maximum > 0, maximum, 1.0) / ((1 << (bits - 1)) - 1)
    code = (grouped / scale).round().clamp_(-((1 << (bits - 1)) - 1), (1 << (bits - 1)) - 1)
    code = code.reshape(nr * gr, nc * gc)[:rows, :columns].to(torch.int32)
    expanded_scale = scale.expand(nr, gr, nc, gc).reshape(nr * gr, nc * gc)
    return code, expanded_scale[:rows, :columns]


def weight_tiles(weight, config):
    # PyTorch weights [O,K] become physical [K,O] before grouping.
    k, o = weight.shape[1], weight.shape[0]
    r, c = config.array_size
    m, p = math.ceil(k / r), math.ceil(o / c)
    code, scale = grouped_quantize(
        weight.T.contiguous(), config.weight_bits, config.weight_quant_gran
    )
    padded = F.pad(code, (0, p * c - o, 0, m * r - k))
    packed = (padded & ((1 << config.weight_bits) - 1)).to(torch.uint8)
    packed = packed.reshape(m, r, p, c).permute(0, 2, 1, 3).contiguous()
    # A reduction partition has one scale per output column. Column groups are independent.
    scales = F.pad(scale[::r], (0, p * c - o)).reshape(m, p, c).contiguous()
    return packed, scales


def input_tiles(value, config):
    n, k = value.shape
    r = config.array_size[0]
    m = math.ceil(k / r)
    code, scale = grouped_quantize(value, config.activation_bits, config.input_quant_gran)
    padded = F.pad(code, (0, m * r - k))
    packed = (padded & ((1 << config.activation_bits) - 1)).to(torch.uint8)
    return packed.reshape(n, m, r).contiguous(), scale[:, ::r].contiguous()
