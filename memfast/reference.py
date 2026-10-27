"""Readable PyTorch implementation of the array model."""

import math

import torch

from .random import mix, normal, uniform


def restored(packed, config, seed, event, begin, end):
    ni = config.nonidealities
    m, p, r, c = packed.shape
    w = config.weight_bits
    array = torch.arange(m, device=packed.device)[:, None, None, None, None]
    tile = torch.arange(begin, end, device=packed.device)[None, :, None, None, None]
    bit = torch.arange(w, device=packed.device)[None, None, :, None, None]
    row = torch.arange(r, device=packed.device)[None, None, None, :, None]
    col = torch.arange(c, device=packed.device)[None, None, None, None, :]
    address = (((array * p + tile) * w + bit) * r + row) * c + col
    state = (packed[:, begin:end, None].long() >> bit) & 1
    raw = torch.where(state.bool(), config.high_conductance, config.low_conductance).float()
    if ni.write_variation:
        raw = raw * torch.exp(ni.write_variation * normal(address, seed, 10))
    raw = raw * (1 + ni.drift_time / ni.drift_reference_time) ** (-ni.drift_coefficient)
    if ni.stuck_at_low + ni.stuck_at_high:
        sample = uniform(address, seed, 30)
        raw = torch.where(sample < ni.stuck_at_low, config.low_conductance, raw)
        raw = torch.where(
            (sample >= ni.stuck_at_low) & (sample < ni.stuck_at_low + ni.stuck_at_high),
            config.high_conductance,
            raw,
        )
    read_seed = seed ^ mix(event & 0xFFFFFFFF) ^ mix(event >> 32)
    if ni.read_mode == "shared" and ni.read_variation:
        raw = raw * torch.exp(ni.read_variation * normal(address, read_seed, 20))
    return raw, address, read_seed


def adc(current, config):
    ni = config.nonidealities
    value = current * (1 / config.current_unit)
    if ni.adc_nonlinearity:
        value = value / (1 + ((ni.adc_nonlinearity / config.adc_maximum) * value).abs())
    return value.clamp(0, config.adc_maximum).round()


def partial(q, packed, config, schedule, seed, event, begin, end):
    ni = config.nonidealities
    n, m, r = q.shape
    p, c, a, w = packed.shape[1], config.array_size[1], config.activation_bits, config.weight_bits
    width = end - begin
    raw, cell, read_seed = restored(packed, config, seed, event, begin, end)
    totals = torch.zeros((n, m, width, c), device=q.device, dtype=torch.int64)
    vector = torch.arange(n, device=q.device)[:, None, None, None]
    array = torch.arange(m, device=q.device)[None, :, None, None]
    tile = torch.arange(begin, end, device=q.device)[None, None, :, None]
    col = torch.arange(c, device=q.device)[None, None, None, :]
    for ai in range(a):
        voltage = ((q.long() >> ai) & 1).float() * config.read_voltage
        if ni.voltage_variation:
            row = torch.arange(r, device=q.device)[None, None, :]
            address = ((vector[..., 0] * m + array[..., 0]) * a + ai) * r + row
            voltage *= 1 + ni.voltage_variation * normal(address, read_seed, 40)
        v = voltage.permute(1, 0, 2).contiguous()
        for wi in range(w):
            g = raw[:, :, wi]
            if ni.read_mode == "slice_pair" and ni.read_variation:
                g = g * torch.exp(
                    ni.read_variation * normal(cell[:, :, wi] * a + ai, read_seed, 20)
                )
            effective = (g - config.low_conductance).permute(0, 2, 1, 3).reshape(m, r, width * c)
            current = torch.bmm(v, effective).permute(1, 0, 2).reshape(n, m, width, c)
            address = ((((vector * m + array) * p + tile) * a + ai) * w + wi) * c + col
            if ni.read_mode == "output" and ni.read_variation:
                base_g = g.permute(0, 2, 1, 3).reshape(m, r, width * c)
                mean = torch.bmm(v, base_g) * math.expm1(0.5 * ni.read_variation**2)
                variance = torch.bmm(v.square(), base_g.square())
                variance *= math.exp(ni.read_variation**2) * math.expm1(ni.read_variation**2)
                mean = mean.permute(1, 0, 2).reshape_as(current)
                variance = variance.permute(1, 0, 2).reshape_as(current)
                current += mean + variance.sqrt() * normal(address, read_seed, 60)
            if ni.output_noise:
                current += (
                    ni.output_noise * config.current_unit * r * normal(address, read_seed, 50)
                )
            sa = -(1 << ai) if ai == a - 1 else (1 << ai)
            sw = -(1 << wi) if wi == w - 1 else (1 << wi)
            totals += adc(current, config).long() * (sa * sw)
    return totals.float()
