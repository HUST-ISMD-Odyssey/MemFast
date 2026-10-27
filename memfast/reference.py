"""Readable PyTorch implementation of the array model."""

import math

import torch

from .random import event_seed, normal, uniform


def restored(packed, config, seed, event, begin, end):
    ni = config.nonidealities
    m, p, r, c = packed.shape
    w = len(config.weight_layout)
    array = torch.arange(m, device=packed.device)[:, None, None, None, None]
    tile = torch.arange(begin, end, device=packed.device)[None, :, None, None, None]
    bit = torch.arange(w, device=packed.device)[None, None, :, None, None]
    row = torch.arange(r, device=packed.device)[None, None, None, :, None]
    col = torch.arange(c, device=packed.device)[None, None, None, None, :]
    address = (((array * p + tile) * w + bit) * r + row) * c + col
    shifts = torch.tensor([entry[0] for entry in config.weight_layout], device=packed.device)
    masks = torch.tensor([entry[1] for entry in config.weight_layout], device=packed.device)
    shifts, masks = shifts[None, None, :, None, None], masks[None, None, :, None, None]
    state = (packed[:, begin:end, None].long() >> shifts) & masks
    dtype = torch.float64 if config.precise_currents else torch.float32
    raw = config.low_conductance + (config.high_conductance - config.low_conductance) * (
        state.to(dtype) / masks
    )
    raw = torch.where(
        masks == 1,
        torch.where(
            state.bool(),
            torch.tensor(config.high_conductance, dtype=dtype, device=packed.device),
            torch.tensor(config.low_conductance, dtype=dtype, device=packed.device),
        ),
        raw,
    ).to(dtype)
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
    read_seed = event_seed(seed, event)
    if ni.read_mode == "shared" and ni.read_variation:
        raw = raw * torch.exp(ni.read_variation * normal(address, read_seed, 20))
    return raw, address, read_seed


def adc(current, config, weight_slice=0):
    ni = config.nonidealities
    maximum = config.adc_maxima[weight_slice]
    value = current * (1 / config.current_unit)
    if ni.adc_nonlinearity:
        value = value / (1 + ((ni.adc_nonlinearity / maximum) * value).abs())
    return value.clamp(0, maximum).round()


def partial(q, packed, config, schedule, seed, event, begin, end):
    """Restore ADC counts using signed input and configured weight significance.

    Weight digits are always physical nonnegative conductance states. Their
    signed or unsigned interpretation enters only through the layout factors.
    """
    ni = config.nonidealities
    n, m, r = q.shape
    p, c = packed.shape[1], config.array_size[1]
    a, w = len(config.input_layout), len(config.weight_layout)
    width = end - begin
    raw, cell, read_seed = restored(packed, config, seed, event, begin, end)
    groups = config.reads_per_array
    totals = torch.zeros((n, m, groups, width, c), device=q.device, dtype=torch.int64)
    vector = torch.arange(n, device=q.device)[:, None, None, None]
    array = torch.arange(m, device=q.device)[None, :, None, None]
    tile = torch.arange(begin, end, device=q.device)[None, None, :, None]
    col = torch.arange(c, device=q.device)[None, None, None, :]
    for ai, (ashift, amask, sa) in enumerate(config.input_layout):
        voltage = ((q.long() >> ashift) & amask).to(raw.dtype) / amask * config.read_voltage
        if ni.voltage_variation:
            row = torch.arange(r, device=q.device)[None, None, :]
            address = ((vector[..., 0] * m + array[..., 0]) * a + ai) * r + row
            voltage *= 1 + ni.voltage_variation * normal(address, read_seed, 40)
        v = voltage.permute(1, 0, 2).contiguous()
        for wi, (_, wmask, sw) in enumerate(config.weight_layout):
            g = raw[:, :, wi]
            if ni.read_mode == "slice_pair" and ni.read_variation:
                g = g * torch.exp(
                    ni.read_variation * normal(cell[:, :, wi] * a + ai, read_seed, 20)
                )
            for group in range(groups):
                start = group * config.read_rows
                end = min(start + config.read_rows, r)
                length = end - start
                sub_g, sub_v = g[:, :, start:end], v[:, :, start:end]
                effective = (sub_g - config.low_conductance).permute(0, 2, 1, 3)
                effective = effective.reshape(m, length, width * c)
                current = torch.bmm(sub_v, effective).permute(1, 0, 2)
                current = current.reshape(n, m, width, c)
                read_part = array * groups + group
                address = (
                    (((vector * m * groups + read_part) * p + tile) * a + ai) * w + wi
                ) * c + col
                if ni.read_mode == "output" and ni.read_variation:
                    base_g = sub_g.permute(0, 2, 1, 3).reshape(m, length, width * c)
                    mean = torch.bmm(sub_v, base_g) * math.expm1(0.5 * ni.read_variation**2)
                    variance = torch.bmm(sub_v.square(), base_g.square())
                    variance *= math.exp(ni.read_variation**2) * math.expm1(ni.read_variation**2)
                    mean = mean.permute(1, 0, 2).reshape_as(current)
                    variance = variance.permute(1, 0, 2).reshape_as(current)
                    current += mean + variance.sqrt() * normal(address, read_seed, 60)
                if ni.output_noise:
                    current += (
                        ni.output_noise
                        * config.current_unit
                        * length
                        * normal(address, read_seed, 50)
                    )
                totals[:, :, group] += adc(current * (amask * wmask), config, wi).long() * (sa * sw)
    dtype = torch.float64 if config.wide else torch.float32
    return totals.reshape(n, m * groups, width, c).to(dtype)
