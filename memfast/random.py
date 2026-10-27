"""Addressed random samples: scheduling does not change physical events."""

import math

import torch


def mix(value):
    value = value & 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def uniform(address, seed, stream):
    key = mix((int(seed) ^ (int(stream) * 0x9E3779B9)) & 0xFFFFFFFF)
    high = address >> 32
    seed_high = mix(int(seed) >> 32)
    folded = mix(high ^ seed_high)
    # Salt the high word by stream so large addresses do not alias in every field.
    folded = folded ^ torch.where(high != 0, mix(high ^ key), torch.zeros_like(high))
    hashed = mix((address & 0xFFFFFFFF) ^ folded ^ key)
    return ((hashed >> 8).float() + 0.5) / 16777216


def event_seed(seed, event):
    return (seed ^ mix(event & 0xFFFFFFFF)) | (mix(event >> 32) << 32)


def normal(address, seed, stream):
    first = uniform(address, seed, stream)
    second = uniform(address, seed, stream + 1)
    return torch.sqrt(-2 * torch.log(first)) * torch.cos((2 * math.pi) * second)
