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
    hashed = mix((address & 0xFFFFFFFF) ^ mix(address >> 32) ^ key)
    return ((hashed >> 8).float() + 0.5) / 16777216


def normal(address, seed, stream):
    first = uniform(address, seed, stream)
    second = uniform(address, seed, stream + 1)
    return torch.sqrt(-2 * torch.log(first)) * torch.cos((2 * math.pi) * second)
