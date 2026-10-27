"""Physical geometry, quantization, and device nonidealities."""

import math
from dataclasses import dataclass, fields


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class Nonidealities:
    """Dimensionless noise magnitudes; retention times are in seconds."""

    write_variation: float = 0.0
    read_variation: float = 0.0
    read_mode: str = "shared"
    voltage_variation: float = 0.0
    output_noise: float = 0.0
    adc_nonlinearity: float = 0.0
    stuck_at_low: float = 0.0
    stuck_at_high: float = 0.0
    drift_coefficient: float = 0.0
    drift_time: float = 0.0
    drift_reference_time: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            if field.name != "read_mode":
                nonnegative(getattr(self, field.name), field.name)
        if self.read_mode not in ("shared", "slice_pair", "output"):
            raise ValueError("read_mode must be shared, slice_pair, or output")
        if self.stuck_at_low + self.stuck_at_high > 1:
            raise ValueError("stuck-at probabilities must sum to at most one")
        if self.drift_reference_time <= 0:
            raise ValueError("drift_reference_time must be positive")


@dataclass(frozen=True)
class Config:
    """Matrices use [reduction rows, output columns] for geometry and scale groups."""

    activation_bits: int = 8
    weight_bits: int = 8
    array_size: tuple[int, int] = (64, 64)
    weight_quant_gran: tuple[int | None, int | None] = (None, 1)
    input_quant_gran: tuple[int | None, int | None] = (1, None)
    adc_bits: int = 6
    high_conductance: float = 1e-5
    low_conductance: float = 1e-7
    read_voltage: float = 0.2
    seed: int = 135
    nonidealities: Nonidealities = Nonidealities()

    def __post_init__(self):
        integer(self.activation_bits, "activation_bits", 2, 8)
        integer(self.weight_bits, "weight_bits", 2, 8)
        integer(self.adc_bits, "adc_bits", 1, 16)
        integer(self.seed, "seed", 0, 2**32 - 1)
        for name in ("array_size", "weight_quant_gran", "input_quant_gran"):
            pair = getattr(self, name)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"{name} must be a pair")
            for value in pair:
                if value is None and name != "array_size":
                    continue
                integer(value, name, 1, 65536)
        rows, columns = self.array_size
        if columns % 4:
            raise ValueError("array columns must be divisible by four")
        if self.input_quant_gran[0] != 1:
            raise ValueError("input_quant_gran starts with 1 (one scale per vector)")
        for value in (self.weight_quant_gran[0], self.input_quant_gran[1]):
            if value is not None and value % rows:
                raise ValueError("reduction scale groups must span whole array-row partitions")
        for name in ("high_conductance", "low_conductance", "read_voltage"):
            value = getattr(self, name)
            nonnegative(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.high_conductance <= self.low_conductance:
            raise ValueError("high_conductance must exceed low_conductance")
        if not isinstance(self.nonidealities, Nonidealities):
            raise TypeError("nonidealities must be Nonidealities")

    @property
    def current_unit(self):
        return (self.high_conductance - self.low_conductance) * self.read_voltage

    @property
    def adc_maximum(self):
        return (1 << self.adc_bits) - 1
