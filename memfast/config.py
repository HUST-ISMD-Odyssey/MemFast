"""Physical geometry, quantization, and device nonidealities."""

import math
import operator
import struct
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
    adc_bits: int | tuple[int, ...] = 6
    high_conductance: float = 1e-5
    low_conductance: float = 1e-7
    read_voltage: float = 0.2
    seed: int = 135
    nonidealities: Nonidealities = Nonidealities()
    rows_per_read: int | None = None
    input_slice: tuple[int, ...] | None = None
    weight_slice: tuple[int, ...] | None = None
    bw_e: int | None = None

    def __post_init__(self):
        integer(self.activation_bits, "activation_bits", 2, 32)
        integer(self.weight_bits, "weight_bits", 2, 32)
        if self.bw_e is not None:
            integer(self.bw_e, "bw_e", 2, 11)
        for name, bits in (
            ("input_slice", self.activation_bits),
            ("weight_slice", self.weight_bits),
        ):
            value = getattr(self, name)
            if value is not None:
                try:
                    original = tuple(value)
                    if any(isinstance(item, bool) for item in original):
                        raise ValueError
                    widths = tuple(operator.index(item) for item in original)
                except (TypeError, ValueError):
                    raise ValueError(f"{name} must contain integer slice widths") from None
                if not widths or widths[0] != 1 or sum(widths) != bits:
                    raise ValueError(
                        f"{name} must start with a one-bit sign slice and sum to {bits}"
                    )
                if any(width < 1 or width > 16 for width in widths):
                    raise ValueError(f"{name} slice widths must be in [1,16]")
                object.__setattr__(self, name, widths)
        if isinstance(self.adc_bits, (tuple, list)):
            widths = tuple(self.adc_bits)
            if len(widths) != len(self.weight_layout):
                raise ValueError("adc_bits must have one entry per weight slice, MSB first")
            for value in widths:
                integer(value, "adc_bits", 1, 16)
            object.__setattr__(self, "adc_bits", widths)
        else:
            integer(self.adc_bits, "adc_bits", 1, 16)
        bound = sum(abs(entry[2]) for entry in self.input_layout)
        bound *= sum(
            abs(entry[2]) * maximum for entry, maximum in zip(self.weight_layout, self.adc_maxima)
        )
        if bound > 2**63 - 1:
            raise ValueError(
                "slice reconstruction exceeds the signed 64-bit accumulator; "
                "reduce precision, ADC bits, or use wider multibit slices"
            )
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
        if self.rows_per_read is not None:
            integer(self.rows_per_read, "rows_per_read", 1, rows)
        if columns % 4:
            raise ValueError("array columns must be divisible by four")
        for value in (self.weight_quant_gran[0], self.input_quant_gran[1]):
            if value is not None and value % rows != 0:
                if rows % self.read_rows != 0 or value % self.read_rows != 0:
                    raise ValueError(
                        "reduction scale groups must align with every ADC read boundary; "
                        "use a multiple of array rows, or a multiple of rows_per_read "
                        "when it divides array rows"
                    )
        for name in ("high_conductance", "low_conductance", "read_voltage"):
            value = getattr(self, name)
            nonnegative(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.high_conductance <= self.low_conductance:
            raise ValueError("high_conductance must exceed low_conductance")
        try:
            high = struct.unpack("f", struct.pack("f", self.high_conductance))[0]
            low = struct.unpack("f", struct.pack("f", self.low_conductance))[0]
        except OverflowError:
            raise ValueError("conductances must be representable in FP32") from None
        maximum = 3.4028234663852886e38
        if not (0 < low < high <= maximum):
            raise ValueError("conductances must remain distinct and positive in FP32")
        if not 1.401298464324817e-45 <= self.read_voltage <= maximum:
            raise ValueError("read_voltage must be representable in FP32")
        if not 1 / maximum <= self.current_unit <= maximum:
            raise ValueError("current unit and reciprocal must be representable in FP32")
        if not isinstance(self.nonidealities, Nonidealities):
            raise TypeError("nonidealities must be Nonidealities")

    @property
    def current_unit(self):
        return (self.high_conductance - self.low_conductance) * self.read_voltage

    @property
    def adc_maximum(self):
        return max(self.adc_maxima)

    @property
    def adc_maxima(self):
        """ADC count limits in the internal least-significant-slice-first order."""
        widths = (
            (self.adc_bits,) * len(self.weight_layout)
            if type(self.adc_bits) is int
            else reversed(self.adc_bits)
        )
        return tuple((1 << bits) - 1 for bits in widths)

    @property
    def read_rows(self):
        return self.array_size[0] if self.rows_per_read is None else self.rows_per_read

    @property
    def reads_per_array(self):
        return math.ceil(self.array_size[0] / self.read_rows)

    @staticmethod
    def _layout(widths, bits):
        widths = (1,) * bits if widths is None else widths
        result, shift = [], 0
        for index, width in enumerate(reversed(widths)):
            factor = -(1 << shift) if index == len(widths) - 1 else 1 << shift
            result.append((shift, (1 << width) - 1, factor))
            shift += width
        return tuple(result)

    @property
    def input_layout(self):
        return self._layout(self.input_slice, self.activation_bits)

    @property
    def weight_layout(self):
        return self._layout(self.weight_slice, self.weight_bits)

    @property
    def wide(self):
        return max(self.activation_bits, self.weight_bits) > 8 or self.bw_e is not None

    @property
    def precise_currents(self):
        """Use FP64 where physical FP32 rounding can change ideal ADC counts."""
        return (
            max(entry[1] for entry in (*self.input_layout, *self.weight_layout)) > 255
            or self.high_conductance - self.low_conductance < self.high_conductance * 1e-4
            or self.high_conductance * self.read_voltage * self.read_rows > 8e37
            or min(self.low_conductance, self.read_voltage) < 1e-18
            or max(self.high_conductance, self.read_voltage) > 1e18
        )
