"""CPU regressions for finite quantization, currents, and scale restoration."""

import math

import pytest
import torch

from memfast import Config, Nonidealities, Simulator
from memfast.quantization import grouped_quantize
from memfast.simulator import restore_scales

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "bits,dtype,exponent",
    [(8, torch.float32, -140), (20, torch.float64, -1040)],
    ids=["int8-fp32-subnormal", "int20-fp64-subnormal"],
)
def test_int_quantization_preserves_representable_subnormal_scales(bits, dtype, exponent):
    upper = (1 << (bits - 1)) - 1
    step = math.ldexp(1.0, exponent)
    expected_codes = torch.tensor([[-upper, -1, 0, 1, upper], [0, 0, 0, 0, 0]], dtype=torch.int64)
    value = expected_codes.to(dtype) * step

    codes, scales = grouped_quantize(value, bits, (1, None))

    assert 0 < step < torch.finfo(dtype).tiny
    assert torch.equal(codes, expected_codes)
    torch.testing.assert_close(scales[0], torch.full((5,), step, dtype=dtype), rtol=0, atol=0)
    assert torch.isfinite(scales).all() and (scales > 0).all()
    torch.testing.assert_close(codes * scales, value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bits,dtype,exponent,large",
    [(8, torch.float32, -140, 2.0**100), (20, torch.float64, -1040, 2.0**1000)],
    ids=["int8", "int20"],
)
@pytest.mark.parametrize("tiny_operand", ["input", "weight"])
def test_subnormal_int_groups_restore_finite_nonzero_dot(
    bits, dtype, exponent, large, tiny_operand
):
    upper = (1 << (bits - 1)) - 1
    tiny = upper * math.ldexp(1.0, exponent)
    input_value, weight_value = (tiny, large) if tiny_operand == "input" else (large, tiny)
    x = torch.full((1, 8), input_value, dtype=dtype)
    weight = torch.full((1, 8), weight_value, dtype=dtype)
    config = Config(activation_bits=bits, weight_bits=bits, array_size=(8, 4), adc_bits=8)

    output = Simulator(config, device="cpu").map(weight)(x)
    expected = (x.double() @ weight.double().T).to(dtype)

    assert torch.isfinite(output).all() and (output != 0).all()
    torch.testing.assert_close(output, expected, rtol=2e-6 if bits == 8 else 2e-11, atol=0)


@pytest.mark.parametrize("adc_bits", [16, (1, 16)], ids=["uniform-adc", "msb-first-adc"])
@pytest.mark.parametrize("bw_e", [None, 8], ids=["int", "bfp"])
def test_seventeen_bit_slices_preserve_ideal_counts_and_cancellation(adc_bits, bw_e):
    config = Config(
        activation_bits=17,
        weight_bits=17,
        input_slice=(1, 16),
        weight_slice=(1, 16),
        array_size=(3, 4),
        rows_per_read=1,
        adc_bits=adc_bits,
        bw_e=bw_e,
    )
    # The inactive third row anchors both quantization steps at exactly one.
    weight = torch.tensor([[1.0, 1.0, 65535.0]], dtype=torch.float64)
    x = torch.tensor(
        [[65535.0, 0.0, 0.0], [65535.0, -65535.0, 0.0], [-65535.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    output = Simulator(config, device="cpu").map(weight)(x)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, x @ weight.T, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bits,dtype,input_value,weight_value,rtol",
    [
        pytest.param(8, torch.float32, 1e38, 1e-30, 2e-6, id="fp32-intermediate-overflow"),
        pytest.param(20, torch.float64, 1e308, 1e-100, 2e-11, id="fp64-intermediate-overflow"),
        pytest.param(8, torch.float32, 1e-20, 1e-20, 2e-5, id="fp32-small-scale-product"),
        pytest.param(20, torch.float64, 1e-160, 1e-160, 1e-3, id="fp64-scale-product-underflow"),
    ],
)
def test_scale_restoration_keeps_representable_results(
    bits, dtype, input_value, weight_value, rtol
):
    config = Config(activation_bits=bits, weight_bits=bits, array_size=(8, 4), adc_bits=8)
    x = torch.full((1, 8), input_value, dtype=dtype)
    weight = torch.full((1, 8), weight_value, dtype=dtype)
    expected = (x.double() @ weight.double().T).to(dtype)

    output = Simulator(config, device="cpu").map(weight)(x)

    assert torch.isfinite(expected).all() and (expected != 0).all()
    assert torch.isfinite(output).all() and (output != 0).all()
    torch.testing.assert_close(output, expected, rtol=rtol, atol=0)


@pytest.mark.parametrize(
    "value,input_scale,weight_scale,expected",
    [
        pytest.param(8.0, 2.0**1023, 0.125, 2.0**1023, id="exponent-above-1023"),
        pytest.param(
            1.0,
            torch.finfo(torch.float64).max,
            1.0,
            torch.finfo(torch.float64).max,
            id="largest-finite-result",
        ),
        pytest.param(
            4.0,
            math.ldexp(1.0, -1074),
            0.25,
            math.ldexp(1.0, -1074),
            id="smallest-subnormal-result",
        ),
        pytest.param(0.0, 2.0**1023, 2.0**1023, 0.0, id="zero-with-extreme-scales"),
    ],
)
def test_scale_restoration_exponent_boundaries(value, input_scale, weight_scale, expected):
    output = restore_scales(
        torch.tensor(value, dtype=torch.float64),
        torch.tensor(input_scale, dtype=torch.float64),
        torch.tensor(weight_scale, dtype=torch.float64),
    )

    assert torch.isfinite(output)
    torch.testing.assert_close(output, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize("difference", [1e-14, 2e-14])
def test_near_fp32_ulp_conductances_preserve_ideal_dot(difference):
    config = Config(
        array_size=(8, 4),
        adc_bits=8,
        high_conductance=1e-7 + difference,
        low_conductance=1e-7,
    )
    low = torch.tensor(config.low_conductance, dtype=torch.float32)
    high = torch.tensor(config.high_conductance, dtype=torch.float32)
    assert high > low
    x, weight = torch.ones(1, 8), torch.ones(1, 8)

    output = Simulator(config, device="cpu").map(weight)(x)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, x @ weight.T, rtol=1e-6, atol=0)


@pytest.mark.parametrize("read_voltage", [2 * torch.finfo(torch.float32).max, 1e40])
def test_read_voltage_above_fp32_range_is_rejected_even_with_finite_current_unit(read_voltage):
    # Small conductances keep the derived current unit inside the FP32 range.
    unit = (1e-20 - 1e-22) * read_voltage
    assert math.isfinite(unit) and 0 < unit < torch.finfo(torch.float32).max
    with pytest.raises(ValueError):
        Config(high_conductance=1e-20, low_conductance=1e-22, read_voltage=read_voltage)


def test_largest_finite_fp32_read_voltage_preserves_ideal_dot():
    config = Config(
        array_size=(8, 4),
        adc_bits=8,
        high_conductance=1e-20,
        low_conductance=1e-22,
        read_voltage=torch.finfo(torch.float32).max,
    )
    x, weight = torch.ones(1, 8), torch.ones(1, 8)

    output = Simulator(config, device="cpu").map(weight)(x)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, x @ weight.T, rtol=1e-6, atol=0)


@pytest.mark.parametrize("mode", ["shared", "slice_pair", "output"])
def test_read_noise_preserves_normalized_counts_at_large_finite_voltage(mode):
    noise = Nonidealities(read_mode=mode, read_variation=0.02)
    baseline = Config(array_size=(8, 4), adc_bits=8, nonidealities=noise)
    extreme = Config(
        array_size=(8, 4),
        adc_bits=8,
        high_conductance=1e-20,
        low_conductance=1e-22,
        read_voltage=torch.finfo(torch.float32).max,
        nonidealities=noise,
    )
    x, weight = torch.ones(1, 8), torch.ones(1, 8)
    expected = Simulator(baseline, device="cpu").map(weight, name="edge")(x)

    output = Simulator(extreme, device="cpu").map(weight, name="edge")(x)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=0)


def _assert_cuda_matches_cpu(config, x, weight, *, rtol=0):
    expected = Simulator(config, device="cpu").map(weight, name="edge")(x)
    output = Simulator(config, device="cuda:0").map(weight, name="edge")(x.cuda()).cpu()
    assert torch.isfinite(expected).all() and torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=rtol, atol=0)


@CUDA
@pytest.mark.parametrize("bw_e", [None, 8], ids=["int", "bfp"])
def test_cuda_seventeen_bit_slices_match_cpu(bw_e):
    config = Config(
        activation_bits=17,
        weight_bits=17,
        input_slice=(1, 16),
        weight_slice=(1, 16),
        array_size=(3, 4),
        rows_per_read=1,
        adc_bits=(1, 16),
        bw_e=bw_e,
    )
    weight = torch.tensor([[1.0, 1.0, 65535.0]], dtype=torch.float64)
    x = torch.tensor(
        [[65535.0, 0.0, 0.0], [65535.0, -65535.0, 0.0], [-65535.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    _assert_cuda_matches_cpu(config, x, weight)


@CUDA
@pytest.mark.parametrize("tiny_operand", ["input", "weight"])
def test_cuda_twenty_bit_subnormal_scales_match_cpu(tiny_operand):
    config = Config(activation_bits=20, weight_bits=20, array_size=(8, 4), adc_bits=8)
    tiny = ((1 << 19) - 1) * math.ldexp(1.0, -1040)
    large = 2.0**1000
    input_value, weight_value = (tiny, large) if tiny_operand == "input" else (large, tiny)
    x = torch.full((1, 8), input_value, dtype=torch.float64)
    weight = torch.full((1, 8), weight_value, dtype=torch.float64)
    _assert_cuda_matches_cpu(config, x, weight, rtol=2e-11)


@CUDA
@pytest.mark.parametrize("tiny_operand", ["input", "weight"])
def test_cuda_eight_bit_subnormal_scales_match_cpu(tiny_operand):
    config = Config(array_size=(8, 4), adc_bits=8)
    tiny, large = 127 * math.ldexp(1.0, -140), 2.0**100
    input_value, weight_value = (tiny, large) if tiny_operand == "input" else (large, tiny)
    x = torch.full((1, 8), input_value)
    weight = torch.full((1, 8), weight_value)
    _assert_cuda_matches_cpu(config, x, weight, rtol=2e-6)


@CUDA
@pytest.mark.parametrize(
    "bits,dtype,input_value,weight_value,rtol",
    [
        pytest.param(8, torch.float32, 1e38, 1e-30, 2e-6, id="default-int8"),
        pytest.param(20, torch.float64, 1e308, 1e-100, 2e-11, id="int20"),
    ],
)
def test_cuda_scale_restoration_overflow_matches_cpu(bits, dtype, input_value, weight_value, rtol):
    config = Config(activation_bits=bits, weight_bits=bits, array_size=(8, 4), adc_bits=8)
    x = torch.full((1, 8), input_value, dtype=dtype)
    weight = torch.full((1, 8), weight_value, dtype=dtype)
    _assert_cuda_matches_cpu(config, x, weight, rtol=rtol)


@CUDA
@pytest.mark.parametrize("difference", [1e-14, 2e-14])
def test_cuda_near_fp32_ulp_conductances_match_cpu(difference):
    config = Config(
        array_size=(8, 4),
        adc_bits=8,
        high_conductance=1e-7 + difference,
        low_conductance=1e-7,
    )
    _assert_cuda_matches_cpu(config, torch.ones(1, 8), torch.ones(1, 8), rtol=1e-6)


@CUDA
@pytest.mark.parametrize("mode", ["shared", "slice_pair", "output"])
def test_cuda_largest_finite_voltage_read_noise_matches_cpu(mode):
    config = Config(
        array_size=(8, 4),
        adc_bits=8,
        high_conductance=1e-20,
        low_conductance=1e-22,
        read_voltage=torch.finfo(torch.float32).max,
        nonidealities=Nonidealities(read_mode=mode, read_variation=0.02),
    )
    _assert_cuda_matches_cpu(config, torch.ones(1, 8), torch.ones(1, 8), rtol=1e-6)
