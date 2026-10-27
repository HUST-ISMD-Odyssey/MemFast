from dataclasses import replace

import pytest
import torch

from memfast import Config, Nonidealities, Simulator
from memfast.quantization import grouped_quantize
from memfast.reference import adc


def test_adc_order_follows_public_weight_slice_order():
    config = Config(weight_slice=(1, 1, 2, 2, 2), adc_bits=(6, 7, 8, 9, 10))
    assert config.adc_maxima == (1023, 511, 255, 127, 63)
    assert Config(adc_bits=[6] * 8).adc_bits == (6,) * 8


@pytest.mark.parametrize("bits", [1, 6, 7, 16])
def test_adc_endpoints_and_nearest_even(bits):
    config = Config(adc_bits=bits)
    maximum = 2**bits - 1
    counts = torch.tensor([-2.0, 0.0, 0.5, 1.5, float(maximum), maximum + 20.0])
    actual = adc(counts * config.current_unit, config)
    expected = counts.clamp(0, maximum).round()
    assert torch.equal(actual, expected)


def test_nonlinearity_uses_each_slice_range():
    config = Config(
        weight_bits=2, adc_bits=(6, 7), nonidealities=Nonidealities(adc_nonlinearity=0.5)
    )
    current = torch.tensor([40.0, 60.0, 100.0]) * config.current_unit
    assert not torch.equal(adc(current, config, 0), adc(current, config, 1))
    for index, maximum in enumerate((127, 63)):
        expected = current / config.current_unit
        expected = (expected / (1 + (0.5 / maximum * expected).abs())).clamp(0, maximum).round()
        assert torch.equal(adc(current, config, index), expected)


@pytest.mark.parametrize(
    "value",
    [
        (),
        (6,),
        (6,) * 9,
        (True,) * 8,
        (0,) * 8,
        (17,) * 8,
        (6.0,) * 8,
        True,
        "6",
        None,
    ],
)
def test_invalid_adc_layouts(value):
    with pytest.raises(ValueError):
        Config(adc_bits=value)


@pytest.mark.parametrize("name", ["input_slice", "weight_slice"])
def test_boolean_slice_widths_are_rejected(name):
    with pytest.raises(ValueError):
        Config(**{name: (True,) * 8})


def integer_oracle(x, weight, config):
    """Explicit integer dot products and ADC clipping, without array helpers."""
    xq, xs = grouped_quantize(x, config.activation_bits, config.input_quant_gran, config.bw_e)
    wq, ws = grouped_quantize(weight.T, config.weight_bits, config.weight_quant_gran, config.bw_e)
    output = torch.zeros(x.shape[0], weight.shape[0], dtype=torch.float64)
    for array in range(0, x.shape[1], config.array_size[0]):
        for start in range(array, min(array + config.array_size[0], x.shape[1]), config.read_rows):
            end = min(start + config.read_rows, array + config.array_size[0], x.shape[1])
            partial = torch.zeros_like(output, dtype=torch.int64)
            for ashift, amask, afactor in config.input_layout:
                a = (xq[:, start:end].long() >> ashift) & amask
                for index, (wshift, wmask, wfactor) in enumerate(config.weight_layout):
                    w = (wq[start:end].long() >> wshift) & wmask
                    counts = (a @ w).clamp(0, config.adc_maxima[index])
                    partial += counts * afactor * wfactor
            output += partial * xs[:, start, None] * ws[start, None, :]
    return output.to(x.dtype)


@pytest.mark.parametrize("bw_e", [None, 5])
@pytest.mark.parametrize(
    "slices,adcs",
    [
        ((1,) * 8, (6, 7, 1, 8, 9, 10, 12, 16)),
        ((1, 1, 2, 2, 2), (6, 7, 8, 9, 10)),
        ((1, 3, 4, 4, 4), (6, 7, 8, 9, 10)),
        ((1, 3, 4, 4, 4, 4), (6, 7, 8, 9, 10, 12)),
    ],
)
def test_heterogeneous_adc_against_integer_oracle(slices, adcs, bw_e):
    torch.manual_seed(183)
    config = Config(
        activation_bits=8,
        weight_bits=sum(slices),
        weight_slice=slices,
        input_slice=(1, 1, 2, 2, 2),
        adc_bits=adcs,
        array_size=(8, 8),
        rows_per_read=2,
        bw_e=bw_e,
        input_quant_gran=(2, 2),
        weight_quant_gran=(2, 3),
    )
    x, weight = torch.randn(3, 13), torch.randn(11, 13)
    actual = Simulator(config, device="cpu").map(weight)(x)
    torch.testing.assert_close(actual, integer_oracle(x, weight, config), rtol=2e-5, atol=2e-5)


def test_uniform_sequence_preserves_scalar_adc_outputs_and_checkpoint():
    torch.manual_seed(91)
    config = Config(array_size=(8, 8), rows_per_read=2)
    x, weight = torch.randn(3, 13), torch.randn(11, 13)
    scalar = Simulator(config, device="cpu").map(weight, name="same")
    sequence = Simulator(replace(config, adc_bits=(6,) * 8), device="cpu").map(weight, name="same")
    assert torch.equal(scalar(x), sequence(x))
    with pytest.raises(ValueError, match="configuration"):
        sequence.load_state_dict(scalar.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["shared", "slice_pair", "output"])
def test_cuda_heterogeneous_adc_all_read_modes(mode):
    config = Config(
        weight_bits=10,
        weight_slice=(1, 1, 2, 2, 2, 2),
        adc_bits=(6, 7, 8, 9, 10, 12),
        array_size=(8, 8),
        rows_per_read=2,
        nonidealities=Nonidealities(
            read_mode=mode, read_variation=0.02, adc_nonlinearity=0.02, output_noise=0.01
        ),
    )
    torch.manual_seed(188)
    x, weight = torch.randn(3, 13), torch.randn(11, 13)
    cpu = Simulator(config, device="cpu").map(weight, name="same")
    gpu = Simulator(config, device="cuda:0").map(weight, name="same")
    torch.testing.assert_close(gpu(x.cuda()).cpu(), cpu(x), rtol=2e-5, atol=2e-5)
