import copy
from dataclasses import replace

import pytest
import torch

from memfast import Config, Nonidealities, Simulator, reference
from memfast.quantization import grouped_quantize, input_tiles
from memfast.simulator import default_schedule

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def oracle(value, bits, granularity, bw_e=None):
    """Direct block dequantization, independent of physical tiling and slicing."""
    value = value.double()
    result = torch.empty_like(value)
    nr, nk = [size if group is None else group for size, group in zip(value.shape, granularity)]
    upper = 2 ** (bits - 1) - 1
    for row in range(0, value.shape[0], nr):
        for column in range(0, value.shape[1], nk):
            block = value[row : row + nr, column : column + nk]
            maximum = float(block.abs().max())
            if bw_e is None:
                scale, lower = (maximum or 1) / upper, -upper
            else:
                import math

                limit = 2 ** (bw_e - 1) - 1
                exponent = math.floor(math.log2(maximum)) if maximum else 1 - limit
                exponent = max(1 - limit, min(limit, exponent))
                scale, lower = 2.0 ** (exponent + 2 - bits), -(2 ** (bits - 1))
            result[row : row + nr, column : column + nk] = (block / scale).round().clamp(
                lower, upper
            ) * scale
    return result


def test_four_partials_restore_distinct_scales():
    config = Config(
        array_size=(128, 128),
        rows_per_read=32,
        weight_quant_gran=(32, 1),
        input_quant_gran=(1, 32),
    )
    x = torch.repeat_interleave(torch.tensor([1.0, 2.0, 4.0, 8.0]), 32)[None]
    weight = torch.repeat_interleave(torch.tensor([1.0, 3.0, 9.0, 27.0]), 32)[None]
    array = Simulator(config, device="cpu").map(weight)
    torch.testing.assert_close(array(x), torch.tensor([[8288.0]]), rtol=1e-6, atol=0)
    assert array.weight_scale.shape == (4, 1, 128)
    assert torch.unique(array.weight_scale[:, 0, 0]).numel() == 4


@pytest.mark.parametrize("granularity", [(None, None), (None, 32), (3, 64), (2, None)])
def test_tensor_channel_and_group_input_scales(granularity):
    config = Config(
        array_size=(128, 8),
        rows_per_read=32,
        input_quant_gran=granularity,
        weight_quant_gran=(32, 3),
    )
    torch.manual_seed(43)
    x, weight = torch.randn(5, 149), torch.randn(11, 149)
    expected = oracle(x, 8, granularity) @ oracle(weight.T, 8, (32, 3))
    output = Simulator(config, device="cpu").map(weight)(x)
    torch.testing.assert_close(output.double(), expected, rtol=2e-5, atol=2e-5)


def test_input_channel_scales_with_single_row_reads():
    config = Config(array_size=(8, 8), rows_per_read=1, input_quant_gran=(None, 1))
    x, weight = torch.randn(4, 13), torch.randn(7, 13)
    expected = oracle(x, 8, (None, 1)) @ oracle(weight.T, 8, (None, 1))
    actual = Simulator(config, device="cpu").map(weight)(x)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("rows,read_rows,k", [(7, 3, 5), (7, 3, 17), (64, 16, 8), (128, 32, 259)])
def test_physical_and_logical_tails(rows, read_rows, k):
    config = Config(array_size=(rows, 8), rows_per_read=read_rows, adc_bits=8)
    x, weight = torch.randn(2, k), torch.randn(11, k)
    expected = oracle(x, 8, (1, None)) @ oracle(weight.T, 8, (None, 1))
    actual = Simulator(config, device="cpu").map(weight)(x)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)


def test_padding_reads_retain_physical_output_noise():
    config = Config(
        array_size=(64, 8), rows_per_read=16, nonidealities=Nonidealities(output_noise=0.1)
    )
    array = Simulator(config, device="cpu").map(torch.ones(8, 8))
    assert (array.weight_scale[:, 0, 0] > 0).all()
    q, scale = input_tiles(torch.ones(2, 8), config)
    assert (scale > 0).all()
    partials = reference.partial(
        q, array.packed, config, default_schedule(config), array.seed, 0, 0, 1
    )
    assert torch.count_nonzero(partials[:, 1:]) > 0


@pytest.mark.parametrize(
    "bits,slices",
    [
        (8, (1, 1, 2, 2, 2)),
        (10, (1, 1, 2, 2, 2, 2)),
        (16, (1, 3, 4, 4, 4)),
        (20, (1, 3, 4, 4, 4, 4)),
        (10, None),
        (16, None),
        (20, None),
    ],
)
@pytest.mark.parametrize("bw_e", [None, 5, 8])
def test_precision_slicing_and_quantization_oracle(bits, slices, bw_e):
    torch.manual_seed(149)
    config = Config(
        activation_bits=bits,
        weight_bits=bits,
        input_slice=slices,
        weight_slice=slices,
        bw_e=bw_e,
        array_size=(8, 8),
        rows_per_read=4,
        adc_bits=12,
        input_quant_gran=(2, 4),
        weight_quant_gran=(4, 1),
    )
    x, weight = torch.randn(3, 13, dtype=torch.float64), torch.randn(11, 13, dtype=torch.float64)
    expected = oracle(x, bits, (2, 4), bw_e) @ oracle(weight.T, bits, (4, 1), bw_e)
    output = Simulator(config, device="cpu").map(weight)(x)
    tolerance = 2e-5 if not config.wide else 2e-11
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "change",
    [
        {"rows_per_read": 0},
        {"rows_per_read": 65},
        {"rows_per_read": True},
        {"rows_per_read": 24, "weight_quant_gran": (48, 1)},
        {"rows_per_read": 32, "input_quant_gran": (2, 48)},
        {"input_slice": (2, 2, 2, 2)},
        {"weight_slice": (1, 2, 2)},
        {"activation_bits": 32, "weight_bits": 32},
        {"bw_e": 0},
    ],
)
def test_invalid_read_and_slice_layouts(change):
    with pytest.raises(ValueError):
        replace(Config(), **change)


def test_nonfinite_inputs_and_output_dtype_overflow_do_not_consume_events():
    array = Simulator(device="cpu").map(torch.full((4, 64), 1000.0))
    with pytest.raises(ValueError):
        array(torch.full((1, 64), float("inf")))
    with pytest.raises(FloatingPointError):
        array(torch.full((1, 64), 1000.0, dtype=torch.float16))
    assert array._event == 0


def test_wide_checkpoint_and_physical_identity():
    config = Config(activation_bits=16, weight_bits=16, bw_e=8, array_size=(8, 8), rows_per_read=2)
    weight, x = torch.randn(7, 13), torch.randn(2, 13)
    array = Simulator(config, device="cpu").map(weight, name="same")
    state = copy.deepcopy(array.state_dict())
    clone = Simulator(config, device="cpu").map(weight, name="same")
    clone.load_state_dict(state)
    assert torch.equal(array(x), clone(x))
    other = replace(config, rows_per_read=4)
    packed = Simulator(other, device="cpu").map(weight, name="same").packed
    first = reference.restored(array.packed, config, array.seed, 0, 0, 1)[0]
    second = reference.restored(packed, other, array.seed, 0, 0, 1)[0]
    assert torch.equal(first, second)


def test_read_event_high_word_and_large_addresses():
    from memfast.random import event_seed, normal

    addresses = torch.arange(128)
    events = (0, 1, 2**32, 2**32 + 1)
    fields = [normal(addresses, event_seed(135, event), 20) for event in events]
    assert all(not torch.equal(fields[i], fields[j]) for i in range(4) for j in range(i))
    streams = range(10, 20)
    first = [float(normal(torch.tensor(0), 135, stream)) for stream in streams]
    other = [float(normal(torch.tensor(6048813248), 135, stream)) for stream in streams]
    assert first != other


def test_legacy_default_checkpoint_metadata():
    array = Simulator(device="cpu").map(torch.ones(4, 64))
    state = copy.deepcopy(array.state_dict())
    for name in ("rows_per_read", "input_slice", "weight_slice", "bw_e"):
        del state["_extra_state"]["config"][name]
    array.load_state_dict(state)
    assert torch.isfinite(array(torch.ones(2, 64))).all()


@pytest.mark.parametrize(
    "change",
    [
        {"high_conductance": 1e100},
        {"low_conductance": 1e-100},
        {"read_voltage": 1e-100},
        {"high_conductance": 1e-7 + 1e-20, "low_conductance": 1e-7},
    ],
)
def test_unrepresentable_current_parameters(change):
    with pytest.raises(ValueError):
        replace(Config(), **change)


def test_large_groups_do_not_require_large_padding():
    value = torch.randn(3, 7)
    code, scale = grouped_quantize(value, 8, (65536, 65536))
    expected_code, expected_scale = grouped_quantize(value, 8, (None, None))
    assert torch.equal(code, expected_code) and torch.equal(scale, expected_scale)


@pytest.mark.parametrize("bw_e,bits", [(None, 8), (8, 8), (None, 20)])
def test_model_dtype_casts_preserve_programmed_values(bw_e, bits):
    config = Config(activation_bits=bits, weight_bits=bits, bw_e=bw_e, array_size=(8, 8))
    matrix = Simulator(config, device="cpu").map(torch.randn(11, 13), bias=torch.randn(11))
    x = torch.randn(2, 13)
    expected = matrix(x)
    buffers = {name: value.clone() for name, value in matrix.named_buffers()}
    for dtype in (torch.float16, torch.float64, torch.float32):
        matrix.to(dtype=dtype)
        for name, value in matrix.named_buffers():
            assert value.dtype == buffers[name].dtype and torch.equal(value, buffers[name])
        matrix.reset_read()
        assert torch.equal(expected, matrix(x))


@CUDA
def test_channels_last_storage_and_high_read_event():
    config = Config(
        array_size=(8, 8),
        rows_per_read=2,
        nonidealities=Nonidealities(read_variation=0.05, read_mode="slice_pair"),
    )
    weight, x = torch.randn(19, 13), torch.randn(2, 13, device="cuda:0")
    array = Simulator(config, device="cuda:0").map(weight)
    array.reset_read(2**32 + 1)
    expected = array(x)
    array.to(memory_format=torch.channels_last)
    assert not array.packed.is_contiguous()
    array.reset_read(2**32 + 1)
    assert torch.equal(expected, array(x))


@CUDA
@pytest.mark.parametrize("rows,read_rows,k", [(7, 3, 17), (128, 32, 259), (64, 1, 9)])
def test_cuda_physical_tails_and_per_read_scales(rows, read_rows, k):
    config = Config(
        array_size=(rows, 128),
        rows_per_read=read_rows,
        adc_bits=8,
        weight_quant_gran=(rows if rows % read_rows else read_rows, 1),
        input_quant_gran=(None, rows if rows % read_rows else read_rows),
    )
    weight, x = torch.randn(131, k), torch.randn(3, k)
    gpu = Simulator(config, device="cuda:0").map(weight, name="tail")
    cpu = Simulator(config, device="cpu").map(weight, name="tail")
    torch.testing.assert_close(gpu(x.cuda()).cpu(), cpu(x), rtol=2e-5, atol=2e-5)


@CUDA
@pytest.mark.parametrize(
    "bits,slices,bw_e",
    [
        (8, None, None),
        (8, (1, 1, 2, 2, 2), None),
        (10, None, 5),
        (16, (1, 3, 4, 4, 4), 8),
        (20, None, None),
        (20, (1, 3, 4, 4, 4, 4), 8),
    ],
)
@pytest.mark.parametrize("mode", ["shared", "slice_pair", "output"])
def test_cuda_partial_dynamic_and_wide(bits, slices, bw_e, mode):
    config = Config(
        activation_bits=bits,
        weight_bits=bits,
        input_slice=slices,
        weight_slice=slices,
        bw_e=bw_e,
        array_size=(8, 8),
        rows_per_read=2,
        adc_bits=12,
        input_quant_gran=(2, 4),
        weight_quant_gran=(2, 3),
        nonidealities=Nonidealities(
            read_variation=0.02,
            read_mode=mode,
            write_variation=0.02,
            adc_nonlinearity=0.02,
            stuck_at_low=0.01,
            drift_coefficient=0.01,
            drift_time=1,
        ),
    )
    x, weight = torch.randn(3, 13), torch.randn(19, 13)
    cpu = Simulator(config, device="cpu").map(weight, name="wide")
    gpu = Simulator(config, device="cuda:0").map(weight, name="wide")
    actual = gpu(x.cuda())
    torch.testing.assert_close(actual.cpu(), cpu(x), rtol=2e-5, atol=2e-5)
    q, scales = input_tiles(x.cuda(), config)
    for schedule in (
        dict(arrays=4, vectors=1, columns=2, workers=1),
        dict(arrays=2, vectors=2, columns=4, workers=3),
    ):
        result = gpu._execute(q, scales, schedule, 0).to(x.dtype)
        assert torch.equal(actual, result)
