import copy
import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from memfast import Config, Nonidealities, Simulator, convert_model, reference
from memfast.quantization import grouped_quantize, input_tiles
from memfast.simulator import default_schedule, schedule_grid


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(135)
    torch.set_num_threads(2)


@pytest.mark.parametrize("granularity", [(64, 1), (128, 16), (None, 1), (64, 128)])
def test_quantization_groups_and_array_geometry(granularity):
    config = Config(weight_quant_gran=granularity, adc_bits=8)
    weight, x = torch.randn(79, 150), torch.randn(3, 150)
    array = Simulator(config, device="cpu").map(weight)
    output = array(x)
    qw, sw = grouped_quantize(weight.T, config.weight_bits, granularity)
    qa, sa = grouped_quantize(x, config.activation_bits, config.input_quant_gran)
    expected = (qa * sa) @ (qw * sw)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_signed_integer_oracle(bits):
    config = Config(activation_bits=bits, weight_bits=bits, array_size=(8, 8), adc_bits=6)
    weight = torch.randint(-3, 4, (9, 17)).float()
    x = torch.randint(-3, 4, (2, 17)).float()
    array = Simulator(config, device="cpu").map(weight)
    qw, sw = grouped_quantize(weight.T, bits, config.weight_quant_gran)
    qa, sa = grouped_quantize(x, bits, config.input_quant_gran)
    torch.testing.assert_close(array(x), (qa * sa) @ (qw * sw), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "ni",
    [
        Nonidealities(write_variation=0.05),
        Nonidealities(read_variation=0.05),
        Nonidealities(read_variation=0.05, read_mode="slice_pair"),
        Nonidealities(read_variation=0.05, read_mode="output"),
        Nonidealities(voltage_variation=0.05),
        Nonidealities(output_noise=0.01),
        Nonidealities(adc_nonlinearity=0.05),
        Nonidealities(stuck_at_low=0.02, stuck_at_high=0.02),
        Nonidealities(drift_coefficient=0.05, drift_time=100),
    ],
)
def test_noise_replay_and_schedule_independence(ni):
    config = Config(activation_bits=4, weight_bits=4, array_size=(8, 8), nonidealities=ni)
    sim = Simulator(config, device="cpu")
    array = sim.map(torch.randn(19, 17))
    x = torch.randn(3, 17)
    q, scales = input_tiles(x, config)
    first = array._execute(q, scales, default_schedule(config), 0)
    other = array._execute(q, scales, dict(arrays=4, vectors=1, columns=2, workers=1), 0)
    assert torch.equal(first, other)
    assert torch.equal(first, array(x))
    array.reset_read()
    assert torch.equal(first, array(x))


@pytest.mark.parametrize("mode", ["shared", "slice_pair", "output"])
def test_read_noise_batch_granularity(mode):
    config = Config(
        activation_bits=4,
        weight_bits=4,
        array_size=(8, 8),
        nonidealities=Nonidealities(read_variation=0.2, read_mode=mode),
    )
    array = Simulator(config, device="cpu").map(torch.randn(8, 8))
    x = torch.ones(2, 8)
    first = array(x)
    second = array(x)
    assert not torch.equal(first, second)
    if mode in ("shared", "slice_pair"):
        assert torch.equal(first[0], first[1])
    else:
        assert not torch.equal(first[0], first[1])


def test_slice_pair_fields_are_independent():
    from memfast.random import normal

    address = torch.arange(100000)
    fields = torch.stack([normal(address * 4 + a, 135, 20) for a in range(4)])
    correlation = torch.corrcoef(fields)
    assert (correlation - torch.eye(4)).abs().max() < 0.02
    assert torch.unique(fields[:, :1000]).numel() == 4000


def test_output_read_first_two_moments():
    from memfast.random import normal

    ni = Nonidealities(read_variation=0.2, read_mode="output")
    raw = torch.tensor([1e-5, 2e-6, 1e-7])
    voltage = torch.tensor([0.2, 0.1, 0.2])
    address = torch.arange(80000)[:, None] * 3 + torch.arange(3)[None]
    exact = (voltage * raw * (ni.read_variation * normal(address, 135, 20)).exp()).sum(1)
    mean = (voltage * raw).sum() * math.exp(0.5 * ni.read_variation**2)
    variance = (
        (voltage * raw).square().sum()
        * math.exp(ni.read_variation**2)
        * math.expm1(ni.read_variation**2)
    )
    assert abs(float(exact.mean() / mean) - 1) < 0.003
    assert abs(float(exact.var() / variance) - 1) < 0.03


@pytest.mark.parametrize("groups,padding_mode", [(1, "zeros"), (2, "reflect"), (4, "circular")])
def test_model_conversion_and_checkpoint(groups, padding_mode):
    model = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1, groups=groups, padding_mode=padding_mode),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 3),
    ).eval()
    before = copy.deepcopy(model.state_dict())
    config = Config(array_size=(8, 8), adc_bits=8)
    analog = convert_model(model, Simulator(config, device="cpu"))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    x = torch.randn(2, 4, 8, 8)
    y = analog(x)
    assert y.shape == (2, 3) and torch.isfinite(y).all()
    state = copy.deepcopy(analog.state_dict())
    expected = analog(x)
    fresh = convert_model(model, Simulator(config, device="cpu"))
    fresh.load_state_dict(state)
    assert torch.equal(expected, fresh(x))


def test_shared_module_conversion():
    linear = nn.Linear(8, 8)
    model = nn.ModuleDict({"left": linear, "right": linear})
    analog = convert_model(model, Simulator(device="cpu"))
    assert analog["left"] is analog["right"]
    assert isinstance(model["left"], nn.Linear)


@pytest.mark.parametrize(
    "change",
    [
        {"weight_quant_gran": (32, 1)},
        {"input_quant_gran": (2, 64)},
        {"array_size": (64, 7)},
        {"adc_bits": 0},
        {"activation_bits": 1},
        {"high_conductance": float("nan")},
    ],
)
def test_invalid_config(change):
    with pytest.raises((ValueError, TypeError)):
        replace(Config(), **change)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@CUDA
@pytest.mark.parametrize(
    "ni",
    [
        Nonidealities(),
        Nonidealities(write_variation=0.02),
        Nonidealities(read_variation=0.05, read_mode="shared"),
        Nonidealities(read_variation=0.05, read_mode="slice_pair"),
        Nonidealities(read_variation=0.05, read_mode="output"),
        Nonidealities(voltage_variation=0.05, output_noise=0.005),
        Nonidealities(
            adc_nonlinearity=0.05,
            stuck_at_low=0.01,
            stuck_at_high=0.01,
            drift_coefficient=0.05,
            drift_time=100,
        ),
    ],
)
def test_cuda_all_schedules_and_reference(ni):
    config = Config(
        activation_bits=4,
        weight_bits=4,
        array_size=(8, 8),
        adc_bits=8,
        weight_quant_gran=(8, 1),
        nonidealities=ni,
    )
    sim = Simulator(config, device="cuda:0")
    array = sim.map(torch.randn(19, 17), name="check")
    x = torch.randn(3, 17, device="cuda:0")
    q, scale = input_tiles(x, config)
    baseline = array._execute(q, scale, default_schedule(config), 0)
    for schedule in [
        dict(arrays=2, vectors=1, columns=2, workers=1),
        dict(arrays=4, vectors=4, columns=4, workers=3),
        dict(arrays=8, vectors=2, columns=8, workers=4),
    ]:
        actual = array._execute(q, scale, schedule, 0)
        assert torch.equal(baseline.view(torch.int32), actual.view(torch.int32))
    expected = reference.partial(
        q.cpu(), array.packed.cpu(), config, default_schedule(config), array.seed, 0, 0, array.p
    )
    expected *= scale.cpu()[:, :, None, None]
    expected *= array.weight_scale.cpu()[None]
    expected = expected.sum(1).flatten(1)[:, : array.out_features]
    torch.testing.assert_close(baseline.cpu(), expected, rtol=1e-5, atol=2e-5)


@CUDA
def test_tuning_cache_and_argmin(tmp_path):
    config = Config(activation_bits=4, weight_bits=4, array_size=(8, 8))
    weight, x = torch.randn(13, 8), torch.randn(3, 8, device="cuda:0")
    sim = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path)
    array = sim.map(weight, name="cache")
    expected = array(x)
    record = sim.records[0]
    candidates = record["candidates"]
    assert record["complete"] and not record["cache_hit"]
    assert len(candidates) == len(list(schedule_grid(config, 3, array.p)))
    accepted = [item for item in candidates if item["status"] == "accepted"]
    assert record["selected"] == min(accepted, key=lambda item: item["seconds"])["schedule"]
    assert all(item["timed_samples"] == 1 for item in accepted)
    fresh = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path)
    actual = fresh.map(weight, name="cache")(x)
    assert torch.equal(expected, actual)
    assert fresh.records[0]["cache_hit"] and not fresh.records[0]["candidates"]


@CUDA
@pytest.mark.parametrize(
    "activation,weight,rows,columns,adc_bits",
    [
        (2, 8, 32, 64, 6),
        (8, 4, 64, 32, 8),
        (8, 8, 128, 64, 16),
    ],
)
def test_cuda_rectangular_arrays_and_mixed_precision(activation, weight, rows, columns, adc_bits):
    config = Config(
        activation_bits=activation,
        weight_bits=weight,
        array_size=(rows, columns),
        adc_bits=adc_bits,
        weight_quant_gran=(rows, 1),
    )
    values, x = torch.randn(columns + 3, rows + 7), torch.randn(3, rows + 7)
    cpu = Simulator(config, device="cpu").map(values, name="mixed")
    gpu = Simulator(config, device="cuda:0").map(values, name="mixed")
    torch.testing.assert_close(gpu(x.cuda()).cpu(), cpu(x), rtol=1e-5, atol=2e-5)


def test_float32_internal_storage_with_float64_default():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        array = Simulator(device="cpu").map(torch.randn(8, 64, dtype=torch.float32))
        result = array(torch.randn(2, 64, dtype=torch.float32))
        assert result.dtype == torch.float32 and torch.isfinite(result).all()
    finally:
        torch.set_default_dtype(previous)


def test_cpu_device_alias():
    array = Simulator(device="cpu:0").map(torch.randn(8, 64))
    assert array(torch.randn(2, 64)).shape == (2, 8)
