import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from memfast import Config, Nonidealities, Simulator
from memfast.nn import (
    AnalogConv2d,
    AnalogLinear,
    AnalogRNN,
    AnalogRNNCell,
    HATConv2d,
    HATLinear,
    TrainableAnalogConv2d,
    TrainableAnalogLinear,
    convert_model,
    reset_read_events,
)
from memfast.quantization import grouped_quantize


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(135)
    torch.set_num_threads(2)


def simulator(noisy=False):
    noise = Nonidealities(write_variation=0.02, read_variation=0.05) if noisy else Nonidealities()
    return Simulator(
        Config(
            activation_bits=4, weight_bits=4, array_size=(8, 8), adc_bits=8, nonidealities=noise
        ),
        device="cpu",
    )


@pytest.mark.parametrize("bias", [False, True])
def test_hat_linear_exact_analog_forward_and_digital_gradients(bias):
    source = nn.Linear(5, 4, bias=bias)
    layer = HATLinear(source, simulator(True), name="same", gradient_mode="digital")
    reference = AnalogLinear(source, simulator(True), name="same")
    x = torch.randn(2, 3, 5, requires_grad=True)
    upstream = torch.randn(2, 3, 4)
    actual = layer(x)
    assert torch.equal(actual.detach(), reference(x.detach()))
    assert not torch.equal(actual.detach(), F.linear(x, source.weight, source.bias).detach())
    (actual * upstream).sum().backward()
    rx = x.detach().clone().requires_grad_()
    digital = copy.deepcopy(source)
    (digital(rx) * upstream).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    torch.testing.assert_close(layer.weight.grad, digital.weight.grad)
    if bias:
        torch.testing.assert_close(layer.bias.grad, digital.bias.grad)
    assert source.weight.grad is None


def test_hat_optimizer_reprograms_without_replacing_parameters_or_buffers():
    layer = HATLinear(nn.Linear(5, 3), simulator(True), name="update")
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    names = {name: id(value) for name, value in layer.named_parameters()}
    buffers = {name: id(value) for name, value in layer.named_buffers()}
    x = torch.randn(4, 5)
    first = layer(x)
    first.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    event = layer.matrix._event
    with torch.no_grad():
        expected = layer.matrix.simulator.map(layer.weight, bias=layer.bias, name="update")
        expected.reset_read(event)
        actual = layer(x)
    assert torch.equal(actual, expected(x))
    assert not torch.equal(first.detach(), actual)
    assert names == {name: id(value) for name, value in layer.named_parameters()}
    assert buffers == {name: id(value) for name, value in layer.named_buffers()}
    assert layer.matrix._event == event + 1


@pytest.mark.parametrize(
    "groups,padding_mode,stride,dilation",
    [(1, "zeros", 2, 1), (2, "reflect", 1, 2), (4, "circular", 2, 1)],
)
@pytest.mark.parametrize("bias", [False, True])
def test_hat_conv_exact_forward_and_gradients(groups, padding_mode, stride, dilation, bias):
    source = nn.Conv2d(
        4,
        4,
        3,
        stride=stride,
        padding=2,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode,
        bias=bias,
    )
    layer = HATConv2d(source, simulator(True), name="same", gradient_mode="digital")
    reference = AnalogConv2d(source, simulator(True), name="same")
    x = torch.randn(2, 4, 7, 8, requires_grad=True)
    actual = layer(x)
    assert torch.equal(actual.detach(), reference(x.detach()))
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    rx = x.detach().clone().requires_grad_()
    digital = copy.deepcopy(source)
    (digital(rx) * upstream).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    torch.testing.assert_close(layer.weight.grad, digital.weight.grad)
    if bias:
        torch.testing.assert_close(layer.bias.grad, digital.bias.grad)


def dequantized_operand(value, bits, granularity, config):
    code, scale = grouped_quantize(value.detach(), bits, granularity, config.bw_e)
    return code * scale.to(torch.float64 if config.wide else torch.float32)


@pytest.mark.parametrize("activation,weight,bw_e", [(4, 4, None), (4, 4, 4), (4, 10, None)])
@pytest.mark.parametrize("bias", [False, True])
def test_default_linear_quantized_operand_gradient_oracle(activation, weight, bw_e, bias):
    config = Config(
        activation_bits=activation,
        weight_bits=weight,
        bw_e=bw_e,
        array_size=(8, 4),
        rows_per_read=4,
        adc_bits=8,
        input_quant_gran=(2, 4),
        weight_quant_gran=(4, 2),
        nonidealities=Nonidealities(write_variation=0.02, read_variation=0.05),
    )
    source = nn.Linear(11, 5, bias=bias)
    layer = HATLinear(source, Simulator(config, device="cpu"), name="oracle")
    reference = AnalogLinear(source, Simulator(config, device="cpu"), name="oracle")
    x = torch.randn(2, 3, 11, requires_grad=True)
    upstream = torch.randn(2, 3, 5)
    output = layer(x)
    assert layer.gradient_mode == "quantized"
    assert torch.equal(output.detach(), reference(x.detach()))
    (output * upstream).sum().backward()
    dx = dequantized_operand(x.reshape(-1, 11), activation, config.input_quant_gran, config)
    dw = dequantized_operand(source.weight.T, weight, config.weight_quant_gran, config)
    grad_output = upstream.reshape(-1, 5).to(dx.dtype)
    expected_input = (grad_output @ dw.T).reshape(x.shape).to(x.dtype)
    expected_weight = (dx.T @ grad_output).T.to(source.weight.dtype)
    torch.testing.assert_close(x.grad, expected_input)
    torch.testing.assert_close(layer.weight.grad, expected_weight)
    if bias:
        torch.testing.assert_close(layer.bias.grad, upstream.sum((0, 1)))


def convolution_gradient_oracle(source, x, upstream, config):
    rx = x.detach().clone().requires_grad_()
    padding = source.padding
    if source.padding_mode != "zeros":
        ph, pw = padding
        padded = F.pad(rx, (pw, pw, ph, ph), mode=source.padding_mode)
        padding = (0, 0)
    else:
        padded = rx
    patches = F.unfold(padded, source.kernel_size, source.dilation, padding, source.stride)
    n, _, positions = patches.shape
    features = source.weight[0].numel()
    outputs = source.out_channels // source.groups
    input_gradients, weight_gradients = [], []
    for group in range(source.groups):
        sl = slice(group * outputs, (group + 1) * outputs)
        px = patches[:, group * features : (group + 1) * features]
        dx = dequantized_operand(
            px.transpose(1, 2).reshape(n * positions, features),
            config.activation_bits,
            config.input_quant_gran,
            config,
        )
        dw = dequantized_operand(
            source.weight[sl].flatten(1).T,
            config.weight_bits,
            config.weight_quant_gran,
            config,
        )
        grad = upstream[:, sl].flatten(2).transpose(1, 2).reshape(n * positions, outputs)
        grad = grad.to(dx.dtype)
        input_gradients.append((grad @ dw.T).reshape(n, positions, features))
        weight_gradients.append((dx.T @ grad).T.reshape(source.weight[sl].shape))
    grad_patches = torch.cat(input_gradients, dim=-1).transpose(1, 2).to(patches.dtype)
    # Use only unfold/padding's adjoint; no STE graph defines this oracle.
    expected_input = torch.autograd.grad(patches, rx, grad_patches)[0]
    expected_weight = torch.cat(weight_gradients).to(source.weight.dtype)
    return expected_input, expected_weight, upstream.sum((0, 2, 3))


@pytest.mark.parametrize("bw_e", [None, 4])
@pytest.mark.parametrize(
    "groups,outputs,padding_mode",
    [(1, 4, "zeros"), (2, 6, "reflect"), (4, 4, "circular")],
)
@pytest.mark.parametrize("bias", [False, True])
def test_default_conv_quantized_patch_gradient_oracle(groups, outputs, padding_mode, bw_e, bias):
    config = Config(
        activation_bits=4,
        weight_bits=4,
        bw_e=bw_e,
        array_size=(8, 4),
        rows_per_read=4,
        adc_bits=8,
        input_quant_gran=(5, 4),
        weight_quant_gran=(4, 2),
        nonidealities=Nonidealities(write_variation=0.02, read_variation=0.05),
    )
    source = nn.Conv2d(
        4,
        outputs,
        (3, 2),
        stride=(2, 1),
        dilation=(2, 1),
        padding=(2, 1),
        groups=groups,
        padding_mode=padding_mode,
        bias=bias,
    )
    layer = HATConv2d(source, Simulator(config, device="cpu"), name="oracle")
    reference = AnalogConv2d(source, Simulator(config, device="cpu"), name="oracle")
    x = torch.randn(2, 4, 7, 8, requires_grad=True)
    output = layer(x)
    assert layer.gradient_mode == "quantized"
    assert torch.equal(output.detach(), reference(x.detach()))
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    expected_input, expected_weight, expected_bias = convolution_gradient_oracle(
        source, x, upstream, config
    )
    torch.testing.assert_close(x.grad, expected_input)
    torch.testing.assert_close(layer.weight.grad, expected_weight)
    if bias:
        torch.testing.assert_close(layer.bias.grad, expected_bias)


@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_hat_checkpoint_restores_next_read_after_optimizer_step(kind):
    source = nn.Linear(5, 3) if kind == "linear" else nn.Conv2d(2, 4, 3, padding=1, groups=2)
    x = torch.randn(2, 5) if kind == "linear" else torch.randn(2, 2, 4, 4)
    layer = convert_model(source, simulator(True), training=True)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)
    layer(x).sum().backward()
    optimizer.step()
    state = copy.deepcopy(layer.state_dict())
    expected = layer(x).detach()
    fresh = convert_model(source, simulator(True), training=True)
    fresh.load_state_dict(state)
    assert torch.equal(fresh(x).detach(), expected)
    reset_read_events(fresh)
    reset_read_events(layer)
    assert torch.equal(fresh(x).detach(), layer(x).detach())


def test_hat_repeated_forward_graphs_and_empty_batch():
    layer = HATLinear(nn.Linear(5, 3), simulator())
    x = torch.randn(2, 5, requires_grad=True)
    (layer(x).sum() + layer(x).sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    layer.zero_grad()
    result = layer(torch.empty(0, 5))
    assert result.shape == (0, 3)
    result.sum().backward()
    assert torch.equal(layer.weight.grad, torch.zeros_like(layer.weight))
    assert torch.equal(layer.bias.grad, torch.zeros_like(layer.bias))


def test_training_conversion_preserves_frozen_parameters_and_source():
    model = nn.Sequential(nn.Linear(5, 5), nn.BatchNorm1d(5), nn.Linear(5, 2))
    model[0].weight.requires_grad_(False)
    model.eval()
    before = copy.deepcopy(model.state_dict())
    analog = convert_model(model, simulator(), exclude=("2",), training=True)
    assert isinstance(analog[0], TrainableAnalogLinear)
    assert isinstance(analog[2], nn.Linear)
    assert analog.training and analog[1].training
    assert not analog[0].weight.requires_grad and analog[0].bias.requires_grad
    assert analog[2].weight.requires_grad
    assert not model.training and isinstance(model[0], nn.Linear)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    frozen = convert_model(model, simulator())
    assert not frozen.training and not any(p.requires_grad for p in frozen.parameters())


@pytest.mark.parametrize("training", [False, True])
def test_shared_modules_and_exclusion_aliases(training):
    shared = nn.Sequential(nn.Linear(5, 3))
    model = nn.ModuleDict({"left": shared, "right": shared})
    converted = convert_model(model, simulator(), training=training)
    assert converted["left"] is converted["right"]
    assert converted["left"][0] is converted["right"][0]
    excluded = convert_model(model, simulator(), exclude=("right.0",), training=training)
    assert excluded["left"] is excluded["right"]
    assert isinstance(excluded["left"][0], nn.Linear)
    assert excluded["left"][0] is excluded["right"][0]


@pytest.mark.parametrize("training", [False, True])
def test_root_and_subtree_exclusions(training):
    root = nn.Linear(5, 3)
    excluded = convert_model(root, simulator(), exclude=("",), training=training)
    assert isinstance(excluded, nn.Linear)
    model = nn.Sequential(nn.Sequential(nn.Linear(5, 3)), nn.Linear(3, 2))
    converted = convert_model(model, simulator(), exclude=("0",), training=training)
    assert isinstance(converted[0][0], nn.Linear)
    assert not isinstance(converted[1], nn.Linear)
    complete = convert_model(model, simulator(), exclude=("",), training=training)
    assert isinstance(complete[1], nn.Linear)


class LegacyLlamaProjection(nn.Module):
    """Exercise the old Llama direct-weight branch without transformers."""

    def __init__(self, *, model_type="llama", pretraining_tp=2):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, pretraining_tp=pretraining_tp)
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            weights = self.projection.weight.chunk(self.config.pretraining_tp, dim=0)
            return torch.cat([F.linear(x, weight) for weight in weights], dim=-1)
        return self.projection(x)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_llama_tensor_partitions_invoke_converted_projection(training, inplace):
    source = LegacyLlamaProjection()
    original_config = source.config
    original_state = copy.deepcopy(source.state_dict())
    x = torch.randn(2, 4)
    digital = source(x).detach()
    expected = AnalogLinear(source.projection, simulator(True), name="projection")(x)
    converted = convert_model(source, simulator(True), training=training, inplace=inplace)
    assert converted.config.pretraining_tp == 1
    if inplace:
        assert converted is source and original_config.pretraining_tp == 1
    else:
        assert converted.config is not original_config
        assert source.config.pretraining_tp == 2
        assert isinstance(source.projection, nn.Linear)
        for key, value in source.state_dict().items():
            assert torch.equal(value, original_state[key])
        torch.testing.assert_close(source(x), digital)
    output = converted(x)
    assert torch.equal(output.detach(), expected)
    assert not torch.equal(output.detach(), digital)
    matrix = converted.projection.matrix if training else converted.projection
    assert matrix._event == 1
    if training:
        output.sum().backward()
        assert converted.projection.weight.grad is not None
        assert torch.isfinite(converted.projection.weight.grad).all()


@pytest.mark.parametrize("exclude", [("",), ("projection",)])
def test_llama_tensor_partitions_unchanged_when_fully_excluded(exclude):
    source = LegacyLlamaProjection()
    x = torch.randn(2, 4)
    converted = convert_model(source, simulator(), exclude=exclude)
    assert converted.config.pretraining_tp == source.config.pretraining_tp == 2
    assert isinstance(converted.projection, nn.Linear)
    torch.testing.assert_close(converted(x), source(x), rtol=0, atol=0)


def test_llama_tensor_partition_normalization_is_nested_and_family_specific():
    llama = LegacyLlamaProjection()
    other = LegacyLlamaProjection(model_type="other")
    target = nn.ModuleDict({"llama": llama, "other": other})
    converted = convert_model(target, simulator())
    assert converted["llama"].config.pretraining_tp == 1
    assert converted["other"].config.pretraining_tp == 2
    assert llama.config.pretraining_tp == other.config.pretraining_tp == 2
    assert torch.isfinite(converted["llama"](torch.ones(1, 4))).all()


@pytest.mark.parametrize("training", [False, True])
def test_optional_llama_tensor_partitions_run_converted_layers(training):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        pretraining_tp=2,
    )
    config._attn_implementation = "eager"
    source = transformers.LlamaForCausalLM(config).eval()
    converted = convert_model(source, simulator(), training=training)
    assert source.config.pretraining_tp == 2
    assert converted.config.pretraining_tp == 1
    logits = converted(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits
    assert logits.shape == (1, 3, 16) and torch.isfinite(logits).all()
    projections = [
        module
        for module in converted.modules()
        if isinstance(module, (AnalogLinear, TrainableAnalogLinear))
    ]
    matrices = [
        module.matrix if isinstance(module, TrainableAnalogLinear) else module
        for module in projections
    ]
    assert matrices and all(matrix._event == 1 for matrix in matrices)
    if training:
        logits.sum().backward()
        assert converted.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_tied_parameters_survive_training_conversion_and_digital_exclusion():
    left, right = nn.Linear(5, 3), nn.Linear(5, 3)
    right.weight, right.bias = left.weight, left.bias
    model = nn.ModuleDict({"left": left, "right": right})
    converted = convert_model(model, simulator(), training=True)
    assert converted["left"].weight is converted["right"].weight
    assert converted["left"].bias is converted["right"].bias
    partial = convert_model(model, simulator(), exclude=("right",), training=True)
    assert partial["left"].weight is partial["right"].weight
    assert partial["left"].weight is not left.weight


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"exclude": "0"}, TypeError),
        ({"exclude": ("missing",)}, ValueError),
        ({"exclude": (1,)}, TypeError),
        ({"training": "yes"}, TypeError),
        ({"gradient_mode": "other"}, ValueError),
        ({"gradient_mode": None}, ValueError),
    ],
)
def test_invalid_conversion_options(kwargs, exception):
    with pytest.raises(exception):
        convert_model(nn.Sequential(nn.Linear(5, 3)), simulator(), **kwargs)


def test_inplace_conversion_retains_optimizer_parameter_references():
    model = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
    parameter = model[0].weight
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = convert_model(model, simulator(), training=True, inplace=True)
    assert result is model and result[0].weight is parameter
    result(torch.ones(2, 5)).sum().backward()
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(parameter, before)


def test_recurrent_tied_parameters_and_optimizer_references():
    source = nn.RNN(3, 3)
    source.weight_hh_l0 = source.weight_ih_l0
    parameter = source.weight_ih_l0
    layer = convert_model(source, simulator(), training=True, inplace=True)
    assert layer.cells[0].input.weight is parameter
    assert layer.cells[0].input.weight is layer.cells[0].hidden.weight
    output, hidden = layer(torch.randn(2, 1, 3))
    (output.sum() + hidden.sum()).backward()
    assert parameter.grad is not None


def test_training_conversion_rejects_inference_only_weights():
    inference = convert_model(nn.Linear(3, 2), simulator())
    with pytest.raises(ValueError, match="original digital"):
        convert_model(inference, simulator(), training=True)


@pytest.mark.parametrize("gradient_mode", ["quantized", "digital"])
@pytest.mark.parametrize("kind", ["linear", "conv", "cell", "rnn"])
def test_gradient_mode_propagates_through_conversion_and_reconversion(kind, gradient_mode):
    source = {
        "linear": lambda: nn.Linear(3, 2),
        "conv": lambda: nn.Conv2d(2, 2, 1),
        "cell": lambda: nn.RNNCell(3, 2),
        "rnn": lambda: nn.RNN(3, 2, num_layers=2, bidirectional=True),
    }[kind]()
    model = convert_model(source, simulator(), training=True, gradient_mode=gradient_mode)
    assert model.gradient_mode == gradient_mode
    assert all(
        child.gradient_mode == gradient_mode
        for child in model.modules()
        if isinstance(child, (TrainableAnalogLinear, TrainableAnalogConv2d, AnalogRNNCell))
    )
    other = "digital" if gradient_mode == "quantized" else "quantized"
    converted = convert_model(model, simulator(), training=True, gradient_mode=other)
    assert converted.gradient_mode == other
    assert all(
        child.gradient_mode == other
        for child in converted.modules()
        if isinstance(child, (TrainableAnalogLinear, TrainableAnalogConv2d, AnalogRNNCell))
    )
    assert model.gradient_mode == gradient_mode


@pytest.mark.parametrize("bits,bw_e", [(10, None), (16, None), (20, None), (8, 4)])
@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_hat_wide_and_bfp_preserve_scale_precision(bits, bw_e, kind):
    config = Config(
        activation_bits=bits,
        weight_bits=bits,
        array_size=(4, 4),
        adc_bits=8,
        input_slice=(1, 1, *((2,) * ((bits - 2) // 2))),
        weight_slice=(1, 1, *((2,) * ((bits - 2) // 2))),
        bw_e=bw_e,
    )
    sim = Simulator(config, device="cpu")
    source = nn.Linear(3, 2) if kind == "linear" else nn.Conv2d(2, 2, 1)
    x = torch.randn(2, 3) if kind == "linear" else torch.randn(1, 2, 2, 2)
    layer = convert_model(source, sim, training=True)
    result = layer(x)
    result.sum().backward()
    assert torch.isfinite(result).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    matrices = [module for module in layer.modules() if hasattr(module, "weight_scale")]
    assert matrices and all(matrix.weight_scale.dtype == torch.float64 for matrix in matrices)


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_rnn_cell_uses_two_analog_projections_and_biases(nonlinearity, training, bias):
    source = nn.RNNCell(5, 3, nonlinearity=nonlinearity, bias=bias)
    cell = AnalogRNNCell(source, simulator(True), training=training, name="cell")
    x, hidden = torch.randn(2, 5), torch.randn(2, 3)
    actual = cell(x, hidden)
    reset_read_events(cell)
    expected = cell.input(x) + cell.hidden(hidden)
    expected = torch.tanh(expected) if nonlinearity == "tanh" else F.relu(expected)
    assert torch.equal(actual.detach(), expected.detach())
    reset_read_events(cell)
    assert cell(x[0]).shape == (3,)
    if training:
        actual.sum().backward()
        assert all(p.grad is not None for p in cell.parameters())


@pytest.mark.parametrize("gradient_mode", ["quantized", "digital"])
@pytest.mark.parametrize("bw_e", [None, 4])
def test_rnn_cell_gradient_mode_matches_closed_form(gradient_mode, bw_e):
    config = Config(
        activation_bits=4,
        weight_bits=4,
        bw_e=bw_e,
        array_size=(8, 4),
        rows_per_read=4,
        adc_bits=8,
        input_quant_gran=(2, 4),
        weight_quant_gran=(4, 2),
        nonidealities=Nonidealities(write_variation=0.02, read_variation=0.05),
    )
    source = nn.RNNCell(5, 3)
    cell = AnalogRNNCell(
        source, Simulator(config, device="cpu"), training=True, gradient_mode=gradient_mode
    )
    x = torch.randn(3, 5, requires_grad=True)
    hidden = torch.randn(3, 3, requires_grad=True)
    output = cell(x, hidden)
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    grad_pre = upstream * (1 - output.detach().square())
    for projection, operand, original in (
        (cell.input, x, source.weight_ih),
        (cell.hidden, hidden, source.weight_hh),
    ):
        if gradient_mode == "quantized":
            dx = dequantized_operand(
                operand, config.activation_bits, config.input_quant_gran, config
            )
            dw = dequantized_operand(
                original.T, config.weight_bits, config.weight_quant_gran, config
            )
        else:
            dx, dw = operand.detach(), original.detach().T
        grad = grad_pre.to(dx.dtype)
        torch.testing.assert_close(operand.grad, (grad @ dw.T).to(operand.dtype))
        torch.testing.assert_close(projection.weight.grad, (dx.T @ grad).T.to(original.dtype))
        torch.testing.assert_close(projection.bias.grad, grad_pre.sum(0))


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("unbatched", [False, True])
def test_rnn_sequence_topology_against_pytorch(monkeypatch, batch_first, bidirectional, unbatched):
    # Isolate traversal from quantization with independent digital projections.
    def digital_projection(weight, bias, simulator, name, training, gradient_mode):
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        with torch.no_grad():
            layer.weight.copy_(weight)
            if bias is not None:
                layer.bias.copy_(bias)
        return layer

    monkeypatch.setattr("memfast.nn.recurrent._linear", digital_projection)
    source = nn.RNN(3, 4, num_layers=2, batch_first=batch_first, bidirectional=bidirectional)
    layer = AnalogRNN(source, simulator())
    x = torch.randn(5, 3) if unbatched else torch.randn(2, 5, 3)
    if not unbatched and not batch_first:
        x = x.transpose(0, 1)
    directions = 2 if bidirectional else 1
    hx = torch.randn(2 * directions, 4) if unbatched else torch.randn(2 * directions, 2, 4)
    expected, expected_h = source(x, hx)
    actual, actual_h = layer(x, hx)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_h, expected_h)


def test_real_recurrent_training_backpropagates_and_checkpoints():
    source = nn.RNN(3, 4, num_layers=2, bidirectional=True, batch_first=True)
    layer = convert_model(source, simulator(True), training=True)
    x = torch.randn(2, 3, 3, requires_grad=True)
    output, hidden = layer(x)
    (output.square().sum() + hidden.square().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    state = copy.deepcopy(layer.state_dict())
    expected, expected_h = layer(x)
    fresh = convert_model(source, simulator(True), training=True)
    fresh.load_state_dict(state)
    actual, actual_h = fresh(x)
    assert torch.equal(actual.detach(), expected.detach())
    assert torch.equal(actual_h.detach(), expected_h.detach())


def test_recurrent_real_inference_zero_weights_matches_bias_oracle():
    source = nn.RNN(3, 4, num_layers=2, bidirectional=True, nonlinearity="relu")
    with torch.no_grad():
        for name, value in source.named_parameters():
            value.fill_(0.25 if name.startswith("bias") else 0)
    x = torch.randn(3, 2, 3)
    layer = convert_model(source, simulator())
    output, hidden = layer(x)
    assert torch.equal(output, torch.full((3, 2, 8), 0.5))
    assert torch.equal(hidden, torch.full((4, 2, 4), 0.5))
    for matrix in layer.modules():
        if isinstance(matrix, AnalogLinear):
            assert matrix._event == 3


@pytest.mark.parametrize("kind", ["cell", "rnn"])
def test_recurrent_invalid_state_and_empty_sequence(kind):
    layer = (
        AnalogRNNCell(nn.RNNCell(3, 4), simulator())
        if kind == "cell"
        else AnalogRNN(nn.RNN(3, 4), simulator())
    )
    x = torch.randn(2, 3) if kind == "cell" else torch.randn(2, 2, 3)
    with pytest.raises(ValueError):
        layer(x, torch.randn(9, 4))
    with pytest.raises(ValueError):
        layer(torch.randn(2, 8))
    if kind == "rnn":
        with pytest.raises(ValueError):
            layer(torch.empty(0, 2, 3))


@pytest.mark.parametrize("gradient_mode", ["quantized", "digital"])
def test_hat_all_parameters_used_by_cpu_ddp(tmp_path, gradient_mode):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU DDP requires Gloo")
    if dist.is_initialized():
        pytest.skip("test requires an isolated default process group")
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "ddp_init").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=10),
    )
    try:
        source = nn.Sequential(nn.Conv2d(2, 2, 1, groups=2), nn.Flatten(), nn.Linear(8, 3))
        model = convert_model(source, simulator(), training=True, gradient_mode=gradient_mode)
        ddp = DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            ddp(torch.randn(2, 2, 2, 2)).square().sum().backward()
            assert all(p.grad is not None for p in model.parameters())
            optimizer.step()
    finally:
        dist.destroy_process_group()
