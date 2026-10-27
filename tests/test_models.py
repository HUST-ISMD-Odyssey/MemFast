"""CPU task/model contracts and optional integration checks."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from examples import _common
from examples.llama_lm_eval import harness_evaluate, perplexity
from memfast import Config, Simulator, convert_model
from memfast.models import MLP, RNNClassifier, cifar_model, imagenet_model, load_causal_lm


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(135)
    torch.set_num_threads(2)


def test_mlp_logits_and_gradients():
    model = MLP(input_features=12, hidden_sizes=(7, 5), num_classes=3)
    output = model(torch.randn(4, 1, 3, 4))
    assert output.shape == (4, 3)
    nn.functional.cross_entropy(output, torch.tensor([0, 1, 2, 0])).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("dimensions", [(0, (4,), 2), (4, (-1,), 2), (4, (), False)])
def test_mlp_rejects_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        MLP(*dimensions)


@pytest.mark.parametrize("name", ["vgg11_bn", "vgg16_bn", "resnet18", "resnet50"])
def test_cifar_architectures(name):
    model = cifar_model(name, num_classes=100).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 100)
    assert torch.isfinite(output).all()
    if name.startswith("resnet"):
        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.padding == (0, 0)
        assert isinstance(model.maxpool, nn.Identity)


def test_cifar_vgg_accepts_existing_scan_state():
    from examples.cifar10 import vgg11

    source = vgg11().eval()
    target = cifar_model().eval()
    target.load_state_dict(source.state_dict(), strict=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(target(x), source(x), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["vgg11_bn", "resnet18", "mobilenet_v2"])
def test_imagenet_architecture_shapes_without_weights(name):
    with torch.device("meta"):
        model = imagenet_model(name).eval()
        assert model(torch.empty(1, 3, 224, 224)).shape == (1, 1000)


def test_factories_reject_unknown_architectures():
    with pytest.raises(ValueError):
        cifar_model("unknown")
    with pytest.raises(ValueError):
        imagenet_model("unknown")


def test_deit_uses_upstream_timm(monkeypatch):
    factory = Mock(return_value=nn.Identity())
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=factory))
    imagenet_model("deit_tiny_patch16_224", weights="DEFAULT")
    factory.assert_called_once_with("deit_tiny_patch16_224", pretrained=True, num_classes=1000)


def test_rnn_forward_and_analog_conversion():
    from memfast.nn import AnalogRNN

    model = RNNClassifier(input_size=4, hidden_size=5, num_layers=2, num_classes=3).eval()
    x = torch.randn(2, 3, 4)
    with torch.inference_mode():
        digital = model(x)
    analog = convert_model(model, Simulator(Config(), device="cpu"))
    assert isinstance(analog.rnn, AnalogRNN)
    output = analog(x)
    assert output.shape == digital.shape == (2, 3)
    assert torch.isfinite(output).all()


def test_hat_checkpoint_round_trip_uses_master_parameters(tmp_path):
    digital = MLP(input_features=12, hidden_sizes=(8,), num_classes=3)
    keys = tuple(digital.state_dict())
    analog = convert_model(digital, Simulator(Config(), device="cpu"), training=True)
    initial = {key: value.detach().clone() for key, value in analog.named_parameters()}
    options = SimpleNamespace(learning_rate=0.01, limit=None, batch_size=2, workers=0, seed=135)
    batches = _common.loader(_common.synthetic((12,), 3, 4, 135), options, training=True)
    _common.classification_epoch(
        analog, batches, "cpu", optimizer=_common.optimizer_for(analog, options)
    )
    assert any(not torch.equal(initial[key], value) for key, value in analog.named_parameters())
    checkpoint = tmp_path / "hat.pt"
    _common.save_checkpoint(analog, checkpoint, state_keys=keys, architecture="mlp")
    restored = MLP(input_features=12, hidden_sizes=(8,), num_classes=3)
    assert _common.load_checkpoint(restored, checkpoint)["architecture"] == "mlp"
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, analog.state_dict()[key], rtol=0, atol=0)
    assert set(torch.load(checkpoint, weights_only=True)["state_dict"]) == set(keys)


def test_checkpoint_module_alias_and_strict_shape(tmp_path):
    model = MLP(input_features=3, hidden_sizes=(), num_classes=2)
    checkpoint = tmp_path / "ddp.pt"
    torch.save(
        {"state_dict": {"module." + key: value for key, value in model.state_dict().items()}},
        checkpoint,
    )
    _common.load_checkpoint(MLP(3, (), 2), checkpoint)
    with pytest.raises(RuntimeError):
        _common.load_checkpoint(MLP(4, (), 2), checkpoint)


def test_upstream_model_wrapper_checkpoint(tmp_path):
    model = MLP(3, (), 2)
    path = tmp_path / "model.pt"
    torch.save(dict(model=model.state_dict(), epoch=1), path)
    restored = MLP(3, (), 2)
    _common.load_checkpoint(restored, path)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


@pytest.mark.parametrize("world_size", [2, 3, 8])
def test_distributed_evaluation_has_no_padding_or_duplicates(world_size):
    args = SimpleNamespace(limit=None, batch_size=2, workers=0, seed=135)
    dataset = torch.utils.data.TensorDataset(torch.arange(5)[:, None], torch.arange(5))
    seen = []
    for rank in range(world_size):
        batches = _common.loader(dataset, args, rank=rank, world_size=world_size)
        seen.extend(int(value) for _, targets in batches for value in targets)
    assert sorted(seen) == list(range(5))


def test_local_language_loader_disables_downloads_and_remote_code(monkeypatch, tmp_path):
    config = SimpleNamespace(model_type="qwen2")
    model = nn.Linear(2, 2)
    tokenizer = object()
    library = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
    )
    monkeypatch.setitem(sys.modules, "transformers", library)
    actual, actual_tokenizer = load_causal_lm(tmp_path)
    assert actual is model and actual_tokenizer is tokenizer
    for factory in (library.AutoConfig, library.AutoModelForCausalLM, library.AutoTokenizer):
        assert factory.from_pretrained.call_args.kwargs["local_files_only"] is True
        assert factory.from_pretrained.call_args.kwargs["trust_remote_code"] is False
    config.model_type = "unknown"
    with pytest.raises(ValueError):
        load_causal_lm(tmp_path)


@pytest.mark.parametrize("length", [2, 16, 17, 33, 51])
def test_perplexity_counts_shifted_targets_once(length):
    class UniformLM(nn.Module):
        def forward(self, input_ids, use_cache):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 5))

    result = perplexity(UniformLM(), torch.arange(length)[None] % 5, max_length=16, stride=7)
    assert result["target_tokens"] == length - 1
    assert result["negative_log_likelihood"] == pytest.approx(math.log(5))
    assert result["perplexity"] == pytest.approx(5)


def test_harness_adapter_receives_live_converted_model(monkeypatch):
    model, tokenizer = nn.Linear(2, 2), object()
    adapter = Mock(return_value=object())
    evaluate = Mock(return_value={"results": {"task": {"acc,none": 0.5}}})
    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=evaluate))
    monkeypatch.setitem(sys.modules, "lm_eval.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", SimpleNamespace(HFLM=adapter))
    args = SimpleNamespace(
        device="cpu",
        batch_size=2,
        max_length=16,
        tasks="one,two",
        num_fewshot=0,
        limit=4,
        seed=135,
    )
    result = harness_evaluate(model, tokenizer, args)
    assert adapter.call_args.kwargs["pretrained"] is model
    assert adapter.call_args.kwargs["tokenizer"] is tokenizer
    assert evaluate.call_args.kwargs["tasks"] == ["one", "two"]
    assert result["results"]["task"]["acc,none"] == 0.5


@pytest.mark.parametrize("value,expected", [("6", 6), ("6,8,10", (6, 8, 10))])
def test_adc_cli_scalar_or_slice(value, expected):
    assert _common.parser("test").parse_args(["--adc-bits", value]).adc_bits == expected


@pytest.mark.parametrize(
    "module,extra",
    [
        ("matrix", []),
        ("mnist", []),
        ("mlp_hat", []),
        ("mlp_hat_ddp", ["--limit", "3"]),
        ("rnn", []),
        ("cifar", ["--digital", "--model", "resnet18"]),
        ("imagenet", ["--digital", "--model", "mobilenet_v2"]),
    ],
)
def test_core_cli_smoke(module, extra):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", f"examples.{module}", "--smoke", *extra],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["synthetic"] is True
    if "evaluation" in result:
        assert result["evaluation"]["samples"] == (3 if module == "mlp_hat_ddp" else 4)


def test_optional_harness_smoke_runs_real_adapter():
    pytest.importorskip("transformers")
    pytest.importorskip("lm_eval")
    from examples.llama_lm_eval import smoke_model

    model, tokenizer = smoke_model("llama", "cpu")
    args = SimpleNamespace(
        device="cpu",
        batch_size=2,
        max_length=16,
        tasks=None,
        num_fewshot=0,
        limit=4,
        seed=135,
    )
    result = harness_evaluate(model, tokenizer, args, smoke=True)
    assert "memfast_in_memory" in result["results"]


def test_imagenet_rejects_class_count_mismatch(tmp_path):
    from PIL import Image

    from examples._classification import dataset

    directory = tmp_path / "class_a"
    directory.mkdir()
    Image.new("RGB", (4, 4)).save(directory / "image.png")
    args = SimpleNamespace(
        smoke=False,
        task="imagenet",
        model="resnet18",
        weights=None,
        data_root=tmp_path,
        num_classes=2,
    )
    with pytest.raises(ValueError, match="classes"):
        dataset(args, None)
