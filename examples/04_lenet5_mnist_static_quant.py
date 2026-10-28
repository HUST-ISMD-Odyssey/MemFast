"""Calibrate LeNet on MNIST training images, save static INT4 weights, and infer."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from memfast import Config, Nonidealities, Simulator, convert_model
from memfast.models import LeNet5
from memfast.nn import reset_read_events
from memfast.simulator import AnalogMatrix

if __package__:
    from .common import evaluate, load_checkpoint
else:
    from common import evaluate, load_checkpoint


EXAMPLES = Path(__file__).resolve().parent
DEFAULT_SOURCE = EXAMPLES / "checkpoints/lenet5_mnist.pt"
DEFAULT_STATIC = EXAMPLES / "checkpoints/lenet5_mnist_static_int4.pt"
FORMAT = "memfast.lenet5.static-int4.v1"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/mnist")
CHECKPOINT = DEFAULT_SOURCE
STATIC_CHECKPOINT = DEFAULT_STATIC
CALIBRATED_CHECKPOINT = EXAMPLES / "output/lenet5_mnist_static_int4.pt"
BATCH_SIZE = 128
CALIBRATION_SAMPLES = 1024
CALIBRATION_SEED = 135
INCLUDE = None
EXCLUDE = ()
CONFIG = Config(
    activation_bits=8,
    weight_bits=4,
    input_slice=None,
    weight_slice=None,  # Binary slices, matching the shipped checkpoint.
    array_size=(64, 64),
    rows_per_read=None,  # All 64 rows per read.
    input_quant_gran=(1, None),
    weight_quant_gran=(None, 1),
    adc_bits=6,
    bw_e=None,
    nonidealities=Nonidealities(write_variation=0.02),
)


def matrix_layers(model):
    return {
        name: module for name, module in model.named_modules() if isinstance(module, AnalogMatrix)
    }


@torch.inference_mode()
def calibrate(model, batches, *, activation_bits):
    """Observe each programmed matrix before quantization and then freeze its scale."""
    matrices = matrix_layers(model)
    if not matrices:
        raise ValueError("static calibration requires at least one converted layer")
    maxima = {
        name: torch.zeros((), device=module.packed.device) for name, module in matrices.items()
    }
    calls = {name: 0 for name in matrices}

    def observer(name):
        def observe(module, inputs):
            value = inputs[0].detach()
            if not value.numel() or not torch.isfinite(value).all():
                raise ValueError("calibration inputs must be finite and nonempty")
            maxima[name] = torch.maximum(maxima[name], value.abs().amax())
            calls[name] += 1

        return observe

    handles = [
        module.register_forward_pre_hook(observer(name)) for name, module in matrices.items()
    ]
    seen = 0
    started = time.perf_counter()
    model.eval()
    device = next(iter(matrices.values())).packed.device
    try:
        for inputs, _ in batches:
            output = model(inputs.to(device))
            if not torch.isfinite(output).all():
                raise FloatingPointError("nonfinite calibration output")
            seen += len(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if not seen or any(not count for count in calls.values()):
        raise ValueError("calibration must exercise every converted layer")
    upper = (1 << (activation_bits - 1)) - 1
    layers = {}
    for name, module in matrices.items():
        maximum = float(maxima[name])
        scale = max(maximum if maximum > 0 else 1.0, torch.finfo(torch.float32).tiny) / upper
        module.set_input_scale(scale)
        layers[name] = dict(
            input_features=module.in_features,
            output_features=module.out_features,
            maximum=maximum,
            input_scale=scale,
            calls=calls[name],
        )
    reset_read_events(model)
    return dict(samples=seen, seconds=time.perf_counter() - started, layers=layers)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indices_sha256(indices):
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode("ascii")).hexdigest()


def static_payload(model, engine, calibration, indices):
    """The state contains packed integer weights and scales, without FP32 weight copies."""
    state = {
        name: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for name, value in model.state_dict().items()
    }
    return dict(
        format=FORMAT,
        architecture="lenet5_mnist",
        state_dict=state,
        config=asdict(engine.config),
        conversion=dict(include=INCLUDE, exclude=list(EXCLUDE)),
        normalization=dict(mean=[0.5], std=[0.5]),
        calibration=dict(
            calibration,
            split="train",
            seed=CALIBRATION_SEED,
            indices=indices,
            indices_sha256=indices_sha256(indices),
        ),
        source=dict(
            checkpoint=str(CHECKPOINT),
            sha256=file_sha256(CHECKPOINT),
        ),
        synthetic=False,
    )


def restore_static(model, payload, engine):
    if payload.get("format") != FORMAT or payload.get("architecture") != "lenet5_mnist":
        raise ValueError("expected a LeNet static INT4 checkpoint produced by this example")
    saved_config = dict(payload["config"])
    effects = dict(saved_config["nonidealities"])
    if effects.pop("output_noise", 0) != 0:
        raise ValueError("checkpoint uses a removed noise model")
    saved_config["nonidealities"] = effects
    if saved_config != asdict(engine.config):
        raise ValueError("simulator configuration must match the saved static checkpoint")
    if payload["conversion"] != dict(include=INCLUDE, exclude=list(EXCLUDE)):
        raise ValueError("include/exclude must match the saved static checkpoint")
    matrices = matrix_layers(model)
    if set(matrices) != set(payload["calibration"]["layers"]):
        raise ValueError("checkpoint calibration layers differ from the converted model")
    for name in matrices:
        scale = payload["state_dict"].get(name + ".input_scale")
        if (
            not torch.is_tensor(scale)
            or scale.shape != (1, matrices[name].in_features)
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError(f"{name}: checkpoint has no valid fixed input scale")
    model.load_state_dict(payload["state_dict"], strict=True)
    # Revalidate loaded scale grouping through the public API before inference.
    for module in matrices.values():
        module.set_input_scale(module.input_scale)
    reset_read_events(model)
    return {name: value for name, value in payload["calibration"].items() if name != "indices"}


def main(stage="inference"):
    torch.manual_seed(135)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    model = LeNet5().float().to(DEVICE)
    if stage == "calibrate":
        load_checkpoint(model, CHECKPOINT)
    engine = Simulator(CONFIG, device=DEVICE)
    model = convert_model(
        model,
        engine,
        include=INCLUDE,
        exclude=EXCLUDE,
        inplace=True,
    )
    if stage == "calibrate":
        dataset = datasets.MNIST(DATA_ROOT, train=True, transform=transform, download=True)
        generator = torch.Generator().manual_seed(CALIBRATION_SEED)
        indices = torch.randperm(len(dataset), generator=generator)[:CALIBRATION_SAMPLES].tolist()
        calibration = calibrate(
            model,
            DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE),
            activation_bits=engine.config.activation_bits,
        )
        payload = static_payload(model, engine, calibration, indices)
        CALIBRATED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, CALIBRATED_CHECKPOINT)
        print(f"Saved packed weights and fixed input scales: {CALIBRATED_CHECKPOINT}")
    else:
        payload = torch.load(STATIC_CHECKPOINT, map_location="cpu", weights_only=True)
        restore_static(model, payload, engine)
    test_data = datasets.MNIST(DATA_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    command = argparse.ArgumentParser(__doc__)
    command.add_argument("--stage", choices=("calibrate", "inference"), default="inference")
    main(command.parse_args().stage)
