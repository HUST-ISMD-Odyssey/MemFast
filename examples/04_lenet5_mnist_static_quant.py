"""Calibrate LeNet on MNIST training images, save static INT4 weights, and infer."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Subset

from memfast import convert_model
from memfast.models import LeNet5
from memfast.nn import reset_read_events
from memfast.simulator import AnalogMatrix

if __package__:
    from .common import (
        classification_epoch,
        load_checkpoint,
        loader,
        parser,
        positive,
        report,
        setup,
        simulator,
    )
    from .common.mnist import mnist_dataset
else:
    from common import (
        classification_epoch,
        load_checkpoint,
        loader,
        parser,
        positive,
        report,
        setup,
        simulator,
    )
    from common.mnist import mnist_dataset


EXAMPLES = Path(__file__).resolve().parent
DEFAULT_SOURCE = EXAMPLES / "checkpoints/lenet5_mnist.pt"
DEFAULT_STATIC = EXAMPLES / "checkpoints/lenet5_mnist_static_int4.pt"
FORMAT = "memfast.lenet5.static-int4.v1"


def matrix_layers(model):
    return {
        name: module for name, module in model.named_modules() if isinstance(module, AnalogMatrix)
    }


def calibration_data(args):
    """Select only training indices; evaluation limits do not choose calibration data."""
    dataset = mnist_dataset(args, training=True)
    count = min(args.calibration_samples, len(dataset))
    generator = torch.Generator().manual_seed(args.calibration_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices), indices


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


def static_payload(model, engine, calibration, args, indices):
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
        conversion=dict(include=args.include, exclude=args.exclude),
        normalization=dict(mean=[0.5], std=[0.5]),
        calibration=dict(
            calibration,
            split="train",
            seed=args.calibration_seed,
            indices=indices,
            indices_sha256=indices_sha256(indices),
        ),
        source=dict(
            checkpoint=str(args.checkpoint) if args.checkpoint else None,
            sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
        ),
        synthetic=bool(args.smoke),
    )


def restore_static(model, payload, engine, args):
    if payload.get("format") != FORMAT or payload.get("architecture") != "lenet5_mnist":
        raise ValueError("expected a LeNet static INT4 checkpoint produced by this example")
    if payload["config"] != asdict(engine.config):
        raise ValueError("simulator configuration must match the saved static checkpoint")
    if payload["conversion"] != dict(include=args.include, exclude=args.exclude):
        raise ValueError("include/exclude must match the saved static checkpoint")
    if bool(payload["synthetic"]) != bool(args.smoke):
        raise ValueError("a synthetic checkpoint is usable only with --smoke")
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


def main():
    command = parser(__doc__)
    command.set_defaults(
        batch_size=128,
        activation_bits=8,
        weight_bits=4,
        write_variation=0.02,
    )
    command.add_argument("--calibrate", action="store_true", help="observe training data and save")
    command.add_argument("--static-checkpoint", type=Path, help="packed weights and fixed scales")
    command.add_argument("--calibration-samples", type=positive, default=1024)
    command.add_argument("--calibration-seed", type=int, default=135)
    args = command.parse_args()
    setup(args)
    if args.stage != "inference" or args.digital:
        command.error("this example performs static analog inference")
    if args.calibration_seed < 0:
        command.error("calibration-seed must be nonnegative")
    explicit_static = "static_checkpoint" in args.explicit_options
    if args.smoke and args.calibrate and not explicit_static:
        command.error("--calibrate --smoke requires an explicit --static-checkpoint")
    if args.smoke and not args.calibrate and not explicit_static:
        args.static_checkpoint = None
    calibrating = args.calibrate or (args.smoke and args.static_checkpoint is None)
    if args.checkpoint is None and calibrating and not args.smoke:
        args.checkpoint = DEFAULT_SOURCE
    if args.static_checkpoint is None and not args.smoke:
        args.static_checkpoint = DEFAULT_STATIC
    if calibrating and args.static_checkpoint and args.checkpoint:
        if args.static_checkpoint.resolve() == args.checkpoint.resolve():
            command.error(
                "the static checkpoint destination must differ from the source checkpoint"
            )
    if not calibrating and (args.static_checkpoint is None or not args.static_checkpoint.is_file()):
        command.error("static checkpoint is missing; create it once with --calibrate")

    model = LeNet5().float()
    if calibrating:
        load_checkpoint(model, args.checkpoint)
    engine = simulator(args)
    if engine.config.weight_bits != 4 or not engine.config.weight_signed:
        command.error("this example stores signed 4-bit weights")
    if engine.config.bw_e is not None:
        command.error("this calibrated checkpoint uses INT scales; omit bw-e")
    model = convert_model(
        model,
        engine,
        include=args.include,
        exclude=args.exclude,
        training=False,
        inplace=True,
    )
    if calibrating:
        dataset, indices = calibration_data(args)
        # --limit applies to evaluation; calibration has its own explicit sample budget.
        from argparse import Namespace

        calibration_args = Namespace(**vars(args))
        calibration_args.limit = None
        calibration = calibrate(
            model,
            loader(dataset, calibration_args),
            activation_bits=engine.config.activation_bits,
        )
        calibration.update(
            split="train",
            seed=args.calibration_seed,
            indices_sha256=indices_sha256(indices),
        )
        if args.static_checkpoint:
            payload = static_payload(model, engine, calibration, args, indices)
            args.static_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.static_checkpoint)
    else:
        payload = torch.load(args.static_checkpoint, map_location="cpu", weights_only=True)
        calibration = restore_static(model, payload, engine, args)
    evaluation = classification_epoch(model, loader(mnist_dataset(args), args), args.device)
    report(
        args,
        dict(
            model="lenet5_mnist",
            stage="inference",
            quantization="static_signed_int4",
            static_checkpoint=str(args.static_checkpoint) if args.static_checkpoint else None,
            calibrated_now=bool(calibrating),
            online_scale_estimation=False,
            calibration=calibration,
            evaluation=evaluation,
        ),
        engine,
    )


if __name__ == "__main__":
    main()
