"""Shared command-line options, checkpoint handling, and classification loops."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset

from memfast import Config, Nonidealities, Simulator, convert_model


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def group_pair(value):
    try:
        result = tuple(
            None if item.lower() == "none" else positive(item) for item in value.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected two comma-separated integers or none") from exc
    if len(result) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers or none")
    return result


def bit_slices(value):
    try:
        result = tuple(positive(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive slice widths") from exc
    return result


def adc_bits(value):
    widths = bit_slices(value)
    return widths if "," in value else widths[0]


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--device", default="cpu")
    result.add_argument("--digital", action="store_true", help="use the original PyTorch model")
    result.add_argument("--smoke", action="store_true", help="use a small synthetic task")
    result.add_argument("--data-root", type=Path, default=Path("data"))
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output", type=Path, help="write the actual result as JSON")
    result.add_argument("--download", action="store_true")
    result.add_argument("--limit", type=positive, help="maximum examples per dataset split")
    result.add_argument("--batch-size", type=positive, default=32)
    result.add_argument("--workers", type=int, default=0)
    result.add_argument("--threads", type=positive, default=2)
    result.add_argument("--seed", type=int, default=135)
    result.add_argument("--mode", choices=("default", "tuned"), default="default")
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--activation-bits", type=positive, default=8)
    result.add_argument("--weight-bits", type=positive, default=8)
    result.add_argument("--input-slice", type=bit_slices)
    result.add_argument("--weight-slice", type=bit_slices)
    result.add_argument("--bw-e", type=positive, help="BFP exponent bits; omit for INT")
    result.add_argument("--array-size", type=group_pair, default=(64, 64))
    result.add_argument("--rows-per-read", type=positive)
    result.add_argument("--input-quant-gran", type=group_pair, default=(1, None))
    result.add_argument("--weight-quant-gran", type=group_pair, default=(None, 1))
    result.add_argument(
        "--adc-bits",
        type=adc_bits,
        default=6,
        help="one width, or comma-separated widths per weight slice (MSB first)",
    )
    result.add_argument("--write-variation", type=float, default=0.0)
    return result


def setup(args):
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.smoke:
        args.batch_size = min(args.batch_size, 2)
        args.limit = min(args.limit or 4, 4)


def simulator(args):
    options = dict(
        activation_bits=args.activation_bits,
        weight_bits=args.weight_bits,
        array_size=args.array_size,
        rows_per_read=args.rows_per_read,
        input_quant_gran=args.input_quant_gran,
        weight_quant_gran=args.weight_quant_gran,
        adc_bits=args.adc_bits,
        seed=args.seed,
        nonidealities=Nonidealities(write_variation=args.write_variation),
    )
    for name in ("input_slice", "weight_slice", "bw_e"):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    return Simulator(
        Config(**options), device=args.device, mode=args.mode, cache_dir=args.cache_dir
    )


def prepare_model(model, args, *, training=False, exclude=()):
    model = model.to(args.device)
    if args.digital:
        return model, None
    engine = simulator(args)
    options = dict(exclude=exclude, inplace=True)
    if training:
        options["training"] = True
        options["gradient_mode"] = getattr(args, "gradient_mode", "quantized")
    return convert_model(model, engine, **options), engine


def load_checkpoint(model, checkpoint):
    if checkpoint is None:
        return {}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    state = payload.get(
        "state_dict", payload.get("model_state_dict", payload.get("model", payload))
    )
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint state_dict must map module paths to tensors")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return (
        {key: value for key, value in payload.items() if key != "state_dict"}
        if ("state_dict" in payload)
        else {}
    )


def synthetic(shape, num_classes, count, seed):
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(count, *shape, generator=generator),
        torch.randint(num_classes, (count,), generator=generator),
    )


def loader(dataset, args, *, training=False, rank=0, world_size=1):
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    if not len(dataset):
        raise ValueError("dataset is empty")
    sampler = None
    if world_size > 1:
        if training:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
            )
        else:
            # Evaluation visits each sample once, including uneven final shards.
            dataset = Subset(dataset, range(rank, len(dataset), world_size))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed + rank),
    )


def classification_epoch(model, batches, device, *, optimizer=None, distributed=False):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    started = time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for x, targets in batches:
            x, targets = x.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.ndim != 2 or not torch.isfinite(logits).all():
                raise FloatingPointError("expected finite [batch, classes] logits")
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite classification loss")
            if training:
                loss.backward()
                optimizer.step()
            count = targets.numel()
            totals[0] += loss.detach().double() * count
            totals[1] += (logits.argmax(1) == targets).sum()
            totals[2] += (
                (logits.topk(min(5, logits.shape[1]), dim=1).indices == targets[:, None])
                .any(1)
                .sum()
            )
            totals[3] += count
    if distributed:
        dist.all_reduce(totals)
    loss_sum, correct, top5, samples = totals.tolist()
    if not samples:
        raise ValueError("evaluation has no samples")
    return dict(
        samples=int(samples),
        correct=int(correct),
        top1=correct / samples,
        top5=top5 / samples,
        cross_entropy=loss_sum / samples,
        seconds=time.perf_counter() - started,
    )


def training_options(result):
    result.add_argument("--epochs", type=positive, default=1)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--save-checkpoint", type=Path)
    result.add_argument("--gradient-mode", choices=("quantized", "digital"), default="quantized")


def optimizer_for(model, args):
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")
    return torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def save_checkpoint(model, path, *, state_keys=None, **metadata):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if state_keys is not None:
        # HAT master parameters use the original digital module paths.
        state = {key: state[key] for key in state_keys}
    state = {key: value.detach().cpu() for key, value in state.items()}
    torch.save(dict(state_dict=state, **metadata), path)


def report(args, result, engine=None):
    result = dict(
        **result,
        synthetic=bool(args.smoke),
        seed=args.seed,
        device=str(args.device),
        torch=str(torch.__version__),
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        execution="digital" if args.digital else "analog",
    )
    if engine is not None:
        result["config"] = asdict(engine.config)
        result["simulator"] = engine.report()
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return result
