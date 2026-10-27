"""Evaluate pretrained VGG11-BN and scan device parameters on CIFAR-10."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from memfast import Config, Nonidealities, Simulator, convert_model

WEIGHTS_URL = (
    "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/"
    "vgg/cifar10_vgg11_bn-eaeebf42.pt"
)
SCANS = {
    "write_variation": [0, 0.02, 0.05, 0.08],
    "read_shared": [0, 0.02, 0.05, 0.08],
    "read_slice_pair": [0, 0.02, 0.05, 0.08],
    "read_output": [0, 0.02, 0.05, 0.08],
    "voltage_variation": [0, 0.02, 0.05, 0.08],
    "output_noise": [0, 0.001, 0.005, 0.01],
    "adc_nonlinearity": [0, 0.02, 0.05, 0.08],
    "stuck_at_low": [0, 0.001, 0.005, 0.01],
    "stuck_at_high": [0, 0.001, 0.005, 0.01],
    "drift_time": [0, 1, 100, 10000],
}


def vgg11():
    layers, channels = [], 3
    for width in (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"):
        if width == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.extend(
                [
                    nn.Conv2d(channels, width, 3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = width
    model = nn.Sequential()
    model.add_module("features", nn.Sequential(*layers))
    model.add_module("flatten", nn.Flatten())
    model.add_module(
        "classifier",
        nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 10),
        ),
    )
    return model


def noise_for(scan, value):
    if scan.startswith("read_"):
        return Nonidealities(read_variation=value, read_mode=scan.removeprefix("read_"))
    if scan == "drift_time":
        return Nonidealities(drift_coefficient=0.05, drift_time=value)
    return Nonidealities(**{scan: value})


def evaluate(model, loader, device, output_file=None):
    correct, count, loss_sum = 0, 0, 0.0
    logits_hash = hashlib.sha256()
    outputs, targets = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for x, y in loader:
            output = model(x.to(device)).float()
            if not torch.isfinite(output).all():
                raise FloatingPointError("nonfinite model output")
            cpu_output = output.cpu().contiguous()
            logits_hash.update(cpu_output.numpy().tobytes())
            if output_file is not None:
                outputs.append(cpu_output)
                targets.append(y.clone())
            correct += int((output.argmax(1).cpu() == y).sum())
            count += len(y)
            loss_sum += float(nn.functional.cross_entropy(output, y.to(device), reduction="sum"))
    if output_file is not None:
        torch.save(dict(logits=torch.cat(outputs), targets=torch.cat(targets)), output_file)
    return dict(
        samples=count,
        correct=correct,
        top1=100 * correct / count,
        cross_entropy=loss_sum / count,
        seconds=time.perf_counter() - start,
        logits_sha256=logits_hash.hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("default", "tuned"), default="default")
    parser.add_argument("--cache-dir")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--scan", choices=("all", *SCANS), default=None)
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument("--output", type=Path, default=Path("results/cifar10"))
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="save logits and targets for independent result verification",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 10000 or args.batch_size < 1:
        parser.error("limit must be 1..10000 and batch-size must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = datasets.CIFAR10(args.data, train=False, transform=transform, download=args.download)
    loader = DataLoader(
        Subset(dataset, range(args.limit)), batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = vgg11().eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint = args.checkpoint
    else:
        state = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, map_location="cpu", check_hash=True, weights_only=True
        )
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_URL.rsplit("/", 1)[1]
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.checkpoint is None and not checkpoint_sha256.startswith("eaeebf42"):
        raise ValueError("pretrained checkpoint checksum mismatch")
    model.load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    for key, tensor in state.items():
        digest.update(key.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        model="VGG11-BN",
        dataset="CIFAR-10 test",
        weights_url=WEIGHTS_URL,
        state_sha256=digest.hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
        samples=args.limit,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.mode,
        torch=torch.__version__,
    )
    digital = evaluate(
        model.to(args.device),
        loader,
        args.device,
        args.output / "digital.pt" if args.save_outputs else None,
    )
    model.cpu()
    rows = [dict(effect="digital", value=0, **digital)]
    scans = list(SCANS) if args.scan == "all" else ([args.scan] if args.scan else [])
    settings = [(scan, value) for scan in scans for value in SCANS[scan]]
    if not settings:
        settings = [("write_variation", 0)]
    zero_result, zero_output_file = None, None
    for effect, value in settings:
        config = replace(Config(), seed=args.seed, nonidealities=noise_for(effect, value))
        output_file = args.output / f"{effect}_{value:g}.pt" if args.save_outputs else None
        if value == 0 and zero_result is not None:
            result = dict(zero_result)
            if output_file is not None:
                shutil.copyfile(zero_output_file, output_file)
        else:
            sim = Simulator(config, device=args.device, mode=args.mode, cache_dir=args.cache_dir)
            analog = convert_model(model, sim)
            result = evaluate(
                analog,
                loader,
                args.device,
                output_file,
            )
            del analog
            if value == 0:
                zero_result = dict(result)
                zero_output_file = output_file
        rows.append(dict(effect=effect, value=value, **result))
        metadata.setdefault("configurations", {})[f"{effect}:{value}"] = asdict(config)
        fields = list(rows[0])
        with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(rows[-1]), flush=True)
    text = [
        "# VGG11-BN on CIFAR-10",
        "",
        f"Test samples: {args.limit}; batch size: {args.batch_size}; seed: {args.seed}.",
        "",
        f"Digital top-1: {digital['top1']:.2f}%.",
        "A8W8, 64 x 64 arrays, ADC6, per-channel weights, per-vector inputs.",
        "Each scan changes one effect; other nonidealities are zero.",
        "Drift uses exponent 0.05 and reference time 1 second.",
        "",
    ]
    for effect in scans or ["write_variation"]:
        text += [
            f"## {effect}",
            "",
            "| Value | Top-1 (%) | Cross entropy |",
            "| ---: | ---: | ---: |",
        ]
        for row in rows:
            if row["effect"] == effect:
                text.append(
                    f"| {row['value']:g} | {row['top1']:.2f} | {row['cross_entropy']:.6f} |"
                )
        text.append("")
    (args.output / "tables.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
