"""Run an Ultralytics detector and evaluate a local VOC-format task definition."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from memfast.models import load_detector

if __package__:
    from ._common import parser, positive, prepare_model, report, setup
else:
    from _common import parser, positive, prepare_model, report, setup


def validation_yaml(source, destination, *, split="test", limit=None):
    """Resolve an upstream detection dataset and optionally limit its image list."""
    import yaml
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

    data = check_det_dataset(str(source), autodownload=False)
    if len(data["names"]) != 20:
        raise ValueError("VOC validation requires a dataset definition with 20 classes")
    if not data.get(split):
        raise ValueError(f"dataset definition has no {split} split")
    paths = data[split] if isinstance(data[split], list) else [data[split]]
    images = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            images.extend(
                file.resolve()
                for file in sorted(path.rglob("*"))
                if file.is_file() and file.suffix[1:].lower() in IMG_FORMATS
            )
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    images.append(
                        (path.parent / line[2:]).resolve()
                        if line.startswith("./")
                        else Path(line).resolve()
                    )
        else:
            raise ValueError(f"expected an image directory or image-list file: {path}")
    images = sorted(set(images))
    if limit is not None:
        images = images[:limit]
    if not images or any(not image.is_file() for image in images):
        raise ValueError("selected validation images are empty or missing")
    image_list = destination / "images.txt"
    image_list.write_text("".join(str(image) + "\n" for image in images), encoding="utf-8")
    specification = {
        "path": str(data["path"]),
        "train": data.get("train") or str(image_list),
        "val": str(image_list),
        "test": str(image_list),
        "names": data["names"],
    }
    # Upstream dataset resolution may return Path objects.
    if isinstance(specification["train"], list):
        specification["train"] = [str(item) for item in specification["train"]]
    else:
        specification["train"] = str(specification["train"])
    resolved = destination / "voc.yaml"
    resolved.write_text(yaml.safe_dump(specification), encoding="utf-8")
    return resolved, len(images), data["names"]


def main():
    command = parser(__doc__)
    command.add_argument("--data-yaml", type=Path, help="local Ultralytics VOC dataset YAML")
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument(
        "--architecture", default="yolo11n.yaml", help="upstream architecture for smoke"
    )
    command.add_argument("--image-size", type=positive, default=640)
    command.add_argument("--confidence", type=float, default=0.001)
    command.add_argument("--nms-iou", type=float, default=0.7)
    command.add_argument("--run-dir", type=Path, default=Path("results/yolo"))
    command.set_defaults(batch_size=1)
    args = command.parse_args()
    setup(args)
    if args.image_size % 32:
        command.error("image-size must be a multiple of 32")
    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        command.error("confidence and nms-iou must be in [0, 1]")
    if not args.smoke and (args.checkpoint is None or args.data_yaml is None):
        command.error("VOC evaluation requires --checkpoint and --data-yaml")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        command.error("--checkpoint must be an existing local Ultralytics checkpoint")
    detector = load_detector(args.checkpoint, architecture=args.architecture)
    detector.model.float().eval()
    # Fuse upstream Conv/BatchNorm pairs before replacing their Conv2d modules.
    detector.model.fuse()
    detector.model, engine = prepare_model(detector.model, args)
    if args.smoke:
        args.image_size = 64
        image = np.random.default_rng(args.seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        outputs = detector.predict(
            source=image,
            imgsz=args.image_size,
            device=args.device,
            half=False,
            conf=args.confidence,
            iou=args.nms_iou,
            save=False,
            verbose=False,
        )
        if any(not torch.isfinite(output.boxes.data).all() for output in outputs):
            raise FloatingPointError("nonfinite detector predictions")
        report(
            args,
            dict(
                task="detection",
                untrained=args.checkpoint is None,
                images=len(outputs),
                detections=[len(output.boxes) for output in outputs],
            ),
            engine,
        )
        return
    with tempfile.TemporaryDirectory(prefix="memfast-voc-") as temporary:
        definition, count, names = validation_yaml(
            args.data_yaml, Path(temporary), split=args.split, limit=args.limit
        )
        if len(detector.names) != len(names):
            raise ValueError("checkpoint and VOC dataset class counts differ")
        checkpoint_names = (
            list(detector.names.values()) if isinstance(detector.names, dict) else detector.names
        )
        dataset_names = list(names.values()) if isinstance(names, dict) else names
        if checkpoint_names != dataset_names:
            raise ValueError("checkpoint class order must match the VOC dataset definition")
        metrics = detector.val(
            data=str(definition),
            split=args.split,
            imgsz=args.image_size,
            batch=args.batch_size,
            workers=args.workers,
            device=args.device,
            half=False,
            conf=args.confidence,
            iou=args.nms_iou,
            plots=False,
            save_json=False,
            project=str(args.run_dir),
            name="validation",
            exist_ok=True,
            verbose=False,
        )
    report(
        args,
        dict(
            task="voc_detection",
            images=count,
            metric="Ultralytics AP at IoU 0.50 and IoU 0.50:0.95",
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            per_class_map50_95=metrics.box.maps.tolist(),
        ),
        engine,
    )


if __name__ == "__main__":
    main()
