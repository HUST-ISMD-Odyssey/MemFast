"""Ultralytics YOLO on local VOC data with visible MemFast model conversion."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from memfast import convert_model
from memfast.models import load_detector

if __package__:
    from .common import (
        load_checkpoint,
        parser,
        positive,
        report,
        setup,
        simulator,
    )
else:
    from common import (
        load_checkpoint,
        parser,
        positive,
        report,
        setup,
        simulator,
    )


def voc_yaml(source, destination, *, split, limit):
    """Resolve local upstream data and select a deterministic image list."""
    import yaml
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

    data = check_det_dataset(str(source), autodownload=False)
    if len(data["names"]) != 20:
        raise ValueError("VOC dataset must define exactly 20 ordered classes")
    if not data.get(split):
        raise ValueError(f"VOC dataset has no {split} split")
    paths = data[split] if isinstance(data[split], list) else [data[split]]
    images = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            images.extend(
                image.resolve()
                for image in path.rglob("*")
                if image.is_file() and image.suffix[1:].lower() in IMG_FORMATS
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
            raise ValueError("VOC splits must be image directories or image-list files")
    images = sorted(set(images))
    images = images[:limit] if limit else images
    if not images or any(not image.is_file() for image in images):
        raise ValueError("VOC split is empty or contains missing images")
    image_list = destination / f"{split}_images.txt"
    image_list.write_text("".join(str(image) + "\n" for image in images), encoding="utf-8")
    training = data.get("train") or str(image_list)
    if split == "train":
        training = str(image_list)
    training = [str(value) for value in training] if isinstance(training, list) else str(training)
    definition = destination / f"{split}.yaml"
    definition.write_text(
        yaml.safe_dump(
            dict(
                path=str(data["path"]),
                train=training,
                val=str(image_list),
                test=str(image_list),
                names=data["names"],
            )
        ),
        encoding="utf-8",
    )
    return definition, len(images), data["names"]


def main():
    command = parser(__doc__)
    command.add_argument("--architecture", default="yolo11n.yaml")
    command.add_argument(
        "--state-checkpoint", type=Path, help="saved fused-model FP32 master state"
    )
    command.add_argument("--data-yaml", type=Path)
    command.add_argument("--split", choices=("val", "test"), default="test")
    command.add_argument("--image-size", type=positive, default=640)
    command.add_argument("--confidence", type=float, default=0.001)
    command.add_argument("--nms-iou", type=float, default=0.7)
    command.add_argument("--run-dir", type=Path, default=Path("examples/output/yolo"))
    command.set_defaults(batch_size=1)
    args = command.parse_args()
    setup(args)
    if args.stage != "inference":
        command.error("YOLO example supports inference only")
    if args.image_size % 32 or not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        command.error("image-size must be divisible by 32; confidence and nms-iou must be in [0,1]")
    if not args.smoke and (args.checkpoint is None or args.data_yaml is None):
        command.error("VOC requires a local --checkpoint and --data-yaml")
    if args.smoke:
        args.image_size = 64
    detector = load_detector(args.checkpoint, architecture=args.architecture)
    model = detector.model.float().eval()
    # Fuse in FP32 before conversion, matching the upstream inference graph.
    model.fuse()
    load_checkpoint(model, args.state_checkpoint)
    model.to(args.device)
    engine = None
    if not args.digital:
        engine = simulator(args)
        model = convert_model(
            model,
            engine,
            include=args.include,
            exclude=args.exclude,
            training=args.stage == "hat",
            inplace=True,
        )
    detector.model = model
    with tempfile.TemporaryDirectory(prefix="memfast-voc-") as directory:
        directory = Path(directory)
        model.eval()
        if args.smoke:
            image = np.random.default_rng(args.seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            predictions = detector.predict(
                source=image,
                imgsz=64,
                device=args.device,
                half=False,
                conf=args.confidence,
                iou=args.nms_iou,
                save=False,
                verbose=False,
            )
            if any(not torch.isfinite(item.boxes.data).all() for item in predictions):
                raise FloatingPointError("nonfinite detector predictions")
            result = dict(
                images=len(predictions), detections=[len(item.boxes) for item in predictions]
            )
        else:
            definition, count, names = voc_yaml(
                args.data_yaml, directory, split=args.split, limit=args.limit
            )
            expected = list(names.values()) if isinstance(names, dict) else list(names)
            actual = detector.names
            actual = list(actual.values()) if isinstance(actual, dict) else list(actual)
            if actual != expected:
                raise ValueError("checkpoint classes and order must match the VOC dataset")
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
            result = dict(
                images=count,
                metric="Ultralytics AP50 and AP50:95",
                map50=float(metrics.box.map50),
                map50_95=float(metrics.box.map),
                per_class_map50_95=metrics.box.maps.tolist(),
            )
    report(
        args,
        dict(
            task="voc_detection",
            model="yolo",
            stage=args.stage,
            fused_model=True,
            evaluation=result,
        ),
        engine,
    )


if __name__ == "__main__":
    main()
