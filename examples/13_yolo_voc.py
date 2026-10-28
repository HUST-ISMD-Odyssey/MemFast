"""Ultralytics YOLO on local VOC data with visible MemFast model conversion."""

import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

from memfast import Config, Nonidealities, Simulator, convert_model

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHECKPOINT = Path("models/yolo_voc.pt")  # A detector trained for VOC's 20 classes.
DATA_YAML = Path("data/voc.yaml")
RUN_DIR = Path(__file__).resolve().parent / "output/yolo"
SPLIT = "test"
IMAGE_SIZE = 640
BATCH_SIZE = 1
INCLUDE = None
EXCLUDE = ()
CONFIG = Config(
    activation_bits=8,
    weight_bits=8,
    input_slice=(1,) * 8,
    weight_slice=(1,) * 8,
    array_size=(64, 64),
    rows_per_read=64,
    input_quant_gran=(1, None),
    weight_quant_gran=(None, 1),
    adc_bits=6,
    bw_e=None,
    nonidealities=Nonidealities(write_variation=0.02),
)


def voc_yaml(source, destination, *, split):
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
    torch.manual_seed(135)
    detector = YOLO(str(CHECKPOINT), task="detect")
    model = detector.model.float().eval()
    # Fuse in FP32 before conversion, matching the upstream inference graph.
    model.fuse()
    model.to(DEVICE)
    simulator = Simulator(CONFIG, device=DEVICE)
    model = convert_model(model, simulator, include=INCLUDE, exclude=EXCLUDE, inplace=True)
    detector.model = model
    with tempfile.TemporaryDirectory(prefix="memfast-voc-") as directory:
        directory = Path(directory)
        model.eval()
        definition, count, names = voc_yaml(DATA_YAML, directory, split=SPLIT)
        expected = list(names.values()) if isinstance(names, dict) else list(names)
        actual = detector.names
        actual = list(actual.values()) if isinstance(actual, dict) else list(actual)
        if actual != expected:
            raise ValueError("checkpoint classes and order must match the VOC dataset")
        metrics = detector.val(
            data=str(definition),
            split=SPLIT,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=0,
            device=DEVICE,
            half=False,
            conf=0.001,
            iou=0.7,
            plots=False,
            save_json=False,
            project=str(RUN_DIR),
            name="validation",
            exist_ok=True,
            verbose=False,
        )
        result = dict(
            images=count,
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            per_class_map50_95=metrics.box.maps.tolist(),
        )
    print(result)
    return result


if __name__ == "__main__":
    main()
