"""Optional Ultralytics detector loading without a bundled detection engine."""

from pathlib import Path


def load_detector(checkpoint=None, *, architecture="yolo11n.yaml"):
    """Return an upstream YOLO wrapper using local weights or an untrained YAML.

    Supplying an existing checkpoint is explicit. With no checkpoint, the YAML
    architecture is initialized with random weights for synthetic smoke runs.
    """
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"expected a local detector checkpoint: {checkpoint}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("detection models require the optional ultralytics package") from exc
    return YOLO(str(checkpoint) if checkpoint is not None else architecture, task="detect")
