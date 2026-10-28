"""Load pretrained MobileNetV2 and select its CIM conversion boundaries."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models

from memfast import Config, Nonidealities, Simulator, convert_model

if __package__:
    from .common import evaluate, load_checkpoint
else:
    from common import evaluate, load_checkpoint

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/imagenet/val")
MODEL_NAME = "mobilenet_v2"
WEIGHTS = "DEFAULT"
CHECKPOINT = None
BATCH_SIZE = 32
INCLUDE = None
EXCLUDE = ()  # ("features.0", "classifier") keeps both digital.
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


def main():
    weights = models.get_model_weights(MODEL_NAME)[WEIGHTS]
    model = (
        models.get_model(
            MODEL_NAME,
            weights=weights if CHECKPOINT is None else None,
        )
        .float()
        .to(DEVICE)
    )
    if CHECKPOINT is not None:
        load_checkpoint(model, CHECKPOINT)
    test_data = datasets.ImageFolder(DATA_ROOT, transform=weights.transforms())
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    simulator = Simulator(CONFIG, device=DEVICE)
    model = convert_model(model, simulator, include=INCLUDE, exclude=EXCLUDE, inplace=True)
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    main()
