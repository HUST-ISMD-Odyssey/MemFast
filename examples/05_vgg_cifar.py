"""Load a custom CIFAR VGG checkpoint and evaluate CIM inference."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from memfast import Config, Nonidealities, Simulator, convert_model
from memfast.models import CifarVGG

if __package__:
    from .common import evaluate, load_checkpoint
else:
    from common import evaluate, load_checkpoint

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/cifar10")
CHECKPOINT = Path(torch.hub.get_dir()) / "checkpoints/cifar10_vgg11_bn-eaeebf42.pt"
MODEL_NAME = "vgg11_bn"
NUM_CLASSES = 10
BATCH_SIZE = 128
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


def main():
    # Use CIFAR100 and its checkpoint/normalization together for CIFAR-100.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_data = datasets.CIFAR10(DATA_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = CifarVGG(MODEL_NAME, num_classes=NUM_CLASSES).float().to(DEVICE)
    load_checkpoint(model, CHECKPOINT)
    simulator = Simulator(CONFIG, device=DEVICE)
    model = convert_model(model, simulator, include=INCLUDE, exclude=EXCLUDE, inplace=True)
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    main()
