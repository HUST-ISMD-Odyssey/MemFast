"""LeNet-5 digital training, saved-weight inference and FP32 HAT."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from memfast import Config, Nonidealities, Simulator, convert_model
from memfast.models import LeNet5

if __package__:
    from .common import evaluate, load_checkpoint, save_checkpoint, train
else:
    from common import evaluate, load_checkpoint, save_checkpoint, train

EXAMPLES = Path(__file__).resolve().parent
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/mnist")
CHECKPOINT = EXAMPLES / "checkpoints/lenet5_mnist.pt"
TRAIN_CHECKPOINT = EXAMPLES / "output/lenet5_mnist.pt"
HAT_CHECKPOINT = EXAMPLES / "output/lenet5_mnist_hat.pt"
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001
INCLUDE = None
# To convert only the first four eligible layers:
# INCLUDE = ("features.0", "features.3", "classifier.0", "classifier.2")
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


def main(stage="inference"):
    torch.manual_seed(135)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_data = datasets.MNIST(DATA_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = LeNet5().float().to(DEVICE)
    if stage != "train":
        load_checkpoint(model, CHECKPOINT)
    state_keys = tuple(model.state_dict())
    if stage in ("inference", "hat"):
        simulator = Simulator(CONFIG, device=DEVICE)
        model = convert_model(
            model,
            simulator,
            include=INCLUDE,
            exclude=EXCLUDE,
            training=stage == "hat",
            inplace=True,
        )
    if stage in ("train", "hat"):
        train_data = datasets.MNIST(DATA_ROOT, train=True, transform=transform, download=True)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(model, train_loader, optimizer, device=DEVICE, epochs=EPOCHS)
        save_checkpoint(
            model,
            HAT_CHECKPOINT if stage == "hat" else TRAIN_CHECKPOINT,
            state_keys=state_keys,
            architecture="lenet5_mnist",
            normalization=dict(mean=[0.5], std=[0.5]),
        )
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    command = argparse.ArgumentParser(__doc__)
    command.add_argument("--stage", choices=("train", "inference", "hat"), default="inference")
    main(command.parse_args().stage)
