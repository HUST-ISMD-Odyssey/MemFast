"""Define a three-layer MLP, train it, restore its weights and run CIM."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from memfast import Config, Nonidealities, Simulator, convert_model

if __package__:
    from .common import evaluate, load_checkpoint, save_checkpoint, train
else:
    from common import evaluate, load_checkpoint, save_checkpoint, train

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/mnist")
CHECKPOINT = Path(__file__).resolve().parent / "output/mlp_mnist.pt"
HAT_CHECKPOINT = CHECKPOINT.with_name("mlp_mnist_hat.pt")
HIDDEN_SIZES = (512, 128)
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.001
INCLUDE = None  # ("fc1", "fc2") keeps fc3 in software.
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


class ThreeLayerMLP(nn.Module):
    def __init__(self, hidden_sizes=(512, 128), input_features=784, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(self.flatten(x)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main(stage="train"):
    torch.manual_seed(135)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_data = datasets.MNIST(DATA_ROOT, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = ThreeLayerMLP(HIDDEN_SIZES).float().to(DEVICE)
    if stage != "train":
        load_checkpoint(model, CHECKPOINT)
    state_keys = tuple(model.state_dict())
    if stage in ("inference", "hat"):
        simulator = Simulator(CONFIG, device=DEVICE)
        # All eligible layers by default; change INCLUDE or EXCLUDE above.
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
            HAT_CHECKPOINT if stage == "hat" else CHECKPOINT,
            state_keys=state_keys,
            hidden_sizes=HIDDEN_SIZES,
        )
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    command = argparse.ArgumentParser(__doc__)
    command.add_argument("--stage", choices=("train", "inference", "hat"), default="train")
    main(command.parse_args().stage)
