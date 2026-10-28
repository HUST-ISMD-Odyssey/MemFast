"""Adapt a trained CIFAR VGG with analog forward and FP32 backward."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from memfast import Config, Nonidealities, Simulator, convert_model
from memfast.models import CifarVGG

if __package__:
    from .common import evaluate, load_checkpoint, save_checkpoint, train
else:
    from common import evaluate, load_checkpoint, save_checkpoint, train

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data/cifar10")
CHECKPOINT = Path(torch.hub.get_dir()) / "checkpoints/cifar10_vgg11_bn-eaeebf42.pt"
SAVE_CHECKPOINT = Path(__file__).resolve().parent / "output/vgg11_bn_cifar10_hat.pt"
MODEL_NAME = "vgg11_bn"
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.0001
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
    torch.manual_seed(135)
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalization])
    train_data = datasets.CIFAR10(DATA_ROOT, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(DATA_ROOT, train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = CifarVGG(MODEL_NAME, num_classes=NUM_CLASSES).float().to(DEVICE)
    load_checkpoint(model, CHECKPOINT)
    state_keys = tuple(model.state_dict())
    simulator = Simulator(CONFIG, device=DEVICE)
    model = convert_model(
        model,
        simulator,
        include=INCLUDE,
        exclude=EXCLUDE,
        training=True,
        gradient_mode="quantized",
        inplace=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, train_loader, optimizer, device=DEVICE, epochs=EPOCHS)
    save_checkpoint(
        model,
        SAVE_CHECKPOINT,
        state_keys=state_keys,
        architecture=MODEL_NAME,
        num_classes=NUM_CLASSES,
    )
    return evaluate(model, test_loader, device=DEVICE)


if __name__ == "__main__":
    main()
