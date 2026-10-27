"""Convert a small PyTorch network without downloading data."""

import argparse

import torch
from torch import nn

from memfast import Config, Simulator, convert_model, reset_read_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("default", "tuned"), default="default")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    ).eval()
    simulator = Simulator(Config(), device=args.device, mode=args.mode, cache_dir=args.cache_dir)
    analog = convert_model(model, simulator)
    x = torch.randn(2, 3, 8, 8, device=simulator.device)
    result = analog(x)
    assert result.shape == (2, 2) and torch.isfinite(result).all()
    reset_read_events(analog)
    assert torch.equal(result, analog(x))
    print(result.cpu())
    print("Converted Conv2d and Linear: passed")


if __name__ == "__main__":
    main()
