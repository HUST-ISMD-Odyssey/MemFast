"""Program and read a matrix without downloading data."""

import argparse
import json

import torch

from memfast import Config, Nonidealities, Simulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("default", "tuned"), default="default")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    torch.manual_seed(7)
    config = Config(weight_quant_gran=(64, 1), nonidealities=Nonidealities(write_variation=0.02))
    simulator = Simulator(config, mode=args.mode, device=args.device, cache_dir=args.cache_dir)
    weight = torch.randn(80, 128, device=simulator.device)
    x = torch.randn(8, 128, device=simulator.device)
    array = simulator.map(weight, name="example")
    result = array(x)
    assert result.shape == (8, 80) and torch.isfinite(result).all()
    array.reset_read()
    assert torch.equal(result, array(x))
    error = (result - x @ weight.T).norm() / (x @ weight.T).norm()
    print(
        json.dumps(
            dict(shape=list(result.shape), relative_l2=float(error), execution=simulator.report()),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
