"""One independent matrix example for each nonideality."""

import argparse
import json
from dataclasses import replace

import torch

from memfast import Config, Nonidealities, Simulator

CASES = {
    "ideal": {},
    "write": {"write_variation": 0.05},
    "read_shared": {"read_variation": 0.05, "read_mode": "shared"},
    "read_slice_pair": {"read_variation": 0.05, "read_mode": "slice_pair"},
    "read_output": {"read_variation": 0.05, "read_mode": "output"},
    "voltage": {"voltage_variation": 0.05},
    "output": {"output_noise": 0.01},
    "adc": {"adc_nonlinearity": 0.05},
    "stuck_low": {"stuck_at_low": 0.01},
    "stuck_high": {"stuck_at_high": 0.01},
    "drift": {"drift_coefficient": 0.05, "drift_time": 100.0},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    weight, x = torch.randn(32, 64), torch.randn(4, 64)
    baseline = None
    rows = []
    for name, values in CASES.items():
        config = replace(Config(), nonidealities=Nonidealities(**values))
        sim = Simulator(config, device=args.device)
        array = sim.map(weight, name="nonidealities")
        result = array(x.to(sim.device)).cpu()
        assert torch.isfinite(result).all()
        array.reset_read()
        assert torch.equal(result, array(x.to(sim.device)).cpu())
        if baseline is None:
            baseline = result
        rows.append(
            dict(case=name, relative_l2=float((result - baseline).norm() / baseline.norm()))
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
