"""Compare a CIM matrix product with torch.mm."""

import torch

from memfast import Config, Nonidealities, Simulator

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONFIG = Config(
    activation_bits=8,
    weight_bits=8,
    input_slice=(1,) * 8,
    weight_slice=(1,) * 8,
    bw_e=None,  # INT; use an exponent width for BFP.
    array_size=(128, 128),
    rows_per_read=32,
    input_quant_gran=(1, 32),
    weight_quant_gran=(32, 1),
    adc_bits=6,  # Or a tuple with one width per weight slice.
    nonidealities=Nonidealities(write_variation=0.02),
)
EXECUTION_MODE = "default"  # "tuned" searches once and caches the schedule.


def main():
    torch.manual_seed(135)
    a = torch.randn(8, 128, device=DEVICE)
    b = torch.randn(128, 80, device=DEVICE)
    simulator = Simulator(CONFIG, device=DEVICE, mode=EXECUTION_MODE)
    expected = torch.mm(a, b)
    result = simulator.mm(a, b)
    relative_error = (result - expected).norm() / expected.norm().clamp_min(1e-12)
    print(f"Output shape: {tuple(result.shape)}")
    print(f"Relative L2 error: {relative_error.item():.6f}")
    return result


if __name__ == "__main__":
    main()
