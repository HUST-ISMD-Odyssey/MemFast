# MemFast

MemFast is a functional simulator for neural-network inference on bit-sliced
analog in-memory computing arrays. It connects device and converter parameters
to matrix outputs and network accuracy through a PyTorch interface.

The simulation follows quantization, binary weight programming, bit-serial
array currents, per-slice ADC conversion, and signed digital reconstruction.
Weights remain compact between reads; CUDA kernels restore and consume arrays
in bounded groups.

## Capabilities

- Signed 2-8 bit activations and weights with one-bit slices.
- Configurable rectangular arrays, quantization groups, and ADC resolution.
- Programming and read variation, voltage noise, current output noise,
  S-shaped ADC nonlinearity, stuck-at faults, and conductance drift.
- Matrix operations and conversion of PyTorch `Linear` and `Conv2d` layers.
- CPU execution and accelerated CUDA execution.
- Fixed default execution or exhaustive schedule tuning with a persistent cache.

MemFast models array computation and inference accuracy. Circuit timing, energy,
area, and interconnect parasitics require separate models.

## Installation

Install Python 3.10 or newer and a PyTorch build for your CPU or CUDA platform.
Then install MemFast:

```bash
git clone https://github.com/HUST-ISMD-Odyssey/MemFast.git
cd MemFast
python -m pip install .
```

For the CIFAR-10 example, install the vision extra:

```bash
python -m pip install ".[examples]"
```

CUDA execution compiles the included kernels with NVIDIA NVRTC on first use.
The CUDA-enabled PyTorch distribution or CUDA Toolkit must provide NVRTC.

## A First Simulation

```python
import torch
from memfast import Config, Nonidealities, Simulator

config = Config(
    array_size=(64, 64),
    weight_quant_gran=(64, 1),
    nonidealities=Nonidealities(write_variation=0.02),
)
simulator = Simulator(config, device="cpu")
array = simulator.map(torch.randn(32, 128))
output = array(torch.randn(4, 128))
```

`array_size` defines physical parallelism. `weight_quant_gran` defines which
weights share a quantization scale. The example uses a separate scale for each
64-row group in each output column, while the array computes 64 columns in
parallel.

## Network Simulation

```python
from memfast import convert_model

analog_model = convert_model(model.eval(), simulator)
output = analog_model(inputs)
```

Conversion returns a copy by default. Biases, normalization, nonlinear
activations, pooling, attention, and other operations retain their PyTorch
behavior. Select layers can remain digital with `exclude=("classifier",)`.

## Execution Modes

| Mode | Behavior |
| --- | --- |
| `default` | Execute immediately with fixed scheduling parameters. |
| `tuned` | Traverse the declared schedule grid for each new shape and select the fastest measured feasible schedule. |

```python
simulator = Simulator(config, mode="tuned", device="cuda:0")
```

Tuning checks outputs against default execution before timing each candidate
once. Matching cached selections skip traversal on later runs and receive an
untimed output check. Cache keys include geometry, quantization, nonidealities,
input shape, software, GPU properties, and implementation identity.

## Documentation And Examples

- [User guide](docs/user-guide.md): configuration, physical models, sampling,
  tuning, and reproducibility.
- [Examples](examples/README.md): matrix operations, model conversion,
  individual nonidealities, and pretrained VGG11-BN on CIFAR-10.
- [Parameter scans](examples/results/README.md): measured CIFAR-10 results and
  their configurations.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

CUDA tests run when a CUDA device is available.

## License

MemFast is released under the [MIT License](LICENSE). Example datasets and
pretrained weights retain their original licenses.
