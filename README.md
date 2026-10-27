# MemFast

MemFast is a functional simulator for neural-network inference on bit-sliced
analog in-memory computing arrays. It connects device and converter parameters
to matrix outputs and network accuracy through a PyTorch interface.

The simulation follows quantization, sliced weight programming, sliced
array currents, per-slice ADC conversion, and signed digital reconstruction.
Weights remain compact between reads; CUDA kernels restore and consume arrays
in bounded groups.

## Capabilities

- Signed INT and block floating-point (BFP) quantization, including 10-, 16-,
  and 20-bit configurations.
- Independent input and weight slice layouts, including multibit cells and
  input levels such as `(1,1,2,2,2)` for eight-bit data.
- Independent physical array geometry, rows per ADC read, quantization groups,
  and uniform or per-weight-slice ADC resolution.
- Programming and read variation, voltage noise, current output noise,
  S-shaped ADC nonlinearity, stuck-at faults, and conductance drift.
- Matrix operations, neural-network layers, model conversion, and
  hardware-aware training.
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

PyTorch and torchvision are included in the standard dependencies. Task-specific
integrations for additional model libraries are described with their examples.

CUDA execution compiles the included kernels with NVIDIA NVRTC on first use.
The CUDA-enabled PyTorch distribution or CUDA Toolkit must provide NVRTC.

## A First Simulation

```python
import torch
from memfast import Config, Nonidealities, Simulator

config = Config(
    array_size=(128, 128),
    rows_per_read=32,
    weight_quant_gran=(32, 1),
    input_quant_gran=(1, 32),
    nonidealities=Nonidealities(write_variation=0.02),
)
simulator = Simulator(config, device="cpu")
array = simulator.map(torch.randn(32, 128))
output = array(torch.randn(4, 128))
```

`array_size` defines the physical array. `rows_per_read` defines how many rows
are summed before each ADC conversion. Quantization groups define scale
sharing. Each ADC partial is reconstructed and multiplied by its own input and
weight scales before digital accumulation.

Set `bw_e=None` for INT or an exponent width such as `bw_e=8` for BFP.
Slice layouts are ordered from the sign slice to the least significant slice;
`input_slice=(1,1,2,2,2)` and `weight_slice=(1,1,2,2,2)` each represent eight bits.
Use `adc_bits=7` for one resolution throughout, or
`adc_bits=(6,7,7,8,8)` for the five weight slices in that same order.
The [user guide](docs/user-guide.md) defines mapping, exponent ranges, and
accumulator limits.

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

Tuning checks outputs against an admitted reference schedule before timing each candidate
once. Matching cached selections skip traversal on later runs and receive an
untimed output check. Cache keys include geometry, quantization, nonidealities,
input shape, software, GPU properties, and implementation identity.

## Documentation And Examples

- [User guide](docs/user-guide.md): configuration, physical models, sampling,
  tuning, and reproducibility.
- [Examples](examples/README.md): matrix operations, model conversion,
  training, vision models, recurrent networks, and language-model evaluation.
- [Parameter scans](examples/results/README.md): measured CIFAR-10 results and
  their configurations.
- [Validation](docs/validation.md): numerical coverage, workflow checks, and
  real-data evaluation scope.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

CUDA tests run when a CUDA device is available.

## Package Layout

`memfast` contains the physical configuration and numerical engine.
`memfast.nn` provides network layers, conversion, and training;
`memfast.models` provides model construction. Task workflows live in `examples`.

## License

MemFast is released under the [MIT License](LICENSE). Example datasets and
pretrained weights retain their original licenses.
