# Project Layout

The public API offers matrix multiplication, PyTorch-style layers,
and complete network conversion. Model and dataset choices belong in examples;
device and quantization behavior belongs in the simulator.

| Path | Responsibility |
| --- | --- |
| `memfast/config.py` | Validated physical geometry, scales, slices, ADCs and nonidealities |
| `memfast/functional.py` | PyTorch-order `mm` and `matmul`, broadcasting and FP32 backward |
| `memfast/quantization.py` | INT/BFP codes and scale groups |
| `memfast/simulator.py` | Matrix programming, forward, schedule selection and persistent cache |
| `memfast/reference.py` | Readable PyTorch implementation of array currents and ADCs |
| `memfast/cuda.py`, `kernels.cu` | CUDA loading, compilation and accelerated kernels |
| `memfast/random.py` | Reproducible physical random coordinates |
| `memfast/nn/linear.py` | Inference Linear |
| `memfast/nn/modules.py` | PyTorch-style trainable Linear and Conv2d constructors |
| `memfast/nn/convolution.py` | Grouped and depthwise Conv2d |
| `memfast/nn/recurrent.py` | Stateful RNN and GRU layers |
| `memfast/nn/training.py` | Trainable master weights, analog forward and FP32 HAT surrogate |
| `memfast/nn/conversion.py` | Full or selected model conversion and read-event reset |
| `memfast/models/` | Reusable ordinary PyTorch architectures and upstream loaders |
| `examples/01_...15_*.py` | Complete numbered task tutorials |
| `examples/common/` | Data preparation, checkpoint I/O, metrics and configuration |
| `examples/configs/` | Optional JSON configurations, one per tutorial |
| `examples/checkpoints/` | Small released digital/static LeNet checkpoints and provenance |
| `examples/notebooks/` | Consecutive learning workflows and intermediate results |
| `docs/` | User manual and project layout |

Example names use `NN_model_dataset.py`; general matrix and language benchmark
names describe the operation when there is no single fixed dataset.
VGG inference is followed by a separate HAT example that loads a checkpoint
and trains with analog forward values. MLP and LeNet use `--stage` for
their training, inference and HAT workflows.
Task libraries remain optional; torch and torchvision are standard dependencies.

## Configuration

Each tutorial creates its model and converts it explicitly in `main`.
The common utilities do not choose conversion boundaries.

| JSON group | Contents | Examples |
| --- | --- | --- |
| `model` | Architecture, dimensions, pretrained weights or checkpoint | `hidden_sizes`, `checkpoint`, `model`, `weights` |
| `conversion` | Layers to map or retain digitally | `include`, `exclude` |
| `simulator` | `Config` fields; nested `nonidealities` | `array_size`, `bw_e`, `rows_per_read`, `adc_bits` |
| `run` | Device, stage, optimizer, data and execution | `device`, `stage`, `epochs`, `learning_rate`, `mode` |

The JSON is optional. Defaults are declared next to the example's arguments;
command-line arguments override JSON values. Relative paths are interpreted
from the working directory, except the bundled LeNet checkpoint, which is
resolved relative to the examples directory. Common simulator flags have command-line
forms; all physical parameters are available in the `simulator` object.
Run one example with `--help` to see its task-specific fields.

```bash
python examples/03_lenet5_mnist.py \
  --config examples/configs/03_lenet5_mnist.json \
  --data-root data --exclude classifier.4
```

## Migration

The old MemIntelli `NN_layers/linear.py` and `NN_layers/convolution.py`
were replaced by MemFast's layer adapters. Before this reorganization both
adapters lived in `memfast/nn/layers.py`; their public exports remain
`AnalogLinear` and `AnalogConv2d`. They retain bias, convolution groups,
stride, dilation, numeric padding and padding-mode behavior.
Conv1d, Conv3d and string padding are outside the current conversion API.

CIFAR VGG retains the legacy three-layer classifier. CIFAR ResNet retains
the unpadded 3x3 stem and no max pooling. Architectures are built first and
digital checkpoints load strictly before conversion. A similarly named
ImageNet checkpoint is not interchangeable.

Example outputs use user-supplied paths or `examples/output/`.

The VGG notebooks demonstrate nonideality sweeps and per-layer mixed precision.
