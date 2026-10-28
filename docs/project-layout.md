# Project Layout

MemFast separates numerical simulation, PyTorch layers, custom architectures
and runnable workflows.

| Path | Responsibility |
| --- | --- |
| `memfast/config.py` | Validated geometry, scales, slices, ADCs and nonidealities |
| `memfast/functional.py` | PyTorch-order matrix multiplication and backward |
| `memfast/quantization.py` | INT/BFP codes and scale groups |
| `memfast/simulator.py` | Programming, forward, schedule search and cache |
| `memfast/reference.py` | Readable PyTorch currents and ADCs |
| `memfast/cuda.py`, `kernels.cu` | CUDA compilation and accelerated kernels |
| `memfast/random.py` | Reproducible physical sampling coordinates |
| `memfast/nn/linear.py` | Compact inference Linear |
| `memfast/nn/convolution.py` | Grouped/depthwise inference Conv2d |
| `memfast/nn/modules.py` | PyTorch-style trainable layer constructors |
| `memfast/nn/recurrent.py` | RNN and GRU layers |
| `memfast/nn/training.py` | HAT master weights and FP32 surrogate backward |
| `memfast/nn/conversion.py` | Full/partial conversion and read-event reset |
| `memfast/models/lenet.py`, `mlp.py`, `recurrent.py` | Custom baseline architectures |
| `memfast/models/vgg_cifar.py`, `resnet_cifar.py` | Custom CIFAR architectures |
| `memfast/models/speech.py` | Speech Commands GRU and checkpoint audio frontend |
| `examples/01_...15_*.py` | Complete numbered workflows |
| `examples/common/training.py` | `train(..., epochs=...)` and `evaluate(...)` |
| `examples/common/checkpoints.py` | Strict state loading and FP32 checkpoint saving |
| `examples/checkpoints/` | Two released LeNet tutorial assets |
| `examples/notebooks/` | Stepwise tutorials with executed outputs |
| `docs/` | User manual and layout |

Names follow `NN_model_dataset.py`; matrix and language examples name the
operation or benchmark. Custom architectures live in `memfast/models`.
torchvision, timm, Hugging Face and Ultralytics models load directly in
examples and use their upstream implementations.

## Example Structure

Each example starts with editable paths and settings, including a visible
`Config`. Its `main` constructs the dataset and DataLoader, builds or loads
the model, restores weights, converts selected layers and runs the task.
`common` does not choose models, data, simulator settings or conversion scope.

MLP, LeNet and Speech Commands GRU accept `--stage train|inference|hat`.
VGG HAT follows its inference example. The static LeNet example accepts
`--stage calibrate|inference`. Other examples run their inference task directly.
User outputs go to `examples/output/` by default; released checkpoints are
kept separate. Dataset paths and preprocessing are visible in each example.

## Layer Migration

The old MemIntelli `NN_layers/linear.py` and `NN_layers/convolution.py`
are represented by MemFast's `nn/linear.py`, `nn/convolution.py` and trainable
constructors in `nn/modules.py`. Public `AnalogLinear` and `AnalogConv2d`
exports remain available through `nn/layers.py`.

Layers retain bias, groups, stride, dilation, numeric padding and padding
modes. HAT supports analog forward with FP32 surrogate backward. The custom
CIFAR ResNet18 retains its unpadded 3x3 stride-one stem and no maxpool.
Build the matching architecture and restore its weights before conversion.
