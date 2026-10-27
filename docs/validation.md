# Validation

The 0.2.0 release was checked on October 27, 2026, using a separate environment
with Python 3.13, PyTorch 2.10.0 and torchvision 0.25.0. Optional integration
checks used timm 1.0.16, transformers 5.8.0, lm-evaluation-harness 0.4.10 and
Ultralytics 8.4.176. CPU-only checks also ran with PyTorch 2.5.1.

The built wheel was installed and imported from its environment's
`site-packages`, outside the source tree. All 328 tests passed there, including
CUDA and optional integrations, with no skipped cases. A separate BFP YOLO
smoke check and dependency consistency check also passed.

## Numerical Coverage

The automated tests compare the simulator with independent integer-dot-product
and grouped-dequantization calculations. They cover:

- Physical arrays and shorter ADC reads, including `R=128,P=32`, logical tails,
  nondivisible physical reads, and scale restoration before accumulation.
- Tensor, channel and rectangular group quantization for inputs and weights.
- INT and BFP, mixed activation/weight precision, 10-, 16- and 20-bit data,
  binary slicing, and multibit layouts including `(1,1,2,2,2)`.
- Uniform and per-weight-slice ADC resolutions, nearest-even rounding,
  saturation, sign slices and S-shaped ADC response.
- Programming/read/voltage/output noise, read sampling modes, stuck-at faults,
  drift, deterministic replay, large random coordinates and read-event values.
- Subnormal quantization scales, finite results with extreme scale factors,
  wide slice cells, closely spaced conductances and extreme physical constants.
- Checkpoint restoration, shared parameters, grouped and depthwise convolution,
  recurrent states, HAT gradient calculations and distributed training.
- Complete tuning-domain coverage, strict measured argmin selection, persistent
  cache reuse, corrupt cache entries and recovery from cached resource failures.

CPU and CUDA use the same physical model. Their floating reduction orders can
differ at ADC boundaries. Schedule changes within CUDA are checked for exact
output equality for the same input and random event.

## Example Workflows

Thirty-two execution checks cover the matrix, model conversion, nonideality,
MLP, HAT, two-process DDP, checkpoint reload, recurrent, vision, detection and
language-model entry points. Vision smoke checks execute the complete named
architecture, including VGG, ResNet, MobileNetV2 and DeiT. Llama, Qwen2 and Qwen3
checks use tiny upstream decoders and run perplexity, generation and the actual
HFLM harness adapter against an in-memory task.

These smoke checks use synthetic inputs or random weights. They establish
workflow execution and output validity. Model accuracy requires a matching
pretrained checkpoint and task dataset.

The VOC validation API also ran on two explicitly synthetic annotated images
and an untrained 20-class detector. Prediction, dataset loading, label matching
and AP calculation executed; its scores are not pretrained VOC accuracy.

## Real Data

The following inference checks use pretrained checkpoints and actual data.
The analog configuration is signed A8W8 INT, binary slices, 64-by-64 arrays,
ADC6, per-output-channel weight scales, per-vector input scales, and zero
nonidealities.

| Model and data | Samples | Analog top-1 |
| --- | ---: | ---: |
| VGG11-BN, CIFAR-10 test | 10,000 | 92.79% |
| ResNet18, CIFAR-10 test | 10,000 | 94.60% |
| DeiT-Tiny, ImageNet validation subset | 32 | 93.75% |

The ImageNet row is a limited workflow check on the first 32 sorted validation
images, not an estimate of complete ImageNet accuracy.

VGG11-BN digital top-1 was 92.78%. Its complete analog logits retained SHA-256
`7ca4cdac7917dc5da11582ac3cedbc1e0ae57ff8aa9618b69f6f017841ec9072`,
matching the earlier validated default, tuned and cached runs.
The existing [nonideality scan tables](../examples/results/README.md) retain
their measured configurations and sample counts.

## Reproducing Checks

```bash
python -m pip install -e ".[dev,models]"
python -m pytest -q
```

CUDA cases run when a CUDA device is available. The [example guide](../examples/README.md)
provides commands and checkpoint/data requirements for each workflow.
Execution checks ran with shared device load; this report makes no throughput
comparison between models or scheduling choices.
