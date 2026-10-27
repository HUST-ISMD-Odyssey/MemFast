# Tutorial Checkpoint

`lenet5_mnist.pt` is a MemFast tutorial checkpoint trained from scratch on
MNIST. It is provided under this repository's MIT License. MNIST retains
its original dataset terms.

| Property | Value |
| --- | --- |
| Architecture | `memfast.models.LeNet5`, 28x28 adaptation |
| Training | 60,000 MNIST training images, 8 epochs, Adam, learning rate 0.001 |
| Batch / seed | 128 / 135 |
| Input transform | ToTensor, `(x - 0.5) / 0.5` |
| Training arithmetic | PyTorch FP32, digital, CUDA |
| Evaluation | 10,000 MNIST test images, no augmentation |
| Digital accuracy | 9,849 / 10,000 = 98.49% |
| Default CIM accuracy | 9,850 / 10,000 = 98.50% |
| SHA-256 | `d8eca094258b1a4850183ae5447ba199e9ddb39c6f246b5c68b2e42626f810a1` |

The CIM result uses A8W8 INT, binary slices, 64x64 arrays, 64 rows per read,
ADC6, per-vector input scales, per-output-channel weight scales, and
write variation 0.02. All other nonidealities are disabled. The result is one
fixed programming realization, not an average over seeds.

```bash
python examples/03_lenet5_mnist.py --stage train --device cuda:0 \
  --data-root data --download --epochs 8 --batch-size 128 --seed 135
python examples/03_lenet5_mnist.py --device cuda:0 --data-root data --digital
python examples/03_lenet5_mnist.py --device cuda:0 --data-root data
```

The example resolves this checkpoint by default. User checkpoints do not need
to live here. Save them to a chosen path and pass `--checkpoint PATH`.
HAT checkpoints retain the original digital parameter names and can be loaded
before conversion. Rebuild the matching architecture before loading strictly.

## Static INT4 Checkpoint

`lenet5_mnist_static_int4.pt` contains packed signed INT4 weights and fixed
INT8 input scales for all five CIM matrices, without FP32 weight copies.
It is derived from the digital checkpoint above. Calibration observes 1,024
randomly selected MNIST training images with seed 135; test images are
used only for evaluation.

With ADC6, 64x64 arrays, write variation 0.02 and all other effects off,
static inference correctly classifies **9,835 / 10,000 (98.35%)**.
Reloading the artifact with batch 97 reproduces this result without online
scale estimation. Its SHA-256 is
`e7a8d182b4047e46a56ce5c9162e7c6eda131e62a5c2cfaf88ba127af219bb2c`.

```bash
python examples/04_lenet5_mnist_static_quant.py --device cuda:0 --data-root data
python examples/04_lenet5_mnist_static_quant.py --calibrate \
  --device cuda:0 --data-root data --calibration-samples 1024 \
  --checkpoint examples/checkpoints/lenet5_mnist.pt \
  --static-checkpoint examples/output/lenet5_static_int4.pt
```

The first command loads the included static artifact. For a custom artifact,
pass `--static-checkpoint PATH` during inference.
