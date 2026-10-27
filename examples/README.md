# Examples

Run commands from the repository root after installation.

| Example | Purpose | Downloads |
| --- | --- | --- |
| `matrix.py` | Program and read a matrix with independent quantization groups | None |
| `model.py` | Convert a PyTorch convolutional network | None |
| `nonidealities.py` | Exercise each device and converter effect separately | None |
| `cifar10.py` | Evaluate pretrained VGG11-BN and scan nonidealities | CIFAR-10 and weights |

## Matrix And Model Conversion

```bash
python examples/matrix.py
python examples/model.py
python examples/nonidealities.py
```

For CUDA execution, add `--device cuda:0`. For tuning, use:

```bash
python examples/matrix.py --device cuda:0 --mode tuned
python examples/model.py --device cuda:0 --mode tuned
```

The first tuned invocation searches each new shape. Running the same command
again uses its cached schedules.

## VGG11-BN On CIFAR-10

Install `python -m pip install ".[examples]"`. Evaluate all 10,000 test images:

```bash
python examples/cifar10.py --download --device cuda:0
```

The script downloads weights from
[pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models)
and checks the checkpoint filename hash. Supply an existing checkpoint with
`--checkpoint path/to/cifar10_vgg11_bn-eaeebf42.pt`.
The checkpoint architecture is VGG11 with batch normalization and three
classifier layers; loading is strict. CIFAR-10 uses its test split and the
normalization constants in the example.

To check the workflow on a subset, use `--limit 32`. The output records the
actual sample count. Use `--limit 10000` for full-test results.

## Parameter Scans

```bash
python examples/cifar10.py --download --device cuda:0 \
    --scan all --output results/cifar10
```

Each scan changes one nonideality while retaining A8W8, 64-by-64 arrays,
ADC6, per-output-channel weight quantization, and per-vector input
quantization. Programming variation, all three read models, voltage
variation, current output noise, ADC nonlinearity, both stuck-at rates,
and drift are evaluated separately.

Choose one scan with `--scan read_slice_pair`, `--scan drift_time`, or another
name listed by `--help`. Drift fixes the exponent at `0.05` and the reference
time at one second, then varies elapsed time.

The output directory contains:

- `results.csv`: exact correct/sample counts, top-1 accuracy, cross entropy,
  elapsed time, and full-output hashes.
- `metadata.json`: model identity, complete configurations, sample count,
  batch size, software version, and seed.
- `tables.md`: one table for each nonideality.

Add `--save-outputs` to retain per-configuration logits and targets for
independent recomputation of the reported metrics.

Zero-effect rows share one evaluated ideal-array result within a scan run.
The digital network is evaluated separately. These are accuracy scans;
reported wall times include evaluation and first-call preparation.

Measured full-test tables are in [results](results/README.md). Dataset and
checkpoint files are stored outside the source tree and are not distributed
with MemFast.
