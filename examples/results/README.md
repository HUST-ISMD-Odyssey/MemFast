# VGG11-BN Parameter Scans

CIFAR-10 test set, all 10,000 images per row, batch size 16, seed 135.
Pretrained weights: `cifar10_vgg11_bn-eaeebf42.pt` from
[pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models).

A8W8; binary cells and one-bit input slices; 64-by-64 arrays; ADC6;
per-output-channel weight quantization and per-vector activation quantization.
The complete convolutional and linear network, including its classifier, is simulated.

Digital top-1 accuracy: **92.78%**.
Each scan changes one effect; all other nonidealities are zero.
Results use one fixed programming realization. Read events advance with each batch.
Drift fixes its exponent at 0.05 and reference time at one second.

Every row was checked by independently recomputing correct counts, cross entropy,
and output hashes from saved logits and targets on CPU.
The ideal-array output hash also matches full-test tuned and cached execution.

## Programming Variation

Log-conductance standard deviation.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.02 | 92.79 | 0.330616 |
| 0.05 | 92.75 | 0.332284 |
| 0.08 | 92.18 | 0.361474 |

## Read Variation: Shared Conductance

Log-conductance standard deviation.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.02 | 92.79 | 0.330617 |
| 0.05 | 92.75 | 0.338049 |
| 0.08 | 91.43 | 0.403277 |

## Read Variation: Independent Slice Pairs

Log-conductance standard deviation.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.02 | 92.79 | 0.330616 |
| 0.05 | 92.53 | 0.338187 |
| 0.08 | 91.26 | 0.411198 |

## Read Variation: Current Approximation

Log-conductance standard deviation.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.02 | 92.79 | 0.330616 |
| 0.05 | 92.65 | 0.333998 |
| 0.08 | 90.41 | 0.433730 |

## Voltage Variation

Relative standard deviation.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.02 | 92.79 | 0.330616 |
| 0.05 | 92.77 | 0.331375 |
| 0.08 | 91.63 | 0.379825 |

## Current Output Noise

Standard deviation relative to full row current.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.001 | 92.79 | 0.330616 |
| 0.005 | 10.54 | 3.977808 |
| 0.01 | 10.29 | 268.307167 |

## S-Shaped ADC Nonlinearity

Transfer strength.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.02 | 92.79 | 0.330618 |
| 0.05 | 92.81 | 0.330374 |
| 0.08 | 92.37 | 0.342981 |

## Stuck-At-Low Faults

Fault probability.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.001 | 92.66 | 0.328708 |
| 0.005 | 92.00 | 0.341649 |
| 0.01 | 90.35 | 0.404311 |

## Stuck-At-High Faults

Fault probability.

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 0.001 | 92.56 | 0.336680 |
| 0.005 | 91.10 | 0.392078 |
| 0.01 | 89.37 | 0.450699 |

## Conductance Drift

Elapsed time (seconds).

| Value | Top-1 (%) | Cross entropy |
| ---: | ---: | ---: |
| 0 | 92.79 | 0.330616 |
| 1 | 92.28 | 0.344928 |
| 100 | 64.33 | 1.115182 |
| 10000 | 18.04 | 2.235586 |

## Reproduction

```bash
python examples/cifar10.py --download --device cuda:0 --scan all --save-outputs
```

[results.csv](results.csv) contains counts and hashes; [metadata.json](metadata.json)
contains the checkpoint identity, full configurations, and software version.
Separate scan runs repeat the digital and zero-effect control rows.
Recorded evaluation wall times include preparation and are not a timing benchmark.
