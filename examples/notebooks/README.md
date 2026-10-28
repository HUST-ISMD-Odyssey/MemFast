# Guided Workflows

Run cells from top to bottom in the same environment where MemFast is
installed. These notebooks emphasize intermediate steps and comparisons;
the numbered Python examples provide command-line task entry points.
Both call the same `Config`, `Simulator` and `convert_model` interfaces.

```bash
python -m pip install -e .
python -m pip install jupyterlab matplotlib
python -m jupyterlab examples/notebooks
```

| Notebook | Inputs | Output |
| --- | --- | --- |
| `01_custom_model_mnist.ipynb` | MNIST, downloaded by torchvision if absent | Custom-model training, saved/reloaded checkpoint, four accuracy comparisons |
| `02_pretrained_mobilenet_imagenet.ipynb` | torchvision MobileNetV2 weights and local ImageNet validation split | Digital, all-layer CIM and first/head-digital accuracy |
| `03_dft_image_denoising.ipynb` | Bundled lake landscape with periodic stripes | DFT notch filtering, NumPy/ideal-CIM/write-0.05 reconstructions, images and PSNR |
| `04_vgg_nonidealities.ipynb` | VGG11-BN checkpoint and CIFAR-10 test data | Device models, sampling granularity, independent parameter scans and tables |
| `05_vgg_mixed_precision.ipynb` | The same VGG11-BN checkpoint and CIFAR-10 data | Explicit per-layer 8/4-bit choices, per-layer ADC overrides and accuracy comparisons |

The first notebook uses `MEMFAST_MNIST_ROOT` to override its data directory.
The second uses `MEMFAST_IMAGENET_ROOT`, defaulting to `data/imagenet/val`.
Set these environment variables before launching Jupyter, or edit the visible
settings cell. Use ImageNet's standard class-directory ordering.

MobileNet evaluates the complete validation split by default
(`max_samples=None`). Set a positive integer in its settings cell for a subset.
The DFT notebook uses a 256 x 256 photograph and selects CUDA when available,
with CPU as the fallback. Set `MEMFAST_NOTEBOOK_DEVICE=cpu` to select CPU.
Its two-bin notch filter removes interference at a known frequency while
preserving the remaining image frequencies. Input scales cover groups of
64 values; all slices are one bit, with A8W8 and 64x64 arrays.

The VGG notebooks use `MEMFAST_CIFAR10_ROOT`, `MEMFAST_VGG_CHECKPOINT` and
`MEMFAST_NOTEBOOK_DEVICE`. Both evaluate all 10,000 CIFAR-10 test images by
default. The nonideality notebook evaluates 36 distinct CIM configurations
and presents a separate table for each parameter sweep. The mixed-precision
notebook evaluates ten policy/write-variation combinations and one editable
user assignment, alongside a digital baseline. Each selected layer has its
own simulator. All comparisons use the same images and seed. Only the
mixed-precision notebook accepts `MEMFAST_VGG_SAMPLES`; its default of `0`
selects the full test split.

Checkpoints created while learning are written under `output/` in the
notebook's working directory and are not distributed. The released LeNet
checkpoint is separate, in `examples/checkpoints/`.

Ideal CIM retains quantization, slicing and ADC conversion. Write-noisy CIM
adds a fixed programmed conductance realization. Rounding can make the
ideal and noisy outputs coincide for a particular configuration; the
notebooks show the corresponding accuracy and reconstruction results.
