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
| `03_dft_image_denoising.ipynb` | Photograph included in Matplotlib | NumPy/CIM DFT reconstructions, images and PSNR |
| `04_vgg_nonidealities.ipynb` | VGG11-BN checkpoint and CIFAR-10 test data | Device models, sampling granularity, individual/joint scans and tables |
| `05_vgg_mixed_precision.ipynb` | The same VGG11-BN checkpoint and CIFAR-10 data | Explicit per-layer 8/4-bit choices, per-layer ADC overrides and accuracy comparisons |

The first notebook uses `MEMFAST_MNIST_ROOT` to override its data directory.
The second uses `MEMFAST_IMAGENET_ROOT`, defaulting to `data/imagenet/val`.
Set these environment variables before launching Jupyter, or edit the visible
settings cell. Use ImageNet's standard class-directory ordering.

MobileNet evaluates the first 128 images by default. Set `max_samples=None`
for the complete validation split.
The DFT notebook defaults to CPU and a fixed 64x64 photograph.

The VGG notebooks use `MEMFAST_CIFAR10_ROOT`, `MEMFAST_VGG_CHECKPOINT` and
`MEMFAST_NOTEBOOK_DEVICE`. They evaluate 512 fixed test images by default;
set `MEMFAST_VGG_SAMPLES=0` for the full test split. All compared configurations
use the same images and seed. The nonideality notebook declares 59 unique CIM
configurations; the mixed-precision notebook compares 30 configurations and
one editable user example. Each selected layer has its own simulator.

Checkpoints created while learning are written under `output/` in the
notebook's working directory and are not distributed. The released LeNet
checkpoint is separate, in `examples/checkpoints/`.

Ideal CIM retains quantization, slicing and ADC conversion. Write-noisy CIM
adds a fixed programmed conductance realization. Rounding can make the
ideal and noisy outputs coincide for a particular configuration; the
notebooks show the corresponding accuracy and reconstruction results.
