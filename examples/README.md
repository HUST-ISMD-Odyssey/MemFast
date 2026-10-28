# Examples And Notebooks

Install MemFast from the repository root:

```bash
python -m pip install -e .
```

Each Python example has editable settings at the top: device, dataset path,
checkpoint, batch size, CIM configuration and conversion boundaries. Model
construction, preprocessing, DataLoader, checkpoint loading and conversion are
written in the example itself. There is no separate JSON configuration.
Paths are relative to the working directory unless based on `__file__`.

PyTorch and torchvision are standard dependencies. Speech, DeiT, Hugging Face
and YOLO examples use task libraries available through:

```bash
python -m pip install -e ".[models]"
```

## Python Examples

Run these commands from the repository root after setting the data and weight
paths in the corresponding file.

| Example | Run | Workflow |
| --- | --- | --- |
| `01_matrix_multiplication.py` | `python examples/01_matrix_multiplication.py` | Compare `simulator.mm(A, B)` with `torch.mm(A, B)` |
| `02_mlp_mnist.py` | `python examples/02_mlp_mnist.py --stage train` | Define three layers, train, save, restore, infer or run HAT |
| `03_lenet5_mnist.py` | `python examples/03_lenet5_mnist.py` | Load the supplied LeNet checkpoint; also supports training and HAT |
| `04_lenet5_mnist_static_quant.py` | `python examples/04_lenet5_mnist_static_quant.py` | Load packed INT4 weights and fixed activation scales |
| `05_vgg_cifar.py` | `python examples/05_vgg_cifar.py` | Custom CIFAR VGG checkpoint and CIM inference |
| `06_vgg_cifar_hat.py` | `python examples/06_vgg_cifar_hat.py` | Adapt a trained CIFAR VGG with FP32 backward |
| `07_resnet_cifar.py` | `python examples/07_resnet_cifar.py` | Released custom CIFAR ResNet18 and CIM inference |
| `08_vgg_imagenet.py` | `python examples/08_vgg_imagenet.py` | torchvision pretrained model and ImageNet evaluation |
| `09_resnet_imagenet.py` | `python examples/09_resnet_imagenet.py` | torchvision ResNet and ImageNet evaluation |
| `10_mobilenetv2_imagenet.py` | `python examples/10_mobilenetv2_imagenet.py` | MobileNetV2 with editable full/partial conversion |
| `11_deit_imagenet.py` | `python examples/11_deit_imagenet.py` | timm DeiT and ImageNet evaluation |
| `12_gru_speech.py` | `python examples/12_gru_speech.py` | Custom Speech Commands GRU; training, inference and HAT |
| `13_yolo_voc.py` | `python examples/13_yolo_voc.py` | Local VOC detector and upstream AP evaluation |
| `14_bert_squad.py` | `python examples/14_bert_squad.py` | Hugging Face BERT QA and SQuAD EM/F1 |
| `15_llm_lm_eval.py` | `python examples/15_llm_lm_eval.py` | Hugging Face causal LM and lm-evaluation-harness |

## Settings And Conversion

Every example constructs its simulator configuration visibly:

```python
CONFIG = Config(
    activation_bits=8,
    weight_bits=8,
    input_slice=(1,) * 8,
    weight_slice=(1,) * 8,
    array_size=(64, 64),
    rows_per_read=64,
    input_quant_gran=(1, None),
    weight_quant_gran=(None, 1),
    adc_bits=6,
    bw_e=None,
    nonidealities=Nonidealities(write_variation=0.02),
)
```

Only write variation is enabled by default. Set it to zero for ideal devices;
quantization and ADC conversion remain active. Use `bw_e=None` for INT or an
exponent width for BFP. Slice widths are MSB first; signed layouts start with
a one-bit sign slice. For eight bits, `(1, 1, 2, 2, 2)` is another layout.
`adc_bits` accepts one width or a tuple with one width per weight slice.
The simulator validates these settings.

`01_matrix_multiplication.py` also illustrates 128-row arrays with 32 rows
per ADC read and independent scale groups. Set its `EXECUTION_MODE` to
`"tuned"` for exhaustive schedule selection and persistent caching.
The [manual](../docs/user-guide.md) explains physical and quantization parameters.

All supported layers convert by default. Change the example's `INCLUDE` and
`EXCLUDE` to choose named layers or subtrees. Exclusions take precedence.
For the three-layer MLP:

```python
INCLUDE = ("fc1", "fc2")
EXCLUDE = ()
model = convert_model(model, simulator, include=INCLUDE, exclude=EXCLUDE)
```

`fc3` stays digital. For LeNet, the first four eligible layers are
`features.0`, `features.3`, `classifier.0`, and `classifier.2`; its last
classifier is `classifier.4`. For MobileNetV2, set
`EXCLUDE = ("features.0", "classifier")` to keep the first convolution and
classification head digital. Inspect `model.named_modules()` for other models.

## Custom Models, Data And Checkpoints

The MLP defines its dimensions in the example. LeNet, custom CIFAR VGG,
custom CIFAR ResNet18 and Speech Commands GRU are ordinary PyTorch models in
`memfast/models/`. Construct the architecture first, load its digital
checkpoint, then convert it.

The MLP defaults to digital training:

```bash
python examples/02_mlp_mnist.py --stage train
python examples/02_mlp_mnist.py --stage inference
python examples/02_mlp_mnist.py --stage hat
```

Change `HIDDEN_SIZES`, `DATA_ROOT`, `CHECKPOINT`, `EPOCHS` and
`LEARNING_RATE` in that file. Training saves to `CHECKPOINT`; HAT saves to
`HAT_CHECKPOINT`. Set `CHECKPOINT` to the HAT output to evaluate it later.
Restore the same architecture dimensions used during training.

LeNet and GRU also accept `--stage train|inference|hat`. LeNet inference
defaults to the supplied checkpoint; its training and HAT destinations are
`TRAIN_CHECKPOINT` and `HAT_CHECKPOINT`. The VGG HAT example loads `CHECKPOINT`
and writes `SAVE_CHECKPOINT`. Training uses `train(..., epochs=EPOCHS)`;
software training and HAT share the same FP32 training loop.
HAT uses analog forward values and FP32 master weights/surrogate backward.

To use a custom dataset, replace the example's Dataset and preprocessing,
retain a DataLoader yielding `(inputs, targets)`, and set the model's input
shape and class count accordingly. Paths and transforms are not selected by
`common`. Its helpers only train, evaluate and read/write checkpoints.
The loader accepts raw state dictionaries and dictionaries containing
`state_dict`, `model_state_dict` or `model`; loading is strict.

### Static LeNet Quantization

Default inference loads `examples/checkpoints/lenet5_mnist_static_int4.pt`
without online scale estimation:

```bash
python examples/04_lenet5_mnist_static_quant.py
python examples/04_lenet5_mnist_static_quant.py --stage calibrate
```

Calibration observes `CALIBRATION_SAMPLES` MNIST training images, freezes
per-layer input scales and saves packed weights/scales to
`CALIBRATED_CHECKPOINT` under `examples/output/`. Set `STATIC_CHECKPOINT` to
that path to restore the new artifact. Use the same configuration and
conversion selection that created it. Released tutorial assets are never
overwritten by calibration.

### CIFAR Models

The VGG source is `memfast/models/vgg_cifar.py`. The default VGG11-BN
checkpoint is `cifar10_vgg11_bn-eaeebf42.pt` from the
[pytorch-cifar-models VGG release](https://github.com/chenyaofo/pytorch-cifar-models/releases/tag/vgg).
Place it at `CHECKPOINT`; the default path uses PyTorch's hub checkpoint cache.

The custom ResNet18 is in `memfast/models/resnet_cifar.py`. Its 3x3 stem
uses stride one, no padding and no maxpool. Download
`resnet18-cifar10_9498.bin` or `resnet18-cifar100_7926.bin` from the
[MemIntelli pretrained release](https://github.com/HUST-ISMD-Odyssey/MemIntelli/releases/tag/pretrained_model)
and set `CHECKPOINT`. These are CIFAR checkpoints for this architecture.
The ResNet example preserves their ImageNet-style normalization;
VGG uses CIFAR-specific normalization.

For CIFAR-100, change the dataset constructor to `datasets.CIFAR100`,
`NUM_CLASSES` to 100, and use matching weights and preprocessing.

### Upstream Models

ImageNet examples use torchvision's `get_model` and weight enum preprocessing
directly. Set `MODEL_NAME` to another available classifier; there is no MemFast
model whitelist. `CHECKPOINT=None` loads upstream pretrained weights. Set a
local checkpoint path to restore your own weights instead. Adjust preprocessing
for custom training. `DATA_ROOT` points to an ImageFolder validation split
with the checkpoint's class ordering.

DeiT uses timm's model and data configuration. BERT and LLM examples use
Hugging Face `from_pretrained`; `MODEL_ID` can be a hub identifier or a local
`save_pretrained` directory. They do not redefine those architectures in
`memfast/models`. Conversion operates on supported PyTorch layers; the other
operators retain their upstream implementation.

BERT uses `DATA_FILE` for SQuAD v1 JSON or parquet, reconstructs spans across
overflow windows and reports EM/F1. LLM evaluation uses `TASKS` for harness
tasks, `TEXT_FILE` for a local perplexity corpus and `PROMPT` for optional
generation. LLaMA and Qwen use the same example; the language head stays
digital by default. These task examples provide inference and evaluation.

### Speech And Detection

For Speech Commands v0.02, extract audio and split lists under `DATA_ROOT`.
Run the GRU example with `--stage train` to build its weights. It estimates
log-mel normalization on training frames only and saves the feature settings,
statistics, class order and model dimensions with the weights. Inference
loads those settings. Set `WAV` for a single mono 16 kHz recording.

YOLO uses a local checkpoint trained for VOC's 20 classes and an Ultralytics
dataset YAML in `DATA_YAML`. Class names and ordering must agree. The example
fuses Conv/BatchNorm before conversion and runs upstream AP evaluation.

## Notebooks

Notebooks explain workflows cell by cell and retain printed results, tables
and images. Python scripts are shorter runnable task examples. Both use the
same MemFast APIs.

```bash
python -m pip install -e . jupyterlab matplotlib
python -m jupyter lab examples/notebooks
```

| Notebook | Workflow |
| --- | --- |
| `01_custom_model_mnist.ipynb` | Define, train, save/load, compare digital and CIM |
| `02_pretrained_mobilenet_imagenet.ipynb` | Pretrained model, full/partial CIM and accuracy |
| `03_dft_image_denoising.ipynb` | DFT matrices on CIM, periodic stripe removal and PSNR |
| `04_vgg_nonidealities.ipynb` | Device models, independent parameter sweeps and full-test tables |
| `05_vgg_mixed_precision.ipynb` | User-defined layer precisions and full-test comparisons |

Notebook settings accept `MEMFAST_MNIST_ROOT`, `MEMFAST_IMAGENET_ROOT`,
`MEMFAST_CIFAR10_ROOT` and `MEMFAST_VGG_CHECKPOINT` environment overrides.
The DFT notebook uses a bundled lake landscape.

## Example Results

### LeNet-5 / MNIST

The supplied digital checkpoint was trained on all 60,000 training images for
8 epochs, batch 128, Adam at 0.001, seed 135 and FP32 software execution.
Test normalization is `(x-0.5)/0.5`.

| Evaluation | Test images | Correct | Top-1 |
| --- | ---: | ---: | ---: |
| Digital FP32 | 10,000 | 9,849 | 98.49% |
| Static A8W4, ADC6, write variation 0.02 | 10,000 | 9,835 | 98.35% |

Static calibration uses 1,024 training images and seed 135. The artifact
contains signed INT4 weights and fixed INT8 input scales for five matrices.

### BERT / SQuAD

The `csarron/bert-base-uncased-squad-v1` checkpoint on the first 32 SQuAD v1
validation questions gave **EM 87.50, F1 96.5625**. This measurement uses FP32,
length 384, stride 128, batch 4, A8W8 INT, binary slices, 64x64 arrays,
64 rows per read, ADC6, per-vector input scales, per-channel weight scales,
write variation 0.02 and seed 135. Other device effects are disabled.

### Notebook Results

The VGG nonideality sweeps and mixed-precision comparisons evaluate all
10,000 CIFAR-10 test images. Results appear in their executed notebooks.

The MobileNetV2 notebook evaluates all 50,000 ImageNet validation images:

| Conversion | Correct | Top-1 |
| --- | ---: | ---: |
| Digital FP32 | 36,075 | 72.150% |
| Full CIM | 35,941 | 71.882% |
| CIM with first convolution and classifier digital | 35,940 | 71.880% |

These CIM runs use A8W8 INT, binary slices, 64x64 arrays, ADC6 and
write variation 0.02. Other effects are disabled.

The 256x256 DFT photograph has additive periodic stripes of amplitude 0.18,
with 12 cycles across its width. A notch filter rejects the two known
interference frequencies. Both NumPy and CIM subtract the observed image
mean before transforming and restore it after the inverse:

| Reconstruction | PSNR |
| --- | ---: |
| Striped input | 17.905 dB |
| NumPy DFT notch filter and inverse | 50.876 dB |
| CIM, ideal devices | 46.241 dB |
| CIM, write variation 0.05 | 33.501 dB |

CIM uses A8W8, binary slices, 64x64 arrays, ADC8, per-64-value input
scales, per-output-channel weight scales and seed 135. All forward/inverse
DFT matrix products run through CIM; centering and frequency masking are
digital. The notebook retains the images and spectrum. With write variation
0.05, reconstruction improves PSNR by 15.596 dB over the striped input.
