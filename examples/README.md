# Examples And Notebooks

Run from the repository root after `python -m pip install -e .`.
PyTorch and torchvision are standard dependencies. Speech, DeiT, Hugging Face
and YOLO integrations use optional libraries:
`python -m pip install -e ".[models]"`.
MLP, LeNet, CIFAR and torchvision ImageNet examples need no extra.

## Choose A Starting Point

**Notebooks** teach an end-to-end workflow in small, consecutive steps, with
intermediate outputs and plots. Open them in Jupyter and run cells in order.
Install JupyterLab and Matplotlib in the same environment as MemFast.

```bash
python -m pip install -e . jupyterlab matplotlib
python -m jupyter lab examples/notebooks
```

Each notebook starts with editable settings. Data paths use `pathlib` on
Windows and Linux; environment variables can override the defaults:

| Input | Setting | Default |
| --- | --- | --- |
| MNIST | `MEMFAST_MNIST_ROOT` | `data`; the custom-model notebook downloads MNIST when needed |
| ImageNet validation split | `MEMFAST_IMAGENET_ROOT` | `data/imagenet/val`, with the standard 1,000 class directories |
| CIFAR-10 | `MEMFAST_CIFAR10_ROOT` | `data/cifar10`, containing the extracted torchvision dataset |
| VGG11-BN weights | `MEMFAST_VGG_CHECKPOINT` | PyTorch hub's cached `cifar10_vgg11_bn-eaeebf42.pt` |

The VGG checkpoint comes from the
[pytorch-cifar-models VGG release](https://github.com/chenyaofo/pytorch-cifar-models/releases/tag/vgg).
MobileNetV2 uses torchvision's pretrained weight cache. The DFT notebook uses
a photograph bundled with Matplotlib and requires no dataset download.

| Notebook | What it demonstrates |
| --- | --- |
| [01_custom_model_mnist](notebooks/01_custom_model_mnist.ipynb) | Define dimensions, train digitally, save/load a checkpoint, compare digital, ideal-device CIM and noisy CIM |
| [02_pretrained_mobilenet_imagenet](notebooks/02_pretrained_mobilenet_imagenet.ipynb) | Load an upstream model, evaluate full CIM, retain the first layer and classifier digitally, compare accuracy |
| [03_dft_image_denoising](notebooks/03_dft_image_denoising.ipynb) | Map a complex DFT into real CIM matrices, filter image noise, compare reconstructions and PSNR |
| [04_vgg_nonidealities](notebooks/04_vgg_nonidealities.ipynb) | Explain each device model and its sampling, independently sweep its parameters and compare full-test accuracy |
| [05_vgg_mixed_precision](notebooks/05_vgg_mixed_precision.ipynb) | Choose 8/4-bit activations and weights by layer name, override ADC widths and compare accuracy |

**Numbered Python examples** run complete tasks from the command line. They
support repeatable evaluation, explicit arguments and JSON configurations.
They use the same public APIs as the notebooks. Model construction,
checkpoint loading and CIM conversion remain visible in each task source;
`common/` shares data, checkpoint and metric utilities.

| Example | Purpose |
| --- | --- |
| `01_matrix_multiplication.py` | Matrix multiplication, INT/BFP, dynamic slices, per-slice ADC and scale groups |
| `02_mlp_mnist.py` | Custom three-layer MLP, checkpoint restoration, full/partial CIM and HAT |
| `03_lenet5_mnist.py` | LeNet-5 and the supplied MNIST checkpoint |
| `04_lenet5_mnist_static_quant.py` | Training-set calibration, saved INT4 weights/fixed activation scales and inference |
| `05_vgg_cifar.py` | CIFAR VGG checkpoint loading and CIM inference |
| `06_vgg_cifar_hat.py` | CIFAR VGG HAT with analog forward and FP32 backward |
| `07_resnet_cifar.py` | CIFAR ResNet checkpoint loading and CIM inference |
| `08_vgg_imagenet.py` | Pretrained torchvision VGG inference |
| `09_resnet_imagenet.py` | Pretrained torchvision ResNet inference |
| `10_mobilenetv2_imagenet.py` | Pretrained MobileNetV2 inference |
| `11_deit_imagenet.py` | Pretrained timm DeiT inference |
| `12_gru_speech.py` | Speech Commands GRU and single-WAV inference |
| `13_yolo_voc.py` | Ultralytics detector and local VOC AP evaluation |
| `14_bert_squad.py` | Pretrained BERT QA, answer reconstruction and SQuAD EM/F1 |
| `15_llm_lm_eval.py` | Llama/Qwen perplexity, generation and harness evaluation |

Vision inference files concentrate on loading and evaluation. The VGG HAT
example adapts a trained checkpoint to the configured CIM model. MLP and LeNet retain
`--stage train|inference|hat` for custom-model and HAT workflows.
HAT uses analog forward values and FP32 master weights/surrogate backward.
Speech, detection and language examples are inference-only: they do not
implement the upstream task training loops.

## Configuration

JSON files live together in `examples/configs/`, named after their example.
They share a four-group schema, but architecture and task settings differ:

| Group | Contents |
| --- | --- |
| `model` | Architecture, dimensions, checkpoint or upstream weights |
| `conversion` | Named `include` and `exclude` paths |
| `simulator` | Physical/quantization `Config` fields and nested `nonidealities` |
| `run` | Device, data, stage, optimizer and execution mode |

JSON is optional. Command-line options override its values. Relative paths
are resolved from the working directory; the supplied LeNet checkpoint is
resolved relative to the example directory.

```bash
python examples/01_matrix_multiplication.py --config examples/configs/01_matrix_multiplication.json
python examples/03_lenet5_mnist.py --config examples/configs/03_lenet5_mnist.json \
  --data-root data --exclude classifier.4
```

Only write variation is enabled by default, at 0.02.
`--write-variation 0` selects ideal devices, while quantization and ADC remain
active. Slice widths are MSB first, begin with a one-bit sign slice and sum
to the selected precision. ADC widths can be uniform or per weight slice.
Omit `--bw-e` for INT; set it for BFP.

```bash
python examples/01_matrix_multiplication.py --activation-bits 20 --weight-bits 20 \
  --input-slice 1,3,4,4,4,4 --weight-slice 1,3,4,4,4,4 \
  --bw-e 8 --adc-bits 16 --write-variation 0
```

The [manual](../docs/user-guide.md) defines array geometry, per-read scale
alignment, quantization groups, nonidealities and cache behavior.

## Full And Partial Conversion

Build a digital model, load its checkpoint, then convert:

```python
model = ThreeLayerMLP(hidden_sizes=(512, 128))
load_checkpoint(model, checkpoint)
engine = Simulator(Config(), device="cuda:0")
analog = convert_model(model, engine, include=("fc1", "fc2"))
```

The classifier `fc3` remains digital. Without `include`, every supported layer
converts. `exclude` takes precedence; named paths include descendants.
Use `model.named_modules()` to inspect names.

The supplied LeNet has five eligible layers. Select the first four while
retaining its final classifier:

```bash
python examples/03_lenet5_mnist.py --device cuda:0 --data-root data \
  --include features.0 features.3 classifier.0 classifier.2
```

## MLP And LeNet Checkpoints

The MLP source defines its three Linear layers directly. Restore the same
hidden dimensions used for training:

```bash
python examples/02_mlp_mnist.py --stage train --device cuda:0 \
  --download --data-root data --epochs 8 --batch-size 128 \
  --hidden-sizes 512 128 --save-checkpoint examples/output/mlp_mnist.pt
python examples/02_mlp_mnist.py --device cuda:0 --data-root data \
  --checkpoint examples/output/mlp_mnist.pt --hidden-sizes 512 128
python examples/02_mlp_mnist.py --stage hat --device cuda:0 --data-root data \
  --checkpoint examples/output/mlp_mnist.pt --epochs 2 \
  --save-checkpoint examples/output/mlp_mnist_hat.pt
```

LeNet's architecture is in `memfast/models/lenet.py`: two Conv2d layers, Tanh
and average pooling, followed by three Linear layers producing raw logits.
It is adapted to 28x28 input. The default
`examples/checkpoints/lenet5_mnist.pt` achieved **98.49% digital accuracy**
on all 10,000 MNIST test images. Training settings and measured example results
are listed below.

```bash
python examples/03_lenet5_mnist.py --device cuda:0 --download --data-root data
python examples/03_lenet5_mnist.py --device cuda:0 --data-root data --digital
python examples/03_lenet5_mnist.py --stage train --device cuda:0 --data-root data \
  --epochs 8 --batch-size 128
python examples/03_lenet5_mnist.py --stage hat --device cuda:0 --data-root data \
  --epochs 1 --save-checkpoint examples/output/lenet5_mnist_hat.pt
```

LeNet training and HAT save under `examples/output/` unless
`--save-checkpoint` specifies another path. Pass the resulting path with
`--checkpoint` to evaluate your trained model. Default inference loads the
released checkpoint. User checkpoints may be anywhere; `checkpoints/` contains
only the two released tutorial assets.

## Vision

CIFAR VGG is `CifarVGG` in `memfast/models/vision.py`, with the earlier
three-layer classifier. CIFAR ResNet uses the earlier unpadded 3x3 stem and no
max pooling. Both construct the architecture and load task weights strictly
before conversion. CIFAR-100 uses `--num-classes 100`; ImageNet weights do not
substitute for CIFAR weights.

```bash
python examples/05_vgg_cifar.py --device cuda:0 --data-root data/cifar \
  --checkpoint /path/to/vgg11_bn_cifar10.pt
python examples/06_vgg_cifar_hat.py --device cuda:0 --data-root data/cifar \
  --checkpoint /path/to/vgg11_bn_cifar10.pt --epochs 1 \
  --save-checkpoint examples/output/vgg11_bn_cifar10_hat.pt
python examples/05_vgg_cifar.py --device cuda:0 --data-root data/cifar \
  --checkpoint examples/output/vgg11_bn_cifar10_hat.pt
python examples/07_resnet_cifar.py --device cuda:0 --data-root data/cifar \
  --checkpoint /path/to/resnet18_cifar10.pt
python examples/08_vgg_imagenet.py --device cuda:0 --weights DEFAULT --data-root data/imagenet/val
python examples/09_resnet_imagenet.py --device cuda:0 --weights DEFAULT --data-root data/imagenet/val
python examples/10_mobilenetv2_imagenet.py --device cuda:0 --weights DEFAULT --data-root data/imagenet/val
python examples/11_deit_imagenet.py --device cuda:0 --weights DEFAULT --data-root data/imagenet/val
```

ImageNet models come from torchvision or timm. `--weights DEFAULT` explicitly
requests upstream weights and preprocessing. `--checkpoint PATH` loads local
weights instead. The data root is an ImageFolder evaluation split arranged as
`val/class_name/image.JPEG`, with the checkpoint's class index ordering.
`--limit 32` evaluates the first 32 images.
CIFAR examples load task-specific checkpoints; HAT saves weights with the
same digital model keys for subsequent inference.

## Speech And Detection

The GRU tutorial preserves the two-layer Speech Commands model and its feature
preprocessing. Use the earlier matching checkpoint containing `state_dict`,
class names and normalization statistics.

```bash
python examples/12_gru_speech.py --device cuda:0 \
  --checkpoint /path/to/gru_speech_commands.pt --data-root data/speech
python examples/12_gru_speech.py --device cuda:0 \
  --checkpoint /path/to/gru_speech_commands.pt --wav sample.wav
python examples/13_yolo_voc.py --device cuda:0 \
  --checkpoint /path/to/yolo_voc.pt --data-yaml data/voc.yaml
```

Speech Commands v0.02 needs its `testing_list.txt`; WAV input is mono 16 kHz.
The detector needs a checkpoint trained for VOC's 20 classes and an
Ultralytics dataset YAML. It resolves local data without dataset downloads,
fuses Conv/BatchNorm before conversion and invokes upstream AP evaluation.

## BERT And Language Models

BERT/SQuAD loads a local QA checkpoint and fast tokenizer, converts the
network, then reconstructs answer spans using context offsets and overflow
windows. SQuAD v1 JSON and Hugging Face parquet inputs are supported.

```bash
python examples/14_bert_squad.py --device cuda:0 \
  --checkpoint /path/to/bert-base-uncased-squad-v1 \
  --data-file /path/to/squad/validation.parquet --limit 32
python examples/15_llm_lm_eval.py --device cuda:0 \
  --checkpoint /path/to/llama --text-file data/evaluation.txt
python examples/15_llm_lm_eval.py --device cuda:0 \
  --checkpoint /path/to/qwen --tasks hellaswag --limit 100
```

Language checkpoints contain config, weights and tokenizer. These loaders
disable downloads and remote-code execution. The language head is digital
by default. Evaluation supports a local text corpus or task suites through
lm-evaluation-harness.
The measured BERT example and simulation settings are listed below.

## Small Execution Checks

`--smoke` uses synthetic inputs or tiny random models and reports
`synthetic: true`.

```bash
python examples/01_matrix_multiplication.py --smoke
python examples/02_mlp_mnist.py --stage hat --smoke --hidden-sizes 16 8
python examples/03_lenet5_mnist.py --smoke
python examples/05_vgg_cifar.py --smoke --digital
python examples/10_mobilenetv2_imagenet.py --smoke --digital
python examples/15_llm_lm_eval.py --family qwen2 --smoke
```

## Example Results

### LeNet-5 / MNIST

The included digital checkpoint was trained from scratch on all 60,000 training
images for 8 epochs, with batch 128, Adam at 0.001, seed 135 and FP32 digital
execution. Test normalization is `(x-0.5)/0.5`, without augmentation.

| Evaluation | Test images | Correct | Top-1 |
| --- | ---: | ---: | ---: |
| Digital FP32 | 10,000 | 9,849 | 98.49% |
| Static A8W4, ADC6, write variation 0.02 | 10,000 | 9,835 | 98.35% |

The static artifact contains signed INT4 weights and calibrated INT8 input
scales for all five matrices. Calibration uses 1,024 training images and seed
135. Reloading with batch 97 reproduces the score without online scale
estimation or FP32 weight copies.

### BERT / SQuAD

The local `csarron/bert-base-uncased-squad-v1` checkpoint evaluated on the first
32 SQuAD v1 validation questions gives **EM 87.50, F1 96.5625**.
The example reconstructs answer spans from context offsets and overflow windows.

The model uses FP32, context length 384, stride 128 and batch 4. CIM uses
A8W8 INT, binary slices, 64x64 arrays, 64 rows per read, ADC6,
per-vector input scales, per-output-channel weight scales, and seed 135.
Write variation is 0.02; other device effects are disabled.

### Notebook Results

Each notebook retains its executed tables, printed metrics and images.
The VGG physical-effects notebook presents a separate model and parameter table
for each nonideality. Its sweeps and the mixed-precision comparisons evaluate
all 10,000 CIFAR-10 test images.

The MobileNetV2 notebook evaluates all 50,000 ImageNet validation images with
torchvision's default pretrained weights and preprocessing:

| Conversion | Correct | Top-1 |
| --- | ---: | ---: |
| Digital FP32 | 36,075 | 72.150% |
| Full CIM | 35,941 | 71.882% |
| CIM with first convolution and classifier digital | 35,940 | 71.880% |

These CIM results use A8W8 INT, one-bit slices, 64 x 64 arrays, ADC6 and
write variation 0.02; other device effects are disabled.

The DFT notebook maps real and imaginary DFT components to CIM matrix products,
then applies a fixed Gaussian low-pass filter. The 256 x 256 photograph uses
Gaussian input noise with standard deviation 0.22 and seed 135.

| Reconstruction | PSNR |
| --- | ---: |
| Noisy image | 14.680 dB |
| NumPy DFT and inverse | 20.684 dB |
| CIM, ideal devices | 20.604 dB |
| CIM, write variation 0.02 | 20.604 dB |

Both CIM runs use A8W8, one-bit slices, 64 x 64 arrays and ADC8.
The notebook displays the source, noisy image, reconstructions and filter
response together, with a 5.924 dB improvement for the CIM reconstruction.
