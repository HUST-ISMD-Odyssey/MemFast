# Examples

Run commands from the repository root after `python -m pip install -e .`.
PyTorch and torchvision are standard dependencies. Install optional
integrations with `python -m pip install -e ".[models]"` for DeiT, local
Hugging Face language models, lm-evaluation-harness, and Ultralytics detection.
Individual optional packages can also be installed separately.

| Entry point | Task |
| --- | --- |
| `matrix.py` | Matrix programming, INT/BFP, slices, ADCs, geometry and scale groups |
| `model.py` | Small convolutional model conversion |
| `nonidealities.py` | Independent device-effect examples |
| `mnist.py` | MNIST MLP training and inference |
| `mlp_hat.py` | MNIST hardware-aware training |
| `mlp_hat_ddp.py` | Distributed hardware-aware training with torchrun |
| `cifar.py` | VGG/ResNet evaluation and digital fine-tuning on CIFAR-10/100 |
| `imagenet.py` | VGG, ResNet, MobileNetV2 or DeiT ImageFolder evaluation |
| `rnn.py` | Recurrent MNIST row-sequence training and inference |
| `llama_lm_eval.py` | Local Llama/Qwen generation, perplexity and harness tasks |
| `yolo_voc.py` | Ultralytics YOLO inference and VOC validation |
| `cifar10.py` | Existing pretrained VGG11-BN nonideality scan |

## Synthetic Smoke Runs

`--smoke` uses synthetic inputs and small sample counts. MLP and recurrent
examples also reduce hidden dimensions. Vision examples retain the named
upstream architecture. Language-model smoke runs create a tiny random decoder
and an in-memory harness task; YOLO creates an untrained upstream architecture.
Smoke results are marked `synthetic: true` and are execution checks.

```bash
python -m examples.matrix --smoke
python -m examples.mnist --smoke
python -m examples.mlp_hat --smoke
python -m examples.mlp_hat_ddp --smoke
python -m examples.rnn --smoke
python -m examples.cifar --model vgg11_bn --smoke --digital
python -m examples.cifar --model resnet18 --smoke --digital
python -m examples.imagenet --model mobilenet_v2 --smoke --digital
python -m examples.imagenet --model deit_tiny_patch16_224 --smoke --digital
python -m examples.llama_lm_eval --family llama --smoke
python -m examples.llama_lm_eval --family qwen2 --smoke
python -m examples.yolo_voc --smoke --digital
```

Remove `--digital` from a vision smoke command to exercise analog conversion
of the entire architecture. The small matrix, MLP, HAT and recurrent smoke
commands use analog execution by default.

## Matrix Configuration

```bash
python -m examples.matrix --activation-bits 10 --weight-bits 10 \
  --input-slice 1,3,3,3 --weight-slice 1,4,5 --adc-bits 6,8,10 \
  --array-size 16,32 --rows-per-read 4 \
  --input-quant-gran none,4 --weight-quant-gran 4,2

python -m examples.matrix --activation-bits 16 --weight-bits 16 \
  --input-slice 1,5,5,5 --weight-slice 1,5,5,5 --bw-e 8 \
  --adc-bits 12 --array-size 32,64 --rows-per-read 8 \
  --input-quant-gran 2,8 --weight-quant-gran 8,4
```

Bit widths range from 2 through 32, subject to the engine's signed 64-bit
accumulator bound. Slice tuples are MSB first, begin with a one-bit sign
slice, and sum to the corresponding bit width. Omit `--bw-e` for INT;
specifying an exponent width selects BFP. `--adc-bits` accepts a scalar
shared by all weight slices or a comma-separated tuple with exactly one
entry per weight slice in MSB-first order.

`--array-size` is physical rows and columns. `--rows-per-read` sets the
physical rows participating in each ADC read. Quantization groups use two
comma-separated sizes; `none` spans the corresponding dimension. Inputs
use `[vectors, reduction]` groups and weights use `[reduction, outputs]`.
Output includes the actual configuration, replay check and error relative
to the original floating-point matrix product.

## MNIST And Hardware-Aware Training

Train a digital model and evaluate its converted counterpart:

```bash
python -m examples.mnist --train --download --data-root data/mnist \
  --epochs 10 --save-checkpoint results/mnist.pt
python -m examples.mnist --data-root data/mnist --checkpoint results/mnist.pt
```

Hardware-aware training uses trainable master parameters, analog forward
values, and quantized-operand surrogate gradients by default. Select
`--gradient-mode digital` explicitly for unquantized surrogate gradients.
`--digital` instead trains and evaluates the original PyTorch network.

```bash
python -m examples.mlp_hat --download --data-root data/mnist \
  --epochs 10 --write-variation 0.02 --save-checkpoint results/mnist_hat.pt
python -m examples.mnist --data-root data/mnist --checkpoint results/mnist_hat.pt

torchrun --standalone --nproc-per-node=2 -m examples.mlp_hat_ddp \
  --data-root data/mnist --epochs 10 --save-checkpoint results/mnist_ddp.pt
torchrun --standalone --nproc-per-node=2 -m examples.mlp_hat_ddp --smoke --limit 3
```

Each process receives a training shard. Evaluation shards visit each sample
exactly once, even with uneven sample counts. Metrics are reduced across
ranks, and rank zero saves master weights and reports results. Pass the same
`--hidden-sizes` when loading a custom MLP architecture. Smoke MLPs use a
single hidden layer of size 16.

## CIFAR And ImageNet

CIFAR models require a matching task checkpoint, or `--train` to train or
fine-tune. Checkpoints load strictly. VGG uses the legacy three-layer CIFAR
classifier; ResNet uses the legacy unpadded 3-by-3 stem without max pooling.

```bash
python -m examples.cifar --model vgg16_bn --num-classes 100 \
  --data-root data/cifar --checkpoint checkpoints/vgg16_bn_cifar100.pt
python -m examples.cifar --model resnet18 --num-classes 10 \
  --data-root data/cifar --checkpoint checkpoints/resnet18_cifar10.pt
python -m examples.cifar --model vgg16_bn --num-classes 100 --train --download \
  --data-root data/cifar --epochs 10 --save-checkpoint results/vgg16_bn_cifar100.pt
```

The ImageNet root is the evaluation split itself, arranged as
`val/class_name/image.JPEG`. Class directories must match the checkpoint's
class index order. The class count is checked. `--weights DEFAULT` explicitly
requests upstream pretrained weights and their evaluation transforms.
Local torchvision checkpoints use standard resize/crop and ImageNet
normalization; DeiT uses timm's resolved data configuration.

```bash
python -m examples.imagenet --model resnet50 --weights DEFAULT --data-root data/imagenet/val
python -m examples.imagenet --model vgg16_bn --checkpoint checkpoints/vgg16_bn.pt \
  --data-root data/imagenet/val
python -m examples.imagenet --model mobilenet_v2 --weights DEFAULT --data-root data/imagenet/val
python -m examples.imagenet --model deit_base_patch16_224 --weights DEFAULT \
  --data-root data/imagenet/val
```

New classification examples emit actual sample counts, top-1/top-5 fractions,
cross entropy, training history, and configuration metadata. Use `--limit 32`
for limited real-data evaluation and `--output results/task.json` to retain
results. Random initialization is permitted for synthetic smoke checks
and explicit training.

## Recurrent Classification

MNIST images become 28 timesteps of 28 features. The digital model uses
`torch.nn.RNN`; conversion maps recurrent input/hidden projections and the
classification head to MemFast.

```bash
python -m examples.rnn --train --download --data-root data/mnist \
  --epochs 10 --save-checkpoint results/rnn.pt
python -m examples.rnn --data-root data/mnist --checkpoint results/rnn.pt
```

`--hidden-size` and `--num-layers` must match the checkpoint. Training here
uses the digital recurrent model before analog inference.

## Local Language Models

Use a local Hugging Face directory containing configuration, tokenizer, and
weights for Llama, Qwen2 or Qwen3. Loading uses `local_files_only=True` and
does not enable remote model code. Attention and generation use transformers.
The vocabulary head stays digital by default; `--analog-head` converts it too.

```bash
python -m examples.llama_lm_eval --checkpoint checkpoints/llama \
  --text-file data/evaluation.txt --max-length 512 --stride 256 \
  --output results/llama_ppl.json
python -m examples.llama_lm_eval --checkpoint checkpoints/qwen \
  --prompt "Explain analog computing in one sentence." --max-new-tokens 32
python -m examples.llama_lm_eval --checkpoint checkpoints/llama \
  --tasks hellaswag,arc_easy --num-fewshot 0 --limit 32
```

Perplexity uses overlapping context windows and counts every next-token
target once. With `--text-file`, `--limit` caps input tokens; with `--tasks`,
it caps harness documents per task. Named harness tasks use their own data
sources. The `--smoke` path runs the actual HFLM adapter and harness against
four in-memory multiple-choice documents, without downloading task data.

## VOC Detection

Supply a local Ultralytics detector checkpoint with 20 VOC classes and a local
Ultralytics dataset YAML pointing to prepared images and YOLO-format labels.
Dataset class names and ordering must match the checkpoint.

```bash
python -m examples.yolo_voc --checkpoint checkpoints/yolo_voc.pt \
  --data-yaml data/voc.yaml --split test --limit 32 --output results/voc.json
```

Ultralytics handles decoding, suppression and validation. Results report its
AP50 and AP50:95 metrics. `--limit` creates a temporary deterministic image
list. The synthetic smoke path uses the same upstream prediction API and
reports detection counts.

## Existing CIFAR-10 Scan

The existing numerical scan remains available:

```bash
python examples/cifar10.py --download --limit 10000
python examples/cifar10.py --download --scan all --output results/cifar10
python examples/cifar10.py --checkpoint checkpoints/cifar10_vgg11_bn-eaeebf42.pt \
  --scan read_slice_pair --save-outputs
```

It loads pretrained VGG11-BN weights, checks the download hash, and evaluates
the actual selected CIFAR-10 test samples. CSV, metadata, Markdown tables and
optional raw logits go to the output directory. Its `top1` CSV field is a
percentage. Measured full-test tables remain in [results](results/README.md).
