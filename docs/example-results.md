# Example Results

These examples use Python 3.13, PyTorch 2.10.0+cu130 and torchvision 0.25.0.
The BERT example uses transformers 5.8.0.

## LeNet-5 / MNIST

The included [checkpoint](../examples/checkpoints/README.md) was trained
from scratch using all 60,000 MNIST training images, 8 epochs, batch 128,
Adam at 0.001, seed 135 and FP32 digital execution. Test normalization is
`(x-0.5)/0.5`, with no augmentation.

| Evaluation | Samples | Correct | Top-1 |
| --- | ---: | ---: | ---: |
| Digital FP32 | 10,000 | 9,849 | 98.49% |
| Default CIM | 10,000 | 9,850 | 98.50% |

CIM uses A8W8 INT, binary slices, 64x64 arrays, 64 rows per ADC read,
ADC6, per-vector input scales and per-output-channel weight scales.
Only write variation is enabled, at 0.02, with seed 135.

```bash
python examples/03_lenet5_mnist.py --device cuda:0 --data-root data --digital
python examples/03_lenet5_mnist.py --device cuda:0 --data-root data
```

## BERT / SQuAD

The pretrained local `csarron/bert-base-uncased-squad-v1` checkpoint was
evaluated on the first 32 SQuAD v1 validation questions: **EM 87.50,
F1 96.5625**. There were 32 tokenized features.

The model uses FP32, context length 384, document stride 128, batch 4,
and the same default CIM configuration as above. The example performs
context-offset answer reconstruction, including overflow windows.

```bash
python examples/14_bert_squad.py --device cuda:0 \
  --checkpoint /path/to/bert-base-uncased-squad-v1 \
  --data-file /path/to/squad/validation.parquet --limit 32 --batch-size 4
```

## Static Quantization

The bundled static LeNet artifact contains signed INT4 weights and calibrated
INT8 input scales for all five CIM matrices. Calibration used 1,024 MNIST
training images, seed 135. Evaluation on all 10,000 test images gave
**98.35% (9,835 correct)** with ADC6 and write variation 0.02.
Reloading with batch 97 reproduced the score with zero online scale-estimation
calls. The original FP32 weights are not required for inference.

## Guided Notebooks

| Workflow | Data and settings | Result |
| --- | --- | --- |
| Custom MLP | All 10,000 MNIST test images | Digital 96.61%; ideal CIM 96.65%; write-0.02 CIM 96.65% |
| MobileNetV2 | First 128 ImageNet validation images | Digital 88.28%; all CIM 86.72%; first convolution/head digital 87.50% |
| DFT denoising | Fixed 64x64 photograph | Noisy 17.5205 dB; NumPy 18.4623 dB; ideal/write-0.02 CIM 18.4523 dB |
| VGG nonidealities | Same 512 CIFAR-10 test images, 59 unique CIM configurations | Digital 93.945%; default CIM 94.141%; write-0.20 CIM 66.406% |
| VGG mixed precision | Same 512 images, 30 configurations plus custom ADC example | ADC8/write-0.02: uniform 8-bit 94.141%; uniform 4-bit 92.969%; editable mixed policy 94.336% |

VGG uses the pretrained VGG11-BN checkpoint
`cifar10_vgg11_bn-eaeebf42.pt`, SHA-256
`eaeebf42370c92fdfbb5dbe8eba7c27d4eb7366a1dca0b5be435364ccdf54378`.
The editable policy assigns eight bits to the first two convolutions and final
classifier, and four bits to the remaining mapped layers.
