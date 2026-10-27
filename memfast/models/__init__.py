"""PyTorch model factories; optional task libraries are imported on demand."""

from .detection import load_detector
from .language import load_causal_lm
from .lenet import LeNet5
from .mlp import MLP
from .recurrent import RNNClassifier
from .vision import CIFAR_MODELS, IMAGENET_MODELS, cifar_model, imagenet_model

__all__ = [
    "MLP",
    "LeNet5",
    "RNNClassifier",
    "CIFAR_MODELS",
    "IMAGENET_MODELS",
    "cifar_model",
    "imagenet_model",
    "load_causal_lm",
    "load_detector",
]
