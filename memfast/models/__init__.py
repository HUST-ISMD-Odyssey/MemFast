"""PyTorch model factories; optional task libraries are imported on demand."""

from .detection import load_detector
from .language import load_causal_lm
from .mlp import MLP
from .recurrent import RNNClassifier
from .vision import CIFAR_MODELS, IMAGENET_MODELS, cifar_model, imagenet_model

__all__ = [
    "MLP",
    "RNNClassifier",
    "CIFAR_MODELS",
    "IMAGENET_MODELS",
    "cifar_model",
    "imagenet_model",
    "load_causal_lm",
    "load_detector",
]
