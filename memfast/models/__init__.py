"""Custom PyTorch architectures used by the MemFast tutorials."""

from .lenet import LeNet5
from .mlp import MLP
from .recurrent import RNNClassifier
from .resnet_cifar import CifarResNet18
from .speech import AudioGRU
from .vgg_cifar import CifarVGG


def cifar_model(name="vgg11_bn", *, num_classes=10):
    """Build a custom CIFAR architecture; used by the VGG notebooks."""
    if name == "resnet18":
        return CifarResNet18(num_classes=num_classes)
    return CifarVGG(name, num_classes=num_classes)


__all__ = [
    "MLP",
    "LeNet5",
    "RNNClassifier",
    "AudioGRU",
    "CifarVGG",
    "CifarResNet18",
    "cifar_model",
]
