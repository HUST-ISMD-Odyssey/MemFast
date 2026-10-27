"""CIFAR adaptations and upstream ImageNet architectures."""

from torch import nn
from torchvision import models
from torchvision.models.vgg import cfgs, make_layers

CIFAR_MODELS = (
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
)
IMAGENET_MODELS = (
    "vgg11_bn",
    "vgg16_bn",
    "resnet18",
    "resnet50",
    "resnet101",
    "mobilenet_v2",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
)


class CifarVGG(nn.Module):
    """Legacy CIFAR VGG layout with torchvision feature layers and raw logits."""

    def __init__(self, name="vgg11_bn", num_classes=10):
        super().__init__()
        key = {"vgg11_bn": "A", "vgg13_bn": "B", "vgg16_bn": "D", "vgg19_bn": "E"}[name]
        self.features = make_layers(cfgs[key], batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)).flatten(1))


def cifar_model(name="vgg11_bn", *, num_classes=10):
    """Build an untrained CIFAR model; load task-specific weights explicitly.

    ResNet uses the legacy 3x3, stride-one, unpadded stem without max pooling.
    This preserves the state layout and spatial convention of the old models.
    ImageNet pretrained weights are not substituted for CIFAR weights.
    """
    if name not in CIFAR_MODELS:
        raise ValueError(f"unknown CIFAR model: {name}")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if name.startswith("vgg"):
        return CifarVGG(name, num_classes)
    model = models.get_model(name, weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=0, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model


def imagenet_model(name="resnet18", *, weights=None, num_classes=1000):
    """Use torchvision, or timm for DeiT; weights are never downloaded implicitly.

    ``weights="DEFAULT"`` explicitly requests upstream pretrained weights.
    DeiT also accepts ``"DEFAULT"``; custom DeiT weights use a local checkpoint.
    """
    if name not in IMAGENET_MODELS:
        raise ValueError(f"unknown ImageNet model: {name}")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if name.startswith("deit_"):
        if weights not in (None, "DEFAULT"):
            raise ValueError("DeiT weights must be None or DEFAULT")
        try:
            import timm
        except ImportError as exc:
            raise ImportError("DeiT requires the optional timm package") from exc
        return timm.create_model(name, pretrained=weights is not None, num_classes=num_classes)
    if isinstance(weights, str):
        weights = models.get_model_weights(name)[weights]
    return models.get_model(name, weights=weights, num_classes=num_classes)
