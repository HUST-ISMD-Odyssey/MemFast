"""CIFAR VGG with the legacy three-layer classifier."""

from torch import nn
from torchvision.models.vgg import cfgs, make_layers


class CifarVGG(nn.Module):
    """Legacy CIFAR VGG layout with torchvision feature layers and raw logits."""

    def __init__(self, name="vgg11_bn", num_classes=10):
        super().__init__()
        layouts = {"vgg11_bn": "A", "vgg13_bn": "B", "vgg16_bn": "D", "vgg19_bn": "E"}
        if name not in layouts:
            raise ValueError(f"unknown CIFAR VGG architecture: {name}")
        key = layouts[name]
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
