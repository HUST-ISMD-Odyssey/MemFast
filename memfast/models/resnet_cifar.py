"""The ResNet18 architecture used by the released CIFAR checkpoints."""

from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class CifarResNet18(ResNet):
    """Unpadded 3x3 stride-one stem, no maxpool, and four groups of two blocks."""

    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.maxpool = nn.Identity()
