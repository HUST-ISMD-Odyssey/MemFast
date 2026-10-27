"""A logits-producing multilayer perceptron for image classification."""

from torch import nn


class MLP(nn.Sequential):
    """Flatten inputs and apply Linear/ReLU layers followed by class logits."""

    def __init__(self, input_features=784, hidden_sizes=(512, 128), num_classes=10):
        sizes = (input_features, *hidden_sizes, num_classes)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("layer dimensions must be positive integers")
        layers = [nn.Flatten()]
        for index, (inputs, outputs) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if index < len(sizes) - 2:
                layers.append(nn.ReLU())
        super().__init__(*layers)
