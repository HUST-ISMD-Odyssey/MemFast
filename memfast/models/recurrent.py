"""Sequence classification with PyTorch's recurrent implementation."""

from torch import nn


class RNNClassifier(nn.Module):
    """Classify [batch, time, features] sequences using the final hidden state.

    The digital model uses ``torch.nn.RNN``. MemFast conversion maps both
    recurrent projections at each timestep and the Linear classification head.
    """

    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10):
        super().__init__()
        for value in (input_size, hidden_size, num_layers, num_classes):
            if type(value) is not int or value < 1:
                raise ValueError("RNN dimensions must be positive integers")
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("expected [batch, time, features]")
        _, hidden = self.rnn(x)
        return self.classifier(hidden[-1])
