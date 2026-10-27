"""Elman recurrence composed from independently programmed analog projections."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import AnalogLinear
from .training import TrainableAnalogLinear, _validate_gradient_mode


def _linear(weight, bias, simulator, name, training, gradient_mode):
    source = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    source.weight, source.bias = weight, bias
    if training:
        return TrainableAnalogLinear(source, simulator, name=name, gradient_mode=gradient_mode)
    return AnalogLinear(source, simulator, name=name)


class AnalogRNNCell(nn.Module):
    """Wrap an nn.RNNCell using input and hidden analog Linear projections.

    ``training=True`` enables the feedforward layers' quantized-operand STE
    by default; ``gradient_mode="digital"`` opts into unquantized derivatives.
    Both biases remain separate, as in PyTorch's Elman RNN cell.
    """

    def __init__(
        self, source, simulator, *, training=False, name="rnn_cell", gradient_mode="quantized"
    ):
        super().__init__()
        if not isinstance(source, nn.RNNCell):
            raise TypeError("source must be torch.nn.RNNCell")
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self._hardware_aware_training = training
        self.input_size, self.hidden_size = source.input_size, source.hidden_size
        self.nonlinearity, self.bias = source.nonlinearity, source.bias
        self.input = _linear(
            source.weight_ih, source.bias_ih, simulator, f"{name}.input", training, gradient_mode
        )
        self.hidden = _linear(
            source.weight_hh, source.bias_hh, simulator, f"{name}.hidden", training, gradient_mode
        )
        self.train(source.training)

    def forward(self, x, hx=None):
        if x.ndim not in (1, 2) or x.shape[-1] != self.input_size:
            raise ValueError("RNNCell input must be [input_size] or [batch, input_size]")
        shape = (*x.shape[:-1], self.hidden_size)
        if hx is None:
            hx = x.new_zeros(shape)
        if hx.shape != shape or hx.device != x.device or hx.dtype != x.dtype:
            raise ValueError("hidden state must match the input batch, device, and dtype")
        value = self.input(x) + self.hidden(hx)
        return torch.tanh(value) if self.nonlinearity == "tanh" else F.relu(value)


class AnalogRNN(nn.Module):
    """Wrap an nn.RNN, supporting stacked, bidirectional, and unbatched input.

    Returns ``(output, h_n)`` with PyTorch RNN shapes. PackedSequence inputs are
    not supported. Every timestep invokes each direction's two physical layers.
    ``gradient_mode`` is propagated to every trainable projection.
    """

    def __init__(self, source, simulator, *, training=False, name="rnn", gradient_mode="quantized"):
        super().__init__()
        if not isinstance(source, nn.RNN):
            raise TypeError("source must be torch.nn.RNN")
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self._hardware_aware_training = training
        for attr in (
            "input_size",
            "hidden_size",
            "num_layers",
            "nonlinearity",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
        ):
            setattr(self, attr, getattr(source, attr))
        self.cells = nn.ModuleList()
        directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            width = self.input_size if layer == 0 else self.hidden_size * directions
            for direction in range(directions):
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                cell = nn.RNNCell(
                    width,
                    self.hidden_size,
                    bias=self.bias,
                    nonlinearity=self.nonlinearity,
                    device="meta",
                )
                for attr in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    setattr(cell, attr, getattr(source, attr + suffix, None))
                self.cells.append(
                    AnalogRNNCell(
                        cell,
                        simulator,
                        training=training,
                        name=f"{name}.layer{layer}.direction{direction}",
                        gradient_mode=gradient_mode,
                    )
                )
        self.train(source.training)

    def forward(self, x, hx=None):
        if not torch.is_tensor(x) or x.ndim not in (2, 3) or x.shape[-1] != self.input_size:
            raise ValueError("RNN input must be a dense sequence with matching input_size")
        unbatched = x.ndim == 2
        sequence = x.unsqueeze(1) if unbatched else x.transpose(0, 1) if self.batch_first else x
        if sequence.shape[0] == 0:
            raise ValueError("RNN sequence must contain at least one timestep")
        directions = 2 if self.bidirectional else 1
        shape = (self.num_layers * directions, sequence.shape[1], self.hidden_size)
        expected = (shape[0], shape[2]) if unbatched else shape
        if hx is None:
            state = x.new_zeros(shape)
        else:
            if hx.shape != expected or hx.device != x.device or hx.dtype != x.dtype:
                raise ValueError(
                    "hidden state must match layers, directions, batch, device, and dtype"
                )
            state = hx.unsqueeze(1) if unbatched else hx
        final = []
        for layer in range(self.num_layers):
            outputs = []
            for direction in range(directions):
                index = layer * directions + direction
                hidden = state[index]
                steps = range(sequence.shape[0])
                if direction:
                    steps = reversed(steps)
                values = []
                for step in steps:
                    hidden = self.cells[index](sequence[step], hidden)
                    values.append(hidden)
                if direction:
                    values.reverse()
                outputs.append(torch.stack(values))
                final.append(hidden)
            sequence = torch.cat(outputs, dim=-1) if directions == 2 else outputs[0]
            if self.dropout and layer + 1 < self.num_layers:
                sequence = F.dropout(sequence, p=self.dropout, training=self.training)
        hidden = torch.stack(final)
        if unbatched:
            return sequence.squeeze(1), hidden.squeeze(1)
        output = sequence.transpose(0, 1) if self.batch_first else sequence
        return output, hidden
