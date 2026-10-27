"""Elman and GRU recurrence using independently programmed analog projections."""

import torch
from torch import nn
from torch.nn import functional as F

from .linear import AnalogLinear
from .training import TrainableAnalogLinear, _SurrogateGradient, _validate_gradient_mode


def _linear(weight, bias, simulator, name, training, gradient_mode):
    source = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    source.weight, source.bias = weight, bias
    if training:
        return TrainableAnalogLinear(source, simulator, name=name, gradient_mode=gradient_mode)
    return AnalogLinear(source, simulator, name=name)


def _hidden_state(x, hx, input_size, hidden_size, kind):
    if (
        not torch.is_tensor(x)
        or x.ndim not in (1, 2)
        or x.shape[-1] != input_size
        or not x.is_floating_point()
    ):
        raise ValueError(f"{kind} input must be floating [input_size] or [batch, input_size]")
    shape = (*x.shape[:-1], hidden_size)
    if hx is None:
        return x.new_zeros(shape)
    if not torch.is_tensor(hx) or hx.shape != shape or hx.device != x.device or hx.dtype != x.dtype:
        raise ValueError("hidden state must match the input batch, device, and dtype")
    return hx


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
        hx = _hidden_state(x, hx, self.input_size, self.hidden_size, "RNNCell")
        value = self.input(x) + self.hidden(hx)
        activation = torch.tanh if self.nonlinearity == "tanh" else F.relu
        analog = activation(value.detach())
        if self._hardware_aware_training and torch.is_grad_enabled():
            return _SurrogateGradient.apply(analog, activation(value.float()))
        return analog


class AnalogGRUCell(nn.Module):
    """PyTorch's reset/update/new gates with input and hidden analog projections.

    The reset gate multiplies the full hidden candidate projection, including
    its bias, after the matrix product, matching torch.nn.GRUCell. Gate order
    is reset, update, new. HAT uses FP32 gate derivatives evaluated at the
    analog projection values; the exact analog forward retains the input dtype.
    """

    def __init__(
        self, source, simulator, *, training=False, name="gru_cell", gradient_mode="quantized"
    ):
        super().__init__()
        if not isinstance(source, nn.GRUCell):
            raise TypeError("source must be torch.nn.GRUCell")
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self._hardware_aware_training = training
        self.input_size, self.hidden_size, self.bias = (
            source.input_size,
            source.hidden_size,
            source.bias,
        )
        self.input = _linear(
            source.weight_ih, source.bias_ih, simulator, f"{name}.input", training, gradient_mode
        )
        self.hidden = _linear(
            source.weight_hh, source.bias_hh, simulator, f"{name}.hidden", training, gradient_mode
        )
        self.train(source.training)

    @staticmethod
    def _gates(inputs, hidden, hx):
        ir, iz, inn = inputs.chunk(3, dim=-1)
        hr, hz, hn = hidden.chunk(3, dim=-1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        return (1 - update) * candidate + update * hx

    def forward(self, x, hx=None):
        hx = _hidden_state(x, hx, self.input_size, self.hidden_size, "GRUCell")
        inputs, hidden = self.input(x), self.hidden(hx)
        analog = self._gates(inputs.detach(), hidden.detach(), hx.detach())
        if self._hardware_aware_training and torch.is_grad_enabled():
            surrogate = self._gates(inputs.float(), hidden.float(), hx.float())
            return _SurrogateGradient.apply(analog, surrogate)
        return analog


class _AnalogRecurrent(nn.Module):
    """Common dense sequence traversal for recurrent cells."""

    def __init__(self, source, simulator, *, training=False, name=None, gradient_mode="quantized"):
        super().__init__()
        if not isinstance(source, self.source_type):
            raise TypeError(f"source must be torch.nn.{self.kind}")
        name = self.kind.lower() if name is None else name
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self._hardware_aware_training = training
        for attr in (
            "input_size",
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
        ):
            setattr(self, attr, getattr(source, attr))
        if isinstance(source, nn.RNN):
            self.nonlinearity = source.nonlinearity
        self.cells = nn.ModuleList()
        directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            width = self.input_size if layer == 0 else self.hidden_size * directions
            for direction in range(directions):
                suffix = f"_l{layer}" + ("_reverse" if direction else "")
                options = {"nonlinearity": self.nonlinearity} if isinstance(source, nn.RNN) else {}
                cell = self.cell_type(
                    width,
                    self.hidden_size,
                    bias=self.bias,
                    device="meta",
                    **options,
                )
                for attr in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    setattr(cell, attr, getattr(source, attr + suffix, None))
                self.cells.append(
                    self.analog_cell_type(
                        cell,
                        simulator,
                        training=training,
                        name=f"{name}.layer{layer}.direction{direction}",
                        gradient_mode=gradient_mode,
                    )
                )
        self.train(source.training)

    def forward(self, x, hx=None):
        if (
            not torch.is_tensor(x)
            or x.ndim not in (2, 3)
            or x.shape[-1] != self.input_size
            or not x.is_floating_point()
        ):
            raise ValueError(
                f"{self.kind} input must be a floating dense sequence with matching input_size"
            )
        unbatched = x.ndim == 2
        sequence = x.unsqueeze(1) if unbatched else x.transpose(0, 1) if self.batch_first else x
        if sequence.shape[0] == 0:
            raise ValueError(f"{self.kind} sequence must contain at least one timestep")
        directions = 2 if self.bidirectional else 1
        shape = (self.num_layers * directions, sequence.shape[1], self.hidden_size)
        expected = (shape[0], shape[2]) if unbatched else shape
        if hx is None:
            state = x.new_zeros(shape)
        else:
            if (
                not torch.is_tensor(hx)
                or hx.shape != expected
                or hx.device != x.device
                or hx.dtype != x.dtype
            ):
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
                if self._hardware_aware_training and torch.is_grad_enabled():
                    # Share one physical-precision dropout mask between the
                    # exact forward and its FP32 derivative graph.
                    mask = F.dropout(
                        torch.ones_like(sequence), p=self.dropout, training=self.training
                    )
                    sequence = _SurrogateGradient.apply(
                        sequence.detach() * mask, sequence.float() * mask.float()
                    )
                else:
                    sequence = F.dropout(sequence, p=self.dropout, training=self.training)
        hidden = torch.stack(final)
        if unbatched:
            return sequence.squeeze(1), hidden.squeeze(1)
        output = sequence.transpose(0, 1) if self.batch_first else sequence
        return output, hidden


class AnalogRNN(_AnalogRecurrent):
    """Wrap an nn.RNN with stacked, bidirectional, and unbatched dense input.

    Returns ``(output, h_n)`` with PyTorch shapes. PackedSequence is unsupported.
    Every timestep reads both projections of every direction. HAT gradient mode
    propagates to all projections.
    """

    source_type, cell_type, analog_cell_type, kind = nn.RNN, nn.RNNCell, AnalogRNNCell, "RNN"


class AnalogGRU(_AnalogRecurrent):
    """Wrap an nn.GRU using PyTorch gate equations and dense sequence shapes.

    Supports stacked layers, both directions, dropout between layers, explicit
    initial state, and unbatched input. PackedSequence is unsupported. Both
    projections (all three gates) execute in analog at each timestep. Training
    retains trainable master parameters with FP32 HAT surrogate arithmetic.
    """

    source_type, cell_type, analog_cell_type, kind = nn.GRU, nn.GRUCell, AnalogGRUCell, "GRU"
