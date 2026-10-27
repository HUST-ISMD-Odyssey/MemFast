"""Model conversion with shared-module and exclusion semantics."""

import copy

from torch import nn

from ..simulator import AnalogMatrix
from .layers import AnalogConv2d, AnalogLinear
from .recurrent import AnalogRNN, AnalogRNNCell
from .training import TrainableAnalogConv2d, TrainableAnalogLinear, _validate_gradient_mode


def convert_model(
    model, simulator, *, exclude=(), inplace=False, training=False, gradient_mode="quantized"
):
    """Convert Linear, Conv2d, RNNCell, and RNN to analog execution.

    Default conversion is frozen inference. ``training=True`` retains trainable
    master parameters and uses analog forward values with quantized-operand STE
    gradients. ``gradient_mode="digital"`` opts into unquantized derivatives.
    Conv2d gradients use independently quantized unfolded patches, without the
    legacy overlap-averaging reconstruction. See the training module's docs.
    Exclusions include descendants; the empty path excludes the entire model.
    If any alias of a shared module is excluded, all its aliases stay digital.
    Shared modules and shared master parameters retain their identities.
    Llama subtrees containing analog Linear layers use ``pretraining_tp=1``:
    legacy tensor partition execution reads weights directly and bypasses layer
    forwards, including analog execution and HAT. This changes execution config,
    not checkpoint tensor shapes; floating summation order can differ. Default
    copied conversion leaves the source config unchanged; ``inplace=True`` updates it.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch module")
    if not isinstance(training, bool):
        raise TypeError("training must be a boolean")
    _validate_gradient_mode(gradient_mode)
    if isinstance(exclude, str):
        raise TypeError("exclude must be a sequence of module paths")
    excluded = set(exclude)
    if any(not isinstance(path, str) for path in excluded):
        raise TypeError("excluded module paths must be strings")
    paths = {name for name, _ in model.named_modules(remove_duplicate=False)}
    missing = excluded - paths
    if missing:
        raise ValueError(f"unknown excluded modules: {sorted(missing)}")

    def excluded_path(name):
        return any(path == "" or name == path or name.startswith(path + ".") for path in excluded)

    target = model if inplace else copy.deepcopy(model)
    excluded_ids = {
        id(module)
        for name, module in target.named_modules(remove_duplicate=False)
        if excluded_path(name)
    }
    analog_types = (AnalogMatrix, AnalogConv2d, AnalogRNN, AnalogRNNCell)
    trainable_types = (TrainableAnalogLinear, TrainableAnalogConv2d)
    for name, module in target.named_modules():
        if id(module) in excluded_ids:
            continue
        if isinstance(module, nn.Conv2d) and isinstance(module.padding, str):
            raise ValueError(f"{name}: numeric Conv2d padding is required")
    target.to(simulator.device)
    replacements = {}

    def bind_cell(result, source, suffix=""):
        for projection, prefix in ((result.input, "ih"), (result.hidden, "hh")):
            projection.weight = getattr(source, "weight_" + prefix + suffix)
            projection.bias = getattr(source, "bias_" + prefix + suffix, None)

    def convert(module, name):
        if id(module) in excluded_ids:
            return module
        if isinstance(module, trainable_types):
            if training:
                module.gradient_mode = gradient_mode
            return module
        if isinstance(module, analog_types):
            if training and not getattr(module, "_hardware_aware_training", False):
                raise ValueError("training conversion requires the original digital layer weights")
            if training:
                for child in module.modules():
                    if isinstance(child, trainable_types + (AnalogRNN, AnalogRNNCell)):
                        child.gradient_mode = gradient_mode
            return module
        if id(module) in replacements:
            return replacements[id(module)]
        if isinstance(module, nn.Linear):
            if training:
                result = TrainableAnalogLinear(
                    module, simulator, name=name or "linear", gradient_mode=gradient_mode
                )
            else:
                result = AnalogLinear(module, simulator, name=name or "linear")
        elif isinstance(module, nn.Conv2d):
            if training:
                result = TrainableAnalogConv2d(
                    module, simulator, name=name or "conv", gradient_mode=gradient_mode
                )
            else:
                result = AnalogConv2d(module, simulator, name=name or "conv")
        elif isinstance(module, nn.RNNCell):
            result = AnalogRNNCell(
                module,
                simulator,
                training=training,
                name=name or "rnn_cell",
                gradient_mode=gradient_mode,
            )
        elif isinstance(module, nn.RNN):
            result = AnalogRNN(
                module,
                simulator,
                training=training,
                name=name or "rnn",
                gradient_mode=gradient_mode,
            )
        else:
            result = module
            replacements[id(module)] = result
            for attr, child in list(module._modules.items()):
                if child is not None:
                    child_name = f"{name}.{attr}" if name else attr
                    setattr(module, attr, convert(child, child_name))
            config = getattr(module, "config", None)
            if (
                getattr(config, "model_type", None) == "llama"
                and getattr(config, "pretraining_tp", 1) > 1
                and any(
                    isinstance(child, (AnalogLinear, TrainableAnalogLinear))
                    for child in module.modules()
                )
            ):
                # Legacy tensor partitions call F.linear on .weight instead of
                # invoking the converted projections, bypassing analog and HAT.
                config.pretraining_tp = 1
        if training and isinstance(result, trainable_types):
            # Reuse the copied model's Parameters, including ties to digital layers.
            result.weight, result.bias = module.weight, module.bias
        elif training and isinstance(result, AnalogRNNCell):
            bind_cell(result, module)
        elif training and isinstance(result, AnalogRNN):
            directions = 2 if result.bidirectional else 1
            for index, cell in enumerate(result.cells):
                suffix = f"_l{index // directions}"
                if index % directions:
                    suffix += "_reverse"
                bind_cell(cell, module, suffix)
        replacements[id(module)] = result
        return result

    converted = convert(target, "")
    if training:
        return converted.train()
    return converted.eval().requires_grad_(False)


def reset_read_events(model, event=0):
    """Replay a read event across all programmed matrices, including HAT layers."""
    for module in model.modules():
        if isinstance(module, AnalogMatrix):
            module.reset_read(event)
