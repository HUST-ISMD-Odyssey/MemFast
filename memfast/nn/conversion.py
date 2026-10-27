"""Model conversion with explicit selection and shared-module semantics."""

import copy

from torch import nn

from ..simulator import AnalogMatrix
from .convolution import AnalogConv2d
from .linear import AnalogLinear
from .recurrent import AnalogGRU, AnalogGRUCell, AnalogRNN, AnalogRNNCell
from .training import TrainableAnalogConv2d, TrainableAnalogLinear, _validate_gradient_mode


def _selection_paths(value, name):
    if isinstance(value, str):
        raise TypeError(f"{name} must be a sequence of module paths")
    try:
        paths = tuple(value)
    except TypeError:
        raise TypeError(f"{name} must be a sequence of module paths") from None
    if any(not isinstance(path, str) for path in paths):
        raise TypeError(f"{name} module paths must be strings")
    return set(paths)


def convert_model(
    model,
    simulator,
    *,
    include=None,
    exclude=(),
    inplace=False,
    training=False,
    gradient_mode="quantized",
):
    """Convert Linear, Conv2d, RNN/GRU cells and sequences to analog execution.

    Default conversion is frozen inference. ``training=True`` retains trainable
    master parameters and uses analog forward values with quantized-operand STE
    gradients. ``gradient_mode="digital"`` opts into unquantized derivatives.
    Conv2d gradients use independently quantized unfolded patches, without the
    legacy overlap-averaging reconstruction. See the training module's docs.
    ``include=None`` selects everything; an empty sequence selects nothing.
    Named paths select their entire subtrees, and ``""`` selects the root.
    Exclusions take precedence and also include descendants. A shared module
    converts only if every alias is included and none is excluded, keeping
    unselected aliases digital. Existing analog layers are atomic: select their
    own paths, not their internal programmed matrices. Reconversion preserves
    programming and requires matching physical config, device, and mode.
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
    included = None if include is None else _selection_paths(include, "include")
    excluded = _selection_paths(exclude, "exclude")
    analog_types = (AnalogMatrix, AnalogConv2d, AnalogRNN, AnalogRNNCell, AnalogGRU, AnalogGRUCell)
    trainable_types = (TrainableAnalogLinear, TrainableAnalogConv2d)
    supported_types = (nn.Linear, nn.Conv2d, nn.RNNCell, nn.RNN, nn.GRUCell, nn.GRU)
    by_path = dict(model.named_modules(remove_duplicate=False))
    for label, selection in (("included", included), ("excluded", excluded)):
        missing = (selection or set()) - by_path.keys()
        if missing:
            raise ValueError(f"unknown {label} modules: {sorted(missing)}")
        for path in selection or ():
            parts = path.split(".")
            for depth in range(len(parts)):
                ancestor = ".".join(parts[:depth])
                if ancestor != path and isinstance(
                    by_path[ancestor], analog_types + trainable_types
                ):
                    raise ValueError(
                        f"{path}: selection inside a converted layer is unsupported; "
                        f"select its path {ancestor!r}"
                    )

    def excluded_path(name):
        return any(path == "" or name == path or name.startswith(path + ".") for path in excluded)

    def included_path(name):
        return included is None or any(
            path == "" or name == path or name.startswith(path + ".") for path in included
        )

    target = model if inplace else copy.deepcopy(model)
    aliases = {}
    for name, module in target.named_modules(remove_duplicate=False):
        aliases.setdefault(id(module), []).append(name)
    excluded_ids = {
        identity
        for identity, names in aliases.items()
        if any(excluded_path(name) for name in names)
    }
    selected_ids = {
        identity
        for identity, names in aliases.items()
        if identity not in excluded_ids and all(included_path(name) for name in names)
    }
    internal_ids = {
        id(child)
        for module in target.modules()
        if isinstance(module, analog_types + trainable_types)
        for child in module.modules()
        if child is not module
    }
    for name, module in target.named_modules():
        if isinstance(module, AnalogMatrix) and module.simulator.device != simulator.device:
            raise ValueError(
                f"{name}: existing analog matrices are bound to their simulator device"
            )
        if id(module) in internal_ids or id(module) not in selected_ids:
            continue
        if isinstance(module, nn.Conv2d) and isinstance(module.padding, str):
            raise ValueError(f"{name}: numeric Conv2d padding is required")
        if isinstance(module, analog_types + trainable_types):
            if (
                training
                and isinstance(module, analog_types)
                and not getattr(module, "_hardware_aware_training", False)
            ):
                raise ValueError("training conversion requires the original digital layer weights")
            for child in module.modules():
                if isinstance(child, AnalogMatrix) and (
                    child.config != simulator.config or child.simulator.mode != simulator.mode
                ):
                    raise ValueError(
                        f"{name}: reconversion requires matching physical config and mode"
                    )
    target.to(simulator.device)
    replacements = {}

    def bind_cell(result, source, suffix=""):
        for projection, prefix in ((result.input, "ih"), (result.hidden, "hh")):
            projection.weight = getattr(source, "weight_" + prefix + suffix)
            projection.bias = getattr(source, "bias_" + prefix + suffix, None)

    def convert(module, name):
        if id(module) in excluded_ids:
            return module
        if (
            isinstance(module, supported_types + analog_types + trainable_types)
            and id(module) not in selected_ids
        ):
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
                    if isinstance(
                        child,
                        trainable_types + (AnalogRNN, AnalogRNNCell, AnalogGRU, AnalogGRUCell),
                    ):
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
        elif isinstance(module, (nn.RNNCell, nn.GRUCell)):
            cell_type = AnalogGRUCell if isinstance(module, nn.GRUCell) else AnalogRNNCell
            result = cell_type(
                module,
                simulator,
                training=training,
                name=name or ("gru_cell" if isinstance(module, nn.GRUCell) else "rnn_cell"),
                gradient_mode=gradient_mode,
            )
        elif isinstance(module, (nn.RNN, nn.GRU)):
            recurrent_type = AnalogGRU if isinstance(module, nn.GRU) else AnalogRNN
            result = recurrent_type(
                module,
                simulator,
                training=training,
                name=name or ("gru" if isinstance(module, nn.GRU) else "rnn"),
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
        elif training and isinstance(result, (AnalogRNNCell, AnalogGRUCell)):
            bind_cell(result, module)
        elif training and isinstance(result, (AnalogRNN, AnalogGRU)):
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
