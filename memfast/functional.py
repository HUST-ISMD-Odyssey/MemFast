"""Natural-order analog matrix products with FP32 quantized-operand STE.

``mm(A, B)`` and ``matmul(A, B)`` accept operands in ordinary A @ B order.
An explicit simulator can cache RHS programming through its internal matrix
cache. Without one, each call creates a new simulator; no global state is used.
Shapes follow torch.mm and torch.matmul; operands must be real floating dense
tensors with the same dtype and device.

A shared RHS quantizes all its input vectors together in flattened row order.
Each independently indexed batched RHS is programmed separately. Broadcast
copies of one RHS share that program and quantize their input vectors together.
The default grouped input quantization therefore agrees with the matrix core.
Batch-coupled quantization and stochastic addressing follow these program
boundaries; they do not promise equivalence to separately issued calls.

Forward values come directly from the analog simulator. Backward uses the
existing quantized-operand STE, with FP32 arithmetic even for wide/BFP codes.
No derivatives pass through device effects. Leaf gradients retain leaf dtype.
"""

import itertools
import weakref
from pathlib import Path

import torch

from .config import Config
from .nn.training import _quantized_linear, _SurrogateGradient
from .simulator import Simulator


def _operands(a, b):
    for name, value in (("A", a), ("B", b)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided or value.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise TypeError(f"{name} must be a real floating dense tensor")
        if value.ndim == 0:
            raise ValueError(f"{name} must have at least one dimension")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    if a.device != b.device:
        raise ValueError("A and B must use the same device")
    if a.device.type not in ("cpu", "cuda"):
        raise ValueError("matrix simulation requires CPU or CUDA operands")
    for name, value in (("A", a), ("B", b)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def _engine(a, simulator, config, mode, cache_dir):
    if mode not in ("default", "tuned"):
        raise ValueError("mode must be default or tuned")
    if config is not None and not isinstance(config, Config):
        raise TypeError("config must be Config")
    if simulator is None:
        return Simulator(config, mode=mode, device=a.device, cache_dir=cache_dir)
    if not isinstance(simulator, Simulator):
        raise TypeError("simulator must be Simulator")
    if simulator.device != a.device:
        raise ValueError("operands and simulator must use the same device")
    if config is not None and config != simulator.config:
        raise ValueError("config differs from the supplied simulator")
    if mode != "default" and mode != simulator.mode:
        raise ValueError("mode differs from the supplied simulator")
    if cache_dir is not None and Path(cache_dir) != simulator.cache_dir:
        raise ValueError("cache_dir differs from the supplied simulator")
    return simulator


def _out(a, b, out, shape):
    if out is None:
        return
    if not isinstance(out, torch.Tensor) or out.layout != torch.strided:
        raise TypeError("out must be a dense torch.Tensor")
    if out.shape != shape or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("out must have the result shape, dtype, and device")
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad or out.requires_grad):
        raise RuntimeError("out= does not support automatic differentiation")


def _map_rhs(simulator, b):
    mapper = getattr(simulator, "_matrix_for", None)
    if mapper is not None:
        return mapper(b)
    # Legacy map takes PyTorch [O,K] weights; sampling names remain automatic.
    return simulator.map(b.detach().transpose(0, 1))


def _rhs_view(simulator, owner, rhs, index):
    """Keep stable detached cache keys without retaining the caller's tensor."""
    cache = getattr(simulator, "_functional_rhs_views", None)
    if cache is None:
        cache = simulator._functional_rhs_views = {}
    identity = id(owner)
    geometry = (
        tuple(owner.shape),
        tuple(owner.stride()),
        owner.storage_offset(),
        owner.data_ptr(),
        owner.dtype,
        owner.device,
        owner.is_neg(),
    )
    record = cache.get(identity)
    if record is None or record[0]() is not owner or record[1] != geometry:
        engine = weakref.ref(simulator)

        def release(reference):
            instance = engine()
            if instance is not None:
                entries = instance._functional_rhs_views
                previous = entries.get(identity)
                if previous is not None and previous[0] is reference:
                    entries.pop(identity, None)

        record = (weakref.ref(owner, release), geometry, {})
        cache[identity] = record
    if index not in record[2]:
        # detach shares storage/version but has no Python reference to owner.
        record[2][index] = rhs.detach()
    return record[2][index]


def _product(a, b, simulator, cached_rhs=None):
    matrix = _map_rhs(simulator, b if cached_rhs is None else cached_rhs)
    analog = matrix(a.detach())
    if not torch.is_grad_enabled() or not (a.requires_grad or b.requires_grad):
        return analog
    surrogate = _quantized_linear(a, b.transpose(0, 1), None, simulator.config)
    return _SurrogateGradient.apply(analog, surrogate)


def _matmul(a, b, *, simulator, config, mode, cache_dir, out):
    _operands(a, b)
    left_vector, right_vector = a.ndim == 1, b.ndim == 1
    left = a.unsqueeze(0) if left_vector else a
    right = b.unsqueeze(-1) if right_vector else b
    if left.shape[-1] != right.shape[-2]:
        raise ValueError("A and B have incompatible reduction dimensions")
    try:
        batch = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    except RuntimeError:
        raise ValueError("A and B batch dimensions cannot broadcast") from None
    rows, reduction, columns = left.shape[-2], left.shape[-1], right.shape[-1]
    shape = batch + (() if left_vector else (rows,)) + (() if right_vector else (columns,))
    _out(a, b, out, shape)
    engine = _engine(a, simulator, config, mode, cache_dir)
    with torch.autocast(device_type=a.device.type, enabled=False):
        if a.numel() == 0 or b.numel() == 0 or 0 in shape:
            result = a.new_zeros(shape)
            if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
                result = _SurrogateGradient.apply(result, torch.matmul(a.float(), b.float()))
        else:
            left = left.expand(batch + (rows, reduction))
            rhs_batch = right.shape[:-2]
            if all(size == 1 for size in rhs_batch):
                rhs = right if right.ndim == 2 else right.reshape(reduction, columns)
                cached = None if b.ndim == 2 else _rhs_view(engine, b, rhs, ())
                result = _product(left, rhs, engine, cached)
            else:
                groups = {}
                padding = len(batch) - len(rhs_batch)
                for index in itertools.product(*(range(size) for size in batch)):
                    key = tuple(
                        0 if size == 1 else index[padding + axis]
                        for axis, size in enumerate(rhs_batch)
                    )
                    groups.setdefault(key, []).append(index)
                outputs = {}
                for key, indices in groups.items():
                    inputs = torch.stack([left[index] for index in indices])
                    rhs = right[key]
                    values = _product(inputs, rhs, engine, _rhs_view(engine, b, rhs, key))
                    outputs.update((index, values[row]) for row, index in enumerate(indices))
                result = torch.stack(
                    [
                        outputs[index]
                        for index in itertools.product(*(range(size) for size in batch))
                    ]
                ).reshape(batch + (rows, columns))
            result = result.reshape(shape)
    if out is not None:
        out.copy_(result)
        return out
    return result


def mm(input, mat2, *, simulator=None, config=None, mode="default", cache_dir=None, out=None):
    """Simulate a rank-two A @ B with automatic RHS programming.

    A supplied simulator owns configuration and cache selection. Explicit
    configuration/cache options must agree with it; the default mode leaves
    its selected mode unchanged. ``out`` requires the exact result shape and
    supports inference only. An explicit simulator retains its RHS cache.
    """
    if not isinstance(input, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor operands")
    if input.ndim != 2 or mat2.ndim != 2:
        raise ValueError("mm requires two rank-two tensors")
    return _matmul(
        input, mat2, simulator=simulator, config=config, mode=mode, cache_dir=cache_dir, out=out
    )


def matmul(input, other, *, simulator=None, config=None, mode="default", cache_dir=None, out=None):
    """Simulate torch.matmul vector promotion and broadcast batch semantics.

    Each independently indexed RHS matrix has its own analog program. Broadcast
    copies reuse that program within this call. See this module's docs for
    input grouping and FP32 STE semantics. No caller transpose/name is needed.
    """
    return _matmul(
        input, other, simulator=simulator, config=config, mode=mode, cache_dir=cache_dir, out=out
    )


__all__ = ["mm", "matmul"]
