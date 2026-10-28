"""Mapping, complete schedule traversal, and persistent selection cache."""

import hashlib
import itertools
import json
import math
import time
import weakref
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import reference
from .config import Config
from .quantization import (
    check_read_scales,
    expanded_parameter,
    input_tiles,
    pack_weight,
    read_starts,
    weight_tiles,
)


def default_schedule(config):
    return dict(arrays=2, vectors=2, columns=4, workers=min(2, len(config.weight_layout)))


def schedule_grid(config, n, partitions):
    seen = set()
    for t, v, columns, workers in itertools.product(
        (1, 2, 4, 8, 16, 32, 64),
        (1, 2, 4, 8, 16, 32),
        (2, 4, 8),
        range(1, len(config.weight_layout) + 1),
    ):
        if config.array_size[1] % columns:
            continue
        effective = min(t, partitions), (v if v <= n else 1), columns, workers
        if effective in seen:
            continue
        seen.add(effective)
        yield dict(arrays=t, vectors=v if v <= n else 1, columns=columns, workers=workers)


def source_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.cu")]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def restore_scales(value, input_scale, weight_scale):
    """Restore extremes without overflowing an otherwise finite product."""
    vm, ve = torch.frexp(value.double())
    xm, xe = torch.frexp(input_scale.double())
    wm, we = torch.frexp(weight_scale.double())
    mantissa = vm * xm * wm
    exponent = ve + xe + we
    first = exponent.clamp(-1022, 1023)
    restored = torch.ldexp(torch.ldexp(mantissa, first), exponent - first)
    return torch.where(mantissa == 0, 0.0, restored)


class Simulator:
    """Use fixed defaults or traverse the complete declared execution grid.

    ``tuned`` selects per input shape, saves the winner, and validates cached
    winners against default execution without timing candidates again.
    """

    def __init__(
        self, config=None, *, mode="default", device=None, cache_dir=None, workspace_mb=1024
    ):
        self.config = Config() if config is None else config
        if not isinstance(self.config, Config):
            raise TypeError("config must be Config")
        if mode not in ("default", "tuned"):
            raise ValueError("mode must be default or tuned")
        if isinstance(workspace_mb, bool) or not isinstance(workspace_mb, (int, float)):
            raise ValueError("workspace_mb must be positive and finite")
        if not math.isfinite(workspace_mb) or workspace_mb <= 0:
            raise ValueError("workspace_mb must be positive and finite")
        self.mode = mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            index = torch.cuda.current_device() if self.device.index is None else self.device.index
            self.device = torch.device("cuda", index)
        else:
            self.device = torch.device("cpu")
        if mode == "tuned" and self.device.type != "cuda":
            raise ValueError("schedule tuning requires CUDA; default mode supports CPU")
        from .cuda import cache_root

        self.cache_dir = Path(cache_dir) if cache_dir is not None else cache_root() / "tuning"
        self.workspace_bytes = int(workspace_mb * 1024**2)
        self.records = []
        self._selections = {}
        self._fingerprint = source_hash()
        self._next_id = 0
        self._matrix_cache = {}

    def map(self, weight, *, bias=None, name=None):
        """Low-level layer-weight adapter using PyTorch [O,K] weight layout.

        Use ``mm(A, B)`` for a matrix product in its natural operand order.
        """
        if name is None:
            name = f"matrix.{self._next_id}"
            self._next_id += 1
        return AnalogMatrix(weight, self, bias=bias, name=name)

    def mm(self, input, mat2, *, out=None):
        """Simulate a two-dimensional matrix product, following torch.mm."""
        from .functional import mm

        return mm(input, mat2, simulator=self, out=out)

    def matmul(self, input, other, *, out=None):
        """Simulate a vector, matrix, or broadcast batch product like torch.matmul."""
        from .functional import matmul

        return matmul(input, other, simulator=self, out=out)

    def _matrix_for(self, matrix):
        """Validate current packed weights before reusing RHS programming."""
        identity = id(matrix)
        record = self._matrix_cache.get(identity)
        same_tensor = record is not None and record[0]() is matrix
        metadata = (
            tuple(matrix.shape),
            tuple(matrix.stride()),
            matrix.storage_offset(),
            matrix.data_ptr(),
            matrix.dtype,
            matrix.device,
        )
        previous = record[2] if same_tensor else None
        # NumPy aliases and .data writes need not advance the tensor version.
        # Mapping uses device-local weight_tiles and retains no floating RHS copy.
        programmed = self.map(matrix.T, name=None if previous is None else previous.name)
        if (
            same_tensor
            and record[1] == metadata
            and previous.config == self.config
            and previous.packed.device == self.device
            and torch.equal(previous.packed, programmed.packed)
            and torch.equal(previous.weight_scale, programmed.weight_scale)
        ):
            return previous
        if previous is not None:
            programmed._event = previous._event
        owner = weakref.ref(self)

        def release(reference):
            simulator = owner()
            if simulator is not None:
                cached = simulator._matrix_cache.get(identity)
                if cached is not None and cached[0] is reference:
                    simulator._matrix_cache.pop(identity, None)

        self._matrix_cache[identity] = (weakref.ref(matrix, release), metadata, programmed)
        return programmed

    def map_quantized(self, codes, scales, *, zero_points=None, bias=None, name=None):
        """Import integer [O,K] weights as ``scales * (codes - zero_points)``.

        Scales and zero points broadcast to [O,K]. They must remain constant
        along each ADC read. No weight scale is estimated during this import.
        Set ``Config(weight_signed=False)`` for unsigned integer codes.
        """
        if name is None:
            name = f"matrix.{self._next_id}"
            self._next_id += 1
        return AnalogMatrix(
            codes, self, bias=bias, name=name, scales=scales, zero_points=zero_points
        )

    def report(self):
        return dict(
            mode=self.mode,
            device=str(self.device),
            config=asdict(self.config),
            selections=self.records,
        )

    def _key(self, matrix, x):
        from .cuda import identity

        payload = dict(
            config=asdict(self.config),
            hardware=identity(self.device),
            shape=list(x.shape),
            output=matrix.out_features,
            dtype=str(x.dtype),
            sources=self._fingerprint,
            workspace_bytes=self.workspace_bytes,
            grid=dict(
                arrays=[1, 2, 4, 8, 16, 32, 64],
                vectors=[1, 2, 4, 8, 16, 32],
                columns=[2, 4, 8],
                workers=len(self.config.weight_layout),
            ),
        )
        payload = json.loads(json.dumps(payload))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

    def _admit(self, matrix, q, schedule):
        from .cuda import ResourceError, program

        c = self.config
        n, m, r = q.shape
        width = min(schedule["arrays"], matrix.p)
        size = (8 if c.precise_currents else 4) * m * width * len(
            c.weight_layout
        ) * r * c.array_size[1] + (
            32 if c.wide else 16
        ) * n * m * c.reads_per_array * width * c.array_size[1]
        if size > self.workspace_bytes:
            raise ResourceError("workspace budget")
        if schedule["vectors"] * len(c.input_layout) * schedule["columns"] > 128:
            raise ResourceError("accumulator policy (vectors * activation bits * columns <= 128)")
        with torch.cuda.device(self.device):
            resource = program(
                c, schedule["vectors"], schedule["columns"], self.device.index
            ).resources(c, schedule["vectors"], schedule["columns"], schedule["workers"])
        if resource["local_bytes"]:
            raise ResourceError("register spill policy")
        return resource

    def _select(self, matrix, x, q, xscale, event):
        from .cuda import ResourceError, atomic_write

        started = time.perf_counter()
        key, payload = self._key(matrix, x)
        population = list(schedule_grid(self.config, q.shape[0], matrix.p))
        reference_output = None
        for baseline in [default_schedule(self.config), *population]:
            try:
                self._admit(matrix, q, baseline)
                reference_output = matrix._execute(q, xscale, baseline, event)
                break
            except (ResourceError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        if reference_output is None:
            raise RuntimeError("no feasible reference schedule in the declared grid")
        if not torch.isfinite(reference_output).all():
            raise FloatingPointError("nonfinite default output during tuning")
        path = self.cache_dir / f"{key}.json"
        record = dict(
            key=key,
            layer=matrix.name,
            shape=list(x.shape),
            cache_hit=False,
            candidates=[],
            selected=None,
            complete=False,
        )
        self.records.append(record)
        cached = self._selections.get(key)
        if cached is None and path.is_file():
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
                if (
                    content["fingerprint"] == payload
                    and content["complete"]
                    and content["selected"]
                    in list(schedule_grid(self.config, q.shape[0], matrix.p))
                ):
                    cached = content["selected"]
            except (KeyError, ValueError, TypeError):
                cached = None
        if cached is not None:
            try:
                self._admit(matrix, q, cached)
                output = matrix._execute(q, xscale, cached, event)
                if not torch.equal(reference_output.view(torch.int32), output.view(torch.int32)):
                    raise RuntimeError("cached schedule changes output")
                record.update(
                    cache_hit=True,
                    selected=cached,
                    complete=True,
                    preparation_seconds=time.perf_counter() - started,
                )
                self._selections[key] = cached
                return cached
            except (ResourceError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                cached = None
        for schedule in population:
            candidate = dict(schedule=schedule, status="rejected", timed_samples=0)
            record["candidates"].append(candidate)
            try:
                candidate["resources"] = self._admit(matrix, q, schedule)
                output = matrix._execute(q, xscale, schedule, event)
                if not torch.equal(reference_output.view(torch.int32), output.view(torch.int32)):
                    raise RuntimeError("schedule changes output")
                del output
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                output = matrix._execute(q, xscale, schedule, event)
                end.record()
                end.synchronize()
                candidate.update(
                    status="accepted", seconds=start.elapsed_time(end) / 1000, timed_samples=1
                )
                del output
            except ResourceError as error:
                candidate["reason"] = str(error)
            except torch.cuda.OutOfMemoryError:
                candidate["reason"] = "CUDA out of memory"
                torch.cuda.empty_cache()
        accepted = [item for item in record["candidates"] if item["status"] == "accepted"]
        if not accepted:
            raise RuntimeError("no feasible schedule in the declared grid")
        selected = min(accepted, key=lambda item: item["seconds"])["schedule"]
        record.update(
            selected=selected,
            complete=True,
            domain_count=len(population),
            preparation_seconds=time.perf_counter() - started,
        )
        atomic_write(
            path,
            json.dumps(
                dict(
                    fingerprint=payload,
                    complete=True,
                    selected=selected,
                    candidates=record["candidates"],
                ),
                sort_keys=True,
            ).encode(),
        )
        self._selections[key] = selected
        return selected


class AnalogMatrix(nn.Module):
    """Programmed matrix with optional digital zero-point compensation."""

    def __init__(
        self, weight, simulator, *, bias=None, name="matrix", scales=None, zero_points=None
    ):
        super().__init__()
        if weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError("weight must be a nonempty [O,K] matrix")
        if not torch.isfinite(weight).all():
            raise ValueError("weights must be finite")
        self.simulator, self.config, self.name = simulator, simulator.config, name
        self.out_features, self.in_features = weight.shape
        self.p = math.ceil(self.out_features / self.config.array_size[1])
        dtype = torch.float64 if self.config.wide else torch.float32
        zero_correction = None
        if scales is None:
            if not weight.is_floating_point():
                raise ValueError("unquantized weight must be floating")
            if zero_points is not None:
                raise ValueError("zero_points requires imported integer codes and scales")
            packed, scale = weight_tiles(weight.detach().to(simulator.device), self.config)
        else:
            if weight.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("imported weight codes must have an integer dtype")
            if self.config.bw_e is not None:
                raise ValueError("native INT weight import requires bw_e=None")
            lower = -(1 << (self.config.weight_bits - 1)) if self.config.weight_signed else 0
            upper = (1 << (self.config.weight_bits - int(self.config.weight_signed))) - 1
            imported_codes = weight.detach().to(device=simulator.device, dtype=torch.int64)
            if (imported_codes < lower).any() or (imported_codes > upper).any():
                raise ValueError("weight codes exceed configured signedness or bit width")
            code = imported_codes.T.contiguous()
            scale_full = (
                expanded_parameter(
                    scales, weight.shape, simulator.device, dtype, "scales", positive=True
                )
                .detach()
                .T.contiguous()
            )
            check_read_scales(scale_full, self.config, "weight scale")
            packed, scale = pack_weight(code, scale_full, self.config)
            if zero_points is not None:
                zeros = (
                    expanded_parameter(
                        zero_points, weight.shape, simulator.device, dtype, "zero_points"
                    )
                    .detach()
                    .T.contiguous()
                )
                check_read_scales(zeros, self.config, "weight zero point")
                r = self.config.array_size[0]
                starts = read_starts(math.ceil(self.in_features / r), self.config, code.device)
                indices = starts.clamp_max(self.in_features - 1)
                zero_correction = -(zeros[indices] * scale_full[indices])
                if not torch.isfinite(zero_correction).all():
                    raise ValueError("scaled zero points exceed the reconstruction dtype")
        self.register_buffer("packed", packed)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("zero_correction", zero_correction)
        self.register_buffer("input_scale", None)
        if bias is not None and (
            not bias.is_floating_point()
            or bias.shape != (self.out_features,)
            or not torch.isfinite(bias).all()
        ):
            raise ValueError("bias must be finite and match out_features")
        self.register_buffer(
            "bias",
            None
            if bias is None
            else bias.detach()
            .to(device=simulator.device, dtype=torch.float64 if self.config.wide else torch.float32)
            .clone(),
        )
        if self.bias is not None and not torch.isfinite(self.bias).all():
            raise ValueError("bias exceeds the reconstruction dtype")
        self._event = 0
        tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        self.seed = self.config.seed ^ tag
        self._prepared = {}

    def set_input_scale(self, scale):
        """Fix symmetric activation scales for subsequent calls; None restores dynamic scales.

        A scalar or [1,K] tensor is shared across all input vectors. The saved
        scales must be positive and constant within each analog read.
        """
        if scale is None:
            self.input_scale = None
        else:
            dtype = torch.float64 if self.config.wide else torch.float32
            expanded = expanded_parameter(
                scale, (1, self.in_features), self.packed.device, dtype, "input scale"
            ).detach()
            if (expanded <= 0).any():
                raise ValueError("input scale must be strictly positive")
            check_read_scales(expanded.T, self.config, "input scale")
            self.input_scale = expanded.clone()
        self._prepared.clear()
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        packed = state_dict.get(prefix + "packed")
        if packed is not None:
            if not torch.is_tensor(packed) or packed.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise ValueError("checkpoint packed weights must have an integer dtype")
            integer_codes = packed.to(torch.int64)
            if (integer_codes < 0).any() or (
                integer_codes > (1 << self.config.weight_bits) - 1
            ).any():
                raise ValueError("checkpoint packed weights exceed the configured bit width")
        scales = state_dict.get(prefix + "weight_scale")
        if scales is not None and (
            not torch.is_tensor(scales)
            or not scales.is_floating_point()
            or not torch.isfinite(scales).all()
            or (scales < 0).any()
        ):
            raise ValueError("checkpoint weight scales must be finite and nonnegative")
        # Optional buffers acquire their serialized shape before PyTorch copies them.
        for name in ("input_scale", "zero_correction"):
            key = prefix + name
            if key in state_dict and self._buffers[name] is None:
                self._buffers[name] = torch.empty_like(state_dict[key], device=self.packed.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if not torch.isfinite(self.weight_scale).all() or (self.weight_scale < 0).any():
            raise ValueError("checkpoint weight scales exceed the reconstruction dtype")
        if self.bias is not None and not torch.isfinite(self.bias).all():
            raise ValueError("checkpoint bias exceeds the reconstruction dtype")
        if self.input_scale is not None:
            self.set_input_scale(self.input_scale)
        if self.zero_correction is not None:
            expected = (self.weight_scale.shape[0], self.out_features)
            if tuple(self.zero_correction.shape) != expected:
                raise ValueError("checkpoint zero-point correction has an invalid shape")
            if not torch.isfinite(self.zero_correction).all():
                raise ValueError("checkpoint zero-point correction must be finite")

    def _apply(self, fn, recurse=True):
        programmed = {name: buffer for name, buffer in self._buffers.items() if buffer is not None}
        super()._apply(fn, recurse=recurse)
        for name, original in programmed.items():
            transformed = self._buffers[name]
            if transformed.dtype != original.dtype:
                self._buffers[name] = original.to(device=transformed.device)
            if transformed.device != original.device:
                self._prepared.clear()
        return self

    def get_extra_state(self):
        return dict(
            config=asdict(self.config),
            in_features=self.in_features,
            out_features=self.out_features,
            name=self.name,
            seed=self.seed,
            event=self._event,
        )

    def set_extra_state(self, state):
        for key in ("in_features", "out_features"):
            if key in state and (type(state[key]) is not int or state[key] != getattr(self, key)):
                raise ValueError("checkpoint logical matrix dimensions differ")
        loaded_config = dict(state["config"])
        for name in ("rows_per_read", "input_slice", "weight_slice", "bw_e"):
            loaded_config.setdefault(name, None)
        loaded_config.setdefault("weight_signed", True)
        if loaded_config != asdict(self.config):
            raise ValueError("checkpoint physical configuration differs")
        if type(state["seed"]) is not int or not 0 <= state["seed"] < 2**32:
            raise ValueError("invalid checkpoint sampling seed")
        self.reset_read(state["event"])
        self.name, self.seed, self._event = state["name"], state["seed"], state["event"]
        self._prepared.clear()

    def reset_read(self, event=0):
        if type(event) is not int or not 0 <= event < 2**63:
            raise ValueError("event must be a nonnegative signed 64-bit integer")
        self._event = event

    @torch.no_grad()
    def update_weight(self, matrix):
        """Reprogram a right-hand [K,O] matrix without replacing this module.

        Shape, physical settings, fixed input scales, and sampling identity are
        retained. Read events restart at zero after programming.
        """
        if not isinstance(matrix, torch.Tensor) or tuple(matrix.shape) != (
            self.in_features,
            self.out_features,
        ):
            raise ValueError("replacement matrix must have the original [K,O] shape")
        replacement = AnalogMatrix(matrix.T, self.simulator, bias=self.bias, name=self.name)
        self.packed.copy_(replacement.packed)
        self.weight_scale.copy_(replacement.weight_scale)
        self.zero_correction = None
        self._prepared.clear()
        self.reset_read()
        return self

    def _execute(self, q, xscale, schedule, event, stable=False):
        runner = reference.partial
        if q.is_cuda:
            from .cuda import partial

            runner = partial
        output = torch.empty(
            (q.shape[0], self.p * self.config.array_size[1]),
            device=q.device,
            dtype=torch.float64 if self.config.wide or stable else torch.float32,
        )
        width = min(schedule["arrays"], self.p)
        c = self.config.array_size[1]
        packed = self.packed.contiguous()
        for begin in range(0, self.p, width):
            end = min(begin + width, self.p)
            value = runner(q, packed, self.config, schedule, self.seed, event, begin, end)
            if self.config.wide or stable:
                value = restore_scales(
                    value, xscale[:, :, None, None], self.weight_scale[None, :, begin:end, :]
                )
            else:
                value = value * xscale[:, :, None, None]
                value = value * self.weight_scale[None, :, begin:end, :]
            output[:, begin * c : end * c] = value.sum(1).flatten(1)
        output = output[:, : self.out_features].contiguous()
        if self.zero_correction is not None:
            # The same quantized input drives both the analog product and its
            # digital zero-point correction, including partially filled reads.
            bits = self.config.activation_bits
            signed = q.to(torch.int64).reshape(q.shape[0], -1)
            signed = (signed ^ (1 << (bits - 1))) - (1 << (bits - 1))
            r, step = self.config.array_size[0], self.config.read_rows
            padded = torch.nn.functional.pad(
                signed.reshape(q.shape[0], -1, r),
                (0, self.config.reads_per_array * step - r),
            )
            sums = padded.reshape(q.shape[0], -1, step).sum(-1)
            dtype = output.dtype
            if self.config.wide or stable:
                for begin in range(0, self.out_features, c):
                    end = min(begin + c, self.out_features)
                    correction = restore_scales(
                        sums[:, :, None],
                        xscale[:, :, None],
                        self.zero_correction[None, :, begin:end],
                    ).sum(1)
                    output[:, begin:end] += correction
            else:
                correction = (sums.to(dtype) * xscale.to(dtype)) @ self.zero_correction.to(dtype)
                output = output + correction
        if self.bias is not None:
            output = output + self.bias
        if not self.config.wide and not stable and not torch.isfinite(output).all():
            return self._execute(q, xscale, schedule, event, stable=True)
        return output

    @torch.no_grad()
    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError("input must be floating with matching in_features")
        if x.device != self.packed.device or x.device != self.simulator.device:
            raise ValueError("input and simulator must use the same device")
        scale_dtype = torch.float64 if self.config.wide else torch.float32
        if self.weight_scale.dtype != scale_dtype:
            raise RuntimeError("keep simulator scale buffers at their mapped dtype")
        shape = x.shape
        if x.numel() == 0:
            return x.new_empty((*shape[:-1], self.out_features))
        if self._event >= 2**63:
            raise OverflowError("read-event counter exhausted")
        q, xscale = input_tiles(x.reshape(-1, self.in_features), self.config, self.input_scale)
        event = self._event
        schedule = default_schedule(self.config)
        key = tuple(shape), str(x.dtype)
        if self.simulator.mode == "tuned":
            if key not in self._prepared:
                with torch.cuda.device(x.device):
                    self._prepared[key] = self.simulator._select(self, x, q, xscale, event)
            schedule = self._prepared[key]
        output = self._execute(q, xscale, schedule, event)
        output = output.reshape(*shape[:-1], self.out_features).to(x.dtype)
        if not torch.isfinite(output).all():
            raise FloatingPointError("nonfinite analog output")
        self._event += 1
        return output
