"""Mapping, complete schedule traversal, and persistent selection cache."""

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import reference
from .config import Config
from .quantization import input_tiles, weight_tiles


def default_schedule(config):
    return dict(arrays=2, vectors=2, columns=4, workers=min(2, config.weight_bits))


def schedule_grid(config, n, partitions):
    seen = set()
    for t, v, columns, workers in itertools.product(
        (2, 4, 8, 16, 32, 64),
        (1, 2, 4, 8, 16, 32),
        (2, 4, 8),
        range(1, config.weight_bits + 1),
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
    for path in sorted([*root.glob("*.py"), *root.glob("*.cu")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
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

    def map(self, weight, *, bias=None, name=None):
        """Program a PyTorch [out_features, in_features] weight matrix."""
        if name is None:
            name = f"matrix.{self._next_id}"
            self._next_id += 1
        return AnalogMatrix(weight, self, bias=bias, name=name)

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
                arrays=[2, 4, 8, 16, 32, 64],
                vectors=[1, 2, 4, 8, 16, 32],
                columns=[2, 4, 8],
                workers=self.config.weight_bits,
            ),
        )
        payload = json.loads(json.dumps(payload))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

    def _admit(self, matrix, q, schedule):
        from .cuda import ResourceError, program

        c = self.config
        n, m, r = q.shape
        width = min(schedule["arrays"], matrix.p)
        size = (
            4 * m * width * c.weight_bits * r * c.array_size[1]
            + 16 * n * m * width * c.array_size[1]
        )
        if size > self.workspace_bytes:
            raise ResourceError("workspace budget")
        if schedule["vectors"] * c.activation_bits * schedule["columns"] > 128:
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

        key, payload = self._key(matrix, x)
        baseline = default_schedule(self.config)
        reference_output = matrix._execute(q, xscale, baseline, event)
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
        started = time.perf_counter()
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
            except ResourceError:
                cached = None
        population = list(schedule_grid(self.config, q.shape[0], matrix.p))
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
    """Programmed signed matrix. One forward call is one read event."""

    def __init__(self, weight, simulator, *, bias=None, name="matrix"):
        super().__init__()
        if weight.ndim != 2 or min(weight.shape) < 1 or not weight.is_floating_point():
            raise ValueError("weight must be a nonempty floating [O,K] matrix")
        if not torch.isfinite(weight).all():
            raise ValueError("weights must be finite")
        self.simulator, self.config, self.name = simulator, simulator.config, name
        self.out_features, self.in_features = weight.shape
        self.p = math.ceil(self.out_features / self.config.array_size[1])
        packed, scale = weight_tiles(weight.detach().float().to(simulator.device), self.config)
        self.register_buffer("packed", packed)
        self.register_buffer("weight_scale", scale)
        if bias is not None and (
            bias.shape != (self.out_features,) or not torch.isfinite(bias).all()
        ):
            raise ValueError("bias must be finite and match out_features")
        self.register_buffer(
            "bias", None if bias is None else bias.detach().float().to(simulator.device).clone()
        )
        self._event = 0
        tag = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        self.seed = self.config.seed ^ tag
        self._prepared = {}

    def get_extra_state(self):
        return dict(config=asdict(self.config), name=self.name, seed=self.seed, event=self._event)

    def set_extra_state(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("checkpoint physical configuration differs")
        self.name, self.seed, self._event = state["name"], state["seed"], state["event"]
        self._prepared.clear()

    def reset_read(self, event=0):
        if type(event) is not int or not 0 <= event < 2**63:
            raise ValueError("event must be a nonnegative signed 64-bit integer")
        self._event = event

    def _execute(self, q, xscale, schedule, event):
        runner = reference.partial
        if q.is_cuda:
            from .cuda import partial

            runner = partial
        output = torch.empty(
            (q.shape[0], self.p * self.config.array_size[1]), device=q.device, dtype=torch.float32
        )
        width = min(schedule["arrays"], self.p)
        c = self.config.array_size[1]
        for begin in range(0, self.p, width):
            end = min(begin + width, self.p)
            value = runner(q, self.packed, self.config, schedule, self.seed, event, begin, end)
            value = value * xscale[:, :, None, None]
            value = value * self.weight_scale[None, :, begin:end, :]
            output[:, begin * c : end * c] = value.sum(1).flatten(1)
        output = output[:, : self.out_features].contiguous()
        if self.bias is not None:
            output = output + self.bias
        return output

    @torch.no_grad()
    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError("input must be floating with matching in_features")
        if x.device != self.packed.device or x.device != self.simulator.device:
            raise ValueError("input and simulator must use the same device")
        if self.weight_scale.dtype != torch.float32:
            raise RuntimeError("keep simulator buffers in FP32; convert the model before mapping")
        shape = x.shape
        if x.numel() == 0:
            return x.new_empty((*shape[:-1], self.out_features))
        q, xscale = input_tiles(x.reshape(-1, self.in_features).float(), self.config)
        event = self._event
        schedule = default_schedule(self.config)
        key = tuple(shape), str(x.dtype)
        if self.simulator.mode == "tuned":
            if key not in self._prepared:
                with torch.cuda.device(x.device):
                    self._prepared[key] = self.simulator._select(self, x, q, xscale, event)
            schedule = self._prepared[key]
        output = self._execute(q, xscale, schedule, event)
        self._event += 1
        return output.reshape(*shape[:-1], self.out_features).to(x.dtype)
