"""Runtime-compiled CUDA kernels, loaded on first CUDA use."""

import ctypes as ct
import ctypes.util
import hashlib
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch


class ResourceError(RuntimeError):
    """A launch exceeds a compiled kernel or device resource limit."""


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed (CUDA status {status})")


@lru_cache(None)
def libraries():
    if os.name == "nt":
        driver = ct.WinDLL("nvcuda.dll")
        roots = [Path(os.environ.get("CUDA_PATH", "")), Path(torch.__file__).parent]
        candidates = [
            p
            for root in roots
            for pattern in ("bin/nvrtc64*.dll", "lib/nvrtc64*.dll")
            for p in root.glob(pattern)
        ]
    else:
        driver = ct.CDLL("libcuda.so.1")
        roots = [Path(p) / "nvidia/cuda_nvrtc/lib" for p in sys.path if p]
        roots += [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "lib64"]
        candidates = [p for root in roots for p in root.glob("libnvrtc.so*")]
    name = str(sorted(candidates)[0]) if candidates else ctypes.util.find_library("nvrtc")
    if not name:
        raise RuntimeError("CUDA runtime compilation requires the NVIDIA NVRTC library")
    compiler = ct.WinDLL(name) if os.name == "nt" else ct.CDLL(name)
    return driver, compiler


def cache_root():
    base = os.environ.get("MEMFAST_CACHE_DIR")
    if base:
        return Path(base).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "memfast"


def atomic_write(path, content):
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source(config):
    ni = config.nonidealities
    drift = (1 + ni.drift_time / ni.drift_reference_time) ** (-ni.drift_coefficient)
    values = dict(
        R=config.array_size[0],
        ROWS=config.read_rows,
        GROUPS=config.reads_per_array,
        C=config.array_size[1],
        A=len(config.input_layout),
        W=len(config.weight_layout),
        WIDE_CODES=int(max(config.activation_bits, config.weight_bits) > 8),
        WIDE_OUTPUT=int(config.wide),
        PRECISE_CURRENTS=int(config.precise_currents),
        READ_MODE=("shared", "slice_pair", "output").index(ni.read_mode),
        HIGH=config.high_conductance,
        LOW=config.low_conductance,
        VOLTAGE=config.read_voltage,
        UNIT=config.current_unit,
        INV_UNIT=1 / config.current_unit,
        WRITE=ni.write_variation,
        READ=ni.read_variation,
        VNOISE=ni.voltage_variation,
        ONOISE=ni.output_noise,
        NONLINEAR=ni.adc_nonlinearity,
        SAF_LOW=ni.stuck_at_low,
        SAF_HIGH=ni.stuck_at_high,
        DRIFT=drift,
    )
    defines = []
    for key, value in values.items():
        suffix = (
            ""
            if config.precise_currents and key in ("HIGH", "LOW", "VOLTAGE", "UNIT", "INV_UNIT")
            else "f"
        )
        literal = str(value) if type(value) is int else f"{value:.17e}{suffix}"
        defines.append(f"#define {key} {literal}")
    defines = "\n".join(defines)
    tables = [
        f"__device__ __constant__ int ADC_MAXIMUM[{len(config.weight_layout)}]"
        f" = {{{', '.join(map(str, config.adc_maxima))}}};"
    ]
    for prefix, layout in (("A", config.input_layout), ("W", config.weight_layout)):
        for index, name in enumerate(("SHIFT", "MASK", "FACTOR")):
            ctype = "long long" if name == "FACTOR" else "int"
            initializer = ", ".join(str(entry[index]) for entry in layout)
            tables.append(
                f"__device__ __constant__ {ctype} {prefix}_{name}[{len(layout)}]"
                f" = {{{initializer}}};"
            )
    return (
        defines
        + "\n"
        + "\n".join(tables)
        + "\n"
        + Path(__file__).with_name("kernels.cu").read_text(encoding="utf-8")
    )


class Program:
    def __init__(self, config, vectors, columns):
        self.driver, compiler = libraries()
        text = source(config)
        self.expression = f"consume<{vectors},{columns}>"
        capability = torch.cuda.get_device_capability()
        major, minor = ct.c_int(), ct.c_int()
        check(compiler.nvrtcVersion(ct.byref(major), ct.byref(minor)), "NVRTC version")
        options = [
            b"--std=c++17",
            b"--fmad=false",
            f"--gpu-architecture=compute_{capability[0]}{capability[1]}".encode(),
        ]
        identity = repr((text, self.expression, options, major.value, minor.value))
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = cache_root() / "kernels" / f"{key}.ptx"
        names_path = path.with_suffix(".name")
        if path.is_file() and names_path.is_file():
            ptx, name = path.read_bytes(), names_path.read_bytes()
        else:
            program = ct.c_void_p()
            compiler.nvrtcCreateProgram.argtypes = [
                ct.POINTER(ct.c_void_p),
                ct.c_char_p,
                ct.c_char_p,
                ct.c_int,
                ct.c_void_p,
                ct.c_void_p,
            ]
            check(
                compiler.nvrtcCreateProgram(
                    ct.byref(program), text.encode(), b"memfast.cu", 0, None, None
                ),
                "create NVRTC program",
            )
            try:
                compiler.nvrtcAddNameExpression.argtypes = [ct.c_void_p, ct.c_char_p]
                check(
                    compiler.nvrtcAddNameExpression(program, self.expression.encode()),
                    "register kernel",
                )
                compiler.nvrtcCompileProgram.argtypes = [
                    ct.c_void_p,
                    ct.c_int,
                    ct.POINTER(ct.c_char_p),
                ]
                argv = (ct.c_char_p * len(options))(*options)
                status = compiler.nvrtcCompileProgram(program, len(options), argv)
                length = ct.c_size_t()
                if status:
                    compiler.nvrtcGetProgramLogSize(program, ct.byref(length))
                    log = ct.create_string_buffer(length.value)
                    compiler.nvrtcGetProgramLog(program, log)
                    raise RuntimeError(log.value.decode(errors="replace"))
                compiler.nvrtcGetPTXSize(program, ct.byref(length))
                result = ct.create_string_buffer(length.value)
                check(compiler.nvrtcGetPTX(program, result), "get compiled kernel")
                ptx = result.raw
                lowered = ct.c_char_p()
                compiler.nvrtcGetLoweredName.argtypes = [
                    ct.c_void_p,
                    ct.c_char_p,
                    ct.POINTER(ct.c_char_p),
                ]
                check(
                    compiler.nvrtcGetLoweredName(
                        program, self.expression.encode(), ct.byref(lowered)
                    ),
                    "get kernel name",
                )
                name = lowered.value
                atomic_write(path, ptx)
                atomic_write(names_path, name)
            finally:
                compiler.nvrtcDestroyProgram(ct.byref(program))
        self.module = ct.c_void_p()
        self.driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        check(self.driver.cuModuleLoadData(ct.byref(self.module), ct.c_char_p(ptx)), "load kernel")
        self.driver.cuModuleGetFunction.argtypes = [
            ct.POINTER(ct.c_void_p),
            ct.c_void_p,
            ct.c_char_p,
        ]
        self.functions = {}
        for function, lowered in (("restore", b"restore"), ("consume", name)):
            handle = ct.c_void_p()
            check(
                self.driver.cuModuleGetFunction(ct.byref(handle), self.module, lowered),
                "find kernel",
            )
            self.functions[function] = handle

    def __del__(self):
        module = getattr(self, "module", None)
        driver = getattr(self, "driver", None)
        if module and driver:
            try:
                driver.cuModuleUnload(module)
            except (AttributeError, OSError):
                pass

    def attribute(self, number):
        value = ct.c_int()
        check(
            self.driver.cuDeviceGetAttribute(ct.byref(value), number, torch.cuda.current_device()),
            "device attribute",
        )
        return value.value

    def resources(self, config, vectors, columns, workers):
        threads = math.ceil(config.array_size[1] // columns * workers / 32) * 32
        shared = max(
            vectors
            * len(config.input_layout)
            * config.read_rows
            * (8 if config.precise_currents else 4),
            vectors * workers * config.array_size[1] * 8,
        )
        function = self.functions["consume"]

        def get(number):
            value = ct.c_int()
            check(
                self.driver.cuFuncGetAttribute(ct.byref(value), number, function),
                "kernel attribute",
            )
            return value.value

        maximum, static, local = get(0), get(1), get(3)
        limit = max(self.attribute(8), self.attribute(97))
        if threads > maximum or shared + static > limit:
            raise ResourceError("thread or shared-memory limit")
        if shared + static > self.attribute(8):
            check(self.driver.cuFuncSetAttribute(function, 8, shared), "shared-memory opt-in")
        active = ct.c_int()
        check(
            self.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ct.byref(active), function, threads, ct.c_size_t(shared)
            ),
            "occupancy query",
        )
        if active.value < 1:
            raise ResourceError("zero resident blocks")
        return dict(threads=threads, shared=shared, local_bytes=local, registers=get(4))

    def launch(self, name, blocks, threads, shared, values):
        arguments = [
            ct.c_void_p(value.data_ptr()) if torch.is_tensor(value) else value for value in values
        ]
        pointers = (ct.c_void_p * len(arguments))(
            *[ct.cast(ct.pointer(value), ct.c_void_p).value for value in arguments]
        )
        self.driver.cuLaunchKernel.argtypes = [
            ct.c_void_p,
            ct.c_uint,
            ct.c_uint,
            ct.c_uint,
            ct.c_uint,
            ct.c_uint,
            ct.c_uint,
            ct.c_uint,
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
            ct.c_void_p,
        ]
        check(
            self.driver.cuLaunchKernel(
                self.functions[name],
                blocks,
                1,
                1,
                threads,
                1,
                1,
                shared,
                ct.c_void_p(torch.cuda.current_stream().cuda_stream),
                pointers,
                None,
            ),
            "launch kernel",
        )


@lru_cache(maxsize=256)
def program(config, vectors, columns, device):
    with torch.cuda.device(device):
        torch.cuda.init()
        # Establish the PyTorch primary context before loading a driver module.
        torch.empty(0, device=device)
        return Program(config, vectors, columns)


def partial(q, packed, config, schedule, seed, event, begin, end):
    n, m, r = q.shape
    p, c, width = packed.shape[1], config.array_size[1], end - begin
    vectors, columns, workers = schedule["vectors"], schedule["columns"], schedule["workers"]
    with torch.cuda.device(q.device):
        kernel = program(config, vectors, columns, q.device.index)
        resource = kernel.resources(config, vectors, columns, workers)
        g = torch.empty(
            (m, width, len(config.weight_layout), r, c),
            device=q.device,
            dtype=torch.float64 if config.precise_currents else torch.float32,
        )
        count = g.numel()
        kernel.launch(
            "restore",
            min(math.ceil(count / 256), 65535),
            256,
            0,
            [
                packed,
                g,
                ct.c_longlong(count),
                ct.c_int(p),
                ct.c_int(begin),
                ct.c_int(width),
                ct.c_uint(seed),
                ct.c_ulonglong(event),
            ],
        )
        result = torch.empty(
            (n, m * config.reads_per_array, width, c),
            device=q.device,
            dtype=torch.float64 if config.wide else torch.float32,
        )
        blocks = math.ceil(n / vectors) * m * config.reads_per_array * width
        if blocks > kernel.attribute(5):
            raise ResourceError("grid limit")
        kernel.launch(
            "consume",
            blocks,
            resource["threads"],
            resource["shared"],
            [
                q,
                g,
                result,
                *map(ct.c_int, (n, m, p, begin, width, workers)),
                ct.c_uint(seed),
                ct.c_ulonglong(event),
            ],
        )
        return result


def identity(device):
    with torch.cuda.device(device):
        properties = torch.cuda.get_device_properties(device)
        version = ct.c_int()
        check(libraries()[0].cuDriverGetVersion(ct.byref(version)), "driver version")
        return dict(
            name=properties.name,
            capability=torch.cuda.get_device_capability(device),
            multiprocessors=properties.multi_processor_count,
            memory=properties.total_memory,
            driver=version.value,
            torch=torch.__version__,
            cuda=torch.version.cuda,
        )
