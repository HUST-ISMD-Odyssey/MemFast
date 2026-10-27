# User Guide

## Matrix And Network Interfaces

`Simulator.map(weight, bias=None, name=None)` programs a matrix with PyTorch
weight layout `[output_features, input_features]`. Its returned module accepts
inputs `[..., input_features]` and returns `[..., output_features]`.
The last dimension is the reduction dimension. Leading dimensions are flattened
into read vectors for one invocation.

```python
import torch
from memfast import Config, Simulator, convert_model

sim = Simulator(Config(), device="cuda:0")
array = sim.map(torch.randn(128, 256), name="projection")
y = array(torch.randn(2, 16, 256, device=sim.device))

analog = convert_model(model.eval(), sim, exclude=("classifier",))
```

`convert_model` copies the model, converts eligible `Linear` and `Conv2d`
modules, and places the result on the simulator device. `inplace=True` edits
the supplied model. Exclusion paths refer to named modules and include their
descendants. Unknown paths raise an error.

Convolution uses PyTorch unfolding and one matrix per convolution group;
stride, dilation, numeric padding, and groups are preserved. Unfolding can be
substantial for large feature maps. Other layer operations remain digital.
Inputs and outputs may be FP32, FP16, or BF16; analog arithmetic and scale
buffers use FP32. Convert model dtypes before mapping.

The modules implement inference. Program new weights by mapping or converting
the updated digital weights again.

## Quantization And Physical Parallelism

Physical matrices use `[K, O]` coordinates: reduction rows and output columns.
PyTorch weights `[O, K]` are transposed internally.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `activation_bits` | Signed activation precision | `8` |
| `weight_bits` | Signed weight precision | `8` |
| `array_size=(R,C)` | Rows reduced per ADC operation and parallel output columns | `(64,64)` |
| `weight_quant_gran=(QK,QO)` | Rectangular weight region sharing one scale | `(None,1)` |
| `input_quant_gran=(1,QK)` | Input region sharing one scale in each vector | `(1,None)` |
| `adc_bits` | Unsigned ADC resolution per slice pair and array | `6` |

`None` spans the entire corresponding matrix dimension. Defaults therefore
give per-output-channel weight quantization and per-vector activation
quantization.

Quantization uses `scale = max(abs(group)) / (2**(bits-1)-1)` and nearest-even
rounding. Codes range from `-(2**(bits-1)-1)` to `2**(bits-1)-1`. All-zero groups
produce zero codes. Negative codes are stored in two's complement; the most
significant bit has negative significance.

Physical array dimensions and quantization groups describe separate operations.
Examples for `array_size=(64,64)`:

| Weight quantization group | Scale sharing |
| --- | --- |
| `(64,1)` | One scale per 64 reduction rows in each output column |
| `(128,1)` | One scale across two row partitions in each output column |
| `(64,64)` | One scale for a 64-by-64 region |
| `(64,16)` | Four independent column scale groups within one array |
| `(None,1)` | One scale per output channel across all reduction rows |
| `(None,None)` | One scale for the entire matrix |

An explicit reduction group must be a positive multiple of `R` so that a
single array partial sum has one dequantization scale. The column group is
independent of `C`; it can be smaller or larger. Input reduction groups follow
the same row-partition alignment rule. Tail rows and columns are zero-padded
and removed from the returned result. Padding remains part of the physical
array, including its device nonidealities.

Array columns must be divisible by four. Activation and weight precisions can
differ, and each can be 2-8 bits. This release uses binary cells and one-bit
activation slices.

## Array Computation

Each weight bit maps to `low_conductance` or `high_conductance`.
An active input bit applies `read_voltage`; an inactive bit applies zero.
The nominal defaults are `1e-7 S`, `1e-5 S`, and `0.2 V`.

For every input slice, weight slice, row partition, vector, and output column:

```text
I = sum_r V[r] * (G[r] - low_conductance)
z = I / ((high_conductance - low_conductance) * read_voltage)
code = clamp(round(response(z)), 0, 2**adc_bits - 1)
```

The ADC converts each array current before accumulation across row partitions.
Integer ADC codes are combined with signed bit significances; activation and
weight scales then restore the numerical units. CUDA uses ascending-row FP32
fused multiply-add and integer signed reconstruction. CPU uses the same model
with PyTorch matrix operations; floating-point reduction differences can affect
currents near an ADC boundary.

## Nonidealities

Set any combination through `Config(nonidealities=Nonidealities(...))`.
All effects default to zero.

```python
from memfast import Config, Nonidealities, Simulator

config = Config(
    nonidealities=Nonidealities(
        write_variation=0.02,
        read_variation=0.02,
        read_mode="slice_pair",
        adc_nonlinearity=0.02,
        stuck_at_low=0.001,
        drift_coefficient=0.05,
        drift_time=100,
    )
)
sim = Simulator(config, device="cuda:0")
```

### Programming Variation

`G_programmed = G_nominal * exp(write_variation * Z)`, with independent
standard-normal `Z` per physical cell and weight slice. Samples remain fixed
for the mapped matrix. The parameter is the standard deviation of log
conductance. This uncentered lognormal model has mean multiplier
`exp(sigma**2 / 2)`.

### Read Variation

Read events advance once per mapped-matrix invocation. Samples are refreshed
between invocations. A convolution invokes each group matrix once.

| `read_mode` | Independent coordinates | Shared coordinates |
| --- | --- | --- |
| `shared` | Layer, read event, array, cell row/column, weight slice | All vectors in the invocation and all activation slices |
| `slice_pair` | The same coordinates plus activation slice | All vectors in the invocation |
| `output` | Layer, read event, vector, array, output column, activation slice, weight slice | None across current outputs |

The two conductance modes apply
`G_read = G_programmed * exp(read_variation * Z)` before low-conductance
subtraction. With four activation bits and four weight bits, `slice_pair`
creates sixteen independent cell-perturbation fields per read event. It is
still batch-shared: equal vectors in the same call see the same cell field.

`output` approximates independent cell read variation with a Gaussian current
distribution. For each cell let `b_r = V_r * G_programmed,r`. Add the following
mean shift and variance to the otherwise clean current:

```text
mean_shift = sum_r b_r * (exp(sigma**2 / 2) - 1)
variance   = sum_r b_r**2 * exp(sigma**2) * (exp(sigma**2) - 1)
I_read     = I_clean + mean_shift + sqrt(variance) * Z
```

This matches the first two moments of independent lognormal cell perturbations.
It approximates their distribution and removes the cross-vector correlation
of the batch-shared modes. A common sigma does not make these three models
interchangeable.

### Voltage Variation

`V_actual = V_nominal * (1 + voltage_variation * Z)`. Samples are independent
per vector, row partition, input slice, and row, and shared across output
columns and weight slices. An inactive input bit remains at zero voltage.

### Current Output Noise

`output_noise` adds independent Gaussian noise directly to each current before
ADC conversion:

```text
I_noisy = I + output_noise * R * current_unit * Z
current_unit = (high_conductance - low_conductance) * read_voltage
```

It is sampled separately for every vector, array, output column, and slice
pair. Its scale is relative to a fully active nominal binary row partition.
This parameter models output-referred noise directly; `read_mode="output"`
instead derives the noise scale from the programmed conductances and inputs.
The two terms can be combined.

### S-Shaped ADC Nonlinearity

`adc_nonlinearity = alpha` applies the odd saturating transfer
`response(z) = z / (1 + abs(alpha * z / adc_maximum))`
before nearest-even ADC rounding and unsigned clipping. `alpha=0` gives a
linear transfer. The parameter controls a behavioral response curve and
requires calibration for a specific ADC.

### Stuck-At Faults

`stuck_at_low` and `stuck_at_high` are mutually exclusive per-cell probabilities.
A single fixed uniform sample chooses a low fault, a high fault, or a
functioning cell. Their sum must be at most one. Faults are stable across
reads, batches, and execution schedules. Faulty cells are pinned to the
nominal low/high conductance after programming variation and drift;
read variation may subsequently perturb a read from that state.

### Conductance Drift

For functioning cells:

```text
G(t) = G_programmed * (1 + drift_time / drift_reference_time)**(-drift_coefficient)
```

Times are in seconds; defaults are zero elapsed time and a one-second reference.
This is a power-law retention model. It does not resample programming noise,
apply automatic gain compensation, or fit a particular device technology.
Create a simulator with the desired elapsed time to evaluate another
retention condition with the same seed and layer names.

## Default Execution And Tuning

The default schedule restores two output-array tiles at a time, computes two
vectors and four columns per thread, and distributes weight slices across up
to two workers. These scheduling choices do not change physical geometry.

`mode="tuned"` traverses the following finite domain per effective input shape:

| Execution axis | Values |
| --- | --- |
| Restored output-array tiles | `2, 4, 8, 16, 32, 64` |
| Vectors per block tile | `1, 2, 4, 8, 16, 32` |
| Columns per thread | `2, 4, 8` where they divide array columns |
| Weight-slice workers | `1 .. weight_bits` |

Activation and weight slice counts remain fixed by the configured precisions.
Execution-equivalent restoration widths and vector tiles are deduplicated.
Candidates exceeding the workspace budget, actual device resource limits,
or the register policy are rejected. The register policy permits
`vectors * activation_bits * columns <= 128` and excludes compiled spills.
The complete declared grid is visited, including rejected candidates.

Compilation and an untimed output comparison precede one CUDA-event timing
sample for each admitted candidate. Selection is strict minimum measured
latency. The same input and random event are used throughout. A compilation or
output error stops tuning. The fastest measurement is specific to this run,
device, shape, and domain.

`workspace_mb=1024` bounds the estimated restored-conductance and partial-output
workspace. Full inputs, packed weights, model state, unfolding, compiler state,
and allocator reservations are additional memory consumers.

Selections persist under `~/.cache/memfast/tuning` by default. Set
`MEMFAST_CACHE_DIR` to relocate all caches, or pass `cache_dir` to relocate only
schedule selections. The kernel compilation cache is separate from selection
records. A valid selection receives a default-versus-selected output check on
each newly mapped matrix; there is no candidate timing or grid traversal on a
cache hit. An inadmissible cached schedule triggers a fresh traversal.

Inspect `simulator.report()` for domain coverage, candidate statuses, measured
latencies, the selected schedule, and cache hits. Tuning happens on the first
call with a new input shape. Use representative inference inputs to prepare a
model before measuring repeated inference.

## Reproducibility And Checkpoints

`Config.seed`, stable mapped-layer names, physical coordinates, and the read
event identify random samples. Scheduling and cache hits do not consume
random events. Matrix names should be unique within a simulator if layers
should have independent programming and read fields.

```python
from memfast import reset_read_events

reset_read_events(analog_model, event=0)
output = analog_model(inputs)
```

`array.reset_read(event=0)` replays a single matrix read. Resetting changes the
read-event counter; it leaves programmed weights and faults intact.
`state_dict()` stores packed codes, scales, biases, configuration, sampling
identity, and the next read-event counter. Recreate the same converted
architecture and configuration before loading. Execution selections are
revalidated separately.
