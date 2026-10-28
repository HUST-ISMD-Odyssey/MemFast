# User Guide

## Matrix And Network Interfaces

Use `memfast.mm(A, B)` as you would use `torch.mm(A, B)`: both operands
are matrices, with shapes `[N,K]` and `[K,O]`. `memfast.matmul(A, B)` also
accepts vectors and broadcast batch dimensions, following `torch.matmul`.
No transpose, matrix name or separate programming call is required.

```python
import torch
import memfast
from memfast import Config, Simulator

A = torch.randn(16, 256, device="cuda:0", requires_grad=True)
B = torch.randn(256, 128, device="cuda:0", requires_grad=True)
C = memfast.mm(A, B)
C.sum().backward()

sim = Simulator(Config(), device="cuda:0")
C = sim.mm(A, B)
```

An explicit simulator shares configuration and automatically reuses programmed
weights while its quantized codes and scales are unchanged. Device-local checks
also detect changes through NumPy aliases, `.data`, or optimizer updates.
Forward uses the physical array model; backward uses the FP32 quantized-operand
surrogate described below. Device effects are not differentiated.
For broadcast products, each distinct right-hand matrix has its own program;
vectors that share it are quantized together in flattened row order.

Linear and Conv2d use the same matrix engine with their own PyTorch weight
layouts. Their public constructors follow PyTorch's argument order:

```python
from memfast import nn

linear = nn.Linear(256, 128, bias=True, device="cuda:0")
conv = nn.Conv2d(3, 32, 3, stride=1, padding=1, device="cuda:0")
output = linear(A)
output.sum().backward()
```

Pass `config=config` or `simulator=sim` as additional keyword arguments.
The layers expose trainable `.weight` and `.bias`, and work with ordinary
PyTorch optimizers. `layer.update_weight(new_weight)` copies PyTorch-layout
weights without replacing the optimizer parameters; `layer.update_weight()`
refreshes programming from the current parameters. Ordinary forward calls
already refresh programming after optimizer updates.

For an existing network, load its digital checkpoint and convert:

```python
from memfast import convert_model

analog = convert_model(model.eval(), sim, exclude=("classifier",))
```

`convert_model` copies the model, converts eligible `Linear`, `Conv2d`, RNN and GRU
modules, and places the result on the simulator device. `inplace=True` edits
the supplied model. Exclusion paths refer to named modules and include their
descendants. Unknown paths raise an error.

All supported layers are converted unless `include` restricts the scope.
`include=None` means all; `include=()` means none. Inclusion selects named
modules and their descendants, while `exclude` takes precedence. An excluded
alias of a shared module keeps that shared module digital.
A shared module is converted only when all its aliases are included; inspect
aliases with `model.named_modules(remove_duplicate=False)` when selecting them.

```python
from torch import nn

eligible = [
    name for name, layer in model.named_modules() if isinstance(layer, (nn.Linear, nn.Conv2d))
]
analog = convert_model(model, sim, include=eligible[:4], exclude=(eligible[-1],))
```

Construct the digital architecture, load its original checkpoint, then convert.
The [MLP tutorial](../examples/02_mlp_mnist.py) shows each step, with no
pretrained model required.

Convolution uses PyTorch unfolding and one matrix per convolution group;
stride, dilation, numeric padding, and groups are preserved. Unfolding can be
substantial for large feature maps. Other layer operations remain digital.
Inputs and outputs retain their floating dtype. Cell currents normally use FP32.
FP64 currents are selected for slice digits wider than eight bits, closely
spaced conductance levels, subnormal physical constants, or current ranges
that exceed FP32. This preserves ADC counts in those configurations.
Scale restoration uses FP64 for BFP or precisions above eight bits, and FP32
for the default eight-bit INT configuration. Model dtype conversions preserve the
exact programmed codes, scales, and bias storage. Inputs and outputs still
follow the network dtype. Select the simulator device before mapping; moving
a mapped network to a different device requires remapping with that device.

### Layers, Models, And Training

`memfast.nn` contains PyTorch-style trainable `Linear` and `Conv2d`,
compact inference adapters (`AnalogLinear`, `AnalogConv2d`),
recurrent RNN/GRU layers, trainable layers
(`HATLinear`, `HATConv2d`), conversion, and read-event utilities.
`memfast.models` supplies custom MLP, LeNet, CIFAR and Speech Commands
architectures. Examples load torchvision, timm, Hugging Face and Ultralytics
models directly, then convert their supported layers. The numerical engine
remains independent of these model libraries.

```python
from memfast.nn import Linear, convert_model

layer = Linear(128, 32, simulator=sim)
trainable = convert_model(model, sim, training=True, gradient_mode="quantized")
optimizer = torch.optim.Adam(trainable.parameters(), lr=1e-3)
```

HAT keeps floating master parameters and reprograms the analog weights on each
forward invocation. The forward values follow the same physical model as
inference. Default surrogate gradients use dequantized grouped inputs and
weights, with identity derivatives through their quantizers. Set
`gradient_mode="digital"` to use unquantized layer derivatives instead.
The surrogate computation uses FP32, including when the physical forward uses
wide INT or BFP reconstruction. Tutorials use FP32 master parameters. The
layer API preserves source parameter dtype, so leaf gradients accumulate in
that dtype when a caller supplies a different one.
Neither option differentiates the noise, faults, or ADC response.
Convolution gradients use unfolded patches directly. Optimizer parameters and
registered buffers keep stable identities for distributed training.

The low-level compatibility interface `Simulator.map(weight)` accepts layer
weights in `[O,K]` layout. It is used by the layer adapters; application matrix
products should use `mm` or `matmul`.

Converted recurrent networks preserve stacked and bidirectional execution,
hidden states, the RNN tanh/ReLU nonlinearity and PyTorch GRU gate equations.
See the numbered examples for architecture construction and checkpoint loading.

## Quantization And Physical Parallelism

Physical matrices use `[K, O]` coordinates: reduction rows and output columns.
PyTorch weights `[O, K]` are transposed internally.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `activation_bits` | Signed activation precision | `8` |
| `weight_bits` | Weight code precision | `8` |
| `weight_signed` | Two's-complement (`True`) or unsigned (`False`) weight reconstruction | `True` |
| `array_size=(R,C)` | Physical array rows and output columns | `(64,64)` |
| `rows_per_read=P` | Maximum physical rows summed before each ADC | `None`, meaning `R` |
| `weight_quant_gran=(QK,QO)` | Rectangular weight region sharing one scale | `(None,1)` |
| `input_quant_gran=(QN,QK)` | Input region sharing one scale | `(1,None)` |
| `adc_bits` | Uniform unsigned ADC bits, or one entry per weight slice | `6` |
| `input_slice`, `weight_slice` | Slice widths, most significant first | `None`, meaning one-bit slices |
| `bw_e` | Shared exponent width for BFP; `None` selects INT | `None` |

`None` spans the entire corresponding matrix dimension. Defaults therefore
give per-output-channel weight quantization and per-vector activation
quantization.

INT quantization uses `scale = max(abs(group)) / (2**(bits-1)-1)` and nearest-even
rounding. Codes range from `-(2**(bits-1)-1)` to `2**(bits-1)-1`. All-zero groups
produce zero codes. Negative codes are stored in two's complement; the most
significant bit has negative significance.

### Imported Affine Weights And Static Inputs

`Simulator.map_quantized` accepts existing integer codes without estimating
new weight scales. Codes use PyTorch weight layout `[O,K]`; scales and zero
points must broadcast to that shape. Expand a checkpoint's per-group values
using its actual group index before mapping.

```python
config = Config(
    activation_bits=8,
    weight_bits=4,
    weight_signed=False,
    array_size=(128, 64),
    rows_per_read=32,
    input_quant_gran=(1, 32),
    weight_quant_gran=(32, 1),
)
sim = Simulator(config, device="cuda:0")
codes = torch.randint(0, 16, (64, 128))  # Replace with decoded checkpoint codes.
scales = torch.full((64, 128), 0.025)
zero_points = torch.full((64, 128), 8)
array = sim.map_quantized(codes, scales, zero_points=zero_points)
y = array(torch.randn(1, 128, device=sim.device))
```

An affine weight is `w = scale * (q - zero_point)`. With unsigned weights,
all weight slice significances are positive, including the most significant
slice. For example, `weight_slice=(2,2)` represents a four-bit unsigned code
using factors 4 and 1; signed weights require a separate one-bit sign slice.
Activation codes remain signed and have no zero point.

For each read, the analog path evaluates the integer product with `q`.
The digital path subtracts
`weight_scale * zero_point * sum(quantized_input * input_scale)`.
It uses the same activation codes as the analog path. This correction does
not cancel array noise or ADC clipping. Both scale and zero point must stay
constant along each individual ADC read; incompatible imported groups are
rejected rather than silently requantized. Set `rows_per_read` to an aligned
group subdivision when necessary. Imported weights use INT (`bw_e=None`).
Unsigned mapping of floating nonnegative matrices is also supported; use the
integer import API for affine weights that represent negative values.

By default, input scales are estimated per invocation. To use calibrated
scales instead:

```python
array.set_input_scale(0.015)  # One fixed scale for all features and vectors.
# Or supply [1,K] scales that are constant inside every analog read.
y = array(inputs)
torch.save(array.state_dict(), "calibrated_matrix.pt")
array.set_input_scale(None)  # Restore dynamic scale estimation.
```

Fixed-scale inference still rounds and clips each activation into integer
codes; it skips online scale estimation. Scales are buffers saved with the
programmed integer weights. Calibrate on a representative training subset,
then evaluate separately. A complete LeNet example covers calibration,
checkpoint export and inference from the saved artifact.

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

Input groups use the flattened matrix `[N,K]`:

| Input quantization group | Scale sharing |
| --- | --- |
| `(None,None)` | Whole input matrix for this invocation |
| `(1,None)` | Each vector independently |
| `(None,1)` | Each input feature across vectors |
| `(4,64)` | Groups of four vectors and 64 reduction elements |

Leading input dimensions are flattened before grouping. For convolution, these
are unfolded patches; an unfolded feature is a channel/kernel position.
Scales are computed from the current invocation.

`P` may be any positive integer up to `R`. Reads follow contiguous physical row
ranges, with a shorter final range when `P` does not divide `R`. Each read must
have one input scale per vector and one weight scale per output column.
An explicit reduction group therefore aligns with all read boundaries: it
must be a multiple of `R`, or a multiple of `P` when `P` divides `R`.
The input vector group and weight output-column group are independent of these
reduction constraints. Merely being larger than `P` is insufficient.

For `R=128,P=32`, weight groups `(32,1)` and input groups `(1,32)` are valid.
Each physical array produces four ADC partials:

```text
y[n,o] = sum_g input_scale[n,g] * weight_scale[g,o] * reconstructed_partial[n,g,o]
```

Scale restoration precedes summation across reads. Larger aligned quantization
groups share a scale across multiple reads. Per-input-feature quantization is
supported with `P=1`. Incompatible groups raise an error instead of being
silently enlarged.

Tail matrix rows and columns are zero-padded. Physical padding inherits the
last logical scale group and remains part of executed reads, including output
noise. Returned outputs exclude padded columns. Row selection is contiguous;
zero input elements remain in their physical positions.

Array columns must be divisible by four. Activation and weight precisions can
differ; the configuration accepts 2-32 bits subject to a checked signed 64-bit
reconstruction bound. Ten-, sixteen-, and twenty-bit paths are included in the
tests. Wider configurations use packed 64-bit codes. The bound is
`sum(abs(input_significances)) * sum_w(adc_maximum[w] * abs(weight_significance[w]))`;
an overflowing combination is rejected before execution.

### Dynamic Slicing And BFP

Slice widths run from most significant to least significant. The first width
is one for the two's-complement sign cell, as in the original interface.
Each remaining slice holds 1-16 bits; the widths sum to the corresponding
activation or weight precision. `(1,1,2,2,2)` represents eight bits with five
physical slices. Input and weight layouts can differ. ADC conversion and noise
events follow the physical slice pairs.

```python
config = Config(
    activation_bits=16,
    weight_bits=16,
    input_slice=(1, 3, 4, 4, 4),
    weight_slice=(1, 3, 4, 4, 4),
    bw_e=8,
    array_size=(128, 128),
    rows_per_read=32,
    input_quant_gran=(4, 64),
    weight_quant_gran=(32, 1),
    adc_bits=16,
)
```

ADC resolutions may also differ between weight slices. For example:

```python
config = Config(
    weight_slice=(1, 1, 2, 2, 2),
    adc_bits=(6, 7, 8, 9, 10),
)
```

Both sequences run from most significant to least significant. The sign slice
uses six ADC bits, the next slice seven, and the remaining slices eight, nine,
and ten. Every activation slice paired with a given weight slice uses that
weight slice's ADC resolution. A scalar applies to all weight slices.
Sequence length must equal the number of physical weight slices, including
the sign slice. Each resolution is an integer in `1..16`. Per-slice settings
also control S-shaped ADC saturation and are part of cache and checkpoint
identity.

`bw_e=None` selects INT. An exponent width selects BFP: for block maximum `m`,
choose `e=floor(log2(m))`, constrained to
`[1-(2**(bw_e-1)-1), 2**(bw_e-1)-1]`. The block step is `2**(e+2-bits)`;
mantissas use nearest-even rounding and range
`[-2**(bits-1),2**(bits-1)-1]`. All-zero blocks produce zero codes.
`bw_e` is separate from the mantissa/sign bits represented by the slice widths.
This shared-exponent format saturates at the configured exponent limits.

## Array Computation

Binary slices map to `low_conductance` or `high_conductance`.
A weight slice with maximum digit `Mw` maps digit `d` to
`low_conductance + (high_conductance-low_conductance)*d/Mw`.
An input slice with maximum digit `Ma` applies `read_voltage*d/Ma`.
The nominal defaults are `1e-7 S`, `1e-5 S`, and `0.2 V`.

For every input slice, weight slice, physical subread, vector, and output column:

```text
I = sum_r V[r] * (G[r] - low_conductance)
z = I * Ma * Mw / ((high_conductance - low_conductance) * read_voltage)
code = clamp(round(response(z)), 0, 2**adc_bits[weight_slice] - 1)
```

The ADC converts each subread current before accumulation across reads.
Its count step is one ideal input-digit/weight-digit product. Multibit slices
can require a larger ADC range; `adc_bits` remains explicit.
Integer ADC codes are combined with signed bit significances; activation and
weight scales then restore the numerical units. Wide precision uses exponent
decomposition when restoring scales, avoiding intermediate overflow for finite
products. CUDA uses ascending-row fused multiply-add at the selected current
precision and integer signed reconstruction. CPU uses the same model
with PyTorch matrix operations; floating-point reduction differences can affect
currents near an ADC boundary.

## Nonidealities

Set any combination through `Config(nonidealities=Nonidealities(...))`.
Only `write_variation=0.02` is enabled by default. All other effect magnitudes
default to zero. `Nonidealities(write_variation=0)` selects ideal devices.

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

### S-Shaped ADC Nonlinearity

`adc_nonlinearity = alpha` applies the odd saturating transfer
`response(z) = z / (1 + abs(alpha * z / adc_maximum))`
before nearest-even ADC rounding and unsigned clipping. `alpha=0` gives a
linear transfer. The parameter controls a behavioral response curve and
requires calibration for a specific ADC. With per-weight-slice ADC resolutions,
`adc_maximum` is the maximum count of the current weight slice.

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
| Restored output-array tiles | `1, 2, 4, 8, 16, 32, 64` |
| Vectors per block tile | `1, 2, 4, 8, 16, 32` |
| Columns per thread | `2, 4, 8` where they divide array columns |
| Weight-slice workers | `1 .. number_of_weight_slices` |

Activation and weight slice counts remain fixed by the configured precisions.
Execution-equivalent restoration widths and vector tiles are deduplicated.
Candidates exceeding the workspace budget, actual device resource limits,
or the register policy are rejected. The register policy permits
`vectors * number_of_input_slices * columns <= 128` and excludes compiled spills.
The complete declared grid is visited, including rejected candidates.

An admitted reference schedule is selected before traversal. Default execution
is used when it fits the resource and workspace limits; otherwise the first
feasible schedule supplies the reference. Compilation and an untimed comparison precede one CUDA-event timing
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
