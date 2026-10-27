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
Inputs and outputs retain their floating dtype. Cell currents normally use FP32.
FP64 currents are selected for slice digits wider than eight bits, closely
spaced conductance levels, subnormal physical constants, or current ranges
that exceed FP32. This preserves ADC counts in those configurations.
Scale restoration uses FP64 for BFP or precisions above eight bits, and FP32
for the default eight-bit INT configuration. Model dtype conversions preserve the
exact programmed codes, scales, and bias storage. Inputs and outputs still
follow the network dtype. Select the simulator device before mapping; moving
a mapped network to a different device requires remapping with that device.

The modules implement inference. Program new weights by mapping or converting
the updated digital weights again.

### Layers, Models, And Training

`memfast.nn` contains inference layers (`AnalogLinear`, `AnalogConv2d`),
recurrent layers (`AnalogRNNCell`, `AnalogRNN`), trainable layers
(`HATLinear`, `HATConv2d`), conversion, and read-event utilities.
`memfast.models` supplies MLP and recurrent classifiers, CIFAR and ImageNet
architecture factories, and local language-model and detector loaders.
The numerical engine remains independent of these model libraries.

```python
from memfast.nn import HATLinear, convert_model

layer = HATLinear(torch.nn.Linear(128, 32), sim)
trainable = convert_model(model, sim, training=True, gradient_mode="quantized")
optimizer = torch.optim.Adam(trainable.parameters(), lr=1e-3)
```

HAT keeps floating master parameters and reprograms the analog weights on each
forward invocation. The forward values follow the same physical model as
inference. Default surrogate gradients use dequantized grouped inputs and
weights, with identity derivatives through their quantizers. Set
`gradient_mode="digital"` to use unquantized layer derivatives instead.
Neither option differentiates the noise, faults, or ADC response.
Convolution gradients use unfolded patches directly. Optimizer parameters and
registered buffers keep stable identities for distributed training.

Converted recurrent networks preserve stacked and bidirectional RNN execution,
hidden states, and the upstream tanh or ReLU nonlinearity. See the examples for
training, distributed launch, architecture construction, and checkpoint loading.

## Quantization And Physical Parallelism

Physical matrices use `[K, O]` coordinates: reduction rows and output columns.
PyTorch weights `[O, K]` are transposed internally.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `activation_bits` | Signed activation precision | `8` |
| `weight_bits` | Signed weight precision | `8` |
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
I_noisy = I + output_noise * read_rows * current_unit * Z
current_unit = (high_conductance - low_conductance) * read_voltage
```

It is sampled separately for every vector, subread, output column, and slice
pair. Its scale is relative to a fully active nominal subread. A shorter
physical final subread uses its actual physical row count.
This parameter models output-referred noise directly; `read_mode="output"`
instead derives the noise scale from the programmed conductances and inputs.
The two terms can be combined.

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
