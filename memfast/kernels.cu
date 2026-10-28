// Array-local FP32 currents, per-slice ADC, and exact signed integer reconstruction.
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;
#if PRECISE_CURRENTS
typedef double physical_t;
#else
typedef float physical_t;
#endif
#if WIDE_CODES
typedef unsigned long long code_t;
#else
typedef unsigned char code_t;
#endif
#if WIDE_OUTPUT
typedef double result_t;
#else
typedef float result_t;
#endif

__device__ __forceinline__ u32 mix(u32 x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}
__device__ __forceinline__ float uniform(u64 address, u64 seed, u32 stream) {
    u32 key = mix(u32(seed) ^ (stream * 0x9e3779b9u));
    u32 high = u32(address >> 32);
    u32 folded = mix(high ^ mix(u32(seed >> 32)));
    if (high) folded ^= mix(high ^ key);
    u32 h = mix(u32(address) ^ folded ^ key);
    return (float(h >> 8) + .5f) * (1.f / 16777216.f);
}
__device__ __forceinline__ float normal(u64 address, u64 seed, u32 stream) {
    float u = uniform(address, seed, stream), v = uniform(address, seed, stream + 1);
    return sqrtf(-2.f * logf(u)) * cosf(6.283185307179586f * v);
}
__device__ __forceinline__ u64 event_seed(u32 seed, u64 event) {
    return u64(seed ^ mix(u32(event))) | (u64(mix(u32(event >> 32))) << 32);
}

extern "C" __global__ void restore(
    const code_t* packed, physical_t* g, i64 count, int p, int begin, int width,
    u32 seed, u64 event) {
    for (i64 i = i64(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += i64(gridDim.x) * blockDim.x) {
        i64 t = i;
        int c = t % C; t /= C;
        int r = t % R; t /= R;
        int w = t % W; t /= W;
        int tile = begin + t % width, array = t / width;
        i64 cell = (((i64(array) * p + tile) * W + w) * R + r) * C + c;
        code_t q = packed[((i64(array) * p + tile) * R + r) * C + c];
        int digit = (q >> W_SHIFT[w]) & W_MASK[w];
        physical_t raw = W_MASK[w] == 1 ? (digit ? HIGH : LOW)
            : LOW + (HIGH - LOW) * (physical_t(digit) / physical_t(W_MASK[w]));
        if constexpr (WRITE != 0.f) raw *= expf(WRITE * normal(cell, seed, 10));
        if constexpr (DRIFT != 1.f) raw *= DRIFT;
        if constexpr (SAF_LOW + SAF_HIGH != 0.f) {
            float u = uniform(cell, seed, 30);
            if (u < SAF_LOW) raw = LOW;
            else if (u < SAF_LOW + SAF_HIGH) raw = HIGH;
        }
        if constexpr (READ_MODE == 0 && READ != 0.f)
            raw *= expf(READ * normal(cell, event_seed(seed, event), 20));
        g[i] = raw;
    }
}

template<int V, int L>
__global__ void consume(
    const code_t* q, const physical_t* g, result_t* out, int n, int m, int p,
    int begin, int width, int workers, u32 seed, u64 event) {
    constexpr int lanes = C / L;
    extern __shared__ __align__(16) unsigned char shared[];
    physical_t* voltage = reinterpret_cast<physical_t*>(shared);
    i64* sums = reinterpret_cast<i64*>(shared);
    int tid = threadIdx.x, worker = tid / lanes, col = tid % lanes;
    bool active = worker < workers;
    i64 vector_tiles = (n + V - 1) / V;
    i64 pair = i64(blockIdx.x) / vector_tiles;
    int first = (blockIdx.x % vector_tiles) * V;
    int pt = pair % width, read_part = pair / width;
    int array = read_part / GROUPS, row_begin = (read_part % GROUPS) * ROWS;
    int read_rows = ROWS < R - row_begin ? ROWS : R - row_begin;
    u64 read_seed = event_seed(seed, event);
    for (int j = tid; j < A * V * ROWS; j += blockDim.x) {
        int a = j / (V * ROWS), v = (j % (V * ROWS)) / ROWS;
        int local_row = j % ROWS, r = row_begin + local_row;
        int vector = first + v;
        physical_t value = 0.f;
        if (vector < n && local_row < read_rows) {
            code_t code = q[(i64(vector) * m + array) * R + r];
            int digit = (code >> A_SHIFT[a]) & A_MASK[a];
            value = A_MASK[a] == 1 ? (digit ? VOLTAGE : 0.f)
                : VOLTAGE * (physical_t(digit) / physical_t(A_MASK[a]));
            if constexpr (VNOISE != 0.f) {
                i64 address = (((i64(vector) * m + array) * A + a) * R + r);
                value *= 1.f + VNOISE * normal(address, read_seed, 40);
            }
        }
        voltage[j] = value;
    }
    __syncthreads();
    i64 totals[V * L] = {};
    if (active) {
        for (int w = worker; w < W; w += workers) {
            physical_t currents[V * A * L] = {};
            physical_t means[V * A * L] = {}, variances[V * A * L] = {};
            #pragma unroll 4
            for (int local_row = 0; local_row < read_rows; ++local_row) {
                int r = row_begin + local_row;
                #pragma unroll
                for (int l = 0; l < L; ++l) {
                    int c = col + l * lanes;
                    physical_t raw = g[(((i64(array) * width + pt) * W + w) * R + r) * C + c];
                    i64 cell = (((i64(array) * p + begin + pt) * W + w) * R + r) * C + c;
                    #pragma unroll
                    for (int a = 0; a < A; ++a) {
                        physical_t read_g = raw;
                        if constexpr (READ_MODE == 1 && READ != 0.f)
                            read_g *= expf(READ * normal(cell * A + a, read_seed, 20));
                        physical_t effective = read_g - LOW;
                        #pragma unroll
                        for (int v = 0; v < V; ++v) {
                            physical_t voltage_value = voltage[(a * V + v) * ROWS + local_row];
                            int index = (v * A + a) * L + l;
                            #if PRECISE_CURRENTS
                            currents[index] = __fma_rn(voltage_value, effective, currents[index]);
                            #else
                            currents[index] = __fmaf_rn(voltage_value, effective, currents[index]);
                            #endif
                            if constexpr (READ_MODE == 2 && READ != 0.f) {
                                physical_t vg = voltage_value * raw;
                                means[index] += vg * expm1f(.5f * READ * READ);
                                variances[index] += vg * vg * expf(READ * READ)
                                                    * expm1f(READ * READ);
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                #pragma unroll
                for (int a = 0; a < A; ++a) {
                    #pragma unroll
                    for (int l = 0; l < L; ++l) {
                        int index = (v * A + a) * L + l;
                        int c = col + l * lanes;
                        i64 address = ((((i64(first + v) * m * GROUPS + read_part) * p + begin + pt)
                                          * A + a) * W + w) * C + c;
                        physical_t current = currents[index];
                        if constexpr (READ_MODE == 2 && READ != 0.f)
                            current += means[index] + sqrt(variances[index])
                                       * normal(address, read_seed, 60);
                        physical_t z = current * INV_UNIT;
                        if (A_MASK[a] != 1 || W_MASK[w] != 1)
                            z *= physical_t(A_MASK[a]) * physical_t(W_MASK[w]);
                        int maximum = ADC_MAXIMUM[w];
                        if constexpr (NONLINEAR != 0.f)
                            z = z / (1.f + fabs((NONLINEAR / physical_t(maximum)) * z));
                        #if PRECISE_CURRENTS
                        int code = __double2int_rn(fmin(fmax(z, 0.), double(maximum)));
                        #else
                        int code = __float2int_rn(fminf(fmaxf(z, 0.f), float(maximum)));
                        #endif
                        totals[v * L + l] += i64(code) * A_FACTOR[a] * W_FACTOR[w];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (active) {
        #pragma unroll
        for (int v = 0; v < V; ++v)
            #pragma unroll
            for (int l = 0; l < L; ++l)
                sums[(v * workers + worker) * C + col + l * lanes] = totals[v * L + l];
    }
    __syncthreads();
    if (active) {
        for (int v = worker; v < V; v += workers) {
            if (first + v >= n) continue;
            #pragma unroll
            for (int l = 0; l < L; ++l) {
                i64 total = 0;
                for (int w = 0; w < workers; ++w)
                    total += sums[(v * workers + w) * C + col + l * lanes];
                i64 index = (((i64(first + v) * m * GROUPS + read_part) * width + pt) * C
                              + col + l * lanes);
                #if WIDE_OUTPUT
                out[index] = __ll2double_rn(total);
                #else
                out[index] = __ll2float_rn(total);
                #endif
            }
        }
    }
}
