// Array-local FP32 currents, per-slice ADC, and exact signed integer reconstruction.
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

__device__ __forceinline__ u32 mix(u32 x) {
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}
__device__ __forceinline__ float uniform(u64 address, u32 seed, u32 stream) {
    u32 key = mix(seed ^ (stream * 0x9e3779b9u));
    u32 h = mix(u32(address) ^ mix(u32(address >> 32)) ^ key);
    return (float(h >> 8) + .5f) * (1.f / 16777216.f);
}
__device__ __forceinline__ float normal(u64 address, u32 seed, u32 stream) {
    float u = uniform(address, seed, stream), v = uniform(address, seed, stream + 1);
    return sqrtf(-2.f * logf(u)) * cosf(6.283185307179586f * v);
}
__device__ __forceinline__ u32 event_seed(u32 seed, u64 event) {
    return seed ^ mix(u32(event)) ^ mix(u32(event >> 32));
}
__device__ __forceinline__ int significance(int bit, int bits) {
    return bit == bits - 1 ? -(1 << bit) : (1 << bit);
}

extern "C" __global__ void restore(
    const u8* packed, float* g, i64 count, int p, int begin, int width,
    u32 seed, u64 event) {
    for (i64 i = i64(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += i64(gridDim.x) * blockDim.x) {
        i64 t = i;
        int c = t % C; t /= C;
        int r = t % R; t /= R;
        int w = t % W; t /= W;
        int tile = begin + t % width, array = t / width;
        i64 cell = (((i64(array) * p + tile) * W + w) * R + r) * C + c;
        u8 q = packed[((i64(array) * p + tile) * R + r) * C + c];
        float raw = ((q >> w) & 1) ? HIGH : LOW;
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
    const u8* q, const float* g, float* out, int n, int m, int p,
    int begin, int width, int workers, u32 seed, u64 event) {
    constexpr int lanes = C / L;
    extern __shared__ __align__(16) unsigned char shared[];
    float* voltage = reinterpret_cast<float*>(shared);
    i64* sums = reinterpret_cast<i64*>(shared);
    int tid = threadIdx.x, worker = tid / lanes, col = tid % lanes;
    bool active = worker < workers;
    i64 vector_tiles = (n + V - 1) / V;
    i64 pair = i64(blockIdx.x) / vector_tiles;
    int first = (blockIdx.x % vector_tiles) * V;
    int pt = pair % width, array = pair / width;
    u32 read_seed = event_seed(seed, event);
    for (int j = tid; j < A * V * R; j += blockDim.x) {
        int a = j / (V * R), v = (j % (V * R)) / R, r = j % R;
        int vector = first + v;
        float value = 0.f;
        if (vector < n) {
            u8 code = q[(i64(vector) * m + array) * R + r];
            value = ((code >> a) & 1) ? VOLTAGE : 0.f;
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
            float currents[V * A * L] = {};
            float means[V * A * L] = {}, variances[V * A * L] = {};
            #pragma unroll 4
            for (int r = 0; r < R; ++r) {
                #pragma unroll
                for (int l = 0; l < L; ++l) {
                    int c = col + l * lanes;
                    float raw = g[(((i64(array) * width + pt) * W + w) * R + r) * C + c];
                    i64 cell = (((i64(array) * p + begin + pt) * W + w) * R + r) * C + c;
                    #pragma unroll
                    for (int a = 0; a < A; ++a) {
                        float read_g = raw;
                        if constexpr (READ_MODE == 1 && READ != 0.f)
                            read_g *= expf(READ * normal(cell * A + a, read_seed, 20));
                        float effective = __fsub_rn(read_g, LOW);
                        #pragma unroll
                        for (int v = 0; v < V; ++v) {
                            float voltage_value = voltage[(a * V + v) * R + r];
                            int index = (v * A + a) * L + l;
                            currents[index] = __fmaf_rn(voltage_value, effective, currents[index]);
                            if constexpr (READ_MODE == 2 && READ != 0.f) {
                                float vg = voltage_value * raw;
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
                        i64 address = ((((i64(first + v) * m + array) * p + begin + pt)
                                          * A + a) * W + w) * C + c;
                        float current = currents[index];
                        if constexpr (READ_MODE == 2 && READ != 0.f)
                            current += means[index] + sqrtf(variances[index])
                                       * normal(address, read_seed, 60);
                        if constexpr (ONOISE != 0.f)
                            current += ONOISE * UNIT * R * normal(address, read_seed, 50);
                        float z = current * INV_UNIT;
                        if constexpr (NONLINEAR != 0.f)
                            z = z / (1.f + fabsf((NONLINEAR / float(MAXIMUM)) * z));
                        int code = __float2int_rn(fminf(fmaxf(z, 0.f), float(MAXIMUM)));
                        totals[v * L + l] += i64(code) * significance(a, A)
                                            * significance(w, W);
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
                i64 index = (((i64(first + v) * m + array) * width + pt) * C
                              + col + l * lanes);
                out[index] = __ll2float_rn(total);
            }
        }
    }
}
