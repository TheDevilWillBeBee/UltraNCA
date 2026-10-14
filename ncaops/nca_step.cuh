#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Fused NCA update step for K NCAs with B samples each (image n belongs to NCA n / B):
//   p = [identity, sobel_x, sobel_x^T, laplacian] (x)    3x3, circular padding, feature 4c + f
//   h = relu(W1 p + b1),  u = W2 h,  y = x + u * m,      m = floor(U + update_rate), U ~ Philox(seed, pixel)
// STOCH = false: m = 1. ALIVE (growing NCA, alpha = channel 3): y = (x + u * m) * a with the alive mask
//   a = (maxpool3(x_alpha) > 0.1) & (maxpool3(y_alpha before masking) > 0.1)   (max-pools without wrap-around).
// It needs the neighbours' new alpha, so the forward kernel writes the unmasked y and its alpha plane, and
// alive_kernel then applies a and stores it (uint8) for the backward.
// Matmuls run on tensor cores (mma.sync m16n8k8 TF32, sm_80+). FP32 mode uses 3xTF32 (hi/lo operand split,
// three MMAs per product), which matches fp32 to ~1e-6 relative. One warp computes 16 pixels of an image row
// (the MMA M dimension); the four warps of a block cover a TILE_H x TILE_W tile, whose state (plus a 1-pixel
// circular halo) is staged in shared memory. Weights are staged once per block in "fragment-major" order: each
// lane reads its own pre-converted B-fragment registers, so the inner loops have no bank conflicts or conversions.
// The hidden activations stay in registers between the two matmuls: the accumulator columns (2t, 2t+1) of an
// n-tile are reused as the A-operand columns (t, t+4) of the next matmul by permuting W2's hidden index.
namespace nca_step {

// Tuning knobs (the NCA_* macros override them at build time for sweeps: build.sh "-DNCA_TILE_H=4 ...").
#ifndef NCA_TILE_H
#define NCA_TILE_H 16
#endif
#ifndef NCA_TILES_PER_BLOCK
#define NCA_TILES_PER_BLOCK 2
#endif
#ifndef NCA_BWD_TILES_PER_BLOCK
#define NCA_BWD_TILES_PER_BLOCK 8
#endif
#ifndef NCA_BWD_MIN_BLOCKS
#define NCA_BWD_MIN_BLOCKS 2
#endif
constexpr int TILE_W = 16;                            // pixels per warp row = MMA M; must stay 16
constexpr int TILE_H = NCA_TILE_H;                    // rows per forward block tile
constexpr int WARPS = 4;                              // each warp takes TILE_H / WARPS rows of a tile
constexpr int TILES_PER_BLOCK = NCA_TILES_PER_BLOCK;  // tiles of one image per block (amortises the weight staging)
constexpr int THREADS = 32 * WARPS;
static_assert(TILE_W == 16, "A warp's 16 pixels are the MMA M dimension");
static_assert(TILE_H % WARPS == 0, "Rows of a tile are split evenly over the warps");

constexpr int HALO_W = TILE_W + 2, HALO_H = TILE_H + 2;

// identity, sobel_x, sobel_x^T, laplacian (cross-correlation taps, row-major dy, dx in -1..1)
__constant__ float FILTERS[4][9] = {
    {0, 0, 0, 0, 1, 0, 0, 0, 0},
    {-1, 0, 1, -2, 0, 2, -1, 0, 1},
    {-1, -2, -1, 0, 0, 0, 1, 2, 1},
    {1, 2, 1, 2, -12, 2, 1, 2, 1},
};
// the same taps as a compile-time function (fully unrolled loops drop the zero taps)
__device__ __forceinline__ constexpr float filter_tap(int f, int i)
{
    constexpr float taps[4][9] = {{0, 0, 0, 0, 1, 0, 0, 0, 0}, {-1, 0, 1, -2, 0, 2, -1, 0, 1},
                                  {-1, -2, -1, 0, 0, 0, 1, 2, 1}, {1, 2, 1, 2, -12, 2, 1, 2, 1}};
    return taps[f][i];
}

template <int C, int HID>
struct Shape {
    static constexpr int PIN = 4 * C;            // perception features
    static constexpr int KS1 = PIN / 8;          // k-steps of p -> h
    static constexpr int NT1 = HID / 8;          // n-tiles of p -> h = k-steps of h -> u
    static constexpr int NT2 = (C + 7) / 8;      // n-tiles of h -> u (channels padded to 8)
    static_assert(PIN % 8 == 0, "4 * channels must be a multiple of 8 (even channel count)");
    static_assert(HID % 8 == 0, "hidden size must be a multiple of 8");
};

// ---------------------------------------------------------------- tensor-core helpers
__device__ __forceinline__ uint32_t tf32(float x)
{
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// An MMA operand: TF32 bits, plus the TF32 residual for the 3xTF32 (FP32) mode.
template <bool FP32>
struct Op4 {
    uint32_t hi[4];
    uint32_t lo[4];
    __device__ __forceinline__ void set(int i, float v)
    {
        hi[i] = tf32(v);
        if constexpr (FP32)
            lo[i] = tf32(v - __uint_as_float(hi[i]));
    }
    // v as returned by staged<FP32>()
    __device__ __forceinline__ void set_staged(int i, float v)
    {
        if constexpr (FP32)
            set(i, v);
        else
            hi[i] = __float_as_uint(v);
    }
};

// Value as staged in shared memory for later MMA operands: TF32-rounded in TF32 mode (converted once, at the store),
// unrounded in FP32 mode (split into hi / lo at the load).
template <bool FP32>
__device__ __forceinline__ float staged(float v)
{
    if constexpr (FP32)
        return v;
    else
        return __uint_as_float(tf32(v));
}

// B fragment (b0, b1): TF32 bits, or the fp32 bits in FP32 mode (split into hi / lo at the MMA, which keeps the
// shared-memory footprint of the weights the same in both modes).
template <bool FP32>
struct FragB {
    uint2 v;
};

template <bool FP32>
__device__ __forceinline__ void make_frag(FragB<FP32> &f, float b0, float b1)
{
    f.v = FP32 ? make_uint2(__float_as_uint(b0), __float_as_uint(b1)) : make_uint2(tf32(b0), tf32(b1));
}

template <bool FP32>
__device__ __forceinline__ void mma(float (&d)[4], const Op4<FP32> &a, const FragB<FP32> &b)
{
    if constexpr (FP32) {  // small products first
        const uint32_t h0 = tf32(__uint_as_float(b.v.x)), h1 = tf32(__uint_as_float(b.v.y));
        mma_tf32(d, a.lo, h0, h1);
        mma_tf32(d, a.hi, tf32(__uint_as_float(b.v.x) - __uint_as_float(h0)), tf32(__uint_as_float(b.v.y) - __uint_as_float(h1)));
        mma_tf32(d, a.hi, h0, h1);
    } else {
        mma_tf32(d, a.hi, b.v.x, b.v.y);
    }
}

// ---------------------------------------------------------------- update mask
// Philox4x32-10 on counter (pixel, 0, 0, 0) with key = seed; first output word -> U in [0, 1) with 24 bits.
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t pixel)
{
    uint32_t c0 = (uint32_t)pixel, c1 = (uint32_t)(pixel >> 32), c2 = 0, c3 = 0;
    uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t lo0 = 0xD2511F53u * c0, hi0 = __umulhi(0xD2511F53u, c0);
        const uint32_t lo1 = 0xCD9E8D57u * c2, hi1 = __umulhi(0xCD9E8D57u, c2);
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return (c0 >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float update_mask(uint64_t seed, uint64_t pixel, float update_rate)
{
    return floorf(philox_uniform(seed, pixel) + update_rate);
}

template <bool STOCH>
__device__ __forceinline__ float step_mask(uint64_t seed, uint64_t pixel, float update_rate)
{
    return STOCH ? update_mask(seed, pixel, update_rate) : 1.f;
}

__global__ void mask_kernel(float *__restrict__ mask, uint64_t total, uint64_t seed, float update_rate)
{
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < total; i += (uint64_t)gridDim.x * blockDim.x)
        mask[i] = update_mask(seed, i, update_rate);
}

// ---------------------------------------------------------------- shared-memory layout
template <int C, int HID, bool FP32>
struct ForwardSmem {
    using S = Shape<C, HID>;
    FragB<FP32> w1[S::NT1][S::KS1][32];   // p -> h: b0 = W1[8j+g][8s+t], b1 = W1[8j+g][8s+t+4]
    FragB<FP32> w2[S::NT2][S::NT1][32];   // h -> u: b0 = W2[8n+g][8j+2t], b1 = W2[8n+g][8j+2t+1] (permuted hidden)
    float b1[HID];
    float x[C][HALO_H][HALO_W];           // state tile with circular halo
};

// Stage one NCA's weights in fragment-major order.
template <int C, int HID, bool FP32>
__device__ void stage_forward_weights(ForwardSmem<C, HID, FP32> &sm, const float *__restrict__ w1,
                                      const float *__restrict__ b1, const float *__restrict__ w2)
{
    using S = Shape<C, HID>;
    for (int i = threadIdx.x; i < S::NT1 * S::KS1 * 32; i += THREADS) {
        const int lane = i % 32, s = (i / 32) % S::KS1, j = i / (32 * S::KS1), g = lane / 4, t = lane % 4;
        const float *row = w1 + (8 * j + g) * S::PIN + 8 * s;
        make_frag<FP32>(sm.w1[j][s][lane], row[t], row[t + 4]);
    }
    for (int i = threadIdx.x; i < S::NT2 * S::NT1 * 32; i += THREADS) {
        const int lane = i % 32, j = (i / 32) % S::NT1, n = i / (32 * S::NT1), g = lane / 4, t = lane % 4;
        const int ch = 8 * n + g;
        const float v0 = ch < C ? w2[ch * HID + 8 * j + 2 * t] : 0.f;
        const float v1 = ch < C ? w2[ch * HID + 8 * j + 2 * t + 1] : 0.f;
        make_frag<FP32>(sm.w2[n][j][lane], v0, v1);
    }
    for (int i = threadIdx.x; i < HID; i += THREADS)
        sm.b1[i] = b1[i];
}

// Load a CH x (TH + 2) x (TILE_W + 2) tile of an image (CH channel planes of H x W) with circular wrap.
template <int CH, int TH>
__device__ void stage_tile(float (&xs)[CH][TH + 2][HALO_W], const float *__restrict__ x, int H, int W, int y0, int x0)
{
    for (int i = threadIdx.x; i < CH * (TH + 2) * HALO_W; i += THREADS) {
        const int col = i % HALO_W, row = (i / HALO_W) % (TH + 2), c = i / (HALO_W * (TH + 2));
        int gy = y0 - 1 + row, gx = x0 - 1 + col;
        gy = gy < 0 ? gy + H : (gy >= H ? gy - H : gy);
        gx = gx < 0 ? gx + W : (gx >= W ? gx - W : gx);
        xs[c][row][col] = x[((size_t)c * H + gy) * W + gx];
    }
}

// 9-tap filter f of channel c at tile pixel (row, col) (halo coordinates of the centre are row + 1, col + 1).
template <int C, int TH>
__device__ __forceinline__ float perceive(const float (&xs)[C][TH + 2][HALO_W], int f, int c, int row, int col)
{
    float v = 0.f;
#pragma unroll
    for (int dy = 0; dy < 3; ++dy)
#pragma unroll
        for (int dx = 0; dx < 3; ++dx)
            v = fmaf(FILTERS[f][3 * dy + dx], xs[c][row + dy][col + dx], v);
    return v;
}

// ---------------------------------------------------------------- forward
template <int C, int HID, bool FP32, bool STOCH, bool ALIVE>
__global__ void __launch_bounds__(THREADS) forward_kernel(
    const float *__restrict__ x, const float *__restrict__ w1, const float *__restrict__ b1, const float *__restrict__ w2,
    float *__restrict__ y, float *__restrict__ alpha, int B, int H, int W, uint64_t seed, float update_rate)
{
    using S = Shape<C, HID>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto &sm = *reinterpret_cast<ForwardSmem<C, HID, FP32> *>(smem_raw);

    const int n = blockIdx.y, k = n / B;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int tiles_x = (W + TILE_W - 1) / TILE_W, tiles = tiles_x * ((H + TILE_H - 1) / TILE_H);
    const float *xn = x + (size_t)n * C * H * W;
    float *yn = y + (size_t)n * C * H * W;

    stage_forward_weights<C, HID, FP32>(sm, w1 + (size_t)k * HID * S::PIN, b1 + (size_t)k * HID, w2 + (size_t)k * C * HID);

    for (int tile = blockIdx.x * TILES_PER_BLOCK; tile < min(tiles, (blockIdx.x + 1) * TILES_PER_BLOCK); ++tile) {
        const int y0 = (tile / tiles_x) * TILE_H, x0 = (tile % tiles_x) * TILE_W;
        __syncthreads();  // previous tile's reads (and the weight staging) are complete
        stage_tile<C, TILE_H>(sm.x, xn, H, W, y0, x0);
        __syncthreads();

        for (int row = warp * (TILE_H / WARPS); row < (warp + 1) * (TILE_H / WARPS); ++row) {
            // A operand of p -> h: this lane computes filter t of channels (2s, 2s + 1) at pixels g and g + 8.
            Op4<FP32> a[S::KS1];
#pragma unroll
            for (int s = 0; s < S::KS1; ++s) {
                a[s].set(0, perceive<C, TILE_H>(sm.x, t, 2 * s, row, g));
                a[s].set(1, perceive<C, TILE_H>(sm.x, t, 2 * s, row, g + 8));
                a[s].set(2, perceive<C, TILE_H>(sm.x, t, 2 * s + 1, row, g));
                a[s].set(3, perceive<C, TILE_H>(sm.x, t, 2 * s + 1, row, g + 8));
            }
            // h = relu(W1 p + b1): accumulator (pixel g / g + 8, hidden 8j + 2t / 2t + 1)
            float h[S::NT1][4];
#pragma unroll
            for (int j = 0; j < S::NT1; ++j) {
                h[j][0] = h[j][2] = sm.b1[8 * j + 2 * t];
                h[j][1] = h[j][3] = sm.b1[8 * j + 2 * t + 1];
            }
#pragma unroll
            for (int s = 0; s < S::KS1; ++s)  // k outer: consecutive MMAs are independent
#pragma unroll
                for (int j = 0; j < S::NT1; ++j)
                    mma<FP32>(h[j], a[s], sm.w1[j][s][lane]);
#pragma unroll
            for (int j = 0; j < S::NT1; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    h[j][e] = fmaxf(h[j][e], 0.f);
            // u = W2 h: accumulator columns (2t, 2t + 1) of n-tile j are the A columns (t, t + 4) of k-step j.
            float u[S::NT2][4] = {};
#pragma unroll
            for (int j = 0; j < S::NT1; ++j) {
                Op4<FP32> ah;
                ah.set(0, h[j][0]); ah.set(1, h[j][2]); ah.set(2, h[j][1]); ah.set(3, h[j][3]);
#pragma unroll
                for (int nt = 0; nt < S::NT2; ++nt)
                    mma<FP32>(u[nt], ah, sm.w2[nt][j][lane]);
            }
            // y = x + u * m for pixels g, g + 8 and channels 8nt + 2t, 8nt + 2t + 1
            const int gy = y0 + row;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int px = g + 8 * half, gx = x0 + px;
                if (gy >= H || gx >= W)
                    continue;
                const float m = step_mask<STOCH>(seed, ((uint64_t)n * H + gy) * W + gx, update_rate);
#pragma unroll
                for (int nt = 0; nt < S::NT2; ++nt)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int ch = 8 * nt + 2 * t + e;
                        if (ch >= C)
                            continue;
                        const float v = sm.x[ch][row + 1][px + 1] + u[nt][2 * half + e] * m;
                        yn[((size_t)ch * H + gy) * W + gx] = v;
                        if (ALIVE && ch == 3)
                            alpha[((size_t)n * H + gy) * W + gx] = v;
                    }
            }
        }
    }
}

// Alive mask of a growing-NCA step: a = (maxpool3(x_alpha) > 0.1) & (maxpool3(alpha) > 0.1) without wrap-around,
// alpha = the unmasked new alpha written by forward_kernel. y *= a; alive = a (for the backward). One thread per pixel.
template <int C>
__global__ void alive_kernel(const float *__restrict__ x, const float *__restrict__ alpha, float *__restrict__ y,
                             uint8_t *__restrict__ alive, int H, int W, uint64_t total)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;  // n * H * W + Y * W + X
    if (i >= total)
        return;
    const int X = i % W, Y = (i / W) % H;
    const uint64_t plane = (uint64_t)H * W, n = i / plane;
    const float *xa = x + (n * C + 3) * plane, *ya = alpha + n * plane;
    float mx = 0.f, my = 0.f;  // only the comparison with 0.1 matters
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int yy = Y + dy, xx = X + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                mx = fmaxf(mx, xa[yy * W + xx]);
                my = fmaxf(my, ya[yy * W + xx]);
            }
        }
    const bool a = mx > 0.1f && my > 0.1f;
    alive[i] = a;
    if (!a)
#pragma unroll
        for (int c = 0; c < C; ++c)
            y[(n * C + c) * plane + (i - n * plane)] = 0.f;
}

// ================================================================ backward
// Main kernel (one 16-pixel row per warp, BWD_TILE_H rows per block tile): recompute p and h = W1 p + b1, stage
// p (with a ones column, so that db1 falls out of the dW1 product), relu(h) and g*m in shared memory, then
//   dh = (g * m) W2 * [h > 0]          (MMA, registers)
//   dp = dh W1                          (MMA, permuted hidden index as in the forward) -> global (N, 4C, H, W)
//   dW2 += (g * m)^T relu(h),  [dW1 | db1] += dh^T [p | 1]   (MMA over the pixels of the tile, per-block accumulators)
// Each block writes its weight-gradient partials; reduce_kernel sums them per NCA in a fixed order (deterministic).
// adjoint_kernel then forms dx = g + sum_f K_f^T * dp_f (transpose of the circular 3x3 filters).
constexpr int BWD_TILE_H = WARPS;          // one 16-pixel row per warp
constexpr int BWD_PIX = BWD_TILE_H * TILE_W;
constexpr int BWD_TILES_PER_BLOCK = NCA_BWD_TILES_PER_BLOCK;  // tiles of one image per block (amortises accumulator write-out)
constexpr int BWD_MIN_BLOCKS = NCA_BWD_MIN_BLOCKS;            // resident blocks per SM (register bound via __launch_bounds__)
#ifndef NCA_ABLATE
#define NCA_ABLATE 0   // profiling only: 1 = skip weight gradients, 2 = skip dp stores, 3 = both
#endif

// smallest stride >= n with stride % 32 in {8, 24}: lanes (g, t) reading [8k + t][g] hit 32 distinct banks
constexpr int conflict_free_stride(int n)
{
    return (n % 32 == 8 || n % 32 == 24) ? n : conflict_free_stride(n + 1);
}

template <int C, int HID>
struct BwdShape {
    using S = Shape<C, HID>;
    static constexpr int PIN = S::PIN, KS1 = S::KS1, NT1 = S::NT1, NT2 = S::NT2;
    static constexpr int NTP = PIN / 8;                      // n-tiles of dh -> dp
    static constexpr int PAUG = PIN + 8;                     // [p | 1 | 0 ... 0]
    static constexpr int MTA = (PAUG + 15) / 16;             // weight-gradient m-tiles of [p | 1]^T (features)
    static constexpr int MT2 = (C + 15) / 16;                // weight-gradient m-tiles of gm^T (channels)
    static constexpr int CPAD = 16 * MT2;
    static constexpr int NIW = NT1 / WARPS;                  // hidden n-tiles owned by each warp
    static_assert(NT1 % WARPS == 0, "hidden n-tiles are split evenly over the warps");
    static constexpr int P_STRIDE = conflict_free_stride(PAUG), H_STRIDE = conflict_free_stride(HID);
    static constexpr int G_STRIDE = conflict_free_stride(CPAD);
    static constexpr int PARTIAL = HID * (PIN + 1) + C * HID;  // [dW1 | db1] then dW2, per block
    static_assert(CPAD >= 8 * NT2, "channel padding");
};

// Per-NCA B fragments for the backward matmuls (packed by pack_kernel).
template <int C, int HID, bool FP32>
struct BackwardWeights {
    using BS = BwdShape<C, HID>;
    FragB<FP32> w1h[BS::NT1][BS::KS1][32];   // h = W1 p: b0 = W1[8j+g][8s+t], b1 = W1[8j+g][8s+t+4]
    FragB<FP32> w2d[BS::NT2][BS::NT1][32];   // dh = gm W2: b0 = W2[8q+t][8j+g], b1 = W2[8q+t+4][8j+g]
    FragB<FP32> w1p[BS::NT1][BS::NTP][32];   // dp = dh W1: b0 = W1[8j+2t][8n+g], b1 = W1[8j+2t+1][8n+g]
    float b1[HID];
};

template <int C, int HID, bool FP32>
__global__ void pack_kernel(const float *__restrict__ w1, const float *__restrict__ b1, const float *__restrict__ w2,
                            BackwardWeights<C, HID, FP32> *__restrict__ packed)
{
    using BS = BwdShape<C, HID>;
    const int k = blockIdx.x;
    w1 += (size_t)k * HID * BS::PIN; b1 += (size_t)k * HID; w2 += (size_t)k * C * HID;
    auto &pk = packed[k];
    for (int i = threadIdx.x; i < BS::NT1 * BS::KS1 * 32; i += blockDim.x) {
        const int lane = i % 32, s = (i / 32) % BS::KS1, j = i / (32 * BS::KS1), g = lane / 4, t = lane % 4;
        make_frag<FP32>(pk.w1h[j][s][lane], w1[(8 * j + g) * BS::PIN + 8 * s + t], w1[(8 * j + g) * BS::PIN + 8 * s + t + 4]);
    }
    for (int i = threadIdx.x; i < BS::NT2 * BS::NT1 * 32; i += blockDim.x) {
        const int lane = i % 32, j = (i / 32) % BS::NT1, q = i / (32 * BS::NT1), g = lane / 4, t = lane % 4;
        const int c0 = 8 * q + t, c1 = 8 * q + t + 4;
        make_frag<FP32>(pk.w2d[q][j][lane], c0 < C ? w2[c0 * HID + 8 * j + g] : 0.f, c1 < C ? w2[c1 * HID + 8 * j + g] : 0.f);
    }
    for (int i = threadIdx.x; i < BS::NT1 * BS::NTP * 32; i += blockDim.x) {
        const int lane = i % 32, nt = (i / 32) % BS::NTP, j = i / (32 * BS::NTP), g = lane / 4, t = lane % 4;
        make_frag<FP32>(pk.w1p[j][nt][lane], w1[(8 * j + 2 * t) * BS::PIN + 8 * nt + g], w1[(8 * j + 2 * t + 1) * BS::PIN + 8 * nt + g]);
    }
    for (int i = threadIdx.x; i < HID; i += blockDim.x)
        pk.b1[i] = b1[i];
}

template <int C, int HID>
struct BackwardSmem {
    using BS = BwdShape<C, HID>;
    float x[C][BWD_TILE_H + 2][HALO_W];
    float p[BWD_PIX][BS::P_STRIDE];      // [p | 1 | 0]
    float h[BWD_PIX][BS::H_STRIDE];      // relu(h), later dh
    float gm[BWD_PIX][BS::G_STRIDE];     // g * m (channels zero-padded)
};

template <bool FP32>
__device__ __forceinline__ FragB<FP32> frag_from_staged(float b0, float b1)
{
    return {make_uint2(__float_as_uint(b0), __float_as_uint(b1))};
}

template <int C, int HID, bool FP32, bool STOCH, bool ALIVE>
__global__ void __launch_bounds__(THREADS, BWD_MIN_BLOCKS) backward_kernel(
    const float *__restrict__ x, const float *__restrict__ gy, const uint8_t *__restrict__ alive,
    const BackwardWeights<C, HID, FP32> *__restrict__ packed, float *__restrict__ dp, float *__restrict__ partial,
    int B, int H, int W, uint64_t seed, float update_rate)
{
    using BS = BwdShape<C, HID>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto &sm = *reinterpret_cast<BackwardSmem<C, HID> *>(smem_raw);

    const int n = blockIdx.y, k = n / B;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int tiles_x = (W + TILE_W - 1) / TILE_W, tiles = tiles_x * ((H + BWD_TILE_H - 1) / BWD_TILE_H);
    const size_t plane = (size_t)H * W;
    const float *xn = x + n * C * plane, *gyn = gy + n * C * plane;
    float *dpn = dp + n * BS::PIN * plane;
    const auto &wk = packed[k];

    float accp[BS::MTA][BS::NIW][4] = {};   // [p | 1]^T dh  -> [dW1 | db1]^T, hidden n-tiles of this warp
    float accg[BS::MT2][BS::NIW][4] = {};   // gm^T relu(h)  -> dW2
    const int row = warp;
    for (int tile = blockIdx.x * BWD_TILES_PER_BLOCK; tile < min(tiles, (blockIdx.x + 1) * BWD_TILES_PER_BLOCK); ++tile) {
        const int y0 = (tile / tiles_x) * BWD_TILE_H, x0 = (tile % tiles_x) * TILE_W, Y = y0 + row;
        __syncthreads();  // previous tile's shared reads are complete
        stage_tile<C, BWD_TILE_H>(sm.x, xn, H, W, y0, x0);
        __syncthreads();

        // ---- p (A fragments + shared copy with the ones column), h = W1 p + b1
        Op4<FP32> a[BS::KS1];
#pragma unroll
        for (int s = 0; s < BS::KS1; ++s) {
            const float v0 = perceive<C, BWD_TILE_H>(sm.x, t, 2 * s, row, g), v1 = perceive<C, BWD_TILE_H>(sm.x, t, 2 * s, row, g + 8);
            const float v2 = perceive<C, BWD_TILE_H>(sm.x, t, 2 * s + 1, row, g), v3 = perceive<C, BWD_TILE_H>(sm.x, t, 2 * s + 1, row, g + 8);
            a[s].set(0, v0); a[s].set(1, v1); a[s].set(2, v2); a[s].set(3, v3);
            float *p0 = sm.p[16 * row + g], *p1 = sm.p[16 * row + g + 8];
            p0[8 * s + t] = staged<FP32>(v0); p1[8 * s + t] = staged<FP32>(v1);
            p0[8 * s + t + 4] = staged<FP32>(v2); p1[8 * s + t + 4] = staged<FP32>(v3);
        }
        for (int i = lane; i < 16 * (BS::P_STRIDE - BS::PIN); i += 32)
            sm.p[16 * row + i % 16][BS::PIN + i / 16] = (i / 16 == 0) ? 1.f : 0.f;
        uint32_t positive[(BS::NT1 * 4 + 31) / 32] = {};
#pragma unroll
        for (int j = 0; j < BS::NT1; ++j) {
            float h[4] = {wk.b1[8 * j + 2 * t], wk.b1[8 * j + 2 * t + 1], wk.b1[8 * j + 2 * t], wk.b1[8 * j + 2 * t + 1]};
#pragma unroll
            for (int s = 0; s < BS::KS1; ++s)
                mma<FP32>(h, a[s], wk.w1h[j][s][lane]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                if (h[e] > 0.f)
                    positive[(4 * j + e) / 32] |= 1u << ((4 * j + e) % 32);
                sm.h[16 * row + g + 8 * (e / 2)][8 * j + 2 * t + e % 2] = staged<FP32>(fmaxf(h[e], 0.f));
            }
        }

        // ---- g * m (* a) (A fragments of dh = gm W2, shared copy for dW2); zero outside the image
        float mrow[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int X = x0 + g + 8 * half;
            const uint64_t pix = ((uint64_t)n * H + Y) * W + X;
            mrow[half] = (Y < H && X < W) ? step_mask<STOCH>(seed, pix, update_rate) * (ALIVE ? alive[pix] : 1) : 0.f;
        }
        Op4<FP32> ag[BS::NT2];
#pragma unroll
        for (int q = 0; q < BS::NT2; ++q)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int px = g + 8 * (e % 2), ch = 8 * q + t + 4 * (e / 2), X = x0 + px;
                const float v = (ch < C && mrow[e % 2] != 0.f) ? gyn[ch * plane + (size_t)Y * W + X] * mrow[e % 2] : 0.f;
                ag[q].set(e, v);
                sm.gm[16 * row + px][ch] = staged<FP32>(v);
            }
        for (int i = lane; i < 16 * (BS::CPAD - 8 * BS::NT2); i += 32)
            sm.gm[16 * row + i % 16][8 * BS::NT2 + i / 16] = 0.f;

        // ---- dh = gm W2 * [h > 0] (registers), dp = dh W1 -> global
        float dh[BS::NT1][4];
#pragma unroll
        for (int j = 0; j < BS::NT1; ++j) {
            dh[j][0] = dh[j][1] = dh[j][2] = dh[j][3] = 0.f;
#pragma unroll
            for (int q = 0; q < BS::NT2; ++q)
                mma<FP32>(dh[j], ag[q], wk.w2d[q][j][lane]);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                if (!((positive[(4 * j + e) / 32] >> ((4 * j + e) % 32)) & 1u))
                    dh[j][e] = 0.f;
        }
        {
            float d[BS::NTP][4] = {};
#pragma unroll
            for (int j = 0; j < BS::NT1; ++j) {
                Op4<FP32> ad;
                ad.set(0, dh[j][0]); ad.set(1, dh[j][2]); ad.set(2, dh[j][1]); ad.set(3, dh[j][3]);
#pragma unroll
                for (int nt = 0; nt < BS::NTP; ++nt)
                    mma<FP32>(d[nt], ad, wk.w1p[j][nt][lane]);
            }
            if (Y < H && !(NCA_ABLATE & 2))
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int X = x0 + g + 8 * half;
                    if (X < W)
#pragma unroll
                        for (int nt = 0; nt < BS::NTP; ++nt)
#pragma unroll
                            for (int e = 0; e < 2; ++e)
                                dpn[(8 * nt + 2 * t + e) * plane + (size_t)Y * W + X] = d[nt][2 * half + e];
                }
        }

        // ---- dW2 += gm^T relu(h) over the tile's pixels. Each warp owns NIW hidden n-tiles, so one B fragment
        //      serves all m-tiles and the A fragments are shared by the warp's n-tiles.
        __syncthreads();
        if constexpr (NCA_ABLATE & 1) continue;
#pragma unroll
        for (int kk = 0; kk < BWD_PIX / 8; ++kk) {
            Op4<FP32> ag2[BS::MT2];
#pragma unroll
            for (int mi = 0; mi < BS::MT2; ++mi) {
                ag2[mi].set_staged(0, sm.gm[8 * kk + t][16 * mi + g]); ag2[mi].set_staged(1, sm.gm[8 * kk + t][16 * mi + g + 8]);
                ag2[mi].set_staged(2, sm.gm[8 * kk + t + 4][16 * mi + g]); ag2[mi].set_staged(3, sm.gm[8 * kk + t + 4][16 * mi + g + 8]);
            }
#pragma unroll
            for (int q = 0; q < BS::NIW; ++q) {
                const int ni = warp * BS::NIW + q;
                const auto b = frag_from_staged<FP32>(sm.h[8 * kk + t][8 * ni + g], sm.h[8 * kk + t + 4][8 * ni + g]);
#pragma unroll
                for (int mi = 0; mi < BS::MT2; ++mi)
                    mma<FP32>(accg[mi][q], ag2[mi], b);
            }
        }
        // ---- [dW1 | db1]^T += [p | 1]^T dh
        __syncthreads();
#pragma unroll
        for (int j = 0; j < BS::NT1; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                sm.h[16 * row + g + 8 * (e / 2)][8 * j + 2 * t + e % 2] = staged<FP32>(dh[j][e]);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BWD_PIX / 8; ++kk) {
            Op4<FP32> ap[BS::MTA];
#pragma unroll
            for (int mi = 0; mi < BS::MTA; ++mi)
#pragma unroll
                for (int e = 0; e < 4; ++e) {  // feature rows 16 mi + g (+ 8): past the [p | 1 | 0] columns they are zero
                    const int fb = 16 * mi + 8 * (e % 2);
                    ap[mi].set_staged(e, fb < BS::PAUG ? sm.p[8 * kk + t + 4 * (e / 2)][fb + g] : 0.f);
                }
#pragma unroll
            for (int q = 0; q < BS::NIW; ++q) {
                const int ni = warp * BS::NIW + q;
                const auto b = frag_from_staged<FP32>(sm.h[8 * kk + t][8 * ni + g], sm.h[8 * kk + t + 4][8 * ni + g]);
#pragma unroll
                for (int mi = 0; mi < BS::MTA; ++mi)
                    mma<FP32>(accp[mi][q], ap[mi], b);
            }
        }
    }

    // ---- write this block's partial weight gradients: [dW1 | db1] (HID x (PIN + 1)), then dW2 (C x HID)
    float *out = partial + ((size_t)n * gridDim.x + blockIdx.x) * BS::PARTIAL;
#pragma unroll
    for (int q = 0; q < BS::NIW; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int hid = 8 * (warp * BS::NIW + q) + 2 * t + e % 2;
#pragma unroll
            for (int mi = 0; mi < BS::MTA; ++mi) {
                const int f = 16 * mi + g + 8 * (e / 2);
                if (f <= BS::PIN)
                    out[hid * (BS::PIN + 1) + f] = accp[mi][q][e];
            }
#pragma unroll
            for (int mi = 0; mi < BS::MT2; ++mi) {
                const int ch = 16 * mi + g + 8 * (e / 2);
                if (ch < C)
                    out[HID * (BS::PIN + 1) + ch * HID + hid] = accg[mi][q][e];
            }
        }
}

// Sum the partials of each NCA's blocks in block order -> dw1 (K, HID, PIN), db1 (K, HID), dw2 (K, C, HID).
template <int C, int HID>
__global__ void reduce_kernel(const float *__restrict__ partial, int blocks_per_nca, float *__restrict__ dw1,
                              float *__restrict__ db1, float *__restrict__ dw2)
{
    using BS = BwdShape<C, HID>;
    const int k = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= BS::PARTIAL)
        return;
    const float *src = partial + (size_t)k * blocks_per_nca * BS::PARTIAL + e;
    float s = 0.f;
    for (int b = 0; b < blocks_per_nca; ++b)
        s += src[(size_t)b * BS::PARTIAL];
    if (e < HID * (BS::PIN + 1)) {
        const int hid = e / (BS::PIN + 1), f = e % (BS::PIN + 1);
        if (f < BS::PIN)
            dw1[((size_t)k * HID + hid) * BS::PIN + f] = s;
        else
            db1[(size_t)k * HID + hid] = s;
    } else {
        dw2[(size_t)k * C * HID + (e - HID * (BS::PIN + 1))] = s;
    }
}

// dx = g (* a) + sum_f sum_o K_f(o) dp_{4c+f}(y - o), the transpose of the circular perception filters. One thread
// per (pixel, channel); neighbouring taps are served by L1 (faster than shared-memory staging of all 4C planes).
template <int C, bool ALIVE>
__global__ void __launch_bounds__(256) adjoint_kernel(const float *__restrict__ dp, const float *__restrict__ gy,
                                                      const uint8_t *__restrict__ alive, float *__restrict__ gx,
                                                      int H, int W, uint64_t total)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int X = i % W, Y = (i / W) % H;
    const uint64_t nc = i / ((uint64_t)H * W);           // n * C + c
    const uint64_t n = nc / C, c = nc % C;
    const float *d = dp + (n * 4 * C + 4 * c) * (uint64_t)H * W;
    float v = ALIVE ? (alive[n * H * W + (uint64_t)Y * W + X] ? gy[i] : 0.f) : gy[i];
#pragma unroll
    for (int dy = 0; dy < 3; ++dy) {
        int yy = Y + 1 - dy;
        yy = yy < 0 ? yy + H : (yy >= H ? yy - H : yy);
#pragma unroll
        for (int dx = 0; dx < 3; ++dx) {
            int xx = X + 1 - dx;
            xx = xx < 0 ? xx + W : (xx >= W ? xx - W : xx);
            const uint64_t o = (uint64_t)yy * W + xx;
#pragma unroll
            for (int f = 0; f < 4; ++f)
                if (filter_tap(f, 3 * dy + dx) != 0.f)
                    v = fmaf(filter_tap(f, 3 * dy + dx), __ldg(d + f * (uint64_t)H * W + o), v);
        }
    }
    gx[i] = v;
}

}  // namespace nca_step
