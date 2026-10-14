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
#ifndef NCA_FWD_WARPS
#define NCA_FWD_WARPS 8
#endif
#ifndef NCA_BWD_TILES_PER_BLOCK
#define NCA_BWD_TILES_PER_BLOCK 8
#endif
#ifndef NCA_BWD_MIN_BLOCKS
#define NCA_BWD_MIN_BLOCKS 2
#endif
constexpr int TILE_W = 16;                            // pixels per warp row = MMA M; must stay 16
constexpr int TILE_H = NCA_TILE_H;                    // rows per forward block tile
constexpr int FWD_WARPS = NCA_FWD_WARPS;              // forward: each warp takes TILE_H / FWD_WARPS rows of a tile
constexpr int TILES_PER_BLOCK = NCA_TILES_PER_BLOCK;  // tiles of one image per block (amortises the weight staging)
constexpr int FWD_THREADS = 32 * FWD_WARPS;
static_assert(TILE_W == 16, "A warp's 16 pixels are the MMA M dimension");
static_assert(TILE_H % FWD_WARPS == 0, "Rows of a tile are split evenly over the warps");

constexpr int HALO_W = TILE_W + 2, HALO_H = TILE_H + 2;

// identity, sobel_x, sobel_x^T, laplacian (cross-correlation taps, row-major dy, dx in -1..1); fully unrolled loops
// drop the zero taps
__device__ __forceinline__ constexpr float filter_tap(int f, int i)
{
    constexpr float taps[4][9] = {{0, 0, 0, 0, 1, 0, 0, 0, 0}, {-1, 0, 1, -2, 0, 2, -1, 0, 1},
                                  {-1, -2, -1, 0, 0, 0, 1, 2, 1}, {1, 2, 1, 2, -12, 2, 1, 2, 1}};
    return taps[f][i];
}

// smallest stride >= n with stride % 32 in {8, 24}: lanes (g, t) reading [8k + t][g] hit 32 distinct banks
constexpr int conflict_free_stride(int n)
{
    return (n % 32 == 8 || n % 32 == 24) ? n : conflict_free_stride(n + 1);
}

// K order of the perception features in the p -> h MMA: in k-steps (2q, 2q + 1), lane t computes all four filters of
// channel 4q + t (its 3 x 3 neighbourhood is loaded once). Column k of k-step s is channel 4 (s / 2) + k % 4, filter
// 2 (s % 2) + k / 4, i.e. feature 4 c + f of W1.
__host__ __device__ constexpr int feature(int s, int k) { return 4 * (4 * (s / 2) + k % 4) + 2 * (s % 2) + k / 4; }

template <int C, int HID>
struct Shape {
    static constexpr int PIN = 4 * C;            // perception features
    static constexpr int KS1 = PIN / 8;          // k-steps of p -> h
    static constexpr int NT1 = HID / 8;          // n-tiles of p -> h = k-steps of h -> u
    static constexpr int NT2 = (C + 7) / 8;      // n-tiles of h -> u (channels padded to 8)
    static_assert(C % 4 == 0, "channels must be a multiple of 4 (one channel per lane of a k-step pair)");
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
// m = floor(U + update_rate) with U uniform in [0, 1) (24 bits). Pixel (n, Y, X) uses word X % 4 of the Philox4x32-10
// block (key = seed) of counter (n H + Y) ceil(W / 4) + X / 4: one Philox call serves four neighbouring pixels.
__device__ __forceinline__ uint4 philox(uint64_t seed, uint64_t counter)
{
    uint32_t c0 = (uint32_t)counter, c1 = (uint32_t)(counter >> 32), c2 = 0, c3 = 0;
    uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t lo0 = 0xD2511F53u * c0, hi0 = __umulhi(0xD2511F53u, c0);
        const uint32_t lo1 = 0xCD9E8D57u * c2, hi1 = __umulhi(0xCD9E8D57u, c2);
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return make_uint4(c0, c1, c2, c3);
}

__device__ __forceinline__ bool mask_bit(uint32_t word, float update_rate)
{
    return floorf((word >> 8) * (1.0f / 16777216.0f) + update_rate) != 0.f;
}

__device__ __forceinline__ uint64_t mask_counter(int n, int H, int W, int Y, int X)
{
    return ((uint64_t)n * H + Y) * ((W + 3) / 4) + X / 4;
}

// The masks of the R x 16 pixels (rows Y0 .. Y0 + R - 1, columns x0 .. x0 + 15, x0 % 4 == 0) of a warp, one Philox
// call per lane: lane 4 r + q draws the block of row r, columns x0 + 4 q .. x0 + 4 q + 3. Bit 4 r + q of word w of the
// result is the mask of row r, column x0 + 4 q + w (read with mask_at()). STOCH = false: all ones.
template <bool STOCH, int R>
__device__ __forceinline__ uint4 warp_masks(uint64_t seed, float update_rate, int n, int H, int W, int Y0, int x0)
{
    static_assert(R <= 8, "one Philox block per lane");
    if constexpr (!STOCH)
        return make_uint4(~0u, ~0u, ~0u, ~0u);
    const int lane = threadIdx.x % 32;
    const uint4 c = philox(seed, mask_counter(n, H, W, Y0 + lane / 4, x0 + 4 * (lane % 4)));
    return make_uint4(__ballot_sync(~0u, mask_bit(c.x, update_rate)), __ballot_sync(~0u, mask_bit(c.y, update_rate)),
                      __ballot_sync(~0u, mask_bit(c.z, update_rate)), __ballot_sync(~0u, mask_bit(c.w, update_rate)));
}

// mask of row r, column x0 + col of warp_masks()
__device__ __forceinline__ float mask_at(const uint4 &m, int r, int col)
{
    const int w = col % 4;
    const uint32_t word = w == 0 ? m.x : w == 1 ? m.y : w == 2 ? m.z : m.w;
    return (word >> (4 * r + col / 4)) & 1u;
}

__global__ void mask_kernel(float *__restrict__ mask, int N, int H, int W, uint64_t seed, float update_rate)
{
    const uint64_t total = (uint64_t)N * H * W;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < total; i += (uint64_t)gridDim.x * blockDim.x) {
        const int X = i % W, Y = (i / W) % H, n = i / ((uint64_t)H * W);
        const uint4 c = philox(seed, mask_counter(n, H, W, Y, X));
        mask[i] = mask_bit(X % 4 == 0 ? c.x : X % 4 == 1 ? c.y : X % 4 == 2 ? c.z : c.w, update_rate);
    }
}

// ---------------------------------------------------------------- state tile
// CH x (TH + 2) x (TILE_W + 2) tile of an image with its 1-pixel halo. The channel planes are padded to a stride
// that is 8 or 24 mod 32, so that lanes (g, t) reading channel 4q + t at column g hit distinct banks.
template <int CH, int TH>
struct Tile {
    static constexpr int PLANE = conflict_free_stride((TH + 2) * HALO_W);
    float v[CH][PLANE];
    // (row, col) in halo coordinates: the tile pixel (r, c) is at (r + 1, c + 1)
    __device__ __forceinline__ float &at(int c, int row, int col) { return v[c][row * HALO_W + col]; }
    __device__ __forceinline__ float at(int c, int row, int col) const { return v[c][row * HALO_W + col]; }

    // load the tile at (y0, x0) of an image (CH channel planes of H x W) with circular wrap
    __device__ void load(const float *__restrict__ x, int H, int W, int y0, int x0)
    {
        for (int i = threadIdx.x; i < CH * (TH + 2) * HALO_W; i += blockDim.x) {
            const int col = i % HALO_W, row = (i / HALO_W) % (TH + 2), c = i / (HALO_W * (TH + 2));
            int gy = y0 - 1 + row, gx = x0 - 1 + col;
            gy = gy < 0 ? gy + H : (gy >= H ? gy - H : gy);
            gx = gx < 0 ? gx + W : (gx >= W ? gx - W : gx);
            at(c, row, col) = x[((size_t)c * H + gy) * W + gx];
        }
    }

    // the four filters [identity, sobel_x, sobel_x^T, laplacian] of channel c at tile pixel (row, col)
    __device__ __forceinline__ void perceive(int c, int row, int col, float (&p)[4]) const
    {
        float v[3][3];
#pragma unroll
        for (int dy = 0; dy < 3; ++dy)
#pragma unroll
            for (int dx = 0; dx < 3; ++dx)
                v[dy][dx] = at(c, row + dy, col + dx);
        p[0] = v[1][1];
        p[1] = (v[0][2] - v[0][0]) + 2.f * (v[1][2] - v[1][0]) + (v[2][2] - v[2][0]);
        p[2] = (v[2][0] - v[0][0]) + 2.f * (v[2][1] - v[0][1]) + (v[2][2] - v[0][2]);
        p[3] = (v[0][0] + v[0][2] + v[2][0] + v[2][2]) + 2.f * (v[0][1] + v[1][0] + v[1][2] + v[2][1]) - 12.f * v[1][1];
    }

    // The A-operand elements of the p -> h MMA (K order of feature()) for the 16 pixels (row, 0..15) of a warp:
    // emit(s, e, v) for k-step s and fragment element e (pixel g + 8 (e % 2), column t + 4 (e / 2)).
    template <int KS1, typename F>
    __device__ __forceinline__ void perceive_all(int row, int g, int t, F &&emit) const
    {
#pragma unroll
        for (int q = 0; q < KS1 / 2; ++q) {
            float p0[4], p1[4];  // pixels g, g + 8
            perceive(4 * q + t, row, g, p0);
            perceive(4 * q + t, row, g + 8, p1);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                emit(2 * q + h, 0, p0[2 * h]); emit(2 * q + h, 1, p1[2 * h]);
                emit(2 * q + h, 2, p0[2 * h + 1]); emit(2 * q + h, 3, p1[2 * h + 1]);
            }
        }
    }
};

// ---------------------------------------------------------------- shared-memory layout
template <int C, int HID, bool FP32>
struct ForwardSmem {
    using S = Shape<C, HID>;
    FragB<FP32> w1[S::NT1][S::KS1][32];   // p -> h: b0 = W1[8j+g][feature(s, t)], b1 = W1[8j+g][feature(s, t+4)]
    FragB<FP32> w2[S::NT2][S::NT1][32];   // h -> u: b0 = W2[8n+g][8j+2t], b1 = W2[8n+g][8j+2t+1] (permuted hidden)
    float b1[HID];
    Tile<C, TILE_H> x;                    // state tile with circular halo
};

// Stage one NCA's weights in fragment-major order.
template <int C, int HID, bool FP32>
__device__ void stage_forward_weights(ForwardSmem<C, HID, FP32> &sm, const float *__restrict__ w1,
                                      const float *__restrict__ b1, const float *__restrict__ w2)
{
    using S = Shape<C, HID>;
    for (int i = threadIdx.x; i < S::NT1 * S::KS1 * 32; i += blockDim.x) {
        const int lane = i % 32, s = (i / 32) % S::KS1, j = i / (32 * S::KS1), g = lane / 4, t = lane % 4;
        const float *row = w1 + (8 * j + g) * S::PIN;
        make_frag<FP32>(sm.w1[j][s][lane], row[feature(s, t)], row[feature(s, t + 4)]);
    }
    for (int i = threadIdx.x; i < S::NT2 * S::NT1 * 32; i += blockDim.x) {
        const int lane = i % 32, j = (i / 32) % S::NT1, n = i / (32 * S::NT1), g = lane / 4, t = lane % 4;
        const int ch = 8 * n + g;
        const float v0 = ch < C ? w2[ch * HID + 8 * j + 2 * t] : 0.f;
        const float v1 = ch < C ? w2[ch * HID + 8 * j + 2 * t + 1] : 0.f;
        make_frag<FP32>(sm.w2[n][j][lane], v0, v1);
    }
    for (int i = threadIdx.x; i < HID; i += blockDim.x)
        sm.b1[i] = b1[i];
}

// ---------------------------------------------------------------- forward
// Growing NCA: does any cell within 2 pixels of the tile [y0, y0 + TH) x [x0, x0 + TILE_W) have alpha > 0.1 (no wrap)?
// If not, every pixel of the tile and of its 1-pixel ring is dead before the step (pre-alive mask 0), so the tile's
// output is 0 and no living cell depends on it: the step can skip it (alive_kernel writes its zeros). Block-uniform.
template <int TH>
__device__ bool near_alive(const float *__restrict__ alpha, int H, int W, int y0, int x0)
{
    bool any = false;
    for (int i = threadIdx.x; i < (TH + 4) * (TILE_W + 4); i += blockDim.x) {
        const int yy = y0 - 2 + i / (TILE_W + 4), xx = x0 - 2 + i % (TILE_W + 4);
        any |= yy >= 0 && yy < H && xx >= 0 && xx < W && alpha[yy * W + xx] > 0.1f;
    }
    return __syncthreads_or(any);
}

template <int C, int HID, bool FP32, bool STOCH, bool ALIVE>
__global__ void __launch_bounds__(FWD_THREADS) forward_kernel(
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

    bool weights_staged = false;
    for (int tile = blockIdx.x * TILES_PER_BLOCK; tile < min(tiles, (blockIdx.x + 1) * TILES_PER_BLOCK); ++tile) {
        const int y0 = (tile / tiles_x) * TILE_H, x0 = (tile % tiles_x) * TILE_W;
        if (ALIVE && !near_alive<TILE_H>(xn + 3 * (size_t)H * W, H, W, y0, x0))
            continue;
        if (!weights_staged) {  // once per block, unless all its tiles are dead
            stage_forward_weights<C, HID, FP32>(sm, w1 + (size_t)k * HID * S::PIN, b1 + (size_t)k * HID, w2 + (size_t)k * C * HID);
            weights_staged = true;
        }
        __syncthreads();  // previous tile's reads (and the weight staging) are complete
        sm.x.load(xn, H, W, y0, x0);
        __syncthreads();

        constexpr int RW = TILE_H / FWD_WARPS;  // rows of this warp
        const uint4 masks = warp_masks<STOCH, RW>(seed, update_rate, n, H, W, y0 + warp * RW, x0);
        for (int row = warp * RW; row < (warp + 1) * RW; ++row) {
            Op4<FP32> a[S::KS1];  // A operand of p -> h
            sm.x.template perceive_all<S::KS1>(row, g, t, [&](int s, int e, float v) { a[s].set(e, v); });
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
            float u2[2][S::NT2][4] = {};  // even / odd k-steps: twice the independent MMA chains
#pragma unroll
            for (int j = 0; j < S::NT1; ++j) {
                Op4<FP32> ah;
                ah.set(0, h[j][0]); ah.set(1, h[j][2]); ah.set(2, h[j][1]); ah.set(3, h[j][3]);
#pragma unroll
                for (int nt = 0; nt < S::NT2; ++nt)
                    mma<FP32>(u2[j % 2][nt], ah, sm.w2[nt][j][lane]);
            }
            float u[S::NT2][4];
#pragma unroll
            for (int nt = 0; nt < S::NT2; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    u[nt][e] = u2[0][nt][e] + u2[1][nt][e];
            // y = x + u * m for pixels g, g + 8 and channels 8nt + 2t, 8nt + 2t + 1
            const int gy = y0 + row;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int px = g + 8 * half, gx = x0 + px;
                if (gy >= H || gx >= W)
                    continue;
                const float m = mask_at(masks, row - warp * RW, px);
#pragma unroll
                for (int nt = 0; nt < S::NT2; ++nt)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int ch = 8 * nt + 2 * t + e;
                        if (ch >= C)
                            continue;
                        const float v = sm.x.at(ch, row + 1, px + 1) + u[nt][2 * half + e] * m;
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
// p, relu(h) and g*m in shared memory, then
//   dh = (g * m) W2 * [h > 0]          (MMA, registers); db1 += sum of dh over the pixels (warp shuffles)
//   dp = dh W1                          (MMA, permuted hidden index as in the forward) -> global (N, 4C, H, W)
//   dW2 += (g * m)^T relu(h),  dW1 += dh^T p   (MMA over the pixels of the tile, per-block accumulators)
// Each block writes its weight-gradient partials; reduce_kernel sums them per NCA in a fixed order (deterministic).
// adjoint_kernel then forms dx = g + sum_f K_f^T * dp_f (transpose of the circular 3x3 filters).
constexpr int BWD_WARPS = 4, BWD_THREADS = 32 * BWD_WARPS;
constexpr int BWD_TILE_H = BWD_WARPS;      // one 16-pixel row per warp
constexpr int BWD_PIX = BWD_TILE_H * TILE_W;
static_assert(BWD_PIX <= BWD_THREADS, "one thread per pixel checks the alive mask of a tile");
constexpr int BWD_TILES_PER_BLOCK = NCA_BWD_TILES_PER_BLOCK;  // tiles of one image per block (amortises accumulator write-out)
constexpr int BWD_MIN_BLOCKS = NCA_BWD_MIN_BLOCKS;            // resident blocks per SM (register bound via __launch_bounds__)
#ifndef NCA_ABLATE
#define NCA_ABLATE 0   // profiling only: 1 = skip weight gradients, 2 = skip dp stores, 3 = both
#endif

template <int C, int HID>
struct BwdShape {
    using S = Shape<C, HID>;
    static constexpr int PIN = S::PIN, KS1 = S::KS1, NT1 = S::NT1, NT2 = S::NT2;
    static constexpr int NTP = PIN / 8;                      // n-tiles of dh -> dp
    static constexpr int MTA = PIN / 16;                     // weight-gradient m-tiles of p^T (features)
    static constexpr int MT2 = (C + 15) / 16;                // weight-gradient m-tiles of gm^T (channels)
    static constexpr int CPAD = 16 * MT2;
    static constexpr int NIW = NT1 / BWD_WARPS;              // hidden n-tiles owned by each warp
    static_assert(NT1 % BWD_WARPS == 0, "hidden n-tiles are split evenly over the warps");
    // relu(h) and dh are staged for the weight gradients in HPARTS parts of the hidden units (less shared memory:
    // 16 / 128 then fits two blocks per SM on sm_89). With two parts, relu(h)'s second part goes to the p buffer
    // (p is staged later, for dW1), and each warp owns NIWP n-tiles of each part.
    static constexpr int HPARTS = (NIW % 2 == 0 && HID / 2 <= PIN) ? 2 : 1;
    static constexpr int HP = HID / HPARTS, NTH = NT1 / HPARTS, NIWP = NIW / HPARTS;
    // hidden n-tile of the warp's accumulator q
    __device__ static constexpr int owned(int warp, int q) { return (q / NIWP) * NTH + warp * NIWP + q % NIWP; }
    static constexpr int PARTIAL = HID * PIN + HID + C * HID;  // dW1, db1, dW2 per block
    static_assert(CPAD >= 8 * NT2, "channel padding");
};

// Per-NCA B fragments for the backward matmuls (packed by pack_kernel).
template <int C, int HID, bool FP32>
struct BackwardWeights {
    using BS = BwdShape<C, HID>;
    FragB<FP32> w1h[BS::NT1][BS::KS1][32];   // h = W1 p: b0 = W1[8j+g][feature(s, t)], b1 = W1[8j+g][feature(s, t+4)]
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
        make_frag<FP32>(pk.w1h[j][s][lane], w1[(8 * j + g) * BS::PIN + feature(s, t)], w1[(8 * j + g) * BS::PIN + feature(s, t + 4)]);
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

// ROWS x COLS floats (COLS % 16 == 0) without padding: the column is XOR-swizzled by the row, which makes the MMA
// fragment reads [8k + t][8n + g] (t: row, g: column) free of bank conflicts.
template <int ROWS, int COLS>
struct Staged {
    static_assert(COLS % 16 == 0, "swizzle granularity");
    float v[ROWS][COLS];
    __device__ __forceinline__ float &operator()(int row, int col)
    {
        return v[row][col ^ (COLS % 32 == 0 ? 8 * (row % 4) : 8 * (row / 2 % 2))];
    }
};

template <int C, int HID>
struct BackwardSmem {
    using BS = BwdShape<C, HID>;
    Tile<C, BWD_TILE_H> x;
    Staged<BWD_PIX, BS::PIN> p;          // p (K order of feature()); with two parts first relu(h) (second part)
    Staged<BWD_PIX, BS::HP> h;           // relu(h) (first part), later dh (one part at a time)
    Staged<BWD_PIX, BS::CPAD> gm;        // g * m (channels zero-padded)
    float db1[BWD_WARPS][HID];           // per-warp sums of dh over the block's pixels
};

template <bool FP32>
__device__ __forceinline__ FragB<FP32> frag_from_staged(float b0, float b1)
{
    return {make_uint2(__float_as_uint(b0), __float_as_uint(b1))};
}

template <int C, int HID, bool FP32, bool STOCH, bool ALIVE>
__global__ void __launch_bounds__(BWD_THREADS, BWD_MIN_BLOCKS) backward_kernel(
    const float *__restrict__ x, const float *__restrict__ gy, const uint8_t *__restrict__ alive,
    const BackwardWeights<C, HID, FP32> *__restrict__ packed, float *__restrict__ dp, float *__restrict__ partial,
    int n0, int B, int H, int W, uint64_t seed, float update_rate)
{
    using BS = BwdShape<C, HID>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto &sm = *reinterpret_cast<BackwardSmem<C, HID> *>(smem_raw);

    const int n = n0 + blockIdx.y, k = n / B;  // image; dp holds the images n0, n0 + 1, ... of this launch
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int tiles_x = (W + TILE_W - 1) / TILE_W, tiles = tiles_x * ((H + BWD_TILE_H - 1) / BWD_TILE_H);
    const size_t plane = (size_t)H * W;
    const float *xn = x + n * C * plane, *gyn = gy + n * C * plane;
    float *dpn = dp + blockIdx.y * BS::PIN * plane;
    const auto &wk = packed[k];

    float accp[BS::MTA][BS::NIW][4] = {};   // p^T dh        -> dW1^T, hidden n-tiles of this warp
    float accg[BS::MT2][BS::NIW][4] = {};   // gm^T relu(h)  -> dW2
    const int row = warp;
    for (int i = lane; i < HID; i += 32)
        sm.db1[warp][i] = 0.f;
    for (int tile = blockIdx.x * BWD_TILES_PER_BLOCK; tile < min(tiles, (blockIdx.x + 1) * BWD_TILES_PER_BLOCK); ++tile) {
        const int y0 = (tile / tiles_x) * BWD_TILE_H, x0 = (tile % tiles_x) * TILE_W, Y = y0 + row;
        if constexpr (ALIVE) {  // all cells of the tile dead after the step: no gradient through them, dp = 0
            const int X = x0 + threadIdx.x % TILE_W, Yt = y0 + threadIdx.x / TILE_W;
            const bool any = threadIdx.x < BWD_PIX && Yt < H && X < W && alive[((size_t)n * H + Yt) * W + X];
            if (!__syncthreads_or(any)) {
                for (int i = threadIdx.x; i < BS::PIN * BWD_PIX; i += blockDim.x) {
                    const int px = i % BWD_PIX, f = i / BWD_PIX, YY = y0 + px / TILE_W, XX = x0 + px % TILE_W;
                    if (YY < H && XX < W)
                        dpn[f * plane + (size_t)YY * W + XX] = 0.f;
                }
                continue;
            }
        }
        __syncthreads();  // previous tile's shared reads are complete
        sm.x.load(xn, H, W, y0, x0);
        __syncthreads();

        // ---- p (A fragments, and the shared copy in K order unless that buffer holds relu(h) first), h = W1 p + b1
        auto stage_p = [&](int s, int e, float v) { sm.p(16 * row + g + 8 * (e % 2), 8 * s + t + 4 * (e / 2)) = staged<FP32>(v); };
        // staged relu(h) or dh, hidden unit c of pixel px
        auto hbuf = [&](int px, int c) -> float & { return c < BS::HP ? sm.h(px, c) : sm.p(px, c - BS::HP); };
        Op4<FP32> a[BS::KS1];
        sm.x.template perceive_all<BS::KS1>(row, g, t, [&](int s, int e, float v) {
            a[s].set(e, v);
            if constexpr (BS::HPARTS == 1)
                stage_p(s, e, v);
        });
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
                hbuf(16 * row + g + 8 * (e / 2), 8 * j + 2 * t + e % 2) = staged<FP32>(fmaxf(h[e], 0.f));
            }
        }

        // ---- g * m (* a) (A fragments of dh = gm W2, shared copy for dW2); zero outside the image
        float mrow[2];
        const uint4 masks = warp_masks<STOCH, 1>(seed, update_rate, n, H, W, Y, x0);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int X = x0 + g + 8 * half;
            const uint64_t pix = ((uint64_t)n * H + Y) * W + X;
            mrow[half] = (Y < H && X < W) ? mask_at(masks, 0, g + 8 * half) * (ALIVE ? alive[pix] : 1) : 0.f;
        }
        Op4<FP32> ag[BS::NT2];
#pragma unroll
        for (int q = 0; q < BS::NT2; ++q)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int px = g + 8 * (e % 2), ch = 8 * q + t + 4 * (e / 2), X = x0 + px;
                const float v = (ch < C && mrow[e % 2] != 0.f) ? gyn[ch * plane + (size_t)Y * W + X] * mrow[e % 2] : 0.f;
                ag[q].set(e, v);
                sm.gm(16 * row + px, ch) = staged<FP32>(v);
            }
        for (int i = lane; i < 16 * (BS::CPAD - 8 * BS::NT2); i += 32)
            sm.gm(16 * row + i % 16, 8 * BS::NT2 + i / 16) = 0.f;

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
                ag2[mi].set_staged(0, sm.gm(8 * kk + t, 16 * mi + g)); ag2[mi].set_staged(1, sm.gm(8 * kk + t, 16 * mi + g + 8));
                ag2[mi].set_staged(2, sm.gm(8 * kk + t + 4, 16 * mi + g)); ag2[mi].set_staged(3, sm.gm(8 * kk + t + 4, 16 * mi + g + 8));
            }
#pragma unroll
            for (int q = 0; q < BS::NIW; ++q) {
                const int c = 8 * BS::owned(warp, q) + g;
                const auto b = frag_from_staged<FP32>(hbuf(8 * kk + t, c), hbuf(8 * kk + t + 4, c));
#pragma unroll
                for (int mi = 0; mi < BS::MT2; ++mi)
                    mma<FP32>(accg[mi][q], ag2[mi], b);
            }
        }
        // ---- db1 += dh summed over the warp's 16 pixels (fixed order: deterministic); dW1^T += p^T dh
#pragma unroll
        for (int j = 0; j < BS::NT1; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = dh[j][e] + dh[j][e + 2];
                v += __shfl_xor_sync(~0u, v, 4);
                v += __shfl_xor_sync(~0u, v, 8);
                v += __shfl_xor_sync(~0u, v, 16);
                if (g == 0)
                    sm.db1[warp][8 * j + 2 * t + e] += v;
            }
#pragma unroll
        for (int part = 0; part < BS::HPARTS; ++part) {  // dh one part at a time in sm.h
            __syncthreads();
            if (BS::HPARTS == 2 && part == 0)  // p again (cheap), into the buffer that held relu(h)
                sm.x.template perceive_all<BS::KS1>(row, g, t, stage_p);
#pragma unroll
            for (int j = part * BS::NTH; j < (part + 1) * BS::NTH; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    sm.h(16 * row + g + 8 * (e / 2), 8 * (j - part * BS::NTH) + 2 * t + e % 2) = staged<FP32>(dh[j][e]);
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < BWD_PIX / 8; ++kk) {
                Op4<FP32> ap[BS::MTA];
#pragma unroll
                for (int mi = 0; mi < BS::MTA; ++mi)
#pragma unroll
                    for (int e = 0; e < 4; ++e)  // feature rows (K order) 16 mi + g (+ 8)
                        ap[mi].set_staged(e, sm.p(8 * kk + t + 4 * (e / 2), 16 * mi + 8 * (e % 2) + g));
#pragma unroll
                for (int q = part * BS::NIWP; q < (part + 1) * BS::NIWP; ++q) {
                    const int c = 8 * BS::owned(warp, q) - part * BS::HP + g;
                    const auto b = frag_from_staged<FP32>(sm.h(8 * kk + t, c), sm.h(8 * kk + t + 4, c));
#pragma unroll
                    for (int mi = 0; mi < BS::MTA; ++mi)
                        mma<FP32>(accp[mi][q], ap[mi], b);
                }
            }
        }
    }

    // ---- write this block's partial weight gradients: dW1 (HID x PIN), db1 (HID), dW2 (C x HID)
    float *out = partial + ((size_t)n * gridDim.x + blockIdx.x) * BS::PARTIAL;
    __syncthreads();
    for (int i = threadIdx.x; i < HID; i += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < BWD_WARPS; ++w)
            v += sm.db1[w][i];
        out[HID * BS::PIN + i] = v;
    }
#pragma unroll
    for (int q = 0; q < BS::NIW; ++q)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int hid = 8 * BS::owned(warp, q) + 2 * t + e % 2;
#pragma unroll
            for (int mi = 0; mi < BS::MTA; ++mi) {
                const int f = 16 * mi + g + 8 * (e / 2);  // K position of the feature
                out[hid * BS::PIN + feature(f / 8, f % 8)] = accp[mi][q][e];
            }
#pragma unroll
            for (int mi = 0; mi < BS::MT2; ++mi) {
                const int ch = 16 * mi + g + 8 * (e / 2);
                if (ch < C)
                    out[HID * (BS::PIN + 1) + ch * HID + hid] = accg[mi][q][e];  // after dW1 and db1
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
    if (e < HID * BS::PIN)
        dw1[(size_t)k * HID * BS::PIN + e] = s;
    else if (e < HID * (BS::PIN + 1))
        db1[(size_t)k * HID + e - HID * BS::PIN] = s;
    else
        dw2[(size_t)k * C * HID + e - HID * (BS::PIN + 1)] = s;
}

// dx = g (* a) + sum_f sum_o K_f(o) dp_{4c+f}(y - o), the transpose of the circular perception filters. One thread
// per (4 neighbouring pixels of a row, channel): for each tap row of a dp plane it loads the 4 pixels (one float4 if
// W % 4 == 0) and their 2 neighbours, a third of the loads of one thread per pixel. Taps are served by L1.
template <int C, bool ALIVE>
__global__ void __launch_bounds__(256) adjoint_kernel(const float *__restrict__ dp, const float *__restrict__ gy,
                                                      const uint8_t *__restrict__ alive, float *__restrict__ gx,
                                                      int H, int W, uint64_t total)  // total = images * C * H * W4
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int W4 = (W + 3) / 4, X0 = 4 * (i % W4), Y = (i / W4) % H;
    const uint64_t nc = i / ((uint64_t)H * W4), n = nc / C, c = nc % C, plane = (uint64_t)H * W;
    const float *d = dp + (n * 4 * C + 4 * c) * plane;
    const bool vec = W % 4 == 0;
    // pixels X0 - 1 .. X0 + 4 of row yy of plane P (circular)
    auto load_row = [&](const float *P, int yy, float (&v)[6]) {
        P += (uint64_t)yy * W;
        if (vec) {
            const float4 m = __ldg(reinterpret_cast<const float4 *>(P + X0));
            v[1] = m.x; v[2] = m.y; v[3] = m.z; v[4] = m.w;
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k)
                v[k + 1] = __ldg(P + (X0 + k) % W);
        }
        v[0] = __ldg(P + (X0 == 0 ? W - 1 : X0 - 1));
        v[5] = __ldg(P + (X0 + 4) % W);
    };
    float out[4] = {};
#pragma unroll
    for (int dy = 0; dy < 3; ++dy) {
        int yy = Y + 1 - dy;
        yy = yy < 0 ? yy + H : (yy >= H ? yy - H : yy);
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            if (filter_tap(f, 3 * dy) == 0.f && filter_tap(f, 3 * dy + 1) == 0.f && filter_tap(f, 3 * dy + 2) == 0.f)
                continue;  // no taps in this row of the filter
            float v[6];
            load_row(d + f * plane, yy, v);
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int dx = 0; dx < 3; ++dx)  // tap (dy, dx) reads pixel X0 + k + 1 - dx = v[k + 2 - dx]
                    if (filter_tap(f, 3 * dy + dx) != 0.f)
                        out[k] = fmaf(filter_tap(f, 3 * dy + dx), v[k + 2 - dx], out[k]);
        }
    }
    const uint64_t o = nc * plane + (uint64_t)Y * W + X0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        if (X0 + k < W) {
            const float g = ALIVE ? (alive[n * plane + (uint64_t)Y * W + X0 + k] ? gy[o + k] : 0.f) : gy[o + k];
            gx[o + k] = g + out[k];
        }
}

}  // namespace nca_step
