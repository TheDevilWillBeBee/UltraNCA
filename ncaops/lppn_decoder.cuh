#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Fused LPPN decoder (lppn.py) for N images, each with its own SIREN. Per pixel (rho, kappa) of dual cell (i, j):
//   z_0 = sum_c w_c(rho, kappa) G[corner c] + T[rho, kappa],  h_0 = sin(z_0)
//   h_l = sin(V_l h_{l-1} + c_l)  (l = 1 .. LAYERS),  out = W_o h_L + b_o
// G (N, h, w, HID): first-layer pre-activation per cell (corners (i, j), (i, j+1), (i+1, j), (i+1, j+1), mod h, w);
// T (N, S, S, HID): the coordinate term per dual-cell position; V (N, LAYERS, HID, HID), c (N, LAYERS, HID): the
// hidden layers with omega folded in; W_o (N, O, HID), b_o (N, O). Pixel (rho, kappa) of dual cell (i, j) is image
// pixel ((S i + S/2 + rho) mod Sh, (S j + S/2 + kappa) mod Sw) of out (N, O, Sh, Sw).
// Matmuls run on tensor cores (mma.sync m16n8k16, fp16 operands, fp32 accumulation; sines in fp32). One warp
// computes one M-tile: 16 pixels of a dual cell (m = 16 q + row, rho = m / S, kappa = m % S). The accumulator columns
// (2t, 2t+1) of n-tile j are the A-operand columns of the next layer, so activations stay in registers. Weights are
// staged once per block as B fragments in fragment-major order (one 8-byte shared load per lane and MMA).
// Backward: the forward is recomputed. dout is scaled per image by a power of two (gscale: max |dout| in [1, 2)) so
// that the fp16 operands of the backward matmuls neither underflow nor overflow; every gradient is unscaled in fp32.
// A block sweeps a band of BAND dual-cell rows across the image (all columns), in rounds of 64 pixels (4 warps x
// 16): S = 8, one dual cell per round (warp = M-tile q); S = 4, one dual-cell column of the band (warp = dual row).
// Per round, h_0, h_1, dz_0, dz_1, dz_2 are staged (fp16) for the weight gradients dV_l += dz_l^T h_{l-1} (K = 64
// pixels, ldmatrix.trans) and the corner gradients dG = W_c^T dz_0 (the bilinear weights as a constant A operand),
// which accumulate into a window of cell columns; finished columns are written out, the band's first and last cell
// rows (shared with the neighbouring bands) as edges that edge_kernel adds. The other gradients are per-block
// partials that reduce_kernel sums per image in a fixed order: every reduction is deterministic.
// ENS (local ensemble, LIIF-style; Cells2Pixels batched/decoder.LocalEnsembleSiren): the MLP runs once per corner c,
// z_0 = G[corner c] + T_c[rho, kappa] with a coordinate table per corner (T (N, 4, S, S, HID): the pixel's position
// relative to that cell), and out = sum_c w_c(rho, kappa) out_c. The forward blends the 4 evaluations of an M-tile in
// registers; the backward sweeps the band once per corner (dout weighted by w_c) and writes each dual cell's
// corner-gradient sum (N, 4, h, w, HID), which ens_gather_kernel adds per cell in a fixed order.
// offset (N, 2) (optional; training-time jitter): image n samples the field at positions shifted by (offset[n][0],
// offset[n][1]) cells, |offset| <= 1 / (2 S) so that every pixel stays in its dual cell: the bilinear weights use
// lambda + offset (the coordinate tables, computed per image, carry the shift of the coordinates).
// HID (hidden width, a template parameter): a multiple of 8 up to 32 (instantiated: 32, 24): NT = HID / 8 n-tiles,
// KS = ceil(HID / 16) k-steps; a partial last k-step reads zero weights and zero activations.
namespace lppn {

constexpr int LAYERS = 2;              // sine layers after the first
constexpr int WARPS = 4, THREADS = 32 * WARPS;
constexpr int BAND = 4;                // dual-cell rows per block (h % BAND == 0)
constexpr int ROUND = 16 * WARPS;      // pixels per backward round
constexpr int LD = 40;                 // fp16 row stride of the staged tiles (32 columns): 80 bytes, conflict-free ldmatrix rows

template <int HID>
struct Width {
    static_assert(HID % 8 == 0 && HID >= 8 && HID <= 32, "hidden width: a multiple of 8, at most 32");
    static constexpr int NT = HID / 8, KS = (HID + 15) / 16;
};

template <int S, int O, bool ENS, int HID>
struct Layout {  // per-block partial of the backward and per-image result of reduce_kernel (floats)
    static constexpr int V = 0, C = LAYERS * HID * HID, WO = C + LAYERS * HID, BO = WO + O * HID, T = BO + 8,
                         SIZE = T + (ENS ? 4 : 1) * S * S * HID;
};

__device__ __forceinline__ uint32_t half2_bits(float lo, float hi)
{
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const __half *p)
{
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm_x2_trans(uint32_t (&r)[2], const __half *p)
{
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

template <int O, int HID>
struct Weights {
    using W = Width<HID>;
    uint2 f[LAYERS][W::NT][W::KS][32];  // forward B fragments (B[k][n] = V[n][k]): layer, n-tile, k-step, lane
    uint2 b[LAYERS][W::NT][W::KS][32];  // backward B fragments (B[k][n] = V[k][n])
    uint2 o[W::KS][32];                 // output layer (B[k][n] = W_o[n][k], n < O)
    float c[LAYERS][HID], wo[O][HID], bo[8];
};

template <int O, bool BWD, int HID>
__device__ void load_weights(Weights<O, HID> &s, const float *__restrict__ V, const float *__restrict__ c,
                             const float *__restrict__ wo, const float *__restrict__ bo, int n)
{
    using W = Width<HID>;
    const float *Vn = V + (size_t)n * LAYERS * HID * HID;
    for (int x = threadIdx.x; x < LAYERS * W::NT * W::KS * 32; x += THREADS)
    {
        const int lane = x & 31, sk = (x >> 5) % W::KS, jt = (x >> 5) / W::KS % W::NT, l = (x >> 5) / (W::KS * W::NT);
        const int g = lane >> 2, t = lane & 3;
        const float *M = Vn + l * HID * HID;
        auto at = [M](int row, int col) { return row < HID && col < HID ? M[row * HID + col] : 0.f; };  // k padding: 0
        const int nn = 8 * jt + g, k = 16 * sk + 2 * t;
        s.f[l][jt][sk][lane] = make_uint2(half2_bits(at(nn, k), at(nn, k + 1)), half2_bits(at(nn, k + 8), at(nn, k + 9)));
        if (BWD)
            s.b[l][jt][sk][lane] = make_uint2(half2_bits(at(k, nn), at(k + 1, nn)), half2_bits(at(k + 8, nn), at(k + 9, nn)));
    }
    for (int x = threadIdx.x; x < W::KS * 32; x += THREADS)
    {
        const int lane = x & 31, sk = x >> 5, g = lane >> 2, k = 16 * sk + 2 * (lane & 3);
        uint2 v = make_uint2(0, 0);
        if (g < O)
        {
            const float *r = wo + ((size_t)n * O + g) * HID;
            auto at = [r](int col) { return col < HID ? r[col] : 0.f; };
            v = make_uint2(half2_bits(at(k), at(k + 1)), half2_bits(at(k + 8), at(k + 9)));
        }
        s.o[sk][lane] = v;
    }
    for (int x = threadIdx.x; x < LAYERS * HID; x += THREADS)
        s.c[x / HID][x % HID] = c[(size_t)n * LAYERS * HID + x];
    for (int x = threadIdx.x; x < O * HID; x += THREADS)
        s.wo[x / HID][x % HID] = wo[(size_t)n * O * HID + x];
    if (threadIdx.x < 8)
        s.bo[threadIdx.x] = bo && threadIdx.x < O ? bo[(size_t)n * O + threadIdx.x] : 0.f;
}

// The lane's two pixels (rows g and g + 8 of M-tile q): dual-cell position and bilinear corner weights (sample
// positions shifted by (oy, ox) cells).
template <int S>
struct Pixels {
    int rho[2], kap[2];
    float wc[2][4];
    __device__ Pixels(int q, int g, float oy, float ox)
    {
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            const int m = 16 * q + g + 8 * r;
            rho[r] = m / S, kap[r] = m % S;
            const float ly = (rho[r] + 0.5f) / S + oy, lx = (kap[r] + 0.5f) / S + ox;
            wc[r][0] = (1.f - ly) * (1.f - lx), wc[r][1] = (1.f - ly) * lx, wc[r][2] = ly * (1.f - lx), wc[r][3] = ly * lx;
        }
    }
};

// Activations of 16 pixels x HID channels in the accumulator layout: v[j][e] is (row g, channel 8j + 2t) for e = 0,
// (g, 8j + 2t + 1) for e = 1, rows g + 8 for e = 2, 3.
template <int HID>
using Acts = float[Width<HID>::NT][4];
template <int HID>
using Frag = uint32_t[Width<HID>::KS][4];  // the same as an A operand: k-step, register (zero beyond HID)

template <int HID>
__device__ __forceinline__ void to_frag(Frag<HID> &a, const Acts<HID> &v)
{
    constexpr int NT = Width<HID>::NT;
#pragma unroll
    for (int s = 0; s < Width<HID>::KS; ++s)
    {
        const int hi = 2 * s + 1 < NT ? 2 * s + 1 : 2 * s;  // n-tile of the upper 8 columns (absent: zeros)
        a[s][0] = half2_bits(v[2 * s][0], v[2 * s][1]);
        a[s][1] = half2_bits(v[2 * s][2], v[2 * s][3]);
        a[s][2] = 2 * s + 1 < NT ? half2_bits(v[hi][0], v[hi][1]) : 0u;
        a[s][3] = 2 * s + 1 < NT ? half2_bits(v[hi][2], v[hi][3]) : 0u;
    }
}

// the table entries of the lane's pixels; tables = 1 (T (N, S, S, HID)) or 4 (T (N, 4, S, S, HID): corner `table`)
template <int S, int HID>
__device__ __forceinline__ void load_table(float (&tv)[2][Width<HID>::NT][2], const float *__restrict__ T, int n,
                                           const Pixels<S> &px, int t, int tables = 1, int table = 0)
{
    const float *Tn = T + ((size_t)n * tables + table) * S * S * HID;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int j = 0; j < Width<HID>::NT; ++j)
        {
            const float2 v = __ldg(reinterpret_cast<const float2 *>(Tn + (px.rho[r] * S + px.kap[r]) * HID + 8 * j + 2 * t));
            tv[r][j][0] = v.x, tv[r][j][1] = v.y;
        }
}

// z_0 of the lane's pixels for one corner of a dual cell (ensemble): cell (i, j) + its table entries
template <int HID>
__device__ __forceinline__ void first_layer_corner(Acts<HID> &z, const float *__restrict__ Gn, int w, int i, int j,
                                                   const float (&tv)[2][Width<HID>::NT][2], int t)
{
    const float2 *cell = reinterpret_cast<const float2 *>(Gn + ((size_t)i * w + j) * HID) + t;
#pragma unroll
    for (int jt = 0; jt < Width<HID>::NT; ++jt)
    {
        const float2 v = __ldg(cell + 4 * jt);
#pragma unroll
        for (int r = 0; r < 2; ++r)
            z[jt][2 * r] = tv[r][jt][0] + v.x, z[jt][2 * r + 1] = tv[r][jt][1] + v.y;
    }
}

// z_0 of the lane's pixels in dual cell (i, j).
template <int S, int HID>
__device__ __forceinline__ void first_layer(Acts<HID> &z, const float *__restrict__ Gn, int h, int w, int i, int j,
                                            const Pixels<S> &px, const float (&tv)[2][Width<HID>::NT][2], int t)
{
    const int i1 = i + 1 == h ? 0 : i + 1, j1 = j + 1 == w ? 0 : j + 1;
    const float2 *corner[4] = {reinterpret_cast<const float2 *>(Gn + ((size_t)i * w + j) * HID) + t,
                               reinterpret_cast<const float2 *>(Gn + ((size_t)i * w + j1) * HID) + t,
                               reinterpret_cast<const float2 *>(Gn + ((size_t)i1 * w + j) * HID) + t,
                               reinterpret_cast<const float2 *>(Gn + ((size_t)i1 * w + j1) * HID) + t};
#pragma unroll
    for (int jt = 0; jt < Width<HID>::NT; ++jt)
    {
        float2 v[4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
            v[c] = __ldg(corner[c] + 4 * jt);
#pragma unroll
        for (int r = 0; r < 2; ++r)
        {
            z[jt][2 * r] = tv[r][jt][0] + px.wc[r][0] * v[0].x + px.wc[r][1] * v[1].x + px.wc[r][2] * v[2].x + px.wc[r][3] * v[3].x;
            z[jt][2 * r + 1] = tv[r][jt][1] + px.wc[r][0] * v[0].y + px.wc[r][1] * v[1].y + px.wc[r][2] * v[2].y + px.wc[r][3] * v[3].y;
        }
    }
}

// acc = a (16 x HID) @ B + bias, B given as fragments [n-tile][k-step][lane]
template <int HID>
__device__ __forceinline__ void dense(Acts<HID> &acc, const Frag<HID> &a, const uint2 (*frag)[Width<HID>::KS][32],
                                      const float *bias, int t, int lane)
{
    using W = Width<HID>;
#pragma unroll
    for (int jt = 0; jt < W::NT; ++jt)
    {
        const float b0 = bias ? bias[8 * jt + 2 * t] : 0.f, b1 = bias ? bias[8 * jt + 2 * t + 1] : 0.f;
        acc[jt][0] = acc[jt][2] = b0, acc[jt][1] = acc[jt][3] = b1;
    }
#pragma unroll
    for (int s = 0; s < W::KS; ++s)
#pragma unroll
        for (int jt = 0; jt < W::NT; ++jt)
        {
            const uint2 b = frag[jt][s][lane];
            mma(acc[jt], a[s], b.x, b.y);
        }
}

template <int HID>
__device__ __forceinline__ void sine(Acts<HID> &v)
{
#pragma unroll
    for (int jt = 0; jt < Width<HID>::NT; ++jt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            v[jt][e] = __sinf(v[jt][e]);
}

template <int HID>
__device__ __forceinline__ void sine_cos(Acts<HID> &v, Acts<HID> &cs)
{
#pragma unroll
    for (int jt = 0; jt < Width<HID>::NT; ++jt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            __sincosf(v[jt][e], &v[jt][e], &cs[jt][e]);
}

// image pixel of row r of the lane in dual cell (i, j)
template <int S>
__device__ __forceinline__ size_t pixel_offset(const Pixels<S> &px, int r, int i, int j, int h, int w)
{
    int P = S * i + S / 2 + px.rho[r], Q = S * j + S / 2 + px.kap[r];
    P = P >= S * h ? P - S * h : P, Q = Q >= S * w ? Q - S * w : Q;
    return (size_t)P * S * w + Q;
}

// ---------------------------------------------------------------- forward
// grid (bands * segments, N): a block covers BAND dual-cell rows and a segment of the columns; no reductions.
template <int S, int O, bool ENS, int HID>
__global__ void __launch_bounds__(THREADS) forward_kernel(const float *__restrict__ G, const float *__restrict__ T,
                                                          const float *__restrict__ V, const float *__restrict__ c,
                                                          const float *__restrict__ wo, const float *__restrict__ bo,
                                                          const float *__restrict__ offset, float *__restrict__ out,
                                                          int h, int w, int segments)
{
    static_assert(S == 4 || S == 8, "dual cells of one M-tile (S = 4) or of one round (S = 8)");
    using W = Width<HID>;
    __shared__ Weights<O, HID> sw;
    const int n = blockIdx.y, band = blockIdx.x / segments, seg = blockIdx.x % segments;
    load_weights<O, false, HID>(sw, V, c, wo, bo, n);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    constexpr int Q = S * S / 16, CELLS = WARPS / Q;  // M-tiles per dual cell, dual cells in flight per block
    const Pixels<S> px(warp % Q, g, offset ? offset[2 * n] : 0.f, offset ? offset[2 * n + 1] : 0.f);
    float tv[2][W::NT][2];
    if (!ENS)
        load_table<S, HID>(tv, T, n, px, t);
    const float *Gn = G + (size_t)n * h * w * HID;
    const size_t plane = (size_t)S * h * S * w;
    float *outn = out + (size_t)n * O * plane;
    for (int j = seg * w / segments; j < (seg + 1) * w / segments; ++j)
        for (int k = warp / Q; k < BAND; k += CELLS)
        {
            const int i = band * BAND + k;
            float yb[4] = {0.f, 0.f, 0.f, 0.f};  // ensemble: the blend of the corners' outputs
#pragma unroll 1
            for (int cc = 0; cc < (ENS ? 4 : 1); ++cc)
            {
                Acts<HID> z;
                Frag<HID> a;
                if (ENS)
                {
                    load_table<S, HID>(tv, T, n, px, t, 4, cc);
                    const int ic = i + (cc >> 1), jc = j + (cc & 1);
                    first_layer_corner<HID>(z, Gn, w, ic == h ? 0 : ic, jc == w ? 0 : jc, tv, t);
                }
                else
                    first_layer<S, HID>(z, Gn, h, w, i, j, px, tv, t);
                sine<HID>(z);
                to_frag<HID>(a, z);
#pragma unroll
                for (int l = 0; l < LAYERS; ++l)
                {
                    dense<HID>(z, a, sw.f[l], sw.c[l], t, lane);
                    sine<HID>(z);
                    to_frag<HID>(a, z);
                }
                float y[4] = {sw.bo[2 * t], sw.bo[2 * t + 1], sw.bo[2 * t], sw.bo[2 * t + 1]};
#pragma unroll
                for (int s = 0; s < W::KS; ++s)
                    mma(y, a[s], sw.o[s][lane].x, sw.o[s][lane].y);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    yb[e] = ENS ? yb[e] + px.wc[e >> 1][cc] * y[e] : y[e];
            }
            const float *y = yb;
#pragma unroll
            for (int r = 0; r < 2; ++r)
            {
                const size_t p = pixel_offset<S>(px, r, i, j, h, w);
                if (2 * t < O)
                    outn[2 * t * plane + p] = y[2 * r];
                if (2 * t + 1 < O)
                    outn[(2 * t + 1) * plane + p] = y[2 * r + 1];
            }
        }
}

// ---------------------------------------------------------------- backward
template <int O, int HID>
struct BackwardSmem {
    Weights<O, HID> wt;
    uint4 wb[4][32];                   // A fragments of the bilinear corner weights: k-step, lane
    __half stage[5][ROUND][LD];        // h_0, h_1, dz_1, dz_2, dz_0 of the round (fp16)
    float dgc[16][HID];                // corner gradients of the round
    float win[3][BAND + 1][HID];       // cell gradients: column 0 (finished last), then a ring of two columns
};
enum { H0 = 0, H1 = 1, DZ1 = 2, DZ2 = 3, DZ0 = 4 };

template <int HID>
__device__ __forceinline__ void stage_frag(__half (*buf)[LD], const Frag<HID> &a, int row0, int g, int t)
{
#pragma unroll
    for (int s = 0; s < Width<HID>::KS; ++s)
    {
        *reinterpret_cast<uint32_t *>(&buf[row0 + g][16 * s + 2 * t]) = a[s][0];
        *reinterpret_cast<uint32_t *>(&buf[row0 + g + 8][16 * s + 2 * t]) = a[s][1];
        *reinterpret_cast<uint32_t *>(&buf[row0 + g][16 * s + 8 + 2 * t]) = a[s][2];
        *reinterpret_cast<uint32_t *>(&buf[row0 + g + 8][16 * s + 8 + 2 * t]) = a[s][3];
    }
}

// grid (h / BAND, N); partial (N, bands, Layout::SIZE) and edges are workspaces. LPPN: edges (N, bands, 2, w, HID);
// dG (N, h, w, HID) receives every cell row except the bands' first rows, which edge_kernel writes. ENS: edges
// (N, 4, h, w, HID) receives the corner-gradient sums per dual cell, ens_gather_kernel writes dG.
template <int S, int O, bool ENS, int HID>
__global__ void __launch_bounds__(THREADS, 2) backward_kernel(
    const float *__restrict__ G, const float *__restrict__ T, const float *__restrict__ V, const float *__restrict__ c,
    const float *__restrict__ wo, const float *__restrict__ dout, const float *__restrict__ gscale,
    const float *__restrict__ offset, float *__restrict__ dG, float *__restrict__ partial, float *__restrict__ edges,
    int h, int w)
{
    static_assert(S == 4 || S == 8, "dual cells of one M-tile (S = 4) or of one round (S = 8)");
    using L = Layout<S, O, ENS, HID>;
    using W = Width<HID>;
    __shared__ BackwardSmem<O, HID> sm;
    const int n = blockIdx.y, band = blockIdx.x, bands = gridDim.x, i0 = band * BAND;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    load_weights<O, true, HID>(sm.wt, V, c, wo, nullptr, n);
    const float oy = offset ? offset[2 * n] : 0.f, ox = offset ? offset[2 * n + 1] : 0.f;
    // corner weights as an A operand: row = corner (S = 8) or 4 * dual row + corner (S = 4), column = round pixel.
    // ENS: the sum over a dual cell's pixels (dout carries the corner weight): row = dual cell of the round
    for (int x = threadIdx.x; x < 4 * 32; x += THREADS)
    {
        const int ks = x >> 5, ln = x & 31, gg = ln >> 2, k = 16 * ks + 2 * (ln & 3);
        auto wcorner = [oy, ox](int row, int rp) {
            if (ENS)
                return (S == 8 ? row == 0 : (rp >> 4) == row) ? 1.f : 0.f;
            const int corner = S == 8 ? row : row & 3, m = S == 8 ? rp : rp & 15;
            if (S == 8 ? row >= 4 : (rp >> 4) != (row >> 2))
                return 0.f;
            const float ly = (m / S + 0.5f) / S + oy, lx = (m % S + 0.5f) / S + ox;
            return (corner & 2 ? ly : 1.f - ly) * (corner & 1 ? lx : 1.f - lx);
        };
        sm.wb[ks][ln] = make_uint4(half2_bits(wcorner(gg, k), wcorner(gg, k + 1)), half2_bits(wcorner(gg + 8, k), wcorner(gg + 8, k + 1)),
                                   half2_bits(wcorner(gg, k + 8), wcorner(gg, k + 9)), half2_bits(wcorner(gg + 8, k + 8), wcorner(gg + 8, k + 9)));
    }
    for (int x = threadIdx.x; x < 3 * (BAND + 1) * HID; x += THREADS)
        (&sm.win[0][0][0])[x] = 0.f;
    __syncthreads();

    constexpr int Q = S * S / 16;
    const Pixels<S> px(warp % Q, g, oy, ox);
    float tv[2][W::NT][2];
    const float gs = gscale[n], inv = 1.f / gs;
    const float *Gn = G + (size_t)n * h * w * HID;
    const size_t plane = (size_t)S * h * S * w;
    const float *doutn = dout + (size_t)n * O * plane;
    auto slot = [w](int col) { return col == 0 || col == w ? 0 : 1 + (col & 1); };

    float av[W::NT][4] = {};                                 // phase B: this warp's dV tiles
    float dwo[O][W::NT][2] = {}, dbo[O] = {}, dc[LAYERS][W::NT][2] = {};
    constexpr int ROUNDS = S == 8 ? BAND : 1;                // rounds per dual-cell column of the band
    constexpr int R = LAYERS * HID + O * HID + O;
    float *red = reinterpret_cast<float *>(&sm.stage[0][0][0]);  // after the sweeps: WARPS x R, then WARPS x 16 x HID
    float *pb = partial + ((size_t)n * bands + band) * L::SIZE;  // this block's partial
#pragma unroll 1
    for (int cc = 0; cc < (ENS ? 4 : 1); ++cc)  // ensemble: one sweep per corner
    {
        float dt[2][W::NT][2] = {};
        load_table<S, HID>(tv, T, n, px, t, ENS ? 4 : 1, cc);
        for (int j = 0; j < w; ++j)
            for (int r = 0; r < ROUNDS; ++r)
            {
                const int i = i0 + (S == 8 ? r : warp);
                // ---- phase A: this warp's M-tile, forward recomputed, backward to dz_0
                Acts<HID> z, cs0, cs1, cs2, acc;
                Frag<HID> a;
                if (ENS)
                {
                    const int ic = i + (cc >> 1), jc = j + (cc & 1);
                    first_layer_corner<HID>(z, Gn, w, ic == h ? 0 : ic, jc == w ? 0 : jc, tv, t);
                }
                else
                    first_layer<S, HID>(z, Gn, h, w, i, j, px, tv, t);
                sine_cos<HID>(z, cs0);
                to_frag<HID>(a, z);
                stage_frag<HID>(sm.stage[H0], a, 16 * warp, g, t);
                dense<HID>(acc, a, sm.wt.f[0], sm.wt.c[0], t, lane);
                sine_cos<HID>(acc, cs1);
                to_frag<HID>(a, acc);
                stage_frag<HID>(sm.stage[H1], a, 16 * warp, g, t);
                dense<HID>(acc, a, sm.wt.f[1], sm.wt.c[1], t, lane);
                sine_cos<HID>(acc, cs2);  // acc = h_2
                float d[2][O];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                {
                    const size_t p = pixel_offset<S>(px, rr, i, j, h, w);
#pragma unroll
                    for (int o = 0; o < O; ++o)
                        d[rr][o] = __ldg(doutn + o * plane + p) * (ENS ? gs * px.wc[rr][cc] : gs);
                }
                // output layer: dW_o += dout h_2^T, db_o += dout; dh_2 = W_o^T dout; dz_2 = dh_2 cos(z_2)
#pragma unroll
                for (int jt = 0; jt < W::NT; ++jt)
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        float dh0 = 0.f, dh1 = 0.f;
#pragma unroll
                        for (int o = 0; o < O; ++o)
                        {
                            dwo[o][jt][e] += d[0][o] * acc[jt][e] + d[1][o] * acc[jt][2 + e];
                            const float wv = sm.wt.wo[o][8 * jt + 2 * t + e];
                            dh0 += d[0][o] * wv, dh1 += d[1][o] * wv;
                        }
                        z[jt][e] = dh0 * cs2[jt][e], z[jt][2 + e] = dh1 * cs2[jt][2 + e];
                        dc[1][jt][e] += z[jt][e] + z[jt][2 + e];
                    }
                if (t == 0)
#pragma unroll
                    for (int o = 0; o < O; ++o)
                        dbo[o] += d[0][o] + d[1][o];
                to_frag<HID>(a, z);
                stage_frag<HID>(sm.stage[DZ2], a, 16 * warp, g, t);
                dense<HID>(acc, a, sm.wt.b[1], nullptr, t, lane);  // dh_1
#pragma unroll
                for (int jt = 0; jt < W::NT; ++jt)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[jt][e] *= cs1[jt][e];
#pragma unroll
                for (int jt = 0; jt < W::NT; ++jt)
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                        dc[0][jt][e] += acc[jt][e] + acc[jt][2 + e];
                to_frag<HID>(a, acc);
                stage_frag<HID>(sm.stage[DZ1], a, 16 * warp, g, t);
                dense<HID>(acc, a, sm.wt.b[0], nullptr, t, lane);  // dh_0
#pragma unroll
                for (int jt = 0; jt < W::NT; ++jt)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        acc[jt][e] *= cs0[jt][e];
                        dt[e >> 1][jt][e & 1] += acc[jt][e];
                    }
                to_frag<HID>(a, acc);
                stage_frag<HID>(sm.stage[DZ0], a, 16 * warp, g, t);
                __syncthreads();

                // ---- phase B: dV_l += dz_l^T h_{l-1} over the round's 64 pixels (warp: layer warp / 2, rows 16 (warp % 2) ..;
                // rows and columns beyond HID read the zero padding of the staged tiles and are not written),
                // corner gradients dG = W_c^T dz_0 (warp: channels 8 warp ..)
                {
                    const int mat = warp >> 1, mt = warp & 1, mi = lane >> 3, rw = lane & 7;
                    const __half(*A)[LD] = sm.stage[mat ? DZ2 : DZ1];
                    const __half(*B)[LD] = sm.stage[mat ? H1 : H0];
#pragma unroll
                    for (int ks = 0; ks < ROUND / 16; ++ks)
                    {
                        uint32_t af[4], bf[4];
                        ldsm_x4_trans(af, &A[16 * ks + ((mi >> 1) << 3) + rw][16 * mt + ((mi & 1) << 3)]);
#pragma unroll
                        for (int nt = 0; nt < W::NT; nt += 2)
                        {
                            ldsm_x4_trans(bf, &B[16 * ks + ((mi & 1) << 3) + rw][8 * nt + ((mi >> 1) << 3)]);
                            mma(av[nt], af, bf[0], bf[1]);
                            if (nt + 1 < W::NT)
                                mma(av[nt + 1 < W::NT ? nt + 1 : nt], af, bf[2], bf[3]);
                        }
                    }
                    if (warp < W::NT)
                    {
                        float ag[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                        for (int ks = 0; ks < ROUND / 16; ++ks)
                        {
                            uint32_t bf[2];
                            ldsm_x2_trans(bf, &sm.stage[DZ0][16 * ks + (lane & 15)][8 * warp]);
                            const uint4 wa = sm.wb[ks][lane];
                            const uint32_t af[4] = {wa.x, wa.y, wa.z, wa.w};
                            mma(ag, af, bf[0], bf[1]);
                        }
                        *reinterpret_cast<float2 *>(&sm.dgc[g][8 * warp + 2 * t]) = make_float2(ag[0], ag[1]);
                        *reinterpret_cast<float2 *>(&sm.dgc[g + 8][8 * warp + 2 * t]) = make_float2(ag[2], ag[3]);
                    }
                }
                __syncthreads();

                // ---- corner gradients into the cell window (fixed order); ENS: the dual cells' sums, written out
                if (ENS)
                {  // S = 8: dual cell (i0 + r, j) in row 0; S = 4: dual cells (i0 + k, j) in rows k
                    const int k = S == 8 ? 0 : threadIdx.x >> 5, ch = threadIdx.x & 31;
                    if ((S != 8 || threadIdx.x < 32) && ch < HID)
                        edges[((((size_t)n * 4 + cc) * h + i0 + (S == 8 ? r : k)) * w + j) * HID + ch] = sm.dgc[k][ch] * inv;
                }
                else if (S == 8)
                {  // dual cell (i0 + r, j): corner x / 32 -> cell (r + corner / 2, j + corner % 2)
                    const int corner = threadIdx.x >> 5, ch = threadIdx.x & 31;
                    if (ch < HID)
                        sm.win[slot(j + (corner & 1))][r + (corner >> 1)][ch] += sm.dgc[corner][ch];
                }
                else if (threadIdx.x < 2 * HID)
                {  // dual rows 0 .. BAND - 1 of column j: cell (row, j + side) gets corner side of dual row `row` and
                   // corner 2 + side of dual row `row - 1`
                    const int side = threadIdx.x / HID, ch = threadIdx.x % HID;
                    float(*col)[HID] = sm.win[slot(j + side)];
#pragma unroll
                    for (int row = 0; row <= BAND; ++row)
                    {
                        float v = row < BAND ? sm.dgc[4 * row + side][ch] : 0.f;
                        if (row > 0)
                            v += sm.dgc[4 * (row - 1) + 2 + side][ch];
                        col[row][ch] += v;
                    }
                }
                if (!ENS && r == ROUNDS - 1 && j > 0)
                {  // cell column j is finished
                    __syncthreads();
                    float(*col)[HID] = sm.win[slot(j)];
                    for (int x = threadIdx.x; x < (BAND + 1) * HID; x += THREADS)
                    {
                        const int row = x / HID, ch = x % HID;
                        const float v = col[row][ch] * inv;
                        col[row][ch] = 0.f;
                        if (row == 0 || row == BAND)
                            edges[((((size_t)n * bands + band) * 2 + (row == BAND)) * w + j) * HID + ch] = v;
                        else
                            dG[(((size_t)n * h + i0 + row) * w + j) * HID + ch] = v;
                    }
                }
            }
        // ---- table gradient of this sweep: per-block partial
        __syncthreads();  // the staging buffers are free: scratch for the S = 4 warp sum
        float *rt = red + WARPS * R;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
#pragma unroll
            for (int jt = 0; jt < W::NT; ++jt)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    const int col = 8 * jt + 2 * t + e;
                    if (S == 8)  // warp = M-tile: disjoint table rows
                        pb[L::T + cc * S * S * HID + (16 * warp + g + 8 * rr) * HID + col] = dt[rr][jt][e] * inv;
                    else
                        rt[(warp * 16 + g + 8 * rr) * HID + col] = dt[rr][jt][e];
                }
        if (S != 8)
        {
            __syncthreads();
            for (int x = threadIdx.x; x < 16 * HID; x += THREADS)
            {
                float v = 0.f;
#pragma unroll
                for (int wp = 0; wp < WARPS; ++wp)
                    v += rt[wp * 16 * HID + x];
                pb[L::T + cc * S * S * HID + x] = v * inv;
            }
        }
        __syncthreads();  // before the next sweep stages again
    }
    if (!ENS)
        for (int x = threadIdx.x; x < (BAND + 1) * HID; x += THREADS)
        {  // cell column 0 (dual columns w - 1 and 0)
            const int row = x / HID, ch = x % HID;
            const float v = sm.win[0][row][ch] * inv;
            if (row == 0 || row == BAND)
                edges[((((size_t)n * bands + band) * 2 + (row == BAND)) * w) * HID + ch] = v;
            else
                dG[(((size_t)n * h + i0 + row) * w) * HID + ch] = v;
        }

    // ---- per-block partials
    {
        const int mat = warp >> 1, mt = warp & 1;
#pragma unroll
        for (int nt = 0; nt < W::NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int row = 16 * mt + g + 8 * (e >> 1);
                if (row < HID)
                    pb[L::V + mat * HID * HID + row * HID + 8 * nt + 2 * t + (e & 1)] = av[nt][e] * inv;
            }
    }
    // dc, dW_o, db_o: sums over the lanes' rows (shuffles over g), then over the warps (fixed order)
    auto lane_sum = [](float v) {
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        return v + __shfl_xor_sync(0xffffffffu, v, 16);
    };
#pragma unroll
    for (int jt = 0; jt < W::NT; ++jt)
#pragma unroll
        for (int e = 0; e < 2; ++e)
        {
            const int ch = 8 * jt + 2 * t + e;
#pragma unroll
            for (int l = 0; l < LAYERS; ++l)
            {
                const float v = lane_sum(dc[l][jt][e]);
                if (g == 0)
                    red[warp * R + l * HID + ch] = v;
            }
#pragma unroll
            for (int o = 0; o < O; ++o)
            {
                const float v = lane_sum(dwo[o][jt][e]);
                if (g == 0)
                    red[warp * R + LAYERS * HID + o * HID + ch] = v;
            }
        }
#pragma unroll
    for (int o = 0; o < O; ++o)
    {
        const float v = lane_sum(dbo[o]);
        if (lane == 0)
            red[warp * R + LAYERS * HID + O * HID + o] = v;
    }
    __syncthreads();
    for (int x = threadIdx.x; x < R; x += THREADS)
    {
        float v = 0.f;
#pragma unroll
        for (int wp = 0; wp < WARPS; ++wp)
            v += red[wp * R + x];
        pb[L::C + x] = v * inv;  // C, WO, BO are contiguous
    }
}

// per image: the sum of its blocks' partials (fixed order). grid (ceil(SIZE / 256), N)
__global__ void reduce_kernel(const float *__restrict__ partial, int bands, int size, float *__restrict__ out)
{
    const int n = blockIdx.y, x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size)
        return;
    const float *p = partial + (size_t)n * bands * size + x;
    float v = 0.f;
    for (int b = 0; b < bands; ++b)
        v += p[(size_t)b * size];
    out[(size_t)n * size + x] = v;
}

// ensemble: dG of cell (a, b) = sum over the corners c of the corner-gradient sum of dual cell (a - c / 2, b - c % 2)
template <int HID>
__global__ void ens_gather_kernel(const float *__restrict__ cg, float *__restrict__ dG, int N, int h, int w)
{
    const size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x, total = (size_t)N * h * w * HID;
    if (x >= total)
        return;
    const int ch = x % HID, b = (x / HID) % w, a = (x / ((size_t)HID * w)) % h, n = x / ((size_t)HID * w * h);
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        const int i = a - (c >> 1) < 0 ? h - 1 : a - (c >> 1), j = b - (c & 1) < 0 ? w - 1 : b - (c & 1);
        v += cg[((((size_t)n * 4 + c) * h + i) * w + j) * HID + ch];
    }
    dG[x] = v;
}

// dG of the bands' first cell rows: the last edge of the band above (mod) + the first edge of the band
template <int HID>
__global__ void edge_kernel(const float *__restrict__ edges, float *__restrict__ dG, int N, int bands, int h, int w)
{
    const size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x, total = (size_t)N * bands * w * HID;
    if (x >= total)
        return;
    const int ch = x % HID, j = (x / HID) % w, b = (x / ((size_t)HID * w)) % bands, n = x / ((size_t)HID * w * bands);
    const int prev = b == 0 ? bands - 1 : b - 1;
    const float v = edges[((((size_t)n * bands + prev) * 2 + 1) * w + j) * HID + ch] +
                    edges[((((size_t)n * bands + b) * 2 + 0) * w + j) * HID + ch];
    dG[(((size_t)n * h + b * BAND) * w + j) * HID + ch] = v;
}

}  // namespace lppn
