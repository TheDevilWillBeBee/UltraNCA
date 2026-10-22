// VGG16 relu1_1 ... relu5_1 for the OT texture loss (ncaops/vgg_ops.py): fp16 NHWC activations and gradients.
// Convolutions are CUTLASS implicit GEMMs (sm_80+ tensor cores) with fused epilogues, accumulated in fp32 or fp16
// (acc16: fp16 accumulation runs at twice the rate on GeForce cards (Ada, Ampere), the same rate on A100 / H100):
//   forward      a = relu(conv(x, w) + b)                                   (LinearCombinationRelu, bias as a
//                                                                            stride-0 C)
//   backward     dz_prev = conv_input_grad(dz, w) * (a_prev > 0)            (MaskByPositiveSource: C = the forward
//                activation that was the convolution's input, so the output is the gradient at its pre-activation;
//                the gradients carry a per-image power-of-2 scale (vgg_ops.py) that keeps them in fp16's range)
// so no separate bias, relu or relu-backward pass touches the activations. conv1_1 and the image gradient (3 channels)
// are CUDA-core kernels (end of file). 2 x 2 max pooling keeps a 1-byte argmax for the backward; features are gathered
// at sampled positions (rows of the flattened (N h w, C) map) in the forward and their gradients scattered (with the
// relu mask) into the layer gradient in the backward. Tile shapes 128 x 128, 128 x 64 and 128 x 256 per convolution;
// vgg_ops.py picks the fastest per layer shape by timing them once.
// Pointers are passed as integers (data_ptr()); vgg_ops.py allocates every tensor (contiguous NHWC, channels multiples
// of 8).
#include <nanobind/nanobind.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"

namespace nb = nanobind;
using namespace nb::literals;
using f16 = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

#define cuda_check(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cudaError_t error = call;                                                                               \
        if (error != cudaSuccess)                                                                               \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                                     " - " + cudaGetErrorString(error));                                        \
    } while (0)

// ---------------------------------------------------------------- epilogue: D = C > 0 ? alpha * accumulator : 0
// C (fp16) is tested on its bits: > 0 as int16 <=> positive (-0 is 0x8000 < 0)
template <typename ElementOutput_, int Count, typename ElementAccumulator_ = float>
class MaskByPositiveSource {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = float;
    static int const kCount = Count;
    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
    using FragmentSource = cutlass::Array<ElementOutput, kCount>;
    static cutlass::FloatRoundStyle const kRound = cutlass::FloatRoundStyle::round_to_nearest;
    static bool const kIsHeavy = false;
    struct Params {
        float alpha = 1.f;
        CUTLASS_HOST_DEVICE Params() {}
        CUTLASS_HOST_DEVICE Params(float a) : alpha(a) {}
    };
    float alpha_;
    CUTLASS_HOST_DEVICE MaskByPositiveSource(Params const &p) : alpha_(p.alpha) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const &acc, FragmentSource const &src) const
    {
        FragmentOutput out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i)
            out[i] = ElementOutput(int16_t(ElementOutput(src[i]).raw()) > 0 ? alpha_ * float(acc[i]) : 0.f);
        return out;
    }
    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const &acc) const
    {
        FragmentOutput out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i)
            out[i] = ElementOutput(alpha_ * float(acc[i]));
        return out;
    }
};

// ---------------------------------------------------------------- convolution instantiations
template <int TM, int TN, int WM, int WN, int ST>
struct Tiles {
    using Block = cutlass::gemm::GemmShape<TM, TN, 32>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, 32>;
    static constexpr int Stages = ST;
};
using Tiles0 = Tiles<128, 128, 64, 64, 3>;  // config 0
using Tiles1 = Tiles<128, 64, 64, 32, 4>;   // config 1
using Tiles2 = Tiles<128, 256, 64, 64, 3>;  // config 2
using Instr = cutlass::gemm::GemmShape<16, 8, 16>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

template <typename T, typename Acc>
using FpropOp = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv2dFprop<
    f16, NHWC, f16, NHWC, f16, NHWC, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename T::Block,
    typename T::Warp, Instr,
    cutlass::epilogue::thread::LinearCombinationRelu<f16, 8, Acc, Acc, cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
    Swizzle, T::Stages, cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel>;

template <typename T, typename Acc>
using DgradMaskOp = cutlass::conv::device::ImplicitGemmConvolution<typename cutlass::conv::kernel::DefaultConv2dDgrad<
    f16, NHWC, f16, NHWC, f16, NHWC, Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename T::Block,
    typename T::Warp, Instr, MaskByPositiveSource<f16, 8, Acc>, Swizzle, T::Stages, cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized, cutlass::conv::StrideSupport::kUnity>::Kernel>;

static cutlass::conv::Conv2dProblemSize problem(int N, int H, int W, int C, int K)
{  // 3 x 3, padding 1, stride 1
    return cutlass::conv::Conv2dProblemSize({N, H, W, C}, {K, 3, 3, C}, {1, 1, 1, 1}, {1, 1}, {1, 1}, {N, H, W, K},
                                            cutlass::conv::Mode::kCrossCorrelation, 1);
}

template <typename Op>
static void launch(typename Op::Arguments const &args, cudaStream_t stream)
{
    Op op;
    cutlass::Status st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("vgg_cuda: convolution not supported: ") + cutlassGetStatusString(st));
    st = op.initialize(args, nullptr, stream);
    if (st == cutlass::Status::kSuccess)
        st = op(stream);
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("vgg_cuda: convolution failed: ") + cutlassGetStatusString(st));
}

template <typename F>
static void with_tiles(int config, F &&run)
{
    if (config == 0)
        run(Tiles0{});
    else if (config == 1)
        run(Tiles1{});
    else
        run(Tiles2{});
}

// y (N, H, W, K) = relu(conv(x (N, H, W, C), w (K, 3, 3, C)) + b (K)), fp16; acc16: fp16 accumulation
void fprop(int config, bool acc16, uintptr_t x, uintptr_t w, uintptr_t b, uintptr_t y, int N, int H, int W, int C, int K,
           uintptr_t stream)
{
    auto run = [&](auto tiles, auto acc) {
        using Acc = decltype(acc);
        launch<FpropOp<decltype(tiles), Acc>>({problem(N, H, W, C, K), {(f16 *)x, NHWC::packed({N, H, W, C})},
                                               {(f16 *)w, NHWC::packed({K, 3, 3, C})}, {(f16 *)b, NHWC::Stride(0)},
                                               {(f16 *)y, NHWC::packed({N, H, W, K})}, {Acc(1)}},
                                              (cudaStream_t)stream);
    };
    with_tiles(config, [&](auto tiles) { acc16 ? run(tiles, f16{}) : run(tiles, float{}); });
}

// dx (N, H, W, C) = conv_input_grad(dy (N, H, W, K), w) * (act (N, H, W, C) > 0), fp16; acc16: fp16 accumulation
void dgrad_mask(int config, bool acc16, uintptr_t dy, uintptr_t w, uintptr_t act, uintptr_t dx, int N, int H, int W, int C,
                int K, uintptr_t stream)
{
    auto run = [&](auto tiles, auto acc) {
        launch<DgradMaskOp<decltype(tiles), decltype(acc)>>(
            {problem(N, H, W, C, K), {(f16 *)dy, NHWC::packed({N, H, W, K})}, {(f16 *)w, NHWC::packed({K, 3, 3, C})},
             {(f16 *)act, NHWC::packed({N, H, W, C})}, {(f16 *)dx, NHWC::packed({N, H, W, C})}, {1.f}},
            (cudaStream_t)stream);
    };
    with_tiles(config, [&](auto tiles) { acc16 ? run(tiles, f16{}) : run(tiles, float{}); });
}

// ---------------------------------------------------------------- pooling, gather, scatter (8 channels per thread)
union Pack8 {
    uint4 u;
    __nv_bfloat16 h[8];
    __half f[8];
    short s[8];  // fp16 > 0 <=> s > 0
};

// out (N, H/2, W/2, C) = max over 2 x 2 windows of in (N, H, W, C); idx (optional): window position of the first max
__global__ void pool_kernel(const uint4 *__restrict__ in, uint4 *__restrict__ out, uint2 *__restrict__ idx, int H, int W,
                            int C8, long total)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    const int c = t % C8, ow = (t / C8) % (W / 2), oh = (t / C8 / (W / 2)) % (H / 2);
    const long n = t / C8 / (W / 2) / (H / 2);
    Pack8 best, v;
    uint8_t arg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    best.u = in[((n * H + 2 * oh) * W + 2 * ow) * C8 + c];
#pragma unroll
    for (int k = 1; k < 4; ++k)
    {
        v.u = in[((n * H + 2 * oh + (k >> 1)) * W + 2 * ow + (k & 1)) * C8 + c];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (__half2float(v.f[i]) > __half2float(best.f[i]))
                best.f[i] = v.f[i], arg[i] = k;
    }
    out[t] = best.u;
    if (idx)  // only with a backward
        idx[t] = make_uint2(arg[0] | arg[1] << 8 | arg[2] << 16 | arg[3] << 24, arg[4] | arg[5] << 8 | arg[6] << 16 | arg[7] << 24);
}

// out (N, H, W, C): g (N, H/2, W/2, C) at each window's argmax, 0 elsewhere
__global__ void unpool_kernel(const uint4 *__restrict__ g, const uint2 *__restrict__ idx, uint4 *__restrict__ out, int H,
                              int W, int C8, long total)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    const int c = t % C8, ow = (t / C8) % (W / 2), oh = (t / C8 / (W / 2)) % (H / 2);
    const long n = t / C8 / (W / 2) / (H / 2);
    Pack8 v, o;
    v.u = g[t];
    const uint2 a = idx[t];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int arg = ((i < 4 ? a.x : a.y) >> (8 * (i & 3))) & 0xff;
            o.s[i] = arg == k ? v.s[i] : 0;
        }
        out[((n * H + 2 * oh + (k >> 1)) * W + 2 * ow + (k & 1)) * C8 + c] = o.u;
    }
}

// out (R, C) fp32 = f (rows, C) fp16 at rows[r]
__global__ void gather_kernel(const uint4 *__restrict__ f, const long *__restrict__ rows, float4 *__restrict__ out, int C8,
                              long total)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    Pack8 v;
    v.u = f[rows[t / C8] * C8 + t % C8];
    out[2 * t] = make_float4(__half2float(v.f[0]), __half2float(v.f[1]), __half2float(v.f[2]), __half2float(v.f[3]));
    out[2 * t + 1] = make_float4(__half2float(v.f[4]), __half2float(v.f[5]), __half2float(v.f[6]), __half2float(v.f[7]));
}

// 8 gradient values of row t: fp32 (samples), fp16 or bf16 (dense feature-map gradients)
__device__ __forceinline__ void load8(const float4 *g, long t, float (&v)[8])
{
    const float4 a = g[2 * t], b = g[2 * t + 1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
struct Bf16x8 {  // tags for load8()'s overloads
    uint4 u;
};
struct F16x8 {
    uint4 u;
};
__device__ __forceinline__ void load8(const Bf16x8 *g, long t, float (&v)[8])
{
    Pack8 p;
    p.u = g[t].u;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        v[i] = __bfloat162float(p.h[i]);
}
__device__ __forceinline__ void load8(const F16x8 *g, long t, float (&v)[8])
{
    Pack8 p;
    p.u = g[t].u;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        v[i] = __half2float(p.f[i]);
}

// dz (rows, C) fp16 [+]= (act (rows, C) > 0) * scale[row / M] * g (R, C) at rows[r] (rows == nullptr: row r, all
// rows in order; M: rows per image). add: dz += ..., else dz = ... (the top layer). The rows must be distinct.
template <typename G>
__global__ void scatter_kernel(uint4 *__restrict__ dz, const uint4 *__restrict__ act, const long *__restrict__ rows,
                               const G *__restrict__ g, const float *__restrict__ scale, long M, int C8, long total, bool add)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total)
        return;
    const long row = rows ? rows[t / C8] : t / C8, e = row * C8 + t % C8;
    const float sc = scale[row / M];
    Pack8 a, d;
    a.u = act[e];
    d.u = add ? dz[e] : make_uint4(0, 0, 0, 0);
    float gv[8];
    load8(g, t, gv);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d.f[i] = __float2half(__half2float(d.f[i]) + (a.s[i] > 0 ? sc * gv[i] : 0.f));
    dz[e] = d.u;
}

static unsigned blocks(long total) { return (unsigned)((total + 255) / 256); }

void pool(uintptr_t in, uintptr_t out, uintptr_t idx, int N, int H, int W, int C, uintptr_t stream)
{
    const long total = (long)N * (H / 2) * (W / 2) * (C / 8);
    pool_kernel<<<blocks(total), 256, 0, (cudaStream_t)stream>>>((const uint4 *)in, (uint4 *)out, (uint2 *)idx, H, W, C / 8, total);
    cuda_check(cudaGetLastError());
}

void unpool(uintptr_t g, uintptr_t idx, uintptr_t out, int N, int H, int W, int C, uintptr_t stream)
{
    const long total = (long)N * (H / 2) * (W / 2) * (C / 8);
    unpool_kernel<<<blocks(total), 256, 0, (cudaStream_t)stream>>>((const uint4 *)g, (const uint2 *)idx, (uint4 *)out, H, W, C / 8, total);
    cuda_check(cudaGetLastError());
}

void gather(uintptr_t f, uintptr_t rows, uintptr_t out, long R, int C, uintptr_t stream)
{
    const long total = R * (C / 8);
    gather_kernel<<<blocks(total), 256, 0, (cudaStream_t)stream>>>((const uint4 *)f, (const long *)rows, (float4 *)out, C / 8, total);
    cuda_check(cudaGetLastError());
}

void scatter(uintptr_t dz, uintptr_t act, uintptr_t rows, uintptr_t g, int g_type, uintptr_t scale, long M, long R, int C,
             bool add, uintptr_t stream)
{
    const long total = R * (C / 8);
    auto run = [&](auto *gp) {
        scatter_kernel<<<blocks(total), 256, 0, (cudaStream_t)stream>>>((uint4 *)dz, (const uint4 *)act, (const long *)rows,
                                                                        gp, (const float *)scale, M, C / 8, total, add);
    };
    if (g_type == 0)
        run((const float4 *)g);
    else if (g_type == 1)
        run((const F16x8 *)g);
    else
        run((const Bf16x8 *)g);
    cuda_check(cudaGetLastError());
}

// ---------------------------------------------------------------- the first layer (3 input channels, CUDA cores)
// conv1_1 is memory-bound (it writes 64 channels per pixel from 3), so it runs on CUDA cores from the fp32 RGB input
// directly, and so does its input gradient (3 channels per pixel from 64). 16 x 16 pixels per 256-thread block. A
// pixel's 64 fp16 channels (8 x 16 bytes) sit in shared memory with chunk j at position j ^ (pixel & 7): threads on
// consecutive pixels then hit different banks.
constexpr int T1 = 16;
__device__ __forceinline__ int swz(int p, int j) { return p * 8 + (j ^ (p & 7)); }

// y (N, H, W, 64) fp16 = relu(conv3x3(a * x + b) + bias): x (N, 3, H, W) fp32 with strides (sN, sC, sH, sW), a and b
// per channel (normalisation; zero padding after it), w (27, 64) fp32 (tap (ky, kx, c), output channel), bias (64)
__global__ void __launch_bounds__(256) conv_in_kernel(const float *__restrict__ x, long sN, long sC, long sH, long sW,
                                                      const float *__restrict__ ab, const float *__restrict__ w,
                                                      const float *__restrict__ bias, uint4 *__restrict__ y, int H, int W)
{
    __shared__ float xs[3][T1 + 2][T1 + 2];
    __shared__ float4 ws[27][16];
    __shared__ uint4 ys[T1 * T1 * 8];  // the block's output, written out row by row
    const int tid = threadIdx.x, n = blockIdx.z, y0 = blockIdx.y * T1, x0 = blockIdx.x * T1;
    for (int i = tid; i < 27 * 16; i += 256)
        ws[i / 16][i % 16] = reinterpret_cast<const float4 *>(w)[i];
    for (int i = tid; i < 3 * (T1 + 2) * (T1 + 2); i += 256)
    {
        const int c = i / ((T1 + 2) * (T1 + 2)), yy = i / (T1 + 2) % (T1 + 2), xx = i % (T1 + 2);
        const int gy = y0 + yy - 1, gx = x0 + xx - 1;
        xs[c][yy][xx] = gy >= 0 && gy < H && gx >= 0 && gx < W ? ab[c] * x[n * sN + c * sC + gy * sH + gx * sW] + ab[3 + c] : 0.f;
    }
    __syncthreads();
    const int ty = tid / T1, tx = tid % T1;
    float acc[64];
#pragma unroll
    for (int k = 0; k < 64; ++k)
        acc[k] = bias[k];
#pragma unroll
    for (int ky = 0; ky < 3; ++ky)
#pragma unroll
        for (int kx = 0; kx < 3; ++kx)
#pragma unroll
            for (int c = 0; c < 3; ++c)
            {
                const float v = xs[c][ty + ky][tx + kx];
                const int tap = (ky * 3 + kx) * 3 + c;
#pragma unroll
                for (int k4 = 0; k4 < 16; ++k4)
                {
                    const float4 wv = ws[tap][k4];
                    acc[4 * k4] += v * wv.x, acc[4 * k4 + 1] += v * wv.y, acc[4 * k4 + 2] += v * wv.z, acc[4 * k4 + 3] += v * wv.w;
                }
            }
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        Pack8 o;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            o.f[i] = __float2half(fmaxf(acc[8 * j + i], 0.f));
        ys[swz(tid, j)] = o.u;
    }
    __syncthreads();
    for (int i = tid; i < T1 * T1 * 8; i += 256)  // row r of the tile: 16 pixels x 8 uint4, contiguous in y
    {
        const int r = i / (T1 * 8), gy = y0 + r, gx = x0 + i % (T1 * 8) / 8;
        if (gy < H && gx < W)
            y[((long)(n * H + gy) * W + gx) * 8 + i % 8] = ys[swz(i / 8, i % 8)];
    }
}

// dx (N, 3, H, W) fp32 (strides sN, sC, sH, sW) = mult[n] * cmul[c] * conv3x3_input_grad(dz (N, H, W, 64) fp16, w):
// w (9, 64, 3) fp32 (tap (ky, kx), output channel k of conv1_1, input channel c). Dynamic shared memory: the dz tile
// with its halo (fp16, swizzled) and w.
constexpr int DGRAD_IN_SMEM = (T1 + 2) * (T1 + 2) * 8 * 16 + 9 * 64 * 3 * 4;
__global__ void __launch_bounds__(256) dgrad_in_kernel(const uint4 *__restrict__ dz, const float4 *__restrict__ w,
                                                       const float *__restrict__ mult, const float *__restrict__ cmul,
                                                       float *__restrict__ dx, long sN, long sC, long sH, long sW, int H, int W)
{
    extern __shared__ uint4 smem[];
    uint4 *zs = smem;
    float4 *ws = reinterpret_cast<float4 *>(smem + (T1 + 2) * (T1 + 2) * 8);  // per (tap, 8 channels): 6 float4
    const int tid = threadIdx.x, n = blockIdx.z, y0 = blockIdx.y * T1, x0 = blockIdx.x * T1;
    for (int i = tid; i < 9 * 64 * 3 / 4; i += 256)
        ws[i] = w[i];
    for (int i = tid; i < (T1 + 2) * (T1 + 2) * 8; i += 256)
    {
        const int p = i / 8, gy = y0 + p / (T1 + 2) - 1, gx = x0 + p % (T1 + 2) - 1;
        zs[swz(p, i % 8)] = gy >= 0 && gy < H && gx >= 0 && gx < W ? dz[((long)(n * H + gy) * W + gx) * 8 + i % 8] : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    const int ty = tid / T1, tx = tid % T1, gy = y0 + ty, gx = x0 + tx;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
#pragma unroll
    for (int tap = 0; tap < 9; ++tap)
    {
        const int p = (ty + 2 - tap / 3) * (T1 + 2) + tx + 2 - tap % 3;  // dz at pixel - (ky - 1, kx - 1)
#pragma unroll 2
        for (int k8 = 0; k8 < 8; ++k8)
        {
            Pack8 v;
            v.u = zs[swz(p, k8)];
            float z[8], wv[24];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&v.f[2 * i]));
                z[2 * i] = f.x, z[2 * i + 1] = f.y;
            }
#pragma unroll
            for (int i = 0; i < 6; ++i)
                *reinterpret_cast<float4 *>(&wv[4 * i]) = ws[(tap * 8 + k8) * 6 + i];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                a0 += z[i] * wv[3 * i], a1 += z[i] * wv[3 * i + 1], a2 += z[i] * wv[3 * i + 2];
        }
    }
    if (gy < H && gx < W)
    {
        const float m = mult[n];
        float *o = dx + n * sN + gy * sH + gx * sW;
        o[0] = m * cmul[0] * a0, o[sC] = m * cmul[1] * a1, o[2 * sC] = m * cmul[2] * a2;
    }
}

void conv_in(uintptr_t x, long sN, long sC, long sH, long sW, uintptr_t ab, uintptr_t w, uintptr_t bias, uintptr_t y, int N,
             int H, int W, uintptr_t stream)
{
    const dim3 grid((W + T1 - 1) / T1, (H + T1 - 1) / T1, N);
    conv_in_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const float *)x, sN, sC, sH, sW, (const float *)ab,
                                                           (const float *)w, (const float *)bias, (uint4 *)y, H, W);
    cuda_check(cudaGetLastError());
}

void dgrad_in(uintptr_t dz, uintptr_t w, uintptr_t mult, uintptr_t cmul, uintptr_t dx, long sN, long sC, long sH, long sW,
              int N, int H, int W, uintptr_t stream)
{
    const dim3 grid((W + T1 - 1) / T1, (H + T1 - 1) / T1, N);
    cuda_check(cudaFuncSetAttribute(dgrad_in_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DGRAD_IN_SMEM));
    dgrad_in_kernel<<<grid, 256, DGRAD_IN_SMEM, (cudaStream_t)stream>>>((const uint4 *)dz, (const float4 *)w, (const float *)mult,
                                                                        (const float *)cmul, (float *)dx, sN, sC, sH, sW, H, W);
    cuda_check(cudaGetLastError());
}

NB_MODULE(vgg_cuda, m)
{
    m.def("fprop", &fprop, "config"_a, "acc16"_a, "x"_a, "w"_a, "b"_a, "y"_a, "N"_a, "H"_a, "W"_a, "C"_a, "K"_a, "stream"_a,
          "y (N, H, W, K) = relu(conv3x3(x (N, H, W, C), w (K, 3, 3, C)) + b (K)), fp16, fp32 or (acc16) fp16 "
          "accumulation; config 0: 128 x 128 tiles, 1: 128 x 64, 2: 128 x 256");
    m.def("dgrad_mask", &dgrad_mask, "config"_a, "acc16"_a, "dy"_a, "w"_a, "act"_a, "dx"_a, "N"_a, "H"_a, "W"_a, "C"_a, "K"_a,
          "stream"_a, "dx (N, H, W, C) = conv3x3_input_grad(dy (N, H, W, K), w) * (act (N, H, W, C) > 0), fp16, fp32 or "
          "(acc16) fp16 accumulation");
    m.def("conv_in", &conv_in, "x"_a, "sN"_a, "sC"_a, "sH"_a, "sW"_a, "ab"_a, "w"_a, "bias"_a, "y"_a, "N"_a, "H"_a, "W"_a,
          "stream"_a, "conv1_1: y (N, H, W, 64) fp16 = relu(conv3x3(a x + b) + bias), x (N, 3, H, W) fp32 (strides), "
          "ab (6) fp32, w (27, 64) fp32, bias (64) fp32");
    m.def("dgrad_in", &dgrad_in, "dz"_a, "w"_a, "mult"_a, "cmul"_a, "dx"_a, "sN"_a, "sC"_a, "sH"_a, "sW"_a, "N"_a, "H"_a,
          "W"_a, "stream"_a, "conv1_1's input gradient: dx (N, 3, H, W) fp32 (strides) = mult[n] cmul[c] "
          "conv3x3_input_grad(dz (N, H, W, 64) fp16, w (9, 64, 3) fp32)");
    m.def("pool", &pool, "x"_a, "y"_a, "idx"_a, "N"_a, "H"_a, "W"_a, "C"_a, "stream"_a,
          "2 x 2 max pooling of x (N, H, W, C) fp16 -> y (N, H/2, W/2, C), idx (uint8 window position of the first max; "
          "0: not written)");
    m.def("unpool", &unpool, "g"_a, "idx"_a, "out"_a, "N"_a, "H"_a, "W"_a, "C"_a, "stream"_a,
          "gradient of pool: out (N, H, W, C) = g (N, H/2, W/2, C) at the argmax, 0 elsewhere");
    m.def("gather", &gather, "f"_a, "rows"_a, "out"_a, "R"_a, "C"_a, "stream"_a,
          "out (R, C) fp32 = f (rows, C) fp16 at rows (R) int64");
    m.def("scatter", &scatter, "dz"_a, "act"_a, "rows"_a, "g"_a, "g_type"_a, "scale"_a, "M"_a, "R"_a, "C"_a, "add"_a, "stream"_a,
          "dz (rows, C) fp16 [+]= (act (fp16) > 0) * scale[row / M] * g (R, C) (g_type 0: fp32, 1: fp16, 2: bf16) at rows (R) "
          "int64, distinct (rows 0: all rows in order); scale fp32 per image, M rows per image");
}
