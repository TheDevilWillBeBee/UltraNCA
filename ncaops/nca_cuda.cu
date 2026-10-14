#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <type_traits>
#include <nanobind/stl/tuple.h>
#include "nca_step.cuh"

namespace nb = nanobind;
using namespace nb::literals;

using FloatArray2D = nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cuda>;
using FloatArray3D = nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cuda>;
using FloatArray1D = nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FloatArray4D = nb::ndarray<float, nb::shape<-1, -1, -1, -1>, nb::c_contig, nb::device::cuda>;
using ByteArray1D = nb::ndarray<uint8_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using ByteArray3D = nb::ndarray<uint8_t, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cuda>;

#define cuda_check(call)                                                              \
    do                                                                                \
    {                                                                                 \
        cudaError_t error = call;                                                     \
        if (error != cudaSuccess)                                                     \
        {                                                                             \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " +               \
                                     cudaGetErrorString(error));                      \
        }                                                                             \
    } while (0)

template <typename ArrayType>
void validate_array_shape(const ArrayType &array, const std::vector<size_t> &expected, const std::string &name)
{
    bool ok = array.ndim() == expected.size();
    for (size_t i = 0; ok && i < expected.size(); ++i)
        ok = array.shape(i) == expected[i];
    if (!ok)
    {
        auto fmt = [](auto get, size_t n) {
            std::string s = "(";
            for (size_t i = 0; i < n; ++i)
                s += (i ? ", " : "") + std::to_string(get(i));
            return s + ")";
        };
        throw std::runtime_error(name + " must have shape " + fmt([&](size_t i) { return expected[i]; }, expected.size()) +
                                 ", got " + fmt([&](size_t i) { return array.shape(i); }, array.ndim()));
    }
}

// Compile-time NCA configuration handed to generic lambdas by dispatch().
template <int C_, int HID_, bool FP32_, bool STOCH_, bool ALIVE_>
struct Inst
{
    static constexpr int C = C_, HID = HID_;
    static constexpr bool FP32 = FP32_, STOCH = STOCH_, ALIVE = ALIVE_;
};

template <typename F>
auto with_bool(bool v, F &&f) { return v ? f(std::true_type{}) : f(std::false_type{}); }

// (channels, hidden) instantiations, each with every (fp32, stochastic, alive) combination; add a line here for
// another NCA size.
template <typename F>
auto dispatch(size_t C, size_t HID, bool fp32, bool stoch, bool alive, F &&f)
{
    return with_bool(fp32, [&](auto FP) { return with_bool(stoch, [&](auto ST) { return with_bool(alive, [&](auto AL) {
        if (C == 12 && HID == 96)
            return f(Inst<12, 96, FP, ST, AL>{});
        if (C == 16 && HID == 128)
            return f(Inst<16, 128, FP, ST, AL>{});
        throw std::runtime_error("no kernel instantiated for channels=" + std::to_string(C) + ", hidden=" + std::to_string(HID));
    }); }); });
}

template <typename Kernel>
void set_smem(Kernel kernel, int bytes)
{
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

static int sm_count()
{
    static int n = [] { int dev, v; cudaGetDevice(&dev); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev); return v; }();
    return n;
}

// One step x -> y (N images of C x H x W, K = N / B NCAs) on `stream`; growing: alpha workspace and alive mask (N, H, W).
static void launch_forward(const float *x, const float *w1, const float *b1, const float *w2, float *y, float *alpha,
                           uint8_t *alive, size_t N, size_t C, size_t HID, size_t H, size_t W, uint32_t B, uint64_t seed,
                           float update_rate, bool fp32, bool growing, cudaStream_t stream)
{
    dispatch(C, HID, fp32, update_rate < 1.f, growing, [&](auto I) {
        using D = decltype(I);
        auto kernel = nca_step::forward_kernel<D::C, D::HID, D::FP32, D::STOCH, D::ALIVE>;
        constexpr int smem = sizeof(nca_step::ForwardSmem<D::C, D::HID, D::FP32>);
        static bool init = (set_smem(kernel, smem), true);
        (void)init;
        static const int blocks_per_sm = [&] { int b; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, nca_step::FWD_THREADS, smem); return b; }();
        // segments of whole columns, unless that gives less than two waves of blocks (shorter segments: more partial rounds)
        const int strips = ceil_div(W, nca_step::TILE_W);
        int seg = H;
        while (seg > nca_step::FWD_PHASE && (size_t)N * strips * ceil_div(H, seg) < 2 * (size_t)blocks_per_sm * sm_count())
            seg = ceil_div(seg, 2);
        kernel<<<dim3(strips * ceil_div(H, seg), N), nca_step::FWD_THREADS, smem, stream>>>(
            x, w1, b1, w2, y, alpha, B, H, W, seed, update_rate, seg);
        cuda_check(cudaGetLastError());
        if constexpr (D::ALIVE)
        {
            const uint64_t total = (uint64_t)N * H * W;
            nca_step::alive_kernel<D::C><<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(x, alpha, y, alive, H, W, total);
            cuda_check(cudaGetLastError());
        }
    });
}

static void validate_forward(FloatArray4D x, FloatArray3D w1, FloatArray2D b1, FloatArray3D w2, FloatArray4D y,
                             FloatArray3D alpha, ByteArray3D alive, uint32_t batch_per_nca, bool growing)
{
    const size_t N = x.shape(0), C = x.shape(1), H = x.shape(2), W = x.shape(3), K = w1.shape(0), HID = w1.shape(1);
    if (N != K * batch_per_nca)
        throw std::runtime_error("x.shape[0] must equal (number of NCAs) * batch_per_nca");
    validate_array_shape(w1, {K, HID, 4 * C}, "w1");
    validate_array_shape(b1, {K, HID}, "b1");
    validate_array_shape(w2, {K, C, HID}, "w2");
    validate_array_shape(y, {N, C, H, W}, "y");
    if (growing)
    {
        if (C < 4)
            throw std::runtime_error("a growing NCA needs at least 4 channels (alpha = channel 3)");
        validate_array_shape(alpha, {N, H, W}, "alpha");
        validate_array_shape(alive, {N, H, W}, "alive");
    }
}

void forward(FloatArray4D x, FloatArray3D w1, FloatArray2D b1, FloatArray3D w2, FloatArray4D y,
             FloatArray3D alpha, ByteArray3D alive, uint32_t batch_per_nca, uint64_t seed, float update_rate, bool fp32,
             bool growing, uintptr_t stream)
{
    validate_forward(x, w1, b1, w2, y, alpha, alive, batch_per_nca, growing);
    launch_forward(x.data(), w1.data(), b1.data(), w2.data(), y.data(), alpha.data(), alive.data(), x.shape(0), x.shape(1),
                   w1.shape(1), x.shape(2), x.shape(3), batch_per_nca, seed, update_rate, fp32, growing,
                   reinterpret_cast<cudaStream_t>(stream));
}

// splitmix64: the seed of step i of a rollout
static uint64_t step_seed(uint64_t seed, uint64_t i)
{
    uint64_t z = seed + (i + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// `steps` steps from x, ping-ponging between y and tmp so that the result ends in y (x is not modified).
void rollout(FloatArray4D x, FloatArray3D w1, FloatArray2D b1, FloatArray3D w2, FloatArray4D y, FloatArray4D tmp,
             FloatArray3D alpha, ByteArray3D alive, uint32_t steps, uint32_t batch_per_nca, uint64_t seed,
             float update_rate, bool fp32, bool growing, uintptr_t stream_handle)
{
    validate_forward(x, w1, b1, w2, y, alpha, alive, batch_per_nca, growing);
    if (steps > 1)
        validate_array_shape(tmp, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)}, "tmp");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (steps == 0)
        cuda_check(cudaMemcpyAsync(y.data(), x.data(), x.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    const float *src = x.data();
    for (uint32_t i = 0; i < steps; ++i)
    {
        float *dst = (steps - i) % 2 ? y.data() : tmp.data();
        launch_forward(src, w1.data(), b1.data(), w2.data(), dst, alpha.data(), alive.data(), x.shape(0), x.shape(1),
                       w1.shape(1), x.shape(2), x.shape(3), batch_per_nca, step_seed(seed, i), update_rate, fp32,
                       growing, stream);
        src = dst;
    }
}

static int backward_blocks_per_image(int H, int W)
{
    const int tiles = ceil_div(W, nca_step::TILE_W) * ceil_div(H, nca_step::BWD_TILE_H);
    return ceil_div(tiles, nca_step::BWD_TILES_PER_BLOCK);
}

// Images per backward chunk: the backward and adjoint kernels run on chunks of images, alternating between two
// streams with a dp buffer each, so that the adjoint reads dp from L2 rather than DRAM and each chunk's backward
// overlaps the other's tail. A chunk is about one wave of resident backward blocks, and the dp (4C planes per image)
// of both chunks in flight fits in L2 (measured on sm_89: 13-23 % faster than one launch).
static size_t chunk_images(size_t N, size_t C, size_t H, size_t W, int blocks_per_image, int blocks_per_sm)
{
    int dev, sms, l2;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    const size_t wave = sms * blocks_per_sm / blocks_per_image, fit = l2 / 2 / (4 * C * H * W * sizeof(float));
    return std::clamp<size_t>(std::min(wave, fit), 1, N);
}

// The two streams of the chunked backward (created once, for the current device).
static cudaStream_t side_stream(int i)
{
    static cudaStream_t s[2] = {[] { cudaStream_t t; cudaStreamCreateWithFlags(&t, cudaStreamNonBlocking); return t; }(),
                                [] { cudaStream_t t; cudaStreamCreateWithFlags(&t, cudaStreamNonBlocking); return t; }()};
    return s[i];
}

// `waiter` waits for the work enqueued so far on `stream`.
static void join(cudaStream_t waiter, cudaStream_t stream)
{
    cudaEvent_t e;
    cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    cuda_check(cudaEventRecord(e, stream));
    cuda_check(cudaStreamWaitEvent(waiter, e, 0));
    cuda_check(cudaEventDestroy(e));  // released once complete
}

// Workspace sizes for backward(): (packed weight bytes per NCA, partial floats per image, images per chunk).
std::tuple<size_t, size_t, size_t> backward_workspace(size_t C, size_t HID, bool fp32, size_t N, size_t H, size_t W)
{
    return dispatch(C, HID, fp32, true, false, [&](auto I) {
        using D = decltype(I);
        auto kernel = nca_step::backward_kernel<D::C, D::HID, D::FP32, D::STOCH, D::ALIVE>;
        constexpr int smem = sizeof(nca_step::BackwardSmem<D::C, D::HID>);
        set_smem(kernel, smem);
        int blocks_per_sm;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, nca_step::BWD_THREADS, smem));
        const int blocks = backward_blocks_per_image(H, W);
        return std::make_tuple(sizeof(nca_step::BackwardWeights<D::C, D::HID, D::FP32>),
                               (size_t)blocks * nca_step::BwdShape<D::C, D::HID>::PARTIAL,
                               chunk_images(N, C, H, W, blocks, blocks_per_sm));
    });
}

void backward(FloatArray4D x, FloatArray4D gy, ByteArray3D alive, FloatArray3D w1, FloatArray2D b1, FloatArray3D w2,
              FloatArray4D gx, FloatArray3D dw1, FloatArray2D db1, FloatArray3D dw2,
              ByteArray1D packed, FloatArray4D dp, FloatArray1D partial,
              uint32_t batch_per_nca, uint64_t seed, float update_rate, bool fp32, bool growing, uintptr_t stream_handle)
{
    const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const size_t N = x.shape(0), C = x.shape(1), H = x.shape(2), W = x.shape(3), K = w1.shape(0), HID = w1.shape(1);
    if (N != K * batch_per_nca)
        throw std::runtime_error("x.shape[0] must equal (number of NCAs) * batch_per_nca");
    validate_array_shape(gy, {N, C, H, W}, "gy");
    validate_array_shape(gx, {N, C, H, W}, "gx");
    validate_array_shape(w1, {K, HID, 4 * C}, "w1");
    validate_array_shape(dw1, {K, HID, 4 * C}, "dw1");
    validate_array_shape(b1, {K, HID}, "b1");
    validate_array_shape(db1, {K, HID}, "db1");
    validate_array_shape(w2, {K, C, HID}, "w2");
    validate_array_shape(dw2, {K, C, HID}, "dw2");
    const auto workspace = backward_workspace(C, HID, fp32, N, H, W);
    const size_t packed_bytes = std::get<0>(workspace), partial_floats = std::get<1>(workspace);
    // dp: N images (one launch on `stream`), or two chunks of dp.shape[0] / 2 images (alternating side streams)
    const size_t chunk = dp.shape(0) >= N ? N : dp.shape(0) / 2;
    if (chunk == 0 || (chunk < N && dp.shape(0) % 2))
        throw std::runtime_error("dp must hold N images or two chunks of images");
    validate_array_shape(dp, {chunk < N ? 2 * chunk : N, 4 * C, H, W}, "dp");
    validate_array_shape(packed, {K * packed_bytes}, "packed");
    validate_array_shape(partial, {N * partial_floats}, "partial");
    if (growing)
        validate_array_shape(alive, {N, H, W}, "alive");
    dispatch(C, HID, fp32, update_rate < 1.f, growing, [&](auto I) {
        using D = decltype(I);
        using BW = nca_step::BackwardWeights<D::C, D::HID, D::FP32>;
        using BS = nca_step::BwdShape<D::C, D::HID>;
        BW *pk = reinterpret_cast<BW *>(packed.data());
        nca_step::pack_kernel<D::C, D::HID, D::FP32><<<K, 256, 0, stream>>>(w1.data(), b1.data(), w2.data(), pk);
        cuda_check(cudaGetLastError());

        auto kernel = nca_step::backward_kernel<D::C, D::HID, D::FP32, D::STOCH, D::ALIVE>;
        constexpr int smem = sizeof(nca_step::BackwardSmem<D::C, D::HID>);
        set_smem(kernel, smem);
        const int blocks = backward_blocks_per_image(H, W);
        const size_t plane = H * W;
        const bool split = chunk < N;
        if (split)
            for (int i = 0; i < 2; ++i)
                join(side_stream(i), stream);
        for (size_t n0 = 0, i = 0; n0 < N; n0 += chunk, ++i)
        {
            const size_t n = std::min(chunk, N - n0);
            const cudaStream_t s = split ? side_stream(i % 2) : stream;
            float *dpc = dp.data() + (i % 2) * chunk * 4 * D::C * plane;  // in-stream order protects the buffer
            kernel<<<dim3(blocks, n), nca_step::BWD_THREADS, smem, s>>>(x.data(), gy.data(), alive.data(), pk, dpc,
                                                                       partial.data(), n0, batch_per_nca, H, W, seed, update_rate);
            cuda_check(cudaGetLastError());
            const uint64_t total = (uint64_t)n * D::C * H * ceil_div(W, 4);  // 4 pixels per thread
            nca_step::adjoint_kernel<D::C, D::ALIVE><<<(unsigned)((total + 255) / 256), 256, 0, s>>>(
                dpc, gy.data() + n0 * D::C * plane, alive.data() + (growing ? n0 * plane : 0),
                gx.data() + n0 * D::C * plane, H, W, total);
            cuda_check(cudaGetLastError());
        }
        if (split)
            for (int i = 0; i < 2; ++i)
                join(stream, side_stream(i));
        nca_step::reduce_kernel<D::C, D::HID><<<dim3(ceil_div(BS::PARTIAL, 256), K), 256, 0, stream>>>(
            partial.data(), batch_per_nca * blocks, dw1.data(), db1.data(), dw2.data());
        cuda_check(cudaGetLastError());
    });
}

void update_mask(FloatArray3D mask, uint64_t seed, float update_rate)
{
    const uint64_t total = mask.shape(0) * mask.shape(1) * mask.shape(2);
    nca_step::mask_kernel<<<(unsigned)std::min<uint64_t>((total + 255) / 256, 65535), 256>>>(
        mask.data(), mask.shape(0), mask.shape(1), mask.shape(2), seed, update_rate);
    cuda_check(cudaGetLastError());
}

NB_MODULE(nca_cuda, m)
{
    m.def("forward", &forward, "x"_a, "w1"_a, "b1"_a, "w2"_a, "y"_a, "alpha"_a, "alive"_a, "batch_per_nca"_a, "seed"_a,
          "update_rate"_a, "fp32"_a, "growing"_a, "stream"_a,
          "One NCA step on CUDA stream `stream` (handle). x, y: (N, C, H, W) with N = K * batch_per_nca; w1: (K, hidden, 4C); b1: (K, hidden); "
          "w2: (K, C, hidden). fp32=False uses TF32 tensor cores, fp32=True 3xTF32 (fp32 accuracy). update_rate >= 1: "
          "every cell updates. growing: alive masking (alpha = channel 3); alpha (N, H, W) is a workspace and alive "
          "(N, H, W) receives the mask for backward() (both unused otherwise).");
    m.def("rollout", &rollout, "x"_a, "w1"_a, "b1"_a, "w2"_a, "y"_a, "tmp"_a, "alpha"_a, "alive"_a, "steps"_a,
          "batch_per_nca"_a, "seed"_a, "update_rate"_a, "fp32"_a, "growing"_a, "stream"_a,
          "`steps` forward steps from x into y (tmp: workspace like x if steps > 1), with the step seeds derived from "
          "`seed`; no gradients.");
    m.def("backward_workspace", &backward_workspace, "channels"_a, "hidden"_a, "fp32"_a, "N"_a, "H"_a, "W"_a,
          "(packed weight bytes per NCA, partial floats per image, images of dp) for backward().");
    m.def("backward", &backward, "x"_a, "gy"_a, "alive"_a, "w1"_a, "b1"_a, "w2"_a, "gx"_a, "dw1"_a, "db1"_a, "dw2"_a,
          "packed"_a, "dp"_a, "partial"_a, "batch_per_nca"_a, "seed"_a, "update_rate"_a, "fp32"_a, "growing"_a, "stream"_a,
          "Gradients of one NCA step (same seed, update_rate and, if growing, alive mask as the forward). Writes gx, dw1, db1, dw2; packed, dp and partial "
          "are workspaces sized by backward_workspace(): dp holds all N images (one launch) or two chunks of images. Weight gradients are reduced in a fixed order.");
    m.def("update_mask", &update_mask, "mask"_a, "seed"_a, "update_rate"_a,
          "The stochastic update mask used by forward / backward: mask (N, H, W) = floor(U + update_rate).");
}
