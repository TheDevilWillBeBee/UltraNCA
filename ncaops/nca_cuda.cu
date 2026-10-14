#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <nanobind/stl/pair.h>
#include "nca_step.cuh"

namespace nb = nanobind;
using namespace nb::literals;

using FloatArray2D = nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cuda>;
using FloatArray3D = nb::ndarray<float, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cuda>;
using FloatArray1D = nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;
using FloatArray4D = nb::ndarray<float, nb::shape<-1, -1, -1, -1>, nb::c_contig, nb::device::cuda>;
using ByteArray1D = nb::ndarray<uint8_t, nb::shape<-1>, nb::c_contig, nb::device::cuda>;

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
template <int C_, int HID_, bool FP32_>
struct Inst
{
    static constexpr int C = C_, HID = HID_;
    static constexpr bool FP32 = FP32_;
};

// (channels, hidden) instantiations; add a line here for another NCA size.
template <typename F>
auto dispatch(size_t C, size_t HID, bool fp32, F &&f)
{
    if (C == 12 && HID == 96)
        return fp32 ? f(Inst<12, 96, true>{}) : f(Inst<12, 96, false>{});
    throw std::runtime_error("no kernel instantiated for channels=" + std::to_string(C) + ", hidden=" + std::to_string(HID));
}

template <typename Kernel>
void set_smem(Kernel kernel, int bytes)
{
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

void forward(FloatArray4D x, FloatArray3D w1, FloatArray2D b1, FloatArray3D w2, FloatArray4D y,
             uint32_t batch_per_nca, uint64_t seed, float update_rate, bool fp32)
{
    const size_t N = x.shape(0), C = x.shape(1), H = x.shape(2), W = x.shape(3), K = w1.shape(0), HID = w1.shape(1);
    if (N != K * batch_per_nca)
        throw std::runtime_error("x.shape[0] must equal (number of NCAs) * batch_per_nca");
    validate_array_shape(w1, {K, HID, 4 * C}, "w1");
    validate_array_shape(b1, {K, HID}, "b1");
    validate_array_shape(w2, {K, C, HID}, "w2");
    validate_array_shape(y, {N, C, H, W}, "y");
    dispatch(C, HID, fp32, [&](auto I) {
        using D = decltype(I);
        auto kernel = nca_step::forward_kernel<D::C, D::HID, D::FP32>;
        constexpr int smem = sizeof(nca_step::ForwardSmem<D::C, D::HID, D::FP32>);
        set_smem(kernel, smem);
        const int tiles = ceil_div(W, nca_step::TILE_W) * ceil_div(H, nca_step::TILE_H);
        kernel<<<dim3(ceil_div(tiles, nca_step::TILES_PER_BLOCK), N), nca_step::THREADS, smem>>>(
            x.data(), w1.data(), b1.data(), w2.data(), y.data(), batch_per_nca, H, W, seed, update_rate);
        cuda_check(cudaGetLastError());
    });
}

static int backward_blocks_per_image(int H, int W)
{
    const int tiles = ceil_div(W, nca_step::TILE_W) * ceil_div(H, nca_step::BWD_TILE_H);
    return ceil_div(tiles, nca_step::BWD_TILES_PER_BLOCK);
}

// Workspace sizes for backward(): (packed weight bytes per NCA, partial floats per image).
std::pair<size_t, size_t> backward_workspace(size_t C, size_t HID, bool fp32, size_t H, size_t W)
{
    return dispatch(C, HID, fp32, [&](auto I) {
        using D = decltype(I);
        return std::make_pair(sizeof(nca_step::BackwardWeights<D::C, D::HID, D::FP32>),
                              (size_t)backward_blocks_per_image(H, W) * nca_step::BwdShape<D::C, D::HID>::PARTIAL);
    });
}

void backward(FloatArray4D x, FloatArray4D gy, FloatArray3D w1, FloatArray2D b1, FloatArray3D w2,
              FloatArray4D gx, FloatArray3D dw1, FloatArray2D db1, FloatArray3D dw2,
              ByteArray1D packed, FloatArray4D dp, FloatArray1D partial,
              uint32_t batch_per_nca, uint64_t seed, float update_rate, bool fp32)
{
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
    validate_array_shape(dp, {N, 4 * C, H, W}, "dp");
    const auto [packed_bytes, partial_floats] = backward_workspace(C, HID, fp32, H, W);
    validate_array_shape(packed, {K * packed_bytes}, "packed");
    validate_array_shape(partial, {N * partial_floats}, "partial");
    dispatch(C, HID, fp32, [&](auto I) {
        using D = decltype(I);
        using BW = nca_step::BackwardWeights<D::C, D::HID, D::FP32>;
        using BS = nca_step::BwdShape<D::C, D::HID>;
        BW *pk = reinterpret_cast<BW *>(packed.data());
        nca_step::pack_kernel<D::C, D::HID, D::FP32><<<K, 256>>>(w1.data(), b1.data(), w2.data(), pk);
        cuda_check(cudaGetLastError());

        auto kernel = nca_step::backward_kernel<D::C, D::HID, D::FP32>;
        constexpr int smem = sizeof(nca_step::BackwardSmem<D::C, D::HID>);
        set_smem(kernel, smem);
        const int blocks = backward_blocks_per_image(H, W);
        kernel<<<dim3(blocks, N), nca_step::THREADS, smem>>>(x.data(), gy.data(), pk, dp.data(), partial.data(),
                                                            batch_per_nca, H, W, seed, update_rate);
        cuda_check(cudaGetLastError());

        nca_step::reduce_kernel<D::C, D::HID><<<dim3(ceil_div(BS::PARTIAL, 256), K), 256>>>(
            partial.data(), batch_per_nca * blocks, dw1.data(), db1.data(), dw2.data());
        cuda_check(cudaGetLastError());

        const uint64_t total = (uint64_t)N * D::C * H * W;
        nca_step::adjoint_kernel<D::C><<<(unsigned)((total + 255) / 256), 256>>>(dp.data(), gy.data(), gx.data(), H, W, total);
        cuda_check(cudaGetLastError());
    });
}

void update_mask(FloatArray3D mask, uint64_t seed, float update_rate)
{
    const uint64_t total = mask.shape(0) * mask.shape(1) * mask.shape(2);
    nca_step::mask_kernel<<<(unsigned)std::min<uint64_t>((total + 255) / 256, 65535), 256>>>(mask.data(), total, seed, update_rate);
    cuda_check(cudaGetLastError());
}

NB_MODULE(nca_cuda, m)
{
    m.def("forward", &forward, "x"_a, "w1"_a, "b1"_a, "w2"_a, "y"_a, "batch_per_nca"_a, "seed"_a, "update_rate"_a, "fp32"_a,
          "One NCA step. x, y: (N, C, H, W) with N = K * batch_per_nca; w1: (K, hidden, 4C); b1: (K, hidden); "
          "w2: (K, C, hidden). fp32=False uses TF32 tensor cores, fp32=True 3xTF32 (fp32 accuracy).");
    m.def("backward_workspace", &backward_workspace, "channels"_a, "hidden"_a, "fp32"_a, "H"_a, "W"_a,
          "(packed weight bytes per NCA, partial floats per image) for backward().");
    m.def("backward", &backward, "x"_a, "gy"_a, "w1"_a, "b1"_a, "w2"_a, "gx"_a, "dw1"_a, "db1"_a, "dw2"_a,
          "packed"_a, "dp"_a, "partial"_a, "batch_per_nca"_a, "seed"_a, "update_rate"_a, "fp32"_a,
          "Gradients of one NCA step (same seed as the forward). Writes gx, dw1, db1, dw2; packed, dp (N, 4C, H, W) "
          "and partial are workspaces sized by backward_workspace(). Weight gradients are reduced in a fixed order.");
    m.def("update_mask", &update_mask, "mask"_a, "seed"_a, "update_rate"_a,
          "The stochastic update mask used by forward / backward: mask (N, H, W) = floor(U + update_rate).");
}
