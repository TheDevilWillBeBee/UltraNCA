#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "lppn_decoder.cuh"

namespace nb = nanobind;
using namespace nb::literals;

using FloatArray = nb::ndarray<float, nb::c_contig, nb::device::cuda>;

#define cuda_check(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cudaError_t error = call;                                                                               \
        if (error != cudaSuccess)                                                                               \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                                     " - " + cudaGetErrorString(error));                                        \
    } while (0)

static void check_shape(const FloatArray &a, const std::vector<size_t> &expected, const char *name)
{
    bool ok = a.ndim() == expected.size();
    for (size_t i = 0; ok && i < expected.size(); ++i)
        ok = a.shape(i) == expected[i];
    if (!ok)
    {
        std::string s = std::string(name) + " must have shape (";
        for (size_t i = 0; i < expected.size(); ++i)
            s += (i ? ", " : "") + std::to_string(expected[i]);
        s += "), got (";
        for (size_t i = 0; i < a.ndim(); ++i)
            s += (i ? ", " : "") + std::to_string(a.shape(i));
        throw std::runtime_error(s + ")");
    }
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

static int sm_count()
{
    static int n = [] { int dev, v; cudaGetDevice(&dev); cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev); return v; }();
    return n;
}

// (S, out channels, hidden width) instantiations, each with and without the local ensemble; add a line here for others
// (S = 4 or 8 for the backward's rounds, out <= 8, hidden a multiple of 8 up to 32).
template <typename F>
auto dispatch(size_t S, size_t O, bool ens, size_t HID, F &&f)
{
    auto with_ens = [&](auto S_, auto O_, auto H_) {
        return ens ? f(S_, O_, std::true_type{}, H_) : f(S_, O_, std::false_type{}, H_);
    };
    auto with_hid = [&](auto S_, auto O_) {
        if (HID == 32)
            return with_ens(S_, O_, std::integral_constant<int, 32>{});
        if (HID == 24)
            return with_ens(S_, O_, std::integral_constant<int, 24>{});
        throw std::runtime_error("no LPPN kernel instantiated for hidden=" + std::to_string(HID) + " (32, 24)");
    };
    if (S == 8 && O == 3)
        return with_hid(std::integral_constant<int, 8>{}, std::integral_constant<int, 3>{});
    if (S == 4 && O == 3)
        return with_hid(std::integral_constant<int, 4>{}, std::integral_constant<int, 3>{});
    throw std::runtime_error("no LPPN kernel instantiated for scale=" + std::to_string(S) + ", out=" + std::to_string(O));
}

struct Dims {
    size_t N, h, w, S, O, HID;
    bool ens;  // local ensemble: T (N, 4, S, S, hidden)
};

static Dims validate(const FloatArray &G, const FloatArray &T, const FloatArray &V, const FloatArray &c, const FloatArray &wo)
{
    if (G.ndim() != 4 || (T.ndim() != 4 && T.ndim() != 5) || wo.ndim() != 3)
        throw std::runtime_error("G (N, h, w, hidden), T (N, S, S, hidden) or (N, 4, S, S, hidden), wo (N, out, hidden) expected");
    const bool ens = T.ndim() == 5;
    const Dims d{G.shape(0), G.shape(1), G.shape(2), T.shape(ens ? 2 : 1), wo.shape(1), G.shape(3), ens};
    if (ens)
        check_shape(T, {d.N, 4, d.S, d.S, d.HID}, "T");
    else
        check_shape(T, {d.N, d.S, d.S, d.HID}, "T");
    check_shape(V, {d.N, (size_t)lppn::LAYERS, d.HID, d.HID}, "V");
    check_shape(c, {d.N, (size_t)lppn::LAYERS, d.HID}, "c");
    check_shape(wo, {d.N, d.O, d.HID}, "wo");
    if (d.h % lppn::BAND || d.w < 2)
        throw std::runtime_error("the grid height must be a multiple of " + std::to_string(lppn::BAND) + " and its width at least 2");
    return d;
}

// the per-image sample offsets (N, 2), or nullptr for an empty array
static const float *offsets(const FloatArray &offset, const Dims &d)
{
    if (offset.size() == 0)
        return nullptr;
    check_shape(offset, {d.N, 2}, "offset");
    return offset.data();
}

void forward(FloatArray G, FloatArray T, FloatArray V, FloatArray c, FloatArray wo, FloatArray bo, FloatArray offset,
             FloatArray out, uintptr_t stream)
{
    const Dims d = validate(G, T, V, c, wo);
    check_shape(bo, {d.N, d.O}, "bo");
    check_shape(out, {d.N, d.O, d.S * d.h, d.S * d.w}, "out");
    dispatch(d.S, d.O, d.ens, d.HID, [&](auto S, auto O, auto E, auto H) {
        auto kernel = lppn::forward_kernel<decltype(S)::value, decltype(O)::value, decltype(E)::value, decltype(H)::value>;
        static const int blocks_per_sm = [&] { int b; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, kernel, lppn::THREADS, 0); return b; }();
        // column segments: at least two waves of blocks
        const int bands = d.h / lppn::BAND, slots = blocks_per_sm * sm_count();
        const int segments = std::max(1, std::min<int>(d.w / 4, ceil_div(2 * slots, (int)(d.N * bands))));
        kernel<<<dim3(bands * segments, d.N), lppn::THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            G.data(), T.data(), V.data(), c.data(), wo.data(), bo.data(), offsets(offset, d), out.data(), d.h, d.w, segments);
        cuda_check(cudaGetLastError());
    });
}

// (floats per image of the reduced result: dV, dc, dwo, dbo (padded to 8), dT; bands = blocks per image)
std::tuple<size_t, size_t> workspace(size_t S, size_t O, size_t h, bool ensemble, size_t hidden)
{
    return dispatch(S, O, ensemble, hidden, [&](auto S_, auto O_, auto E_, auto H_) {
        return std::make_tuple(
            (size_t)lppn::Layout<decltype(S_)::value, decltype(O_)::value, decltype(E_)::value, decltype(H_)::value>::SIZE,
            h / lppn::BAND);
    });
}

void backward(FloatArray G, FloatArray T, FloatArray V, FloatArray c, FloatArray wo, FloatArray dout, FloatArray gscale,
              FloatArray offset, FloatArray dG, FloatArray partial, FloatArray edges, FloatArray reduced, uintptr_t stream_handle)
{
    const Dims d = validate(G, T, V, c, wo);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    check_shape(dout, {d.N, d.O, d.S * d.h, d.S * d.w}, "dout");
    check_shape(gscale, {d.N}, "gscale");
    check_shape(dG, {d.N, d.h, d.w, d.HID}, "dG");
    dispatch(d.S, d.O, d.ens, d.HID, [&](auto S, auto O, auto E, auto H) {
        constexpr int s = decltype(S)::value, o = decltype(O)::value, hid = decltype(H)::value;
        constexpr bool ens = decltype(E)::value;
        using L = lppn::Layout<s, o, ens, hid>;
        const int bands = d.h / lppn::BAND;
        check_shape(partial, {d.N, (size_t)bands, (size_t)L::SIZE}, "partial");
        if (ens)
            check_shape(edges, {d.N, 4, d.h, d.w, d.HID}, "edges (ensemble: corner gradients)");
        else
            check_shape(edges, {d.N, (size_t)bands, 2, d.w, d.HID}, "edges");
        check_shape(reduced, {d.N, (size_t)L::SIZE}, "reduced");
        lppn::backward_kernel<s, o, ens, hid><<<dim3(bands, d.N), lppn::THREADS, 0, stream>>>(
            G.data(), T.data(), V.data(), c.data(), wo.data(), dout.data(), gscale.data(), offsets(offset, d), dG.data(),
            partial.data(), edges.data(), d.h, d.w);
        cuda_check(cudaGetLastError());
        lppn::reduce_kernel<<<dim3(ceil_div(L::SIZE, 256), d.N), 256, 0, stream>>>(partial.data(), bands, L::SIZE, reduced.data());
        cuda_check(cudaGetLastError());
        if (ens)
        {
            const size_t total = d.N * d.h * d.w * hid;
            lppn::ens_gather_kernel<hid><<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(edges.data(), dG.data(), d.N, d.h, d.w);
        }
        else
        {
            const size_t total = d.N * bands * d.w * hid;
            lppn::edge_kernel<hid><<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(edges.data(), dG.data(), d.N, bands, d.h, d.w);
        }
        cuda_check(cudaGetLastError());
    });
}

NB_MODULE(lppn_cuda, m)
{
    m.attr("HIDDEN") = nb::make_tuple(32, 24);  // instantiated hidden widths
    m.attr("LAYERS") = lppn::LAYERS;
    m.def("forward", &forward, "G"_a, "T"_a, "V"_a, "c"_a, "wo"_a, "bo"_a, "offset"_a, "out"_a, "stream"_a,
          "LPPN decoder forward (lppn.py, ncaops/lppn_decoder.cuh): G (N, h, w, hidden), T (N, S, S, hidden) or, local "
          "ensemble, (N, 4, S, S, hidden), V (N, layers, hidden, hidden), c (N, layers, hidden), wo (N, out, hidden), bo "
          "(N, out); offset (N, 2): per-image sample offsets in cells, |offset| <= 1 / (2 S), or empty -> out (N, out, S h, S w).");
    m.def("workspace", &workspace, "S"_a, "out"_a, "h"_a, "ensemble"_a, "hidden"_a,
          "(floats per image of backward's reduced result, bands = backward blocks per image).");
    m.def("backward", &backward, "G"_a, "T"_a, "V"_a, "c"_a, "wo"_a, "dout"_a, "gscale"_a, "offset"_a, "dG"_a, "partial"_a, "edges"_a,
          "reduced"_a, "stream"_a,
          "Gradients of forward() for dout (N, out, S h, S w) scaled by gscale (N,) inside (results unscaled): dG (N, h, w, "
          "hidden); reduced (N, size) = [dV, dc, dwo, dbo (8), dT ((4,) S, S, hidden)]; partial (N, bands, size) and edges "
          "(N, bands, 2, w, hidden), local ensemble (N, 4, h, w, hidden), are workspaces. Deterministic.");
}
