#pragma once

// Keep CUTLASS's source loads, fragment ordering, and coordinate advancement. Only destination stores change:
// each pooled output writes its disjoint 2x2 window, preserving the rounded fp16 gradient and argmax bits.
template <typename Base>
class UnpoolOutputIterator : public Base {
public:
    using ThreadMap = typename Base::ThreadMap;
    using Element = typename Base::Element;
    using TensorCoord = typename Base::TensorCoord;
    using Fragment = typename Base::Fragment;
    struct Params : Base::Params {
        using Base::Params::Params;
        uint8_t const *argmax = nullptr;
        int width = 0;
        CUTLASS_HOST_DEVICE Params() {}
    };

private:
    Element *out_;
    uint8_t const *argmax_;
    int width_;

public:
    CUTLASS_DEVICE
    UnpoolOutputIterator(Params const &p, Element *ptr, TensorCoord extent, int thread,
                         TensorCoord offset = TensorCoord(), int const *indices = nullptr)
        : Base(p, ptr, extent, thread, offset, indices), out_(ptr), argmax_(p.argmax), width_(p.width) {}

    CUTLASS_DEVICE
    void store(Fragment const &frag) const {
        static_assert(Base::kElementsPerAccess == 8, "unpool stores eight fp16 channels per access");
        using Access = cutlass::AlignedArray<Element, 8>;
        auto const *values = reinterpret_cast<Access const *>(&frag);
        typename Base::Mask mask;
        this->get_mask(mask);
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
            CUTLASS_PRAGMA_UNROLL
            for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
                CUTLASS_PRAGMA_UNROLL
                for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
                    int fr = row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster);
                    int r = this->thread_start_row() + row * ThreadMap::Delta::kRow
                        + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
                    CUTLASS_PRAGMA_UNROLL
                    for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                        if (r < this->extent_row() && mask.predicates[column]) {
                            int c = this->thread_start_column() + column * ThreadMap::Delta::kColumn;
                            int C = this->extent_column();
                            uint2 idx = *reinterpret_cast<uint2 const *>(argmax_ + int64_t(r) * C + c);
                            Access const &v = values[fr * ThreadMap::Iterations::kColumn + column];
                            int64_t dst = (int64_t(r / width_) * 4 * width_ + 2 * (r % width_)) * C + c;
                            CUTLASS_PRAGMA_UNROLL
                            for (int k = 0; k < 4; ++k) {
                                Access result;
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < 8; ++i) {
                                    int arg = ((i < 4 ? idx.x : idx.y) >> (8 * (i & 3))) & 0xff;
                                    result[i] = arg == k ? v[i] : Element(0);
                                }
                                *reinterpret_cast<Access *>(out_ + dst + ((k >> 1) * 2 * width_ + (k & 1)) * C) = result;
                            }
                        }
                    }
                }
            }
        }
    }
};

template <typename T, typename Acc>
struct DgradUnpool {
    using Original = typename DgradMaskOp<T, Acc>::UnderlyingKernel;
    using E = typename Original::Epilogue;
    using Epilogue = cutlass::epilogue::threadblock::Epilogue<
        typename E::Shape, typename E::WarpMmaOperator, E::kPartitionsK,
        UnpoolOutputIterator<typename E::OutputTileIterator>, typename E::AccumulatorFragmentIterator,
        typename E::WarpTileIterator, typename E::SharedLoadIterator, typename E::OutputOp, typename E::Padding>;
    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
        typename Original::Mma, Epilogue, typename Original::ThreadblockSwizzle, cutlass::conv::Operator::kDgrad>;
};
