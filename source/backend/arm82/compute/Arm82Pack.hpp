#ifndef Arm82Pack_hpp
#define Arm82Pack_hpp

#include <cstddef>
#include <cstring>

namespace MNN {
namespace Arm82 {

template <typename T>
inline void packSmallChannelForMatMulA(T* destBase, const T* sourceBase, int e, int l, int eDest, size_t srcRowStride) {
    static_assert(sizeof(T) == 2, "The SME2 FP16 packer requires 16-bit elements");
    constexpr int lP = 2;
    const size_t dstColBlockStride = static_cast<size_t>(eDest) * lP;
    int yR = 0;
    for (int y = 0; y < e; ++y) {
        auto src = sourceBase + static_cast<size_t>(y) * srcRowStride;
        auto dst = destBase + static_cast<size_t>(yR) * lP;
        int x = 0;
        for (; x + 1 < l; x += lP) {
            std::memcpy(dst, src + x, lP * sizeof(T));
            dst += dstColBlockStride;
        }
        if (x < l) {
            *dst = src[x];
        }
        if (++yR == eDest) {
            yR = 0;
        }
    }
}

} // namespace Arm82
} // namespace MNN

#endif // Arm82Pack_hpp
