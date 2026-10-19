#ifndef MNN_SAMPLER_C4_BILINEAR_RVV_HPP
#define MNN_SAMPLER_C4_BILINEAR_RVV_HPP

#include <stddef.h>
#include <stdint.h>

namespace MNN {
namespace CV {
namespace RVV {

static inline bool canUseC4BilinearIndexedLoad(const unsigned char* source, size_t iw, size_t ih, size_t yStride) {
    if (iw == 0 || ih == 0) {
        return false;
    }

    constexpr size_t kPixelBytes = sizeof(uint32_t);
    constexpr size_t kAlignmentMask = kPixelBytes - 1;
    if ((reinterpret_cast<uintptr_t>(source) & kAlignmentMask) != 0 || (yStride & kAlignmentMask) != 0) {
        return false;
    }

    const size_t maxIndexedOffset = static_cast<size_t>(UINT32_MAX);
    const size_t maxColumn = iw - 1;
    const size_t maxRow = ih - 1;
    if (maxColumn > maxIndexedOffset / kPixelBytes) {
        return false;
    }

    const size_t maxColumnOffset = maxColumn * kPixelBytes;
    if (maxRow != 0 && yStride > (maxIndexedOffset - maxColumnOffset) / maxRow) {
        return false;
    }
    return true;
}

} // namespace RVV
} // namespace CV
} // namespace MNN

#endif // MNN_SAMPLER_C4_BILINEAR_RVV_HPP
