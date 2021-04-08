#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#include "./FunctionSummary.hpp"
extern "C" {
void MNNTranspose32Bit4x4(int32_t* dstO, const int32_t* srcO, int32_t* dim);
}
void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    auto wC4 = w / 4;
    auto hC4 = h / 4;
    int srcStride = dim[2];
    int dstStride = dim[3];
    if (wC4 > 0 && hC4 > 0) {
        MNNTranspose32Bit4x4(dstO, srcO, dim);
    }
    // Down
    for (int i=hC4 * 4; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
    // Right
    for (int i=0; i<hC4 * 4; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=wC4 * 4; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 12;
    *lP = 1;
#ifdef __aarch64__
    *hP = 8;
#else
    *hP = 4;
#endif
}

#ifdef __aarch64__

// input shape is (l, h) when transpose=false, else input shape is (h, l)
// output shape is (UP_DIV(h, 8), l, 8)
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    auto hP = (int)h / 8;
    auto hR = (int)hP * 8;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 8)*8*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 8 * l;
            auto sourceY = source + y * 8;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, 8 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 8 * l;
            auto sourceY = source + hP * 8;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int lC8 = (int)l / 8;
    auto lR = lC8 * 8;
    if (hP > 0 && lC8 > 0) {
        MNNPackC8(dest, source, l, h);
    }
    for (int y=hR; y<h; ++y) {
        auto yR = y % 8;
        auto yC = hP;
        for (int x=0; x<l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
    for (int y=0; y<hR; ++y) {
        auto yR = y % 8;
        auto yC = y / 8;
        for (int x=lR; x<l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
}
#else
void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    if (!transpose) {
        auto hP = h / 4;
        auto hR = hP * 4;
        if (hR != h) {
            ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(float));
        }
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    MNNPackC4(dest, source, l, h);
}
#endif


#endif
