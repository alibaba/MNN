

#if defined(MNN_SUPPORT_BF16) // CmakeList.txt does not work for ios, this file has to be self-filted, MNN.podspec doesnot filter this.

#include "core/Macro.h"
#include "../compute/CommonOptFunction.h"
#include "./FunctionSummary.hpp"

// todo: search for proper value for bf16
void NEON_MNNGetMatMulPackMode_BF16(int* eP, int* lP, int* hP) {
    *eP = 12;
    *lP = 1;
#ifdef __aarch64__
    *hP = 8;
#else
    *hP = 4;
#endif
}

#ifdef __aarch64__
void NEON_MNNPackForMatMul_B_BF16(float* destFloat, const float* sourceFloat, size_t h, size_t l, bool transpose) {
    auto hP         = (int)h / 8;
    auto hR         = (int)hP * 8;
    int16_t* dest   = (int16_t*)destFloat;
    int16_t* source = (int16_t*)sourceFloat;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 8) * 8 * l * sizeof(int16_t));
    }
    if (!transpose) {
        for (int y = 0; y < hP; ++y) {
            auto destY   = dest + y * 8 * l;
            auto sourceY = source + y * 8;
            for (int x = 0; x < l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, 8 * sizeof(int16_t));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY   = dest + hP * 8 * l;
            auto sourceY = source + hP * 8;
            for (int x = 0; x < l; ++x) {
                ::memcpy(destY + 8 * x, sourceY + x * h, hRemain * sizeof(int16_t));
            }
        }
        return;
    }
    int lC8 = (int)l / 8;
    auto lR = lC8 * 8;
    if (hP > 0 && lC8 > 0) {
        MNNPackC8_BF16(destFloat, sourceFloat, l, h);
    }
    for (int y = hR; y < h; ++y) {
        auto yR = y % 8;
        auto yC = hP;
        for (int x = 0; x < l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }
    for (int y = 0; y < hR; ++y) {
        auto yR = y % 8;
        auto yC = y / 8;
        for (int x = lR; x < l; ++x) {
            dest[x * 8 + yR + yC * 8 * l] = source[x + y * l];
        }
    }


}

#else
void NEON_MNNPackForMatMul_B_BF16(float* destFloat, const float* sourceFloat, size_t h, size_t l, bool transpose) {
    int16_t* dest   = (int16_t*)destFloat;
    int16_t* source = (int16_t*)sourceFloat;
    if (!transpose) {
        auto hP = h / 4;
        auto hR = hP * 4;
        if (hR != h) {
            ::memset(dest, 0, UP_DIV(h, 4) * 4 * l * sizeof(int16_t));
        }
        for (int y = 0; y < hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x = 0; x < l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(int16_t));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x = 0; x < l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(int16_t));
            }
        }
        return;
    }
    int offset[2] = {
        (int)l,
        (int)l,
    };
    MNNPackC4_BF16(destFloat, sourceFloat, l, h, offset);
}
#endif
#endif // MNN_SUPPORT_BF16

