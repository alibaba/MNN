

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
#define EP 12
#define HP 8
#define LP 4
void ARMV86_MNNGetMatMulPackMode_BF16(int* eP, int* lP, int* hP) {
    *eP = EP;
    *hP = HP;
    *lP = LP;
}
void ARMV86_MNNPackForMatMul_B_BF16(float* destF, const float* sourceF, size_t h, size_t l, bool transpose) {
    // [l, h] -> [h/hp, l/lp, hp, lp]
    auto dest = (int16_t*)destF;
    auto source = (const int16_t*)sourceF;
    auto lCP = UP_DIV(l, LP);
    auto hCP = UP_DIV(h, HP);
    int sYstride = 1;
    int sXstride = h;
    if (transpose) {
        sYstride = l;
        sXstride = 1;
    }
    ::memset(dest, 0, lCP * hCP * sizeof(int16_t) * HP * LP);
    for (int y = 0; y < h; ++y) {
        int yC = y / HP;
        int yR = y % HP;
        for (int x = 0; x < l; ++x) {
            int xC = x / LP;
            int xR = x % LP;
            dest[xR + yR * LP + xC * HP * LP + yC * HP * LP * lCP] = source[sXstride * x + sYstride * y];
        }
    }
}
void ARMV86_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    // [l/4, e, 4] -> [l/4, ep, 4]
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    if (1 == number) {
        int l = el[1];
        if (l % 8 != 0) {
            auto lAigin = UP_DIV(l, LP) * LP;
            ::memset(destOrigin, 0, eDest * lAigin * sizeof(int16_t));
        }
    }

    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto lC4         = l / 4;
        auto lDiv        = UP_DIV(l, 4);
        auto lOC = lOffset / LP;
        auto lOR = lOffset % LP;
        auto source = (int16_t*)(sourceGroup[n]);
        auto dest = ((int16_t*)destOrigin) + eOffset * LP + lOC * eDest * LP;
        if (lOR == 0) {
            // [l/4, e, 4] -> [l/4, ep, 4]
            for (int x = 0; x < lDiv; ++x) {
                auto destX = (int64_t*)(dest + x * eDest * 4);
                auto srcX  = (int64_t*)(source + x * eReal * 4);
                for (int y = 0; y < e; ++y) {
                    destX[y] = srcX[y * offset];
                }
            }
            continue;
        }
        for (int x = 0; x < l; ++x) {
            auto dl = lOR + x;
            auto dlC = dl / LP;
            auto dlR = dl % LP;
            auto xC = x / LP;
            auto xR = x % LP;
            auto destX = dest + dlC * eDest * LP + dlR;
            auto srcX = source + xC * eReal * LP + xR;
            for (int y = 0; y < e; ++y) {
                destX[y * 4] = srcX[y * 4 * offset];
            }
        }
    }
}
#undef EP
#undef HP
#undef LP
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
#endif // __aarch64__
#endif // MNN_SUPPORT_BF16

