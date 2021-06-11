//
//  GemmCommonBF16.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"

void _AVX_MNNPackForMatMul_B_BF16(float* destF, const float* sourceF, size_t h, size_t l, bool transpose) {
    auto dest = (int16_t*)destF;
    auto source = (const int16_t*)sourceF;
    auto lC8 = UP_DIV(l, 8);
    auto hC4 = UP_DIV(h, 4);
    int sYstride = 1;
    int sXstride = h;
    if (transpose) {
        sYstride = l;
        sXstride = 1;
    }
    ::memset(dest, 0, lC8 * hC4 * sizeof(int16_t) * 32);
    for (int y = 0; y < h; ++y) {
        int yC = y / 4;
        int yR = y % 4;
        for (int x = 0; x < l; ++x) {
            int xC = x / 8;
            int xR = x % 8;
            dest[xR + yR * 8 + xC * 32 + yC * 32 * lC8] = source[sXstride * x + sYstride * y];
        }
    }
}

void _AVX_MNNGetMatMulPackMode_BF16(int* eP, int *lP, int* hP) {
    *eP = 3;
    *lP = 8;
    *hP = 4;
}

void _AVX_MNNPackC4ForMatMul_A_BF16(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    int pOffset = 4 * offset;
    if (1 == number) {
        int l = el[1];
        if (l % 8 != 0) {
            auto lAigin = UP_DIV(l, 8) * 8;
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
        auto lOC = lOffset / 8;
        auto lOR = lOffset % 8;
        auto source = (int16_t*)(sourceGroup[n]);
        auto dest = ((int16_t*)destOrigin) + eOffset * 8 + lOC * eDest * 8;
        if (lOR == 0) {
            // Fast way
            int alignLC4 = UP_DIV(l, 4);
            int lC8 = alignLC4 / 2;
            int lC8R = alignLC4 % 2;
            for (int x=0; x<lC8; ++x) {
                auto destX = (int64_t*)(dest + x * eDest * 8);
                auto srcX0 = (int64_t*)(source + (2 * x + 0) * eReal * 4);
                auto srcX1 = (int64_t*)(source + (2 * x + 1) * eReal * 4);

                for (int y=0; y<e; ++y) {
                    destX[2*y+0] = srcX0[y*offset];
                    destX[2*y+1] = srcX1[y*offset];
                }
            }
            if (lC8R > 0) {
                auto destX = (int64_t*)(dest + lC8 * eDest * 8);
                auto srcX0 = (int64_t*)(source + (2 * lC8 + 0) * eReal * 4);

                for (int y=0; y<e; ++y) {
                    destX[2*y+0] = srcX0[y*offset];
                }
            }
            continue;
        }
        for (int x=0; x<l; ++x) {
            auto dl = lOR + x;
            auto dlC = dl / 8;
            auto dlR = dl % 8;
            auto xC = x / 4;
            auto xR = x % 4;
            auto destX = dest + dlC * eDest * 8 + dlR;
            auto srcX = source + xC * eReal * 4 + xR;
            for (int y=0; y<e; ++y) {
                destX[y*8] = srcX[y*4*offset];
            }
        }
    }
}
