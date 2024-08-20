//
//  GemmCommon.cpp
//  MNN
//
//  Created by MNN on 2020/09/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GemmCommon.hpp"
#include "FunctionSummary.hpp"
#include "core/Macro.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <algorithm>
#include <cmath>

void _SSE_MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int xStride = info[3];
    int xS4 = xStride * 4;
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto source = sourceGroup[n];
        auto dest = destOrigin + eOffset + lOffset * eDest;
        const int pack   = 12;
        const int mid    = 1; // Deprecate
        const int packC4 = pack / 4;
        auto ePack       = e / pack;
        auto lC4         = l / 4;
        auto lDiv        = UP_DIV(l, 4);
        auto eRemain     = ePack * pack;
        auto lRemain     = lC4 * 4;
        auto lRes        = l - lRemain;
        for (int y = 0; y < ePack; ++y) {
            auto dstY = dest + y * l * pack;
            auto srcY = source + y * pack * 4;
            for (int x = 0; x < lC4; ++x) {
                auto srcX = srcY + x * 4 * eReal;
                auto dstX = dstY + x * pack * 4;
                auto s00  = _mm_loadu_ps(srcX + 0 * xS4);
                auto s01  = _mm_loadu_ps(srcX + 1 * xS4);
                auto s02  = _mm_loadu_ps(srcX + 2 * xS4);
                auto s03  = _mm_loadu_ps(srcX + 3 * xS4);
                auto s10  = _mm_loadu_ps(srcX + 4 * xS4);
                auto s11  = _mm_loadu_ps(srcX + 5 * xS4);
                auto s12  = _mm_loadu_ps(srcX + 6 * xS4);
                auto s13  = _mm_loadu_ps(srcX + 7 * xS4);
                auto s20  = _mm_loadu_ps(srcX + 8 * xS4);
                auto s21  = _mm_loadu_ps(srcX + 9 * xS4);
                auto s22  = _mm_loadu_ps(srcX + 10 * xS4);
                auto s23  = _mm_loadu_ps(srcX + 11 * xS4);

                _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
                _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
                _MM_TRANSPOSE4_PS(s20, s21, s22, s23);

    #define STORE_TEMP(i)                               \
        _mm_storeu_ps(dstX + 4 * (3 * i + 0), s##0##i); \
        _mm_storeu_ps(dstX + 4 * (3 * i + 1), s##1##i); \
        _mm_storeu_ps(dstX + 4 * (3 * i + 2), s##2##i);

                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
                STORE_TEMP(3);
            }
            if (lRes == 0) {
                continue;
            }
            auto srcX = srcY + lC4 * 4 * eReal;
            auto dstX = dstY + lC4 * eDest * 4;
            auto s00  = _mm_loadu_ps(srcX + 0 * xS4);
            auto s01  = _mm_loadu_ps(srcX + 1 * xS4);
            auto s02  = _mm_loadu_ps(srcX + 2 * xS4);
            auto s03  = _mm_loadu_ps(srcX + 3 * xS4);
            auto s10  = _mm_loadu_ps(srcX + 4 * xS4);
            auto s11  = _mm_loadu_ps(srcX + 5 * xS4);
            auto s12  = _mm_loadu_ps(srcX + 6 * xS4);
            auto s13  = _mm_loadu_ps(srcX + 7 * xS4);
            auto s20  = _mm_loadu_ps(srcX + 8 * xS4);
            auto s21  = _mm_loadu_ps(srcX + 9 * xS4);
            auto s22  = _mm_loadu_ps(srcX + 10 * xS4);
            auto s23  = _mm_loadu_ps(srcX + 11 * xS4);

            _MM_TRANSPOSE4_PS(s00, s01, s02, s03);
            _MM_TRANSPOSE4_PS(s10, s11, s12, s13);
            _MM_TRANSPOSE4_PS(s20, s21, s22, s23);
            if (lRes == 3) {
                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
            } else if (lRes == 2) {
                STORE_TEMP(0);
                STORE_TEMP(1);
            } else {
                STORE_TEMP(0);
            }
        }
        // Down
        {
            auto eLast    = e - eRemain;
            auto lastDest = dest + ePack * pack * l;
            for (int y = eRemain; y < e; ++y) {
                auto yR = y - eRemain;
                for (int x = 0; x < l; ++x) {
                    auto xR                  = x % 4;
                    auto xC                  = x / 4;
                    lastDest[x * eDest + yR] = source[xC * eReal * 4 + y * 4 * xStride + xR];
                }
            }
        }
    }
}

void _SSE_GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                        const float* bias) {
    if (nullptr == postParameters) {
        return;
    }
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 4;
    auto hC4          = UP_DIV(h, 4);
    auto minValue     = _mm_set1_ps(postParameters[2]);
    auto maxValue     = _mm_set1_ps(postParameters[3]);
    if (nullptr != bias) {
        for (int y = 0; y < hC4; ++y) {
            auto biasValue = _mm_loadu_ps(bias + 4 * y);
            auto dst       = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst + 4 * x));
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst + 4 * x, sum);
            }
        }
    } else {
        for (int y = 0; y < hC4; ++y) {
            auto dst = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm_loadu_ps(dst + 4 * x);
                sum      = _mm_max_ps(sum, minValue);
                sum      = _mm_min_ps(sum, maxValue);
                _mm_storeu_ps(dst + 4 * x, sum);
            }
        }
    }
}

void _SSE_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    int offset[2] = {
        (int)l,
        (int)l
    };
    if (!transpose) {
        MNNUnpackTranspose(dest, source, l, h, offset);
        return;
    }
    MNNPackC4(dest, source, l, h, offset);
}

void _SSE_MNNPackForMatMul_B_BF16(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    int offset[] = {
        (int)l,
        (int)l
    };
    if (!transpose) {
        MNNUnpackTransposeInt16((int16_t*)dest, (const int16_t*)source, l, h, offset);
        return;
    }
    MNNPackC4Int16((int16_t*)dest, (const int16_t*)source, l, h, offset);
}

void _SSE_MNNPackedSparseMatMul(float* C, const float* A, const float* B, unsigned int* NNZMap, int* dataOffsetMap, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    // sse version
    MNN_ASSERT(false);
    return;
}

void _SSE_MNNComputeScaleZeroScalar(float* source, float* min, float* max, size_t size) {
    int pack = 4;
    int sizeDiv4 = size / pack;
    __m128 minVal = _mm_set1_ps(source[0]);
    __m128 maxVal = minVal;
    float maxArr[4], minArr[4];
    for (int i = 0; i < sizeDiv4; ++i) {
        auto src0 = source + pack * i;
        __m128 vecA = _mm_loadu_ps(src0);
        __m128 maskMax = _mm_cmpgt_ps(maxVal, vecA);
        __m128 maskMin = _mm_cmplt_ps(minVal, vecA);
        maxVal = _mm_blendv_ps(vecA, maxVal, maskMax);
        minVal = _mm_blendv_ps(vecA, minVal, maskMin);
    }
    _mm_storeu_ps(maxArr, maxVal);
    _mm_storeu_ps(minArr, minVal);
    float max_ = maxArr[0], min_ = minArr[0];
    for (int k = 1; k < pack; ++k) {
        if (max_ < maxArr[k]) {
            max_ = maxArr[k];
        }
        if (min_ > minArr[k]) {
            min_ = minArr[k];
        }
    }
    for (int i = pack * sizeDiv4; i < size; ++i) {
        max_ = std::max(max_, source[i]);
        min_ = std::min(min_, source[i]);
    }
    min[0] = min_;
    max[0] = max_;
}

