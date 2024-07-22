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
#include "Vec8.hpp"

void AVX2GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* bias) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto hC4          = UP_DIV(h, 4);
    auto hC8          = hC4 / 2;
    auto hR           = hC4 % 2;
    if (nullptr == postParameters) {
        if (hR > 0) {
            auto zero = _mm_set1_ps(0.0f);
            // Set Last H4 = 0
            auto dst = C + hC8 * cStride;
            for (int x = 0; x < eSize; ++x) {
                _mm_storeu_ps(dst + 8 * x + 4, zero);
            }
        }
        return;
    }
    auto minV2        = _mm256_broadcast_ss(postParameters + 2);
    auto maxV2        = _mm256_broadcast_ss(postParameters + 3);
    for (int y = 0; y < hC8; ++y) {
        auto biasValue = _mm256_loadu_ps(bias + 8 * y);
        auto dst       = C + y * cStride;
        for (int x = 0; x < eSize; ++x) {
            auto sum = _mm256_add_ps(biasValue, _mm256_loadu_ps(dst));
            sum      = _mm256_max_ps(sum, minV2);
            sum      = _mm256_min_ps(sum, maxV2);
            _mm256_storeu_ps(dst, sum);
            dst += 8;
        }
    }
    if (hR > 0) {
        auto zero = _mm_set1_ps(0.0f);
        // Set Last H4 = 0
        auto dst = C + hC8 * cStride;
        auto biasValue = _mm_loadu_ps(bias + 8 * hC8);
        auto minV1 = _mm256_extractf128_ps(minV2, 0);
        auto maxV1 = _mm256_extractf128_ps(maxV2, 0);
        for (int x = 0; x < eSize; ++x) {
            auto sum = _mm_add_ps(biasValue, _mm_loadu_ps(dst));
            sum      = _mm_max_ps(sum, minV1);
            sum      = _mm_min_ps(sum, maxV1);
            _mm_storeu_ps(dst, sum);
            _mm_storeu_ps(dst + 4, zero);
            dst += 8;
        }
    }
}
void _AVX_MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int unit = 8;
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    int pOffset = unit * offset;

    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto lC4         = l / unit;
        auto lDiv        = UP_DIV(l, unit);
        auto lRemain     = lC4 * unit;
        auto lRes        = l - lRemain;
        auto source = sourceGroup[n];
        auto dest = destOrigin + eOffset + lOffset * eDest;
#define MAIN_COMPUTE                        \
auto r00 = _mm256_loadu_ps(srcX + 0 * pOffset);  \
auto r01 = _mm256_loadu_ps(srcX + 1 * pOffset);  \
auto r02 = _mm256_loadu_ps(srcX + 2 * pOffset);  \
auto r03 = _mm256_loadu_ps(srcX + 3 * pOffset);  \
auto r04 = _mm256_loadu_ps(srcX + 4 * pOffset);  \
auto r05 = _mm256_loadu_ps(srcX + 5 * pOffset);  \
auto r06 = _mm256_loadu_ps(srcX + 6 * pOffset);  \
auto r07 = _mm256_loadu_ps(srcX + 7 * pOffset);  \
auto r10 = _mm256_loadu_ps(srcX + 8 * pOffset);  \
auto r11 = _mm256_loadu_ps(srcX + 9 * pOffset);  \
auto r12 = _mm256_loadu_ps(srcX + 10 * pOffset);  \
auto r13 = _mm256_loadu_ps(srcX + 11 * pOffset);  \
auto r14 = _mm256_loadu_ps(srcX + 12 * pOffset);  \
auto r15 = _mm256_loadu_ps(srcX + 13 * pOffset);  \
auto r16 = _mm256_loadu_ps(srcX + 14 * pOffset);  \
auto r17 = _mm256_loadu_ps(srcX + 15 * pOffset);  \
auto r20 = _mm256_loadu_ps(srcX + 16 * pOffset);  \
auto r21 = _mm256_loadu_ps(srcX + 17 * pOffset);  \
auto r22 = _mm256_loadu_ps(srcX + 18 * pOffset);  \
auto r23 = _mm256_loadu_ps(srcX + 19 * pOffset);  \
auto r24 = _mm256_loadu_ps(srcX + 20 * pOffset);  \
auto r25 = _mm256_loadu_ps(srcX + 21 * pOffset);  \
auto r26 = _mm256_loadu_ps(srcX + 22 * pOffset);  \
auto r27 = _mm256_loadu_ps(srcX + 23 * pOffset);  \
TRANSPOSE_8x8_REPLACE(r00, r01, r02, r03, r04, r05, r06, r07);\
TRANSPOSE_8x8_REPLACE(r10, r11, r12, r13, r14, r15, r16, r17);\
TRANSPOSE_8x8_REPLACE(r20, r21, r22, r23, r24, r25, r26, r27);\

#define STORE_TEMP(i)                               \
_mm256_storeu_ps(dstX + 24 * i + 0 * 8, r0##i);\
_mm256_storeu_ps(dstX + 24 * i + 1 * 8, r1##i);\
_mm256_storeu_ps(dstX + 24 * i + 2 * 8, r2##i);\

        const int pack   = 24;
        MNN_ASSERT(e <= pack);
        if (e == pack) {
            for (int x = 0; x < lC4; ++x) {
                auto srcX = source + x * unit * eReal;
                auto dstX = dest + x * eDest * unit;
                
                MAIN_COMPUTE;

                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
                STORE_TEMP(3);
                STORE_TEMP(4);
                STORE_TEMP(5);
                STORE_TEMP(6);
                STORE_TEMP(7);
            }
            if (lRes > 4) {
                auto lastLc4Src = source + lC4 * unit * eReal;
                auto lastLc4Dst = dest + lC4 * eDest * unit;
                auto dstX = lastLc4Dst;
                auto srcX = lastLc4Src;
                MAIN_COMPUTE;
                switch (lRes) {
                    case 7:
                        STORE_TEMP(6);
                    case 6:
                        STORE_TEMP(5);
                    case 5:
                        STORE_TEMP(4);
                        STORE_TEMP(3);
                        STORE_TEMP(2);
                        STORE_TEMP(1);
                        STORE_TEMP(0);
                    default:
                        break;
                }
            } else if (lRes > 0) {
                auto lastLc4Src = source + lC4 * unit * eReal;
                auto lastLc4Dst = dest + lC4 * eDest * unit;
                auto dstX = lastLc4Dst;
                auto srcX = lastLc4Src;
                auto r00 = _mm_loadu_ps(srcX + 0 * pOffset);
                auto r01 = _mm_loadu_ps(srcX + 1 * pOffset);
                auto r02 = _mm_loadu_ps(srcX + 2 * pOffset);
                auto r03 = _mm_loadu_ps(srcX + 3 * pOffset);
                auto r10 = _mm_loadu_ps(srcX + 4 * pOffset);
                auto r11 = _mm_loadu_ps(srcX + 5 * pOffset);
                auto r12 = _mm_loadu_ps(srcX + 6 * pOffset);
                auto r13 = _mm_loadu_ps(srcX + 7 * pOffset);
                auto r20 = _mm_loadu_ps(srcX + 8 * pOffset);
                auto r21 = _mm_loadu_ps(srcX + 9 * pOffset);
                auto r22 = _mm_loadu_ps(srcX + 10 * pOffset);
                auto r23 = _mm_loadu_ps(srcX + 11 * pOffset);
                auto r30 = _mm_loadu_ps(srcX + 12 * pOffset);
                auto r31 = _mm_loadu_ps(srcX + 13 * pOffset);
                auto r32 = _mm_loadu_ps(srcX + 14 * pOffset);
                auto r33 = _mm_loadu_ps(srcX + 15 * pOffset);
                auto r40 = _mm_loadu_ps(srcX + 16 * pOffset);
                auto r41 = _mm_loadu_ps(srcX + 17 * pOffset);
                auto r42 = _mm_loadu_ps(srcX + 18 * pOffset);
                auto r43 = _mm_loadu_ps(srcX + 19 * pOffset);
                auto r50 = _mm_loadu_ps(srcX + 20 * pOffset);
                auto r51 = _mm_loadu_ps(srcX + 21 * pOffset);
                auto r52 = _mm_loadu_ps(srcX + 22 * pOffset);
                auto r53 = _mm_loadu_ps(srcX + 23 * pOffset);
                
                _MM_TRANSPOSE4_PS(r00, r01, r02, r03);
                _MM_TRANSPOSE4_PS(r10, r11, r12, r13);
                _MM_TRANSPOSE4_PS(r20, r21, r22, r23);
                _MM_TRANSPOSE4_PS(r30, r31, r32, r33);
                _MM_TRANSPOSE4_PS(r40, r41, r42, r43);
                _MM_TRANSPOSE4_PS(r50, r51, r52, r53);

#define STORE_TEMP_TEMP(i)                               \
_mm_storeu_ps(dstX + 24 * i + 0 * 4, r0##i);\
_mm_storeu_ps(dstX + 24 * i + 1 * 4, r1##i);\
_mm_storeu_ps(dstX + 24 * i + 2 * 4, r2##i);\
_mm_storeu_ps(dstX + 24 * i + 3 * 4, r3##i);\
_mm_storeu_ps(dstX + 24 * i + 4 * 4, r4##i);\
_mm_storeu_ps(dstX + 24 * i + 5 * 4, r5##i);\

                switch (lRes) {
                    case 4:
                        STORE_TEMP_TEMP(3);
                    case 3:
                        STORE_TEMP_TEMP(2);
                    case 2:
                        STORE_TEMP_TEMP(1);
                    case 1:
                        STORE_TEMP_TEMP(0);
                    default:
                        break;
                }
#undef STORE_TEMP_TEMP
            }
        }
        // Down
        else {
            auto eRemain     = 0;
            auto eLast    = e - eRemain;
            auto lastDest = dest;
            for (int xC = 0; xC < lC4; ++xC) {
                for (int y = 0; y < e; ++y) {
                    auto yR = y - eRemain;
                    for (int xR = 0; xR < unit; ++xR) {
                        lastDest[(xC * unit + xR) * eDest + yR] = source[xC * eReal * unit + y * unit * offset + xR];
                    }
                }
            }
            for (int x = lC4 * unit, xR = 0; x < l; ++x, ++xR) {
                for (int y = 0; y < e; ++y) {
                    auto yR                  = y - eRemain;
                    lastDest[x * eDest + yR] = source[lC4 * eReal * unit + y * unit * offset + xR];
                }
            }
        }
    }
#undef MAIN_COMPUTE
#undef STORE_TEMP
}


// C8 -> E6
void _AVX_MNNPackC4ForMatMul_A_EShort(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int unit = 8;
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    int pOffset = unit * offset;
    float temp2[64];

    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto lC4         = l / unit;
        auto lDiv        = UP_DIV(l, unit);
        auto lRemain     = lC4 * unit;
        auto lRes        = l - lRemain;
        auto source = sourceGroup[n];
        auto dest = destOrigin + eOffset + lOffset * eDest;
        __m256 t0, t1, t2, t3, t4, t5, t6, t7;
#define MAIN_COMPUTE                        \
auto r0 = _mm256_loadu_ps(srcX + 0 * pOffset);  \
auto r1 = _mm256_loadu_ps(srcX + 1 * pOffset);  \
auto r2 = _mm256_loadu_ps(srcX + 2 * pOffset);  \
auto r3 = _mm256_loadu_ps(srcX + 3 * pOffset);  \
auto r6 = _mm256_setzero_ps();\
auto r7 = _mm256_setzero_ps();\
auto r4 = _mm256_loadu_ps(srcX + 4 * pOffset);  \
auto r5 = _mm256_loadu_ps(srcX + 5 * pOffset);  \
TRANSPOSE_8x8;\
_mm256_storeu_ps(temp2 + 0 * 8, t0);\
_mm256_storeu_ps(temp2 + 1 * 8, t1);\
_mm256_storeu_ps(temp2 + 2 * 8, t2);\
_mm256_storeu_ps(temp2 + 3 * 8, t3);\
_mm256_storeu_ps(temp2 + 4 * 8, t4);\
_mm256_storeu_ps(temp2 + 5 * 8, t5);\
_mm256_storeu_ps(temp2 + 6 * 8, t6);\
_mm256_storeu_ps(temp2 + 7 * 8, t7);\

#define STORE_TEMP(i)                               \
::memcpy(dstX + 6 * i, temp2 + 8 * i, 6 * sizeof(float));\

        const int pack   = 6;
        MNN_ASSERT(e <= pack);
        if (e == pack) {
            for (int x = 0; x < lC4; ++x) {
                auto srcX = source + x * unit * eReal;
                auto dstX = dest + x * eDest * unit;
                
                MAIN_COMPUTE;

                STORE_TEMP(0);
                STORE_TEMP(1);
                STORE_TEMP(2);
                STORE_TEMP(3);
                STORE_TEMP(4);
                STORE_TEMP(5);
                STORE_TEMP(6);
                STORE_TEMP(7);
            }
            if (lRes > 0) {
                auto lastLc4Src = source + lC4 * unit * eReal;
                auto lastLc4Dst = dest + lC4 * eDest * unit;
                auto dstX = lastLc4Dst;
                auto srcX = lastLc4Src;
                MAIN_COMPUTE;
                for (int i=0; i<lRes; ++i) {
                    ::memcpy(dstX + 6 * i, temp2 + 8 * i, 6 * sizeof(float));
                }
            }
        }
        // Down
        else {
            auto eRemain     = 0;
            auto eLast    = e - eRemain;
            auto lastDest = dest;
            for (int xC = 0; xC < lC4; ++xC) {
                for (int y = 0; y < e; ++y) {
                    auto yR = y - eRemain;
                    for (int xR = 0; xR < unit; ++xR) {
                        lastDest[(xC * unit + xR) * eDest + yR] = source[xC * eReal * unit + y * unit * offset + xR];
                    }
                }
            }
            for (int x = lC4 * unit; x < l; ++x) {
                auto xR = x % unit;
                auto xC = lC4;
                for (int y = 0; y < e; ++y) {
                    auto yR                  = y - eRemain;
                    lastDest[x * eDest + yR] = source[xC * eReal * unit + y * unit * offset + xR];
                }
            }
        }
    }
}

void _AVX_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
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

void _AVX_MNNPackForMatMul_B_EShort(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    const int unit = 16;
    auto hP = h / unit;
    auto hR = hP * unit;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, unit)*unit*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * unit * l;
            auto sourceY = source + y * unit;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + unit * x, sourceY + x * h, unit * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * unit * l;
            auto sourceY = source + hP * unit;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + unit * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    auto originDest = dest;
    int depthC4     = h / unit;
    int depthRemain = depthC4 * unit;
    int remain      = h - depthRemain;
    int z, x, y;
    int area = l;
    const float* srcChannel[16];
    const float* srcOffset = source;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 16; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < unit; ++y) {
                dest[0] = srcChannel[y][0];
                srcChannel[y]++;
                dest++;
            }
        }
        srcOffset += area * unit;
    }
    if(remain > 0){
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dest[0] = srcChannel[y][0];
                srcChannel[y]++;
                dest++;
            }
            for(y = remain; y < unit; ++y) {
                dest[0] = 0;
                dest++;
            }
        }
    }
}

void _AVX_MNNPackedSparseMatMul(float* C, const float* A, const float* B, unsigned int* NNZMap, int* dataOffsetMap, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    // sse version
    MNN_ASSERT(false);
    return;
}

void _AVX_MNNComputeScaleZeroScalar(float* source, float* min, float* max, size_t size) {
    int pack = 8;
    int sizeDiv8 = size / pack;
    __m256 minVal = _mm256_set1_ps(source[0]);
    __m256 maxVal = minVal;
    float maxArr[8], minArr[8];
    for (int i = 0; i < sizeDiv8; ++i) {
        auto src0 = source + pack * i;
        __m256 vecA = _mm256_loadu_ps(src0);
        __m256 maskMax = _mm256_cmp_ps(vecA, maxVal, 14);
        __m256 maskMin = _mm256_cmp_ps(vecA, minVal, 1);
        maxVal = _mm256_blendv_ps(maxVal, vecA, maskMax);
        minVal = _mm256_blendv_ps(minVal, vecA, maskMin);
    }
    _mm256_storeu_ps(maxArr, maxVal);
    _mm256_storeu_ps(minArr, minVal);
    float max_ = maxArr[0], min_ = minArr[0];
    for (int k = 1; k < pack; ++k) {
        if (max_ < maxArr[k]) {
            max_ = maxArr[k];
        }
        if (min_ > minArr[k]) {
            min_ = minArr[k];
        }
    }
    for (int i = pack * sizeDiv8; i < size; ++i) {
        max_ = std::max(max_, source[i]);
        min_ = std::min(min_, source[i]);
    }
    min[0] = min_;
    max[0] = max_;

}
