//
//  GemmCommon.cpp
//  MNN
//
//  Created by MNN on 2021/01/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FunctionSummary.hpp"
#include "Gemm48_8.hpp"
#include "Gemm10_32.h"
#include "Gemm31_16.h"
#include "Gemm9_48.h"
#include "core/Macro.h"
#include <limits>
#include <string.h>
#include <algorithm>
#include <vector>
#include "../AVX2Functions.hpp"
#include "Vec16.hpp"
//#define AVX512_TEST

#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX512_MNNGemmFloatUnit48x8(float* C, const float* A, const float* B, const size_t* parameter);
void _AVX512_MNNGemmFloatUnit48x8Fused(float* C, const float* A, const float* B, const size_t* parameter, const float* p, const float* bias);
void _AVX512_MNNGemmFloatUnit32x8(float* C, const float* A, const float* B, const size_t* parameter);
void _AVX512_MNNGemmFloatUnit16x8(float* C, const float* A, const float* B, const size_t* parameter);
}
#endif

void AVX512GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* bias) {
    auto h            = parameter[2];
    auto hC           = UP_DIV(h, 8);
    auto hR           = hC % 2;
    auto hCUnit       = hC / 2;
    auto cStride      = parameter[3] / sizeof(float);
    if (nullptr == postParameters) {
        if (hR > 0) {
            auto zero = _mm256_setzero_ps();
            // Set Last H4 = 0
            auto dst = C + hCUnit * cStride;
            for (int x = 0; x < eSize; ++x) {
                _mm256_storeu_ps(dst + 16 * x + 8, zero);
            }
        }
        return;
    }
    auto minValue        = _mm512_broadcastss_ps(_mm_load_ss(postParameters + 2));
    auto maxValue        = _mm512_broadcastss_ps(_mm_load_ss(postParameters + 3));
    for (int y = 0; y < hCUnit; ++y) {
        auto biasValue = _mm512_loadu_ps(bias + 16 * y);
        auto dst       = C + y * cStride;
        for (int x = 0; x < eSize; ++x) {
            auto sum = _mm512_add_ps(biasValue, _mm512_loadu_ps(dst));
            sum      = _mm512_max_ps(sum, minValue);
            sum      = _mm512_min_ps(sum, maxValue);
            _mm512_storeu_ps(dst, sum);
            dst += 16;
        }
    }
    if (hR > 0) {
        auto zero = _mm256_setzero_ps();
        // Set Last H4 = 0
        auto dst = C + hCUnit * cStride;
        auto biasValue = _mm256_loadu_ps(bias + 16 * hCUnit);
        auto minV1 = _mm256_broadcast_ss(postParameters + 2);
        auto maxV1 = _mm256_broadcast_ss(postParameters + 3);
        for (int x = 0; x < eSize; ++x) {
            auto sum = _mm256_add_ps(biasValue, _mm256_loadu_ps(dst));
            sum      = _mm256_max_ps(sum, minV1);
            sum      = _mm256_min_ps(sum, maxV1);
            _mm256_storeu_ps(dst, sum);
            _mm256_storeu_ps(dst + 8, zero);
            dst += 16;
        }
    }
}

#define LOAD_CASE(i, j) auto r##i##j = _mm512_loadu_ps(srcX + (i*16+j) * pOffset)
#define LOAD_GROUP(i)\
LOAD_CASE(i,0);\
LOAD_CASE(i,1);\
LOAD_CASE(i,2);\
LOAD_CASE(i,3);\
LOAD_CASE(i,4);\
LOAD_CASE(i,5);\
LOAD_CASE(i,6);\
LOAD_CASE(i,7);\
LOAD_CASE(i,8);\
LOAD_CASE(i,9);\
LOAD_CASE(i,10);\
LOAD_CASE(i,11);\
LOAD_CASE(i,12);\
LOAD_CASE(i,13);\
LOAD_CASE(i,14);\
LOAD_CASE(i,15);

#define MAIN_COMPUTE                        \
LOAD_GROUP(0);\
LOAD_GROUP(1);\
LOAD_GROUP(2);\
transpose16x16F(r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r010, r011, r012, r013, r014, r015);\
transpose16x16F(r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r110, r111, r112, r113, r114, r115);\
transpose16x16F(r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r210, r211, r212, r213, r214, r215);\

#define STORE_TEMP(i)                               \
_mm512_storeu_ps(dstX + 48 * i + 0 * 16, r0##i);\
_mm512_storeu_ps(dstX + 48 * i + 1 * 16, r1##i);\
_mm512_storeu_ps(dstX + 48 * i + 2 * 16, r2##i);

extern "C" {
void _AVX512_TransposeMain(const float* source, float* dest, const size_t* info, size_t lC4);
}

#ifndef MNN_X86_USE_ASM
void _AVX512_TransposeMain(const float* source, float* dest, const size_t* info, size_t lC4) {
    const int unit = 16;
    int srcStride = info[0] / sizeof(float);
    int dstStride = info[1] / sizeof(float);
    int pOffset = info[2] / sizeof(float);

    for (int x = 0; x < lC4; ++x) {
        auto srcX = source + x * srcStride;
        auto dstX = dest + x * dstStride;

        MAIN_COMPUTE;

        STORE_TEMP(0);
        STORE_TEMP(1);
        STORE_TEMP(2);
        STORE_TEMP(3);

        STORE_TEMP(4);
        STORE_TEMP(5);
        STORE_TEMP(6);
        STORE_TEMP(7);

        STORE_TEMP(8);
        STORE_TEMP(9);
        STORE_TEMP(10);
        STORE_TEMP(11);

        STORE_TEMP(12);
        STORE_TEMP(13);
        STORE_TEMP(14);
        STORE_TEMP(15);
    }
}

#endif

void _AVX512_MNNPackC8ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    const int unit = 16;
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    int pOffset = unit * offset;
    size_t second[3];
    second[0] = unit * eReal * sizeof(float);
    second[1] = unit * eDest * sizeof(float);
    second[2] = pOffset * sizeof(float);

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
        const int pack   = 48;
        MNN_ASSERT(e <= pack);
        if (e == pack) {
            _AVX512_TransposeMain(source, dest, second, lC4);
            auto lastLc4Src = source + lC4 * unit * eReal;
            auto lastLc4Dst = dest + lC4 * eDest * unit;
            if (lRes > 4) {
                auto dstX = lastLc4Dst;
                auto srcX = lastLc4Src;
                MAIN_COMPUTE;
                switch (lRes) {
                    case 15:
                        STORE_TEMP(14);
                    case 14:
                        STORE_TEMP(13);
                    case 13:
                        STORE_TEMP(12);
                    case 12:
                        STORE_TEMP(11);
                    case 11:
                        STORE_TEMP(10);
                    case 10:
                        STORE_TEMP(9);
                    case 9:
                        STORE_TEMP(8);
                    case 8:
                        STORE_TEMP(7);
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
            }
            else if (lRes > 0) {
#undef LOAD_CASE
#undef MAIN_COMPUTE
#undef STORE_TEMP
#define LOAD_CASE(i, j) auto r##i##j = _mm_loadu_ps(srcX + (i*16+j) * pOffset)
#define MAIN_COMPUTE                        \
LOAD_GROUP(0);\
LOAD_GROUP(1);\
LOAD_GROUP(2);\
_MM_TRANSPOSE4_PS(r00, r01, r02, r03);  _MM_TRANSPOSE4_PS(r04, r05, r06, r07);  _MM_TRANSPOSE4_PS(r08, r09, r010, r011);  _MM_TRANSPOSE4_PS(r012, r013, r014, r015); \
_MM_TRANSPOSE4_PS(r10, r11, r12, r13);  _MM_TRANSPOSE4_PS(r14, r15, r16, r17);  _MM_TRANSPOSE4_PS(r18, r19, r110, r111);  _MM_TRANSPOSE4_PS(r112, r113, r114, r115); \
_MM_TRANSPOSE4_PS(r20, r21, r22, r23);  _MM_TRANSPOSE4_PS(r24, r25, r26, r27);  _MM_TRANSPOSE4_PS(r28, r29, r210, r211);  _MM_TRANSPOSE4_PS(r212, r213, r214, r215);

#define STORE_TEMP(i)                               \
_mm_storeu_ps(dstX + 48 * i + 0 * 16, r0##i);\
_mm_storeu_ps(dstX + 48 * i + 1 * 16, r1##i);\
_mm_storeu_ps(dstX + 48 * i + 2 * 16, r2##i);\

                auto dstX = lastLc4Dst;
                auto srcX = lastLc4Src;
                MAIN_COMPUTE;
                switch (lRes) {
                    case 4:
                        _mm_storeu_ps(dstX + 48 * 3 + 0 * 4, r03);
                        _mm_storeu_ps(dstX + 48 * 3 + 1 * 4, r07);
                        _mm_storeu_ps(dstX + 48 * 3 + 2 * 4, r011);
                        _mm_storeu_ps(dstX + 48 * 3 + 3 * 4, r015);
                        _mm_storeu_ps(dstX + 48 * 3 + 4 * 4, r13);
                        _mm_storeu_ps(dstX + 48 * 3 + 5 * 4, r17);
                        _mm_storeu_ps(dstX + 48 * 3 + 6 * 4, r111);
                        _mm_storeu_ps(dstX + 48 * 3 + 7 * 4, r115);
                        _mm_storeu_ps(dstX + 48 * 3 + 8 * 4, r23);
                        _mm_storeu_ps(dstX + 48 * 3 + 9 * 4, r27);
                        _mm_storeu_ps(dstX + 48 * 3 + 10 * 4, r211);
                        _mm_storeu_ps(dstX + 48 * 3 + 11 * 4, r215);
                    case 3:
                        _mm_storeu_ps(dstX + 48 * 2 + 0 * 4, r02);
                        _mm_storeu_ps(dstX + 48 * 2 + 1 * 4, r06);
                        _mm_storeu_ps(dstX + 48 * 2 + 2 * 4, r010);
                        _mm_storeu_ps(dstX + 48 * 2 + 3 * 4, r014);
                        _mm_storeu_ps(dstX + 48 * 2 + 4 * 4, r12);
                        _mm_storeu_ps(dstX + 48 * 2 + 5 * 4, r16);
                        _mm_storeu_ps(dstX + 48 * 2 + 6 * 4, r110);
                        _mm_storeu_ps(dstX + 48 * 2 + 7 * 4, r114);
                        _mm_storeu_ps(dstX + 48 * 2 + 8 * 4, r22);
                        _mm_storeu_ps(dstX + 48 * 2 + 9 * 4, r26);
                        _mm_storeu_ps(dstX + 48 * 2 + 10 * 4, r210);
                        _mm_storeu_ps(dstX + 48 * 2 + 11 * 4, r214);
                    case 2:
                        _mm_storeu_ps(dstX + 48 * 1 + 0 * 4, r01);
                        _mm_storeu_ps(dstX + 48 * 1 + 1 * 4, r05);
                        _mm_storeu_ps(dstX + 48 * 1 + 2 * 4, r09);
                        _mm_storeu_ps(dstX + 48 * 1 + 3 * 4, r013);
                        _mm_storeu_ps(dstX + 48 * 1 + 4 * 4, r11);
                        _mm_storeu_ps(dstX + 48 * 1 + 5 * 4, r15);
                        _mm_storeu_ps(dstX + 48 * 1 + 6 * 4, r19);
                        _mm_storeu_ps(dstX + 48 * 1 + 7 * 4, r113);
                        _mm_storeu_ps(dstX + 48 * 1 + 8 * 4, r21);
                        _mm_storeu_ps(dstX + 48 * 1 + 9 * 4, r25);
                        _mm_storeu_ps(dstX + 48 * 1 + 10 * 4, r29);
                        _mm_storeu_ps(dstX + 48 * 1 + 11 * 4, r213);
                    case 1:
                        _mm_storeu_ps(dstX + 48 * 0 + 0 * 4, r00);
                        _mm_storeu_ps(dstX + 48 * 0 + 1 * 4, r04);
                        _mm_storeu_ps(dstX + 48 * 0 + 2 * 4, r08);
                        _mm_storeu_ps(dstX + 48 * 0 + 3 * 4, r012);
                        _mm_storeu_ps(dstX + 48 * 0 + 4 * 4, r10);
                        _mm_storeu_ps(dstX + 48 * 0 + 5 * 4, r14);
                        _mm_storeu_ps(dstX + 48 * 0 + 6 * 4, r18);
                        _mm_storeu_ps(dstX + 48 * 0 + 7 * 4, r112);
                        _mm_storeu_ps(dstX + 48 * 0 + 8 * 4, r20);
                        _mm_storeu_ps(dstX + 48 * 0 + 9 * 4, r24);
                        _mm_storeu_ps(dstX + 48 * 0 + 10 * 4, r28);
                        _mm_storeu_ps(dstX + 48 * 0 + 11 * 4, r212);
                    default:
                        break;
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
            for (int x = lC4 * unit, xR = 0; x < l; ++x, ++xR) {
                auto xC = lC4;
                for (int y = 0; y < e; ++y) {
                    auto yR                  = y - eRemain;
                    lastDest[x * eDest + yR] = source[xC * eReal * unit + y * unit * offset + xR];
                }
            }
        }
    }
#undef MAIN_COMPUTE
#undef STORE_TEMP

}
extern "C" {
void _AVX_MNNPackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNUnpackCUnit(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNPackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
void _AVX_MNNUnpackCUnitTranspose(float* dst, const float* src, size_t area, size_t depth, int* areaOffset);
}

void _AVX512_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    int offset[2] = {
        (int)l,
        (int)l
    };
    if (!transpose) {
        _AVX_MNNPackCUnitTranspose(dest, source, l, h, offset);
        return;
    }
    _AVX_MNNPackCUnit(dest, source, l, h, offset);
}

static void _AVX512_MNNPackedMatMul_48(float* C, const float* A, const float* B, const size_t* parameter) {
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        INIT_MAIN_48_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_48_8;
        }

        AVX512_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(2, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(3, 0, z0, z3, z6, z9, z12, z15, z18, z21);

        AVX512_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(2, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(3, 1, z1, z4, z7, z10, z13, z16, z19, z22);

        AVX512_TRANSPOSE_SAVE(0, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE(1, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE(2, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE(3, 2, z2, z5, z8, z11, z14, z17, z20, z23);
    }
}

static void _AVX512_MNNPackedMatMul_32(float* C, const float* A, const float* B, const size_t* parameter) {
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto aStride      = parameter[0] / sizeof(float);
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        INIT_MAIN_32_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_32_8;
        }

        AVX512_TRANSPOSE_SAVE(0, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(1, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(2, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE(3, 0, z0, z3, z6, z9, z12, z15, z18, z21);

        AVX512_TRANSPOSE_SAVE(0, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(1, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(2, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE(3, 1, z1, z4, z7, z10, z13, z16, z19, z22);
    }
}

static void _AVX512_MNNPackedMatMul_16(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        INIT_MAIN_16_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_16_8;
        }
        AVX512_TRANSPOSE_SAVE(0, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(1, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(2, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(3, 0, z0, z1, z2, z3, z4, z5, z6, z7);
    }
}

#define LOAD_W_4 \
W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_castps_pd (_mm256_loadu_ps(w0))), _mm256_castps_pd(_mm256_loadu_ps(w1)), 1));\
W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));

#define BROAD_CAST_S_2 \
S0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_castps_pd(_mm256_broadcast_ss(srcUse))), _mm256_castps_pd(_mm256_broadcast_ss(srcUse + aStride)), 1));\


#define SAVE_UNIT(i, j, k) _mm512_storeu_ps((C + (unit * y / 2 + j) * cStride + 16 * i), D##i##j)
static void _AVX2_MNNPackedMatMul_8(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    const int unit = 4;
    int hU = hC8 / unit;
    int hR = hU * unit;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        __m512 W0, W1;
        LOAD_W_4;
        auto srcUse = A;
        auto S0 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 0));
        auto S1 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 1));
        auto S2 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 2));
        auto S3 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 3));
        auto S4 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 4));
        auto S5 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 5));
        auto S6 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 6));
        auto S7 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 7));

        auto D00 = _mm512_mul_ps(S0, W0);
        auto D01 = _mm512_mul_ps(S0, W1);
        auto D10 = _mm512_mul_ps(S1, W0);
        auto D11 = _mm512_mul_ps(S1, W1);
        auto D20 = _mm512_mul_ps(S2, W0);
        auto D21 = _mm512_mul_ps(S2, W1);
        auto D30 = _mm512_mul_ps(S3, W0);
        auto D31 = _mm512_mul_ps(S3, W1);
        auto D40 = _mm512_mul_ps(S4, W0);
        auto D41 = _mm512_mul_ps(S4, W1);
        auto D50 = _mm512_mul_ps(S5, W0);
        auto D51 = _mm512_mul_ps(S5, W1);
        auto D60 = _mm512_mul_ps(S6, W0);
        auto D61 = _mm512_mul_ps(S6, W1);
        auto D70 = _mm512_mul_ps(S7, W0);
        auto D71 = _mm512_mul_ps(S7, W1);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        srcUse += aStride;
        for (int sy = 1; sy < l; ++sy) {
            LOAD_W_4;

            S0 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 0));
            S1 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 1));
            S2 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 2));
            S3 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 3));
            S4 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 4));
            S5 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 5));
            S6 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 6));
            S7 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 7));

            D00 = _mm512_fmadd_ps(S0, W0, D00);
            D01 = _mm512_fmadd_ps(S0, W1, D01);
            D10 = _mm512_fmadd_ps(S1, W0, D10);
            D11 = _mm512_fmadd_ps(S1, W1, D11);
            D20 = _mm512_fmadd_ps(S2, W0, D20);
            D21 = _mm512_fmadd_ps(S2, W1, D21);
            D30 = _mm512_fmadd_ps(S3, W0, D30);
            D31 = _mm512_fmadd_ps(S3, W1, D31);
            D40 = _mm512_fmadd_ps(S4, W0, D40);
            D41 = _mm512_fmadd_ps(S4, W1, D41);
            D50 = _mm512_fmadd_ps(S5, W0, D50);
            D51 = _mm512_fmadd_ps(S5, W1, D51);
            D60 = _mm512_fmadd_ps(S6, W0, D60);
            D61 = _mm512_fmadd_ps(S6, W1, D61);
            D70 = _mm512_fmadd_ps(S7, W0, D70);
            D71 = _mm512_fmadd_ps(S7, W1, D71);

            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
            srcUse += aStride;
        }
        auto dst    = C + (unit * y + 0) * cStride;

        SAVE_UNIT(0, 0, 0);
        SAVE_UNIT(1, 0, 0);
        SAVE_UNIT(2, 0, 0);
        SAVE_UNIT(3, 0, 0);
        SAVE_UNIT(4, 0, 0);
        SAVE_UNIT(5, 0, 0);
        SAVE_UNIT(6, 0, 0);
        SAVE_UNIT(7, 0, 0);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
        SAVE_UNIT(2, 1, 0);
        SAVE_UNIT(3, 1, 0);
        SAVE_UNIT(4, 1, 0);
        SAVE_UNIT(5, 1, 0);
        SAVE_UNIT(6, 1, 0);
        SAVE_UNIT(7, 1, 0);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        INIT_MAIN_8_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_8_8;
        }
        AVX2_TRANSPOSE_SAVE(0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX2_TRANSPOSE_SAVE(1, z0, z1, z2, z3, z4, z5, z6, z7);
    }
}

static void _AVX2_MNNPackedMatMul_5(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);

    const int unit = 4;
    int hU = hC8 / unit;
    int hR = hU * unit;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        __m256 W0, W1, W2, W3;
        W0 = _mm256_loadu_ps(w0);
        W1 = _mm256_loadu_ps(w1);
        W2 = _mm256_loadu_ps(w2);
        W3 = _mm256_loadu_ps(w3);
        auto srcUse = A;
        auto S0 = _mm256_broadcast_ss((srcUse + 0));
        auto S1 = _mm256_broadcast_ss((srcUse + 1));
        auto S2 = _mm256_broadcast_ss((srcUse + 2));
        auto S3 = _mm256_broadcast_ss((srcUse + 3));
        auto S4 = _mm256_broadcast_ss((srcUse + 4));

        auto D00 = _mm256_mul_ps(S0, W0);
        auto D01 = _mm256_mul_ps(S0, W1);
        auto D02 = _mm256_mul_ps(S0, W2);
        auto D03 = _mm256_mul_ps(S0, W3);
        auto D10 = _mm256_mul_ps(S1, W0);
        auto D11 = _mm256_mul_ps(S1, W1);
        auto D12 = _mm256_mul_ps(S1, W2);
        auto D13 = _mm256_mul_ps(S1, W3);
        auto D20 = _mm256_mul_ps(S2, W0);
        auto D21 = _mm256_mul_ps(S2, W1);
        auto D22 = _mm256_mul_ps(S2, W2);
        auto D23 = _mm256_mul_ps(S2, W3);
        auto D30 = _mm256_mul_ps(S3, W0);
        auto D31 = _mm256_mul_ps(S3, W1);
        auto D32 = _mm256_mul_ps(S3, W2);
        auto D33 = _mm256_mul_ps(S3, W3);
        auto D40 = _mm256_mul_ps(S4, W0);
        auto D41 = _mm256_mul_ps(S4, W1);
        auto D42 = _mm256_mul_ps(S4, W2);
        auto D43 = _mm256_mul_ps(S4, W3);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        srcUse += aStride;
        for (int sy = 1; sy < l; ++sy) {
            W0 = _mm256_loadu_ps(w0);
            W1 = _mm256_loadu_ps(w1);
            W2 = _mm256_loadu_ps(w2);
            W3 = _mm256_loadu_ps(w3);

            S0 = _mm256_broadcast_ss((srcUse + 0));
            S1 = _mm256_broadcast_ss((srcUse + 1));
            S2 = _mm256_broadcast_ss((srcUse + 2));
            S3 = _mm256_broadcast_ss((srcUse + 3));
            S4 = _mm256_broadcast_ss((srcUse + 4));

            D00 = _mm256_fmadd_ps(S0, W0, D00);
            D01 = _mm256_fmadd_ps(S0, W1, D01);
            D02 = _mm256_fmadd_ps(S0, W2, D02);
            D03 = _mm256_fmadd_ps(S0, W3, D03);

            D10 = _mm256_fmadd_ps(S1, W0, D10);
            D11 = _mm256_fmadd_ps(S1, W1, D11);
            D12 = _mm256_fmadd_ps(S1, W2, D12);
            D13 = _mm256_fmadd_ps(S1, W3, D13);

            D20 = _mm256_fmadd_ps(S2, W0, D20);
            D21 = _mm256_fmadd_ps(S2, W1, D21);
            D22 = _mm256_fmadd_ps(S2, W2, D22);
            D23 = _mm256_fmadd_ps(S2, W3, D23);

            D30 = _mm256_fmadd_ps(S3, W0, D30);
            D31 = _mm256_fmadd_ps(S3, W1, D31);
            D32 = _mm256_fmadd_ps(S3, W2, D32);
            D33 = _mm256_fmadd_ps(S3, W3, D33);

            D40 = _mm256_fmadd_ps(S4, W0, D40);
            D41 = _mm256_fmadd_ps(S4, W1, D41);
            D42 = _mm256_fmadd_ps(S4, W2, D42);
            D43 = _mm256_fmadd_ps(S4, W3, D43);

            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
            srcUse += aStride;
        }
        auto dst    = C + 2 * y * cStride;
        _mm256_storeu_ps(dst + 0 * cStride + 0 * 8 + 0 * 16, D00);
        _mm256_storeu_ps(dst + 0 * cStride + 1 * 8 + 0 * 16, D01);
        _mm256_storeu_ps(dst + 0 * cStride + 0 * 8 + 1 * 16, D10);
        _mm256_storeu_ps(dst + 0 * cStride + 1 * 8 + 1 * 16, D11);
        _mm256_storeu_ps(dst + 0 * cStride + 0 * 8 + 2 * 16, D20);
        _mm256_storeu_ps(dst + 0 * cStride + 1 * 8 + 2 * 16, D21);
        _mm256_storeu_ps(dst + 0 * cStride + 0 * 8 + 3 * 16, D30);
        _mm256_storeu_ps(dst + 0 * cStride + 1 * 8 + 3 * 16, D31);
        _mm256_storeu_ps(dst + 0 * cStride + 0 * 8 + 4 * 16, D40);
        _mm256_storeu_ps(dst + 0 * cStride + 1 * 8 + 4 * 16, D41);

        _mm256_storeu_ps(dst + 1 * cStride + 0 * 8 + 0 * 16, D02);
        _mm256_storeu_ps(dst + 1 * cStride + 1 * 8 + 0 * 16, D03);
        _mm256_storeu_ps(dst + 1 * cStride + 0 * 8 + 1 * 16, D12);
        _mm256_storeu_ps(dst + 1 * cStride + 1 * 8 + 1 * 16, D13);
        _mm256_storeu_ps(dst + 1 * cStride + 0 * 8 + 2 * 16, D22);
        _mm256_storeu_ps(dst + 1 * cStride + 1 * 8 + 2 * 16, D23);
        _mm256_storeu_ps(dst + 1 * cStride + 0 * 8 + 3 * 16, D32);
        _mm256_storeu_ps(dst + 1 * cStride + 1 * 8 + 3 * 16, D33);
        _mm256_storeu_ps(dst + 1 * cStride + 0 * 8 + 4 * 16, D42);
        _mm256_storeu_ps(dst + 1 * cStride + 1 * 8 + 4 * 16, D43);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        INIT_MAIN_5_8;
        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_5_8;
        }
        _mm256_storeu_ps(dst + 16 * 0, z0);
        _mm256_storeu_ps(dst + 16 * 1, z1);
        _mm256_storeu_ps(dst + 16 * 2, z2);
        _mm256_storeu_ps(dst + 16 * 3, z3);
        _mm256_storeu_ps(dst + 16 * 4, z4);
    }
}

static void _AVX2_MNNPackedMatMul_4(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    const int unit = 4;
    int hU = hC8 / unit;
    int hR = hU * unit;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        __m512 W0, W1;
        LOAD_W_4;
        auto srcUse = A;
        auto S0 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 0));
        auto S1 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 1));
        auto S2 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 2));
        auto S3 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 3));

        auto D00 = _mm512_mul_ps(S0, W0);
        auto D01 = _mm512_mul_ps(S0, W1);
        auto D10 = _mm512_mul_ps(S1, W0);
        auto D11 = _mm512_mul_ps(S1, W1);
        auto D20 = _mm512_mul_ps(S2, W0);
        auto D21 = _mm512_mul_ps(S2, W1);
        auto D30 = _mm512_mul_ps(S3, W0);
        auto D31 = _mm512_mul_ps(S3, W1);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        srcUse += aStride;
        for (int sy = 1; sy < l; ++sy) {
            LOAD_W_4;

            S0 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 0));
            S1 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 1));
            S2 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 2));
            S3 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 3));

            D00 = _mm512_fmadd_ps(S0, W0, D00);
            D01 = _mm512_fmadd_ps(S0, W1, D01);
            D10 = _mm512_fmadd_ps(S1, W0, D10);
            D11 = _mm512_fmadd_ps(S1, W1, D11);
            D20 = _mm512_fmadd_ps(S2, W0, D20);
            D21 = _mm512_fmadd_ps(S2, W1, D21);
            D30 = _mm512_fmadd_ps(S3, W0, D30);
            D31 = _mm512_fmadd_ps(S3, W1, D31);

            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
            srcUse += aStride;
        }
        SAVE_UNIT(0, 0, 0);
        SAVE_UNIT(1, 0, 0);
        SAVE_UNIT(2, 0, 0);
        SAVE_UNIT(3, 0, 0);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
        SAVE_UNIT(2, 1, 0);
        SAVE_UNIT(3, 1, 0);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        INIT_MAIN_4_8;
        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_4_8;
        }
        _mm256_storeu_ps(dst + 16 * 0, z0);
        _mm256_storeu_ps(dst + 16 * 1, z1);
        _mm256_storeu_ps(dst + 16 * 2, z2);
        _mm256_storeu_ps(dst + 16 * 3, z3);
    }
}

static void _AVX2_MNNPackedMatMul_3(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    const int unit = 4;
    int hU = hC8 / unit;
    int hR = hU * unit;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        __m512 W0, W1;
        LOAD_W_4;
        auto srcUse = A;
        auto S0 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 0));
        auto S1 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 1));
        auto S2 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 2));

        auto D00 = _mm512_mul_ps(S0, W0);
        auto D01 = _mm512_mul_ps(S0, W1);
        auto D10 = _mm512_mul_ps(S1, W0);
        auto D11 = _mm512_mul_ps(S1, W1);
        auto D20 = _mm512_mul_ps(S2, W0);
        auto D21 = _mm512_mul_ps(S2, W1);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        srcUse += aStride;
        for (int sy = 1; sy < l; ++sy) {
            LOAD_W_4;

            S0 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 0));
            S1 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 1));
            S2 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 2));

            D00 = _mm512_fmadd_ps(S0, W0, D00);
            D01 = _mm512_fmadd_ps(S0, W1, D01);
            D10 = _mm512_fmadd_ps(S1, W0, D10);
            D11 = _mm512_fmadd_ps(S1, W1, D11);
            D20 = _mm512_fmadd_ps(S2, W0, D20);
            D21 = _mm512_fmadd_ps(S2, W1, D21);

            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
            srcUse += aStride;
        }
        SAVE_UNIT(0, 0, 0);
        SAVE_UNIT(1, 0, 0);
        SAVE_UNIT(2, 0, 0);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
        SAVE_UNIT(2, 1, 0);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);
        auto w1 = _mm256_broadcast_ss(A + 0 * aStride + 1);
        auto w2 = _mm256_broadcast_ss(A + 0 * aStride + 2);
        auto z0 = _mm256_mul_ps(s0, w0);
        auto z1 = _mm256_mul_ps(s0, w1);
        auto z2 = _mm256_mul_ps(s0, w2);
        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm256_loadu_ps(weight + sy * 8);
            w0 = _mm256_broadcast_ss(A + sy * aStride + 0);
            w1 = _mm256_broadcast_ss(A + sy * aStride + 1);
            w2 = _mm256_broadcast_ss(A + sy * aStride + 2);
            z0 = MNNAVXFMA(s0, w0, z0);
            z1 = MNNAVXFMA(s0, w1, z1);
            z2 = MNNAVXFMA(s0, w2, z2);
        }
        _mm256_storeu_ps(dst + 16 * 0, z0);
        _mm256_storeu_ps(dst + 16 * 1, z1);
        _mm256_storeu_ps(dst + 16 * 2, z2);
    }
}

static void _AVX2_MNNPackedMatMul_2(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    const int unit = 4;
    int hU = hC8 / unit;
    int hR = hU * unit;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        __m512 W0, W1;
        LOAD_W_4;
        auto srcUse = A;
        auto S0 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 0));
        auto S1 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 1));

        auto D00 = _mm512_mul_ps(S0, W0);
        auto D01 = _mm512_mul_ps(S0, W1);
        auto D10 = _mm512_mul_ps(S1, W0);
        auto D11 = _mm512_mul_ps(S1, W1);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        srcUse += aStride;
        for (int sy = 1; sy < l; ++sy) {
            LOAD_W_4;

            S0 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 0));
            S1 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 1));

            D00 = _mm512_fmadd_ps(S0, W0, D00);
            D01 = _mm512_fmadd_ps(S0, W1, D01);
            D10 = _mm512_fmadd_ps(S1, W0, D10);
            D11 = _mm512_fmadd_ps(S1, W1, D11);

            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
            srcUse += aStride;
        }
        SAVE_UNIT(0, 0, 0);
        SAVE_UNIT(1, 0, 0);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);
        auto w1 = _mm256_broadcast_ss(A + 0 * aStride + 1);
        auto z0 = _mm256_mul_ps(s0, w0);
        auto z1 = _mm256_mul_ps(s0, w1);
        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm256_loadu_ps(weight + sy * 8);
            w0 = _mm256_broadcast_ss(A + sy * aStride + 0);
            w1 = _mm256_broadcast_ss(A + sy * aStride + 1);
            z0 = MNNAVXFMA(s0, w0, z0);
            z1 = MNNAVXFMA(s0, w1, z1);
        }
        _mm256_storeu_ps(dst + 16 * 0, z0);
        _mm256_storeu_ps(dst + 16 * 1, z1);
    }
}

static void _AVX2_MNNPackedMatMul_1(float* C, const float* A, const float* B, const size_t* parameter) {
    auto aStride      = parameter[0] / sizeof(float);
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    const int unit = 8;
    int hU = hC8 / unit;
    int hR = hU * unit;
    int lC2 = l / 2;
    int lR = l % 2;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        auto w4 = B + (unit * y + 4) * bStride;
        auto w5 = B + (unit * y + 5) * bStride;
        auto w6 = B + (unit * y + 6) * bStride;
        auto w7 = B + (unit * y + 7) * bStride;
        __m512 S0;

        auto srcUse = A;

        auto D00 = _mm512_setzero_ps();
        auto D01 = _mm512_setzero_ps();
        auto D02 = _mm512_setzero_ps();
        auto D03 = _mm512_setzero_ps();
        auto D04 = _mm512_setzero_ps();
        auto D05 = _mm512_setzero_ps();
        auto D06 = _mm512_setzero_ps();
        auto D07 = _mm512_setzero_ps();

        for (int sy = 0; sy < lC2; ++sy) {
            BROAD_CAST_S_2;
            auto W0 = _mm512_loadu_ps(w0);
            auto W1 = _mm512_loadu_ps(w1);
            auto W2 = _mm512_loadu_ps(w2);
            auto W3 = _mm512_loadu_ps(w3);
            auto W4 = _mm512_loadu_ps(w4);
            auto W5 = _mm512_loadu_ps(w5);
            auto W6 = _mm512_loadu_ps(w6);
            auto W7 = _mm512_loadu_ps(w7);

            D00 = _mm512_fmadd_ps(S0, W0, D00);
            D01 = _mm512_fmadd_ps(S0, W1, D01);
            D02 = _mm512_fmadd_ps(S0, W2, D02);
            D03 = _mm512_fmadd_ps(S0, W3, D03);
            D04 = _mm512_fmadd_ps(S0, W4, D04);
            D05 = _mm512_fmadd_ps(S0, W5, D05);
            D06 = _mm512_fmadd_ps(S0, W6, D06);
            D07 = _mm512_fmadd_ps(S0, W7, D07);

            w0 += 16;
            w1 += 16;
            w2 += 16;
            w3 += 16;
            w4 += 16;
            w5 += 16;
            w6 += 16;
            w7 += 16;
            srcUse += (aStride * 2);
        }
#define MERGE_TEMP(i, j) auto d##i##j = _mm256_add_ps(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(D##i##j), 0)), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(D##i##j), 1)));
        MERGE_TEMP(0, 0);
        MERGE_TEMP(0, 1);
        MERGE_TEMP(0, 2);
        MERGE_TEMP(0, 3);
        MERGE_TEMP(0, 4);
        MERGE_TEMP(0, 5);
        MERGE_TEMP(0, 6);
        MERGE_TEMP(0, 7);
#undef MERGE_TEMP
        if (lR > 0) {
            auto s0 = _mm256_broadcast_ss(srcUse);
            auto W0 = _mm256_loadu_ps(w0);
            auto W1 = _mm256_loadu_ps(w1);
            auto W2 = _mm256_loadu_ps(w2);
            auto W3 = _mm256_loadu_ps(w3);
            auto W4 = _mm256_loadu_ps(w4);
            auto W5 = _mm256_loadu_ps(w5);
            auto W6 = _mm256_loadu_ps(w6);
            auto W7 = _mm256_loadu_ps(w7);

            d00 = _mm256_fmadd_ps(s0, W0, d00);
            d01 = _mm256_fmadd_ps(s0, W1, d01);
            d02 = _mm256_fmadd_ps(s0, W2, d02);
            d03 = _mm256_fmadd_ps(s0, W3, d03);
            d04 = _mm256_fmadd_ps(s0, W4, d04);
            d05 = _mm256_fmadd_ps(s0, W5, d05);
            d06 = _mm256_fmadd_ps(s0, W6, d06);
            d07 = _mm256_fmadd_ps(s0, W7, d07);
        }
        auto dst    = C + 4 * y * cStride;
        _mm256_storeu_ps(dst, d00);
        _mm256_storeu_ps(dst + 8, d01);
        _mm256_storeu_ps(dst + cStride, d02);
        _mm256_storeu_ps(dst + 8 + cStride, d03);
        _mm256_storeu_ps(dst + cStride * 2, d04);
        _mm256_storeu_ps(dst + 8 + cStride * 2, d05);
        _mm256_storeu_ps(dst + cStride * 3, d06);
        _mm256_storeu_ps(dst + 8 + cStride * 3, d07);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + (y / 2) * cStride + (y % 2) * 8;
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);
        auto z0 = _mm256_mul_ps(s0, w0);
        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm256_loadu_ps(weight + sy * 8);
            w0 = _mm256_broadcast_ss(A + sy * aStride + 0);
            z0 = MNNAVXFMA(s0, w0, z0);;
        }
        _mm256_storeu_ps(dst + 16 * 0, z0);
    }
}

static void _AVX512_MNNPackednMatMulRemainCommon(float* C, const float* A, const float* B, size_t eSize,
                                                 const size_t* parameter, const float* postParameters,
                                                 const float* bias) {
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    auto es           = eSize;
    auto oC           = C;
    auto aStride      = parameter[0] / sizeof(float);
    if (eSize >= 32) {
#ifdef MNN_X86_USE_ASM
        _AVX512_MNNGemmFloatUnit32x8(C, A, B, parameter);
#else
        _AVX512_MNNPackedMatMul_32(C, A, B, parameter);
#endif
        eSize -= 32;
        C += 32 * 16;
        A += 32;
    }
    if (eSize >= 16) {
#ifdef MNN_X86_USE_ASM
        _AVX512_MNNGemmFloatUnit16x8(C, A, B, parameter);
#else
        _AVX512_MNNPackedMatMul_16(C, A, B, parameter);
#endif
        eSize -= 16;
        C += 16 * 16;
        A += 16;
    }
    if (eSize >= 8) {
        _AVX2_MNNPackedMatMul_8(C, A, B, parameter);
        eSize -= 8;
        C += 8 * 16;
        A += 8;
    }
    if (eSize >= 5) {
        _AVX2_MNNPackedMatMul_5(C, A, B, parameter);
        eSize -= 5;
        C += 5 * 16;
        A += 5;
    }
    if (eSize >= 4) {
        _AVX2_MNNPackedMatMul_4(C, A, B, parameter);
        eSize -= 4;
        C += 4 * 16;
        A += 4;
    }
    if (eSize >= 3) {
        _AVX2_MNNPackedMatMul_3(C, A, B, parameter);
        eSize -= 3;
        C += 3 * 16;
        A += 3;
    }
    if (eSize >= 2) {
        _AVX2_MNNPackedMatMul_2(C, A, B, parameter);
        eSize -= 2;
        C += 2 * 16;
        A += 2;
    }
    if (eSize == 1) {
        _AVX2_MNNPackedMatMul_1(C, A, B, parameter);
    }
}

void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
#ifdef MNN_X86_USE_ASM
    if (nullptr == postParameters) {
        _AVX512_MNNGemmFloatUnit48x8(C, A, B, parameter);
    } else {
        _AVX512_MNNGemmFloatUnit48x8Fused(C, A, B, parameter, postParameters, bias);
    }
    // Fill last remain 8 for zero
    AVX512GemmPostTreat(C, 48, parameter, nullptr, nullptr);
#else
    _AVX512_MNNPackedMatMul_48(C, A, B, parameter);
    AVX512GemmPostTreat(C, 48, parameter, postParameters, bias);
#endif
}

//#define MNN_X86_DEBUG
void _AVX512_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
#ifdef MNN_X86_DEBUG
    static std::set<int> gSize;
    if (gSize.find(eSize) == gSize.end()) {
        FUNC_PRINT(eSize);
        gSize.insert(eSize);
    }
#endif
    _AVX512_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, postParameters, bias);
    AVX512GemmPostTreat(C, eSize, parameter, postParameters, bias);
}

MNN::CoreFunctions::MNNPackedMatMulKernel _AVX512_MNNPackedMatMulOC16Functions[AVX512_INPUT_TILE_MAX] = { // oc is 16
    _AVX512_MNNPackedMatMulO16FullLoadKernel<1>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<2>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<3>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<4>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<5>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<6>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<7>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<8>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<9>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<10>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<11>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<12>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<13>,
    _AVX512_MNNPackedMatMulO16FullLoadKernel<14>,
    // _AVX512_MNNPackedMatMulO16FullLoadKernel<15>,
    // _AVX512_MNNPackedMatMulO16FullLoadKernel<16>,
    // _AVX512_MNNPackedMatMulO16FullLoadKernel<31>,  as much as 31
};
MNN::CoreFunctions::MNNPackedMatMulKernel _AVX512_MNNPackedMatMulOC32Functions[AVX512_INPUT_TILE_MAX] = { // oc is 32
    _AVX512_MNNPackedMatMulO32FullLoadKernel<1>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<2>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<3>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<4>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<5>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<6>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<7>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<8>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<9>,
    _AVX512_MNNPackedMatMulO32FullLoadKernel<10>, // above kernel A and B matrix registers are fully loaded
    _AVX512_MNNPackedMatMulO32Swaped6Kernel<11>,
    _AVX512_MNNPackedMatMulO32Swaped6Kernel<12>,
    _AVX512_MNNPackedMatMulO32SwapedKernel<13>,
    _AVX512_MNNPackedMatMulO32SwapedKernel<14>, // registers are swaped and reused
};
MNN::CoreFunctions::MNNPackedMatMulKernel _AVX512_MNNPackedMatMulOC48Functions[AVX512_INPUT_TILE_MAX] = { // oc is 48
    _AVX512_MNNPackedMatMulO48FullLoadKernel<1>,
    _AVX512_MNNPackedMatMulO48FullLoadKernel<2>,
    _AVX512_MNNPackedMatMulO48FullLoadKernel<3>,
    _AVX512_MNNPackedMatMulO48FullLoadKernel<4>,
    _AVX512_MNNPackedMatMulO48FullLoadKernel<5>,
    _AVX512_MNNPackedMatMulO48FullLoadKernel<6>,
    _AVX512_MNNPackedMatMulO48FullLoadKernel<7>, // above kernel A and B matrix registers are fully loaded
    _AVX512_MNNPackedMatMulO48Swaped4Kernel<8>,
    _AVX512_MNNPackedMatMulO48Swaped2Kernel<9>,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};


#ifdef AVX512_TEST
class AVX512_AutoTest {
public:
    AVX512_AutoTest() {
        float temp[16*16];
        for (int x=0; x<16; ++x) {
            for (int y=0; y<16; ++y) {
                temp[x + y * 16] = x + y * 100;
                MNN_PRINT("%f, ", temp[x + y * 16]);
            }
            MNN_PRINT("\n");
        }
        auto r0 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 0));
        auto r1 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 1));
        auto r2 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 2));
        auto r3 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 3));
        auto r4 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 4));
        auto r5 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 5));
        auto r6 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 6));
        auto r7 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 7));
        auto r8 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 8));
        auto r9 = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 9));
        auto ra = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 10));
        auto rb = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 11));
        auto rc = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 12));
        auto rd = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 13));
        auto re = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 14));
        auto rf = _mm512_castps_si512(_mm512_loadu_ps(temp + 16 * 15));

        transpose16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

        _mm512_storeu_ps(temp + 16 * 0, _mm512_castsi512_ps(r0));
        _mm512_storeu_ps(temp + 16 * 1, _mm512_castsi512_ps(r1));
        _mm512_storeu_ps(temp + 16 * 2, _mm512_castsi512_ps(r2));
        _mm512_storeu_ps(temp + 16 * 3, _mm512_castsi512_ps(r3));
        _mm512_storeu_ps(temp + 16 * 4, _mm512_castsi512_ps(r4));
        _mm512_storeu_ps(temp + 16 * 5, _mm512_castsi512_ps(r5));
        _mm512_storeu_ps(temp + 16 * 6, _mm512_castsi512_ps(r6));
        _mm512_storeu_ps(temp + 16 * 7, _mm512_castsi512_ps(r7));
        _mm512_storeu_ps(temp + 16 * 8, _mm512_castsi512_ps(r8));
        _mm512_storeu_ps(temp + 16 * 9, _mm512_castsi512_ps(r9));
        _mm512_storeu_ps(temp + 16 * 10, _mm512_castsi512_ps(ra));
        _mm512_storeu_ps(temp + 16 * 11, _mm512_castsi512_ps(rb));
        _mm512_storeu_ps(temp + 16 * 12, _mm512_castsi512_ps(rc));
        _mm512_storeu_ps(temp + 16 * 13, _mm512_castsi512_ps(rd));
        _mm512_storeu_ps(temp + 16 * 14, _mm512_castsi512_ps(re));
        _mm512_storeu_ps(temp + 16 * 15, _mm512_castsi512_ps(rf));

        MNN_PRINT("Transposed:\n");
        for (int x=0; x<16; ++x) {
            for (int y=0; y<16; ++y) {
                MNN_PRINT("%f, ", temp[x + y * 16]);
            }
            MNN_PRINT("\n");
        }
    }
};

AVX512_AutoTest __t;

#endif
