#include "FunctionSummary.hpp"
#include "Gemm48_8.hpp"
#include "core/Macro.h"
#include <limits>
#include <string.h>
#include <algorithm>
#include <vector>
#include "../AVX2Functions.hpp"
#include "../avx/Vec8.hpp"

void _AVX512_MNNMatrixSub(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            auto am = _mm256_loadu_ps(a + 8 * x);
            auto bm = _mm256_loadu_ps(b + 8 * x);
            auto cm = _mm256_sub_ps(am, bm);
            _mm256_storeu_ps(c + 8 * x, cm);
        }
    }
}

void _AVX512_MNNMatrixAdd(float* C, const float* A, const float* B, size_t widthC4, size_t cStride, size_t aStride,
                  size_t bStride, size_t height) {
    for (int y = 0; y < height; ++y) {
        auto a = A + aStride * y;
        auto b = B + bStride * y;
        auto c = C + cStride * y;
        for (int x = 0; x < widthC4; ++x) {
            auto am = _mm256_loadu_ps(a + 8 * x);
            auto bm = _mm256_loadu_ps(b + 8 * x);
            auto cm = _mm256_add_ps(am, bm);
            _mm256_storeu_ps(c + 8 * x, cm);
        }
    }
}

void _AVX512_MNNStrassenMergeCFunction(float* c11, float* c12, float* c21, float* c22, float* xAddr, size_t cStride,
                               size_t eSub, size_t hSub) {
    for (int y=0; y<hSub; ++y) {
        auto c11Y = c11 + y * cStride;
        auto c12Y = c12 + y * cStride;
        auto c22Y = c22 + y * cStride;
        auto c21Y = c21 + y * cStride;
        auto xY = xAddr + y * eSub * 8;
        for (int x=0; x<eSub; ++x) {
            auto xv = _mm256_loadu_ps(xY + 8*x);
            auto c21v = _mm256_loadu_ps(c21Y + 8*x);
            auto c11v = _mm256_loadu_ps(c11Y + 8*x);
            auto c22v = _mm256_loadu_ps(c22Y + 8*x);
            auto c12v = _mm256_loadu_ps(c12Y + 8*x);
            c12v = _mm256_add_ps(c12v, xv);
            c21v = _mm256_add_ps(c12v, c21v);
            c12v = _mm256_add_ps(c22v, c12v);
            c22v = _mm256_add_ps(c22v, c21v);
            c12v = _mm256_add_ps(c11v, c12v);
            _mm256_storeu_ps(c12Y + 8*x, c12v);
            _mm256_storeu_ps(c12Y + 8*x, c12v);
            _mm256_storeu_ps(c22Y + 8*x, c22v);
            _mm256_storeu_ps(c21Y + 8*x, c21v);
        }
    }
}

void AVX512GemmPostTreat(float* C, size_t eSize, const size_t* parameter, const float* postParameters,
                       const float* bias) {
    if (nullptr == postParameters) {
        return;
    }
    auto h            = parameter[2];
    auto l            = parameter[1];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8          = UP_DIV(h, 8);
    auto minValue        = _mm256_broadcast_ss(postParameters + 2);
    auto maxValue        = _mm256_broadcast_ss(postParameters + 3);
    if (nullptr != bias) {
        for (int y = 0; y < hC8; ++y) {
            auto biasValue = _mm256_loadu_ps(bias + 8 * y);
            auto dst       = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm256_add_ps(biasValue, _mm256_loadu_ps(dst));
                sum      = _mm256_max_ps(sum, minValue);
                sum      = _mm256_min_ps(sum, maxValue);
                _mm256_storeu_ps(dst, sum);
                dst += 8;
            }
        }
    } else {
        for (int y = 0; y < hC8; ++y) {
            auto dst = C + y * cStride;
            for (int x = 0; x < eSize; ++x) {
                auto sum = _mm256_loadu_ps(dst);
                sum      = _mm256_max_ps(sum, minValue);
                sum      = _mm256_min_ps(sum, maxValue);
                _mm256_storeu_ps(dst, sum);
                dst += 8;
            }
        }
    }
}

#ifdef MNN_X86_USE_ASM
extern "C" {
void _AVX512_MNNGemmFloatUnitMainFMA(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
void _AVX512_MNNGemmFloatUnit16(float* C, const float* A, const float* B, const size_t* parameter, size_t hC4);
}
#endif

void _AVX512_MNNPackC8(float* dst, const float* src, size_t area, size_t depth) {
    auto hC8 = UP_DIV(depth, 8);
    int h8 = depth / 8;
    int hR = depth % 8;

    if (hR > 0) {
        ::memset(dst, 0, hC8 * 8 * area * sizeof(float));
    }

    for (int y = 0; y < h8; y++) {
        auto srcY = src + y * 8;
        auto dstY = dst + y * 8 * area;
        for (int x = 0; x < area; x++) {
            auto srcX = srcY + x * depth;
            auto dstX = dstY + x * 8;
            _mm256_storeu_ps(dstX, _mm256_loadu_ps(srcX));
        }
    }
    if (hR > 0) {
        auto srcY = src + h8 * 8;
        auto dstY = dst + h8 * 8 * area;
        for (int x = 0; x < area; x++) {
            auto srcX = srcY + x * depth;
            auto dstX = dstY + x * 8;
            ::memcpy(dstX, srcX, hR * sizeof(float));
        }
    }
    return;
}

void _AVX512_MNNUnPackC8(float* dst, const float* src, size_t area, size_t depth) {
    int hC8 = UP_DIV(depth, 8);
    int h8 = depth / 8;
    int hR = depth % 8;

    for (int y = 0; y < h8; y++) {
        auto srcY = src + y * 8 * area;
        auto dstY = dst + y * 8;
        for (int x = 0; x < area ; x++) {
            auto srcX = srcY + x * 8;
            auto dstX = dstY + x * depth;
            _mm256_storeu_ps(dstX, _mm256_loadu_ps(srcX));
        }
    }

    if (hR > 0) {
        auto srcY = src + h8 * 8 * area;
        auto dstY = dst + h8 * 8;
        for (int x = 0; x < area; x++) {
            auto srcX = srcY + x * 8;
            auto dstX = dstY + x * depth;
            ::memcpy(dstX, srcX, hR * sizeof(float));
        }
    }
}

void _AVX512_MNNPackC8ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
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
auto r30 = _mm256_loadu_ps(srcX + 24 * pOffset);  \
auto r31 = _mm256_loadu_ps(srcX + 25 * pOffset);  \
auto r32 = _mm256_loadu_ps(srcX + 26 * pOffset);  \
auto r33 = _mm256_loadu_ps(srcX + 27 * pOffset);  \
auto r34 = _mm256_loadu_ps(srcX + 28 * pOffset);  \
auto r35 = _mm256_loadu_ps(srcX + 29 * pOffset);  \
auto r36 = _mm256_loadu_ps(srcX + 30 * pOffset);  \
auto r37 = _mm256_loadu_ps(srcX + 31 * pOffset);  \
auto r40 = _mm256_loadu_ps(srcX + 32 * pOffset);  \
auto r41 = _mm256_loadu_ps(srcX + 33 * pOffset);  \
auto r42 = _mm256_loadu_ps(srcX + 34 * pOffset);  \
auto r43 = _mm256_loadu_ps(srcX + 35 * pOffset);  \
auto r44 = _mm256_loadu_ps(srcX + 36 * pOffset);  \
auto r45 = _mm256_loadu_ps(srcX + 37 * pOffset);  \
auto r46 = _mm256_loadu_ps(srcX + 38 * pOffset);  \
auto r47 = _mm256_loadu_ps(srcX + 39 * pOffset);  \
auto r50 = _mm256_loadu_ps(srcX + 40 * pOffset);  \
auto r51 = _mm256_loadu_ps(srcX + 41 * pOffset);  \
auto r52 = _mm256_loadu_ps(srcX + 42 * pOffset);  \
auto r53 = _mm256_loadu_ps(srcX + 43 * pOffset);  \
auto r54 = _mm256_loadu_ps(srcX + 44 * pOffset);  \
auto r55 = _mm256_loadu_ps(srcX + 45 * pOffset);  \
auto r56 = _mm256_loadu_ps(srcX + 46 * pOffset);  \
auto r57 = _mm256_loadu_ps(srcX + 47 * pOffset);  \
TRANSPOSE_8x8_REPLACE(r00, r01, r02, r03, r04, r05, r06, r07);\
TRANSPOSE_8x8_REPLACE(r10, r11, r12, r13, r14, r15, r16, r17);\
TRANSPOSE_8x8_REPLACE(r20, r21, r22, r23, r24, r25, r26, r27);\
TRANSPOSE_8x8_REPLACE(r30, r31, r32, r33, r34, r35, r36, r37);\
TRANSPOSE_8x8_REPLACE(r40, r41, r42, r43, r44, r45, r46, r47);\
TRANSPOSE_8x8_REPLACE(r50, r51, r52, r53, r54, r55, r56, r57);\

#define STORE_TEMP(i)                               \
_mm256_storeu_ps(dstX + 48 * i + 0 * 8, r0##i);\
_mm256_storeu_ps(dstX + 48 * i + 1 * 8, r1##i);\
_mm256_storeu_ps(dstX + 48 * i + 2 * 8, r2##i);\
_mm256_storeu_ps(dstX + 48 * i + 3 * 8, r3##i);\
_mm256_storeu_ps(dstX + 48 * i + 4 * 8, r4##i);\
_mm256_storeu_ps(dstX + 48 * i + 5 * 8, r5##i);\

        const int pack   = 48;
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
                switch (lRes) {
                    case 7:
                        STORE_TEMP(6);
                    case 6:
                        STORE_TEMP(5);
                    case 5:
                        STORE_TEMP(4);
                    case 4:
                        STORE_TEMP(3);
                    case 3:
                        STORE_TEMP(2);
                    case 2:
                        STORE_TEMP(1);
                    case 1:
                        STORE_TEMP(0);
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
#undef MAIN_COMPUTE
#undef STORE_TEMP

}

void _AVX512_MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    int offset[2] = {
        (int)l,
        (int)l
    };
    if (!transpose) {
        MNN::AVX2Functions::get()->MNNPackCUnitTranspose(dest, source, l, h, offset);
        return;
    }
    MNN::AVX2Functions::get()->MNNPackCUnit(dest, source, l, h, offset);
}

static void _AVX512_MNNPackedMatMul_48(float* C, const float* A, const float* B, const size_t* parameter) {
    auto l            = parameter[1];
    auto h            = parameter[2];
    auto cStride      = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride      = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    auto hR = 0;

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
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

    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + hC8 * cStride;
        INIT_MAIN_48_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_48_8;
        }

        AVX512_TRANSPOSE_SAVE_HALF(0, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE_HALF(1, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE_HALF(2, 0, z0, z3, z6, z9, z12, z15, z18, z21);
        AVX512_TRANSPOSE_SAVE_HALF(3, 0, z0, z3, z6, z9, z12, z15, z18, z21);

        AVX512_TRANSPOSE_SAVE_HALF(0, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE_HALF(1, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE_HALF(2, 1, z1, z4, z7, z10, z13, z16, z19, z22);
        AVX512_TRANSPOSE_SAVE_HALF(3, 1, z1, z4, z7, z10, z13, z16, z19, z22);

        AVX512_TRANSPOSE_SAVE_HALF(0, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE_HALF(1, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE_HALF(2, 2, z2, z5, z8, z11, z14, z17, z20, z23);
        AVX512_TRANSPOSE_SAVE_HALF(3, 2, z2, z5, z8, z11, z14, z17, z20, z23);
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
    auto hR = 0;

    for (int y = 0; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_16_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_16_8;
        }
        AVX512_TRANSPOSE_SAVE(0, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(1, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(2, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE(3, 0, z0, z1, z2, z3, z4, z5, z6, z7);
    }
    if (hR > 0) {
        auto weight = B + hC8 * bStride;
        auto dst    = C + hC8 * cStride;
        INIT_MAIN_16_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_16_8;
        }
        AVX512_TRANSPOSE_SAVE_HALF(0, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE_HALF(1, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE_HALF(2, 0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX512_TRANSPOSE_SAVE_HALF(3, 0, z0, z1, z2, z3, z4, z5, z6, z7);
    }
}

#define SAVE_UNIT(i, j, k) _mm256_storeu_pd((double*)(C + (unit * y + (2*j+k)) * cStride + 8 * i), _mm512_extractf64x4_pd(_mm512_castps_pd(D##i##j), k))

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
        auto W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
        auto W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));
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
            W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
            W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));

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

        SAVE_UNIT(0, 0, 1);
        SAVE_UNIT(1, 0, 1);
        SAVE_UNIT(2, 0, 1);
        SAVE_UNIT(3, 0, 1);
        SAVE_UNIT(4, 0, 1);
        SAVE_UNIT(5, 0, 1);
        SAVE_UNIT(6, 0, 1);
        SAVE_UNIT(7, 0, 1);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
        SAVE_UNIT(2, 1, 0);
        SAVE_UNIT(3, 1, 0);
        SAVE_UNIT(4, 1, 0);
        SAVE_UNIT(5, 1, 0);
        SAVE_UNIT(6, 1, 0);
        SAVE_UNIT(7, 1, 0);

        SAVE_UNIT(0, 1, 1);
        SAVE_UNIT(1, 1, 1);
        SAVE_UNIT(2, 1, 1);
        SAVE_UNIT(3, 1, 1);
        SAVE_UNIT(4, 1, 1);
        SAVE_UNIT(5, 1, 1);
        SAVE_UNIT(6, 1, 1);
        SAVE_UNIT(7, 1, 1);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_8_8;

        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_8_8;
        }
        AVX2_TRANSPOSE_SAVE(0, z0, z1, z2, z3, z4, z5, z6, z7);
        AVX2_TRANSPOSE_SAVE(1, z0, z1, z2, z3, z4, z5, z6, z7);
    }
}

static void _AVX2_MNNPackedMatMul_5(float *C, const float *A, const float *B, const size_t *parameter)
{
    auto aStride = parameter[0] / sizeof(float);
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 8;
    auto hC8 = UP_DIV(h, 8);
    const int hC8Unit = 4;
    auto hC32 = hC8 / hC8Unit;
    auto hR = hC32 * hC8Unit;

    for (int y = 0; y < hC32; ++y)
    {
        auto weight0 = B + (hC8Unit * y + 0) * bStride;
        auto weight1 = B + (hC8Unit * y + 1) * bStride;
        auto weight2 = B + (hC8Unit * y + 2) * bStride;
        auto weight3 = B + (hC8Unit * y + 3) * bStride;
        auto dst0 = C + (hC8Unit * y + 0) * cStride;
        auto dst1 = C + (hC8Unit * y + 1) * cStride;
        auto dst2 = C + (hC8Unit * y + 2) * cStride;
        auto dst3 = C + (hC8Unit * y + 3) * cStride;

        auto sumAvx00 = _mm256_set1_ps(0.0f);
        auto sumAvx01 = _mm256_set1_ps(0.0f);
        auto sumAvx02 = _mm256_set1_ps(0.0f);
        auto sumAvx03 = _mm256_set1_ps(0.0f);

        auto sumAvx10 = _mm256_set1_ps(0.0f);
        auto sumAvx11 = _mm256_set1_ps(0.0f);
        auto sumAvx12 = _mm256_set1_ps(0.0f);
        auto sumAvx13 = _mm256_set1_ps(0.0f);

        auto sumAvx20 = _mm256_set1_ps(0.0f);
        auto sumAvx21 = _mm256_set1_ps(0.0f);
        auto sumAvx22 = _mm256_set1_ps(0.0f);
        auto sumAvx23 = _mm256_set1_ps(0.0f);

        auto sumAvx30 = _mm256_set1_ps(0.0f);
        auto sumAvx31 = _mm256_set1_ps(0.0f);
        auto sumAvx32 = _mm256_set1_ps(0.0f);
        auto sumAvx33 = _mm256_set1_ps(0.0f);

        auto sumAvx40 = _mm256_set1_ps(0.0f);
        auto sumAvx41 = _mm256_set1_ps(0.0f);
        auto sumAvx42 = _mm256_set1_ps(0.0f);
        auto sumAvx43 = _mm256_set1_ps(0.0f);

        auto srcUse = A;
        for (int sy = 0; sy < l; ++sy)
        {
            auto S0 = _mm256_broadcast_ss(srcUse + 0);
            auto S1 = _mm256_broadcast_ss(srcUse + 1);
            auto S2 = _mm256_broadcast_ss(srcUse + 2);
            auto S3 = _mm256_broadcast_ss(srcUse + 3);
            auto S4 = _mm256_broadcast_ss(srcUse + 4);
            auto W0 = _mm256_loadu_ps(weight0);
            auto W1 = _mm256_loadu_ps(weight1);
            auto W2 = _mm256_loadu_ps(weight2);
            auto W3 = _mm256_loadu_ps(weight3);

            sumAvx00 = MNNAVXFMA(S0, W0, sumAvx00);
            sumAvx01 = MNNAVXFMA(S0, W1, sumAvx01);
            sumAvx02 = MNNAVXFMA(S0, W2, sumAvx02);
            sumAvx03 = MNNAVXFMA(S0, W3, sumAvx03);

            sumAvx10 = MNNAVXFMA(S1, W0, sumAvx10);
            sumAvx11 = MNNAVXFMA(S1, W1, sumAvx11);
            sumAvx12 = MNNAVXFMA(S1, W2, sumAvx12);
            sumAvx13 = MNNAVXFMA(S1, W3, sumAvx13);

            sumAvx20 = MNNAVXFMA(S2, W0, sumAvx20);
            sumAvx21 = MNNAVXFMA(S2, W1, sumAvx21);
            sumAvx22 = MNNAVXFMA(S2, W2, sumAvx22);
            sumAvx23 = MNNAVXFMA(S2, W3, sumAvx23);

            sumAvx30 = MNNAVXFMA(S3, W0, sumAvx30);
            sumAvx31 = MNNAVXFMA(S3, W1, sumAvx31);
            sumAvx32 = MNNAVXFMA(S3, W2, sumAvx32);
            sumAvx33 = MNNAVXFMA(S3, W3, sumAvx33);

            sumAvx40 = MNNAVXFMA(S4, W0, sumAvx40);
            sumAvx41 = MNNAVXFMA(S4, W1, sumAvx41);
            sumAvx42 = MNNAVXFMA(S4, W2, sumAvx42);
            sumAvx43 = MNNAVXFMA(S4, W3, sumAvx43);

            srcUse += aStride;
            weight0 += 8;
            weight1 += 8;
            weight2 += 8;
            weight3 += 8;
        }

        _mm256_storeu_ps(dst0 + 8 * 0, sumAvx00);
        _mm256_storeu_ps(dst0 + 8 * 1, sumAvx10);
        _mm256_storeu_ps(dst0 + 8 * 2, sumAvx20);
        _mm256_storeu_ps(dst0 + 8 * 3, sumAvx30);
        _mm256_storeu_ps(dst0 + 8 * 4, sumAvx40);

        _mm256_storeu_ps(dst1 + 8 * 0, sumAvx01);
        _mm256_storeu_ps(dst1 + 8 * 1, sumAvx11);
        _mm256_storeu_ps(dst1 + 8 * 2, sumAvx21);
        _mm256_storeu_ps(dst1 + 8 * 3, sumAvx31);
        _mm256_storeu_ps(dst1 + 8 * 4, sumAvx41);

        _mm256_storeu_ps(dst2 + 8 * 0, sumAvx02);
        _mm256_storeu_ps(dst2 + 8 * 1, sumAvx12);
        _mm256_storeu_ps(dst2 + 8 * 2, sumAvx22);
        _mm256_storeu_ps(dst2 + 8 * 3, sumAvx32);
        _mm256_storeu_ps(dst2 + 8 * 4, sumAvx42);

        _mm256_storeu_ps(dst3 + 8 * 0, sumAvx03);
        _mm256_storeu_ps(dst3 + 8 * 1, sumAvx13);
        _mm256_storeu_ps(dst3 + 8 * 2, sumAvx23);
        _mm256_storeu_ps(dst3 + 8 * 3, sumAvx33);
        _mm256_storeu_ps(dst3 + 8 * 4, sumAvx43);
    }
    for (int y = hR; y < hC8; ++y)
    {
        auto weight = B + y * bStride;
        auto dst = C + y * cStride;
        INIT_MAIN_5_8;
        for (int sy = 1; sy < l; ++sy)
        {
            COMPUTE_5_8;
        }
        _mm256_storeu_ps(dst + 8 * 0, z0);
        _mm256_storeu_ps(dst + 8 * 1, z1);
        _mm256_storeu_ps(dst + 8 * 2, z2);
        _mm256_storeu_ps(dst + 8 * 3, z3);
        _mm256_storeu_ps(dst + 8 * 4, z4);
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
        auto W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
        auto W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));
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
            W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
            W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));

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

        SAVE_UNIT(0, 0, 1);
        SAVE_UNIT(1, 0, 1);
        SAVE_UNIT(2, 0, 1);
        SAVE_UNIT(3, 0, 1);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
        SAVE_UNIT(2, 1, 0);
        SAVE_UNIT(3, 1, 0);

        SAVE_UNIT(0, 1, 1);
        SAVE_UNIT(1, 1, 1);
        SAVE_UNIT(2, 1, 1);
        SAVE_UNIT(3, 1, 1);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        INIT_MAIN_4_8;
        for (int sy = 1; sy < l; ++sy) {
            COMPUTE_4_8;
        }
        _mm256_storeu_ps(dst + 8 * 0, z0);
        _mm256_storeu_ps(dst + 8 * 1, z1);
        _mm256_storeu_ps(dst + 8 * 2, z2);
        _mm256_storeu_ps(dst + 8 * 3, z3);
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
        auto W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
        auto W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));
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
            W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
            W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));

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

        SAVE_UNIT(0, 0, 1);
        SAVE_UNIT(1, 0, 1);
        SAVE_UNIT(2, 0, 1);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);
        SAVE_UNIT(2, 1, 0);

        SAVE_UNIT(0, 1, 1);
        SAVE_UNIT(1, 1, 1);
        SAVE_UNIT(2, 1, 1);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
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
        _mm256_storeu_ps(dst + 8 * 0, z0);
        _mm256_storeu_ps(dst + 8 * 1, z1);
        _mm256_storeu_ps(dst + 8 * 2, z2);
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
        auto W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
        auto W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));
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
            W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
            W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));

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

        SAVE_UNIT(0, 0, 1);
        SAVE_UNIT(1, 0, 1);

        SAVE_UNIT(0, 1, 0);
        SAVE_UNIT(1, 1, 0);

        SAVE_UNIT(0, 1, 1);
        SAVE_UNIT(1, 1, 1);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
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
        _mm256_storeu_ps(dst + 8 * 0, z0);
        _mm256_storeu_ps(dst + 8 * 1, z1);
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
    const int unit = 4;
    int hU = hC8 / unit;
    int hR = hU * unit;

    for (int y = 0; y < hU; ++y) {
        auto w0 = B + (unit * y + 0) * bStride;
        auto w1 = B + (unit * y + 1) * bStride;
        auto w2 = B + (unit * y + 2) * bStride;
        auto w3 = B + (unit * y + 3) * bStride;
        auto W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
        auto W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));
        auto srcUse = A;
        auto S0 = _mm512_broadcastss_ps(_mm_load_ss(srcUse + 0));

        auto D00 = _mm512_mul_ps(S0, W0);
        auto D01 = _mm512_mul_ps(S0, W1);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        srcUse += aStride;
        for (int sy = 1; sy < l; ++sy) {
            W0 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w0)), _mm256_loadu_pd((double*)w1), 1));
            W1 =  _mm512_castpd_ps(_mm512_insertf64x4(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)w2)), _mm256_loadu_pd((double*)w3), 1));

            S0 = _mm512_broadcastss_ps(_mm_broadcast_ss(srcUse + 0));

            D00 = _mm512_fmadd_ps(S0, W0, D00);
            D01 = _mm512_fmadd_ps(S0, W1, D01);

            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
            srcUse += aStride;
        }
        SAVE_UNIT(0, 0, 0);

        SAVE_UNIT(0, 0, 1);

        SAVE_UNIT(0, 1, 0);

        SAVE_UNIT(0, 1, 1);
    }
    for (int y = hR; y < hC8; ++y) {
        auto weight = B + y * bStride;
        auto dst    = C + y * cStride;
        auto s0 = _mm256_loadu_ps(weight + 0 * 8);
        auto w0 = _mm256_broadcast_ss(A + 0 * aStride + 0);
        auto z0 = _mm256_mul_ps(s0, w0);
        for (int sy = 1; sy < l; ++sy) {
            s0 = _mm256_loadu_ps(weight + sy * 8);
            w0 = _mm256_broadcast_ss(A + sy * aStride + 0);
            z0 = MNNAVXFMA(s0, w0, z0);;
        }
        _mm256_storeu_ps(dst + 8 * 0, z0);
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
    while (eSize >= 16) {
        _AVX512_MNNPackedMatMul_16(C, A, B, parameter);
        eSize -= 16;
        C += 16 * 8;
        A += 16;
    }

    while (eSize >= 8) {
        _AVX2_MNNPackedMatMul_8(C, A, B, parameter);
        eSize -= 8;
        C += 8 * 8;
        A += 8;
    }
    if (eSize >= 5) {
        _AVX2_MNNPackedMatMul_5(C, A, B, parameter);
        eSize -= 5;
        C += 5 * 8;
        A += 5;
    }
    if (eSize >= 4) {
        _AVX2_MNNPackedMatMul_4(C, A, B, parameter);
        eSize -= 4;
        C += 4 * 8;
        A += 4;
    }
    if (eSize >= 3) {
        _AVX2_MNNPackedMatMul_3(C, A, B, parameter);
        eSize -= 3;
        C += 3 * 8;
        A += 3;
    }
    if (eSize >= 2) {
        _AVX2_MNNPackedMatMul_2(C, A, B, parameter);
        eSize -= 2;
        C += 2 * 8;
        A += 2;
    }
    if (eSize == 1) {
        _AVX2_MNNPackedMatMul_1(C, A, B, parameter);
        eSize -= 1;
    }
    if (eSize == 0) {
        return;
    }
}

void _AVX512_MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias) {
    auto h       = parameter[2];
    auto hC4     = UP_DIV(h, 4);
    auto cStride = parameter[3] / sizeof(float);
    _AVX512_MNNPackedMatMul_48(C, A, B, parameter);
    AVX512GemmPostTreat(C, 48, parameter, postParameters, bias);
}

void _AVX512_MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias) {
    _AVX512_MNNPackednMatMulRemainCommon(C, A, B, eSize, parameter, postParameters, bias);
    AVX512GemmPostTreat(C, eSize, parameter, postParameters, bias);
}
